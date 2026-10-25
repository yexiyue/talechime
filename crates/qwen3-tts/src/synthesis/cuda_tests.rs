//! Explicit local-model CUDA investigations; ordinary tests never load weights.
use super::*;

#[test]
#[ignore = "requires CUDA; no weights"]
fn cuda_integer_remainder_matches_host() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let values = [
        i64::MIN,
        -4097,
        -1,
        0,
        2047,
        2048,
        3071,
        3072,
        4096,
        i64::MAX,
    ];
    let codes = Tensor::new(&values, &device)?;
    let divisor = Tensor::new(2048i64, &device)?;
    let remainder = (&codes - codes.broadcast_div(&divisor)?.broadcast_mul(&divisor)?)?;
    assert_eq!(remainder.to_vec1::<i64>()?, values.map(|code| code % 2048));
    Ok(())
}

/// Isolates padding error in the CUDA attention arithmetic, before proposing
/// any runtime KV or RoPE changes. Synthetic Q/K/V is not a quality test.
#[test]
#[ignore = "requires CUDA; numerical investigation only"]
fn cuda_attention_padding_probe() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let (heads, dim) = (16usize, 128usize);
    let fixture = |len, seed: f32| -> Result<Tensor> {
        Ok(Tensor::from_vec(
            (0..heads * len * dim)
                .map(|i| ((i as f32 * 0.071 + seed).sin()) * 0.5)
                .collect::<Vec<_>>(),
            (1, heads, len, dim),
            &device,
        )?
        .to_dtype(DType::BF16)?)
    };
    for (length, bucket) in [(31, 32), (35, 64), (127, 128)] {
        for seed in 0..4 {
            let q = fixture(1, seed as f32)?;
            let k = fixture(length, seed as f32 + 1.0)?;
            let v = fixture(length, seed as f32 + 2.0)?;
            let attention = |k: &Tensor, v: &Tensor, mask: Option<&Tensor>| -> Result<Tensor> {
                let logits =
                    (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * (dim as f64).powf(-0.5))?;
                let logits = match mask {
                    Some(mask) => logits.broadcast_add(mask)?,
                    None => logits,
                };
                Ok(candle_nn::ops::softmax_last_dim(&logits)?.matmul(v)?)
            };
            let baseline = attention(&k, &v, None)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let zeros = Tensor::zeros((1, heads, bucket - length, dim), DType::BF16, &device)?;
            let kp = Tensor::cat(&[&k, &zeros], 2)?;
            let vp = Tensor::cat(&[&v, &zeros], 2)?;
            let mask = Tensor::from_vec(
                (0..bucket)
                    .map(|i| {
                        if i < length {
                            0.0f32
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect::<Vec<_>>(),
                (1, 1, 1, bucket),
                &device,
            )?
            .to_dtype(DType::BF16)?;
            let padded = attention(&kp, &vp, Some(&mask))?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max_abs = baseline
                .iter()
                .zip(&padded)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let changed = baseline
                .iter()
                .zip(&padded)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            println!(
                "{}",
                serde_json::json!({"probe":"attention_padding", "length":length, "bucket":bucket, "seed":seed,
                "changed":changed, "elements":baseline.len(), "max_abs":max_abs})
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a local CUDA model; run serially"]
fn decoder_graph_preserves_outputs_across_shapes_and_sessions() -> Result<()> {
    let path = std::env::var("QWEN_CP_TEST_MODEL")?;
    anyhow::ensure!(Path::new(&path).is_dir(), "local model required");
    let device = Device::new_cuda(0)?;
    let model = Qwen3TTS::from_pretrained(&path, device.clone())?;
    let bits = |tensor: &Tensor| -> Result<Vec<u32>> {
        Ok(tensor
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    let mut retained = None;
    let mut large_visits = std::collections::HashMap::new();
    for (seed, frames) in [
        35, 35, 35, 34, 34, 35, 33, 33, 32, 32, 31, 31, 35, 35, 35, 106, 106, 106, 128, 128, 128,
        129, 129, 106, 106,
    ]
    .into_iter()
    .enumerate()
    {
        let codes = Tensor::from_vec(
            (0..16 * frames)
                .map(|i| ((i * 37 + seed * 71) % 2048) as i64)
                .collect::<Vec<_>>(),
            (1, 16, frames),
            &device,
        )?;
        let expected = bits(&model.decoder.decode(&codes)?)?;
        let actual = model.decoder.decode_streaming(&codes)?;
        assert_eq!(bits(&actual)?, expected, "seed {seed}, frames {frames}");
        if let Some((tensor, values)) = &retained {
            assert_eq!(
                bits(tensor)?,
                *values,
                "a later replay overwrote an earlier result"
            );
        }
        if seed == 2 {
            retained = Some((actual, expected));
        }
        assert!(model.decoder.recorded_shapes() <= 4);
        if [106, 128].contains(&frames) {
            let visits = large_visits.entry(frames).or_insert(0);
            *visits += 1;
            if *visits >= 2 {
                assert!(
                    model.decoder.recorded_frame_sizes().contains(&frames),
                    "large shape {frames} did not record"
                );
            }
        }
        assert!(!model.decoder.recorded_frame_sizes().contains(&129));
    }
    assert!(
        model.decoder.recorded_shapes() > 0,
        "unexpected graph fallback"
    );
    // Evict all table-cache entries while the graph still owns its constants.
    for frames in 40..45 {
        model
            .decoder
            .decode(&Tensor::zeros((1, 16, frames), DType::I64, &device)?)?;
    }
    let codes = Tensor::zeros((1, 16, 35), DType::I64, &device)?;
    assert_eq!(
        bits(&model.decoder.decode_streaming(&codes)?)?,
        bits(&model.decoder.decode(&codes)?)?
    );
    Ok(())
}

#[test]
#[ignore = "requires local CUDA model and ICL prompt; run serially"]
fn decoder_graph_survives_cancelled_session() -> Result<()> {
    let path = std::env::var("QWEN_CP_TEST_MODEL")?;
    let prompt_path = std::env::var("QWEN_CP_TEST_PROMPT")?;
    anyhow::ensure!(Path::new(&path).is_dir(), "local model required");
    let model = Qwen3TTS::from_pretrained(&path, Device::new_cuda(0)?)?;
    let prompt = VoiceClonePrompt::load(Path::new(&prompt_path), model.device())?;
    let options = SynthesisOptions {
        seed: Some(42),
        chunk_frames: 10,
        ..Default::default()
    };
    let text = "山风吹过松林，星光照亮归途。";
    let mut cancelled = model.synthesize_voice_clone_streaming(
        text,
        &prompt,
        Language::Chinese,
        options.clone(),
    )?;
    for _ in 0..4 {
        assert!(cancelled.next_chunk()?.is_some());
    }
    assert!(cancelled.next_chunk_with_cancel(|| true)?.is_none());
    drop(cancelled);
    let mut repeated = Vec::new();
    for _ in 0..2 {
        let mut stream = model.synthesize_voice_clone_streaming(
            text,
            &prompt,
            Language::Chinese,
            options.clone(),
        )?;
        let mut samples = Vec::new();
        while let Some(audio) = stream.next_chunk()? {
            samples.extend(audio.samples.into_iter().map(f32::to_bits));
        }
        assert!(stream.is_done());
        assert_eq!(samples.len(), stream.frames_generated() * SAMPLES_PER_FRAME);
        repeated.push(samples);
    }
    assert_eq!(repeated[0], repeated[1]);
    assert!(model.decoder.recorded_shapes() > 0);
    Ok(())
}

/// Measures the dispatch headroom of the two loops left after the code
/// predictor frame graph: one talker step at a fixed KV length, and the
/// 12 Hz decoder at the shapes streaming chunks and full synthesis use.
///
/// Evaluation only. Both paths rewrite the same KV position, every replay
/// must reproduce the eager result bit for bit, and the reported numbers
/// are per-call wall time rather than a synthesis benchmark.
#[test]
#[ignore = "requires a local CUDA model; run serially"]
#[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
fn cuda_graph_headroom_probe() -> Result<()> {
    use anyhow::Context;
    use std::time::Instant;

    /// Per-call wall time in milliseconds, one untimed call first.
    fn per_call_ms(
        device: &Device,
        calls: usize,
        mut run: impl FnMut(usize) -> Result<Tensor>,
    ) -> Result<f64> {
        run(0)?;
        device.synchronize()?;
        let started = Instant::now();
        for call in 1..=calls {
            run(call)?;
        }
        device.synchronize()?;
        Ok(started.elapsed().as_secs_f64() * 1000.0 / calls as f64)
    }

    let model_path = std::env::var("QWEN_CP_TEST_MODEL").context("QWEN_CP_TEST_MODEL")?;
    anyhow::ensure!(Path::new(&model_path).is_dir(), "local model required");
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &device else {
        unreachable!("new_cuda returns a cuda device")
    };
    let model = Qwen3TTS::from_pretrained(&model_path, device.clone())?;

    // A talker step is only shape-stable at a fixed KV length. Every stage
    // below reads its own cache set warmed to the same offset, because a
    // set that a warm-up already advanced would hand the capture a longer
    // KV length than the one recorded.
    let offset = 30usize;
    let step_input = model
        .talker
        .get_codec_embedding_from_tensor(&Tensor::new(&[7u32], &device)?)?;
    let warm_caches = || -> Result<Vec<AnyKVCache>> {
        let mut caches = model.talker.new_kv_caches(offset + 64);
        for filled in 0..offset {
            model
                .talker
                .generate_step_with_embed(&step_input, &mut caches, filled)?;
        }
        Ok(caches)
    };
    let talker_step = |caches: &mut [AnyKVCache], at: usize| {
        model
            .talker
            .generate_step_with_embed(&step_input, caches, at)
            .map(|(_, logits)| logits)
    };
    let measure_eager = || -> Result<f64> {
        let mut samples = Vec::new();
        for _ in 0..12 {
            let mut caches = warm_caches()?;
            device.synchronize()?;
            let started = Instant::now();
            talker_step(&mut caches, offset)?;
            device.synchronize()?;
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(samples.iter().sum::<f64>() / samples.len() as f64)
    };
    let eager_a = measure_eager()?;
    let mut verify_caches = warm_caches()?;
    let expected = talker_step(&mut verify_caches, offset)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut capture_caches = warm_caches()?;
    // The warm-up has to run under the guard, on a cache set the capture
    // has not advanced, so the host data the step uploads is cached.
    let guard = cuda.enable_cuda_graph_htod_cache();
    let mut warmup_caches = warm_caches()?;
    let warmup = talker_step(&mut warmup_caches, offset)?;
    let talker_record = crate::cuda_graph::replay::Replay::record(
        cuda,
        &[&step_input],
        &warmup,
        vec![],
        |inputs| {
            model
                .talker
                .generate_step_with_embed(&inputs[0], &mut capture_caches, offset)
                .map(|(_, logits)| logits)
                .map_err(|error| candle_core::Error::Msg(error.to_string()))
        },
    );
    drop(guard);
    let talker_replay = match talker_record {
        Ok(replay) => replay,
        Err(error) => {
            println!(
                "{}",
                serde_json::json!({"probe": "talker_step", "captured": false, "error": error.to_string()})
            );
            return Ok(());
        }
    };
    let replayed = talker_replay
        .launch(&[&step_input])?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(replayed, expected, "talker replay differs from eager");
    let replay_a = per_call_ms(&device, 12, |_| {
        talker_replay
            .launch(&[&step_input])
            .map_err(anyhow::Error::from)
    })?;
    let eager_b = measure_eager()?;
    let replay_b = per_call_ms(&device, 12, |_| {
        talker_replay
            .launch(&[&step_input])
            .map_err(anyhow::Error::from)
    })?;
    println!(
        "{}",
        serde_json::json!({"probe": "talker_step", "captured": true,
            "recorded_kv_len": offset + 1, "bitwise_equal": true,
            "eager_ms": [eager_a, eager_b], "replay_ms": [replay_a, replay_b]})
    );
    drop(talker_replay);

    // Streaming decodes 25 frames of left context plus 10 new frames on
    // every chunk after the first; full synthesis decodes all frames once.
    for frames in [35usize, 185] {
        let codes = Tensor::from_vec(
            (0..16 * frames)
                .map(|index| (index % 2048) as i64)
                .collect(),
            (1, 16, frames),
            &device,
        )?;
        let decode_eager = || model.decoder.decode(&codes);
        let eager_a = per_call_ms(&device, 6, |_| decode_eager())?;
        let guard = cuda.enable_cuda_graph_htod_cache();
        let warmup = decode_eager()?;
        let decode_record =
            crate::cuda_graph::replay::Replay::record(cuda, &[&codes], &warmup, vec![], |inputs| {
                model
                    .decoder
                    .decode(&inputs[0])
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))
            });
        drop(guard);
        let decode_replay = match decode_record {
            Ok(replay) => replay,
            Err(error) => {
                println!(
                    "{}",
                    serde_json::json!({"probe": "codec_decode", "frames": frames,
                        "captured": false, "error": error.to_string()})
                );
                continue;
            }
        };
        let expected = warmup
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let replayed = decode_replay
            .launch(&[&codes])?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(replayed, expected, "decode replay differs from eager");
        let replay_a = per_call_ms(&device, 6, |_| {
            decode_replay.launch(&[&codes]).map_err(anyhow::Error::from)
        })?;
        let eager_b = per_call_ms(&device, 6, |_| decode_eager())?;
        let replay_b = per_call_ms(&device, 6, |_| {
            decode_replay.launch(&[&codes]).map_err(anyhow::Error::from)
        })?;
        println!(
            "{}",
            serde_json::json!({"probe": "codec_decode", "frames": frames,
                "captured": true, "bitwise_equal": true,
                "eager_ms": [eager_a, eager_b], "replay_ms": [replay_a, replay_b]})
        );
    }
    Ok(())
}

#[test]
#[ignore = "explicit CUDA Base model and cached ICL prompt; no downloads"]
fn decoder_graph_large_clone_preserves_codes_pcm_and_eos() -> Result<()> {
    let path = std::env::var("QWEN_CP_TEST_MODEL")?;
    let prompt_path = std::env::var("QWEN_CP_TEST_PROMPT")?;
    let model = Qwen3TTS::from_pretrained(&path, Device::new_cuda(0)?)?;
    let prompt = VoiceClonePrompt::load(Path::new(&prompt_path), model.device())?;
    let options = SynthesisOptions {
        seed: Some(7_330_414_270_409_262_323),
        chunk_frames: 20,
        repetition_penalty: 1.5,
        max_length: 375,
        ..Default::default()
    };
    let render = |eager_decoder| -> Result<_> {
        let mut stream = model.synthesize_voice_clone_streaming(
            "你好，欢迎收听今天的故事。",
            &prompt,
            Language::Chinese,
            options.clone(),
        )?;
        stream.capture_codes_for_test(eager_decoder);
        let mut pcm = Vec::new();
        while let Some(audio) = stream.next_chunk()? {
            anyhow::ensure!(audio.samples.iter().all(|v| v.is_finite()));
            pcm.extend(audio.samples.iter().map(|v| v.to_bits()));
        }
        anyhow::ensure!(stream.is_done(), "missing natural EOS");
        let codes = stream.captured_codes_for_test().to_vec();
        assert_eq!(codes.len(), stream.frames_generated());
        assert_eq!(pcm.len(), codes.len() * SAMPLES_PER_FRAME);
        Ok((codes, pcm))
    };
    let expected = render(true)?;
    for pass in 0..3 {
        assert_eq!(render(false)?, expected, "decode graph pass {pass}");
    }
    assert!(
        model.decoder.recorded_shapes() > 0,
        "incremental vocoder graph did not record"
    );
    eprintln!(
        "{} generated code frames and {} raw PCM F32 patterns match eager across miss/capture/replay; natural EOS observed",
        expected.0.len(),
        expected.1.len()
    );
    Ok(())
}
