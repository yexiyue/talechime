//! Explicit local diagnostic; synchronized spans are not production timings.
use super::*;
use std::{collections::BTreeMap, time::Instant};

#[test]
#[ignore = "explicit local Nano CUDA model; no downloads"]
fn real_cuda_generated_embedding_reuse_matches_generic_rows() -> anyhow::Result<()> {
    let directory = std::env::var_os("TALECHIME_NANO_PROFILE_MODEL")
        .ok_or_else(|| anyhow::anyhow!("TALECHIME_NANO_PROFILE_MODEL required"))?;
    let device = Device::new_cuda(0)?;
    let model = Nano::load(&std::path::PathBuf::from(directory).join("tts"), &device)?;
    let slot = model.text.forward(&Tensor::new(&[[9u32]], &device)?)?;
    // Cover every entry in each audio codebook, plus uniform boundary rows.
    for offset in 0..1026 {
        let mut row = vec![9];
        let mut reused = slot.clone();
        for channel in 0..16 {
            let code = match offset {
                1024 => 0,
                1025 => 1023,
                _ => (offset + 37 * channel) % 1024,
            } as u32;
            row.push(code);
            let embedding = model.audio[channel].forward(&Tensor::new(&[[code]], &device)?)?;
            reused = (&reused + embedding)?;
        }
        let generic = model.embeds(&[row])?.flatten_all()?.to_vec1::<f32>()?;
        let reused = reused.flatten_all()?.to_vec1::<f32>()?;
        assert!(generic.iter().all(|v| v.is_finite()));
        assert_eq!(
            reused.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            generic.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "generated embedding differs at row {offset}"
        );
    }
    eprintln!(
        "1026 generated rows match generic embeddings bitwise; all 1024 entries of each of 16 codebooks covered"
    );
    Ok(())
}

#[test]
#[ignore = "explicit Nano CUDA model, prompt rows and new output path; no downloads"]
fn real_cuda_generation_stage_profile_preserves_codes_and_pcm() -> anyhow::Result<()> {
    let required = |name| {
        std::env::var_os(name)
            .map(std::path::PathBuf::from)
            .ok_or_else(|| anyhow::anyhow!("{name} required"))
    };
    let directory = required("TALECHIME_NANO_PROFILE_MODEL")?;
    let input = required("TALECHIME_NANO_PROFILE_ROWS")?;
    let output = required("TALECHIME_NANO_PROFILE_OUTPUT")?;
    anyhow::ensure!(!output.exists(), "output already exists");
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(input)?)?;
    let device = Device::new_cuda(0)?;
    let mut model = Nano::load(&directory.join("tts"), &device)?;
    let mut codec = crate::codec::AudioCodec::load(&directory.join("codec"), &device, DType::F32)?;
    let seed = 3_560_097_011;
    let mut rounds = Vec::new();
    for round in 0..6 {
        let mut expected_codes = Vec::new();
        device.synchronize()?;
        let started = Instant::now();
        model.generate(&rows, 375, seed, &|| false, |frame| {
            expected_codes.push(frame);
            Ok(())
        })?;
        codec.reset_decoder();
        let mut expected_pcm = Vec::new();
        for batch in expected_codes.chunks(3) {
            expected_pcm.extend(codec.decode(batch, &|| false)?);
        }
        device.synchronize()?;
        let baseline_seconds = started.elapsed().as_secs_f64();

        let mut stages: BTreeMap<&str, (u64, f64)> = BTreeMap::new();
        macro_rules! stage {
            ($name:literal, $operation:expr) => {{
                device.synchronize()?;
                let started = Instant::now();
                let result = $operation;
                device.synchronize()?;
                let result = result?;
                let entry = stages.entry($name).or_default();
                entry.0 += 1;
                entry.1 += started.elapsed().as_secs_f64();
                result
            }};
        }
        // Follow generate's operation order, retaining embeddings for the next
        // global input. Only timing/synchronization is added.
        model.global.reset();
        codec.reset_decoder();
        let started = Instant::now();
        let mut x = stage!("prompt_embed", model.embeds(&rows));
        let mut history = vec![Vec::new(); 16];
        let mut rng = sampling::Sampler::new(seed);
        let mut codes = Vec::new();
        let mut pcm = Vec::new();
        let mut pending = Vec::new();
        let mut eos = false;
        for _ in 0..375 {
            let hidden = stage!("global", model.global.forward(&x, &|| false));
            model.local.reset();
            let hidden = hidden.i((.., hidden.dim(1)? - 1, ..))?.unsqueeze(1)?;
            let local = stage!("local_text", model.local.forward(&hidden, &|| false)).squeeze(1)?;
            let logits = stage!("text_head", model.text_head.forward(&local)).flatten_all()?;
            let candidates = stage!(
                "text_candidates",
                Tensor::cat(&[logits.narrow(0, 9, 1)?, logits.narrow(0, 7, 1)?], 0)
            );
            let token = stage!("text_sample", rng.sample(&candidates, &[], 1., 2, 1., 1.));
            if token == 1 {
                eos = true;
                break;
            }
            let mut local_input = stage!(
                "slot_embed",
                model.text.forward(&Tensor::new(&[[9u32]], &device)?)
            );
            let mut next_global = local_input.clone();
            let mut frame = Vec::with_capacity(16);
            for (channel, previous) in history.iter_mut().enumerate() {
                let local = stage!("local_audio", model.local.forward(&local_input, &|| false))
                    .squeeze(1)?;
                let logits = stage!("audio_head", model.audio_heads[channel].forward(&local));
                let code = stage!(
                    "audio_sample",
                    rng.sample(&logits, previous, 0.8, 25, 0.95, 1.2)
                );
                previous.push(code);
                frame.push(code);
                local_input = stage!(
                    "audio_embed",
                    model.audio[channel].forward(&Tensor::new(&[[code]], &device)?)
                );
                next_global = stage!("row_accumulate", &next_global + &local_input);
            }
            pending.push(frame.clone());
            codes.push(frame);
            if pending.len() == 3 {
                pcm.extend(stage!("codec", codec.decode(&pending, &|| false)));
                pending.clear();
            }
            x = next_global;
        }
        anyhow::ensure!(eos && !codes.is_empty(), "no nonempty natural EOS");
        if !pending.is_empty() {
            pcm.extend(stage!("codec", codec.decode(&pending, &|| false)));
        }
        device.synchronize()?;
        let diagnostic_seconds = started.elapsed().as_secs_f64();
        assert_eq!(
            codes, expected_codes,
            "synchronized diagnostic changed codes"
        );
        assert_eq!(
            pcm.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            expected_pcm.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            "synchronized diagnostic changed PCM"
        );
        rounds.push(serde_json::json!({
            "index":round,"warmup":round==0,"baseline_seconds":baseline_seconds,
            "diagnostic_seconds":diagnostic_seconds,"stages":stages,
            "code_frames":codes.len(),"pcm_samples":pcm.len(),"eos":true,
            "codes_equal":true,"pcm_bits_equal":true
        }));
    }
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&serde_json::json!({
            "kind":"synchronized_model_stage_diagnostic_not_production",
            "seed":seed,"streaming_codec_frames":3,"sample_rate":codec.sample_rate,
            "channels":codec.channels,"rounds":rounds
        }))?,
    )?;
    Ok(())
}

#[test]
#[ignore = "explicit local Nano CUDA model and prompt rows; no downloads"]
fn real_cuda_unmasked_decode_preserves_codes_pcm_and_eos() -> anyhow::Result<()> {
    let directory = std::path::PathBuf::from(
        std::env::var_os("TALECHIME_NANO_PROFILE_MODEL")
            .ok_or_else(|| anyhow::anyhow!("TALECHIME_NANO_PROFILE_MODEL required"))?,
    );
    let rows = std::env::var_os("TALECHIME_NANO_PROFILE_ROWS")
        .ok_or_else(|| anyhow::anyhow!("TALECHIME_NANO_PROFILE_ROWS required"))?;
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(rows)?)?;
    let device = Device::new_cuda(0)?;
    let mut model = Nano::load(&directory.join("tts"), &device)?;
    let mut codec = crate::codec::AudioCodec::load(&directory.join("codec"), &device, DType::F32)?;
    let mut generate = |seed| -> anyhow::Result<_> {
        let mut codes = Vec::new();
        // generate returns success only on model EOS.
        model.generate(&rows, 375, seed, &|| false, |frame| {
            codes.push(frame);
            Ok(())
        })?;
        anyhow::ensure!(!codes.is_empty());
        codec.reset_decoder();
        let mut pcm = Vec::new();
        for chunk in codes.chunks(3) {
            pcm.extend(codec.decode(chunk, &|| false)?);
        }
        anyhow::ensure!(pcm.iter().all(|x| x.is_finite()));
        Ok((codes, pcm.into_iter().map(f32::to_bits).collect::<Vec<_>>()))
    };
    for seed in [3_560_097_011, 42, 7] {
        let expected = crate::transformer::cuda_mask_tests::with_original_mask(|| generate(seed))?;
        let actual = generate(seed)?;
        assert_eq!(actual, expected, "seed={seed}");
        eprintln!(
            "seed={seed}: {} code frames and {} raw PCM samples match original mask; EOS observed",
            actual.0.len(),
            actual.1.len()
        );
    }
    Ok(())
}
