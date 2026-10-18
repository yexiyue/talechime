//! Serial migration benchmark with the production resampler and segmentation.
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
use voxcpm::{Model, Options, Outcome, SAMPLE_RATE};
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 5,
        "MODEL_DIRECTORY cpu|cuda OUTPUT_DIRECTORY TEXT_FILE [REFERENCE.wav [TRANSCRIPT_FILE]]"
    );
    let device_started = Instant::now();
    let device = match args[2].as_str() {
        "cpu" => candle_core::Device::Cpu,
        "cuda" => candle_core::Device::new_cuda(0)?,
        "metal" => candle_core::Device::new_metal(0)?,
        _ => anyhow::bail!("invalid device"),
    };
    let device_init = device_started.elapsed();
    let output = PathBuf::from(&args[3]);
    std::fs::create_dir_all(&output)?;
    let text = std::fs::read_to_string(&args[4])?;
    let segments = tts_core::text::preprocess_text(&text, 180);
    let start = Instant::now();
    let precision = std::env::var("VOXCPM_BENCH_PRECISION").unwrap_or_else(|_| "q8".into());
    let mut model = match precision.as_str() {
        "q8" => Model::load(Path::new(&args[1]), &device)?,
        "bf16" | "f16" | "f32" => {
            let dtype = match precision.as_str() {
                "bf16" => candle_core::DType::BF16,
                "f16" => candle_core::DType::F16,
                _ => candle_core::DType::F32,
            };
            Model::load_original(Path::new(&args[1]), &device, dtype, &|| false)?
        }
        _ => anyhow::bail!("precision must be q8, bf16, f16 or f32"),
    };
    device.synchronize()?;
    let load = start.elapsed();
    let start = Instant::now();
    let reference = if args.len() >= 6 {
        let samples = talechime_backends::reference::load(Path::new(&args[5]), 16000)?;
        let transcript = args.get(6).map(std::fs::read_to_string).transpose()?;
        Some(model.reference(&samples, transcript, &|| false)?)
    } else {
        None
    };
    device.synchronize()?;
    let encoding = start.elapsed();
    let options = Options {
        measure_modules: std::env::var_os("VOXCPM_BENCH_PROFILE").is_some(),
        ..Options::default()
    };
    let rounds = std::env::var("VOXCPM_BENCH_ROUNDS")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(5);
    anyhow::ensure!((1..=30).contains(&rounds), "rounds must be 1..30");
    // Rounds before this index still run and are recorded, but stay out of the
    // measured aggregate so warmup effects remain visible in the raw log.
    let warmup_rounds = std::env::var("VOXCPM_BENCH_WARMUP")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    anyhow::ensure!(
        warmup_rounds <= rounds,
        "warmup rounds must not exceed rounds"
    );
    let mut results = Vec::new();
    for round in 0..=rounds {
        let directory = output.join(format!("round-{round}"));
        std::fs::create_dir_all(&directory)?;
        let started = Instant::now();
        let mut first = None;
        let mut count = 0;
        let mut index = Vec::new();
        for (i, segment) in segments.iter().enumerate() {
            let mut samples = Vec::new();
            let status = model.generate(
                &segment.text,
                reference.as_ref(),
                &options,
                &|| false,
                |pcm| {
                    first.get_or_insert(started.elapsed());
                    samples.extend_from_slice(pcm);
                    Ok(())
                },
            )?;
            if options.measure_modules {
                eprintln!(
                    "round {round} segment {} modules: {:?}",
                    i + 1,
                    model.timings()
                );
            }
            anyhow::ensure!(
                status == Outcome::Eos,
                "segment {} did not reach EOS: {status:?}",
                i + 1
            );
            count += samples.len();
            let filename = format!("{:03}.wav", i + 1);
            let mut writer = hound::WavWriter::create(
                directory.join(&filename),
                hound::WavSpec {
                    channels: 1,
                    sample_rate: SAMPLE_RATE,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )?;
            for v in samples {
                writer.write_sample(v)?;
            }
            writer.finalize()?;
            index.push(serde_json::json!({"file":filename,"text":segment.text}));
        }
        device.synchronize()?;
        let elapsed = started.elapsed();
        let seconds = count as f64 / SAMPLE_RATE as f64;
        let graphs = model.cuda_graph_status();
        let result = serde_json::json!({"round":round,"warmup":round < warmup_rounds,"first_pcm_ms":first.map(|v|v.as_millis()),"generate_ms":elapsed.as_millis(),"audio_seconds":seconds,"rtf":elapsed.as_secs_f64()/seconds,"segments":segments.len(),"cuda_graph":graphs.cfm,"cuda_decoder_graph":graphs.decoder});
        println!("{result}");
        results.push(result);
        std::fs::write(
            directory.join("index.json"),
            serde_json::to_vec_pretty(&index)?,
        )?;
    }
    let measured: Vec<f64> = results
        .iter()
        .filter(|round| {
            round["round"]
                .as_u64()
                .is_some_and(|index| index >= warmup_rounds as u64)
        })
        .filter_map(|round| round["rtf"].as_f64())
        .collect();
    let measured_summary = (!measured.is_empty()).then(|| {
        serde_json::json!({
            "rounds": measured.len(),
            "rtf_mean": measured.iter().sum::<f64>() / measured.len() as f64,
            "rtf_min": measured.iter().copied().fold(f64::INFINITY, f64::min),
            "rtf_max": measured
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max),
        })
    });
    let cancelled = std::cell::Cell::new(false);
    let cancel_start = std::cell::Cell::new(None);
    let status = model.generate(
        "山风吹过松林，星光照亮归途。",
        reference.as_ref(),
        &options,
        &|| cancelled.get(),
        |_| {
            cancel_start.set(Some(Instant::now()));
            cancelled.set(true);
            Ok(())
        },
    )?;
    anyhow::ensure!(
        status == Outcome::Cancelled,
        "cancel after first PCM failed"
    );
    let cancel_ms = cancel_start.get().expect("first PCM").elapsed().as_millis();
    let status = model.generate(
        "下一次请求仍然正常。",
        reference.as_ref(),
        &options,
        &|| false,
        |_| Ok(()),
    )?;
    anyhow::ensure!(status == Outcome::Eos, "request after cancellation failed");
    let mut delivered = 0;
    anyhow::ensure!(
        model
            .generate("", None, &options, &|| false, |pcm| {
                delivered += pcm.len();
                Ok(())
            })
            .is_err()
            && delivered == 0,
        "empty text must fail without PCM"
    );
    let invalid = Options {
        temperature: f64::NAN,
        ..options.clone()
    };
    anyhow::ensure!(
        model
            .generate("测试。", None, &invalid, &|| false, |_| Ok(()))
            .is_err(),
        "invalid sampling must fail"
    );
    let limited = Options {
        max_frames: 1,
        ..options.clone()
    };
    anyhow::ensure!(
        model.generate(
            "这是用于检查生成上限的长句子。",
            None,
            &limited,
            &|| false,
            |_| Ok(())
        )? == Outcome::Truncated,
        "frame limit must not report EOS"
    );
    let checks = std::cell::Cell::new(0);
    let mut delivered = 0;
    // Cancellation granularity differs by path: the eager sampler polls the
    // flag inside every transformer layer, while a replayed CUDA graph can only
    // poll it once per frame. Trip the flag inside prefill, before the frame
    // loop, so both paths are held to the same guarantee. Cancelling after the
    // first frame is covered by the earlier callback-driven case.
    let status = model.generate(
        "检查扩散迭代取消。",
        None,
        &options,
        &|| {
            checks.set(checks.get() + 1);
            checks.get() > 8
        },
        |pcm| {
            delivered += pcm.len();
            Ok(())
        },
    )?;
    anyhow::ensure!(
        status == Outcome::Cancelled && delivered == 0,
        "cancel during prefill/CFM must not emit PCM: status={status:?} delivered={delivered}"
    );
    let release = Instant::now();
    drop(reference);
    drop(model);
    device.synchronize()?;
    let release_ms = release.elapsed().as_millis();
    anyhow::ensure!(
        (if precision == "q8" {
            Model::load_with_cancel(Path::new(&args[1]), &device, &|| true)
        } else {
            Model::load_original(
                Path::new(&args[1]),
                &device,
                candle_core::DType::BF16,
                &|| true,
            )
        })
        .is_err(),
        "initialization cancellation must fail loading"
    );
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"precision":precision,"load_ms":load.as_millis(),"device_init_ms":device_init.as_millis(),"cold_load_ms":(load+device_init).as_millis(),"reference_ms":encoding.as_millis(),"cancel_ms":cancel_ms,"release_ms":release_ms,"regressions":"passed","warmup_rounds":warmup_rounds,"measured":measured_summary,"rounds":results}),
        )?,
    )?;
    Ok(())
}
