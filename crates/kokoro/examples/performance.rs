//! Fixed official inputs; no downloads. Save complete PCM outside timed regions.
use candle_core::{DType, Device, Result};
use std::{path::PathBuf, time::Instant};
use talechime_kokoro::{ExcitationNoise, Model, SynthesisOptions, Trace};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 4 {
        candle_core::bail!("usage: performance OFFICIAL_EXPORT cpu|cuda|metal NEW_OUTPUT_DIR RUNS");
    }
    let root = PathBuf::from(&args[0]);
    let destination = PathBuf::from(&args[2]);
    let runs: usize = args[3]
        .to_string_lossy()
        .parse()
        .map_err(candle_core::Error::wrap)?;
    if !(1..=20).contains(&runs) {
        candle_core::bail!("expected 1..=20 measured runs");
    }
    let device_label = args[1].to_string_lossy();
    let device = match device_label.as_ref() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "cuda")]
        "cuda" => Device::new_cuda(0)?,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal" => Device::new_metal(0)?,
        other => candle_core::bail!("unavailable device {other}"),
    };
    // Refuse to overwrite an earlier experiment, including its raw outputs.
    std::fs::create_dir(&destination).map_err(candle_core::Error::wrap)?;
    let model = Model::load(
        &root.join("results/weights-folded.safetensors"),
        &root.join("config.json"),
        &device,
    )?;
    let reference =
        candle_core::safetensors::load(root.join("results/official-f32.safetensors"), &device)?;
    let get = |name: &str| {
        reference
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing fixture {name}")))
    };
    let ids = get("input_ids")?
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let noise = ExcitationNoise {
        initial_phase: get("random.0.rand")?.clone(),
        gaussian: get("random.1.randn_like")?.clone(),
    };
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("results/report.json")).map_err(candle_core::Error::wrap)?,
    )
    .map_err(candle_core::Error::wrap)?;
    let options = SynthesisOptions {
        speed: metadata["speed"].as_f64().unwrap_or(1.0) as f32,
        ..Default::default()
    };
    let forward = || {
        model.forward(
            &ids,
            get("style")?,
            &options,
            Some(&noise),
            &mut Trace::default(),
        )
    };
    let warmup = forward()?;
    let expected = warmup.audio.to_vec1::<f32>()?;
    let mut seconds = Vec::new();
    for round in 0..runs {
        let start = Instant::now();
        let output = forward()?;
        // Includes completion of all device work and acquisition of host PCM.
        let pcm = output.audio.to_vec1::<f32>()?;
        seconds.push(start.elapsed().as_secs_f64());
        if output.durations != warmup.durations || !same_bits(&pcm, &expected) {
            candle_core::bail!("repeated request differs at round {round}");
        }
        let bytes: Vec<_> = pcm.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(destination.join(format!("round-{round}.f32le")), bytes)
            .map_err(candle_core::Error::wrap)?;
    }
    // Interrupt after encoder stages, then require exact recovery on this owner.
    let checks = std::cell::Cell::new(0);
    let interrupted = model.forward_cancellable(
        &ids,
        get("style")?,
        &options,
        Some(&noise),
        &mut Trace::default(),
        &|| {
            checks.set(checks.get() + 1);
            checks.get() == 3
        },
    );
    match interrupted {
        Err(error) if error.to_string().contains("synthesis cancelled") => {}
        _ => candle_core::bail!("expected stage cancellation"),
    }
    let recovered = forward()?;
    if recovered.durations != warmup.durations || !same_bits(&recovered.audio.to_vec1()?, &expected)
    {
        candle_core::bail!("cancelled owner did not recover bitwise");
    }
    let report = serde_json::json!({
        "device": device_label, "fixture": root.canonicalize().map_err(candle_core::Error::wrap)?,
        "warmups": 1, "seconds": seconds, "samples": expected.len(),
        "audio_seconds": expected.len() as f64 / 24000.0,
        "durations": warmup.durations, "same_owner_repeat_and_cancel_recovery": true,
        "rayon_num_threads": std::env::var("RAYON_NUM_THREADS").ok(),
        "scope": "fixed official ids/style/excitation; acoustic forward through host PCM; excludes load, frontend, output validation and file writes",
    });
    std::fs::write(
        destination.join("report.json"),
        serde_json::to_vec_pretty(&report).map_err(candle_core::Error::wrap)?,
    )
    .map_err(candle_core::Error::wrap)?;
    println!("{report}");
    Ok(())
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits())
}
