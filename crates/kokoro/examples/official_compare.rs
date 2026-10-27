//! Run only against explicitly supplied local official-model fixtures.
use candle_core::{DType, Device, Result};
use std::{path::PathBuf, time::Instant};
use talechime_kokoro::{ExcitationNoise, Model, SynthesisOptions, Trace};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let root = args.next().map(PathBuf::from).ok_or_else(|| {
        candle_core::Error::Msg("usage: official_compare EXPORTED_MODEL_DIR".into())
    })?;
    let device_name = args.next().unwrap_or_else(|| "cpu".into());
    let benchmark_runs = args
        .next()
        .map(|v| v.to_string_lossy().parse::<usize>())
        .transpose()
        .map_err(candle_core::Error::wrap)?
        .unwrap_or(0);
    if benchmark_runs > 20 {
        candle_core::bail!("benchmark run limit is 20");
    }
    let device_label = device_name
        .to_str()
        .ok_or_else(|| candle_core::Error::Msg("invalid comparison device".into()))?;
    let device = match device_label {
        "cpu" => Device::Cpu,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal" => Device::new_metal(0)?,
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        "cuda" => Device::new_cuda(0)?,
        other => candle_core::bail!("unavailable comparison device {other}"),
    };
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
    let noise = ExcitationNoise {
        initial_phase: get("random.0.rand")?.clone(),
        gaussian: get("random.1.randn_like")?.clone(),
    };
    let ids = get("input_ids")?
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let mut trace = Trace::enabled();
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("results/report.json")).map_err(candle_core::Error::wrap)?,
    )
    .map_err(candle_core::Error::wrap)?;
    let options = SynthesisOptions {
        speed: metadata
            .get("speed")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0) as f32,
        ..Default::default()
    };
    let start = Instant::now();
    let result = model.forward(&ids, get("style")?, &options, Some(&noise), &mut trace)?;
    eprintln!(
        "native elapsed {:?}, samples {}",
        start.elapsed(),
        result.audio.elem_count()
    );
    let exported: std::collections::HashMap<_, _> = trace
        .tensors()
        .into_iter()
        .flatten()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    candle_core::safetensors::save(
        &exported,
        root.join(format!("results/native-{device_label}-f32.safetensors")),
    )?;
    let mut failed = false;
    for (name, tensor) in trace.tensors().into_iter().flatten() {
        let Some(expected) = reference.get(name) else {
            continue;
        };
        let a = tensor.flatten_all()?.to_vec1::<f32>()?;
        let b = expected.flatten_all()?.to_vec1::<f32>()?;
        if tensor.dims() != expected.dims() {
            eprintln!(
                "{name}: shape mismatch {:?} {:?}",
                tensor.dims(),
                expected.dims()
            );
            failed = true;
            continue;
        }
        let mut max = 0f32;
        let mut mse = 0f64;
        let mut power = 0f64;
        for (a, b) in a.iter().zip(&b) {
            let error = a - b;
            max = max.max(error.abs());
            mse += f64::from(error).powi(2);
            power += f64::from(*b).powi(2);
        }
        let snr = 10.0 * (power / mse).log10();
        println!(
            "{name}: max_abs={max:.8e} rmse={:.8e} snr_db={snr:.3}",
            (mse / a.len() as f64).sqrt()
        );
        if !snr.is_finite() && mse != 0.0 || snr < 40.0 {
            failed = true;
        }
    }
    let expected_durations = get("durations")?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
    println!("duration_equal={}", result.durations == expected_durations);
    if failed || result.durations != expected_durations {
        candle_core::bail!(
            "official numerical comparison failed (40 dB stage SNR / exact durations)"
        );
    }
    if benchmark_runs > 0 {
        model.forward(
            &ids,
            get("style")?,
            &options,
            Some(&noise),
            &mut Trace::default(),
        )?;
        let mut seconds = Vec::new();
        for _ in 0..benchmark_runs {
            let start = Instant::now();
            let output = model.forward(
                &ids,
                get("style")?,
                &options,
                Some(&noise),
                &mut Trace::default(),
            )?;
            seconds.push(start.elapsed().as_secs_f64());
            assert_eq!(output.durations, result.durations);
        }
        let report = serde_json::json!({
            "precision": "float32",
            "device": device_label,
            "placement": "CPU encoders/prosody/spectral; selected decoder device",
            "noise": "independently captured official draws",
            "speed": options.speed,
            "warmups": 1,
            "measured_runs": benchmark_runs,
            "traced": false,
            "seconds": seconds,
            "audio_samples": result.audio.elem_count(),
            "sample_rate": 24000,
            "veclib_maximum_threads": std::env::var("VECLIB_MAXIMUM_THREADS").ok(),
            "rayon_num_threads": std::env::var("RAYON_NUM_THREADS").ok(),
        });
        std::fs::write(
            root.join(format!("results/native-benchmark-{device_label}.json")),
            serde_json::to_vec_pretty(&report).map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut wav = hound::WavWriter::create(
            root.join(format!("results/native-{device_label}.wav")),
            spec,
        )
        .map_err(candle_core::Error::wrap)?;
        for value in result.audio.to_vec1::<f32>()? {
            wav.write_sample(value).map_err(candle_core::Error::wrap)?;
        }
        wav.finalize().map_err(candle_core::Error::wrap)?;
        println!("benchmark: {report}");
    }
    Ok(())
}
