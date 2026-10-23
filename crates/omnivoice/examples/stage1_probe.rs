//! Offline fixed-code decoder comparison. Model loading and WAV export are untimed.
use std::{error::Error, fs, path::PathBuf, time::Instant};

use serde::Deserialize;
use talechime_omnivoice::{
    DeviceSpec, RuntimeOptions,
    artifacts::RuntimeArtifacts,
    contracts::{GeneratedTokens, I64Tensor2},
    stage1_decoder::{Stage1DecoderBundle, Stage1RuntimePlan},
};

#[derive(Deserialize)]
struct Codes {
    codebooks: usize,
    frames: usize,
    values: Vec<i64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("usage: stage1_probe MODEL CODES_JSON cpu|metal OUTPUT_DIR HOT_ROUNDS".into());
    }
    let codes: Codes = serde_json::from_slice(&fs::read(&args[1])?)?;
    let tokens = GeneratedTokens::Single(I64Tensor2::new(
        (codes.codebooks, codes.frames),
        codes.values,
    )?);
    let spec = match args[2].as_str() {
        "cpu" => DeviceSpec::Cpu,
        "metal" => DeviceSpec::Metal,
        _ => return Err("device must be cpu or metal".into()),
    };
    let rounds: usize = args[4].parse()?;
    if !(1..=100).contains(&rounds) {
        return Err("hot rounds must be in 1..=100".into());
    }
    let output = PathBuf::from(&args[3]);
    fs::create_dir_all(&output)?;
    let options = RuntimeOptions::new(&args[0]).with_device(spec);
    let device = options.resolve_device()?;
    let runtime = RuntimeArtifacts::from_model_root(&args[0])?;
    let bundle = Stage1DecoderBundle::from_runtime_artifacts(&runtime)?;
    let plan = Stage1RuntimePlan::from_runtime_artifacts_with_device(
        options,
        &runtime,
        bundle,
        device.clone(),
    )?;
    // The first decode includes lazy weight loading and kernel warmup.
    for index in 0..=rounds {
        device.synchronize()?;
        let started = Instant::now();
        let audio = plan.decode_raw(&tokens, None)?;
        device.synchronize()?;
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "{}",
            serde_json::json!({
                "index": index, "warmup": index == 0, "device": args[2],
                "frames": codes.frames, "codebooks": codes.codebooks,
                "seconds": seconds, "samples": audio.samples.len(),
                "sample_rate": audio.sample_rate,
                "rayon_threads": std::env::var("RAYON_NUM_THREADS").ok(),
                "veclib_threads": std::env::var("VECLIB_MAXIMUM_THREADS").ok(),
            })
        );
        if index == 1 {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: audio.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut wav = hound::WavWriter::create(output.join("raw.wav"), spec)?;
            for sample in &audio.samples {
                wav.write_sample(*sample)?;
            }
            wav.finalize()?;
        }
    }
    Ok(())
}
