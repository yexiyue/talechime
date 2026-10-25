//! Development-only adapter for the pinned retired C++ reference, outside production.
use anyhow::{Context, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};
use voxcpm_sys::{Device, Model, Outcome, Reference};

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(args.len() == 3, "REQUEST.json OUTPUT_DIRECTORY");
    let request: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let output = Path::new(&args[2]);
    let config = &request["request"];
    let field = |key: &str| -> anyhow::Result<&str> {
        config[key]
            .as_str()
            .with_context(|| format!("missing {key}"))
    };
    ensure!(
        config["native_seed"] == 42
            && config["steps"] == 10
            && config["cfg"] == 2.0
            && config["temperature"] == 1.0
            && config["max_frames"] == 200,
        "adapter requires fixed native bridge defaults"
    );
    let clock = Instant::now();
    let mut sequence = 0;
    let mut event = |kind: &str, payload: Value| -> anyhow::Result<()> {
        println!(
            "{}",
            json!({"schema":"talechime/performance/1", "attempt_id":request["attempt_id"],
            "sequence":sequence,"offset_seconds":clock.elapsed().as_secs_f64(),"kind":kind,"payload":payload})
        );
        std::io::stdout().flush()?;
        sequence += 1;
        Ok(())
    };
    let root = Path::new(field("model")?);
    let started = Instant::now();
    let mut model = Model::load(
        &root.join("VoxCPM2-BaseLM-Q8_0.gguf"),
        &root.join("VoxCPM2-Acoustic-F16.gguf"),
        Device::Cuda,
    )
    .map_err(anyhow::Error::msg)?;
    let load = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let voice: Value = serde_json::from_slice(&fs::read(field("voice_metadata")?)?)?;
    let transcript = voice["transcript"]
        .as_str()
        .context("voice transcript missing")?;
    let reference = talechime_backends::reference::load(Path::new(field("reference_wav")?), 16000)?;
    let features = model
        .encode_reference(&reference, 16000)
        .map_err(anyhow::Error::msg)?;
    let preparation = started.elapsed().as_secs_f64();
    let runtime = json!({"engine":"pinned llama.cpp-omni via retired voxcpm-sys",
        "device":"cuda","native_seed":42,"steps":10,"cfg":2.0,"temperature":1.0,"max_frames":200,
        "load_seconds":load,"reference_prepare_seconds":preparation,
        "reference_boundary":"prepared once before warmup; cached hot requests",
        "delivery":"native streaming callback; full PCM collection before completion"});
    fs::write(
        output.join("runtime.json"),
        serde_json::to_vec_pretty(&runtime)?,
    )?;
    event("prepared", runtime)?;
    let text = fs::read_to_string(field("text_file")?)?;
    let warmup = request["warmup_rounds"].as_u64().context("warmup rounds")?;
    let hot = request["hot_rounds"].as_u64().context("hot rounds")?;
    for index in 0..warmup + hot {
        let mut pcm = Vec::new();
        let mut first = None;
        let started = Instant::now();
        let outcome = model
            .generate(
                text.trim(),
                Some(Reference {
                    samples: &features,
                    sample_rate: 16000,
                    transcript,
                    encoded: true,
                }),
                200,
                |chunk| {
                    if pcm.len() + chunk.len() > 48_000 * 30 {
                        return false;
                    }
                    if !chunk.is_empty() {
                        first.get_or_insert(started.elapsed().as_secs_f64());
                    }
                    pcm.extend_from_slice(chunk);
                    true
                },
            )
            .map_err(anyhow::Error::msg)?;
        let elapsed = started.elapsed().as_secs_f64();
        ensure!(
            outcome == Outcome::Complete
                && !pcm.is_empty()
                && pcm.iter().all(|sample| sample.is_finite())
                && pcm.iter().any(|sample| *sample != 0.0),
            "invalid native completion: {outcome:?}"
        );
        let filename = format!("round-{index}-f32.wav");
        let path = output.join(&filename);
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 48000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?;
        for sample in &pcm {
            writer.write_sample(*sample)?;
        }
        writer.finalize()?;
        ensure!(
            hound::WavReader::open(&path)?.duration() as usize == pcm.len(),
            "WAV length mismatch"
        );
        let bytes: Vec<_> = pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect();
        event(
            "round",
            json!({"index":index,"warmup":index<warmup,"outcome":"success",
            "synthesis_seconds":elapsed,"first_pcm_seconds":first,"audio_seconds":pcm.len() as f64/48000.0,
            "delivery_boundary":"native_streaming_first_pcm","output_sha256":digest(&bytes),"observations":[],
            "artifacts":[{"path":filename,"sha256":digest(&fs::read(&path)?),"bytes":fs::metadata(&path)?.len(),
                "kind":"audio_f32","derived_from":null}]}),
        )?;
    }
    event("terminal", json!({"outcome":"success"}))
}
