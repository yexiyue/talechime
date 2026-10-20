//! Full Rust generation with official prompt fixtures, no Python inference.
//! Usage: `cosyvoice3_synthesize <converted-model> <reference-fixtures> [metal]`
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use std::{path::Path, time::Instant};
use talechime_cosyvoice3::{Mode, PromptFeatures, SamplingConfig, pipeline::CosyVoice3};
fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let model = Path::new(args.get(1).context("missing converted model dir")?);
    let dir = Path::new(args.get(2).context("missing reference fixture dir")?);
    let dev = if args.get(3).map(String::as_str) == Some("metal") {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("metadata.json"))?)?;
    let f = candle_core::safetensors::load(dir.join("frontend.safetensors"), &Device::Cpu)?;
    let prompt = PromptFeatures {
        text: meta["prompt_text"]
            .as_str()
            .context("missing prompt_text")?
            .into(),
        speech_tokens: f["prompt_speech_token"]
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1()?,
        mel: f["prompt_speech_feat"].flatten_all()?.to_vec1()?,
        mel_frames: f["prompt_speech_feat"].dim(1)?,
        embedding: f["prompt_embedding"].flatten_all()?.to_vec1()?,
    };
    let t = Instant::now();
    let mut cv = CosyVoice3::load(model, &dev, DType::F32, 0)?;
    println!("load_seconds={:.3}", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let pcm = cv.synthesize(
        meta["text"].as_str().context("missing text")?,
        &prompt,
        Mode::ZeroShot,
        None,
        &SamplingConfig {
            seed: Some(42),
            ..SamplingConfig::default()
        },
        1.0,
    )?;
    let seconds = t.elapsed().as_secs_f64();
    let audio_seconds = pcm.len() as f64 / cv.sample_rate() as f64;
    anyhow::ensure!(
        !pcm.is_empty() && pcm.iter().all(|s| s.is_finite()),
        "invalid generated PCM"
    );
    let mut writer = hound::WavWriter::create(
        dir.join("rust-generated.wav"),
        hound::WavSpec {
            channels: 1,
            sample_rate: cv.sample_rate() as u32,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for s in &pcm {
        writer.write_sample(*s)?;
    }
    writer.finalize()?;
    println!(
        "generation_seconds={seconds:.3} audio_seconds={audio_seconds:.3} RTF={:.3}",
        seconds / audio_seconds
    );
    Ok(())
}
