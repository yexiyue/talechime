//! Decode fixed generated codes with full, independent, and left-context chunks.
//! Development-only audition; does not change the streaming implementation.
use anyhow::{Context, ensure};
use candle_core::DType;
use qwen3_tts::{AudioBuffer, Qwen3TTS, VoiceClonePrompt};
use serde_json::json;
use std::{fs, path::Path, time::Instant};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 5,
        "usage: MODEL PROMPT CODES DEVICE OUTPUT_DIR"
    );
    let model = Qwen3TTS::from_pretrained(&args[0], qwen3_tts::parse_device(&args[3])?)?;
    let prompt = VoiceClonePrompt::load(Path::new(&args[1]), model.device())?;
    let generated: Vec<Vec<u32>> = serde_json::from_slice(&fs::read(&args[2])?)?;
    ensure!(!generated.is_empty(), "empty generated codes");
    let mut combined = prompt
        .ref_codes
        .context("ICL reference codes required")?
        .to_dtype(DType::U32)?
        .to_vec2::<u32>()?;
    let reference_len = combined.len();
    combined.extend(generated.iter().cloned());
    let directory = Path::new(&args[4]);
    fs::create_dir_all(directory)?;
    let mut records = Vec::new();
    for (name, chunk_size, left_context) in [
        ("full", generated.len(), 0),
        ("independent", 10, 0),
        ("context25", 10, 25),
    ] {
        model.device().synchronize()?;
        let started = Instant::now();
        let mut samples = Vec::new();
        for start in (0..generated.len()).step_by(chunk_size) {
            let end = (start + chunk_size).min(generated.len());
            let absolute_start = reference_len + start;
            let context_start = if start == 0 {
                0
            } else {
                absolute_start.saturating_sub(left_context)
            };
            let frames = &combined[context_start..reference_len + end];
            let audio = model.decode_codes(frames)?;
            ensure!(audio.sample_rate == 24000, "unexpected sample rate");
            ensure!(
                audio.len() == frames.len() * 1920,
                "unexpected samples per frame"
            );
            let cut = (absolute_start - context_start) * 1920;
            samples.extend_from_slice(&audio.samples[cut..]);
        }
        model.device().synchronize()?;
        let seconds = started.elapsed().as_secs_f64();
        ensure!(samples.len() == generated.len() * 1920, "incomplete output");
        ensure!(samples.iter().all(|v| v.is_finite()), "non-finite audio");
        let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(directory.join(format!("{name}.f32le")), bytes)?;
        AudioBuffer::new(samples, 24000).save(directory.join(format!("{name}.wav")))?;
        records.push(json!({"name": name, "decode_seconds": seconds,
            "chunk_frames": chunk_size, "left_context_frames": left_context,
            "frames": generated.len(), "reference_frames": reference_len}));
    }
    fs::write(
        directory.join("summary.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    Ok(())
}
