//! Export a fresh reference prompt without generating speech or modifying voice caches.
use anyhow::ensure;
use std::path::Path;
use talechime_qwen3_tts::{AudioBuffer, Qwen3TTS};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 5,
        "usage: MODEL REFERENCE TRANSCRIPT DEVICE OUTPUT_JSON"
    );
    let device = talechime_qwen3_tts::parse_device(&args[3])?;
    let model = Qwen3TTS::from_pretrained(&args[0], device)?;
    let audio = AudioBuffer::load(&args[1])?;
    ensure!(
        audio.sample_rate == 24000,
        "probe requires 24kHz reference audio"
    );
    let prompt = model.create_voice_clone_prompt(&audio, Some(&args[2]))?;
    prompt.save(Path::new(&args[4]))?;
    Ok(())
}
