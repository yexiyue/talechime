//! Measure the public one-shot VoiceDesign API under xtask reference containment.
use anyhow::{Context, ensure};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{io::Write, path::PathBuf, time::Instant};
use talechime_backends::qwen::design;
use talechime_protocol::Device;
use talechime_qwen3_tts::AudioBuffer;

#[derive(Deserialize)]
struct Envelope {
    attempt_id: String,
    warmup_rounds: u32,
    hot_rounds: u32,
    request: Request,
}

#[derive(Deserialize)]
struct Request {
    model: PathBuf,
    text_file: PathBuf,
    description: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 2,
        "expected request JSON and output directory"
    );
    let envelope: Envelope = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let output = PathBuf::from(&args[1]);
    let text = std::fs::read_to_string(&envelope.request.text_file)?;
    let started = Instant::now();
    let mut sequence = 0;
    let mut event = |kind, payload| -> anyhow::Result<()> {
        let mut stdout = std::io::stdout().lock();
        writeln!(
            stdout,
            "{}",
            json!({"schema":"talechime/performance/1",
            "attempt_id":envelope.attempt_id,"sequence":sequence,
            "offset_seconds":started.elapsed().as_secs_f64(),"kind":kind,"payload":payload})
        )?;
        stdout.flush()?;
        sequence += 1;
        Ok(())
    };
    event(
        "prepared",
        json!({"scope":"public VoiceDesign reference API",
        "timing":"each round loads, generates, exports PCM16 and releases model",
        "seed":42,"device":"cuda","first_pcm":"completed reference file",
        "output_hash_domain":"raw F32LE decoded from exported PCM16, no rate/channel prefix"}),
    )?;
    for index in 0..envelope.warmup_rounds + envelope.hot_rounds {
        let name = format!("round-{index}.wav");
        let path = output.join(&name);
        ensure!(!path.exists(), "refuse existing reference output");
        let clock = Instant::now();
        design::reference(
            envelope.request.model.clone(),
            Device::Cuda,
            text.trim().to_owned(),
            envelope.request.description.clone(),
            path.clone(),
            None,
        )
        .await
        .context("production VoiceDesign reference failed")?;
        let elapsed = clock.elapsed().as_secs_f64();
        let audio = AudioBuffer::load(&path)?;
        talechime_backends::qwen::validate_reference(&audio)?;
        let mut pcm_digest = Sha256::new();
        for sample in &audio.samples {
            pcm_digest.update(sample.to_le_bytes());
        }
        let bytes = std::fs::read(&path)?;
        event(
            "round",
            json!({"index":index,"warmup":index<envelope.warmup_rounds,
            "outcome":"success","synthesis_seconds":elapsed,
            "audio_seconds":audio.samples.len() as f64 / f64::from(audio.sample_rate),
            "first_pcm_seconds":elapsed,"delivery_boundary":"segment_delivery_first_pcm",
            "output_sha256":format!("{:x}",pcm_digest.finalize()),"observations":[],
            "artifacts":[{"path":name,"kind":"audio_pcm16","bytes":bytes.len(),
                "sha256":format!("{:x}",Sha256::digest(&bytes)),"derived_from":null}]}),
        )?;
    }
    event("terminal", json!({"outcome":"success"}))
}
