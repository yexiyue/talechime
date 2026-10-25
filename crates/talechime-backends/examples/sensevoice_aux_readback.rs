//! Explicit local auxiliary readback, not dual-model qualification or listening acceptance.
use anyhow::{Context, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path, sync::Arc};
use talechime_backends::asr;
use talechime_core::verification::ReadbackAudio;

#[derive(Deserialize)]
struct Case {
    id: String,
    file: String,
    text: String,
    sha256: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    tokio::task::LocalSet::new().run_until(run()).await
}

async fn run() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(
        args.len() == 4,
        "SENSEVOICE_DIRECTORY CASES.json NEW_OUTPUT.json"
    );
    let output = Path::new(&args[3]);
    ensure!(!output.exists(), "readback output already exists");
    let cases_file = Path::new(&args[2]);
    ensure!(
        std::fs::metadata(cases_file)?.len() <= 1024 * 1024,
        "case manifest too large"
    );
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(cases_file)?)?;
    ensure!(
        !cases.is_empty() && cases.len() <= 16,
        "expected 1..=16 explicit clips"
    );
    let recognizer = asr::load_sensevoice(Path::new(&args[1]), 4).await?;
    let identity = recognizer.identity();
    let mut rows = Vec::new();
    for case in cases {
        ensure!(
            std::fs::metadata(&case.file)?.len() <= 64 * 1024 * 1024,
            "clip too large"
        );
        let mut bytes = Vec::new();
        std::fs::File::open(&case.file)?
            .take(64 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= 64 * 1024 * 1024,
            "clip grew beyond byte budget"
        );
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        ensure!(
            sha256 == case.sha256,
            "clip identity changed for {}",
            case.id
        );
        let reader = hound::WavReader::new(std::io::Cursor::new(bytes))?;
        let spec = reader.spec();
        let samples = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, 32) => reader
                .into_samples::<f32>()
                .collect::<Result<Vec<_>, _>>()?,
            (hound::SampleFormat::Int, 16) => reader
                .into_samples::<i16>()
                .map(|sample| sample.map(|value| value as f32 / 32768.0))
                .collect::<Result<Vec<_>, _>>()?,
            _ => anyhow::bail!("expected PCM16 or F32 WAV"),
        };
        ensure!(
            !samples.is_empty() && samples.iter().all(|value| value.is_finite()),
            "invalid PCM"
        );
        let result = recognizer
            .transcribe(Arc::new(ReadbackAudio {
                samples,
                sample_rate: spec.sample_rate,
                channels: spec.channels,
            }))
            .await
            .with_context(|| format!("readback failed for {}", case.id))?;
        rows.push(
            serde_json::json!({"id":case.id,"file":case.file,"sha256":sha256,"expected":case.text,
            "transcribed":result.text,"labels":result.labels}),
        );
    }
    let report = serde_json::json!({"kind":"auxiliary_single_recognizer_readback",
        "model":identity.model,"revision":identity.revision,"listening_acceptance":"pending",
        "dual_model_qualification":false,"rows":rows});
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.sync_all()?;
    Ok(())
}
