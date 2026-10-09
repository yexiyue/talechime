//! CPU readback model group. Explicit preparation, no inference-time downloads.
mod sensevoice;
mod worker;
use std::{
    path::{Path, PathBuf},
    rc::Rc,
};
use tts_core::verification::{Recognizer, Verifier};
use tts_protocol::{Event, RecognizerIdentity};

pub const QWEN_REVISION: &str = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c";
pub const SENSEVOICE_REVISION: &str = "2365baeacb507f821a0c8120fcee3d484dba7a07";

/// Selected v1 group: Qwen3-ASR 0.6B main, SenseVoiceSmall INT8 reviewer, CPU.
#[derive(Debug, Clone)]
pub struct ReadbackModelOptions {
    pub resources: PathBuf,
    /// SenseVoice ORT intra-op threads; Qwen uses Candle's shared CPU thread pool.
    pub threads: usize,
}
impl ReadbackModelOptions {
    pub fn new(resources: impl Into<PathBuf>) -> Self {
        Self {
            resources: resources.into(),
            threads: 4,
        }
    }
}
/// Download/check immutable manifests, then initialize both models on their owners.
pub async fn prepare(
    options: &ReadbackModelOptions,
    progress: tokio::sync::mpsc::Sender<Event>,
) -> anyhow::Result<Rc<Verifier>> {
    anyhow::ensure!(
        !options.resources.as_os_str().is_empty() && (1..=32).contains(&options.threads),
        "invalid ASR resource path or thread count"
    );
    for (name, manifest) in [
        ("qwen06", include_str!("asr/qwen06.json")),
        ("sensevoice", include_str!("asr/sensevoice.json")),
    ] {
        crate::resources::prepare(
            &options.resources.join("asr").join(name),
            &format!("asr/{name}"),
            serde_json::from_str(manifest)?,
            progress.clone(),
        )
        .await?;
    }
    load_local(
        &options.resources.join("asr/qwen06"),
        &options.resources.join("asr/sensevoice"),
        options.threads,
    )
    .await
}
/// Load already prepared local files, validating the same pinned manifests. No download.
pub async fn load_local(
    qwen: &Path,
    sensevoice: &Path,
    threads: usize,
) -> anyhow::Result<Rc<Verifier>> {
    anyhow::ensure!((1..=32).contains(&threads), "invalid ASR thread count");
    let primary = load_qwen(qwen).await?;
    let reviewer = load_sensevoice(sensevoice, threads).await?;
    Ok(Rc::new(Verifier::new(primary, reviewer)?))
}
async fn verify_local(dir: &Path, manifest: &str) -> anyhow::Result<()> {
    let resources: Vec<crate::resources::Resource> = serde_json::from_str(manifest)?;
    let directory = dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        for resource in &resources {
            crate::resources::verify(&directory.join(&resource.path), resource)?;
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    Ok(())
}
/// Independently load a locally pinned Qwen recognizer (also useful for model probes).
pub async fn load_qwen(directory: &Path) -> anyhow::Result<Rc<dyn Recognizer>> {
    verify_local(directory, include_str!("asr/qwen06.json")).await?;
    worker::spawn(
        directory.to_path_buf(),
        worker::Model::Qwen,
        4,
        RecognizerIdentity {
            family: "qwen3-asr".into(),
            model: "Qwen3-ASR-0.6B-hf".into(),
            revision: QWEN_REVISION.into(),
            implementation: "talechime-qwen3-asr-candle-0.11.0-v1".into(),
        },
    )
    .await
}
/// Independently load a locally pinned SenseVoice recognizer.
pub async fn load_sensevoice(
    directory: &Path,
    threads: usize,
) -> anyhow::Result<Rc<dyn Recognizer>> {
    verify_local(directory, include_str!("asr/sensevoice.json")).await?;
    worker::spawn(
        directory.to_path_buf(),
        worker::Model::SenseVoice,
        threads,
        RecognizerIdentity {
            family: "sensevoice".into(),
            model: "SenseVoiceSmall-int8".into(),
            revision: SENSEVOICE_REVISION.into(),
            implementation: "talechime-sensevoice-ort-rc13-kaldi-fbank-0.1.0-v1".into(),
        },
    )
    .await
}
