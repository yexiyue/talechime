//! CPU readback model group. Explicit preparation, no inference-time downloads.
mod sensevoice;
mod worker;
use std::{path::PathBuf, rc::Rc};
use talechime_core::verification::{Recognizer, Verifier};
use talechime_protocol::{Event, RecognizerIdentity};

pub const QWEN_REVISION: &str = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c";
pub const SENSEVOICE_REVISION: &str = "2365baeacb507f821a0c8120fcee3d484dba7a07";

/// Selected v1 group: Qwen3-ASR 0.6B main, SenseVoiceSmall INT8 reviewer, CPU.
#[derive(Debug, Clone)]
pub struct ReadbackModelOptions {
    /// SenseVoice ORT intra-op threads; Qwen uses Candle's shared CPU thread pool.
    pub threads: usize,
}
impl ReadbackModelOptions {
    pub fn new() -> Self {
        Self { threads: 4 }
    }
}
impl Default for ReadbackModelOptions {
    fn default() -> Self {
        Self::new()
    }
}
/// Download/check immutable manifests, then initialize both models on their owners.
pub async fn prepare(
    options: &ReadbackModelOptions,
    downloads: &crate::downloads::Downloads,
    progress: tokio::sync::mpsc::Sender<Event>,
) -> anyhow::Result<Rc<Verifier>> {
    anyhow::ensure!(
        (1..=32).contains(&options.threads),
        "invalid ASR thread count"
    );
    for (name, manifest) in [
        ("qwen06", include_str!("asr/qwen06.json")),
        ("sensevoice", include_str!("asr/sensevoice.json")),
    ] {
        crate::resources::prepare_cached(
            &format!("asr/{name}"),
            serde_json::from_str(manifest)?,
            downloads,
            progress.clone(),
        )
        .await?;
    }
    let primary = load_qwen(downloads).await?;
    let reviewer = load_sensevoice(downloads, options.threads).await?;
    Ok(Rc::new(Verifier::new(primary, reviewer)?))
}

fn file(
    files: &std::collections::BTreeMap<String, PathBuf>,
    name: &str,
) -> anyhow::Result<PathBuf> {
    files
        .get(name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("ASR manifest lacks {name}"))
}

/// Load the pinned Qwen recognizer from the configured Hub cache without downloading.
pub async fn load_qwen(
    downloads: &crate::downloads::Downloads,
) -> anyhow::Result<Rc<dyn Recognizer>> {
    let qwen =
        crate::resources::resolve_manifest(include_str!("asr/qwen06.json"), downloads.clone())
            .await?;
    spawn_qwen(worker::Model::QwenFiles {
        config: file(&qwen, "config.json")?,
        tokenizer: file(&qwen, "tokenizer.json")?,
        weights: file(&qwen, "model.safetensors")?,
    })
    .await
}
async fn spawn_qwen(model: worker::Model) -> anyhow::Result<Rc<dyn Recognizer>> {
    worker::spawn(
        model,
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
/// Load the pinned SenseVoice recognizer from the configured Hub cache without downloading.
pub async fn load_sensevoice(
    downloads: &crate::downloads::Downloads,
    threads: usize,
) -> anyhow::Result<Rc<dyn Recognizer>> {
    anyhow::ensure!((1..=32).contains(&threads), "invalid ASR thread count");
    let sensevoice =
        crate::resources::resolve_manifest(include_str!("asr/sensevoice.json"), downloads.clone())
            .await?;
    spawn_sensevoice(
        worker::Model::SenseVoiceFiles {
            model: file(&sensevoice, "model.int8.onnx")?,
            tokens: file(&sensevoice, "tokens.txt")?,
        },
        threads,
    )
    .await
}
async fn spawn_sensevoice(
    model: worker::Model,
    threads: usize,
) -> anyhow::Result<Rc<dyn Recognizer>> {
    worker::spawn(
        model,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn offline_recognizer_loading_reports_missing_cache_without_creating_layout() {
        let root = tempfile::tempdir().unwrap();
        let downloads = crate::downloads::Downloads::test(root.path());
        assert!(load_qwen(&downloads).await.is_err());
        assert!(load_sensevoice(&downloads, 4).await.is_err());
        assert!(load_sensevoice(&downloads, 0).await.is_err());
        assert!(!root.path().join("asr").exists());
    }
}
