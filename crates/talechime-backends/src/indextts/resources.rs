//! Check converted local weights before starting the owner thread.
use std::path::Path;
use talechime_protocol::Event;
use tokio::sync::mpsc;

/// Pinned upstream checkpoint plus the local conversion layout.
pub const REVISION: &str = "c39ce5ba-candle-v1";

const FILES: &[&str] = &[
    "config.json",
    "gpt.safetensors",
    "codec.safetensors",
    "s2mel.safetensors",
    "campplus.safetensors",
    "bigvgan.safetensors",
    "bigvgan.config.json",
    "feats.safetensors",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "w2v-bert-2.0/model.safetensors",
    "w2v-bert-2.0/config.json",
];

pub async fn prepare(directory: &Path, progress: mpsc::Sender<Event>) -> anyhow::Result<()> {
    let _ = progress
        .send(Event::ResourceState {
            stage: "校验".into(),
            resource: "indextts/2.5".into(),
        })
        .await;
    let directory = directory.to_owned();
    tokio::task::spawn_blocking(move || {
        for name in FILES {
            let path = directory.join(name);
            anyhow::ensure!(
                path.is_file() && path.metadata()?.len() > 0,
                "missing or empty IndexTTS converted file {}; see docs/indextts.md",
                path.display()
            );
        }
        Ok(())
    })
    .await?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn missing_conversion_does_not_create_resources() {
        let root = tempfile::tempdir().unwrap();
        let (tx, _rx) = mpsc::channel(2);
        assert!(prepare(root.path(), tx).await.is_err());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }
}
