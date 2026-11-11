//! Verified local conversion of the pinned Fun-CosyVoice3 checkpoint.
use std::path::Path;
use talechime_protocol::Event;
use tokio::sync::mpsc;

/// Model revision plus the conversion format (including f64 F0 weights).
pub const REVISION: &str = "2512-runtime-f64-front-v4";

/// Check local converted files without downloading or invoking Python.
pub async fn prepare(directory: &Path, progress: mpsc::Sender<Event>) -> anyhow::Result<()> {
    let _ = progress
        .send(Event::ResourceState {
            stage: "校验".into(),
            resource: "cosyvoice3/0.5b".into(),
        })
        .await;
    super::native_recipe()
        .verify_existing(directory.to_owned(), crate::PreparationControl::default())
        .await
}
