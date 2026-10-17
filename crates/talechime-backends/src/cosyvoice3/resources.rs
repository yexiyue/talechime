//! Verified local conversion of the pinned Fun-CosyVoice3 checkpoint.
use std::path::Path;
use tokio::sync::mpsc;
use tts_protocol::Event;

/// Model revision plus the conversion format (including f64 F0 weights).
pub const REVISION: &str = "2512-candle-f64-v1";

/// Check local converted files without downloading or invoking Python.
pub async fn prepare(directory: &Path, progress: mpsc::Sender<Event>) -> anyhow::Result<()> {
    let resources: Vec<crate::resources::Resource> =
        serde_json::from_str(include_str!("resources.json"))?;
    let _ = progress
        .send(Event::ResourceState {
            stage: "校验".into(),
            resource: "cosyvoice3/0.5b".into(),
        })
        .await;
    let directory = directory.to_owned();
    tokio::task::spawn_blocking(move || {
        for resource in resources {
            crate::resources::verify(&directory.join(&resource.path), &resource)
                .map_err(|error| anyhow::anyhow!("CosyVoice3 local model verification failed: {error}; convert the pinned checkpoint into {} (see docs/cosyvoice3.md)", directory.display()))?;
        }
        Ok(())
    }).await?
}
