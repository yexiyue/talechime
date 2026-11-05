//! Offline verification of the compiled native IndexTTS resource recipe.
use std::path::Path;
use talechime_protocol::Event;
use tokio::sync::mpsc;

pub const REVISION: &str = "c39ce5ba-runtime-f32-v3";

/// Verify an already published version without downloading or converting.
pub async fn prepare(directory: &Path, progress: mpsc::Sender<Event>) -> anyhow::Result<()> {
    let _ = progress
        .send(Event::ResourceState {
            stage: "校验".into(),
            resource: "indextts/2.5".into(),
        })
        .await;
    super::native_recipe()
        .verify_existing(directory.to_owned(), crate::PreparationControl::default())
        .await
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn missing_resources_are_not_created_implicitly() {
        let root = tempfile::tempdir().unwrap();
        let (tx, _) = tokio::sync::mpsc::channel(2);
        assert!(super::prepare(root.path(), tx).await.is_err());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }
}
