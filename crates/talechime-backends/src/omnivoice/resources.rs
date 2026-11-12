use talechime_protocol::Event;
use tokio::sync::mpsc;
pub const REVISION: &str = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4";
pub const MANIFEST: &str = include_str!("resources.json");
pub async fn prepare(
    downloads: &crate::downloads::Downloads,
    progress: mpsc::Sender<Event>,
) -> anyhow::Result<()> {
    crate::resources::prepare_cached(
        "omnivoice/0.6b",
        serde_json::from_str(MANIFEST)?,
        downloads,
        progress,
    )
    .await?;
    Ok(())
}
