use talechime_protocol::Event;
use tokio::sync::mpsc;
pub const REVISION: &str = "169f64d8b98bbaab1761e4ca3a83e6af653456cc";
pub(crate) async fn prepare_model(
    model: super::models::Model,
    downloads: &crate::downloads::Downloads,
    progress: mpsc::Sender<Event>,
) -> anyhow::Result<()> {
    crate::resources::prepare_cached(
        &format!("voxcpm/{}", model.id()),
        serde_json::from_str(model.manifest())?,
        downloads,
        progress,
    )
    .await?;
    Ok(())
}
