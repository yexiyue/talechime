use talechime_protocol::Event;
pub const REVISION: &str = "85e237c12c027371202489a0ec509ded67b5e4b5";
pub async fn prepare_model(
    model: super::models::Model,
    downloads: &crate::downloads::Downloads,
    progress: tokio::sync::mpsc::Sender<Event>,
) -> anyhow::Result<()> {
    crate::resources::prepare_cached(
        &format!("qwen/{}", model.id()),
        serde_json::from_str(model.manifest())?,
        downloads,
        progress,
    )
    .await?;
    Ok(())
}
