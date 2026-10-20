use std::path::Path;
use talechime_protocol::Event;
use tokio::sync::mpsc;
pub const REVISION: &str = "c5fdb5ccb189668d56333f77ba2629f4cd7535f4";
pub async fn prepare(directory: &Path, progress: mpsc::Sender<Event>) -> anyhow::Result<()> {
    crate::resources::prepare(
        directory,
        "omnivoice/0.6b",
        serde_json::from_str(include_str!("resources.json"))?,
        progress,
    )
    .await?;
    let temporary = tempfile::NamedTempFile::new_in(directory)?;
    std::io::Write::write_all(
        &mut temporary.as_file(),
        talechime_omnivoice::ARTIFACTS_JSON,
    )?;
    temporary.persist(directory.join("omnivoice.artifacts.json"))?;
    Ok(())
}
