use super::Mode;
use std::path::PathBuf;
use talechime_protocol::Event;
use tokio::sync::mpsc;
pub const CODEC_REVISION: &str = "3cd226ba2947efa357ef453bcad111b6eafba782";
pub const DESIGN_REVISION: &str = "97521ec2b6f3ec5026ac1f5751f8fc302d82c2d4";
pub(crate) struct Files {
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
    pub codec_config: PathBuf,
    pub codec_weights: Vec<PathBuf>,
}
pub(crate) async fn resolve(
    model_manifest: &str,
    downloads: &crate::downloads::Downloads,
) -> anyhow::Result<Files> {
    let mut model = crate::resources::resolve_manifest(model_manifest, downloads.clone()).await?;
    let mut codec =
        crate::resources::resolve_manifest(talechime_moss_tts::manifests::CODEC, downloads.clone())
            .await?;
    let mut take_model = |name: &str| {
        model
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("missing MOSS resource {name}"))
    };
    let mut take_codec = |name: &str| {
        codec
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("missing MOSS codec resource {name}"))
    };
    Ok(Files {
        config: take_model("config.json")?,
        tokenizer: take_model("tokenizer.json")?,
        weights: vec![take_model("model.safetensors")?],
        codec_config: take_codec("config.json")?,
        codec_weights: vec![
            take_codec("model-00001-of-00002.safetensors")?,
            take_codec("model-00002-of-00002.safetensors")?,
        ],
    })
}
async fn manifest(
    id: &str,
    json: &str,
    downloads: &crate::downloads::Downloads,
    progress: mpsc::Sender<Event>,
) -> anyhow::Result<()> {
    crate::resources::prepare_cached(
        &format!("moss/{id}"),
        serde_json::from_str::<Vec<crate::resources::Resource>>(json)?,
        downloads,
        progress,
    )
    .await?;
    Ok(())
}
pub async fn prepare(
    mode: Mode,
    downloads: &crate::downloads::Downloads,
    progress: mpsc::Sender<Event>,
) -> anyhow::Result<()> {
    let json = talechime_moss_tts::manifests::REALTIME;
    manifest(mode.id(), json, downloads, progress.clone()).await?;
    codec(downloads, progress).await
}
async fn codec(
    downloads: &crate::downloads::Downloads,
    progress: mpsc::Sender<Event>,
) -> anyhow::Result<()> {
    manifest(
        "codec",
        talechime_moss_tts::manifests::CODEC,
        downloads,
        progress,
    )
    .await
}
pub async fn prepare_design(
    downloads: &crate::downloads::Downloads,
    progress: mpsc::Sender<Event>,
) -> anyhow::Result<()> {
    manifest(
        "voice-design-1.7b",
        talechime_moss_tts::manifests::VOICE_DESIGN,
        downloads,
        progress.clone(),
    )
    .await?;
    codec(downloads, progress).await
}
