//! OmniVoice design engine: rides on the prepared narration model (instruct).
use crate::omnivoice::{self, design, resources};
use std::path::Path;
use talechime_core::design::{
    DesignOutput, DesignRequest, Designing, EngineInfo, InputConstraints, Preparing, VoiceDesigner,
};
use talechime_protocol::Event;
use tokio::sync::mpsc;

pub(super) const ID: &str = "omnivoice";

pub(super) struct Engine(pub(crate) crate::downloads::Downloads);
impl VoiceDesigner for Engine {
    fn id(&self) -> &'static str {
        ID
    }
    fn info<'a>(&'a self, _root: &'a Path) -> talechime_core::design::Information<'a> {
        Box::pin(async move {
            Ok({
                let status =
                    crate::resources::cached_manifest_status(resources::MANIFEST, self.0.clone())
                        .await?;
                let ready = status == crate::ResourceStatus::Ready;
                let missing = status.missing_bytes();
                EngineInfo {
                    id: ID,
                    name: "OmniVoice 0.6B · 指令合成",
                    devices: omnivoice::available_devices(),
                    inputs: InputConstraints {
                        text_max_chars: 100,
                        description_max_chars: None,
                        description_max_bytes: 4096,
                        forbid_parentheses: false,
                    },
                    extra_download_bytes: missing,
                    ready,
                    reuses_synthesis_model: true,
                }
            })
        })
    }
    fn prepare<'a>(&'a self, _root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a> {
        let downloads = self.0.clone();
        Box::pin(async move { resources::prepare(&downloads, progress).await })
    }
    fn design<'a>(&'a self, _root: &'a Path, request: DesignRequest) -> Designing<'a> {
        let downloads = self.0.clone();
        Box::pin(async move {
            anyhow::ensure!(
                omnivoice::available_devices().contains(&request.device),
                "requested design device is unavailable for {ID}"
            );
            let output = tempfile::NamedTempFile::new()?;
            let files = crate::resources::resolve_manifest(resources::MANIFEST, downloads)
                .await?
                .into_iter()
                .map(|(name, path)| (name.into(), path))
                .collect();
            design::reference_source(
                files,
                request.device,
                request.text,
                request.description,
                output.path().to_path_buf(),
                request.seed,
            )
            .await?;
            Ok(DesignOutput {
                wav: std::fs::read(output.path())?,
            })
        })
    }
}
