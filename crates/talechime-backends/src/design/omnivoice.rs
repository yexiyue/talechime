//! OmniVoice design engine: rides on the prepared narration model (instruct).
use crate::omnivoice::{self, design, resources};
use std::path::Path;
use talechime_core::design::{
    DesignOutput, DesignRequest, Designing, EngineInfo, InputConstraints, Preparing, VoiceDesigner,
};
use talechime_protocol::Event;
use tokio::sync::mpsc;

pub(super) const ID: &str = "omnivoice";

pub(super) struct Engine;
impl VoiceDesigner for Engine {
    fn id(&self) -> &'static str {
        ID
    }
    fn info(&self, root: &Path) -> EngineInfo {
        let directory = omnivoice::directory(root);
        let (ready, missing) = super::manifest_state(&[(resources::MANIFEST, &directory)]);
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
    }
    fn prepare<'a>(&'a self, root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a> {
        let directory = omnivoice::directory(root);
        Box::pin(async move { resources::prepare(&directory, progress).await })
    }
    fn design<'a>(&'a self, root: &'a Path, request: DesignRequest) -> Designing<'a> {
        let directory = omnivoice::directory(root);
        Box::pin(async move {
            anyhow::ensure!(
                omnivoice::available_devices().contains(&request.device),
                "requested design device is unavailable for {ID}"
            );
            let output = tempfile::NamedTempFile::new()?;
            design::reference(
                directory,
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
