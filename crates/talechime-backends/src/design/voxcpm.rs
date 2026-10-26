//! VoxCPM design engine: rides on the prepared narration model (Q8 variant).
use crate::voxcpm::{self, models::Model};
use std::path::Path;
use talechime_core::design::{
    DesignOutput, DesignRequest, Designing, EngineInfo, InputConstraints, Preparing, VoiceDesigner,
};
use talechime_protocol::Event;
use tokio::sync::mpsc;

pub(super) const ID: &str = "voxcpm";

pub(super) struct Engine;
impl VoiceDesigner for Engine {
    fn id(&self) -> &'static str {
        ID
    }
    fn info(&self, root: &Path) -> EngineInfo {
        let variant = Model::Q8;
        let (ready, missing) =
            super::manifest_state(&[(variant.manifest(), &variant.directory(root))]);
        let devices = voxcpm::available_devices()
            .into_iter()
            .filter(|device| variant.compiled_devices().contains(device))
            .collect();
        EngineInfo {
            id: ID,
            name: "VoxCPM 2B · 描述合成",
            devices,
            inputs: InputConstraints {
                text_max_chars: 100,
                description_max_chars: Some(200),
                description_max_bytes: 4096,
                forbid_parentheses: true,
            },
            extra_download_bytes: missing,
            ready,
            reuses_synthesis_model: true,
        }
    }
    fn prepare<'a>(&'a self, root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a> {
        let directory = Model::Q8.directory(root);
        Box::pin(
            async move { voxcpm::resources::prepare_model(&directory, Model::Q8, progress).await },
        )
    }
    fn design<'a>(&'a self, root: &'a Path, request: DesignRequest) -> Designing<'a> {
        let directory = Model::Q8.directory(root);
        Box::pin(async move {
            let output = tempfile::NamedTempFile::new()?;
            voxcpm::design::reference_model(
                directory,
                Model::Q8,
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
