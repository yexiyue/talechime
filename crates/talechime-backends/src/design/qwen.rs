//! Qwen dedicated design engine (`1.7b-voicedesign`), prepared on demand.
use crate::qwen::{self, models::Model};
use std::path::Path;
use talechime_core::design::{
    DesignOutput, DesignRequest, Designing, EngineInfo, InputConstraints, Preparing, VoiceDesigner,
};
use talechime_protocol::Event;
use tokio::sync::mpsc;

pub(super) const ID: &str = "qwen-voicedesign";

pub(super) struct Engine;
impl VoiceDesigner for Engine {
    fn id(&self) -> &'static str {
        ID
    }
    fn info(&self, root: &Path) -> EngineInfo {
        let (ready, missing) = super::manifest_state(&[(
            Model::Design17.manifest(),
            &Model::Design17.directory(root),
        )]);
        EngineInfo {
            id: ID,
            name: "Qwen3 1.7B · 音色设计",
            devices: qwen::available_devices(),
            inputs: InputConstraints {
                text_max_chars: 100,
                description_max_chars: None,
                description_max_bytes: 4096,
                forbid_parentheses: false,
            },
            extra_download_bytes: missing,
            ready,
            reuses_synthesis_model: false,
        }
    }
    fn prepare<'a>(&'a self, root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a> {
        let directory = Model::Design17.directory(root);
        Box::pin(async move {
            qwen::resources::prepare_model(&directory, Model::Design17, progress).await
        })
    }
    fn design<'a>(&'a self, root: &'a Path, request: DesignRequest) -> Designing<'a> {
        let directory = Model::Design17.directory(root);
        Box::pin(async move {
            anyhow::ensure!(
                qwen::available_devices().contains(&request.device),
                "requested design device is unavailable for {ID}"
            );
            let output = tempfile::NamedTempFile::new()?;
            qwen::design::reference(
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
