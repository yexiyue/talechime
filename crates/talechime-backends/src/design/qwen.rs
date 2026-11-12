//! Qwen dedicated design engine (`1.7b-voicedesign`), prepared on demand.
use crate::qwen::{self, models::Model};
use std::path::Path;
use talechime_core::design::{
    DesignOutput, DesignRequest, Designing, EngineInfo, InputConstraints, Preparing, VoiceDesigner,
};
use talechime_protocol::Event;
use tokio::sync::mpsc;

pub(super) const ID: &str = "qwen-voicedesign";

pub(super) struct Engine(pub(crate) crate::downloads::Downloads);
impl VoiceDesigner for Engine {
    fn id(&self) -> &'static str {
        ID
    }
    fn info<'a>(&'a self, _root: &'a Path) -> talechime_core::design::Information<'a> {
        Box::pin(async move {
            Ok({
                let status = crate::resources::cached_manifest_status(
                    Model::Design17.manifest(),
                    self.0.clone(),
                )
                .await?;
                let ready = status == crate::ResourceStatus::Ready;
                let missing = status.missing_bytes();
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
            })
        })
    }
    fn prepare<'a>(&'a self, _root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a> {
        let downloads = self.0.clone();
        Box::pin(async move {
            qwen::resources::prepare_model(Model::Design17, &downloads, progress).await
        })
    }
    fn design<'a>(&'a self, _root: &'a Path, request: DesignRequest) -> Designing<'a> {
        let downloads = self.0.clone();
        Box::pin(async move {
            anyhow::ensure!(
                qwen::available_devices().contains(&request.device),
                "requested design device is unavailable for {ID}"
            );
            let output = tempfile::NamedTempFile::new()?;
            let paths =
                crate::resources::resolve_manifest(Model::Design17.manifest(), downloads).await?;
            let files = Model::resolved_files(paths)?;
            qwen::design::reference_source(
                files,
                request.device,
                request.text,
                request.description,
                output.path().to_path_buf(),
                qwen::design::Options {
                    seed: request.seed,
                    ..Default::default()
                },
            )
            .await?;
            Ok(DesignOutput {
                wav: std::fs::read(output.path())?,
            })
        })
    }
}
