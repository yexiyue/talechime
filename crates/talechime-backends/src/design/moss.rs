//! MOSS dedicated design engine (`voice-design-1.7b` + shared codec, CUDA only).
use crate::moss::candle::{self, resources};
use std::path::{Path, PathBuf};
use talechime_core::design::{
    DesignOutput, DesignRequest, Designing, EngineInfo, InputConstraints, Preparing, VoiceDesigner,
};
use talechime_protocol::{Device, Event};
use tokio::sync::mpsc;

pub(super) const ID: &str = "moss-voicedesign";

pub(super) struct Engine;
impl VoiceDesigner for Engine {
    fn id(&self) -> &'static str {
        ID
    }
    fn info(&self, root: &Path) -> EngineInfo {
        let design = talechime_moss_tts::manifests::VOICE_DESIGN;
        let codec = talechime_moss_tts::manifests::CODEC;
        let (ready, missing) = super::manifest_state(&[
            (design, &resources::design_directory(root)),
            (codec, &resources::codec_directory(root)),
        ]);
        let devices = if candle::available_devices().contains(&Device::Cuda) {
            vec![Device::Cuda]
        } else {
            Vec::new()
        };
        EngineInfo {
            id: ID,
            name: "MOSS 1.7B · 音色设计（CUDA）",
            devices,
            inputs: InputConstraints {
                text_max_chars: 60,
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
        Box::pin(async move { resources::prepare_design(root, progress).await })
    }
    fn design<'a>(&'a self, root: &'a Path, request: DesignRequest) -> Designing<'a> {
        let root: PathBuf = root.to_path_buf();
        Box::pin(async move {
            let output = tempfile::NamedTempFile::new()?;
            candle::design::reference(
                root,
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
