//! MOSS dedicated design engine (`voice-design-1.7b` + shared codec, CUDA/Metal).
use crate::moss::candle::{self, resources};
use std::path::{Path, PathBuf};
use talechime_core::design::{
    DesignOutput, DesignRequest, Designing, EngineInfo, InputConstraints, Preparing, VoiceDesigner,
};
use talechime_protocol::Event;
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
        let devices = candle::available_devices();
        EngineInfo {
            id: ID,
            name: "MOSS 1.7B · 音色设计",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_uses_compiled_gpu_devices_without_preparing_resources() {
        let root = tempfile::tempdir().unwrap();
        let info = Engine.info(root.path());
        assert_eq!(info.devices, candle::available_devices());
        assert!(!info.devices.contains(&talechime_protocol::Device::Cpu));
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn unsupported_design_device_does_not_prepare_or_publish() {
        let root = tempfile::tempdir().unwrap();
        let result = candle::design::reference(
            root.path().to_path_buf(),
            talechime_protocol::Device::Cpu,
            "你好。".into(),
            "自然男声".into(),
            root.path().join("output.wav"),
            None,
        )
        .await;
        assert!(result.is_err());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }
}
