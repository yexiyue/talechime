//! MOSS dedicated design engine (`voice-design-1.7b` + shared codec, CUDA/Metal).
use crate::moss::candle::{self, resources};
use std::path::Path;
use talechime_core::design::{
    DesignOutput, DesignRequest, Designing, EngineInfo, InputConstraints, Preparing, VoiceDesigner,
};
use talechime_protocol::Event;
use tokio::sync::mpsc;

pub(super) const ID: &str = "moss-voicedesign";

pub(super) struct Engine(pub(crate) crate::downloads::Downloads);
impl VoiceDesigner for Engine {
    fn id(&self) -> &'static str {
        ID
    }
    fn info<'a>(&'a self, _root: &'a Path) -> talechime_core::design::Information<'a> {
        Box::pin(async move {
            Ok({
                let design = talechime_moss_tts::manifests::VOICE_DESIGN;
                let codec = talechime_moss_tts::manifests::CODEC;
                let status =
                    crate::resources::cached_manifests_status(&[design, codec], self.0.clone())
                        .await?;
                let ready = status == crate::ResourceStatus::Ready;
                let missing = status.missing_bytes();
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
            })
        })
    }
    fn prepare<'a>(&'a self, _root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a> {
        let downloads = self.0.clone();
        Box::pin(async move { resources::prepare_design(&downloads, progress).await })
    }
    fn design<'a>(&'a self, _root: &'a Path, request: DesignRequest) -> Designing<'a> {
        let downloads = self.0.clone();
        Box::pin(async move {
            let output = tempfile::NamedTempFile::new()?;
            let files =
                resources::resolve(talechime_moss_tts::manifests::VOICE_DESIGN, &downloads).await?;
            candle::design::reference_files(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn catalog_uses_compiled_gpu_devices_without_preparing_resources() {
        let root = tempfile::tempdir().unwrap();
        let engine = Engine(crate::downloads::Downloads::test(root.path()));
        let info = engine.info(root.path()).await.unwrap();
        assert_eq!(info.devices, candle::available_devices());
        assert!(!info.devices.contains(&talechime_protocol::Device::Cpu));
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn unsupported_design_device_does_not_prepare_or_publish() {
        let root = tempfile::tempdir().unwrap();
        let result = candle::design::reference(
            &crate::downloads::Downloads::test(root.path()),
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
