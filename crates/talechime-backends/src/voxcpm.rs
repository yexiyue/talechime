//! Native VoxCPM2 streaming; the model never leaves its inference thread.
mod cache;
pub mod design;
pub mod models;
pub mod params;
pub mod resources;
mod runtime;
use std::path::{Path, PathBuf};
use tokio::sync::{mpsc, oneshot};
use tts_core::backend::{Backend, BackendError, Segmentation, Streaming};
use tts_protocol::{Capabilities, Device};
pub const MODEL: &str = "2b-q8_0";
/// Invalidate native-engine performance calibration without changing model resources.
pub const CALIBRATION_REVISION: &str = "169f64d8b98bbaab1761e4ca3a83e6af653456cc-candle-v1";
pub fn directory(root: &Path) -> PathBuf {
    models::Model::Q8.directory(root)
}
pub fn voice_store(directory: &Path) -> anyhow::Result<tts_core::voices::VoiceStore> {
    voice_store_for(directory, models::Model::Q8)
}
pub fn voice_store_for(
    directory: &Path,
    model: models::Model,
) -> anyhow::Result<tts_core::voices::VoiceStore> {
    tts_core::voices::VoiceStore::new(directory, "voxcpm", model.id(), model.revision())
}
pub fn capabilities(directory: &Path) -> anyhow::Result<Capabilities> {
    capabilities_for(directory, models::Model::Q8)
}
pub fn capabilities_for(directory: &Path, model: models::Model) -> anyhow::Result<Capabilities> {
    let mut caps = Capabilities {
        backend: "voxcpm".into(),
        model: Some(model.id().into()),
        model_name: model.name().into(),
        voices: vec!["narrator".into()],
        default_voice: "narrator".into(),
        voice_names: [("narrator".into(), "自然音色（模型默认）".into())].into(),
        native_streaming: true,
        cloning: true,
        style: true,
        compiled_devices: Vec::new(),
        pronunciation: false,
        continuation: true,
        parameters: params::catalog(),
    };
    for voice in voice_store_for(directory, model)?.list()? {
        caps.voice_names.insert(voice.id.clone(), voice.name);
        caps.voices.push(voice.id);
    }
    Ok(caps)
}
pub fn compiled_devices() -> Vec<Device> {
    vec![
        Device::Cpu,
        #[cfg(all(
            feature = "voxcpm-cuda",
            any(target_os = "windows", target_os = "linux")
        ))]
        Device::Cuda,
        #[cfg(all(feature = "voxcpm-metal", target_os = "macos"))]
        Device::Metal,
    ]
}
pub fn available_devices() -> Vec<Device> {
    compiled_devices()
        .into_iter()
        .filter(|device| match device {
            Device::Cpu => true,
            Device::Cuda => runtime::candle_device(*device).is_ok(),
            Device::Metal => {
                tts_candle_platform::metal_is_available() && runtime::candle_device(*device).is_ok()
            }
            _ => false,
        })
        .collect()
}
pub struct VoxBackend {
    requests: Option<mpsc::Sender<runtime::Request>>,
    thread: Option<std::thread::JoinHandle<()>>,
    caps: Capabilities,
}
impl VoxBackend {
    pub async fn load_on(directory: PathBuf, device: Device) -> Result<Self, BackendError> {
        Self::load_model_on(directory, models::Model::Q8, device).await
    }
    pub async fn load_model_on(
        directory: PathBuf,
        model: models::Model,
        device: Device,
    ) -> Result<Self, BackendError> {
        if !model.compiled_devices().contains(&device) || !available_devices().contains(&device) {
            return Err(BackendError::Initialize(format!(
                "VoxCPM2 device {device:?} is unavailable"
            )));
        }
        let caps = capabilities_for(&directory, model)
            .map_err(|e| BackendError::Initialize(e.to_string()))?;
        let (requests, jobs) = mpsc::channel(1);
        let (ready, loaded) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("voxcpm-inference".into())
            .spawn(move || runtime::run(directory, model, device, jobs, ready))
            .map_err(|e| BackendError::Initialize(e.to_string()))?;
        // Own the thread before awaiting initialization: cancellation must join it.
        let backend = Self {
            requests: Some(requests),
            thread: Some(thread),
            caps,
        };
        loaded
            .await
            .map_err(|_| BackendError::Initialize("Vox inference thread exited".into()))??;
        Ok(backend)
    }
}
impl Drop for VoxBackend {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Backend for VoxBackend {
    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }
    fn stream<'a>(&'a self, request: tts_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            request.reject_unsupported(self.caps.style, self.caps.continuation)?;
            let (text, voice, context) = (request.text, request.voice, request.context);
            if !self.caps.voices.iter().any(|id| id == voice) || text.trim().is_empty() {
                return Err(BackendError::Unsupported(
                    "unknown Vox voice or empty text".into(),
                ));
            }
            let text = styled_text(request.style, text, context.is_some())?;
            let resolved = params::resolve(request.params, request.seed);
            let (audio, receiver) = mpsc::channel(1);
            self.requests
                .as_ref()
                .expect("live inference thread")
                .send(runtime::Request {
                    context: context.cloned(),
                    text,
                    voice: voice.into(),
                    options: resolved,
                    audio,
                })
                .await
                .map_err(|_| BackendError::Synthesis("Vox inference thread exited".into()))?;
            Ok(receiver)
        })
    }
    fn segments<'a>(&'a self, source: &'a str) -> Segmentation<'a> {
        Box::pin(async move { Ok(tts_core::text::preprocess_text(source, 180)) })
    }
}

/// The documented upstream style-control convention: a parenthesized
/// description prefix on the target text; parentheses inside the description
/// would blur the prefix boundary and are rejected explicitly.
///
/// Combined with execution-local continuation the prefix sits between the
/// previous transcript and the target text; a real-model trial (see
/// docs/records/params-exposure-2026-10-10.md) truncated the continued
/// segment there, so the combination is rejected until listening
/// acceptance proves it.
fn styled_text(style: Option<&str>, text: &str, continued: bool) -> Result<String, BackendError> {
    let styled = !style.is_some_and(|value| value.trim().is_empty());
    if styled && continued {
        return Err(BackendError::Unsupported(
            "Vox style is unavailable together with continuation until accepted".into(),
        ));
    }
    let style = style
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_default();
    if style.contains('(') || style.contains(')') {
        return Err(BackendError::Unsupported(
            "Vox style descriptions must not contain parentheses".into(),
        ));
    }
    if style.is_empty() {
        Ok(text.to_owned())
    } else {
        Ok(format!("({style}){text}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_weight_variants_support_continuation_without_preparing_assets() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        for model in [models::Model::Q8, models::Model::OriginalBf16] {
            let caps = capabilities_for(&model.directory(root.path()), model)?;
            assert!(caps.continuation);
            assert!(caps.style);
            assert_eq!(caps.default_voice, "narrator");
        }
        assert_eq!(std::fs::read_dir(root.path())?.count(), 0);
        Ok(())
    }
    #[test]
    fn style_wraps_the_target_text_like_the_upstream_convention() {
        assert_eq!(styled_text(None, "你好。", false).unwrap(), "你好。");
        assert_eq!(styled_text(Some("  "), "你好。", true).unwrap(), "你好。");
        assert_eq!(
            styled_text(Some("cheerful tone"), "你好。", false).unwrap(),
            "(cheerful tone)你好。"
        );
        assert!(styled_text(Some("半句(话"), "你好。", false).is_err());
        assert!(styled_text(Some("cheerful tone"), "你好。", true).is_err());
    }
}
