//! Native VoxCPM2 streaming; the model never leaves its inference thread.
mod cache;
#[cfg(all(test, feature = "voxcpm-cuda"))]
mod cuda_tests;
pub mod design;
pub mod models;
pub mod params;
pub mod resources;
mod runtime;
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
};
use talechime_core::backend::{Backend, BackendError, Segmentation, Streaming};
use talechime_protocol::{Capabilities, Device};
use tokio::sync::{mpsc, oneshot};
pub const MODEL: &str = "2b-q8_0";
/// Invalidate native-engine performance calibration without changing model resources.
pub const CALIBRATION_REVISION: &str = "169f64d8b98bbaab1761e4ca3a83e6af653456cc-candle-v1";
pub fn directory(root: &Path) -> PathBuf {
    models::Model::Q8.directory(root)
}
pub fn voice_store(directory: &Path) -> anyhow::Result<talechime_core::voices::VoiceStore> {
    voice_store_for(directory, models::Model::Q8)
}
pub fn voice_store_for(
    directory: &Path,
    model: models::Model,
) -> anyhow::Result<talechime_core::voices::VoiceStore> {
    talechime_core::voices::VoiceStore::new(directory, "voxcpm", model.id(), model.revision())
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
                crate::devices::metal::metal_is_available()
                    && runtime::candle_device(*device).is_ok()
            }
            _ => false,
        })
        .collect()
}
pub struct VoxBackend {
    requests: Option<mpsc::Sender<runtime::Request>>,
    thread: Option<std::thread::JoinHandle<()>>,
    caps: RefCell<Capabilities>,
    directory: PathBuf,
    model: models::Model,
    cache_identity: String,
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
        let cache_identity =
            crate::cache_identity::identity("voxcpm-candle", model.revision(), device, &directory);
        let (requests, jobs) = mpsc::channel(1);
        let (ready, loaded) = oneshot::channel();
        let voice_directory = directory.clone();
        let thread = std::thread::Builder::new()
            .name("voxcpm-inference".into())
            .spawn(move || runtime::run(directory, model, device, jobs, ready))
            .map_err(|e| BackendError::Initialize(e.to_string()))?;
        // Own the thread before awaiting initialization: cancellation must join it.
        let backend = Self {
            requests: Some(requests),
            thread: Some(thread),
            caps: RefCell::new(caps),
            directory: voice_directory,
            model,
            cache_identity,
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
    fn cache_identity(&self) -> Option<String> {
        Some(self.cache_identity.clone())
    }
    fn capabilities(&self) -> Capabilities {
        self.caps.borrow().clone()
    }
    fn refresh_reference_voices(&self) -> Result<(), BackendError> {
        let updated = capabilities_for(&self.directory, self.model)
            .map_err(|error| BackendError::Unsupported(error.to_string()))?;
        let mut caps = self.caps.borrow_mut();
        caps.voices = updated.voices;
        caps.voice_names = updated.voice_names;
        Ok(())
    }
    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            let (text, voice) = (request.text, request.voice);
            {
                let caps = self.caps.borrow();
                request.reject_unsupported(caps.style)?;
                if !caps.voices.iter().any(|id| id == voice) || text.trim().is_empty() {
                    return Err(BackendError::Unsupported(
                        "unknown Vox voice or empty text".into(),
                    ));
                }
            }
            let text = styled_text(request.style, text)?;
            let resolved = params::resolve(request.params, request.seed);
            let (audio, receiver) = mpsc::channel(1);
            self.requests
                .as_ref()
                .expect("live inference thread")
                .send(runtime::Request {
                    #[cfg(feature = "performance-tools")]
                    perf: crate::performance::OwnerContext::capture(),

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
    fn next_segment<'a>(
        &'a self,
        request: talechime_core::backend::SegmentationRequest<'a>,
    ) -> Segmentation<'a> {
        Box::pin(async move {
            Ok(talechime_core::text::preprocess_text(request.text, 180)
                .into_iter()
                .next())
        })
    }
}

/// The documented upstream style-control convention: a parenthesized
/// description prefix on the target text; parentheses inside the description
/// would blur the prefix boundary and are rejected explicitly.
///
/// The selected fixed reference and its transcript stay unchanged.
fn styled_text(style: Option<&str>, text: &str) -> Result<String, BackendError> {
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
    fn installed_reference_refreshes_a_loaded_catalog_and_errors_are_atomic() -> anyhow::Result<()>
    {
        let root = tempfile::tempdir()?;
        let model = models::Model::Q8;
        let directory = model.directory(root.path());
        let backend = VoxBackend {
            requests: None,
            thread: None,
            caps: RefCell::new(capabilities_for(&directory, model)?),
            directory: directory.clone(),
            model,
            cache_identity: "fixture".into(),
        };
        let original = backend.capabilities();
        let wav = root.path().join("reference.wav");
        // This checks metadata publication only, not model inference.
        std::fs::write(&wav, b"reference fixture")?;
        let store = voice_store_for(&directory, model)?;
        store.import("custom:late", "Late voice", &wav, "你好。", None)?;
        assert!(
            !backend
                .capabilities()
                .voices
                .contains(&"custom:late".into())
        );
        backend.refresh_reference_voices()?;
        let updated = backend.capabilities();
        assert!(updated.voices.contains(&"custom:late".into()));
        assert_eq!(updated.default_voice, original.default_voice);
        assert_eq!(updated.model, original.model);
        assert_eq!(updated.voice_names["custom:late"], "Late voice");
        std::fs::write(
            store.path("custom:late")?.join("voice.json"),
            b"invalid JSON",
        )?;
        assert!(backend.refresh_reference_voices().is_err());
        assert_eq!(backend.capabilities().voices, updated.voices);
        Ok(())
    }

    #[test]
    fn both_weight_variants_support_fixed_reference_without_preparing_assets() -> anyhow::Result<()>
    {
        let root = tempfile::tempdir()?;
        for model in [models::Model::Q8, models::Model::OriginalBf16] {
            let caps = capabilities_for(&model.directory(root.path()), model)?;
            assert!(caps.cloning);
            assert!(caps.style);
            assert_eq!(caps.default_voice, "narrator");
        }
        assert_eq!(std::fs::read_dir(root.path())?.count(), 0);
        Ok(())
    }
    #[test]
    fn style_wraps_the_target_text_like_the_upstream_convention() {
        assert_eq!(styled_text(None, "你好。").unwrap(), "你好。");
        assert_eq!(styled_text(Some("  "), "你好。").unwrap(), "你好。");
        assert_eq!(
            styled_text(Some("cheerful tone"), "你好。").unwrap(),
            "(cheerful tone)你好。"
        );
        assert!(styled_text(Some("半句(话"), "你好。").is_err());
    }
}
