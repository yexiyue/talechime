//! IndexTTS 2.5 reference-cloning adapter.
pub mod resources;
mod runtime;

use std::path::{Path, PathBuf};
use tokio::sync::{mpsc, oneshot};
use tts_core::backend::{Backend, BackendError, Streaming};
use tts_protocol::{Capabilities, Device, ParamKind, ParamValue, ParameterSpec};

pub const MODEL: &str = "2.5";

pub fn directory(root: &Path) -> PathBuf {
    root.join("indextts/models")
        .join(MODEL)
        .join(resources::REVISION)
}

pub fn voice_store(directory: &Path) -> anyhow::Result<tts_core::voices::VoiceStore> {
    tts_core::voices::VoiceStore::new(directory, "indextts", MODEL, resources::REVISION)
}

pub fn capabilities(directory: &Path) -> anyhow::Result<Capabilities> {
    let voices = voice_store(directory)?.list()?;
    Ok(Capabilities {
        backend: "indextts".into(),
        model: Some(MODEL.into()),
        model_name: "IndexTTS 2.5 (experimental)".into(),
        default_voice: voices
            .first()
            .map(|voice| voice.id.clone())
            .unwrap_or_default(),
        voice_names: voices
            .iter()
            .map(|voice| (voice.id.clone(), voice.name.clone()))
            .collect(),
        voices: voices.into_iter().map(|voice| voice.id).collect(),
        native_streaming: false,
        cloning: true,
        style: false,
        compiled_devices: Vec::new(),
        pronunciation: false,
        continuation: false,
        parameters: vec![ParameterSpec {
            name: "language".into(),
            kind: ParamKind::Text { max_len: 5 },
            default: ParamValue::Text("zh".into()),
            description: "文本语言：zh 或 en".into(),
        }],
    })
}

pub fn compiled_devices() -> Vec<Device> {
    vec![
        Device::Cpu,
        #[cfg(all(feature = "indextts-metal", target_os = "macos"))]
        Device::Metal,
        #[cfg(all(
            feature = "indextts-cuda-cudnn",
            any(target_os = "linux", target_os = "windows")
        ))]
        Device::Cuda,
    ]
}

pub fn available_devices() -> Vec<Device> {
    compiled_devices()
        .into_iter()
        .filter(|device| {
            (*device != Device::Metal || tts_candle_platform::metal_is_available())
                && runtime::candle_device(*device).is_ok()
        })
        .collect()
}

pub struct IndexBackend {
    requests: Option<mpsc::Sender<runtime::Request>>,
    thread: Option<std::thread::JoinHandle<()>>,
    caps: Capabilities,
}

impl IndexBackend {
    pub async fn load_on(directory: PathBuf, device: Device) -> Result<Self, BackendError> {
        if !available_devices().contains(&device) {
            return Err(BackendError::Initialize(format!(
                "IndexTTS device {device:?} is unavailable"
            )));
        }
        let caps = capabilities(&directory)
            .map_err(|error| BackendError::Initialize(error.to_string()))?;
        let (requests, jobs) = mpsc::channel(1);
        let (ready, loaded) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("indextts-inference".into())
            .spawn(move || runtime::run(directory, device, jobs, ready))
            .map_err(|error| BackendError::Initialize(error.to_string()))?;
        let backend = Self {
            requests: Some(requests),
            thread: Some(thread),
            caps,
        };
        loaded
            .await
            .map_err(|_| BackendError::Initialize("IndexTTS owner exited".into()))??;
        Ok(backend)
    }
}

impl Drop for IndexBackend {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Backend for IndexBackend {
    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }

    fn stream<'a>(&'a self, request: tts_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            request.reject_unsupported(false, false)?;
            request
                .params
                .validate(&self.caps.parameters)
                .map_err(BackendError::Unsupported)?;
            if request.text.trim().is_empty()
                || !self.caps.voices.iter().any(|voice| voice == request.voice)
            {
                return Err(BackendError::Unsupported(
                    "IndexTTS needs nonempty text and an imported reference voice".into(),
                ));
            }
            let language = match request.params.get("language") {
                Some(ParamValue::Text(value)) if value == "zh" || value == "en" => value.as_str(),
                None => "zh",
                _ => {
                    return Err(BackendError::Unsupported(
                        "IndexTTS language must be zh or en".into(),
                    ));
                }
            };
            let (audio, receiver) = mpsc::channel(1);
            self.requests
                .as_ref()
                .ok_or_else(|| BackendError::Synthesis("IndexTTS owner closed".into()))?
                .send(runtime::Request {
                    #[cfg(feature = "performance-tools")]
                    perf: crate::performance::OwnerContext::capture(),
                    text: request.text.into(),
                    voice: request.voice.into(),
                    language: language.into(),
                    seed: request.seed,
                    audio,
                })
                .await
                .map_err(|_| BackendError::Synthesis("IndexTTS owner exited".into()))?;
            Ok(receiver)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_reports_cloning_without_creating_files() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let registry = crate::Registry::new(Some(root.path().to_owned()))?;
        let caps = registry.capabilities("indextts")?;
        assert_eq!(caps.model.as_deref(), Some(MODEL));
        assert!(caps.cloning && !caps.native_streaming && !caps.continuation);
        assert!(caps.voices.is_empty() && caps.default_voice.is_empty());
        assert!(
            registry
                .capabilities_for("indextts", Some("wrong"))
                .is_err()
        );
        assert_eq!(std::fs::read_dir(root.path())?.count(), 0);
        Ok(())
    }
}
