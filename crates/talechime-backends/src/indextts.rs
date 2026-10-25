//! IndexTTS 2.5 reference cloning with named emotions and reference/vector controls.
pub mod params;
pub mod resources;
mod runtime;

use std::path::{Path, PathBuf};
use talechime_core::backend::{Backend, BackendError, Streaming};
use talechime_protocol::{Capabilities, Device};
use tokio::sync::{mpsc, oneshot};

pub const MODEL: &str = "2.5";

pub fn directory(root: &Path) -> PathBuf {
    root.join("indextts/models")
        .join(MODEL)
        .join(resources::REVISION)
}

pub fn voice_store(directory: &Path) -> anyhow::Result<talechime_core::voices::VoiceStore> {
    talechime_core::voices::VoiceStore::new(directory, "indextts", MODEL, resources::REVISION)
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
        style: true,
        compiled_devices: Vec::new(),
        pronunciation: false,
        continuation: false,
        parameters: params::catalog(),
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
            (*device != Device::Metal || crate::devices::metal::metal_is_available())
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

    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            request.reject_unsupported(self.caps.style, self.caps.continuation)?;
            let resolved = params::resolve(request.params, request.style, request.seed)?;
            if request.text.trim().is_empty()
                || !self.caps.voices.iter().any(|voice| voice == request.voice)
            {
                return Err(BackendError::Unsupported(
                    "IndexTTS needs nonempty text and an imported reference voice".into(),
                ));
            }
            if let Some(voice) = &resolved.emotion_voice
                && !self.caps.voices.contains(voice)
            {
                return Err(BackendError::Unsupported(
                    "emo_voice must name an imported IndexTTS reference voice".into(),
                ));
            }
            let (audio, receiver) = mpsc::channel(1);
            self.requests
                .as_ref()
                .ok_or_else(|| BackendError::Synthesis("IndexTTS owner closed".into()))?
                .send(runtime::Request {
                    #[cfg(feature = "performance-tools")]
                    perf: crate::performance::OwnerContext::capture(),
                    text: request.text.into(),
                    voice: request.voice.into(),
                    resolved,
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

    #[tokio::test]
    async fn emotion_controls_are_request_local_and_invalid_requests_never_enqueue()
    -> anyhow::Result<()> {
        use talechime_core::{backend::SegmentRequest, params::GenerationParams};
        let root = tempfile::tempdir()?;
        let mut caps = capabilities(root.path())?;
        caps.voices = vec!["custom:reader".into(), "custom:emotion".into()];
        let (tx, mut jobs) = mpsc::channel(1);
        let backend = IndexBackend {
            requests: Some(tx),
            thread: None,
            caps,
        };
        let params = params::IndexTtsParams::default()
            .emotion_voice("custom:emotion")
            .emotion_strength(0.7)
            .to_generation_params();
        let request = |style| SegmentRequest {
            text: "测试。",
            voice: "custom:reader",
            style,
            context: None,
            seed: 12,
            params: &params,
        };
        let _audio = backend.stream(request(None)).await?;
        let job = jobs.try_recv()?;
        assert_eq!(
            job.resolved.emotion_voice.as_deref(),
            Some("custom:emotion")
        );
        assert_eq!(job.resolved.options.emo_alpha, 0.7);
        assert_eq!(job.resolved.options.seed, Some(12));
        assert!(backend.stream(request(Some("sad"))).await.is_err());
        assert!(jobs.try_recv().is_err());
        let defaults = GenerationParams::new();
        let _audio = backend
            .stream(SegmentRequest {
                params: &defaults,
                ..request(None)
            })
            .await?;
        let job = jobs.try_recv()?;
        assert!(job.resolved.emotion_voice.is_none() && job.resolved.options.emo_vector.is_none());
        assert_eq!(job.resolved.options.emo_alpha, 1.0);
        let missing = params::IndexTtsParams::default()
            .emotion_voice("custom:missing")
            .to_generation_params();
        assert!(
            backend
                .stream(SegmentRequest {
                    params: &missing,
                    ..request(None)
                })
                .await
                .is_err()
        );
        assert!(jobs.try_recv().is_err());
        Ok(())
    }

    #[test]
    fn catalog_reports_cloning_without_creating_files() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let registry = crate::Registry::new(Some(root.path().to_owned()))?;
        let caps = registry.capabilities("indextts")?;
        assert_eq!(caps.model.as_deref(), Some(MODEL));
        assert!(caps.cloning && caps.style && !caps.native_streaming && !caps.continuation);
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
