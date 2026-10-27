//! CosyVoice3 reference cloning and style instructions with bounded native streaming.
pub mod resources;
mod runtime;
use std::path::{Path, PathBuf};
use talechime_core::backend::{Backend, BackendError, Streaming};
use talechime_protocol::{Capabilities, Device};
use tokio::sync::{mpsc, oneshot};

pub const MODEL: &str = "0.5b";
pub fn directory(root: &Path) -> PathBuf {
    root.join("cosyvoice3/models")
        .join(MODEL)
        .join(resources::REVISION)
}
pub fn voice_store(directory: &Path) -> anyhow::Result<talechime_core::voices::VoiceStore> {
    talechime_core::voices::VoiceStore::new(directory, "cosyvoice3", MODEL, resources::REVISION)
}
pub fn capabilities(directory: &Path) -> anyhow::Result<Capabilities> {
    let voices = voice_store(directory)?.list()?;
    Ok(Capabilities {
        backend: "cosyvoice3".into(),
        model: Some(MODEL.into()),
        model_name: "Fun-CosyVoice3 0.5B (experimental)".into(),
        default_voice: voices.first().map(|v| v.id.clone()).unwrap_or_default(),
        voice_names: voices
            .iter()
            .map(|v| (v.id.clone(), v.name.clone()))
            .collect(),
        voices: voices.into_iter().map(|v| v.id).collect(),
        native_streaming: true,
        cloning: true,
        style: true,
        compiled_devices: Vec::new(),
        pronunciation: false,

        parameters: Vec::new(),
    })
}
pub fn compiled_devices() -> Vec<Device> {
    vec![
        Device::Cpu,
        #[cfg(all(feature = "cosyvoice3-metal", target_os = "macos"))]
        Device::Metal,
        #[cfg(all(
            feature = "cosyvoice3-cuda",
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

pub struct CosyBackend {
    requests: Option<mpsc::Sender<runtime::Request>>,
    thread: Option<std::thread::JoinHandle<()>>,
    caps: Capabilities,
}
impl CosyBackend {
    pub async fn load_on(directory: PathBuf, device: Device) -> Result<Self, BackendError> {
        if !available_devices().contains(&device) {
            return Err(BackendError::Initialize(format!(
                "CosyVoice3 device {device:?} is unavailable"
            )));
        }
        let caps = capabilities(&directory).map_err(|e| BackendError::Initialize(e.to_string()))?;
        let (requests, jobs) = mpsc::channel(1);
        let (ready, loaded) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("cosyvoice3-inference".into())
            .spawn(move || runtime::run(directory, device, jobs, ready))
            .map_err(|e| BackendError::Initialize(e.to_string()))?;
        let backend = Self {
            requests: Some(requests),
            thread: Some(thread),
            caps,
        };
        loaded
            .await
            .map_err(|_| BackendError::Initialize("CosyVoice3 owner exited".into()))??;
        Ok(backend)
    }
}
impl Drop for CosyBackend {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Backend for CosyBackend {
    // References may be replaced by the host; do not advertise persistent
    // deterministic audio caching without binding their contents to identity.
    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }
    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            request.reject_unsupported(self.caps.style)?;
            let instruction = runtime::instruction(request.style)?;
            request
                .params
                .validate(&self.caps.parameters)
                .map_err(|e| BackendError::Unsupported(e.to_string()))?;
            if request.text.trim().is_empty()
                || !self.caps.voices.iter().any(|v| v == request.voice)
            {
                return Err(BackendError::Unsupported("CosyVoice3 needs nonempty text and an imported reference voice (voices import --text ...)".into()));
            }
            let (audio, receiver) = mpsc::channel(1);
            self.requests
                .as_ref()
                .ok_or_else(|| BackendError::Synthesis("CosyVoice3 owner closed".into()))?
                .send(runtime::Request {
                    #[cfg(feature = "performance-tools")]
                    perf: crate::performance::OwnerContext::capture(),
                    text: request.text.into(),
                    voice: request.voice.into(),
                    seed: request.seed,
                    instruction,
                    audio,
                })
                .await
                .map_err(|_| BackendError::Synthesis("CosyVoice3 owner exited".into()))?;
            Ok(receiver)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn style_is_forwarded_to_owner_without_changing_the_voice() -> anyhow::Result<()> {
        use talechime_core::{backend::SegmentRequest, params::GenerationParams};
        let root = tempfile::tempdir()?;
        let mut caps = capabilities(root.path())?;
        caps.voices = vec!["custom:reader".into()];
        let (tx, mut jobs) = mpsc::channel(1);
        let backend = CosyBackend {
            requests: Some(tx),
            thread: None,
            caps,
        };
        let params = GenerationParams::new();
        let request = |style| SegmentRequest {
            text: "测试。",
            voice: "custom:reader",
            style,

            seed: 7,
            params: &params,
        };
        let _audio = backend.stream(request(Some("请用悲伤的语气说。"))).await?;
        let job = jobs.try_recv()?;
        assert_eq!(job.voice, "custom:reader");
        assert_eq!(job.seed, 7);
        assert_eq!(
            job.instruction.as_deref(),
            Some("You are a helpful assistant. 请用悲伤的语气说。<|endofprompt|>")
        );
        let _audio = backend.stream(request(None)).await?;
        assert!(jobs.try_recv()?.instruction.is_none());
        assert!(
            backend
                .stream(request(Some("bad<|endofprompt|>")))
                .await
                .is_err()
        );
        assert!(jobs.try_recv().is_err());
        Ok(())
    }

    #[test]
    fn catalog_requires_a_real_reference_and_does_not_create_resources() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let registry = crate::Registry::new(Some(root.path().to_owned()))?;
        let caps = registry.capabilities("cosyvoice3")?;
        assert_eq!(caps.model.as_deref(), Some(MODEL));
        assert!(caps.native_streaming && caps.cloning);
        assert!(caps.style);
        assert!(caps.voices.is_empty() && caps.default_voice.is_empty());
        assert_eq!(caps.compiled_devices, compiled_devices());
        assert!(
            registry
                .capabilities_for("cosyvoice3", Some("bad-model"))
                .is_err()
        );
        assert_eq!(std::fs::read_dir(root.path())?.count(), 0);
        Ok(())
    }
    #[tokio::test]
    async fn missing_conversion_fails_without_downloading_or_writing() {
        let root = tempfile::tempdir().unwrap();
        let (tx, _rx) = mpsc::channel(2);
        assert!(
            resources::prepare(root.path(), tx)
                .await
                .unwrap_err()
                .to_string()
                .contains("convert the pinned checkpoint")
        );
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }
}
