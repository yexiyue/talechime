//! Realtime 1.7B Candle synthesis with model-scoped reference voices.
#[cfg(all(test, feature = "moss-candle-cuda", target_os = "windows"))]
mod cuda_tests;
pub mod design;
pub mod params;
pub mod resources;
mod runtime;
use std::path::{Path, PathBuf};
use talechime_core::backend::{Backend, BackendError, Segmentation, Streaming};
use talechime_protocol::{Capabilities, Device};
use tokio::sync::{mpsc, oneshot};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Realtime,
}
impl Mode {
    pub fn parse(id: &str) -> anyhow::Result<Self> {
        match id {
            "realtime-1.7b" => Ok(Self::Realtime),
            _ => anyhow::bail!(
                "unsupported MOSS model {id}; supported Realtime model: realtime-1.7b"
            ),
        }
    }
    pub fn id(self) -> &'static str {
        "realtime-1.7b"
    }
    pub fn revision(self) -> &'static str {
        "75682787d8e2fcc73faca37ba2931453ca9c4022"
    }
    pub fn directory(self, root: &Path) -> PathBuf {
        root.join("moss/models")
            .join(self.id())
            .join(self.revision())
    }
    fn codebooks(self) -> usize {
        16
    }
}
pub fn voice_store(
    directory: &Path,
    mode: Mode,
) -> anyhow::Result<talechime_core::voices::VoiceStore> {
    talechime_core::voices::VoiceStore::new(directory, "moss", mode.id(), mode.revision())
}
pub fn capabilities(root: &Path, mode: Mode) -> anyhow::Result<Capabilities> {
    let mut caps = Capabilities {
        backend: "moss".into(),
        model: Some(mode.id().into()),
        model_name: "MOSS Realtime 1.7B（实验）".into(),
        voices: vec!["narrator".into()],
        default_voice: "narrator".into(),
        voice_names: [("narrator".into(), "随机音色（建议导入或设计音色）".into())].into(),
        native_streaming: true,
        cloning: true,
        style: false,
        pronunciation: false,

        parameters: params::catalog(),
        compiled_devices: compiled_devices(),
    };
    for voice in voice_store(&mode.directory(root), mode)?.list()? {
        caps.voice_names.insert(voice.id.clone(), voice.name);
        caps.voices.push(voice.id);
    }
    Ok(caps)
}
/// Production exposure is GPU-only until full CPU memory/performance acceptance.
pub fn compiled_devices() -> Vec<Device> {
    vec![
        #[cfg(all(
            feature = "moss-candle-cuda",
            any(target_os = "windows", target_os = "linux")
        ))]
        Device::Cuda,
        #[cfg(all(feature = "moss-candle-metal", target_os = "macos"))]
        Device::Metal,
    ]
}
pub fn available_devices() -> Vec<Device> {
    compiled_devices()
        .into_iter()
        .filter(|device| {
            (*device != Device::Metal || crate::devices::metal::metal_is_available())
                && runtime::device(*device).is_ok()
        })
        .collect()
}
pub struct CandleBackend {
    requests: Option<mpsc::Sender<runtime::Request>>,
    thread: Option<std::thread::JoinHandle<()>>,
    caps: Capabilities,
    cache_identity: String,
}
impl CandleBackend {
    pub async fn load_on(root: PathBuf, mode: Mode, device: Device) -> Result<Self, BackendError> {
        if !available_devices().contains(&device) {
            return Err(BackendError::Initialize(format!(
                "MOSS Candle device {device:?} is unavailable; select CUDA/Metal in a matching build"
            )));
        }
        let caps =
            capabilities(&root, mode).map_err(|e| BackendError::Initialize(e.to_string()))?;
        let cache_identity = crate::cache_identity::identity(
            "moss-candle",
            &format!("{}/{}", mode.revision(), resources::CODEC_REVISION),
            device,
            &mode.directory(&root),
        );
        let (requests, jobs) = mpsc::channel(1);
        let (ready, loaded) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("moss-candle-inference".into())
            .spawn(move || runtime::run(root, mode, device, jobs, ready))
            .map_err(|e| BackendError::Initialize(e.to_string()))?;
        let backend = Self {
            requests: Some(requests),
            thread: Some(thread),
            caps,
            cache_identity,
        };
        loaded
            .await
            .map_err(|_| BackendError::Initialize("MOSS Candle thread exited".into()))??;
        Ok(backend)
    }
}
impl Drop for CandleBackend {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Backend for CandleBackend {
    fn cache_identity(&self) -> Option<String> {
        Some(self.cache_identity.clone())
    }
    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }
    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            request.reject_unsupported(self.caps.style)?;
            let (text, voice) = (request.text, request.voice);
            if text.trim().is_empty() || !self.caps.voices.iter().any(|id| id == voice) {
                return Err(BackendError::Unsupported(
                    "unknown MOSS voice or empty text".into(),
                ));
            }
            let resolved = params::resolve(request.params);
            let (audio, receiver) = mpsc::channel(1);
            self.requests
                .as_ref()
                .expect("live inference owner")
                .send(runtime::Request {
                    #[cfg(feature = "performance-tools")]
                    perf: crate::performance::OwnerContext::capture(),

                    text: text.into(),
                    voice: voice.into(),
                    seed: request.seed,
                    resolved,
                    audio,
                })
                .await
                .map_err(|_| BackendError::Synthesis("MOSS Candle thread exited".into()))?;
            Ok(receiver)
        })
    }
    fn next_segment<'a>(
        &'a self,
        request: talechime_core::backend::SegmentationRequest<'a>,
    ) -> Segmentation<'a> {
        Box::pin(async move {
            Ok(talechime_core::text::preprocess_text(request.text, 160)
                .into_iter()
                .next())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn realtime_exposes_no_local_controls_or_cpu_device() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let caps = capabilities(root.path(), Mode::Realtime)?;
        assert_eq!(caps.model.as_deref(), Some("realtime-1.7b"));
        assert!(caps.cloning && caps.native_streaming);
        assert!(!caps.style);
        assert!(!caps.compiled_devices.contains(&Device::Cpu));
        assert_eq!(
            caps.parameters
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["max_duration"]
        );
        for retired in ["nano", "local-1b", "local-1.7b"] {
            assert!(Mode::parse(retired).is_err());
        }
        assert_eq!(std::fs::read_dir(root.path())?.count(), 0);
        Ok(())
    }
    #[cfg(feature = "moss-nano-candle")]
    #[test]
    fn retained_voice_directories_are_independent_and_retired_files_are_untouched()
    -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let retired = root.path().join("moss/voices/retired.json");
        std::fs::create_dir_all(retired.parent().unwrap())?;
        std::fs::write(&retired, b"retired voice must not be read or modified")?;
        let realtime = voice_store(&Mode::Realtime.directory(root.path()), Mode::Realtime)?;
        let nano =
            super::super::voices::VoiceStore::new(&super::super::nano::directory(root.path()));
        nano.save(
            "custom:reader".into(),
            "Reader".into(),
            vec![vec![1; 16]; 20],
        )?;
        assert_eq!(nano.list()?.len(), 1);
        assert!(realtime.list()?.is_empty());
        assert!(realtime.load("custom:reader").is_err());
        assert_eq!(
            std::fs::read(retired)?,
            b"retired voice must not be read or modified"
        );
        Ok(())
    }
}
