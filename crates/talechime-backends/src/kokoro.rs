//! Official-model Kokoro Mandarin synthesis with built-in voices and segment PCM.
pub mod resources;
mod runtime;
use std::path::{Path, PathBuf};
use talechime_core::backend::{Backend, BackendError, Segmentation, Streaming};
use talechime_protocol::{Capabilities, Device, ParamKind, ParamValue, ParameterSpec};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub const MODEL: &str = "1.1-zh";
#[cfg(feature = "kokoro-g2pw")]
pub const CONTEXT_MODEL: &str = "1.1-zh-g2pw";
pub fn directory(root: &Path) -> PathBuf {
    directory_for(root, MODEL)
}
pub fn directory_for(root: &Path, model: &str) -> PathBuf {
    root.join("kokoro/models")
        .join(model)
        .join(resources::REVISION)
}
pub fn capabilities(directory: &Path) -> anyhow::Result<Capabilities> {
    capabilities_for(directory, MODEL)
}
pub fn capabilities_for(directory: &Path, model: &str) -> anyhow::Result<Capabilities> {
    let voices = if directory.join("manifest.json").exists() {
        resources::read_for(directory, model)?
            .voices
            .into_keys()
            .collect::<Vec<_>>()
    } else {
        vec!["zf_001".to_owned()]
    };
    Ok(Capabilities {
        backend: "kokoro".into(),
        model: Some(model.into()),
        model_name: if model == MODEL {
            "Kokoro 82M v1.1 中文 (Candle)"
        } else {
            "Kokoro 82M v1.1 中文 + g2pW (Candle/ORT CPU)"
        }
        .into(),
        default_voice: if voices.iter().any(|v| v == "zf_001") {
            "zf_001".into()
        } else {
            voices.first().cloned().unwrap_or_default()
        },
        voice_names: voices.iter().map(|id| (id.clone(), id.clone())).collect(),
        voices,
        native_streaming: false,
        cloning: false,
        style: false,
        pronunciation: false,
        compiled_devices: Vec::new(),
        parameters: vec![ParameterSpec {
            name: "speed".into(),
            kind: ParamKind::Float { min: 0.5, max: 2.0 },
            default: ParamValue::Float(1.0),
            description: "官方时长预测语速；中文前端，按完整语义段输出".into(),
        }],
    })
}
pub fn compiled_devices() -> Vec<Device> {
    vec![
        Device::Cpu,
        #[cfg(all(feature = "kokoro-metal", target_os = "macos"))]
        Device::Metal,
        #[cfg(all(
            feature = "kokoro-cuda",
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

pub struct KokoroBackend {
    requests: Option<mpsc::Sender<runtime::Request>>,
    thread: Option<std::thread::JoinHandle<()>>,
    shutdown: CancellationToken,
    caps: Capabilities,
}
impl KokoroBackend {
    pub async fn load_on(directory: PathBuf, device: Device) -> Result<Self, BackendError> {
        Self::load_model_on(directory, MODEL, device).await
    }
    pub async fn load_model_on(
        directory: PathBuf,
        model: &str,
        device: Device,
    ) -> Result<Self, BackendError> {
        let contextual = match model {
            MODEL => false,
            #[cfg(feature = "kokoro-g2pw")]
            CONTEXT_MODEL => true,
            _ => {
                return Err(BackendError::Initialize(format!(
                    "unsupported Kokoro model {model}"
                )));
            }
        };
        if !available_devices().contains(&device) {
            return Err(BackendError::Initialize(format!(
                "Kokoro device {device:?} is unavailable"
            )));
        }
        let caps = capabilities_for(&directory, model)
            .map_err(|e| BackendError::Initialize(e.to_string()))?;
        let shutdown = CancellationToken::new();
        let stop = shutdown.clone();
        let (requests, jobs) = mpsc::channel(1);
        let (ready, loaded) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("kokoro-inference".into())
            .spawn(move || runtime::run(directory, device, contextual, jobs, ready, stop))
            .map_err(|e| BackendError::Initialize(e.to_string()))?;
        let backend = Self {
            requests: Some(requests),
            thread: Some(thread),
            shutdown,
            caps,
        };
        loaded
            .await
            .map_err(|_| BackendError::Initialize("Kokoro owner exited".into()))??;
        Ok(backend)
    }
}
impl Drop for KokoroBackend {
    fn drop(&mut self) {
        self.shutdown.cancel();
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Backend for KokoroBackend {
    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }
    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            request.reject_unsupported(false)?;
            request
                .params
                .validate(&self.caps.parameters)
                .map_err(BackendError::Unsupported)?;
            if request.text.trim().is_empty()
                || !self.caps.voices.iter().any(|v| v == request.voice)
            {
                return Err(BackendError::Unsupported(
                    "Kokoro needs nonempty Mandarin text and an installed built-in voice".into(),
                ));
            }
            let speed = match request.params.get("speed") {
                Some(ParamValue::Float(v)) => *v as f32,
                _ => 1.0,
            };
            let (audio, receiver) = mpsc::channel(1);
            self.requests
                .as_ref()
                .ok_or_else(|| BackendError::Synthesis("Kokoro owner closed".into()))?
                .send(runtime::Request::Synthesize {
                    text: request.text.into(),
                    voice: request.voice.into(),
                    speed,
                    seed: request.seed,
                    audio,
                })
                .await
                .map_err(|_| BackendError::Synthesis("Kokoro owner exited".into()))?;
            Ok(receiver)
        })
    }
    fn next_segment<'a>(
        &'a self,
        request: talechime_core::backend::SegmentationRequest<'a>,
    ) -> Segmentation<'a> {
        Box::pin(async move {
            if request.text.trim().is_empty() {
                return Ok(None);
            }
            let start = request.text.len() - request.text.trim_start().len();
            let mut end = request.text[start..]
                .char_indices()
                .nth(510)
                .map_or(request.text.len(), |(i, _)| start + i);
            let mut numeric_extension = 0;
            while end < request.text.len()
                && request.text[end..].chars().next().is_some_and(|c| {
                    talechime_kokoro::frontend::decimal_digit(c).is_some() || ".%/".contains(c)
                })
            {
                numeric_extension += 1;
                if numeric_extension > 64 {
                    return Err(BackendError::Unsupported(
                        "Kokoro numeric expression exceeds normalization budget".into(),
                    ));
                }
                end += request.text[end..].chars().next().map_or(0, char::len_utf8);
            }
            let (answer, response) = oneshot::channel();
            self.requests
                .as_ref()
                .ok_or_else(|| BackendError::Synthesis("Kokoro owner closed".into()))?
                .send(runtime::Request::Segment {
                    text: request.text[..end].into(),
                    answer,
                })
                .await
                .map_err(|_| BackendError::Synthesis("Kokoro owner exited".into()))?;
            response
                .await
                .map_err(|_| BackendError::Synthesis("Kokoro segment owner exited".into()))?
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use talechime_core::{backend::SegmentRequest, params::GenerationParams};
    #[test]
    fn catalog_is_read_only_and_reports_only_supported_capabilities() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let registry = crate::Registry::new(Some(root.path().to_owned()))?;
        let caps = registry.capabilities("kokoro")?;
        assert_eq!(caps.model.as_deref(), Some(MODEL));
        assert_eq!(caps.default_voice, "zf_001");
        assert!(!caps.native_streaming && !caps.cloning && !caps.style && !caps.pronunciation);
        assert_eq!(std::fs::read_dir(root.path())?.count(), 0);
        assert!(
            registry
                .capabilities_for("kokoro", Some("legacy-onnx"))
                .is_err()
        );
        Ok(())
    }
    #[tokio::test]
    async fn request_validation_and_forwarding_do_not_require_a_model() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let (tx, mut jobs) = mpsc::channel(1);
        let backend = KokoroBackend {
            requests: Some(tx),
            thread: None,
            shutdown: CancellationToken::new(),
            caps: capabilities(root.path())?,
        };
        let mut params = GenerationParams::new();
        params.insert("speed", ParamValue::Float(0.8));
        let request = |style| SegmentRequest {
            text: "你好。",
            voice: "zf_001",
            style,
            seed: 9,
            params: &params,
        };
        assert!(backend.stream(request(Some("悲伤"))).await.is_err());
        assert!(jobs.try_recv().is_err());
        let duration = talechime_core::text::duration::DurationEstimator::default();
        for digit in ["1", "１"] {
            let numeric = digit.repeat(600);
            let error = tokio::time::timeout(
                std::time::Duration::from_millis(100),
                backend.next_segment(talechime_core::backend::SegmentationRequest {
                    text: &numeric,
                    voice: "zf_001",
                    duration: &duration,
                }),
            )
            .await?
            .unwrap_err();
            assert!(matches!(error, BackendError::Unsupported(_)));
            assert!(jobs.try_recv().is_err());
        }
        let _receiver = backend.stream(request(None)).await?;
        match jobs.try_recv()? {
            runtime::Request::Synthesize {
                voice, speed, seed, ..
            } => {
                assert_eq!(voice, "zf_001");
                assert_eq!(speed, 0.8);
                assert_eq!(seed, 9);
            }
            _ => panic!("expected synthesis request"),
        }
        Ok(())
    }
    #[tokio::test]
    async fn missing_conversion_fails_without_writing() {
        let root = tempfile::tempdir().unwrap();
        let (tx, _rx) = mpsc::channel(2);
        let error = resources::prepare(root.path(), tx).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("prepare the pinned official runtime bundle")
        );
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }
}
