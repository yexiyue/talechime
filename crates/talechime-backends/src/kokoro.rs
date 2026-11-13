//! Official-model Kokoro Mandarin synthesis with built-in voices and segment PCM.
pub mod resources;
mod runtime;
use std::path::{Path, PathBuf};
use talechime_core::backend::{Backend, BackendError, Segmentation, Streaming};
use talechime_protocol::{Capabilities, Device, ParamKind, ParamValue, ParameterSpec};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub const MODEL: &str = "1.1-zh-g2pw";
pub fn directory(root: &Path) -> PathBuf {
    directory_for(root, MODEL)
}
pub fn directory_for(root: &Path, model: &str) -> PathBuf {
    root.join("kokoro/models")
        .join(model)
        .join(resources::REVISION)
}
pub(crate) fn cached_capabilities(model: &str) -> anyhow::Result<Capabilities> {
    anyhow::ensure!(model == MODEL, "unsupported Kokoro model");
    Ok(capabilities_with_voices(
        model,
        resources::compiled_manifest()?.voices.into_keys().collect(),
    ))
}
fn capabilities_with_voices(model: &str, voices: Vec<String>) -> Capabilities {
    Capabilities {
        backend: "kokoro".into(),
        model: Some(model.into()),
        model_name: "Kokoro 82M v1.1 中文 + g2pW (Candle/ORT CPU)".into(),
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
    }
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
    pub(crate) async fn load_cached(
        files: std::collections::BTreeMap<String, PathBuf>,
        model: &str,
        device: Device,
    ) -> Result<Self, BackendError> {
        let caps =
            cached_capabilities(model).map_err(|e| BackendError::Initialize(e.to_string()))?;
        Self::load_input(files, caps, model, device).await
    }
    async fn load_input(
        input: std::collections::BTreeMap<String, PathBuf>,
        caps: Capabilities,
        model: &str,
        device: Device,
    ) -> Result<Self, BackendError> {
        if model != MODEL {
            return Err(BackendError::Initialize(format!(
                "unsupported Kokoro model {model}"
            )));
        }
        if !available_devices().contains(&device) {
            return Err(BackendError::Initialize(format!(
                "Kokoro device {device:?} is unavailable"
            )));
        }
        let shutdown = CancellationToken::new();
        let stop = shutdown.clone();
        let (requests, jobs) = mpsc::channel(1);
        let (ready, loaded) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("kokoro-inference".into())
            .spawn(move || runtime::run(input, device, jobs, ready, stop))
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
    #[test]
    fn resolved_revision_tracks_text_semantics_without_changing_resources() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let registry = crate::Registry::new(Some(root.path().to_owned()))?;
        let expected = "01e7505-runtime-f32-v3-text-v5-loqui-neural-0.4.0";
        assert_eq!(
            registry.resolve("kokoro", Some(MODEL))?.revision(),
            expected
        );
        assert_eq!(registry.resolve("kokoro", None)?.revision(), expected);
        assert!(registry.resolve("kokoro", Some("1.1-zh")).is_err());
        assert_eq!(resources::REVISION, "01e7505-runtime-f32-v3");
        assert!(directory_for(root.path(), MODEL).ends_with(resources::REVISION));
        Ok(())
    }
    #[tokio::test]
    async fn request_validation_and_forwarding_do_not_require_a_model() -> anyhow::Result<()> {
        let (tx, mut jobs) = mpsc::channel(1);
        let backend = KokoroBackend {
            requests: Some(tx),
            thread: None,
            shutdown: CancellationToken::new(),
            caps: cached_capabilities(MODEL)?,
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
}
