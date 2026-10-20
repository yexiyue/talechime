use std::path::{Path, PathBuf};
use talechime_core::backend::{AudioChunk, BackendError, Pcm};
use talechime_protocol::Device;
use tokio::sync::{mpsc, oneshot};

pub(super) struct Request {
    #[cfg(feature = "performance-tools")]
    pub(crate) perf: crate::performance::OwnerContext,
    pub text: String,
    pub voice: String,
    pub language: String,
    pub seed: u64,
    pub audio: mpsc::Sender<Result<AudioChunk, BackendError>>,
}

pub(super) fn candle_device(device: Device) -> candle_core::Result<candle_core::Device> {
    match device {
        Device::Cpu => Ok(candle_core::Device::Cpu),
        #[cfg(all(feature = "indextts-metal", target_os = "macos"))]
        Device::Metal => candle_core::Device::new_metal(0),
        #[cfg(all(
            feature = "indextts-cuda-cudnn",
            any(target_os = "linux", target_os = "windows")
        ))]
        Device::Cuda => candle_core::Device::new_cuda(0),
        _ => candle_core::bail!("unsupported IndexTTS device {device:?}"),
    }
}

pub(super) fn run(
    directory: PathBuf,
    device: Device,
    mut jobs: mpsc::Receiver<Request>,
    ready: oneshot::Sender<Result<(), BackendError>>,
) {
    let loaded = candle_device(device)
        .and_then(|device| talechime_indextts::Model::load(&directory, &device));
    let mut model = match loaded {
        Ok(model) => model,
        Err(error) => {
            let _ = ready.send(Err(BackendError::Initialize(error.to_string())));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    while let Some(request) = jobs.blocking_recv() {
        #[cfg(feature = "performance-tools")]
        let _owner_context = request.perf.enter();
        if request.audio.is_closed() {
            continue;
        }
        if let Err(error) = generate(&mut model, &directory, &request)
            && !request.audio.is_closed()
        {
            let _ = crate::observation::send_audio(&request.audio, Err(error));
        }
    }
}

fn generate(
    model: &mut talechime_indextts::Model,
    directory: &Path,
    request: &Request,
) -> Result<(), BackendError> {
    let store = super::voice_store(directory)
        .map_err(|error| BackendError::Synthesis(error.to_string()))?;
    store
        .load(&request.voice)
        .map_err(|error| BackendError::Unsupported(error.to_string()))?;
    let wav = store
        .path(&request.voice)
        .map_err(|error| BackendError::Unsupported(error.to_string()))?
        .join("reference.wav");
    let samples = crate::reference::load(&wav, talechime_indextts::SAMPLE_RATE)
        .map_err(|error| BackendError::Synthesis(error.to_string()))?;
    let reference =
        talechime_indextts::Reference::from_pcm(&samples, talechime_indextts::SAMPLE_RATE)
            .map_err(|error| BackendError::Synthesis(error.to_string()))?;
    if request.audio.is_closed() {
        return Ok(());
    }
    model
        .prepare_reference(&reference)
        .map_err(|error| BackendError::Synthesis(error.to_string()))?;
    if request.audio.is_closed() {
        return Ok(());
    }
    let options = talechime_indextts::Options {
        seed: Some(request.seed),
        ..Default::default()
    };
    let outcome = model
        .generate_with_cancel(&request.text, &request.language, &options, &|| {
            request.audio.is_closed()
        })
        .map_err(|error| BackendError::Synthesis(error.to_string()))?;
    if request.audio.is_closed() {
        return Ok(());
    }
    let pcm = Pcm {
        samples: outcome
            .samples
            .into_iter()
            .map(|sample| sample / talechime_indextts::PCM16_MAX)
            .collect(),
        sample_rate: outcome.sample_rate,
        channels: 1,
    };
    pcm.duration_ms()?;
    crate::observation::send_audio(&request.audio, Ok(AudioChunk::Pcm(pcm)))
        .map_err(|_| BackendError::Synthesis("IndexTTS receiver closed".into()))?;
    let _ = crate::observation::send_audio(&request.audio, Ok(AudioChunk::End));
    Ok(())
}
