use candle_core::{DType, IndexOp};
use std::{collections::HashMap, path::PathBuf};
use talechime_core::{
    backend::{AudioChunk, BackendError, Pcm},
    text::TextSegment,
};
use talechime_kokoro::{Model, SynthesisOptions, Trace, frontend::Frontend};
use talechime_protocol::Device;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub(super) enum Request {
    Segment {
        text: String,
        answer: oneshot::Sender<Result<Option<TextSegment>, BackendError>>,
    },
    Synthesize {
        text: String,
        voice: String,
        speed: f32,
        seed: u64,
        audio: mpsc::Sender<Result<AudioChunk, BackendError>>,
    },
}
pub(super) fn candle_device(device: Device) -> candle_core::Result<candle_core::Device> {
    match device {
        Device::Cpu => Ok(candle_core::Device::Cpu),
        #[cfg(all(feature = "kokoro-metal", target_os = "macos"))]
        Device::Metal => candle_core::Device::new_metal(0),
        #[cfg(all(
            feature = "kokoro-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        Device::Cuda => candle_core::Device::new_cuda(0),
        _ => candle_core::bail!("unsupported Kokoro device {device:?}"),
    }
}
pub(super) fn run(
    directory: PathBuf,
    device: Device,
    contextual: bool,
    mut jobs: mpsc::Receiver<Request>,
    ready: oneshot::Sender<Result<(), BackendError>>,
    shutdown: CancellationToken,
) {
    #[cfg(feature = "kokoro-g2pw")]
    let mut context = None;
    let loaded = (|| -> anyhow::Result<_> {
        let manifest =
            super::resources::verify_cancellable(&directory, &|| shutdown.is_cancelled())?;
        anyhow::ensure!(
            manifest.contextual == contextual,
            "Kokoro pronunciation mode changed during loading"
        );
        anyhow::ensure!(!shutdown.is_cancelled(), "Kokoro loading cancelled");
        let device = candle_device(device)?;
        let model = Model::load(
            &directory.join("model.safetensors"),
            &directory.join("config.json"),
            &device,
        )?;
        anyhow::ensure!(!shutdown.is_cancelled(), "Kokoro loading cancelled");
        let frontend = Frontend::load(&directory.join("frontend"))?;
        #[cfg(feature = "kokoro-g2pw")]
        {
            context = if contextual {
                Some(talechime_kokoro::context::ContextModel::load(
                    &directory.join("context"),
                )?)
            } else {
                None
            };
        }
        anyhow::ensure!(!shutdown.is_cancelled(), "Kokoro loading cancelled");
        let mut voices = HashMap::new();
        for (id, voice) in manifest.voices {
            anyhow::ensure!(!shutdown.is_cancelled(), "Kokoro loading cancelled");
            let values = candle_core::safetensors::load(directory.join(voice.path), &device)?;
            let tensor = values
                .get("voice")
                .ok_or_else(|| anyhow::anyhow!("missing Kokoro voice tensor"))?;
            anyhow::ensure!(
                tensor.dims() == [510, 1, 256] && tensor.dtype() == DType::F32,
                "invalid Kokoro voice pack"
            );
            voices.insert(id, tensor.clone());
        }
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        Ok((model, frontend, voices, runtime))
    })();
    let (model, frontend, voices, runtime) = match loaded {
        Ok(value) => value,
        Err(error) => {
            let _ = ready.send(Err(BackendError::Initialize(error.to_string())));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    while !shutdown.is_cancelled() {
        let Some(request) = jobs.blocking_recv() else {
            break;
        };
        match request {
            Request::Segment { text, answer } => {
                if answer.is_closed() {
                    continue;
                }
                let result = frontend
                    .segments(&text, 180)
                    .map(|segments| {
                        segments.into_iter().next().map(|segment| TextSegment {
                            text: text[segment.range.clone()].to_owned(),
                            start: segment.range.start,
                            end: segment.range.end,
                        })
                    })
                    .map_err(|e| BackendError::Unsupported(e.to_string()));
                let _ = answer.send(result);
            }
            Request::Synthesize {
                text,
                voice,
                speed,
                seed,
                audio,
            } => {
                if audio.is_closed() {
                    continue;
                }
                let result = (|| -> anyhow::Result<_> {
                    #[cfg(feature = "kokoro-g2pw")]
                    let phonemes = match &mut context {
                        Some(context) => frontend.phonemes_with_context(&text, context, &|| {
                            shutdown.is_cancelled() || audio.is_closed()
                        })?,
                        None => frontend.phonemes(&text)?,
                    };
                    #[cfg(not(feature = "kokoro-g2pw"))]
                    let phonemes = frontend.phonemes(&text)?;
                    let ids = model.token_ids(&phonemes)?;
                    let voice = voices
                        .get(&voice)
                        .ok_or_else(|| anyhow::anyhow!("Kokoro built-in voice was not prepared"))?;
                    let style = voice.i(phonemes.chars().count() - 1)?.reshape((1, 256))?;
                    let options = SynthesisOptions {
                        speed,
                        seed,
                        // A single PCM block must fit the core's 30 s limit.
                        max_samples: 24000 * 30,
                    };
                    let output = model.forward_cancellable(
                        &ids,
                        &style,
                        &options,
                        None,
                        &mut Trace::default(),
                        &|| shutdown.is_cancelled() || audio.is_closed(),
                    )?;
                    let pcm = Pcm {
                        samples: output.audio.to_vec1::<f32>()?,
                        sample_rate: 24000,
                        channels: 1,
                    };
                    pcm.duration_ms()?;
                    Ok(pcm)
                })();
                if audio.is_closed() || shutdown.is_cancelled() {
                    continue;
                }
                let result = result
                    .map(AudioChunk::Pcm)
                    .map_err(|error| BackendError::Synthesis(error.to_string()));
                runtime.block_on(deliver(&audio, &shutdown, result));
            }
        }
    }
}

async fn deliver(
    audio: &mpsc::Sender<Result<AudioChunk, BackendError>>,
    shutdown: &CancellationToken,
    result: Result<AudioChunk, BackendError>,
) {
    let success = result.is_ok();
    tokio::select! {
        biased;
        _=shutdown.cancelled()=>return,
        permit=audio.reserve()=>{
            let Ok(permit)=permit else{return;};
            if shutdown.is_cancelled(){return;}
            permit.send(result);
        }
    }
    if success {
        tokio::select! {biased;_=shutdown.cancelled()=>{},_=audio.send(Ok(AudioChunk::End))=>{}}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pcm() -> Result<AudioChunk, BackendError> {
        Ok(AudioChunk::Pcm(Pcm {
            samples: vec![0.0; 240],
            sample_rate: 24000,
            channels: 1,
        }))
    }
    #[tokio::test]
    async fn cancellation_releases_a_full_channel_without_successful_end() {
        let (tx, mut rx) = mpsc::channel(1);
        let stop = CancellationToken::new();
        let owner = stop.clone();
        let task = tokio::spawn(async move {
            deliver(&tx, &owner, pcm()).await;
        });
        while rx.is_empty() {
            tokio::task::yield_now().await;
        }
        stop.cancel();
        task.await.unwrap();
        assert!(matches!(rx.recv().await, Some(Ok(AudioChunk::Pcm(_)))));
        assert!(rx.recv().await.is_none());
    }
    #[tokio::test]
    async fn closed_consumer_and_failure_do_not_publish_end() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        deliver(&tx, &CancellationToken::new(), pcm()).await;
        let (tx, mut rx) = mpsc::channel(1);
        deliver(
            &tx,
            &CancellationToken::new(),
            Err(BackendError::Synthesis("failure".into())),
        )
        .await;
        assert!(matches!(rx.recv().await, Some(Err(_))));
        assert!(rx.try_recv().is_err());
    }
}
