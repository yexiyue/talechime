use std::path::PathBuf;
use talechime_core::backend::{AudioChunk, BackendError, Pcm};
use talechime_cosyvoice3::{
    DEFAULT_PROMPT_PREFIX, SamplingConfig,
    frontend::ort_models::FrontendOnnx,
    pipeline::{CosyVoice3, Mode, extract_prompt_features},
};
use talechime_protocol::Device;
use tokio::sync::{mpsc, oneshot};

pub(super) struct Request {
    #[cfg(feature = "performance-tools")]
    pub(crate) perf: crate::performance::OwnerContext,
    pub text: String,
    pub voice: String,
    pub seed: u64,
    pub instruction: Option<String>,
    pub cross_lingual: bool,
    pub text_frontend: bool,
    pub audio: mpsc::Sender<Result<AudioChunk, BackendError>>,
}
/// Wrap a plain style in the official Instruct prompt format.
pub(super) fn instruction(style: Option<&str>) -> Result<Option<String>, BackendError> {
    let Some(style) = style.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    if style.chars().count() > 200 || style.contains("<|") || style.contains("|>") {
        return Err(BackendError::Unsupported(
            "CosyVoice3 style must be at most 200 characters without special prompt markers".into(),
        ));
    }
    Ok(Some(format!(
        "You are a helpful assistant. {style}<|endofprompt|>"
    )))
}

pub(super) fn candle_device(device: Device) -> candle_core::Result<candle_core::Device> {
    match device {
        Device::Cpu => Ok(candle_core::Device::Cpu),
        #[cfg(all(feature = "cosyvoice3-metal", target_os = "macos"))]
        Device::Metal => candle_core::Device::new_metal(0),
        #[cfg(all(
            feature = "cosyvoice3-cuda",
            any(target_os = "linux", target_os = "windows")
        ))]
        Device::Cuda => candle_core::Device::new_cuda(0),
        _ => candle_core::bail!("unsupported CosyVoice3 device {device:?}"),
    }
}
pub(super) fn run(
    root: PathBuf,
    device: Device,
    mut jobs: mpsc::Receiver<Request>,
    ready: oneshot::Sender<Result<(), BackendError>>,
) {
    let directory = super::directory(&root);
    let loaded = (|| -> anyhow::Result<_> {
        let device = candle_device(device)?;
        let pipeline = CosyVoice3::load(&directory, &device, candle_core::DType::F32, 0)?;
        let frontend = FrontendOnnx::load(&directory, 1)?;
        let text_frontend =
            talechime_cosyvoice3::frontend::text::TextFrontend::load(&directory.join("tn/wetext"))?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        Ok((pipeline, frontend, text_frontend, runtime))
    })();
    let (mut pipeline, mut frontend, text_frontend, runtime) = match loaded {
        Ok(parts) => parts,
        Err(error) => {
            let _ = ready.send(Err(BackendError::Initialize(error.to_string())));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    while let Some(request) = jobs.blocking_recv() {
        if request.audio.is_closed() {
            continue;
        }
        let result = (|| -> anyhow::Result<()> {
            let store = super::voice_store(&root)?;
            let voice = store.load(&request.voice)?;
            let samples =
                crate::reference::load(&store.path(&request.voice)?.join("reference.wav"), 24000)?;
            let prompt_text = format!("{DEFAULT_PROMPT_PREFIX}{}", voice.transcript);
            let prompt = extract_prompt_features(&mut frontend, &samples, 24000, &prompt_text)?;
            if request.audio.is_closed() {
                return Ok(());
            }
            let sampling = SamplingConfig {
                seed: Some(request.seed),
                ..Default::default()
            };
            let mode = if request.cross_lingual {
                Mode::CrossLingual
            } else if request.instruction.is_some() {
                Mode::Instruct
            } else {
                Mode::ZeroShot
            };
            let model_text = mode_text(&request.text, mode);
            let texts = text_frontend.prepare(
                &model_text,
                request.text_frontend,
                |text| pipeline.text_token_count(text),
                &|| request.audio.is_closed(),
            )?;
            anyhow::ensure!(
                !texts.is_empty(),
                "CosyVoice3 text contains only punctuation"
            );
            for (index, text) in texts.iter().enumerate() {
                if request.audio.is_closed() {
                    return Ok(());
                }
                let mut session = pipeline.start_stream_with_mode(
                    text,
                    &prompt,
                    mode,
                    request.instruction.as_deref(),
                    &sampling,
                )?;
                let delivery = deliver_segment(
                    &request.audio,
                    || Ok(session.next_pcm()?),
                    index + 1 == texts.len(),
                );
                #[cfg(feature = "performance-tools")]
                let delivery = tracing::Instrument::instrument(delivery, request.perf.span());
                runtime.block_on(delivery)?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            let _ = crate::observation::send_audio(
                &request.audio,
                Err(BackendError::Synthesis(error.to_string())),
            );
        }
    }
}

/// Fixed CosyVoice3 example.py supplies this marker in the target input for
/// cross-lingual mode, whose LLM prompt transcript has been removed.
fn mode_text(text: &str, mode: Mode) -> String {
    if mode == Mode::CrossLingual && !text.contains("<|endofprompt|>") {
        format!("{DEFAULT_PROMPT_PREFIX}{text}")
    } else {
        text.to_owned()
    }
}

/// Reserve a slot before inference. A dropped consumer cancels between blocks;
/// exhaustion or a token budget limit never publishes successful End.
async fn deliver_segment(
    audio: &mpsc::Sender<Result<AudioChunk, BackendError>>,
    mut next: impl FnMut() -> anyhow::Result<Option<talechime_cosyvoice3::StreamChunk>>,
    final_segment: bool,
) -> anyhow::Result<()> {
    loop {
        #[cfg(feature = "performance-tools")]
        let waiting = std::time::Instant::now();
        let permit = match audio.reserve().await {
            Ok(permit) => permit,
            Err(_) => return Ok(()),
        };
        #[cfg(feature = "performance-tools")]
        {
            let seconds = waiting.elapsed().as_secs_f64();
            metrics::histogram!("talechime.channel_wait_seconds","stage"=>"owner_reserve")
                .record(seconds);
            tracing::debug!(
                target:"talechime_perf",
                event="channel_wait",
                duration_ms = waiting.elapsed().as_millis() as u64,
            );
        }
        let chunk = next()?
            .ok_or_else(|| anyhow::anyhow!("CosyVoice3 stream ended without a final block"))?;
        anyhow::ensure!(
            !chunk.hit_token_limit,
            "CosyVoice3 exhausted its speech token budget"
        );
        let pcm = Pcm {
            samples: chunk.samples,
            sample_rate: 24000,
            channels: 1,
        };
        pcm.duration_ms()?;
        #[cfg(feature = "performance-tools")]
        tracing::debug!(target:"talechime_perf",event="owner_pcm_ready");
        permit.send(Ok(AudioChunk::Pcm(pcm)));
        #[cfg(feature = "performance-tools")]
        tracing::debug!(target:"talechime_perf",event="adapter_channel_enqueue");
        if chunk.final_chunk {
            if final_segment {
                let _ = audio.send(Ok(AudioChunk::End)).await;
            }
            return Ok(());
        }
    }
}

#[cfg(test)]
async fn deliver(
    audio: &mpsc::Sender<Result<AudioChunk, BackendError>>,
    next: impl FnMut() -> anyhow::Result<Option<talechime_cosyvoice3::StreamChunk>>,
) -> anyhow::Result<()> {
    deliver_segment(audio, next, true).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use talechime_cosyvoice3::StreamChunk;
    #[test]
    fn cross_lingual_supplies_the_official_target_marker() {
        assert_eq!(
            mode_text("hello", Mode::CrossLingual),
            format!("{DEFAULT_PROMPT_PREFIX}hello")
        );
        let tagged = format!("{DEFAULT_PROMPT_PREFIX}hello");
        assert_eq!(mode_text(&tagged, Mode::CrossLingual), tagged);
        assert_eq!(mode_text("hello", Mode::ZeroShot), "hello");
        assert_eq!(mode_text("hello", Mode::Instruct), "hello");
    }
    fn chunk(final_chunk: bool) -> StreamChunk {
        StreamChunk {
            samples: vec![0.1; 240],
            final_chunk,
            hit_token_limit: false,
            tokens_generated: 1,
        }
    }
    #[test]
    fn style_wraps_once_and_default_keeps_zero_shot() {
        assert_eq!(instruction(None).unwrap(), None);
        assert_eq!(instruction(Some("  ")).unwrap(), None);
        assert_eq!(
            instruction(Some(" 请用悲伤的语气说。 "))
                .unwrap()
                .as_deref(),
            Some("You are a helpful assistant. 请用悲伤的语气说。<|endofprompt|>")
        );
        assert!(instruction(Some("bad<|endofprompt|>")).is_err());
        assert!(instruction(Some(&"字".repeat(201))).is_err());
    }

    #[tokio::test]
    async fn dropped_consumer_prevents_any_inference() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        deliver(&tx, || panic!("consumer already closed"))
            .await
            .unwrap();
    }
    #[tokio::test]
    async fn full_queue_blocks_inference_and_only_final_chunk_emits_end() {
        let (tx, mut rx) = mpsc::channel(1);
        let calls = std::cell::Cell::new(0);
        let producer = deliver(&tx, || {
            calls.set(calls.get() + 1);
            Ok(Some(chunk(calls.get() == 2)))
        });
        tokio::pin!(producer);
        tokio::select! {
            result = &mut producer => panic!("completed before consuming audio: {result:?}"),
            _ = tokio::task::yield_now() => {}
        }
        assert_eq!(calls.get(), 1);
        assert!(matches!(rx.recv().await, Some(Ok(AudioChunk::Pcm(_)))));
        tokio::select! {
            result = &mut producer => panic!("completed before consuming final audio: {result:?}"),
            _ = tokio::task::yield_now() => {}
        }
        assert_eq!(calls.get(), 2);
        assert!(matches!(rx.recv().await, Some(Ok(AudioChunk::Pcm(_)))));
        producer.await.unwrap();
        assert!(matches!(rx.recv().await, Some(Ok(AudioChunk::End))));
    }
    #[tokio::test]
    async fn limits_disconnects_and_inference_errors_never_emit_success() {
        for mode in 0..3 {
            let (tx, mut rx) = mpsc::channel(2);
            let result = deliver(&tx, || match mode {
                0 => {
                    let mut c = chunk(true);
                    c.hit_token_limit = true;
                    Ok(Some(c))
                }
                1 => Ok(None),
                _ => anyhow::bail!("injected failure"),
            })
            .await;
            assert!(result.is_err());
            assert!(rx.try_recv().is_err());
        }
    }
}
