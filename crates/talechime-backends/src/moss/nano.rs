//! Candle Nano adapter; shares Nano voice codes with ONNX.
use candle_core::{DType, Device as CandleDevice};
use sentencepiece_rs::SentencePieceProcessor;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::{mpsc, oneshot};
use tts_core::backend::{AudioChunk, Backend, BackendError, Pcm, Segmentation, Streaming};
use tts_protocol::{Capabilities, Device, Event};
pub const REVISION: &str = "44502f80dbf9743528fa921cc544d662c685ebec";
pub fn directory(root: &Path) -> PathBuf {
    root.join("moss/models/nano-candle").join(REVISION)
}
pub fn capabilities(root: &Path) -> anyhow::Result<Capabilities> {
    let mut caps = super::capabilities(&root.join("moss"))?;
    caps.continuation = true;
    caps.model = Some("nano-candle".into());
    caps.model_name = "MOSS Nano Candle".into();
    caps.compiled_devices = compiled_devices();
    Ok(caps)
}
pub fn compiled_devices() -> Vec<Device> {
    vec![
        Device::Cpu,
        #[cfg(all(feature = "moss-nano-candle-metal", target_os = "macos"))]
        Device::Metal,
        #[cfg(all(
            feature = "moss-nano-candle-cuda",
            any(target_os = "windows", target_os = "linux")
        ))]
        Device::Cuda,
    ]
}
fn device(device: Device) -> anyhow::Result<CandleDevice> {
    Ok(match device {
        Device::Cpu => CandleDevice::Cpu,
        #[cfg(all(feature = "moss-nano-candle-metal", target_os = "macos"))]
        Device::Metal => {
            anyhow::ensure!(
                tts_candle_platform::metal_is_available(),
                "Metal device unavailable"
            );
            CandleDevice::new_metal(0)?
        }
        #[cfg(all(
            feature = "moss-nano-candle-cuda",
            any(target_os = "windows", target_os = "linux")
        ))]
        Device::Cuda => CandleDevice::new_cuda(0)?,
        _ => anyhow::bail!("Nano Candle device is not compiled"),
    })
}
pub fn available_devices() -> Vec<Device> {
    compiled_devices()
        .into_iter()
        .filter(|d| device(*d).is_ok())
        .collect()
}
pub async fn prepare(root: &Path, progress: mpsc::Sender<Event>) -> anyhow::Result<()> {
    crate::resources::prepare(
        &directory(root),
        "moss/nano-candle",
        serde_json::from_str(include_str!("nano/resources.json"))?,
        progress,
    )
    .await
}
struct Request {
    #[cfg(feature = "performance-tools")]
    pub(crate) perf: crate::performance::OwnerContext,
    tokens: Vec<i32>,
    none_tokens: Vec<i32>,
    context: Option<tts_core::SpeechContext>,
    voice: String,
    seed: u64,
    audio: mpsc::Sender<Result<AudioChunk, BackendError>>,
}
pub struct NanoBackend {
    requests: Option<mpsc::Sender<Request>>,
    thread: Option<std::thread::JoinHandle<()>>,
    tokenizer: Option<Arc<SentencePieceProcessor>>,
    caps: Capabilities,
    cache_identity: String,
    estimate: std::cell::RefCell<(String, tts_core::text::duration::DurationEstimator)>,
}
impl NanoBackend {
    pub async fn load_on(root: PathBuf, selected: Device) -> Result<Self, BackendError> {
        let caps = capabilities(&root).map_err(|e| BackendError::Initialize(e.to_string()))?;
        let cache_identity = crate::cache_identity::identity(
            "moss-nano-candle",
            REVISION,
            selected,
            &directory(&root),
        );
        let (requests, mut jobs) = mpsc::channel::<Request>(1);
        let (ready, loaded) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("moss-nano-candle".into())
            .spawn(move || {
                let load = || -> anyhow::Result<_> {
                    let device = device(selected)?;
                    let dir = directory(&root);
                    let model = moss_tts::nano::Nano::load(&dir.join("tts"), &device)?;
                    anyhow::ensure!(!ready.is_closed(), "Nano loading cancelled");
                    let codec =
                        moss_tts::codec::AudioCodec::load(&dir.join("codec"), &device, DType::F32)?;
                    let tokenizer = Arc::new(SentencePieceProcessor::open(
                        dir.join("tts/tokenizer.model"),
                    )?);
                    Ok((model, codec, tokenizer))
                };
                let (mut model, mut codec, tokenizer) = match load() {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = ready.send(Err(BackendError::Initialize(e.to_string())));
                        return;
                    }
                };
                if ready.send(Ok(tokenizer)).is_err() {
                    return;
                }
                while let Some(job) = jobs.blocking_recv() {
                    #[cfg(feature = "performance-tools")]
                    let _owner_context = job.perf.enter();
                    if job.audio.is_closed() {
                        continue;
                    }
                    let result =
                        generate(&root, &mut model, &mut codec, &job, &|| jobs.is_closed());
                    if let Err(e) = result
                        && !job.audio.is_closed()
                        && !jobs.is_closed()
                    {
                        let _ = send_audio(
                            &job.audio,
                            Err(BackendError::Synthesis(e.to_string())),
                            &|| jobs.is_closed(),
                        );
                    }
                }
            })
            .map_err(|e| BackendError::Initialize(e.to_string()))?;
        // Acquire the owner before awaiting so cancellation always joins the thread.
        let backend = Self {
            requests: Some(requests),
            thread: Some(thread),
            tokenizer: None,
            caps,
            cache_identity,
            estimate: std::cell::RefCell::new((String::new(), Default::default())),
        };
        struct Loading {
            receiver: oneshot::Receiver<Result<Arc<SentencePieceProcessor>, BackendError>>,
            backend: Option<NanoBackend>,
        }
        impl Drop for Loading {
            fn drop(&mut self) {
                self.receiver.close();
                self.backend.take();
            }
        }
        let mut loading = Loading {
            receiver: loaded,
            backend: Some(backend),
        };
        let tokenizer = (&mut loading.receiver)
            .await
            .map_err(|_| BackendError::Initialize("Nano thread exited".into()))??;
        let mut backend = loading.backend.take().expect("loading owner");
        backend.tokenizer = Some(tokenizer);
        Ok(backend)
    }
    pub async fn stream_seeded(
        &self,
        text: &str,
        voice: &str,
        seed: u64,
    ) -> Result<tts_core::backend::AudioStream, BackendError> {
        self.stream_seeded_with_context(text, voice, seed, None)
            .await
    }
    pub async fn stream_seeded_with_context(
        &self,
        text: &str,
        voice: &str,
        seed: u64,
        context: Option<&tts_core::SpeechContext>,
    ) -> Result<tts_core::backend::AudioStream, BackendError> {
        let text = super::text::normalize(text);
        if text.is_empty() || !self.caps.voices.iter().any(|v| v == voice) {
            return Err(BackendError::Unsupported(
                "empty text or unknown Nano voice".into(),
            ));
        }
        let effective = context.map_or_else(
            || text.clone(),
            |previous| super::text::normalize(previous.text()) + &text,
        );
        let none_tokens = self
            .tokenizer
            .as_ref()
            .expect("loaded tokenizer")
            .encode_to_ids("None")
            .map_err(|e| BackendError::Synthesis(e.to_string()))?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let tokens = self
            .tokenizer
            .as_ref()
            .expect("loaded tokenizer")
            .encode_to_ids(&effective)
            .map_err(|e| BackendError::Synthesis(e.to_string()))?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let (audio, stream) = mpsc::channel(1);
        self.requests
            .as_ref()
            .expect("live Nano owner")
            .send(Request {
                #[cfg(feature = "performance-tools")]
                perf: crate::performance::OwnerContext::capture(),
                tokens,
                none_tokens,
                context: context.cloned(),
                voice: voice.into(),
                seed,
                audio,
            })
            .await
            .map_err(|_| BackendError::Synthesis("Nano thread exited".into()))?;
        Ok(stream)
    }
}
impl Drop for NanoBackend {
    fn drop(&mut self) {
        self.requests.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Backend for NanoBackend {
    fn cache_identity(&self) -> Option<String> {
        Some(self.cache_identity.clone())
    }
    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }
    fn stream<'a>(&'a self, request: tts_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            request.reject_unsupported(self.caps.style, self.caps.continuation)?;
            self.stream_seeded_with_context(
                request.text,
                request.voice,
                request.seed,
                request.context,
            )
            .await
        })
    }
    fn paragraph_end(&self, segment: &str, remaining: &str) -> bool {
        super::text::paragraph_end(segment, remaining)
    }
    fn select_voice(&self, voice: &str) {
        let mut estimate = self.estimate.borrow_mut();
        if estimate.0 != voice {
            *estimate = (voice.into(), Default::default());
        }
    }
    fn observe_duration(&self, text: &str, voice: &str, seconds: f64) {
        self.select_voice(voice);
        self.estimate.borrow_mut().1.observe(text, seconds);
    }
    fn next_segment<'a>(
        &'a self,
        text: &'a str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<Option<tts_core::text::TextSegment>, BackendError>,
                > + 'a,
        >,
    > {
        Box::pin(async move {
            super::text::first_segment(
                text,
                self.tokenizer.as_ref().expect("loaded tokenizer"),
                &self.estimate.borrow().1,
            )
        })
    }
    fn segments<'a>(&'a self, text: &'a str) -> Segmentation<'a> {
        Box::pin(async move {
            super::text::segments_with_estimate(
                text,
                self.tokenizer.as_ref().expect("loaded tokenizer"),
                &self.estimate.borrow().1,
            )
        })
    }
}
fn generate(
    root: &Path,
    model: &mut moss_tts::nano::Nano,
    codec: &mut moss_tts::codec::AudioCodec,
    job: &Request,
    owner_closed: &impl Fn() -> bool,
) -> anyhow::Result<()> {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("assets/browser_poc_manifest.json"))?;
    let cancelled = || job.audio.is_closed() || owner_closed();
    let codes: Vec<Vec<i32>> = if let Some(context) = &job.context {
        anyhow::ensure!(
            context.pcm().sample_rate == 48000 && context.pcm().channels == 2,
            "Nano continuation requires original stereo 48 kHz PCM"
        );
        codec
            .encode(&context.pcm().samples, 16, &cancelled)?
            .into_iter()
            .map(|frame| frame.into_iter().map(|id| id as i32).collect())
            .collect()
    } else {
        super::voices::VoiceStore::new(&root.join("moss")).codes(&job.voice, &manifest)?
    };
    let flat = super::prompt::rows(
        &manifest,
        &job.tokens,
        &codes,
        job.context.as_ref().map(|_| job.none_tokens.as_slice()),
    )?;
    let rows: Vec<Vec<u32>> = flat
        .as_chunks::<17>()
        .0
        .iter()
        .map(|row| row.iter().map(|id| *id as u32).collect())
        .collect();
    codec.reset_decoder();
    if job.context.is_some() {
        for batch in codes.chunks(3) {
            let frames: Vec<Vec<u32>> = batch
                .iter()
                .map(|row| row.iter().map(|id| *id as u32).collect())
                .collect();
            let _prefix_pcm = codec.decode(&frames, &cancelled)?;
        }
    }
    let mut pending = Vec::new();
    let send = |codec: &mut moss_tts::codec::AudioCodec,
                pending: &mut Vec<Vec<u32>>|
     -> anyhow::Result<()> {
        let samples = codec.decode(pending, &cancelled)?;
        let pcm = Pcm {
            samples,
            sample_rate: codec.sample_rate,
            channels: codec.channels as u16,
        };
        pcm.duration_ms()?;
        send_audio(&job.audio, Ok(AudioChunk::Pcm(pcm)), &cancelled)?;
        pending.clear();
        Ok(())
    };
    generate_nonempty(job.seed, |seed| {
        let mut generated = 0usize;
        model.generate(&rows, 375, seed, &cancelled, |frame| {
            generated += 1;
            pending.push(frame);
            if pending.len() == 3 {
                send(codec, &mut pending)?;
            }
            Ok(())
        })?;
        Ok(generated)
    })?;
    if !pending.is_empty() {
        send(codec, &mut pending)?;
    }
    send_audio(&job.audio, Ok(AudioChunk::End), &cancelled)?;
    Ok(())
}

/// Nano may sample EOS at its first step. Retry only that zero-frame result:
/// the decoder and consumer have received no new frames, so no PCM is replayed.
/// Seed derivation stays reproducible and distinct in Nano's 32-bit sampler.
fn generate_nonempty(
    seed: u64,
    mut generate: impl FnMut(u64) -> anyhow::Result<usize>,
) -> anyhow::Result<()> {
    for attempt in 0..3u32 {
        let attempt_seed = (seed as u32).wrapping_add(attempt.wrapping_mul(0x9e37_79b9));
        if generate(u64::from(attempt_seed))? > 0 {
            return Ok(());
        }
    }
    anyhow::bail!("Nano produced empty speech after 3 attempts")
}

#[cfg(test)]
mod empty_eos_tests {
    use super::*;

    #[test]
    fn empty_eos_retries_with_distinct_reproducible_seeds() {
        let mut seeds = Vec::new();
        generate_nonempty(u64::MAX, |seed| {
            seeds.push(seed);
            Ok(usize::from(seeds.len() == 2))
        })
        .unwrap();
        assert_eq!(seeds, [u64::from(u32::MAX), 0x9e37_79b8]);
    }

    #[test]
    fn empty_eos_has_a_fixed_attempt_limit() {
        let mut attempts = 0;
        let error = generate_nonempty(42, |_| {
            attempts += 1;
            Ok(0)
        })
        .unwrap_err();
        assert_eq!(attempts, 3);
        assert!(error.to_string().contains("empty speech after 3 attempts"));
    }

    #[test]
    fn successful_generation_and_other_errors_are_never_retried() {
        for fail in [false, true] {
            let mut attempts = 0;
            let result = generate_nonempty(42, |_| {
                attempts += 1;
                if fail {
                    anyhow::bail!("cancelled or failed after partial output");
                }
                Ok(1)
            });
            assert_eq!(attempts, 1);
            assert_eq!(result.is_err(), fail);
        }
    }
}

/// Keep bounded backpressure while allowing the owner to cancel a full queue.
fn send_audio(
    sender: &mpsc::Sender<Result<AudioChunk, BackendError>>,
    mut chunk: Result<AudioChunk, BackendError>,
    cancelled: &impl Fn() -> bool,
) -> anyhow::Result<()> {
    #[cfg(feature = "performance-tools")]
    let observation = crate::observation::SendObservation::new(&chunk);
    loop {
        anyhow::ensure!(!cancelled(), "Nano cancelled");
        match sender.try_send(chunk) {
            Ok(()) => {
                #[cfg(feature = "performance-tools")]
                observation.complete(true);
                return Ok(());
            }
            Err(mpsc::error::TrySendError::Closed(_)) => anyhow::bail!("Nano stream closed"),
            Err(mpsc::error::TrySendError::Full(value)) => {
                chunk = value;
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_first_step_eos_recovers_without_skipping_text() -> anyhow::Result<()> {
        let Some(root) = std::env::var_os("TALECHIME_NANO_EMPTY_ROOT") else {
            return Ok(());
        };
        let root = PathBuf::from(root);
        let dir = directory(&root);
        let device = CandleDevice::Cpu;
        let mut model = moss_tts::nano::Nano::load(&dir.join("tts"), &device)?;
        let mut codec = moss_tts::codec::AudioCodec::load(&dir.join("codec"), &device, DType::F32)?;
        let tokenizer = SentencePieceProcessor::open(dir.join("tts/tokenizer.model"))?;
        let tokens: Vec<i32> = tokenizer
            .encode_to_ids(&super::super::text::normalize("陆鸣喃喃。"))?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("assets/browser_poc_manifest.json"))?;
        let codes =
            super::super::voices::VoiceStore::new(&root.join("moss")).codes("Yuewen", &manifest)?;
        let flat = super::super::prompt::rows(&manifest, &tokens, &codes, None)?;
        let rows: Vec<Vec<u32>> = flat
            .as_chunks::<17>()
            .0
            .iter()
            .map(|row| row.iter().map(|id| *id as u32).collect())
            .collect();
        // This LCG seed's first draw is 1 - 2^-24: it selects EOS for
        // this real prompt, reproducing the original zero-frame failure.
        let seed = 653_637_408;
        let mut frames = 0;
        model.generate(&rows, 375, seed, &|| false, |_| {
            frames += 1;
            Ok(())
        })?;
        assert_eq!(frames, 0, "probe must first reproduce empty EOS");
        let (audio, mut stream) = mpsc::channel::<Result<AudioChunk, BackendError>>(1);
        let consumer = std::thread::spawn(move || {
            let mut samples = 0;
            let mut ended = false;
            while let Some(chunk) = stream.blocking_recv() {
                match chunk.unwrap() {
                    AudioChunk::Pcm(pcm) => samples += pcm.samples.len(),
                    AudioChunk::End => {
                        ended = true;
                        break;
                    }
                }
            }
            (samples, ended)
        });
        let job = Request {
            #[cfg(feature = "performance-tools")]
            perf: crate::performance::OwnerContext::capture(),
            tokens,
            none_tokens: vec![],
            context: None,
            voice: "Yuewen".into(),
            seed,
            audio,
        };
        let result = generate(&root, &mut model, &mut codec, &job, &|| false);
        drop(job);
        let (samples, ended) = consumer.join().unwrap();
        result?;
        eprintln!("Nano first-step EOS: zero initial frames; recovered {samples} PCM samples");
        assert!(samples > 0 && ended);
        Ok(())
    }

    #[test]
    fn cancellation_releases_a_full_pcm_queue() {
        let (sender, _receiver) = mpsc::channel(1);
        sender.try_send(Ok(AudioChunk::End)).unwrap();
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let signal = cancelled.clone();
        let thread = std::thread::spawn(move || {
            send_audio(&sender, Ok(AudioChunk::End), &|| {
                signal.load(std::sync::atomic::Ordering::Relaxed)
            })
        });
        cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(thread.join().unwrap().is_err());
    }
    #[test]
    fn catalog_shares_nano_voices_and_defaults_to_candle() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let registry = crate::Registry::new(Some(root.path().into()))?;
        let onnx = registry.capabilities_for("moss", Some("nano"))?;
        let candle = registry.capabilities_for("moss", None)?;
        assert_eq!(candle.model.as_deref(), Some("nano-candle"));
        assert_eq!(onnx.model.as_deref(), Some("nano"));
        assert_eq!(onnx.voices, candle.voices);
        assert_eq!(candle.default_voice, "Weiguo");
        assert!(candle.compiled_devices.contains(&Device::Cpu));
        assert!(!root.path().join("moss").exists());
        super::super::voices::VoiceStore::new(&root.path().join("moss")).save(
            "custom:shared".into(),
            "shared".into(),
            vec![vec![42; 16]; 3],
        )?;
        assert!(
            registry
                .capabilities_for("moss", Some("nano-candle"))?
                .voices
                .iter()
                .any(|id| id == "custom:shared")
        );
        Ok(())
    }
}
