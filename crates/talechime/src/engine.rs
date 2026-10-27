use crate::*;
use std::{
    cell::RefCell,
    future::Future,
    path::PathBuf,
    rc::{Rc, Weak},
    sync::Arc,
};
use talechime_core::SynthesisStream;

/// Model preparation is explicit and independent of playback/config persistence.
#[derive(Debug, Clone)]
pub struct ModelOptions {
    /// Compiled adapter ID (e.g. moss, qwen, voxcpm, omnivoice).
    pub backend: String,
    /// Optional model selection; resolved to the prepared capability identity.
    pub model: Option<String>,
    /// Requested inference device; Auto uses the explicit calibration/recovery policy.
    pub device: Device,
    /// Measure CPU/accelerator performance when selecting Auto.
    pub calibrate: bool,
    /// Enable adapter-owned recovery for automatic device selection.
    pub recovery: bool,
    /// Explicit resource directory; no implicit global configuration read/write.
    pub resources: PathBuf,
    /// Voice library root override; defaults to the app root (`~/.talechime/voices`).
    pub voices: Option<PathBuf>,
}
impl ModelOptions {
    /// Select a backend and caller-owned resources, initially on CPU.
    pub fn new(backend: impl Into<String>, resources: impl Into<PathBuf>) -> Self {
        Self {
            backend: backend.into(),
            model: None,
            device: Device::Cpu,
            calibrate: false,
            recovery: false,
            resources: resources.into(),
            voices: None,
        }
    }
}

/// Public failures do not expose native model or audio-library types.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EngineError {
    /// A prepared engine has one owned execution at a time.
    #[error("engine is busy; close its current execution first")]
    Busy,
    /// Engine or execution controls have been explicitly closed.
    #[error("engine or execution is closed")]
    Closed,
    /// Invalid local assembly/collection options.
    #[error("invalid engine options: {0}")]
    InvalidOptions(&'static str),
    /// Preparation failed; the underlying source retains diagnostics.
    #[error("model preparation failed: {0}")]
    Prepare(#[source] anyhow::Error),
    /// A playback device could not be opened.
    #[error("audio output failed: {0}")]
    Output(#[source] BackendError),
    /// Shared plan, synthesis or listening validation/execution failed.
    #[error(transparent)]
    Session(#[from] SessionError),
    /// A local owner task failed unexpectedly.
    #[error("local execution task failed: {0}")]
    Task(#[from] tokio::task::JoinError),
    /// A bounded collector refuses to accumulate more PCM.
    #[error("PCM collection exceeds {0} bytes")]
    CollectionLimit(usize),
    /// Different segment formats cannot be collected into one interleaved buffer.
    #[error("PCM format changed during collection")]
    CollectionFormat,
    /// Voice library operations failed; the message is safe to display.
    #[error("{0}")]
    Voice(String),
    /// Empty/layout-only text produced no PCM to collect.
    #[error("no spoken audio to collect")]
    EmptyAudio,
}

/// Establish a LocalSet in the host's existing Tokio runtime; creates no thread.
/// Advanced hosts can use their existing LocalSet instead.
pub async fn run_local<F: Future>(work: F) -> F::Output {
    tokio::task::LocalSet::new().run_until(work).await
}

pub(crate) enum Active {
    Synthesis {
        owner: Weak<()>,
        cancellation: CancellationHandle,
    },
    Listening(Weak<()>),
}
impl Active {
    fn busy(&self) -> bool {
        match self {
            Self::Synthesis {
                owner,
                cancellation,
            } => owner.upgrade().is_some() || !cancellation.is_finished(),
            Self::Listening(owner) => owner.upgrade().is_some(),
        }
    }
}

/// One prepared model/backend owned by a host-selected local execution thread.
/// No player or checkpoint exists until listening is explicitly attached.
pub struct Engine {
    backend: Option<Rc<dyn Backend>>,
    voice_studio: Option<crate::VoiceStudio>,
    pub(crate) verifier: Option<Rc<crate::Verifier>>,
    active: RefCell<Option<Active>>,
}
impl Engine {
    /// Build open input using this prepared model's exact identity and capabilities.
    /// Append ranges and seal later, or submit it directly for incremental execution.
    pub async fn plan(
        &self,
        source: SourceSnapshot,
        voices: Vec<String>,
        policy: PlaybackPolicy,
    ) -> Result<SpeechPlan, EngineError> {
        Ok(SpeechPlan::new(
            source,
            self.voice_snapshot(voices).await?,
            policy,
        ))
    }
    /// Build a sealed full-text plan using one validated voice and optional style.
    /// Empty text still validates the selected voice/style.
    pub async fn single_voice_plan(
        &self,
        source: SourceSnapshot,
        voice: &str,
        style: Option<String>,
        policy: PlaybackPolicy,
    ) -> Result<SpeechPlan, EngineError> {
        SpeechPlan::single_voice(
            source,
            self.voice_snapshot(vec![voice.to_owned()]).await?,
            policy,
            voice,
            style,
        )
        .map_err(|error| EngineError::Session(SessionError::Plan(error)))
    }
    async fn voice_snapshot(&self, voices: Vec<String>) -> Result<VoiceSnapshot, EngineError> {
        let caps = self.capabilities()?;
        match &self.voice_studio {
            Some(studio) => studio
                .voice_snapshot(&caps, &voices)
                .await
                .map_err(|error| EngineError::Voice(error.to_string())),
            None => VoiceSnapshot::new(&caps.backend, caps.model.as_deref(), &caps, voices)
                .map_err(|error| EngineError::Session(SessionError::Plan(error))),
        }
    }
    /// Explicitly prepare (and if necessary download) the requested model.
    /// Progress is drained concurrently, so no forwarding task needs shutdown.
    /// Requires the host's current LocalSet. No CLI config is read or written.
    pub async fn prepare(
        options: ModelOptions,
        progress: impl FnMut(Event),
    ) -> Result<Self, EngineError> {
        if options.backend.trim().is_empty() || options.resources.as_os_str().is_empty() {
            return Err(EngineError::InvalidOptions(
                "backend and resource path are required",
            ));
        }
        let registry = talechime_backends::Registry::new(Some(options.resources.clone()))
            .map_err(EngineError::Prepare)?;
        let model = registry
            .resolve(&options.backend, options.model.as_deref())
            .map_err(EngineError::Prepare)?;
        crate::model_preparation::with_progress(
            |tx| {
                model.prepare(
                    talechime_backends::PrepareOptions {
                        device: options.device,
                        calibrate: options.calibrate,
                        recovery: options.recovery,
                        voice: None,
                    },
                    tx,
                )
            },
            progress,
        )
        .await
        .and_then(|backend| Self::with_studio(backend, options))
    }
    fn with_studio(backend: Rc<dyn Backend>, options: ModelOptions) -> Result<Self, EngineError> {
        let studio = crate::VoiceStudio::open(Some(options.resources.clone()), options.voices)
            .map_err(|error| EngineError::Voice(error.to_string()))?;
        let mut engine = Self::from_backend(backend);
        engine.voice_studio = Some(studio);
        Ok(engine)
    }
    /// Inject a prepared custom/test backend. Keep one Engine per backend owner.
    pub fn from_backend(backend: Rc<dyn Backend>) -> Self {
        Self {
            backend: Some(backend),
            voice_studio: None,
            verifier: None,
            active: RefCell::new(None),
        }
    }
    /// Voice assets bound to this engine's resources; `None` for injected backends.
    pub fn voices(&self) -> Option<&crate::VoiceStudio> {
        self.voice_studio.as_ref()
    }
    #[cfg(all(test, feature = "qwen"))]
    pub(crate) fn attach_voice_studio(&mut self, studio: crate::VoiceStudio) {
        self.voice_studio = Some(studio);
    }
    async fn resolve_voice(&self, voice: &str) -> Result<String, EngineError> {
        let caps = self.capabilities()?;
        match &self.voice_studio {
            Some(studio) => studio
                .resolve_voice(&caps, voice)
                .await
                .map_err(|error| EngineError::Voice(error.to_string())),
            None => Ok(voice.to_owned()),
        }
    }
    /// Attach reusable prepared ASR resources while idle. Verification remains opt-in per execution.
    pub fn set_verifier(&mut self, verifier: Rc<crate::Verifier>) -> Result<(), EngineError> {
        self.available()?;
        self.verifier = Some(verifier);
        Ok(())
    }
    /// Actual prepared capability identity; constructing/querying never starts audio.
    pub fn capabilities(&self) -> Result<Capabilities, EngineError> {
        Ok(self.backend()?.capabilities())
    }
    pub(crate) fn backend(&self) -> Result<Rc<dyn Backend>, EngineError> {
        self.backend.clone().ok_or(EngineError::Closed)
    }
    pub(crate) fn available(&self) -> Result<(), EngineError> {
        self.backend.as_ref().ok_or(EngineError::Closed)?;
        if self.active.borrow().as_ref().is_some_and(Active::busy) {
            return Err(EngineError::Busy);
        }
        Ok(())
    }
    pub(crate) fn reserve_listening(&self, owner: &Rc<()>) {
        self.active
            .replace(Some(Active::Listening(Rc::downgrade(owner))));
    }
    /// Text + voice + optional style -> bounded PCM, without playback.
    /// Library voices may create or refresh their model reference cache.
    /// Consume blocks on this LocalSet, retaining them only within the 30s/16MiB budget.
    pub async fn synthesize(
        &self,
        text: impl Into<Arc<str>>,
        voice: &str,
        style: Option<String>,
    ) -> Result<PcmStream, EngineError> {
        self.synthesize_verified(text, voice, style, crate::VerificationOptions::default())
            .await
    }
    /// Enable report-only or gate policy at actual backend segment boundaries.
    pub async fn synthesize_verified(
        &self,
        text: impl Into<Arc<str>>,
        voice: &str,
        style: Option<String>,
        verification: crate::VerificationOptions,
    ) -> Result<PcmStream, EngineError> {
        self.synthesize_with_options(
            text,
            voice,
            style,
            crate::SynthesisOptions {
                verification,
                ..Default::default()
            },
        )
        .await
    }
    /// Explicit settings for one isolated execution; history never crosses calls.
    pub async fn synthesize_with_options(
        &self,
        text: impl Into<Arc<str>>,
        voice: &str,
        style: Option<String>,
        options: crate::SynthesisOptions,
    ) -> Result<PcmStream, EngineError> {
        self.available()?;
        let voice = self.resolve_voice(voice).await?;
        self.available()?;
        let stream = SynthesisStream::start_with_options(
            self.backend()?,
            text,
            &voice,
            style,
            self.verifier.clone(),
            options,
        )?;
        let owner = Rc::new(());
        self.active.replace(Some(Active::Synthesis {
            owner: Rc::downgrade(&owner),
            cancellation: stream.cancellation(),
        }));
        Ok(PcmStream {
            stream,
            owner: Some(owner),
        })
    }
    /// Collect only up to a caller-chosen nonzero PCM byte limit.
    pub async fn synthesize_pcm(
        &self,
        text: impl Into<Arc<str>>,
        voice: &str,
        style: Option<String>,
        max_bytes: usize,
    ) -> Result<Pcm, EngineError> {
        self.synthesize_pcm_with_options(text, voice, style, max_bytes, Default::default())
            .await
    }
    /// Collect an isolated execution with explicit verification.
    pub async fn synthesize_pcm_with_options(
        &self,
        text: impl Into<Arc<str>>,
        voice: &str,
        style: Option<String>,
        max_bytes: usize,
        options: crate::SynthesisOptions,
    ) -> Result<Pcm, EngineError> {
        if max_bytes == 0 {
            return Err(EngineError::InvalidOptions(
                "collection limit must be nonzero",
            ));
        }
        self.synthesize_with_options(text, voice, style, options)
            .await?
            .collect(max_bytes)
            .await
    }
    /// Release the prepared owner after executions are explicitly closed.
    /// Dropped synthesis streams are awaited here; live executions return Busy.
    /// Injected backends with external Rc owners remain owned by those callers.
    pub async fn close(&mut self) -> Result<(), EngineError> {
        let pending = match self.active.borrow().as_ref() {
            Some(Active::Synthesis {
                owner,
                cancellation,
            }) if owner.upgrade().is_none() => Some(cancellation.clone()),
            Some(active) if active.busy() => return Err(EngineError::Busy),
            _ => None,
        };
        if let Some(pending) = pending {
            pending.closed().await;
        }
        self.active.take();
        self.backend.take();
        self.verifier.take();
        Ok(())
    }
}

/// Owns a bounded synthesis execution. PCM packets retain their permits until drop.
pub struct PcmStream {
    stream: SynthesisStream,
    owner: Option<Rc<()>>,
}
impl PcmStream {
    /// Most recent readback report, retained independently of PCM packet ownership.
    pub fn last_report(&self) -> Option<&crate::VerificationReport> {
        self.stream.last_report()
    }
    /// Request cancellation from another thread without moving the stream.
    pub fn cancellation(&self) -> CancellationHandle {
        self.stream.cancellation()
    }
    /// Generation state, independent of any playback.
    pub fn state(&self) -> SynthesisState {
        self.stream.state()
    }
    /// Receive blocks or one explicit error. None alone does not imply success:
    /// cancellation also ends delivery; inspect state() for Completed.
    pub async fn recv(&mut self) -> Option<Result<SpeechAudio, EngineError>> {
        let item = self
            .stream
            .recv()
            .await
            .map(|item| item.map_err(EngineError::Session));
        if self.stream.state() != SynthesisState::Running {
            self.owner.take();
        }
        item
    }
    /// Receive ordered reports and PCM; use this when storing all attempt evidence.
    pub async fn next(&mut self) -> Option<Result<crate::SynthesisItem, EngineError>> {
        let item = self
            .stream
            .next()
            .await
            .map(|item| item.map_err(EngineError::Session));
        if self.stream.state() != SynthesisState::Running {
            self.owner.take();
        }
        item
    }
    /// Cancel, release queued packets and await local producer teardown.
    pub async fn cancel(&mut self) {
        self.stream.cancel().await;
        self.owner.take();
    }
    /// Consume this execution into a single PCM buffer within a mandatory byte bound.
    /// On limit, format or synthesis failure, discard accumulated PCM and await cancellation.
    pub async fn collect(mut self, max_bytes: usize) -> Result<Pcm, EngineError> {
        let result = self.collect_inner(max_bytes).await;
        if result.is_err() {
            self.cancel().await;
        }
        result
    }
    async fn collect_inner(&mut self, max_bytes: usize) -> Result<Pcm, EngineError> {
        if max_bytes == 0 {
            return Err(EngineError::InvalidOptions(
                "collection limit must be nonzero",
            ));
        }
        let mut audio: Option<Pcm> = None;
        while let Some(block) = self.recv().await {
            let block = block?;
            let pcm = block.pcm();
            let old_samples = audio.as_ref().map_or(0, |audio| audio.samples.len());
            let bytes = old_samples
                .checked_add(pcm.samples.len())
                .and_then(|samples| samples.checked_mul(size_of::<f32>()))
                .ok_or(EngineError::CollectionLimit(max_bytes))?;
            if bytes > max_bytes {
                return Err(EngineError::CollectionLimit(max_bytes));
            }
            let audio = audio.get_or_insert_with(|| Pcm {
                samples: Vec::new(),
                sample_rate: pcm.sample_rate,
                channels: pcm.channels,
            });
            if (audio.sample_rate, audio.channels) != (pcm.sample_rate, pcm.channels) {
                return Err(EngineError::CollectionFormat);
            }
            audio.samples.extend_from_slice(&pcm.samples);
        }
        if self.state() == SynthesisState::Cancelled {
            return Err(EngineError::Closed);
        }
        audio.ok_or(EngineError::EmptyAudio)
    }
}

#[cfg(all(test, feature = "qwen"))]
mod tests {
    use super::*;
    use crate::VoiceStudio;
    use std::cell::RefCell;
    use talechime_core::backend::{AudioChunk, SegmentRequest};
    use talechime_core::voices::library::{VoiceLibrary, VoiceSource};

    /// Minimal 16-bit mono RIFF header plus samples, for store validation.
    fn riff_wav(samples: i16) -> Vec<u8> {
        let pair = samples.to_le_bytes();
        let data: Vec<u8> = pair.iter().copied().cycle().take(48_000).collect();
        let mut wav = Vec::with_capacity(44 + data.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&24_000u32.to_le_bytes());
        wav.extend_from_slice(&48_000u32.to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        wav
    }

    struct RecordingBackend {
        resources: PathBuf,
        voices: RefCell<Vec<String>>,
    }
    impl Backend for RecordingBackend {
        fn capabilities(&self) -> Capabilities {
            // Mirror real adapters: reference stores are listed dynamically.
            let mut voices = vec!["builtin".to_owned()];
            if let Ok(store) = talechime_backends::qwen::voice_store(
                &talechime_backends::qwen::models::Model::Base17.directory(&self.resources),
                talechime_backends::qwen::models::Model::Base17,
            ) && let Ok(listed) = store.list()
            {
                voices.extend(listed.into_iter().map(|voice| voice.id));
            }
            Capabilities {
                backend: "qwen".into(),
                model: Some("1.7b-base".into()),
                model_name: "stub".into(),
                default_voice: "builtin".into(),
                voices,
                voice_names: Default::default(),
                native_streaming: true,
                style: false,
                cloning: true,
                pronunciation: false,

                compiled_devices: vec![Device::Cpu],
                parameters: vec![],
            }
        }
        fn stream<'a>(
            &'a self,
            request: SegmentRequest<'a>,
        ) -> talechime_core::backend::Streaming<'a> {
            self.voices.borrow_mut().push(request.voice.to_owned());
            Box::pin(async move {
                let (tx, rx) = tokio::sync::mpsc::channel(2);
                tokio::task::spawn_local(async move {
                    let _ = tx
                        .send(Ok(AudioChunk::Pcm(Pcm {
                            samples: vec![0.1, 0.2],
                            sample_rate: 24_000,
                            channels: 1,
                        })))
                        .await;
                    let _ = tx.send(Ok(AudioChunk::End)).await;
                });
                Ok(rx)
            })
        }
    }

    #[tokio::test]
    async fn library_voices_materialize_and_builtin_ids_pass_through() -> anyhow::Result<()> {
        run_local(async {
            let roots = tempfile::tempdir()?;
            let resources = roots.path().join("res");
            let library_root = roots.path().join("lib");
            let library = VoiceLibrary::open(&library_root)?;
            library.create(
                "hero",
                "主角",
                &riff_wav(6000),
                "试音转写",
                Some("低沉"),
                VoiceSource::Imported,
            )?;
            let backend = Rc::new(RecordingBackend {
                resources: resources.clone(),
                voices: RefCell::new(Vec::new()),
            });
            let mut engine = Engine::from_backend(backend.clone());
            engine.attach_voice_studio(VoiceStudio::at(&resources, &library_root)?);
            // A snapshot captured before installation must still accept the installed voice.
            let stale_caps = engine.capabilities()?;
            let snapshot = engine
                .voices()
                .unwrap()
                .voice_snapshot(&stale_caps, &["hero".into()])
                .await?;
            let source = SourceSnapshot::new(
                talechime_protocol::SourceId {
                    namespace: "test".into(),
                    book: "book".into(),
                    chapter: "one".into(),
                },
                "你好",
                &talechime_protocol::text_hash("你好"),
            )?;
            let mut plan = engine
                .plan(
                    source.clone(),
                    vec!["hero".into()],
                    PlaybackPolicy::Streaming,
                )
                .await?;
            plan.append(vec![crate::SpeechSpan::new(
                talechime_protocol::TextRange { start: 0, end: 3 },
                "hero",
                None,
            )])?;
            plan.append(vec![crate::SpeechSpan::new(
                talechime_protocol::TextRange { start: 3, end: 6 },
                "hero",
                None,
            )])?;
            assert!(
                plan.spans()
                    .iter()
                    .all(|span| span.voice() == "custom:hero")
            );
            assert!(
                SpeechPlan::single_voice(source, snapshot, PlaybackPolicy::Streaming, "hero", None)
                    .is_ok()
            );
            // Library id resolves and materializes into the qwen base store.
            let pcm = engine
                .synthesize_pcm("你好。", "hero", None, 1 << 20)
                .await?;
            assert_eq!(pcm.samples.len(), 2);
            assert_eq!(
                backend.voices.borrow().last().map(String::as_str),
                Some("custom:hero")
            );
            // Built-in ids pass through unchanged.
            engine
                .synthesize_pcm("你好。", "builtin", None, 1 << 20)
                .await?;
            assert_eq!(
                backend.voices.borrow().last().map(String::as_str),
                Some("builtin")
            );
            // The materialized cache exists beside its fingerprint marker.
            assert!(resources.join("qwen").exists());
            Ok::<_, anyhow::Error>(())
        })
        .await?;
        Ok(())
    }
}
