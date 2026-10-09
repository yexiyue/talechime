use crate::*;
use std::{
    cell::RefCell,
    future::Future,
    path::PathBuf,
    rc::{Rc, Weak},
    sync::Arc,
};
use tokio::sync::mpsc;
use tts_core::SynthesisStream;

/// Model preparation is explicit and independent of playback/config persistence.
#[derive(Debug, Clone)]
pub struct ModelOptions {
    /// Compiled adapter ID (e.g. moss, qwen, voxcpm, omnivoice).
    pub backend: String,
    /// Optional model selection; resolved to the prepared capability identity.
    pub model: Option<String>,
    /// Concrete inference device. Auto requires host-side selection first.
    pub device: Device,
    /// Explicit resource directory; no implicit global configuration read/write.
    pub resources: PathBuf,
}
impl ModelOptions {
    /// Select a backend and caller-owned resources, initially on CPU.
    pub fn new(backend: impl Into<String>, resources: impl Into<PathBuf>) -> Self {
        Self {
            backend: backend.into(),
            model: None,
            device: Device::Cpu,
            resources: resources.into(),
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
    pub(crate) verifier: Option<Rc<crate::Verifier>>,
    active: RefCell<Option<Active>>,
}
impl Engine {
    /// Build open input using this prepared model's exact identity and capabilities.
    /// Append ranges and seal later, or submit it directly for incremental execution.
    pub fn plan(
        &self,
        source: SourceSnapshot,
        voices: Vec<String>,
        policy: PlaybackPolicy,
    ) -> Result<SpeechPlan, EngineError> {
        Ok(SpeechPlan::new(
            source,
            self.voice_snapshot(voices)?,
            policy,
        ))
    }
    /// Build a sealed full-text plan using one validated voice and optional style.
    /// Empty text still validates the selected voice/style.
    pub fn single_voice_plan(
        &self,
        source: SourceSnapshot,
        voice: &str,
        style: Option<String>,
        policy: PlaybackPolicy,
    ) -> Result<SpeechPlan, EngineError> {
        SpeechPlan::single_voice(
            source,
            self.voice_snapshot(vec![voice.into()])?,
            policy,
            voice,
            style,
        )
        .map_err(|error| EngineError::Session(SessionError::Plan(error)))
    }
    fn voice_snapshot(&self, voices: Vec<String>) -> Result<VoiceSnapshot, EngineError> {
        let caps = self.capabilities()?;
        VoiceSnapshot::new(&caps.backend, caps.model.as_deref(), &caps, voices)
            .map_err(|error| EngineError::Session(SessionError::Plan(error)))
    }
    /// Explicitly prepare (and if necessary download) the requested model.
    /// Progress is drained concurrently, so no forwarding task needs shutdown.
    /// Requires the host's current LocalSet. No CLI config is read or written.
    pub async fn prepare(
        options: ModelOptions,
        mut progress: impl FnMut(Event),
    ) -> Result<Self, EngineError> {
        if options.backend.trim().is_empty()
            || options.resources.as_os_str().is_empty()
            || options.device == Device::Auto
        {
            return Err(EngineError::InvalidOptions(
                "backend, resource path and concrete device are required",
            ));
        }
        let registry =
            tts_backends::Registry::new(Some(options.resources)).map_err(EngineError::Prepare)?;
        let (tx, mut rx) = mpsc::channel(16);
        let prepare = registry.prepare_model_on(
            &options.backend,
            options.model.as_deref(),
            tx,
            options.device,
        );
        tokio::pin!(prepare);
        let mut progress_open = true;
        loop {
            tokio::select! {
                result = &mut prepare => {
                    while let Ok(event) = rx.try_recv() { progress(event); }
                    return result.map(Self::from_backend).map_err(EngineError::Prepare);
                }
                event = rx.recv(), if progress_open => {
                    if let Some(event) = event { progress(event); } else { progress_open = false; }
                }
            }
        }
    }
    /// Inject a prepared custom/test backend. Keep one Engine per backend owner.
    pub fn from_backend(backend: Rc<dyn Backend>) -> Self {
        Self {
            backend: Some(backend),
            verifier: None,
            active: RefCell::new(None),
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
    /// Text + voice + optional style -> bounded PCM, without playback or disk writes.
    /// Consume blocks on this LocalSet, retaining them only within the 30s/16MiB budget.
    pub fn synthesize(
        &self,
        text: impl Into<Arc<str>>,
        voice: &str,
        style: Option<String>,
    ) -> Result<PcmStream, EngineError> {
        self.synthesize_verified(text, voice, style, crate::VerificationOptions::default())
    }
    /// Enable report-only or gate policy at actual backend segment boundaries.
    pub fn synthesize_verified(
        &self,
        text: impl Into<Arc<str>>,
        voice: &str,
        style: Option<String>,
        verification: crate::VerificationOptions,
    ) -> Result<PcmStream, EngineError> {
        self.available()?;
        let stream = SynthesisStream::start_verified(
            self.backend()?,
            text,
            voice,
            style,
            self.verifier.clone(),
            verification,
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
        if max_bytes == 0 {
            return Err(EngineError::InvalidOptions(
                "collection limit must be nonzero",
            ));
        }
        self.synthesize(text, voice, style)?
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
        if let Some(verifier) = self.verifier.take() {
            verifier.settled().await;
        }
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
