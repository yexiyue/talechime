//! Thread-local session state machine with bounded prefetch and durable text positions.
use crate::{
    PlanError, PlanState, Playback, PlaybackPolicy, SourceSnapshot, SpeechPlan, SpeechSpan,
    VoiceSnapshot,
    backend::{Backend, BackendError, Pcm},
    checkpoint::{CheckpointError, CheckpointStore},
};
use std::{
    cell::Cell,
    collections::HashSet,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::{
    sync::{OwnedSemaphorePermit, Semaphore, mpsc},
    task::JoinHandle,
};
use tts_protocol::{Config, EndReason, ErrorInfo, Event, SessionState, TextRange};

/// Advanced core single-voice input, independent of the JSON Lines transport.
/// Public hosts should usually construct an Engine plan instead.
#[derive(Debug, Clone)]
pub struct StartRequest {
    /// Host-owned immutable source identity.
    pub source: tts_protocol::SourceId,
    /// Exact full source without speech normalization.
    pub text: String,
    /// SHA-256 checked by the plan constructor.
    pub text_hash: String,
    /// Explicit source byte to resume at.
    pub resume_byte: Option<usize>,
    /// Use stored actual-playback progress when no explicit byte is provided.
    pub restore_checkpoint: bool,
}

/// Per-session event before the transport adds instance IDs and sequence numbers.
#[derive(Debug)]
pub struct SessionEvent {
    pub session_id: String,
    pub event: Event,
}

/// Errors preserve the failing stage and never implicitly skip text.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    #[error(transparent)]
    Verification(#[from] crate::verification::VerificationError),
    #[error(transparent)]
    Staging(#[from] StagingError),
    #[error("invalid session: {0}")]
    Invalid(String),
    #[error("event consumer disconnected")]
    Disconnected,
}

impl From<std::io::Error> for SessionError {
    fn from(error: std::io::Error) -> Self {
        StagingError::Io(error).into()
    }
}

mod buffering;
mod plan;
mod prefetch;
mod producer;
mod staging;
mod synthesis;
use plan::PlanInput;
pub use plan::{PlanProgress, PlanSessionOptions};
use prefetch::{Budget, Packet};
pub use staging::{StagingError, StagingOptions};
pub use synthesis::{
    CancellationHandle, SpeechAudio, SynthesisItem, SynthesisOptions, SynthesisState,
    SynthesisStream,
};

struct AbortOnDrop(JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Default)]
struct PendingWrites {
    count: AtomicUsize,
    idle: tokio::sync::Notify,
}
struct WriteGuard(Arc<PendingWrites>);
impl Drop for WriteGuard {
    fn drop(&mut self) {
        if self.0.count.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.0.idle.notify_one();
        }
    }
}

struct Job {
    id: String,
    source: SourceSnapshot,
    paused: Rc<Cell<bool>>,
    phase: Rc<Cell<SessionState>>,
    buffering: Rc<Cell<bool>>,
    speed: Rc<Cell<f32>>,
    volume: Cell<f32>,
    position: Rc<Cell<usize>>,
    terminal: Rc<Cell<bool>>,
    writes: Arc<PendingWrites>,
    input: Rc<PlanInput>,
    legacy_settings: Option<(String, Option<String>)>,
    staging: StagingOptions,
    verification: crate::verification::VerificationOptions,
    task: Option<AbortOnDrop>,
}

/// Runs inside a Tokio LocalSet; audio/model objects never move between threads.
pub struct SessionManager {
    verifier: Option<Rc<crate::verification::Verifier>>,
    backend: Rc<dyn Backend>,
    player: Rc<dyn Playback>,
    checkpoints: Option<CheckpointStore>,
    events: mpsc::Sender<SessionEvent>,
    job: Option<Job>,
    used_ids: HashSet<String>,
}

impl SessionManager {
    /// Inject real or test implementations with an explicit event consumer.
    pub fn new(
        backend: Rc<dyn Backend>,
        player: Rc<dyn Playback>,
        checkpoints: CheckpointStore,
        events: mpsc::Sender<SessionEvent>,
    ) -> Self {
        Self::with_optional_checkpoints(backend, player, Some(checkpoints), events)
    }

    /// Assemble local playback with explicit optional checkpoint persistence.
    pub fn with_optional_checkpoints(
        backend: Rc<dyn Backend>,
        player: Rc<dyn Playback>,
        checkpoints: Option<CheckpointStore>,
        events: mpsc::Sender<SessionEvent>,
    ) -> Self {
        Self {
            verifier: None,
            backend,
            player,
            checkpoints,
            events,
            job: None,
            used_ids: HashSet::new(),
        }
    }

    /// Attach prepared ASR resources; configuration cannot change mid-execution.
    pub fn set_verifier(
        &mut self,
        verifier: Rc<crate::verification::Verifier>,
    ) -> Result<(), SessionError> {
        if self.job.as_ref().is_some_and(|job| !job.terminal.get()) {
            return Err(SessionError::Invalid(
                "cannot change verifier during execution".into(),
            ));
        }
        self.verifier = Some(verifier);
        Ok(())
    }

    /// Start explicitly. Restoring a checkpoint never starts this method on its own.
    pub async fn start(
        &mut self,
        id: String,
        request: StartRequest,
        config: &Config,
    ) -> Result<(), SessionError> {
        let capabilities = self.backend.capabilities();
        crate::config::validate(config, &capabilities)
            .map_err(|error| SessionError::Invalid(error.to_string()))?;
        let source = SourceSnapshot::new(request.source, request.text, &request.text_hash)?;
        let voices = VoiceSnapshot::new(
            &capabilities.backend,
            capabilities.model.as_deref(),
            &capabilities,
            vec![config.voice.clone()],
        )?;
        let plan = SpeechPlan::single_voice(
            source,
            voices,
            PlaybackPolicy::Streaming,
            &config.voice,
            config.style.clone(),
        )?;
        self.start_input(
            id,
            plan,
            PlanSessionOptions {
                volume: config.volume,
                speed: config.speed,
                resume_byte: request.resume_byte,
                restore_checkpoint: request.restore_checkpoint,
                continuation: true,
                params: Default::default(),
                seed: Default::default(),
                staging: StagingOptions::default(),
                verification: crate::verification::VerificationOptions::default(),
            },
            Some((config.voice.clone(), config.style.clone())),
        )
        .await
    }

    /// Start a plan on the host's LocalSet, replacing the previous session.
    pub async fn start_plan(
        &mut self,
        id: String,
        plan: SpeechPlan,
        options: PlanSessionOptions,
    ) -> Result<(), SessionError> {
        self.start_input(id, plan, options, None).await
    }

    async fn start_input(
        &mut self,
        id: String,
        plan: SpeechPlan,
        options: PlanSessionOptions,
        legacy_settings: Option<(String, Option<String>)>,
    ) -> Result<(), SessionError> {
        crate::verification::validate_options(&options.verification)?;
        if options.verification.policy != crate::verification::VerificationPolicy::Off
            && self.verifier.is_none()
        {
            return Err(crate::verification::VerificationError::NotPrepared.into());
        }
        if plan.playback_policy() == PlaybackPolicy::AfterChapterReady {
            options.staging.validate()?;
        }
        if plan.state() == PlanState::Failed {
            return Err(PlanError::NotOpen(PlanState::Failed).into());
        }
        plan.validate_capabilities(&self.backend.capabilities())?;
        options
            .params
            .validate(&self.backend.capabilities().parameters)
            .map_err(SessionError::Invalid)?;
        crate::config::validate_playback(options.volume, options.speed)
            .map_err(|error| SessionError::Invalid(error.to_string()))?;
        let source = plan.source().clone();
        if id.is_empty() || self.used_ids.contains(&id) || self.used_ids.len() >= 65_536 {
            return Err(SessionError::Invalid(
                "missing, reused or exhausted session ID".into(),
            ));
        }
        if options.restore_checkpoint && self.checkpoints.is_none() {
            return Err(SessionError::Invalid(
                "checkpoint restoration requires a store".into(),
            ));
        }
        let mut byte = options.resume_byte.unwrap_or(0);
        if options.restore_checkpoint
            && options.resume_byte.is_none()
            && let Some(store) = &self.checkpoints
            && let Some(checkpoint) = store.load(source.source(), source.text())?
        {
            byte = if checkpoint.completed {
                0
            } else {
                checkpoint.resume_byte
            };
        }
        plan.validate_resume_byte(byte)?;
        self.stop().await?;
        self.player.configure(options.volume, options.speed);
        self.player.pause();
        self.used_ids.insert(id.clone());
        let phase = Rc::new(Cell::new(SessionState::Generating));
        let position = Rc::new(Cell::new(byte));
        let paused = Rc::new(Cell::new(false));
        let terminal = Rc::new(Cell::new(false));
        let buffering = Rc::new(Cell::new(true));
        let speed = Rc::new(Cell::new(options.speed));
        let mut input = PlanInput::new(plan, byte);
        input.verifier =
            if options.verification.policy == crate::verification::VerificationPolicy::Off {
                None
            } else {
                self.verifier.as_ref().map(|verifier| verifier.scoped())
            };
        input.verification = options.verification.clone();
        input.continuation = options.continuation;
        input.params = options.params.clone();
        input.seed = options.seed;
        let input = Rc::new(input);
        let writes = Arc::new(PendingWrites::default());
        let runner = Runner {
            id: id.clone(),
            source: source.clone(),
            backend: self.backend.clone(),
            player: self.player.clone(),
            checkpoints: self.checkpoints.clone(),
            events: self.events.clone(),
            paused: paused.clone(),
            phase: phase.clone(),
            buffering: buffering.clone(),
            speed: speed.clone(),
            position: position.clone(),
            text: source.shared_text(),
            writes: writes.clone(),
            staging: options.staging.clone(),
        };
        let finished = terminal.clone();
        let final_phase = phase.clone();
        let running_input = input.clone();
        let task = tokio::task::spawn_local(async move {
            let result = runner.run(byte, running_input).await;
            if let Err(error) = &result {
                runner.player.stop();
                let staging = matches!(error, SessionError::Staging(_));
                let verification = matches!(error, SessionError::Verification(_));
                let _ = runner
                    .emit(Event::Error(ErrorInfo {
                        code: if verification {
                            "verification_failed"
                        } else if staging {
                            "staging_failed"
                        } else {
                            "session_failed"
                        }
                        .into(),
                        stage: if verification {
                            "readback"
                        } else if staging {
                            "chapter_staging"
                        } else {
                            "listening"
                        }
                        .into(),
                        message: error.to_string(),
                        retryable: true,
                    }))
                    .await;
            }
            finished.set(true);
            final_phase.set(if result.is_ok() {
                SessionState::Stopped
            } else {
                SessionState::Failed
            });
            let reason = if result.is_ok() {
                EndReason::Completed
            } else {
                EndReason::Failed
            };
            let _ = runner
                .emit(Event::SessionEnded {
                    reason,
                    text_hash: runner.source.hash().into(),
                })
                .await;
        });
        self.job = Some(Job {
            id,
            source,
            paused,
            phase,
            buffering,
            speed,
            volume: Cell::new(options.volume),
            position,
            terminal,
            writes,
            input,
            legacy_settings,
            staging: options.staging,
            verification: options.verification,
            task: Some(AbortOnDrop(task)),
        });
        Ok(())
    }

    /// Atomically accept another contiguous batch. Stale/terminal sessions fail.
    pub fn append_plan(&self, id: &str, batch: Vec<SpeechSpan>) -> Result<(), SessionError> {
        let job = self.active(id)?;
        job.input.plan.borrow_mut().append(batch)?;
        job.input.changed.notify_one();
        Ok(())
    }

    /// Seal fully covered input; this does not imply generation or playback completion.
    pub fn seal_plan(&self, id: &str) -> Result<(), SessionError> {
        let job = self.active(id)?;
        job.input.plan.borrow_mut().seal()?;
        job.input.changed.notify_one();
        Ok(())
    }

    /// Fail open input and cancel in-flight generation/playback, preserving safe progress.
    pub async fn fail_input(&mut self, id: &str, message: &str) -> Result<(), SessionError> {
        self.active(id)?.input.plan.borrow_mut().fail()?;
        self.finish(EndReason::Failed, Some(message), true).await
    }

    /// Query progress for this session, including after its terminal event.
    pub fn plan_progress(&self, id: &str) -> Result<PlanProgress, SessionError> {
        let job = self
            .job
            .as_ref()
            .filter(|job| job.id == id)
            .ok_or_else(|| SessionError::Invalid("stale session".into()))?;
        let plan = job.input.plan.borrow();
        Ok(PlanProgress {
            input_state: plan.state(),
            accepted_end: plan.accepted_end(),
            generated_end: job.input.generated.get(),
            played_end: job.position.get(),
            waiting_for_input: !job.terminal.get() && job.input.waiting.get(),
            chapter_ready: job.input.ready.get(),
        })
    }

    /// Change only playback settings without touching the fixed voice assignments.
    pub fn configure_playback(
        &self,
        id: &str,
        volume: f32,
        speed: f32,
    ) -> Result<(), SessionError> {
        crate::config::validate_playback(volume, speed)
            .map_err(|error| SessionError::Invalid(error.to_string()))?;
        let job = self.active(id)?;
        job.speed.set(speed);
        job.volume.set(volume);
        self.player.configure(volume, speed);
        Ok(())
    }

    /// Pause actual playback while allowing only bounded prefetch.
    pub async fn pause(&self, id: &str) -> Result<(), SessionError> {
        let job = self.active(id)?;
        job.paused.set(true);
        self.player.pause();
        self.emit(
            id,
            Event::SessionState {
                state: SessionState::Paused,
            },
        )
        .await
    }

    /// Continue the same playback position.
    pub async fn resume(&self, id: &str) -> Result<(), SessionError> {
        let job = self.active(id)?;
        job.paused.set(false);
        if !job.buffering.get() {
            self.player.resume();
        }
        self.emit(
            id,
            Event::SessionState {
                state: job.phase.get(),
            },
        )
        .await
    }

    /// Current playback state is owned by the session, not the transport.
    pub fn status(&self) -> (Option<String>, SessionState) {
        match &self.job {
            Some(job) => (
                Some(job.id.clone()),
                if !job.terminal.get() && job.paused.get() {
                    SessionState::Paused
                } else {
                    job.phase.get()
                },
            ),
            None => (None, SessionState::Idle),
        }
    }

    /// Voice changes replace the session at its unfinished original-text position.
    pub async fn update_settings(
        &mut self,
        new_id: String,
        config: &Config,
    ) -> Result<bool, SessionError> {
        if self
            .job
            .as_ref()
            .is_some_and(|job| !job.terminal.get() && job.legacy_settings.is_none())
        {
            return Err(SessionError::Invalid(
                "planned sessions have fixed voices; use configure_playback or start a new plan"
                    .into(),
            ));
        }
        crate::config::validate(config, &self.backend.capabilities())
            .map_err(|error| SessionError::Invalid(error.to_string()))?;
        if let Some(job) = &self.job
            && !job.terminal.get()
            && job
                .legacy_settings
                .as_ref()
                .is_some_and(|(voice, style)| voice != &config.voice || style != &config.style)
        {
            let paused = job.paused.get();
            let request = StartRequest {
                source: job.source.source().clone(),
                text: job.source.text().into(),
                text_hash: job.source.hash().into(),
                resume_byte: Some(job.position.get()),
                restore_checkpoint: false,
            };
            self.start(new_id.clone(), request, config).await?;
            if paused {
                self.pause(&new_id).await?;
            }
            return Ok(true);
        }
        if let Some(job) = &self.job {
            job.speed.set(config.speed);
            job.volume.set(config.volume);
        }
        self.player.configure(config.volume, config.speed);
        Ok(false)
    }

    fn active(&self, id: &str) -> Result<&Job, SessionError> {
        self.job
            .as_ref()
            .filter(|job| job.id == id && !job.terminal.get())
            .ok_or_else(|| SessionError::Invalid("stale or inactive session".into()))
    }

    /// Cancel generation and clear playback, keeping the model and checkpoint.
    pub async fn stop(&mut self) -> Result<(), SessionError> {
        self.finish(EndReason::Cancelled, None, true).await
    }

    /// Cancel and await owned tasks/I/O without waiting for an event consumer.
    /// Closing discards the execution; it never reports playback completion.
    pub async fn close(&mut self) -> Result<(), SessionError> {
        self.finish(EndReason::Cancelled, None, false).await?;
        self.job.take();
        Ok(())
    }

    async fn finish(
        &mut self,
        reason: EndReason,
        input_error: Option<&str>,
        publish: bool,
    ) -> Result<(), SessionError> {
        self.player.stop();
        if let Some(job) = &mut self.job {
            if let Some(mut task) = job.task.take() {
                task.0.abort();
                let _ = (&mut task.0).await;
            }
            // A blocking atomic write cannot be aborted. Finish it before a new session
            // writes the same source, so a cancelled session cannot replace newer progress.
            if let Some(verifier) = &job.input.verifier {
                verifier.settled().await;
            }
            while job.writes.count.load(Ordering::SeqCst) != 0 {
                job.writes.idle.notified().await;
            }
            if !job.terminal.replace(true) {
                job.phase.set(if reason == EndReason::Failed {
                    SessionState::Failed
                } else {
                    SessionState::Stopped
                });
                job.input.waiting.set(false);
                if !publish {
                    return Ok(());
                }
                if let Some(message) = input_error {
                    self.events
                        .send(SessionEvent {
                            session_id: job.id.clone(),
                            event: Event::Error(ErrorInfo {
                                code: "input_failed".into(),
                                stage: "plan_input".into(),
                                message: message.into(),
                                retryable: true,
                            }),
                        })
                        .await
                        .map_err(|_| SessionError::Disconnected)?;
                }
                self.events
                    .send(SessionEvent {
                        session_id: job.id.clone(),
                        event: Event::SessionEnded {
                            reason,
                            text_hash: job.source.hash().into(),
                        },
                    })
                    .await
                    .map_err(|_| SessionError::Disconnected)?;
            }
        }
        Ok(())
    }

    /// Seek creates a new session and re-synthesizes from a stable original-text byte.
    /// For planned sessions, only config volume/speed are used; voices stay fixed.
    pub async fn seek(
        &mut self,
        old_id: &str,
        new_id: String,
        byte: usize,
        config: &Config,
    ) -> Result<(), SessionError> {
        let job = self.active(old_id)?;
        if job.legacy_settings.is_none() {
            return self
                .restart_plan(old_id, new_id, byte, config.volume, config.speed)
                .await;
        }
        let request = StartRequest {
            source: job.source.source().clone(),
            text: job.source.text().into(),
            text_hash: job.source.hash().into(),
            resume_byte: Some(byte),
            restore_checkpoint: false,
        };
        self.start(new_id, request, config).await
    }

    /// Seek a planned session without global configuration, preserving voices,
    /// volume/speed and user pause. The new ID owns all subsequent input updates.
    pub async fn seek_plan(
        &mut self,
        old_id: &str,
        new_id: String,
        byte: usize,
    ) -> Result<(), SessionError> {
        let job = self.active(old_id)?;
        if job.legacy_settings.is_some() {
            return Err(SessionError::Invalid(
                "use seek for legacy configuration sessions".into(),
            ));
        }
        let (volume, speed) = (job.volume.get(), job.speed.get());
        self.restart_plan(old_id, new_id, byte, volume, speed).await
    }

    async fn restart_plan(
        &mut self,
        old_id: &str,
        new_id: String,
        byte: usize,
        volume: f32,
        speed: f32,
    ) -> Result<(), SessionError> {
        let job = self.active(old_id)?;
        let plan = job.input.plan.borrow().clone();
        let paused = job.paused.get();
        self.start_plan(
            new_id.clone(),
            plan,
            PlanSessionOptions {
                volume,
                speed,
                resume_byte: Some(byte),
                restore_checkpoint: false,
                continuation: job.input.continuation,
                params: job.input.params.clone(),
                seed: job.input.seed,
                staging: job.staging.clone(),
                verification: job.verification.clone(),
            },
        )
        .await?;
        if paused {
            self.pause(&new_id).await?;
        }
        Ok(())
    }

    async fn emit(&self, id: &str, event: Event) -> Result<(), SessionError> {
        self.events
            .send(SessionEvent {
                session_id: id.into(),
                event,
            })
            .await
            .map_err(|_| SessionError::Disconnected)
    }
}

impl Drop for SessionManager {
    fn drop(&mut self) {
        self.player.stop();
        self.job.take();
    }
}

mod playback;
use playback::Runner;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::{AudioChunk, Streaming},
        text::preprocess_text,
    };
    use tts_protocol::{Capabilities, SourceId, text_hash};

    struct FakeBackend {
        calls: Cell<usize>,
        fail_at: Option<usize>,
    }
    impl Backend for FakeBackend {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                model: None,
                model_name: String::new(),
                default_voice: "Weiguo".into(),
                voice_names: Default::default(),
                backend: "moss".into(),
                voices: vec!["Weiguo".into(), "Zf002".into()],
                native_streaming: false,
                style: false,
                compiled_devices: Vec::new(),
                cloning: false,
                pronunciation: false,
                continuation: false,
                parameters: Vec::new(),
            }
        }
        fn stream<'a>(&'a self, _: crate::backend::SegmentRequest<'a>) -> Streaming<'a> {
            Box::pin(async move {
                let index = self.calls.get();
                self.calls.set(index + 1);
                if self.fail_at == Some(index) {
                    return Err(BackendError::Synthesis("injected failure".into()));
                }
                let (tx, rx) = mpsc::channel(2);
                tx.send(Ok(AudioChunk::Pcm(Pcm {
                    samples: vec![0.1; 240],
                    sample_rate: 24000,
                    channels: 1,
                })))
                .await
                .unwrap();
                tx.send(Ok(AudioChunk::End)).await.unwrap();
                Ok(rx)
            })
        }
    }
    #[derive(Default)]
    struct FakePlayer {
        empty: Cell<bool>,
        appended: Cell<usize>,
        cursor: Cell<Duration>,
    }
    impl Playback for FakePlayer {
        fn append(&self, _: Arc<Pcm>) {
            self.appended.set(self.appended.get() + 1);
        }
        fn position(&self) -> Duration {
            self.cursor.get()
        }
        fn is_empty(&self) -> bool {
            self.empty.get()
        }
        fn pause(&self) {}
        fn resume(&self) {}
        fn stop(&self) {}
        fn configure(&self, _: f32, _: f32) {}
    }
    #[derive(Default)]
    struct PausablePlayer {
        paused: Cell<bool>,
        queued: Cell<Duration>,
        cursor: Cell<Duration>,
    }
    impl Playback for PausablePlayer {
        fn append(&self, pcm: Arc<Pcm>) {
            self.queued
                .set(self.queued.get() + Duration::from_millis(pcm.duration_ms().unwrap() as u64));
        }
        fn position(&self) -> Duration {
            self.cursor.get()
        }
        fn is_empty(&self) -> bool {
            self.cursor.get() >= self.queued.get()
        }
        fn pause(&self) {
            self.paused.set(true);
        }
        fn resume(&self) {
            self.paused.set(false);
        }
        fn stop(&self) {
            self.paused.set(true);
        }
        fn configure(&self, _: f32, _: f32) {}
    }
    struct GatedBackend(Arc<tokio::sync::Notify>);
    impl Backend for GatedBackend {
        fn capabilities(&self) -> Capabilities {
            FakeBackend {
                calls: Cell::new(0),
                fail_at: None,
            }
            .capabilities()
        }
        fn stream<'a>(&'a self, _: crate::backend::SegmentRequest<'a>) -> Streaming<'a> {
            Box::pin(async move {
                let (tx, rx) = mpsc::channel(1);
                let gate = self.0.clone();
                tokio::task::spawn_local(async move {
                    for milliseconds in [100, 3000] {
                        tx.send(Ok(AudioChunk::Pcm(Pcm {
                            samples: vec![0.1; milliseconds],
                            sample_rate: 1000,
                            channels: 1,
                        })))
                        .await
                        .unwrap();
                        if milliseconds == 100 {
                            gate.notified().await;
                        }
                    }
                    tx.send(Ok(AudioChunk::End)).await.unwrap();
                });
                Ok(rx)
            })
        }
    }
    #[tokio::test]
    async fn user_resume_cannot_bypass_buffer_and_refill_cannot_override_pause() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let directory = tempfile::tempdir().unwrap();
                let player = Rc::new(PausablePlayer::default());
                let gate = Arc::new(tokio::sync::Notify::new());
                let (tx, mut rx) = mpsc::channel(64);
                let mut manager = SessionManager::new(
                    Rc::new(GatedBackend(gate.clone())),
                    player.clone(),
                    CheckpointStore::new(directory.path()),
                    tx,
                );
                manager
                    .start("buffer".into(), request("正文。"), &Config::default())
                    .await
                    .unwrap();
                loop {
                    if matches!(
                        rx.recv().await.unwrap().event,
                        Event::SessionState {
                            state: SessionState::Buffering
                        }
                    ) {
                        break;
                    }
                }
                manager.resume("buffer").await.unwrap();
                assert!(player.paused.get());
                manager.pause("buffer").await.unwrap();
                gate.notify_one();
                tokio::time::sleep(Duration::from_millis(50)).await;
                assert!(player.paused.get());
                assert_eq!(manager.status().1, SessionState::Paused);
                while let Ok(event) = rx.try_recv() {
                    assert!(!matches!(event.event, Event::SegmentStarted { .. }));
                }
                manager.resume("buffer").await.unwrap();
                assert!(!player.paused.get());
                player.cursor.set(player.queued.get());
                assert_eq!(terminal(&mut rx).await.0, EndReason::Completed);
            })
            .await;
    }
    fn request(text: &str) -> StartRequest {
        StartRequest {
            source: SourceId {
                namespace: "test".into(),
                book: "book".into(),
                chapter: "1".into(),
            },
            text: text.into(),
            text_hash: text_hash(text),
            resume_byte: None,
            restore_checkpoint: false,
        }
    }
    async fn terminal(rx: &mut mpsc::Receiver<SessionEvent>) -> (EndReason, Vec<Event>) {
        let mut events = Vec::new();
        loop {
            let event = tokio::time::timeout(Duration::from_secs(2), rx.recv())
                .await
                .unwrap()
                .unwrap()
                .event;
            if let Event::SessionEnded { reason, .. } = event {
                return (reason, events);
            }
            events.push(event);
        }
    }

    #[tokio::test]
    async fn failures_during_prebuffering_preserve_unplayed_source() {
        tokio::task::LocalSet::new()
            .run_until(async {
                for fail_at in 0..3 {
                    let directory = tempfile::tempdir().unwrap();
                    let player = Rc::new(FakePlayer {
                        empty: Cell::new(true),
                        ..Default::default()
                    });
                    let (tx, mut rx) = mpsc::channel(64);
                    let mut manager = SessionManager::new(
                        Rc::new(FakeBackend {
                            calls: Cell::new(0),
                            fail_at: Some(fail_at),
                        }),
                        player.clone(),
                        CheckpointStore::new(directory.path()),
                        tx,
                    );
                    manager
                        .start(
                            "a".into(),
                            request("第一句。\n第二句。\n第三句。"),
                            &Config::default(),
                        )
                        .await
                        .unwrap();
                    assert_eq!(terminal(&mut rx).await.0, EndReason::Failed);
                    assert_eq!(player.appended.get(), fail_at);
                    manager.stop().await.unwrap();
                    assert!(rx.try_recv().is_err());
                    let mut retry = request("第一句。\n第二句。\n第三句。");
                    let store = CheckpointStore::new(directory.path());
                    let checkpoint = store.load(&retry.source, &retry.text).unwrap().unwrap();
                    let segments = preprocess_text(&retry.text, 200);
                    // Enqueued packets are paused during prebuffering, so none were played.
                    let expected = segments[0].start;
                    assert_eq!(checkpoint.resume_byte, expected);
                    assert!(!checkpoint.completed);
                    retry.restore_checkpoint = true;
                    let (tx, mut rx) = mpsc::channel(64);
                    // A fresh manager models a process restart after the failed attempt.
                    let mut restarted = SessionManager::new(
                        Rc::new(FakeBackend {
                            calls: Cell::new(0),
                            fail_at: None,
                        }),
                        Rc::new(FakePlayer {
                            empty: Cell::new(true),
                            ..Default::default()
                        }),
                        store.clone(),
                        tx,
                    );
                    restarted
                        .start("retry".into(), retry.clone(), &Config::default())
                        .await
                        .unwrap();
                    let (reason, events) = terminal(&mut rx).await;
                    assert_eq!(reason, EndReason::Completed);
                    let first = events.iter().find_map(|event| match event {
                        Event::SegmentStarted { range, .. } => Some(range.start),
                        _ => None,
                    });
                    assert_eq!(first, Some(segments[0].start));
                    assert!(
                        store
                            .load(&retry.source, &retry.text)
                            .unwrap()
                            .unwrap()
                            .completed
                    );
                }
            })
            .await;
    }

    #[tokio::test]
    async fn empty_text_completes_and_bad_resume_is_rejected() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let directory = tempfile::tempdir().unwrap();
                let (tx, mut rx) = mpsc::channel(64);
                let mut manager = SessionManager::new(
                    Rc::new(FakeBackend {
                        calls: Cell::new(0),
                        fail_at: None,
                    }),
                    Rc::new(FakePlayer::default()),
                    CheckpointStore::new(directory.path()),
                    tx,
                );
                manager
                    .start("empty".into(), request("\r\n  "), &Config::default())
                    .await
                    .unwrap();
                assert_eq!(terminal(&mut rx).await.0, EndReason::Completed);
                let mut req = request("中文🙂");
                req.resume_byte = Some(7);
                assert!(
                    manager
                        .start("bad".into(), req, &Config::default())
                        .await
                        .is_err()
                );
            })
            .await;
    }

    #[tokio::test]
    async fn synthesis_does_not_advance_playback_and_cancel_has_one_terminal() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let directory = tempfile::tempdir().unwrap();
                let (tx, mut rx) = mpsc::channel(64);
                let backend = Rc::new(FakeBackend {
                    calls: Cell::new(0),
                    fail_at: None,
                });
                let player = Rc::new(FakePlayer::default());
                let mut manager = SessionManager::new(
                    backend.clone(),
                    player.clone(),
                    CheckpointStore::new(directory.path()),
                    tx,
                );
                manager
                    .start(
                        "old".into(),
                        request("第一句。\n第二句。\n第三句。\n第四句。"),
                        &Config::default(),
                    )
                    .await
                    .unwrap();
                loop {
                    if matches!(rx.recv().await.unwrap().event, Event::SegmentStarted { .. }) {
                        break;
                    }
                }
                manager.pause("old").await.unwrap();
                tokio::time::sleep(Duration::from_millis(40)).await;
                assert!((1..=4).contains(&player.appended.get()));
                assert!(backend.calls.get() <= 4);
                assert_eq!(manager.job.as_ref().unwrap().position.get(), 0);
                manager.stop().await.unwrap();
                assert_eq!(terminal(&mut rx).await.0, EndReason::Cancelled);
                manager.stop().await.unwrap();
                assert!(rx.try_recv().is_err());
                assert!(manager.resume("old").await.is_err());
            })
            .await;
    }

    #[tokio::test]
    async fn seek_and_replacement_cancel_old_sessions_once() {
        tokio::task::LocalSet::new().run_until(async {
            let directory = tempfile::tempdir().unwrap();
            let (tx, mut rx) = mpsc::channel(64);
            let player = Rc::new(FakePlayer::default());
            let mut manager = SessionManager::new(
                Rc::new(FakeBackend { calls: Cell::new(0), fail_at: None }),
                player.clone(), CheckpointStore::new(directory.path()), tx,
            );
            let req = request("第一句。\n第二句。\n第三句。");
            manager.start("first".into(), req.clone(), &Config::default()).await.unwrap();
            loop { if matches!(rx.recv().await.unwrap().event, Event::SegmentStarted{..}) { break; } }
            manager.pause("first").await.unwrap();
            manager.resume("first").await.unwrap();
            assert!(manager.seek("first", "invalid".into(), 1, &Config::default()).await.is_err());
            let byte = preprocess_text(&req.text, 200)[1].start;
            manager.seek("first", "seeked".into(), byte, &Config::default()).await.unwrap();
            let mut cancelled = Vec::new();
            loop {
                let event = rx.recv().await.unwrap();
                if matches!(event.event, Event::SessionEnded{reason: EndReason::Cancelled, ..}) { cancelled.push(event.session_id.clone()); }
                if event.session_id == "seeked" && matches!(event.event, Event::SegmentStarted{..}) {
                    assert!(matches!(event.event, Event::SegmentStarted{range, ..} if range.start == byte));
                    break;
                }
            }
            assert!(manager.pause("first").await.is_err());
            let mut next = request("新章节。 ");
            next.source.chapter = "2".into();
            manager.start("next".into(), next, &Config { voice: "Zf002".into(), ..Config::default() }).await.unwrap();
            manager.stop().await.unwrap();
            manager.stop().await.unwrap();
            while let Ok(event) = rx.try_recv() {
                if matches!(event.event, Event::SessionEnded{reason: EndReason::Cancelled, ..}) { cancelled.push(event.session_id); }
            }
            assert_eq!(cancelled, ["first", "seeked", "next"]);
        }).await;
    }

    #[tokio::test]
    async fn budget_releases_only_after_audio_is_consumed() {
        let budget = Budget::default();
        let audio = || Pcm {
            samples: vec![0.0; 480_000],
            sample_rate: 24_000,
            channels: 1,
        };
        let packet = budget.acquire(audio()).await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), budget.acquire(audio()))
                .await
                .is_err()
        );
        drop(packet);
        assert!(budget.acquire(audio()).await.is_ok());
    }

    #[tokio::test]
    async fn restart_uses_snapshot_position_after_segmentation_changes() {
        tokio::task::LocalSet::new()
            .run_until(async {
                let directory = tempfile::tempdir().unwrap();
                let store = CheckpointStore::new(directory.path());
                let text = "中文🙂\r\n第二句。";
                let mut req = request(text);
                store.save(&req.source, text, 12, false).unwrap();
                req.restore_checkpoint = true;
                let (tx, mut rx) = mpsc::channel(64);
                let mut manager = SessionManager::new(
                    Rc::new(FakeBackend {
                        calls: Cell::new(0),
                        fail_at: None,
                    }),
                    Rc::new(FakePlayer {
                        empty: Cell::new(true),
                        ..Default::default()
                    }),
                    store,
                    tx,
                );
                manager
                    .start("restored".into(), req, &Config::default())
                    .await
                    .unwrap();
                let (reason, events) = terminal(&mut rx).await;
                assert_eq!(reason, EndReason::Completed);
                let ranges: Vec<_> = events
                    .iter()
                    .filter_map(|event| {
                        if let Event::SegmentStarted { range, .. } = event {
                            Some(*range)
                        } else {
                            None
                        }
                    })
                    .collect();
                assert_eq!(ranges, vec![TextRange { start: 12, end: 24 }]);
            })
            .await;
    }
    struct FragmentedBackend {
        ended: bool,
        format_change: bool,
    }
    impl Backend for FragmentedBackend {
        fn capabilities(&self) -> Capabilities {
            FakeBackend {
                calls: Cell::new(0),
                fail_at: None,
            }
            .capabilities()
        }
        fn stream<'a>(&'a self, _: crate::backend::SegmentRequest<'a>) -> Streaming<'a> {
            Box::pin(async move {
                let (tx, rx) = mpsc::channel(4);
                for channels in [1, if self.format_change { 2 } else { 1 }] {
                    tx.send(Ok(AudioChunk::Pcm(Pcm {
                        samples: vec![0.1; 240],
                        sample_rate: 24000,
                        channels,
                    })))
                    .await
                    .unwrap();
                }
                if self.ended {
                    tx.send(Ok(AudioChunk::End)).await.unwrap();
                }
                Ok(rx)
            })
        }
    }
    #[tokio::test(flavor = "current_thread")]
    async fn fragmented_streams_require_completion_and_consistent_format() {
        tokio::task::LocalSet::new()
            .run_until(async {
                for (ended, format_change, reason) in [
                    (true, false, EndReason::Completed),
                    (false, false, EndReason::Failed),
                    (true, true, EndReason::Failed),
                ] {
                    let directory = tempfile::tempdir().unwrap();
                    let checkpoints = CheckpointStore::new(directory.path());
                    let (tx, mut rx) = mpsc::channel(64);
                    let player = Rc::new(FakePlayer {
                        empty: Cell::new(true),
                        appended: Cell::new(0),
                        cursor: Cell::new(Duration::ZERO),
                    });
                    let mut manager = SessionManager::new(
                        Rc::new(FragmentedBackend {
                            ended,
                            format_change,
                        }),
                        player.clone(),
                        checkpoints.clone(),
                        tx,
                    );
                    let req = request("原文。 ");
                    manager
                        .start("stream".into(), req.clone(), &Config::default())
                        .await
                        .unwrap();
                    let (actual, events) = terminal(&mut rx).await;
                    assert_eq!(actual, reason);
                    assert_eq!(
                        events
                            .iter()
                            .filter(|e| matches!(e, Event::SegmentStarted { .. }))
                            .count(),
                        usize::from(reason == EndReason::Completed)
                    );
                    if reason == EndReason::Completed {
                        assert_eq!(player.appended.get(), 2);
                    } else {
                        assert!(
                            !events
                                .iter()
                                .any(|e| matches!(e, Event::SegmentFinished { .. }))
                        );
                    }
                }
            })
            .await;
    }
}
