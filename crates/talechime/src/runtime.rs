//! Worker application state, independent of JSON Lines transport.
#[cfg(test)]
use std::rc::Rc;
use std::time::Duration;
use talechime_backends::Registry;
use talechime_core::{
    checkpoint::CheckpointStore,
    config::{ConfigError, ConfigStore},
    session::SessionEvent,
};
use talechime_protocol::{Command, ErrorInfo, Event, Request, SessionState};
use tokio::{sync::mpsc, task::JoinHandle};

type Preparation = JoinHandle<anyhow::Result<crate::preparation::PreparedModels>>;
type DesignJob = JoinHandle<anyhow::Result<talechime::VoiceAsset>>;

enum BackgroundJob {
    Preparing(Preparation),
    Designing(DesignJob),
}
impl BackgroundJob {
    fn abort(&self) {
        match self {
            Self::Preparing(task) => task.abort(),
            Self::Designing(task) => task.abort(),
        }
    }
}

pub struct Response {
    pub session: Option<String>,
    pub event: Event,
}
impl Response {
    fn global(event: Event) -> Self {
        Self {
            session: None,
            event,
        }
    }
    fn error(code: &str, stage: &str, error: impl ToString) -> Self {
        Self::global(Event::Error(ErrorInfo {
            code: code.into(),
            stage: stage.into(),
            message: error.to_string(),
            retryable: true,
        }))
    }
}

/// A ready command is handed back to transport before any command side effects.
/// The select loop may cancel polling, but cannot lose an already dequeued command.
pub enum BackgroundOutcome {
    Event(Response),
    Command(Request),
}

pub struct Worker {
    readback_models: bool,
    store: ConfigStore,
    checkpoints: CheckpointStore,
    resources: Registry,
    events: mpsc::Sender<SessionEvent>,
    progress: mpsc::Sender<Event>,
    background: Option<BackgroundJob>,
    queued: std::collections::VecDeque<Request>,
    voice_studio: Option<talechime::VoiceStudio>,
    progress_forwarder: Option<JoinHandle<()>>,
    manager: Option<crate::app_session::AppSession>,
    plan_spans: usize,
    resource_state: SessionState,
}
impl Worker {
    pub fn new(
        store: ConfigStore,
        checkpoints: CheckpointStore,
        resources: Registry,
        events: mpsc::Sender<SessionEvent>,
        progress: mpsc::Sender<Event>,
    ) -> Self {
        // Voice assets live beside the config file, never inside model resources.
        let voices = store
            .path()
            .parent()
            .map(|root| root.join("voices"))
            .and_then(|voices| talechime::VoiceStudio::at(resources.root(), voices).ok());
        Self {
            readback_models: false,
            store,
            checkpoints,
            resources,
            events,
            progress,
            background: None,
            queued: Default::default(),
            voice_studio: voices,
            progress_forwarder: None,
            manager: None,
            plan_spans: 0,
            resource_state: SessionState::Idle,
        }
    }
    #[cfg(test)]
    pub fn attach_test(
        &mut self,
        backend: Rc<dyn talechime_core::backend::Backend>,
        player: Rc<dyn talechime::Playback>,
    ) {
        self.manager = Some(
            crate::app_session::AppSession::with_player(
                backend,
                player,
                &self.checkpoints,
                self.events.clone(),
            )
            .unwrap(),
        );
    }
    pub fn catalog(&self) -> anyhow::Result<Vec<talechime_protocol::Capabilities>> {
        self.resources.catalog()
    }
    #[cfg(any(
        feature = "moss",
        feature = "qwen",
        feature = "voxcpm",
        feature = "cosyvoice3",
        feature = "indextts",
        feature = "omnivoice",
    ))]
    pub fn unprepared_device_status(&self) -> anyhow::Result<Event> {
        let config = self.store.load()?;
        Ok(self.resources.device_status_for(
            &config.backend,
            config.model.as_deref(),
            talechime_protocol::Device::Auto,
            Some("resources not prepared".into()),
        ))
    }
    pub fn is_preparing(&self) -> bool {
        matches!(self.background, Some(BackgroundJob::Preparing(_)))
    }
    pub fn is_designing(&self) -> bool {
        matches!(self.background, Some(BackgroundJob::Designing(_)))
    }
    pub fn has_background_work(&self) -> bool {
        self.is_preparing() || self.is_designing() || !self.queued.is_empty()
    }
    async fn session_active(&self) -> bool {
        let (_, state) = self.status().await;
        !matches!(
            state,
            SessionState::Idle | SessionState::Stopped | SessionState::Failed
        )
    }
    /// Test hook: inject a studio with overridden design engines.
    #[cfg(test)]
    pub fn test_design_studio(&mut self, engines: Vec<talechime_backends::design::DesignerHandle>) {
        if let Some(studio) = self.voice_studio.as_mut() {
            studio.set_designers(engines);
        }
    }
    #[cfg(any(
        feature = "moss",
        feature = "qwen",
        feature = "voxcpm",
        feature = "cosyvoice3",
        feature = "indextts",
        feature = "omnivoice",
    ))]
    pub fn has_prepared_model(&self) -> bool {
        self.manager.is_some()
    }
    /// Poll the single background job (preparation or design) when one exists.
    pub async fn poll_background(&mut self) -> BackgroundOutcome {
        if self.is_preparing() {
            return BackgroundOutcome::Event(self.prepared().await);
        }
        if self.is_designing() {
            return BackgroundOutcome::Event(self.designed().await);
        }
        if let Some(request) = self.queued.front() {
            if matches!(request.command, Command::DesignVoice { .. }) {
                while self.session_active().await {
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
            }
            if let Some(request) = self.queued.pop_front() {
                return BackgroundOutcome::Command(request);
            }
        }
        BackgroundOutcome::Event(Response::error(
            "prepare_failed",
            "model",
            "no background job is running",
        ))
    }
    /// Poll only while a design job exists; completion publishes `VoiceReady`.
    pub async fn designed(&mut self) -> Response {
        let Some(BackgroundJob::Designing(task)) = &mut self.background else {
            return Response::error("design_failed", "voice", "no design job is running");
        };
        // Drain progress before consuming the join result. Both awaits remain
        // cancellation-safe when the transport selects another input/event.
        if let Some(forwarder) = &mut self.progress_forwarder {
            let _ = forwarder.await;
        }
        self.progress_forwarder.take();
        let finished = task.await;
        self.background.take();
        match finished {
            Ok(Ok(asset)) => Response::global(Event::VoiceReady {
                voice: asset.id,
                name: asset.name,
            }),
            Ok(Err(error)) => Response::error("design_failed", "voice", error),
            Err(error) => Response::error("design_failed", "voice", error),
        }
    }
    async fn status(&self) -> (Option<String>, SessionState) {
        match &self.manager {
            Some(manager) => manager
                .control
                .status()
                .await
                .unwrap_or((None, SessionState::Failed)),
            None => (None, self.resource_state),
        }
    }
    pub fn enable_readback_models(&mut self, enabled: bool) {
        self.readback_models = enabled;
    }
    /// Poll only while preparation exists; dropping this future keeps the task owned.
    pub async fn prepared(&mut self) -> Response {
        let Some(BackgroundJob::Preparing(task)) = &mut self.background else {
            return Response::error("prepare_failed", "model", "no preparation is running");
        };
        let loaded = task.await;
        self.background.take();
        let prepared = match loaded {
            Ok(Ok(backend)) => backend,
            Ok(Err(error)) => {
                self.resource_state = SessionState::Failed;
                return Response::error("prepare_failed", "model", error);
            }
            Err(error) => {
                self.resource_state = SessionState::Failed;
                return Response::error("prepare_failed", "model", error);
            }
        };
        match crate::app_session::AppSession::open(
            prepared.backend,
            &self.checkpoints,
            self.events.clone(),
            prepared.verifier,
        ) {
            Ok(manager) => {
                self.manager = Some(manager);
                self.resource_state = SessionState::Idle;
                Response::global(Event::ModelReady)
            }
            Err(error) => {
                self.resource_state = SessionState::Failed;
                Response::error("device_unavailable", "audio", error)
            }
        }
    }
    async fn cancel_prepare(&mut self) -> Option<talechime::VoiceAsset> {
        self.queued.clear();
        let completed = match self.background.take() {
            Some(BackgroundJob::Preparing(task)) => {
                task.abort();
                let _ = task.await;
                None
            }
            Some(BackgroundJob::Designing(task)) => {
                task.abort();
                task.await.ok().and_then(Result::ok)
            }
            None => None,
        };
        self.disconnect_progress();
        self.resource_state = SessionState::Idle;
        completed
    }
    fn disconnect_progress(&mut self) {
        if let Some(task) = self.progress_forwarder.take() {
            task.abort();
        }
    }
    pub async fn command(&mut self, request: &Request) -> Response {
        let exclusive = matches!(
            request.command,
            Command::Start(_) | Command::PrepareModel | Command::DesignVoice { .. }
        );
        // Input for a queued Start must retain its order; controls for the
        // current session still pass through so it can finish or stop.
        let follows_start = request.session_id.is_some()
            && self.queued.iter().any(|queued| {
                matches!(queued.command, Command::Start(_))
                    && queued.session_id == request.session_id
            });
        let waits_for_session =
            matches!(request.command, Command::DesignVoice { .. }) && self.session_active().await;
        if follows_start || exclusive && (self.has_background_work() || waits_for_session) {
            if self.queued.len() == 16 {
                return Response::error("busy", "voice", "resource command queue is full");
            }
            self.queued.push_back(request.clone());
            return Response::global(Event::Accepted);
        }
        self.execute_command(request).await
    }
    pub async fn execute_command(&mut self, request: &Request) -> Response {
        match &request.command {
            Command::GetStatus => {
                let (session, state) = self.status().await;
                Response {
                    session,
                    event: Event::SessionState { state },
                }
            }
            Command::GetConfig => match self.store.load() {
                Ok(config) => Response::global(Event::Config(config)),
                Err(error) => Response::error("config_invalid", "config", error),
            },
            Command::UpdateConfig(patch) => {
                if self.is_designing()
                    && (patch.backend.is_some()
                        || patch.model.is_some()
                        || patch.tts_device.is_some())
                {
                    return Response::error(
                        "busy",
                        "voice",
                        "cancel voice design before changing model resources",
                    );
                }
                let old = match self.store.load() {
                    Ok(config) => config,
                    Err(error) => return Response::error("config_invalid", "config", error),
                };
                let target = patch.backend.as_deref().unwrap_or(&old.backend);
                let mut caps = if patch.is_playback_only() {
                    None
                } else {
                    match self
                        .resources
                        .capabilities_for(target, patch.target_model(&old))
                    {
                        Ok(caps) => Some(caps),
                        Err(error) => {
                            return Response::error("backend_unavailable", "config", error);
                        }
                    }
                };
                if let Some(caps) = caps.as_mut()
                    && let Some(studio) = &self.voice_studio
                    && let Err(error) = crate::preparation::include_library_voice(
                        caps,
                        patch.voice.as_deref().unwrap_or(&old.voice),
                        studio,
                    )
                {
                    return Response::error("voice_invalid", "config", error);
                }
                #[cfg(any(
                    feature = "moss",
                    feature = "qwen",
                    feature = "voxcpm",
                    feature = "cosyvoice3",
                    feature = "indextts",
                    feature = "omnivoice",
                ))]
                if let Some(device) = patch.tts_device.or_else(|| {
                    (target != old.backend || patch.target_model(&old) != old.model.as_deref())
                        .then_some(old.tts_device)
                }) && let Err(error) = crate::preparation::validate_device(
                    device,
                    target,
                    patch.target_model(&old),
                    &self.resources,
                ) {
                    return Response::error("device_unavailable", "config", error);
                }
                let update = match caps.as_ref() {
                    Some(caps) => self.store.update(patch, caps),
                    None => self.store.update_playback(patch),
                };
                let config = match update {
                    Ok(config) => config,
                    Err(error) => {
                        let code = if matches!(error, ConfigError::RevisionConflict) {
                            "revision_conflict"
                        } else {
                            "config_invalid"
                        };
                        return Response::error(code, "config", error);
                    }
                };
                if config.model != old.model
                    || config.backend != old.backend
                    || config.tts_device != old.tts_device
                {
                    let _ = self.cancel_prepare().await;
                    self.disconnect_progress();
                    if let Some(mut manager) = self.manager.take()
                        && let Err(error) = manager.stop_and_close().await
                    {
                        self.resource_state = SessionState::Failed;
                        return Response::error("session_cleanup_failed", "session", error);
                    }
                    self.resource_state = SessionState::Idle;
                    self.plan_spans = 0;
                    return Response::global(Event::ConfigChanged(config));
                }
                if (config.volume != old.volume || config.speed != old.speed)
                    && let Some(manager) = &self.manager
                    && let Ok((Some(active), state)) = manager.control.status().await
                    && !matches!(
                        state,
                        SessionState::Stopped | SessionState::Failed | SessionState::Idle
                    )
                    && let Err(error) = manager
                        .control
                        .configure_playback(active, config.volume, config.speed)
                        .await
                {
                    return Response::error("session_invalid", "session", error);
                }
                Response::global(Event::ConfigChanged(config))
            }
            Command::PrepareModel => {
                if self.manager.is_some() {
                    return Response::global(Event::ModelReady);
                }
                if self.is_designing() {
                    return Response::error("busy", "voice", "a design job is running");
                }
                if !self.is_preparing() {
                    let config = match self.store.load() {
                        Ok(config) => config,
                        Err(error) => return Response::error("config_invalid", "config", error),
                    };
                    let mut caps = match self
                        .resources
                        .capabilities_for(&config.backend, config.model.as_deref())
                    {
                        Ok(caps) => caps,
                        Err(error) => {
                            return Response::error("backend_unavailable", "model", error);
                        }
                    };
                    if let Some(studio) = &self.voice_studio
                        && let Err(error) = crate::preparation::include_library_voice(
                            &mut caps,
                            &config.voice,
                            studio,
                        )
                    {
                        return Response::error("voice_invalid", "config", error);
                    }
                    let config = match self.store.initialize(&caps) {
                        Ok(config) => config,
                        Err(error) => return Response::error("config_invalid", "config", error),
                    };
                    let resources = self.resources.clone();
                    let studio = self.voice_studio.clone();
                    self.disconnect_progress();
                    let (progress, mut updates) = mpsc::channel(16);
                    let outgoing = self.progress.clone();
                    self.progress_forwarder = Some(tokio::task::spawn_local(async move {
                        while let Some(event) = updates.recv().await {
                            if outgoing.send(event).await.is_err() {
                                break;
                            }
                        }
                    }));
                    let readback = self.readback_models;
                    self.background = Some(BackgroundJob::Preparing(tokio::task::spawn_local(
                        async move {
                            let mut config = config;
                            if let Some(studio) = studio {
                                let caps = resources
                                    .capabilities_for(&config.backend, config.model.as_deref())?;
                                config.voice = studio.resolve_voice(&caps, &config.voice).await?;
                            }
                            crate::preparation::prepare_models(
                                resources, config, progress, readback,
                            )
                            .await
                        },
                    )));
                    self.resource_state = SessionState::Preparing;
                }
                Response::global(Event::Accepted)
            }
            Command::CancelPrepare => match self.cancel_prepare().await {
                Some(asset) => Response::global(Event::VoiceReady {
                    voice: asset.id,
                    name: asset.name,
                }),
                None => Response::global(Event::Accepted),
            },
            Command::DesignVoice {
                name,
                description,
                text,
                engine,
                device,
                seed,
                id,
            } => {
                if self.is_designing() {
                    return Response::error("busy", "voice", "one design job at a time");
                }
                if self.is_preparing() {
                    return Response::error("busy", "model", "preparation is running");
                }
                let Some(studio) = self.voice_studio.clone() else {
                    return Response::error(
                        "voice_studio_unavailable",
                        "voice",
                        "voice assets are unavailable in this configuration",
                    );
                };
                let options = talechime::DesignOptions {
                    description: description.clone(),
                    engine: engine.clone(),
                    id: id.clone(),
                    name: Some(name.clone()),
                    text: text.clone(),
                    device: device.unwrap_or(talechime_protocol::Device::Auto),
                    seed: *seed,
                };
                self.disconnect_progress();
                let (progress, mut updates) = mpsc::channel(16);
                let outgoing = self.progress.clone();
                self.progress_forwarder = Some(tokio::task::spawn_local(async move {
                    while let Some(event) = updates.recv().await {
                        if outgoing.send(event).await.is_err() {
                            break;
                        }
                    }
                }));
                self.background = Some(BackgroundJob::Designing(tokio::task::spawn_local(
                    async move {
                        studio
                            .design_with_channel(options, progress)
                            .await
                            .map_err(|error| anyhow::anyhow!(error.to_string()))
                    },
                )));
                Response::global(Event::Accepted)
            }
            Command::ListVoices => match &self.voice_studio {
                Some(studio) => match studio.list() {
                    Ok(voices) => Response::global(Event::VoicesListed(
                        voices
                            .into_iter()
                            .map(|asset| talechime_protocol::VoiceAssetRecord {
                                id: asset.id,
                                name: asset.name,
                                transcript: asset.transcript,
                                description: asset.description,
                                designed: matches!(
                                    asset.source,
                                    talechime::VoiceSource::Designed(_)
                                ),
                            })
                            .collect(),
                    )),
                    Err(error) => Response::error("voice_invalid", "voice", error),
                },
                None => Response::error(
                    "voice_studio_unavailable",
                    "voice",
                    "voice assets are unavailable in this configuration",
                ),
            },
            Command::RemoveVoice { voice } => match &self.voice_studio {
                Some(studio) => match studio.remove(voice) {
                    Ok(()) => Response::global(Event::Accepted),
                    Err(error) => Response::error("voice_invalid", "voice", error),
                },
                None => Response::error(
                    "voice_studio_unavailable",
                    "voice",
                    "voice assets are unavailable in this configuration",
                ),
            },
            Command::Hello | Command::Shutdown => Response::error(
                "invalid_request",
                "protocol",
                "transport command reached worker",
            ),
            command => self.session_command(request, command).await,
        }
    }
    async fn session_command(&mut self, request: &Request, command: &Command) -> Response {
        let Some(manager) = &mut self.manager else {
            return Response::error(
                "model_not_ready",
                "session",
                "prepare_model must finish before playback",
            );
        };
        let Some(id) = request.session_id.as_deref().filter(|id| !id.is_empty()) else {
            return Response::error(
                "session_id_required",
                "session",
                "session command requires a nonempty ID",
            );
        };
        let result = match command {
            Command::Start(input) => {
                let caps = match manager.capabilities() {
                    Ok(caps) => caps,
                    Err(error) => return Response::error("session_invalid", "plan", error),
                };
                let plan = match crate::plan_input::build(input, &caps, self.voice_studio.as_ref())
                    .await
                {
                    Ok(result) => result,
                    Err(error) => return Response::error("plan_invalid", "plan", error),
                };
                let config = match self.store.load() {
                    Ok(config) => config,
                    Err(error) => return Response::error("config_invalid", "config", error),
                };
                let (params, seed) = crate::plan_input::generation(input);
                // Chapter reuse lives beside the resources root so tests with
                // temporary homes stay isolated from the user's cache.
                let cache =
                    self.resources
                        .root()
                        .parent()
                        .map(|home| talechime::ChapterCacheOptions {
                            directory: home.join("cache/chapters"),
                            max_bytes: talechime::DEFAULT_MAX_BYTES,
                        });
                let result = manager
                    .control
                    .start(
                        id,
                        plan,
                        talechime::PlanSessionOptions {
                            volume: config.volume,
                            speed: config.speed,
                            resume_byte: input.resume_byte,
                            restore_checkpoint: input.restore_checkpoint,
                            verification: input.verification.clone(),

                            cache,
                            params,
                            seed,
                            ..Default::default()
                        },
                    )
                    .await;
                if result.is_ok() {
                    self.plan_spans = input.spans.len();
                }
                result
            }
            Command::Append { spans } => {
                if let Err(error) = crate::plan_input::check_batch(spans.len(), self.plan_spans) {
                    return Response::error("plan_invalid", "plan", error);
                }
                let result = manager
                    .control
                    .append(id, crate::plan_input::spans(spans))
                    .await;
                if result.is_ok() {
                    self.plan_spans += spans.len();
                }
                result
            }
            Command::Seal => manager.control.seal(id).await,
            Command::FailInput { message } => manager.control.fail_input(id, message).await,
            Command::GetProgress => {
                return match manager.control.progress(id).await {
                    Ok(progress) => Response {
                        session: Some(id.into()),
                        event: Event::Progress(crate::plan_input::progress(progress)),
                    },
                    Err(error) => Response::error("session_invalid", "plan", error),
                };
            }
            Command::Pause => manager.control.pause(id).await,
            Command::Resume => manager.control.resume(id).await,
            Command::Stop => {
                if manager
                    .control
                    .status()
                    .await
                    .ok()
                    .and_then(|status| status.0)
                    .as_deref()
                    == Some(id)
                {
                    manager.control.stop().await
                } else {
                    Err(talechime::EngineError::Session(
                        talechime_core::session::SessionError::Invalid("stale session".into()),
                    ))
                }
            }
            Command::Seek {
                byte,
                new_session_id,
            } => {
                manager
                    .control
                    .seek(id, new_session_id.clone(), *byte)
                    .await
            }
            _ => return Response::error("invalid_request", "session", "not a session command"),
        };
        match result {
            Ok(()) => Response {
                session: Some(match command {
                    Command::Seek { new_session_id, .. } => new_session_id.clone(),
                    _ => id.into(),
                }),
                event: Event::Accepted,
            },
            Err(error) => Response::error("session_invalid", "session", error),
        }
    }
    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        let _ = self.cancel_prepare().await;
        self.disconnect_progress();
        if let Some(manager) = &mut self.manager {
            tokio::time::timeout(Duration::from_secs(3), manager.stop_and_close()).await??;
        }
        Ok(())
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.disconnect_progress();
        if let Some(job) = &self.background {
            job.abort();
        }
    }
}

#[cfg(all(test, feature = "moss"))]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc, sync::Arc};
    use talechime_core::{
        backend::{Backend, BackendError, Pcm, Streaming},
        player::Playback,
    };
    struct SilentBackend(talechime_protocol::Capabilities);
    impl Backend for SilentBackend {
        fn capabilities(&self) -> talechime_protocol::Capabilities {
            self.0.clone()
        }
        fn stream<'a>(&'a self, _: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
            Box::pin(async { Err(BackendError::Synthesis("not used".into())) })
        }
    }
    #[derive(Default)]
    struct IdlePlayer(Cell<usize>);
    impl Playback for IdlePlayer {
        fn append(&self, _: Arc<Pcm>) {}
        fn position(&self) -> Duration {
            Duration::ZERO
        }
        fn is_empty(&self) -> bool {
            true
        }
        fn pause(&self) {}
        fn resume(&self) {}
        fn stop(&self) {
            self.0.set(self.0.get() + 1);
        }
        fn configure(&self, _: f32, _: f32) {}
    }
    #[tokio::test]
    async fn device_change_releases_prepared_manager_and_preserves_checkpoint_files() {
        talechime::run_local(async {

        let root = tempfile::tempdir().unwrap();
        let resources = Registry::new(Some(root.path().join("models"))).unwrap();
        let caps = resources.capabilities("moss").unwrap();
        let store = ConfigStore::new(root.path().join("config.json")).with_defaults(talechime_protocol::Config {
            backend: caps.backend.clone(), model: caps.model.clone(), voice: caps.default_voice.clone(),
            tts_device: talechime_protocol::Device::Cpu, ..Default::default()
        });
        let checkpoints = CheckpointStore::new(root.path().join("checkpoints"));
        std::fs::create_dir_all(root.path().join("checkpoints")).unwrap();
        let progress_file = root.path().join("checkpoints/existing.json");
        std::fs::write(&progress_file, "reliable progress").unwrap();
        let (events, _) = mpsc::channel(32);
        let (progress, _) = mpsc::channel(32);
        let player = Rc::new(IdlePlayer::default());
        let mut worker = Worker::new(
            store,
            checkpoints.clone(),
            resources,
            events.clone(),
            progress,
        );
        worker.manager = Some(
            crate::app_session::AppSession::with_player(
                Rc::new(SilentBackend(caps)),
                player.clone(),
                &checkpoints,
                events,
            )
            .unwrap(),
        );
        let response = worker
            .command(&Request {
                protocol_version: talechime_protocol::PROTOCOL_VERSION,
                request_id: "device".into(),
                session_id: None,
                command: Command::UpdateConfig(talechime_protocol::ConfigPatch {
                    tts_device: Some(talechime_protocol::Device::Auto),
                    ..Default::default()
                }),
            })
            .await;
        assert!(
            matches!(response.event, Event::ConfigChanged(config) if config.tts_device == talechime_protocol::Device::Auto)
        );
        assert!(worker.manager.is_none());
        assert_eq!(worker.resource_state, SessionState::Idle);
        assert!(player.0.get() > 0);
        assert_eq!(
            std::fs::read_to_string(progress_file).unwrap(),
            "reliable progress"
        );

        }).await;
    }
    #[cfg(feature = "moss-nano-candle")]
    #[tokio::test]
    async fn model_change_rejects_inherited_device_before_config_or_manager_changes() {
        talechime::run_local(async {

        let root = tempfile::tempdir().unwrap();
        let resources = Registry::new(Some(root.path().join("models"))).unwrap();
        let defaults = talechime_protocol::Config {
            model: Some("nano-candle".into()),
            tts_device: talechime_protocol::Device::Cpu,
            ..Default::default()
        };
        let store = ConfigStore::new(root.path().join("config.json")).with_defaults(defaults);
        let old = store
            .initialize(&resources.capabilities_for("moss", Some("nano-candle")).unwrap())
            .unwrap();
        let original = std::fs::read(store.path()).unwrap();
        let checkpoints = CheckpointStore::new(root.path().join("checkpoints"));
        let (events, _) = mpsc::channel(32);
        let (progress, _) = mpsc::channel(32);
        let player = Rc::new(IdlePlayer::default());
        let mut worker = Worker::new(
            store.clone(),
            checkpoints.clone(),
            resources.clone(),
            events.clone(),
            progress,
        );
        worker.manager = Some(
            crate::app_session::AppSession::with_player(
                Rc::new(SilentBackend(resources.capabilities_for("moss", Some("nano-candle")).unwrap())),
                player.clone(),
                &checkpoints,
                events,
            )
            .unwrap(),
        );
        let response = worker
            .command(&Request {
                protocol_version: talechime_protocol::PROTOCOL_VERSION,
                request_id: "model".into(),
                session_id: None,
                command: Command::UpdateConfig(talechime_protocol::ConfigPatch {
                    expected_revision: old.revision,
                    model: Some("realtime-1.7b".into()),
                    voice: Some("narrator".into()),
                    ..Default::default()
                }),
            })
            .await;
        assert!(
            matches!(response.event, Event::Error(ref error) if error.code == "device_unavailable")
        );
        assert_eq!(std::fs::read(store.path()).unwrap(), original);
        assert!(worker.manager.is_some());
        assert_eq!(player.0.get(), 0);

        }).await;
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    use crate::fixture::{Fixture, Mode, Player};
    use talechime_protocol::{
        PlanPlayback, PlanRequest, SourceId, TextRange, VoiceSpan, text_hash,
    };

    fn input(sealed: bool, playback: PlanPlayback) -> PlanRequest {
        PlanRequest {
            source: SourceId {
                namespace: "tests".into(),
                book: "book".into(),
                chapter: "chapter".into(),
            },
            text: "甲乙丙".into(),
            text_hash: text_hash("甲乙丙"),
            backend: "fixture".into(),
            model: Some("shared".into()),
            voices: vec!["A".into(), "B".into()],
            spans: if sealed {
                vec![span(0, 3, "A"), span(3, 6, "B"), span(6, 9, "A")]
            } else {
                vec![]
            },
            sealed,
            playback,
            resume_byte: Some(0),
            restore_checkpoint: false,

            params: Default::default(),
            seed: None,
            verification: Default::default(),
        }
    }
    fn span(start: usize, end: usize, voice: &str) -> VoiceSpan {
        VoiceSpan {
            range: TextRange { start, end },
            voice: voice.into(),
            style: None,
        }
    }
    async fn send(worker: &mut Worker, id: Option<&str>, command: Command) -> Response {
        // Exercise the actual JSON DTO boundary before application dispatch.
        let request = Request {
            protocol_version: talechime_protocol::PROTOCOL_VERSION,
            request_id: "test".into(),
            session_id: id.map(str::to_owned),
            command,
        };
        let request =
            talechime_protocol::decode(&talechime_protocol::encode(&request).unwrap()).unwrap();
        worker.command(&request).await
    }
    fn make(
        root: &tempfile::TempDir,
        backend: Rc<Fixture>,
        player: Rc<Player>,
        capacity: usize,
    ) -> (Worker, mpsc::Receiver<SessionEvent>) {
        let resources = Registry::new(Some(root.path().join("models"))).unwrap();
        let store = ConfigStore::new(root.path().join("config.json"));
        let checkpoints = CheckpointStore::new(root.path().join("checkpoints"));
        let (events, receiver) = mpsc::channel(capacity);
        let (progress, _) = mpsc::channel(16);
        let mut worker = Worker::new(
            store,
            checkpoints.clone(),
            resources,
            events.clone(),
            progress,
        );
        worker.manager = Some(
            crate::app_session::AppSession::with_player(backend, player, &checkpoints, events)
                .unwrap(),
        );
        (worker, receiver)
    }
    async fn snapshot(worker: &mut Worker, id: &str) -> talechime_protocol::PlanProgressSnapshot {
        let response = send(worker, Some(id), Command::GetProgress).await;
        if let Event::Progress(progress) = response.event {
            progress
        } else {
            panic!("{:?}", response.event)
        }
    }
    #[tokio::test]
    async fn incremental_transport_is_atomic_and_stale_controls_cannot_mutate_replacement() {
        talechime::run_local(async {
            let root = tempfile::tempdir().unwrap();
            let backend = Rc::new(Fixture::default());
            let (mut worker, mut events) =
                make(&root, backend.clone(), Rc::new(Player::default()), 64);
            let drain = tokio::task::spawn_local(async move {
                while let Some(event) = events.recv().await {
                    if matches!(
                        event.event,
                        Event::SessionEnded {
                            reason: talechime_protocol::EndReason::Completed,
                            ..
                        }
                    ) {
                        return (event, events);
                    }
                }
                panic!("no completion")
            });
            assert!(matches!(
                send(
                    &mut worker,
                    Some("one"),
                    Command::Start(Box::new(input(false, PlanPlayback::Streaming)))
                )
                .await
                .event,
                Event::Accepted
            ));
            assert!(matches!(
                send(&mut worker, Some("one"), Command::Seal).await.event,
                Event::Error(_)
            ));
            assert!(matches!(
                send(
                    &mut worker,
                    Some("one"),
                    Command::Append {
                        spans: vec![span(0, 3, "A"), span(4, 6, "B")]
                    }
                )
                .await
                .event,
                Event::Error(_)
            ));
            assert_eq!(snapshot(&mut worker, "one").await.accepted_end, 0);
            assert!(matches!(
                send(
                    &mut worker,
                    Some("one"),
                    Command::Append {
                        spans: vec![span(0, 3, "A")]
                    }
                )
                .await
                .event,
                Event::Accepted
            ));
            let mut invalid = input(false, PlanPlayback::Streaming);
            invalid.text_hash = "bad".into();
            assert!(matches!(
                send(&mut worker, Some("bad"), Command::Start(Box::new(invalid)))
                    .await
                    .event,
                Event::Error(_)
            ));
            assert_eq!(snapshot(&mut worker, "one").await.accepted_end, 3);
            assert!(matches!(
                send(
                    &mut worker,
                    Some("one"),
                    Command::Seek {
                        byte: 0,
                        new_session_id: "two".into()
                    }
                )
                .await
                .event,
                Event::Accepted
            ));
            assert!(matches!(
                send(
                    &mut worker,
                    Some("one"),
                    Command::FailInput {
                        message: "late".into()
                    }
                )
                .await
                .event,
                Event::Error(_)
            ));
            assert!(matches!(
                send(
                    &mut worker,
                    Some("two"),
                    Command::Append {
                        spans: vec![span(3, 6, "B"), span(6, 9, "A")]
                    }
                )
                .await
                .event,
                Event::Accepted
            ));
            assert!(matches!(
                send(&mut worker, Some("two"), Command::Seal).await.event,
                Event::Accepted
            ));
            let (ended, _events) = tokio::time::timeout(Duration::from_secs(3), drain)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(ended.session_id, "two");
            {
                let calls = backend.calls.borrow();
                assert_eq!(
                    calls
                        .iter()
                        .rev()
                        .take(3)
                        .map(|c| c.1.as_str())
                        .collect::<Vec<_>>(),
                    vec!["A", "B", "A"]
                );
            }
            worker.shutdown().await.unwrap();
            assert!(!root.path().join("models").exists());
        })
        .await;
    }
    #[tokio::test]
    async fn chapter_ready_stays_paused_and_close_drains_without_an_observer() {
        talechime::run_local(async {
            let root = tempfile::tempdir().unwrap();
            let backend = Rc::new(Fixture::default());
            backend.mode.set(Mode::Long);
            let player = Rc::new(Player::default());
            let (mut worker, mut events) = make(&root, backend, player, 64);
            assert!(matches!(
                send(
                    &mut worker,
                    Some("chapter"),
                    Command::Start(Box::new(input(true, PlanPlayback::AfterChapterReady)))
                )
                .await
                .event,
                Event::Accepted
            ));
            assert!(matches!(
                send(&mut worker, Some("chapter"), Command::Pause)
                    .await
                    .event,
                Event::Accepted
            ));
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    while events.try_recv().is_ok() {}
                    let p = snapshot(&mut worker, "chapter").await;
                    if p.chapter_ready {
                        assert_eq!(p.generated_end, 9);
                        assert_eq!(p.played_end, 0);
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            // Owner closure must bypass observer pressure, including both bounded queues.
            worker.manager.as_mut().unwrap().close().await.unwrap();
        })
        .await;
    }
    #[cfg(feature = "moss")]
    #[tokio::test]
    async fn default_voice_update_does_not_replace_or_change_current_plan() {
        talechime::run_local(async {
            let root = tempfile::tempdir().unwrap();
            let (mut worker, _events) = make(
                &root,
                Rc::new(Fixture::default()),
                Rc::new(Player::default()),
                64,
            );
            let caps = worker.resources.capabilities("moss").unwrap();
            worker.store = worker
                .store
                .clone()
                .with_defaults(talechime_protocol::Config {
                    backend: caps.backend,
                    model: caps.model,
                    voice: caps.default_voice.clone(),
                    ..Default::default()
                });
            send(
                &mut worker,
                Some("open"),
                Command::Start(Box::new(input(false, PlanPlayback::Streaming))),
            )
            .await;
            let response = send(
                &mut worker,
                None,
                Command::UpdateConfig(talechime_protocol::ConfigPatch {
                    voice: Some(caps.default_voice),
                    ..Default::default()
                }),
            )
            .await;
            assert!(matches!(response.event, Event::ConfigChanged(_)));
            assert!(worker.store.path().exists());
            assert_eq!(worker.status().await.0.as_deref(), Some("open"));
            assert_eq!(snapshot(&mut worker, "open").await.accepted_end, 0);
            worker.manager.as_mut().unwrap().close().await.unwrap();
        })
        .await;
    }
    #[tokio::test]
    async fn input_failure_and_batch_limits_are_explicit() {
        talechime::run_local(async {
            let root = tempfile::tempdir().unwrap();
            let (mut worker, mut events) = make(
                &root,
                Rc::new(Fixture::default()),
                Rc::new(Player::default()),
                64,
            );
            send(
                &mut worker,
                Some("open"),
                Command::Start(Box::new(input(false, PlanPlayback::Streaming))),
            )
            .await;
            let oversized = vec![span(0, 3, "A"); talechime_protocol::MAX_PLAN_BATCH_SPANS + 1];
            assert!(matches!(
                send(
                    &mut worker,
                    Some("open"),
                    Command::Append { spans: oversized }
                )
                .await
                .event,
                Event::Error(_)
            ));
            assert_eq!(snapshot(&mut worker, "open").await.accepted_end, 0);
            assert!(matches!(
                send(
                    &mut worker,
                    Some("open"),
                    Command::FailInput {
                        message: "analysis failed".into()
                    }
                )
                .await
                .event,
                Event::Accepted
            ));
            let reason = tokio::time::timeout(Duration::from_secs(2), async {
                loop {
                    let e = events.recv().await.unwrap();
                    if let Event::SessionEnded { reason, .. } = e.event {
                        break reason;
                    }
                }
            })
            .await
            .unwrap();
            assert_eq!(reason, talechime_protocol::EndReason::Failed);
            assert!(matches!(
                send(&mut worker, Some("open"), Command::Seal).await.event,
                Event::Error(_)
            ));
            worker.shutdown().await.unwrap();
        })
        .await;
    }
}
