//! Worker application state, independent of JSON Lines transport.
use crate::resources::Resources;
use std::rc::Rc;
use std::time::Duration;
use tokio::{sync::mpsc, task::JoinHandle};
use tts_core::{
    checkpoint::CheckpointStore,
    config::{ConfigError, ConfigStore},
    session::SessionEvent,
};
use tts_protocol::{Command, ErrorInfo, Event, Request, SessionState};

type Preparation = JoinHandle<anyhow::Result<Rc<dyn tts_core::backend::Backend>>>;

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

pub struct Worker {
    store: ConfigStore,
    checkpoints: CheckpointStore,
    resources: Resources,
    events: mpsc::Sender<SessionEvent>,
    progress: mpsc::Sender<Event>,
    preparing: Option<Preparation>,
    progress_forwarder: Option<JoinHandle<()>>,
    manager: Option<crate::app_session::AppSession>,
    plan_spans: usize,
    resource_state: SessionState,
}
impl Worker {
    pub fn new(
        store: ConfigStore,
        checkpoints: CheckpointStore,
        resources: Resources,
        events: mpsc::Sender<SessionEvent>,
        progress: mpsc::Sender<Event>,
    ) -> Self {
        Self {
            store,
            checkpoints,
            resources,
            events,
            progress,
            preparing: None,
            progress_forwarder: None,
            manager: None,
            plan_spans: 0,
            resource_state: SessionState::Idle,
        }
    }
    #[cfg(test)]
    pub fn attach_test(
        &mut self,
        backend: Rc<dyn tts_core::backend::Backend>,
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
    pub fn catalog(&self) -> anyhow::Result<Vec<tts_protocol::Capabilities>> {
        self.resources.catalog()
    }
    #[cfg(any(
        feature = "moss",
        feature = "qwen",
        feature = "voxcpm",
        feature = "omnivoice",
    ))]
    pub fn unprepared_device_status(&self) -> anyhow::Result<Event> {
        let config = self.store.load()?;
        Ok(self.resources.device_status_for(
            &config.backend,
            config.model.as_deref(),
            tts_protocol::Device::Auto,
            Some("resources not prepared".into()),
        ))
    }
    pub fn is_preparing(&self) -> bool {
        self.preparing.is_some()
    }
    #[cfg(any(
        feature = "moss",
        feature = "qwen",
        feature = "voxcpm",
        feature = "omnivoice",
    ))]
    pub fn has_prepared_model(&self) -> bool {
        self.manager.is_some()
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
    /// Poll only while preparation exists; dropping this future keeps the task owned.
    pub async fn prepared(&mut self) -> Response {
        let Some(task) = &mut self.preparing else {
            return Response::error("prepare_failed", "model", "no preparation is running");
        };
        let loaded = task.await;
        self.preparing.take();
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
        match crate::app_session::AppSession::open(prepared, &self.checkpoints, self.events.clone())
        {
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
    async fn cancel_prepare(&mut self) {
        if let Some(task) = self.preparing.take() {
            task.abort();
            let _ = task.await;
            self.disconnect_progress();
        }
        self.resource_state = SessionState::Idle;
    }
    fn disconnect_progress(&mut self) {
        if let Some(task) = self.progress_forwarder.take() {
            task.abort();
        }
    }
    pub async fn command(&mut self, request: &Request) -> Response {
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
                let old = match self.store.load() {
                    Ok(config) => config,
                    Err(error) => return Response::error("config_invalid", "config", error),
                };
                let target = patch.backend.as_deref().unwrap_or(&old.backend);
                let caps = match self
                    .resources
                    .capabilities_for(target, patch.target_model(&old))
                {
                    Ok(caps) => caps,
                    Err(error) => return Response::error("backend_unavailable", "config", error),
                };
                #[cfg(any(
                    feature = "moss",
                    feature = "qwen",
                    feature = "voxcpm",
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
                let config = match self.store.update(patch, &caps) {
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
                    self.cancel_prepare().await;
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
                if self.preparing.is_none() {
                    let config = match self.store.load() {
                        Ok(config) => config,
                        Err(error) => return Response::error("config_invalid", "config", error),
                    };
                    let caps = match self
                        .resources
                        .capabilities_for(&config.backend, config.model.as_deref())
                    {
                        Ok(caps) => caps,
                        Err(error) => {
                            return Response::error("backend_unavailable", "model", error);
                        }
                    };
                    let config = match self.store.initialize(&caps) {
                        Ok(config) => config,
                        Err(error) => return Response::error("config_invalid", "config", error),
                    };
                    let resources = self.resources.clone();
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
                    self.preparing = Some(tokio::task::spawn_local(async move {
                        crate::preparation::prepare(resources, config, progress).await
                    }));
                    self.resource_state = SessionState::Preparing;
                }
                Response::global(Event::Accepted)
            }
            Command::CancelPrepare => {
                self.cancel_prepare().await;
                Response::global(Event::Accepted)
            }
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
                let plan = match crate::plan_input::build(input, &caps) {
                    Ok(plan) => plan,
                    Err(error) => return Response::error("plan_invalid", "plan", error),
                };
                let config = match self.store.load() {
                    Ok(config) => config,
                    Err(error) => return Response::error("config_invalid", "config", error),
                };
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
                        tts_core::session::SessionError::Invalid("stale session".into()),
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
        self.cancel_prepare().await;
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
        if let Some(task) = &self.preparing {
            task.abort();
        }
    }
}

#[cfg(all(test, feature = "moss"))]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc, sync::Arc};
    use tts_core::{
        backend::{Backend, BackendError, Pcm, Streaming},
        player::Playback,
    };
    struct SilentBackend(tts_protocol::Capabilities);
    impl Backend for SilentBackend {
        fn capabilities(&self) -> tts_protocol::Capabilities {
            self.0.clone()
        }
        fn stream<'a>(&'a self, _: &'a str, _: &'a str) -> Streaming<'a> {
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
        let resources = Resources::new(Some(root.path().join("models"))).unwrap();
        let caps = resources.capabilities("moss").unwrap();
        let checkpoints = CheckpointStore::new(root.path().join("checkpoints"));
        std::fs::create_dir_all(root.path().join("checkpoints")).unwrap();
        let progress_file = root.path().join("checkpoints/existing.json");
        std::fs::write(&progress_file, "reliable progress").unwrap();
        let (events, _) = mpsc::channel(32);
        let (progress, _) = mpsc::channel(32);
        let player = Rc::new(IdlePlayer::default());
        let mut worker = Worker::new(
            ConfigStore::new(root.path().join("config.json")),
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
                protocol_version: tts_protocol::PROTOCOL_VERSION,
                request_id: "device".into(),
                session_id: None,
                command: Command::UpdateConfig(tts_protocol::ConfigPatch {
                    tts_device: Some(tts_protocol::Device::Cpu),
                    ..Default::default()
                }),
            })
            .await;
        assert!(
            matches!(response.event, Event::ConfigChanged(config) if config.tts_device == tts_protocol::Device::Cpu)
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
    #[cfg(feature = "moss-candle")]
    #[tokio::test]
    async fn model_change_rejects_inherited_device_before_config_or_manager_changes() {
        talechime::run_local(async {

        let root = tempfile::tempdir().unwrap();
        let resources = Resources::new(Some(root.path().join("models"))).unwrap();
        let defaults = tts_protocol::Config {
            model: Some("nano".into()),
            tts_device: tts_protocol::Device::Cpu,
            ..Default::default()
        };
        let store = ConfigStore::new(root.path().join("config.json")).with_defaults(defaults);
        let old = store
            .initialize(&resources.capabilities("moss").unwrap())
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
                Rc::new(SilentBackend(resources.capabilities("moss").unwrap())),
                player.clone(),
                &checkpoints,
                events,
            )
            .unwrap(),
        );
        let response = worker
            .command(&Request {
                protocol_version: tts_protocol::PROTOCOL_VERSION,
                request_id: "model".into(),
                session_id: None,
                command: Command::UpdateConfig(tts_protocol::ConfigPatch {
                    expected_revision: old.revision,
                    model: Some("local-1.7b".into()),
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
    use tts_protocol::{PlanPlayback, PlanRequest, SourceId, TextRange, VoiceSpan, text_hash};

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
            protocol_version: tts_protocol::PROTOCOL_VERSION,
            request_id: "test".into(),
            session_id: id.map(str::to_owned),
            command,
        };
        let request = tts_protocol::decode(&tts_protocol::encode(&request).unwrap()).unwrap();
        worker.command(&request).await
    }
    fn make(
        root: &tempfile::TempDir,
        backend: Rc<Fixture>,
        player: Rc<Player>,
        capacity: usize,
    ) -> (Worker, mpsc::Receiver<SessionEvent>) {
        let resources = Resources::new(Some(root.path().join("models"))).unwrap();
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
    async fn snapshot(worker: &mut Worker, id: &str) -> tts_protocol::PlanProgressSnapshot {
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
                            reason: tts_protocol::EndReason::Completed,
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
            send(
                &mut worker,
                Some("open"),
                Command::Start(Box::new(input(false, PlanPlayback::Streaming))),
            )
            .await;
            let response = send(
                &mut worker,
                None,
                Command::UpdateConfig(tts_protocol::ConfigPatch {
                    voice: Some("Weiguo".into()),
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
            let oversized = vec![span(0, 3, "A"); tts_protocol::MAX_PLAN_BATCH_SPANS + 1];
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
            assert_eq!(reason, tts_protocol::EndReason::Failed);
            assert!(matches!(
                send(&mut worker, Some("open"), Command::Seal).await.event,
                Event::Error(_)
            ));
            worker.shutdown().await.unwrap();
        })
        .await;
    }
}
