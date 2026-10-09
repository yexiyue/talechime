use crate::{engine::Engine, *};
use std::{path::PathBuf, rc::Rc};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;
use tts_core::{AudioPlayer, checkpoint::CheckpointStore, session::SessionManager};

/// Listening-only assembly; no persistence by default.
#[derive(Debug, Clone)]
pub struct ListeningOptions {
    /// Optional directory for actual-playback checkpoints. None performs no checkpoint I/O.
    pub checkpoints: Option<PathBuf>,
    /// Library-owned event queue, 1..=4096. Ignored with a host-provided sender.
    /// Consume events concurrently with controls.
    pub event_capacity: usize,
}
impl Default for ListeningOptions {
    fn default() -> Self {
        Self {
            checkpoints: None,
            event_capacity: 64,
        }
    }
}

type Reply<T> = oneshot::Sender<Result<T, SessionError>>;
enum Control {
    Start(String, Box<SpeechPlan>, PlanSessionOptions, Reply<()>),
    Append(String, Vec<SpeechSpan>, Reply<()>),
    Seal(String, Reply<()>),
    Fail(String, String, Reply<()>),
    Pause(String, Reply<()>),
    Resume(String, Reply<()>),
    Seek(String, String, usize, Reply<()>),
    Configure(String, f32, f32, Reply<()>),
    Progress(String, Reply<PlanProgress>),
    Stop(Reply<()>),
    Status(Reply<(Option<String>, SessionState)>),
}

/// Send + Sync controls through a bounded queue; each operation has an acknowledgement.
/// Execution IDs and validation are enforced by the shared SessionManager.
#[derive(Clone)]
pub struct ListeningHandle {
    commands: mpsc::Sender<Control>,
}
impl ListeningHandle {
    async fn request<T>(&self, build: impl FnOnce(Reply<T>) -> Control) -> Result<T, EngineError> {
        let (reply, received) = oneshot::channel();
        self.commands
            .send(build(reply))
            .await
            .map_err(|_| EngineError::Closed)?;
        received
            .await
            .map_err(|_| EngineError::Closed)?
            .map_err(EngineError::Session)
    }
    /// Start/replace an execution after validating its fixed plan.
    pub async fn start(
        &self,
        id: impl Into<String>,
        plan: SpeechPlan,
        options: PlanSessionOptions,
    ) -> Result<(), EngineError> {
        self.request(|reply| Control::Start(id.into(), Box::new(plan), options, reply))
            .await
    }
    /// Atomically accept a contiguous batch without waiting for synthesis.
    pub async fn append(
        &self,
        id: impl Into<String>,
        spans: Vec<SpeechSpan>,
    ) -> Result<(), EngineError> {
        self.request(|reply| Control::Append(id.into(), spans, reply))
            .await
    }
    /// Require complete coverage and seal input; not generation/playback completion.
    pub async fn seal(&self, id: impl Into<String>) -> Result<(), EngineError> {
        self.request(|reply| Control::Seal(id.into(), reply)).await
    }
    /// Explicit analysis failure cancels generation/playback and produces Failed.
    pub async fn fail_input(
        &self,
        id: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), EngineError> {
        self.request(|reply| Control::Fail(id.into(), message.into(), reply))
            .await
    }
    /// Pause actual playback without conflating it with generation progress.
    pub async fn pause(&self, id: impl Into<String>) -> Result<(), EngineError> {
        self.request(|reply| Control::Pause(id.into(), reply)).await
    }
    /// Resume subject to the shared prebuffer policy.
    pub async fn resume(&self, id: impl Into<String>) -> Result<(), EngineError> {
        self.request(|reply| Control::Resume(id.into(), reply))
            .await
    }
    /// Replace with a new ID, preserving voice assignments, staging settings and pause.
    pub async fn seek(
        &self,
        old: impl Into<String>,
        new: impl Into<String>,
        byte: usize,
    ) -> Result<(), EngineError> {
        self.request(|reply| Control::Seek(old.into(), new.into(), byte, reply))
            .await
    }
    /// Adjust only volume/speed, independently of voice/style configuration.
    pub async fn configure_playback(
        &self,
        id: impl Into<String>,
        volume: f32,
        speed: f32,
    ) -> Result<(), EngineError> {
        self.request(|reply| Control::Configure(id.into(), volume, speed, reply))
            .await
    }
    /// Query accepted/generated/played boundaries on the execution owner.
    pub async fn progress(&self, id: impl Into<String>) -> Result<PlanProgress, EngineError> {
        self.request(|reply| Control::Progress(id.into(), reply))
            .await
    }
    /// Read the current execution ID and shared session state.
    pub async fn status(&self) -> Result<(Option<String>, SessionState), EngineError> {
        self.request(Control::Status).await
    }
    /// Cancel the current execution, keep the prepared listening owner reusable.
    pub async fn stop(&self) -> Result<(), EngineError> {
        self.request(Control::Stop).await
    }
}

/// Local listening owner with a library-owned bounded event receiver.
/// Consume events concurrently with controls; close waits for cleanup.
pub struct Listening {
    session: ListeningSession,
    events: mpsc::Receiver<SessionEvent>,
}
impl Listening {
    /// Clone controls; models/player remain on the local owner.
    pub fn control(&self) -> ListeningHandle {
        self.session.control()
    }
    /// Receive ordered session events.
    pub async fn recv(&mut self) -> Option<SessionEvent> {
        self.events.recv().await
    }
    /// Close event delivery and await execution cleanup. Pending events are discarded.
    pub async fn close(&mut self) -> Result<(), EngineError> {
        self.events.close();
        while self.events.try_recv().is_ok() {}
        self.session.close().await
    }
}

/// Local execution owner when a host supplies its own bounded event queue.
/// close bypasses observer pressure; the host retains ownership of queued events.
pub struct ListeningSession {
    handle: ListeningHandle,
    closing: CancellationToken,
    task: Option<JoinHandle<Result<(), SessionError>>>,
}
impl ListeningSession {
    /// Clone Send controls without moving the local model/player.
    pub fn control(&self) -> ListeningHandle {
        self.handle.clone()
    }
    /// Cancel and await owned tasks and I/O without waiting for the event consumer.
    pub async fn close(&mut self) -> Result<(), EngineError> {
        self.closing.cancel();
        if let Some(task) = &mut self.task {
            let result = (&mut *task).await;
            self.task.take();
            result??;
        }
        Ok(())
    }
}
impl Drop for ListeningSession {
    fn drop(&mut self) {
        // Detach cleanup, retaining the Engine reservation until the local owner exits.
        self.closing.cancel();
    }
}
impl Engine {
    /// Open the default output only when listening is requested.
    pub fn listen(&self, options: ListeningOptions) -> Result<Listening, EngineError> {
        self.available()?;
        validate(&options)?;
        let player = Rc::new(AudioPlayer::open().map_err(EngineError::Output)?);
        self.listen_with(player, options)
    }
    /// Inject a host/test player; otherwise use the same listening assembly and controls.
    pub fn listen_with(
        &self,
        player: Rc<dyn Playback>,
        options: ListeningOptions,
    ) -> Result<Listening, EngineError> {
        self.available()?;
        validate(&options)?;
        let (events, receiver) = mpsc::channel(options.event_capacity);
        let session = self.listen_with_events(player, options, events)?;
        Ok(Listening {
            session,
            events: receiver,
        })
    }
    /// Open default audio and send directly to the host's bounded event queue.
    /// The host must drain that queue concurrently with controls.
    pub fn listen_to(
        &self,
        options: ListeningOptions,
        events: mpsc::Sender<SessionEvent>,
    ) -> Result<ListeningSession, EngineError> {
        self.available()?;
        validate_checkpoints(&options)?;
        let player = Rc::new(AudioPlayer::open().map_err(EngineError::Output)?);
        self.listen_with_events(player, options, events)
    }
    /// Inject playback and a host-owned queue; no forwarding task or second queue.
    /// event_capacity is used only by listen/listen_with; here the host chooses capacity.
    pub fn listen_with_events(
        &self,
        player: Rc<dyn Playback>,
        options: ListeningOptions,
        events: mpsc::Sender<SessionEvent>,
    ) -> Result<ListeningSession, EngineError> {
        self.available()?;
        validate_checkpoints(&options)?;
        if events.max_capacity() > 4096 {
            return Err(EngineError::InvalidOptions(
                "host event capacity exceeds 4096",
            ));
        }
        let owner = Rc::new(());
        let mut manager = SessionManager::with_optional_checkpoints(
            self.backend()?,
            player,
            options.checkpoints.map(CheckpointStore::new),
            events.clone(),
        );
        let (commands, mut requests) = mpsc::channel(16);
        let closing = CancellationToken::new();
        let closed = closing.clone();
        self.reserve_listening(&owner);
        let task = tokio::task::spawn_local(async move {
            let reservation = owner;
            loop {
                let request = tokio::select! {
                    _ = closed.cancelled() => break,
                    _ = events.closed() => break,
                    request = requests.recv() => match request { Some(request) => request, None => break },
                };
                tokio::select! {
                    _ = closed.cancelled() => break,
                    _ = events.closed() => break,
                    _ = apply(&mut manager, request) => {},
                }
            }
            let result = manager.close().await;
            drop(manager);
            drop(reservation);
            result
        });
        Ok(ListeningSession {
            handle: ListeningHandle { commands },
            closing,
            task: Some(task),
        })
    }
}
fn validate(options: &ListeningOptions) -> Result<(), EngineError> {
    if !(1..=4096).contains(&options.event_capacity) {
        return Err(EngineError::InvalidOptions(
            "event capacity must be 1..=4096",
        ));
    }
    validate_checkpoints(options)
}
fn validate_checkpoints(options: &ListeningOptions) -> Result<(), EngineError> {
    if options
        .checkpoints
        .as_ref()
        .is_some_and(|path| path.as_os_str().is_empty())
    {
        return Err(EngineError::InvalidOptions(
            "checkpoint path must be nonempty",
        ));
    }
    Ok(())
}
async fn apply(manager: &mut SessionManager, control: Control) {
    // A caller dropping its acknowledgement does not roll back an accepted operation.
    match control {
        Control::Start(id, plan, options, reply) => {
            let _ = reply.send(manager.start_plan(id, *plan, options).await);
        }
        Control::Append(id, spans, reply) => {
            let _ = reply.send(manager.append_plan(&id, spans));
        }
        Control::Seal(id, reply) => {
            let _ = reply.send(manager.seal_plan(&id));
        }
        Control::Fail(id, message, reply) => {
            let _ = reply.send(manager.fail_input(&id, &message).await);
        }
        Control::Pause(id, reply) => {
            let _ = reply.send(manager.pause(&id).await);
        }
        Control::Resume(id, reply) => {
            let _ = reply.send(manager.resume(&id).await);
        }
        Control::Seek(old, new, byte, reply) => {
            let _ = reply.send(manager.seek_plan(&old, new, byte).await);
        }
        Control::Configure(id, volume, speed, reply) => {
            let _ = reply.send(manager.configure_playback(&id, volume, speed));
        }
        Control::Progress(id, reply) => {
            let _ = reply.send(manager.plan_progress(&id));
        }
        Control::Status(reply) => {
            let _ = reply.send(Ok(manager.status()));
        }
        Control::Stop(reply) => {
            let _ = reply.send(manager.stop().await);
        }
    }
}
