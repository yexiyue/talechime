//! Thin application assembly using the same public library as embedded hosts.
use std::rc::Rc;
use talechime::{Engine, ListeningHandle, ListeningOptions, ListeningSession};
use tokio::sync::mpsc;
use tts_core::{backend::Backend, checkpoint::CheckpointStore, session::SessionEvent};

pub struct AppSession {
    pub control: ListeningHandle,
    engine: Engine,
    session: ListeningSession,
}
impl AppSession {
    pub fn open(
        backend: Rc<dyn Backend>,
        checkpoints: &CheckpointStore,
        events: mpsc::Sender<SessionEvent>,
        verifier: Option<Rc<talechime::Verifier>>,
    ) -> anyhow::Result<Self> {
        let mut engine = Engine::from_backend(backend);
        if let Some(verifier) = verifier {
            engine.set_verifier(verifier)?;
        }
        let session = engine.listen_to(
            ListeningOptions {
                checkpoints: Some(checkpoints.directory().into()),
                ..Default::default()
            },
            events,
        )?;
        Ok(Self::assemble(engine, session))
    }
    #[cfg(test)]
    pub fn with_player(
        backend: Rc<dyn Backend>,
        player: Rc<dyn talechime::Playback>,
        checkpoints: &CheckpointStore,
        events: mpsc::Sender<SessionEvent>,
    ) -> anyhow::Result<Self> {
        let engine = Engine::from_backend(backend);
        let session = engine.listen_with_events(
            player,
            ListeningOptions {
                checkpoints: Some(checkpoints.directory().into()),
                ..Default::default()
            },
            events,
        )?;
        Ok(Self::assemble(engine, session))
    }
    fn assemble(engine: Engine, session: ListeningSession) -> Self {
        Self {
            control: session.control(),
            engine,
            session,
        }
    }
    pub fn capabilities(&self) -> Result<talechime::Capabilities, talechime::EngineError> {
        self.engine.capabilities()
    }
    /// Stop reliably while the host drains events, then await owner cleanup.
    pub async fn stop_and_close(&mut self) -> anyhow::Result<()> {
        let stopped = self.control.stop().await;
        self.close().await?;
        match stopped {
            Ok(()) | Err(talechime::EngineError::Closed) => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
    pub async fn close(&mut self) -> anyhow::Result<()> {
        self.session.close().await?;
        self.engine.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Fixture, Player};
    #[tokio::test]
    async fn administrative_close_bypasses_full_host_event_queue() {
        talechime::run_local(async {
            let root = tempfile::tempdir().unwrap();
            let (events, _blocked) = mpsc::channel(1);
            events
                .send(SessionEvent {
                    session_id: "occupied".into(),
                    event: tts_protocol::Event::Accepted,
                })
                .await
                .unwrap();
            let mut owner = AppSession::with_player(
                Rc::new(Fixture::default()),
                Rc::new(Player::default()),
                &CheckpointStore::new(root.path()),
                events,
            )
            .unwrap();
            let source = talechime::SourceSnapshot::new(
                tts_protocol::SourceId {
                    namespace: "test".into(),
                    book: "book".into(),
                    chapter: "chapter".into(),
                },
                "甲",
                &tts_protocol::text_hash("甲"),
            )
            .unwrap();
            let caps = owner.capabilities().unwrap();
            let voices = talechime::VoiceSnapshot::new(
                &caps.backend,
                caps.model.as_deref(),
                &caps,
                vec!["A".into()],
            )
            .unwrap();
            owner
                .control
                .start(
                    "open",
                    talechime::SpeechPlan::new(
                        source,
                        voices,
                        talechime::PlaybackPolicy::Streaming,
                    ),
                    Default::default(),
                )
                .await
                .unwrap();
            let controls = owner.control.clone();
            let saturate = tokio::task::spawn_local(async move {
                for _ in 0..100 {
                    if controls.pause("open").await.is_err() {
                        break;
                    }
                }
            });
            for _ in 0..100 {
                tokio::task::yield_now().await;
            }
            tokio::time::timeout(std::time::Duration::from_secs(2), owner.close())
                .await
                .unwrap()
                .unwrap();
            saturate.await.unwrap();
            owner.close().await.unwrap();
        })
        .await;
    }
}
