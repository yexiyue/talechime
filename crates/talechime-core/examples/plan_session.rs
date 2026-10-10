//! Execute an incremental plan with deterministic model/audio substitutes.
//! No model download, real audio device or user configuration is used.
use std::{cell::Cell, rc::Rc, sync::Arc, time::Duration};
use talechime_core::{
    PlanSessionOptions, Playback, PlaybackPolicy, SourceSnapshot, SpeechPlan, SpeechSpan,
    VoiceSnapshot,
    backend::{AudioChunk, Backend, Pcm, Streaming},
    checkpoint::CheckpointStore,
    session::SessionManager,
};
use tokio::sync::mpsc;
use tts_protocol::{Capabilities, EndReason, Event, SourceId, TextRange, text_hash};

struct DemoBackend;
impl Backend for DemoBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            backend: "demo".into(),
            model: None,
            model_name: "Deterministic fixture".into(),
            compiled_devices: vec![],
            default_voice: "A".into(),
            voice_names: Default::default(),
            voices: vec!["A".into(), "B".into()],
            native_streaming: false,
            style: false,
            cloning: false,
            pronunciation: false,
            continuation: false,
            parameters: vec![],
        }
    }
    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            let (text, voice) = (request.text, request.voice);
            println!("generate {voice}: {text}");
            let (tx, rx) = mpsc::channel(2);
            tx.send(Ok(AudioChunk::Pcm(Pcm {
                samples: vec![0.1; 4000],
                sample_rate: 1000,
                channels: 1,
            })))
            .await
            .map_err(|_| {
                talechime_core::backend::BackendError::Synthesis("receiver closed".into())
            })?;
            tx.send(Ok(AudioChunk::End)).await.map_err(|_| {
                talechime_core::backend::BackendError::Synthesis("receiver closed".into())
            })?;
            Ok(rx)
        })
    }
}

#[derive(Default)]
struct DemoPlayer {
    queued: Cell<Duration>,
    cursor: Cell<Duration>,
    paused: Cell<bool>,
}
impl Playback for DemoPlayer {
    fn append(&self, pcm: Arc<Pcm>) {
        self.queued.set(
            self.queued.get()
                + Duration::from_secs_f64(pcm.samples.len() as f64 / pcm.sample_rate as f64),
        );
    }
    fn position(&self) -> Duration {
        if !self.paused.get() {
            self.cursor.set(self.queued.get());
        }
        self.cursor.get()
    }
    fn is_empty(&self) -> bool {
        self.position() >= self.queued.get()
    }
    fn pause(&self) {
        self.paused.set(true);
    }
    fn resume(&self) {
        self.paused.set(false);
    }
    fn stop(&self) {
        self.pause();
        self.queued.set(Duration::ZERO);
        self.cursor.set(Duration::ZERO);
    }
    fn configure(&self, _: f32, _: f32) {}
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tokio::task::LocalSet::new()
        .run_until(async {
            let directory = tempfile::tempdir()?;
            let backend = Rc::new(DemoBackend);
            let caps = backend.capabilities();
            let text = "甲乙丙";
            let plan = SpeechPlan::new(
                SourceSnapshot::new(
                    SourceId {
                        namespace: "demo".into(),
                        book: "book".into(),
                        chapter: "one".into(),
                    },
                    text,
                    &text_hash(text),
                )?,
                VoiceSnapshot::new("demo", None, &caps, caps.voices.clone())?,
                if std::env::args().any(|arg| arg == "--after-chapter") {
                    PlaybackPolicy::AfterChapterReady
                } else {
                    PlaybackPolicy::Streaming
                },
            );
            let (tx, mut events) = mpsc::channel::<talechime_core::session::SessionEvent>(16);
            // Consume bounded events concurrently with control operations.
            let consumer = tokio::task::spawn_local(async move {
                while let Some(event) = events.recv().await {
                    match event.event {
                        Event::SegmentStarted { range, .. } => {
                            println!("play source {}..{}", range.start, range.end)
                        }
                        Event::SessionEnded { reason, .. } => return reason,
                        _ => {}
                    }
                }
                EndReason::Failed
            });
            let mut manager = SessionManager::new(
                backend,
                Rc::new(DemoPlayer::default()),
                CheckpointStore::new(directory.path()),
                tx,
            );
            manager
                .start_plan("demo".into(), plan, PlanSessionOptions::default())
                .await?;
            manager.append_plan(
                "demo",
                vec![SpeechSpan::new(TextRange { start: 0, end: 3 }, "A", None)],
            )?;
            tokio::time::sleep(Duration::from_millis(50)).await;
            println!("prefix progress: {:?}", manager.plan_progress("demo")?);
            manager.append_plan(
                "demo",
                vec![
                    SpeechSpan::new(TextRange { start: 3, end: 6 }, "B", None),
                    SpeechSpan::new(TextRange { start: 6, end: 9 }, "A", None),
                ],
            )?;
            manager.seal_plan("demo")?;
            let reason = consumer.await?;
            println!("terminal: {reason:?}");
            manager.stop().await?;
            Ok::<_, Box<dyn std::error::Error>>(())
        })
        .await
}
