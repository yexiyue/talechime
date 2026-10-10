use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::Arc,
    time::Duration,
};
use talechime_core::{
    PlanSessionOptions, PlanState, Playback, PlaybackPolicy, SourceSnapshot, SpeechPlan,
    SpeechSpan, VoiceSnapshot,
    backend::{AudioChunk, Backend, Pcm, Segmentation, Streaming},
    checkpoint::CheckpointStore,
    session::{SessionEvent, SessionManager},
    text::TextSegment,
};
use tokio::sync::mpsc;
use tts_protocol::{Capabilities, Config, EndReason, Event, SourceId, TextRange, text_hash};

#[derive(Clone, Copy)]
enum Mode {
    Normal,
    Long,
    Tiny,
    Disconnect,
    Omit,
    InvalidRange,
}

#[derive(Debug, PartialEq)]
struct Call {
    text: String,
    voice: String,
    style: Option<String>,
    seed: u64,
    params: Vec<(String, tts_protocol::ParamValue)>,
}

/// Seed and params vary per policy; existing assertions cover text/voice/style.
fn call_shapes(calls: &[Call]) -> Vec<(&str, &str, Option<&str>)> {
    calls
        .iter()
        .map(|call| {
            (
                call.text.as_str(),
                call.voice.as_str(),
                call.style.as_deref(),
            )
        })
        .collect()
}

struct FixtureBackend {
    caps: RefCell<Capabilities>,
    calls: RefCell<Vec<Call>>,
    contexts: RefCell<Vec<Option<String>>>,
    selected: RefCell<Vec<String>>,
    observed: RefCell<Vec<String>>,
    mode: Cell<Mode>,
    chunks: Rc<Cell<usize>>,
    running: Rc<Cell<usize>>,
}

impl FixtureBackend {
    fn new() -> Self {
        Self {
            caps: RefCell::new(Capabilities {
                backend: "fixture".into(),
                model: Some("shared".into()),
                model_name: String::new(),
                compiled_devices: vec![],
                default_voice: "A".into(),
                voice_names: Default::default(),
                voices: vec!["A".into(), "B".into()],
                native_streaming: true,
                style: true,
                cloning: false,
                pronunciation: false,
                continuation: false,
                parameters: vec![],
            }),
            calls: RefCell::new(vec![]),
            contexts: RefCell::new(vec![]),
            selected: RefCell::new(vec![]),
            observed: RefCell::new(vec![]),
            mode: Cell::new(Mode::Normal),
            chunks: Rc::new(Cell::new(0)),
            running: Rc::new(Cell::new(0)),
        }
    }
}

impl Backend for FixtureBackend {
    fn capabilities(&self) -> Capabilities {
        self.caps.borrow().clone()
    }
    fn select_voice(&self, voice: &str) {
        self.selected.borrow_mut().push(voice.into());
    }
    fn observe_duration(&self, _: &str, voice: &str, _: f64) {
        self.observed.borrow_mut().push(voice.into());
    }
    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            let (style, continuation) = {
                let caps = self.caps.borrow();
                (caps.style, caps.continuation)
            };
            request.reject_unsupported(style, continuation)?;
            self.calls.borrow_mut().push(Call {
                text: request.text.into(),
                voice: request.voice.into(),
                style: request.style.map(str::to_owned),
                seed: request.seed,
                params: request
                    .params
                    .iter()
                    .map(|(name, value)| (name.to_owned(), value.clone()))
                    .collect(),
            });
            self.contexts
                .borrow_mut()
                .push(request.context.map(|c| c.text().into()));
            let (tx, rx) = mpsc::channel(1);
            let mode = self.mode.get();
            let chunks = self.chunks.clone();
            let running = self.running.clone();
            running.set(running.get() + 1);
            tokio::task::spawn_local(async move {
                let (count, samples) = if matches!(mode, Mode::Long) {
                    (100, 400)
                } else if matches!(mode, Mode::Tiny) {
                    (1, 10)
                } else {
                    (1, 4000)
                };
                for _ in 0..count {
                    if tx
                        .send(Ok(AudioChunk::Pcm(Pcm {
                            samples: vec![0.1; samples],
                            sample_rate: 1000,
                            channels: 1,
                        })))
                        .await
                        .is_err()
                    {
                        break;
                    }
                    chunks.set(chunks.get() + 1);
                }
                if !matches!(mode, Mode::Disconnect) {
                    let _ = tx.send(Ok(AudioChunk::End)).await;
                }
                running.set(running.get() - 1);
            });
            Ok(rx)
        })
    }
    fn segments<'a>(&'a self, text: &'a str) -> Segmentation<'a> {
        Box::pin(async move {
            Ok(match self.mode.get() {
                Mode::Omit => vec![],
                Mode::InvalidRange => vec![TextSegment {
                    text: text.into(),
                    start: 0,
                    end: text.len() + 1,
                }],
                _ => talechime_core::text::preprocess_text(text, 200),
            })
        })
    }
}

#[derive(Default)]
struct ClockPlayer {
    queued: Cell<Duration>,
    cursor: Cell<Duration>,
    paused: Cell<bool>,
    settings: Cell<(f32, f32)>,
}

impl ClockPlayer {
    fn advance(&self) {
        if !self.paused.get() {
            self.cursor.set(self.queued.get());
        }
    }
}
impl Playback for ClockPlayer {
    fn append(&self, pcm: Arc<Pcm>) {
        self.queued
            .set(self.queued.get() + Duration::from_millis(u64::from(pcm.duration_ms().unwrap())));
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
        self.pause();
        self.queued.set(Duration::ZERO);
        self.cursor.set(Duration::ZERO);
    }
    fn configure(&self, volume: f32, speed: f32) {
        self.settings.set((volume, speed));
    }
}

struct Harness {
    manager: SessionManager,
    backend: Rc<FixtureBackend>,
    player: Rc<ClockPlayer>,
    rx: mpsc::Receiver<SessionEvent>,
    events: Vec<SessionEvent>,
    checkpoints: CheckpointStore,
    _directory: tempfile::TempDir,
}

impl Harness {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let checkpoints = CheckpointStore::new(directory.path());
        let backend = Rc::new(FixtureBackend::new());
        let player = Rc::new(ClockPlayer::default());
        let (tx, rx) = mpsc::channel(256);
        Self {
            manager: SessionManager::new(backend.clone(), player.clone(), checkpoints.clone(), tx),
            backend,
            player,
            rx,
            events: vec![],
            checkpoints,
            _directory: directory,
        }
    }
    fn plan(&self, text: &str) -> SpeechPlan {
        let caps = self.backend.capabilities();
        SpeechPlan::new(
            SourceSnapshot::new(
                SourceId {
                    namespace: "fixture".into(),
                    book: "one".into(),
                    chapter: "one".into(),
                },
                text,
                &text_hash(text),
            )
            .unwrap(),
            VoiceSnapshot::new(
                &caps.backend,
                caps.model.as_deref(),
                &caps,
                caps.voices.clone(),
            )
            .unwrap(),
            PlaybackPolicy::Streaming,
        )
    }
    async fn drive(&mut self, play: bool, ready: impl Fn(&Self) -> bool) {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                while let Ok(event) = self.rx.try_recv() {
                    self.events.push(event);
                }
                if ready(self) {
                    break;
                }
                if play {
                    self.player.advance();
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("session did not reach expected state");
    }
    fn ended(&self, id: &str, reason: EndReason) -> bool {
        self.events.iter().any(|event| {
            event.session_id == id
                && matches!(event.event, Event::SessionEnded { reason: r, .. } if r == reason)
        })
    }
}

fn span(start: usize, end: usize, voice: &str) -> SpeechSpan {
    SpeechSpan::new(TextRange { start, end }, voice, None)
}

#[tokio::test]
async fn same_backend_switches_a_b_a_inside_voice_boundaries() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            let mut plan = h.plan("甲乙丙");
            plan.append(vec![
                span(0, 3, "A"),
                SpeechSpan::new(TextRange { start: 3, end: 6 }, "B", Some("calm".into())),
                span(6, 9, "A"),
            ])
            .unwrap();
            plan.seal().unwrap();
            h.manager
                .start_plan("roles".into(), plan, PlanSessionOptions::default())
                .await
                .unwrap();
            h.drive(true, |h| h.ended("roles", EndReason::Completed))
                .await;
            assert_eq!(
                call_shapes(&h.backend.calls.borrow()),
                [
                    ("甲", "A", None),
                    ("乙", "B", Some("calm")),
                    ("丙", "A", None)
                ]
            );
            assert_eq!(*h.backend.selected.borrow(), ["A", "B", "A"]);
            assert_eq!(*h.backend.observed.borrow(), ["A", "B", "A"]);
            let ranges: Vec<_> = h
                .events
                .iter()
                .filter_map(|e| match e.event {
                    Event::SegmentFinished { range, .. } => Some(range),
                    _ => None,
                })
                .collect();
            assert_eq!(
                ranges,
                [
                    TextRange { start: 0, end: 3 },
                    TextRange { start: 3, end: 6 },
                    TextRange { start: 6, end: 9 }
                ]
            );
            let p = h.manager.plan_progress("roles").unwrap();
            assert_eq!((p.accepted_end, p.generated_end, p.played_end), (9, 9, 9));
        })
        .await;
}

#[tokio::test]
async fn incremental_input_waits_without_completing_then_resumes_and_requires_seal() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            h.manager
                .start_plan("open".into(), h.plan("甲乙"), PlanSessionOptions::default())
                .await
                .unwrap();
            h.drive(false, |h| {
                h.manager.plan_progress("open").unwrap().waiting_for_input
            })
            .await;
            assert!(h.backend.calls.borrow().is_empty());
            assert!(h.manager.seal_plan("open").is_err());
            h.manager
                .append_plan("open", vec![span(0, 3, "A")])
                .unwrap();
            h.drive(true, |h| {
                let p = h.manager.plan_progress("open").unwrap();
                p.played_end == 3 && p.waiting_for_input
            })
            .await;
            assert!(!h.ended("open", EndReason::Completed));
            let before = h.manager.plan_progress("open").unwrap();
            assert!(
                h.manager
                    .append_plan("open", vec![span(3, 6, "missing")])
                    .is_err()
            );
            assert_eq!(h.manager.plan_progress("open").unwrap(), before);
            h.manager
                .append_plan("open", vec![span(3, 6, "B")])
                .unwrap();
            h.drive(true, |h| {
                h.manager.plan_progress("open").unwrap().played_end == 6
            })
            .await;
            assert!(!h.ended("open", EndReason::Completed));
            h.manager.seal_plan("open").unwrap();
            h.drive(true, |h| h.ended("open", EndReason::Completed))
                .await;
            assert!(
                h.manager
                    .append_plan("open", vec![span(0, 3, "A")])
                    .is_err()
            );
        })
        .await;
}

#[tokio::test]
async fn sealed_generated_audio_does_not_complete_or_checkpoint_until_played() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            let mut plan = h.plan("甲乙");
            plan.append(vec![span(0, 3, "A"), span(3, 6, "B")]).unwrap();
            plan.seal().unwrap();
            let source = plan.source().clone();
            h.manager
                .start_plan("paused".into(), plan, PlanSessionOptions::default())
                .await
                .unwrap();
            h.manager.pause("paused").await.unwrap();
            h.drive(false, |h| {
                h.manager.plan_progress("paused").unwrap().generated_end == 6
            })
            .await;
            assert_eq!(h.manager.plan_progress("paused").unwrap().played_end, 0);
            let checkpoint = h
                .checkpoints
                .load(source.source(), source.text())
                .unwrap()
                .unwrap();
            assert_eq!(checkpoint.resume_byte, 0);
            assert!(!checkpoint.completed);
            assert!(!h.ended("paused", EndReason::Completed));
            h.manager.resume("paused").await.unwrap();
            h.drive(true, |h| h.ended("paused", EndReason::Completed))
                .await;
            assert!(
                h.checkpoints
                    .load(source.source(), source.text())
                    .unwrap()
                    .unwrap()
                    .completed
            );
        })
        .await;
}

#[tokio::test]
async fn stop_while_waiting_cancels_once_and_rejects_stale_updates() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            h.manager
                .start_plan("old".into(), h.plan("甲"), PlanSessionOptions::default())
                .await
                .unwrap();
            h.drive(false, |h| {
                h.manager.plan_progress("old").unwrap().waiting_for_input
            })
            .await;
            h.manager.stop().await.unwrap();
            h.manager.stop().await.unwrap();
            h.drive(false, |h| h.ended("old", EndReason::Cancelled))
                .await;
            assert_eq!(
                h.events
                    .iter()
                    .filter(|e| matches!(e.event, Event::SessionEnded { .. }))
                    .count(),
                1
            );
            h.manager
                .start_plan("new".into(), h.plan("甲"), PlanSessionOptions::default())
                .await
                .unwrap();
            assert!(h.manager.append_plan("old", vec![span(0, 3, "A")]).is_err());
            assert!(h.manager.seal_plan("old").is_err());
            assert!(h.manager.fail_input("old", "late failure").await.is_err());
            h.manager.stop().await.unwrap();
        })
        .await;
}

#[tokio::test]
async fn input_failure_aborts_bounded_generation_and_is_not_cancellation() {
    tokio::task::LocalSet::new().run_until(async {
        let mut h = Harness::new(); h.backend.mode.set(Mode::Long);
        let mut plan = h.plan("甲乙"); plan.append(vec![span(0, 3, "A")]).unwrap();
        let source = plan.source().clone();
        h.manager.start_plan("fail".into(), plan, PlanSessionOptions::default()).await.unwrap();
        h.manager.pause("fail").await.unwrap();
        h.drive(false, |h| h.backend.chunks.get() >= 70).await;
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(h.backend.chunks.get() < 80); // 30 seconds plus bounded in-flight chunks.
        h.manager.fail_input("fail", "analysis service unavailable").await.unwrap();
        h.drive(false, |h| h.ended("fail", EndReason::Failed) && h.backend.running.get() == 0).await;
        assert_eq!(h.manager.plan_progress("fail").unwrap().input_state, PlanState::Failed);
        assert_eq!(h.manager.plan_progress("fail").unwrap().played_end, 0);
        assert!(!h.checkpoints.load(source.source(), source.text()).unwrap().unwrap().completed);
        assert!(h.events.iter().any(|e| matches!(&e.event, Event::Error(error) if error.code == "input_failed" && error.stage == "plan_input")));
        h.manager.stop().await.unwrap();
        h.drive(false, |_| true).await;
        assert_eq!(h.events.iter().filter(|e| matches!(e.event, Event::SessionEnded { .. })).count(), 1);
        h.backend.mode.set(Mode::Normal);
        let mut retry = h.plan("甲乙"); retry.append(vec![span(0, 6, "B")]).unwrap(); retry.seal().unwrap();
        h.manager.start_plan("retry".into(), retry, PlanSessionOptions::default()).await.unwrap();
        h.drive(true, |h| h.ended("retry", EndReason::Completed)).await;
    }).await;
}

#[tokio::test]
async fn seek_preserves_roles_and_new_input_identity() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            let mut plan = h.plan("甲乙丙");
            plan.append(vec![span(0, 3, "A"), span(3, 6, "B"), span(6, 9, "A")])
                .unwrap();
            plan.seal().unwrap();
            h.manager
                .start_plan("before".into(), plan, PlanSessionOptions::default())
                .await
                .unwrap();
            h.manager.pause("before").await.unwrap();
            h.drive(false, |h| {
                h.manager.plan_progress("before").unwrap().generated_end == 9
            })
            .await;
            let count = h.backend.calls.borrow().len();
            assert!(
                h.manager
                    .seek("before", "bad".into(), 1, &Config::default())
                    .await
                    .is_err()
            );
            h.manager
                .seek("before", "after".into(), 3, &Config::default())
                .await
                .unwrap();
            assert!(
                h.manager
                    .append_plan("before", vec![span(0, 3, "A")])
                    .is_err()
            );
            assert_eq!(h.manager.status().1, tts_protocol::SessionState::Paused);
            h.manager.resume("after").await.unwrap();
            h.drive(true, |h| h.ended("after", EndReason::Completed))
                .await;
            let calls = h.backend.calls.borrow();
            assert_eq!(
                calls[count..]
                    .iter()
                    .map(|call| call.voice.as_str())
                    .collect::<Vec<_>>(),
                ["B", "A"]
            );
            assert!(h.ended("before", EndReason::Cancelled));
        })
        .await;
}

#[tokio::test]
async fn seek_inside_an_open_assignment_retains_style_and_allows_new_id_append() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            let mut plan = h.plan("甲乙丙丁");
            plan.append(vec![SpeechSpan::new(
                TextRange { start: 0, end: 6 },
                "A",
                Some("calm".into()),
            )])
            .unwrap();
            h.manager
                .start_plan("open-before".into(), plan, PlanSessionOptions::default())
                .await
                .unwrap();
            h.manager.pause("open-before").await.unwrap();
            h.drive(false, |h| {
                h.manager
                    .plan_progress("open-before")
                    .unwrap()
                    .generated_end
                    == 6
            })
            .await;
            let count = h.backend.calls.borrow().len();
            h.manager
                .seek_plan("open-before", "open-after".into(), 3)
                .await
                .unwrap();
            h.manager
                .append_plan("open-after", vec![span(6, 12, "B")])
                .unwrap();
            h.manager.seal_plan("open-after").unwrap();
            h.manager.resume("open-after").await.unwrap();
            h.drive(true, |h| h.ended("open-after", EndReason::Completed))
                .await;
            let calls = h.backend.calls.borrow();
            assert_eq!(
                call_shapes(&calls[count..]),
                [("乙", "A", Some("calm")), ("丙丁", "B", None)]
            );
        })
        .await;
}

#[tokio::test]
async fn long_stream_exceeding_prefetch_budget_finishes_when_playback_consumes() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            h.backend.mode.set(Mode::Long);
            let mut plan = h.plan("甲");
            plan.append(vec![span(0, 3, "A")]).unwrap();
            plan.seal().unwrap();
            h.manager
                .start_plan("long".into(), plan, PlanSessionOptions::default())
                .await
                .unwrap();
            h.drive(true, |h| h.ended("long", EndReason::Completed))
                .await;
            assert_eq!(h.backend.chunks.get(), 100);
            assert_eq!(h.manager.plan_progress("long").unwrap().played_end, 3);
        })
        .await;
}

#[tokio::test]
async fn analysis_failure_after_playback_restores_only_the_confirmed_prefix() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            let mut plan = h.plan("甲乙");
            let source = plan.source().clone();
            plan.append(vec![span(0, 3, "A")]).unwrap();
            h.manager
                .start_plan("partial".into(), plan, PlanSessionOptions::default())
                .await
                .unwrap();
            h.drive(true, |h| {
                h.manager.plan_progress("partial").unwrap().played_end == 3
            })
            .await;
            h.manager
                .fail_input("partial", "analysis failed")
                .await
                .unwrap();
            h.drive(false, |h| h.ended("partial", EndReason::Failed))
                .await;
            let checkpoint = h
                .checkpoints
                .load(source.source(), source.text())
                .unwrap()
                .unwrap();
            assert_eq!(checkpoint.resume_byte, 3);
            assert!(!checkpoint.completed);
            let mut retry = h.plan("甲乙");
            retry
                .append(vec![span(0, 3, "A"), span(3, 6, "B")])
                .unwrap();
            retry.seal().unwrap();
            h.manager
                .start_plan(
                    "restored".into(),
                    retry,
                    PlanSessionOptions {
                        restore_checkpoint: true,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            h.drive(true, |h| h.ended("restored", EndReason::Completed))
                .await;
            assert_eq!(
                h.backend
                    .calls
                    .borrow()
                    .iter()
                    .map(|call| call.voice.as_str())
                    .collect::<Vec<_>>(),
                ["A", "B"]
            );
        })
        .await;
}

#[tokio::test]
async fn invalid_initial_plan_or_options_leave_the_old_session_untouched() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            h.manager
                .start_plan("old".into(), h.plan("甲"), PlanSessionOptions::default())
                .await
                .unwrap();
            let after = SpeechPlan::new(
                h.plan("甲").source().clone(),
                h.plan("甲").voices().clone(),
                PlaybackPolicy::AfterChapterReady,
            );
            assert!(
                h.manager
                    .start_plan(
                        "staging".into(),
                        after,
                        PlanSessionOptions {
                            staging: talechime_core::StagingOptions {
                                max_bytes: 0,
                                ..Default::default()
                            },
                            ..Default::default()
                        }
                    )
                    .await
                    .is_err()
            );
            assert!(
                h.manager
                    .start_plan(
                        "bad".into(),
                        h.plan("甲"),
                        PlanSessionOptions {
                            speed: f32::NAN,
                            ..Default::default()
                        }
                    )
                    .await
                    .is_err()
            );
            assert!(
                h.manager
                    .start_plan(
                        "ahead".into(),
                        h.plan("甲"),
                        PlanSessionOptions {
                            resume_byte: Some(3),
                            ..Default::default()
                        }
                    )
                    .await
                    .is_err()
            );
            let mut mismatch = h.plan("甲");
            mismatch.append(vec![span(0, 3, "A")]).unwrap();
            h.backend.caps.borrow_mut().model = Some("other".into());
            assert!(
                h.manager
                    .start_plan("mismatch".into(), mismatch, PlanSessionOptions::default())
                    .await
                    .is_err()
            );
            assert_eq!(h.manager.status().0.as_deref(), Some("old"));
            h.manager.stop().await.unwrap();
        })
        .await;
}

#[tokio::test]
async fn planned_voice_settings_are_fixed_but_playback_settings_can_change() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            h.manager
                .start_plan("fixed".into(), h.plan("甲"), PlanSessionOptions::default())
                .await
                .unwrap();
            assert!(
                h.manager
                    .update_settings("unexpected".into(), &Config::default())
                    .await
                    .is_err()
            );
            h.manager.configure_playback("fixed", 0.5, 1.5).unwrap();
            assert_eq!(h.player.settings.get(), (0.5, 1.5));
            assert!(h.manager.configure_playback("fixed", -1.0, 1.0).is_err());
            assert_eq!(h.player.settings.get(), (0.5, 1.5));
            h.manager.stop().await.unwrap();
        })
        .await;
}

#[tokio::test]
async fn backend_disconnect_or_invalid_segmentation_fails_without_completion() {
    tokio::task::LocalSet::new()
        .run_until(async {
            for mode in [Mode::Disconnect, Mode::Omit, Mode::InvalidRange] {
                let mut h = Harness::new();
                h.backend.mode.set(mode);
                let mut plan = h.plan("甲");
                plan.append(vec![span(0, 3, "A")]).unwrap();
                plan.seal().unwrap();
                let source = plan.source().clone();
                h.manager
                    .start_plan("broken".into(), plan, PlanSessionOptions::default())
                    .await
                    .unwrap();
                h.drive(false, |h| h.ended("broken", EndReason::Failed))
                    .await;
                assert!(!h.ended("broken", EndReason::Completed));
                assert!(
                    !h.checkpoints
                        .load(source.source(), source.text())
                        .unwrap()
                        .unwrap()
                        .completed
                );
            }
        })
        .await;
}

#[tokio::test]
async fn sanitized_layout_completes_with_original_source_progress() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let text = "====\u{200B}====\n你\u{FEFF}好。\n\u{0}\u{200B}";
            for policy in [PlaybackPolicy::Streaming, PlaybackPolicy::AfterChapterReady] {
                let mut h = Harness::new();
                let base = h.plan(text);
                let mut plan =
                    SpeechPlan::new(base.source().clone(), base.voices().clone(), policy);
                plan.append(vec![span(0, text.len(), "A")]).unwrap();
                plan.seal().unwrap();
                h.manager
                    .start_plan("noise".into(), plan, PlanSessionOptions::default())
                    .await
                    .unwrap();
                h.drive(true, |h| h.ended("noise", EndReason::Completed))
                    .await;
                let progress = h.manager.plan_progress("noise").unwrap();
                assert_eq!(progress.generated_end, text.len());
                assert_eq!(progress.played_end, text.len());
                assert_eq!(
                    h.backend
                        .calls
                        .borrow()
                        .iter()
                        .map(|call| call.text.as_str())
                        .collect::<Vec<_>>(),
                    ["你好。"]
                );
            }
        })
        .await;
}

#[tokio::test]
async fn skipped_layout_waits_for_preceding_audio_and_empty_plan_requires_seal() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            let mut plan = h.plan("甲\n====\n");
            plan.append(vec![span(0, 3, "A"), span(3, 9, "B")]).unwrap();
            plan.seal().unwrap();
            h.manager
                .start_plan("layout".into(), plan, PlanSessionOptions::default())
                .await
                .unwrap();
            h.manager.pause("layout").await.unwrap();
            h.drive(false, |h| {
                h.manager.plan_progress("layout").unwrap().generated_end == 9
            })
            .await;
            assert_eq!(h.manager.plan_progress("layout").unwrap().played_end, 0);
            assert_eq!(h.backend.calls.borrow().len(), 1);
            h.manager.resume("layout").await.unwrap();
            h.drive(true, |h| h.ended("layout", EndReason::Completed))
                .await;
            assert_eq!(h.manager.plan_progress("layout").unwrap().played_end, 9);
            h.manager
                .start_plan("empty".into(), h.plan(""), PlanSessionOptions::default())
                .await
                .unwrap();
            h.drive(false, |h| {
                h.manager.plan_progress("empty").unwrap().waiting_for_input
            })
            .await;
            assert!(!h.ended("empty", EndReason::Completed));
            h.manager.seal_plan("empty").unwrap();
            h.drive(false, |h| h.ended("empty", EndReason::Completed))
                .await;
        })
        .await;
}

fn staged(h: &Harness, text: &str) -> SpeechPlan {
    let base = h.plan(text);
    SpeechPlan::new(
        base.source().clone(),
        base.voices().clone(),
        PlaybackPolicy::AfterChapterReady,
    )
}

#[tokio::test]
async fn staged_long_chapter_prepares_while_paused_then_replays_and_cleans() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            h.backend.mode.set(Mode::Long);
            let root = tempfile::tempdir().unwrap();
            std::fs::write(root.path().join("keep"), "user file").unwrap();
            let mut plan = staged(&h, "甲");
            plan.append(vec![span(0, 3, "A")]).unwrap();
            plan.seal().unwrap();
            h.manager
                .start_plan(
                    "stage".into(),
                    plan,
                    PlanSessionOptions {
                        staging: talechime_core::StagingOptions {
                            directory: Some(root.path().into()),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            h.manager.pause("stage").await.unwrap();
            h.drive(false, |h| {
                h.manager.plan_progress("stage").unwrap().chapter_ready
            })
            .await;
            assert_eq!(h.manager.plan_progress("stage").unwrap().played_end, 0);
            assert!(!h.ended("stage", EndReason::Completed));
            assert_eq!(stage_count(root.path()), 1);
            h.manager.resume("stage").await.unwrap();
            h.drive(true, |h| h.ended("stage", EndReason::Completed))
                .await;
            assert_eq!(h.manager.plan_progress("stage").unwrap().played_end, 3);
            assert_eq!(stage_count(root.path()), 0);
            assert_eq!(
                std::fs::read_to_string(root.path().join("keep")).unwrap(),
                "user file"
            );
        })
        .await;
}

#[tokio::test]
async fn staged_incremental_input_waits_for_seal_and_cancel_removes_storage() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            let root = tempfile::tempdir().unwrap();
            let mut plan = staged(&h, "甲乙");
            plan.append(vec![span(0, 3, "A")]).unwrap();
            h.manager
                .start_plan(
                    "open-stage".into(),
                    plan,
                    PlanSessionOptions {
                        staging: talechime_core::StagingOptions {
                            directory: Some(root.path().into()),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            h.drive(true, |h| {
                h.manager
                    .plan_progress("open-stage")
                    .unwrap()
                    .waiting_for_input
            })
            .await;
            let progress = h.manager.plan_progress("open-stage").unwrap();
            assert_eq!(progress.generated_end, 3);
            assert_eq!(progress.played_end, 0);
            assert!(!progress.chapter_ready);
            assert_eq!(h.player.queued.get(), Duration::ZERO);
            h.manager
                .append_plan("open-stage", vec![span(3, 6, "B")])
                .unwrap();
            h.drive(true, |h| {
                h.manager.plan_progress("open-stage").unwrap().generated_end == 6
            })
            .await;
            assert_eq!(h.player.queued.get(), Duration::ZERO);
            h.manager.stop().await.unwrap();
            assert_eq!(stage_count(root.path()), 0);
        })
        .await;
}

#[tokio::test]
async fn staged_capacity_failure_never_queues_audio_and_cleans() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            let root = tempfile::tempdir().unwrap();
            let mut plan = staged(&h, "甲");
            plan.append(vec![span(0, 3, "A")]).unwrap();
            plan.seal().unwrap();
            h.manager
                .start_plan(
                    "limited".into(),
                    plan,
                    PlanSessionOptions {
                        staging: talechime_core::StagingOptions {
                            directory: Some(root.path().into()),
                            max_bytes: 100,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            h.drive(true, |h| h.ended("limited", EndReason::Failed))
                .await;
            assert_eq!(h.player.queued.get(), Duration::ZERO);
            assert_eq!(h.manager.plan_progress("limited").unwrap().played_end, 0);
            assert!(!h.manager.plan_progress("limited").unwrap().chapter_ready);
            assert_eq!(stage_count(root.path()), 0);
        })
        .await;
}

fn stage_count(parent: &std::path::Path) -> usize {
    std::fs::read_dir(parent.join("talechime-staging-v1"))
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().unwrap().is_dir())
        .count()
}

#[tokio::test]
async fn staged_incremental_a_b_a_seals_and_seek_regenerates_safely() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            let root = tempfile::tempdir().unwrap();
            let mut plan = staged(&h, "甲乙丙");
            plan.append(vec![span(0, 3, "A")]).unwrap();
            h.manager
                .start_plan(
                    "stage-before".into(),
                    plan,
                    PlanSessionOptions {
                        staging: talechime_core::StagingOptions {
                            directory: Some(root.path().into()),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            h.manager.pause("stage-before").await.unwrap();
            h.drive(false, |h| {
                h.manager
                    .plan_progress("stage-before")
                    .unwrap()
                    .waiting_for_input
            })
            .await;
            h.manager
                .append_plan("stage-before", vec![span(3, 6, "B"), span(6, 9, "A")])
                .unwrap();
            h.manager.seal_plan("stage-before").unwrap();
            h.drive(false, |h| {
                h.manager
                    .plan_progress("stage-before")
                    .unwrap()
                    .chapter_ready
            })
            .await;
            h.manager
                .seek_plan("stage-before", "stage-after".into(), 3)
                .await
                .unwrap();
            h.drive(false, |h| {
                h.manager
                    .plan_progress("stage-after")
                    .unwrap()
                    .chapter_ready
            })
            .await;
            assert_eq!(
                h.manager.plan_progress("stage-after").unwrap().played_end,
                3
            );
            assert!(h.player.paused.get());
            assert_eq!(stage_count(root.path()), 1);
            assert!(
                h.manager
                    .append_plan("stage-before", vec![span(0, 3, "A")])
                    .is_err()
            );
            h.manager.resume("stage-after").await.unwrap();
            h.drive(true, |h| h.ended("stage-after", EndReason::Completed))
                .await;
            assert_eq!(
                h.manager.plan_progress("stage-after").unwrap().played_end,
                9
            );
            assert_eq!(stage_count(root.path()), 0);
        })
        .await;
}

#[tokio::test]
async fn staged_input_failure_cleans_without_playing_partial_audio() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            let root = tempfile::tempdir().unwrap();
            let mut plan = staged(&h, "甲乙");
            plan.append(vec![span(0, 3, "A")]).unwrap();
            h.manager
                .start_plan(
                    "input-stage".into(),
                    plan,
                    PlanSessionOptions {
                        staging: talechime_core::StagingOptions {
                            directory: Some(root.path().into()),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            h.drive(true, |h| {
                h.manager
                    .plan_progress("input-stage")
                    .unwrap()
                    .waiting_for_input
            })
            .await;
            h.manager
                .fail_input("input-stage", "analysis failed")
                .await
                .unwrap();
            h.drive(true, |h| h.ended("input-stage", EndReason::Failed))
                .await;
            assert_eq!(
                h.manager.plan_progress("input-stage").unwrap().played_end,
                0
            );
            assert_eq!(h.player.queued.get(), Duration::ZERO);
            assert_eq!(stage_count(root.path()), 0);
        })
        .await;
}

#[tokio::test]
async fn corrupted_staged_prefix_is_rejected_before_any_playback() {
    use std::io::{Seek, SeekFrom, Write};
    tokio::task::LocalSet::new().run_until(async {
        let mut h = Harness::new(); let root = tempfile::tempdir().unwrap();
        let mut plan = staged(&h,"甲"); plan.append(vec![span(0,3,"A")]).unwrap();
        h.manager.start_plan("corrupt-stage".into(), plan, PlanSessionOptions {
            staging: talechime_core::StagingOptions { directory: Some(root.path().into()), ..Default::default() }, ..Default::default()
        }).await.unwrap();
        h.drive(false, |h| h.manager.plan_progress("corrupt-stage").unwrap().waiting_for_input && stage_count(root.path()) == 1).await;
        let execution = std::fs::read_dir(root.path().join("talechime-staging-v1")).unwrap().filter_map(Result::ok).find(|e|e.file_type().unwrap().is_dir()).unwrap();
        let mut file = std::fs::OpenOptions::new().write(true).open(execution.path().join("chapter.pcm")).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap(); file.write_all(b"BADMAGIC").unwrap(); drop(file);
        h.manager.seal_plan("corrupt-stage").unwrap();
        h.drive(true, |h| h.ended("corrupt-stage", EndReason::Failed)).await;
        assert_eq!(h.player.queued.get(),Duration::ZERO); assert!(!h.manager.plan_progress("corrupt-stage").unwrap().chapter_ready);
        assert!(h.events.iter().any(|event| matches!(&event.event, Event::Error(error) if error.code == "staging_failed" && error.stage == "chapter_staging")));
        assert_eq!(stage_count(root.path()),0);
    }).await;
}

#[tokio::test]
async fn staged_many_tiny_segments_drain_when_marker_queue_is_full() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            h.backend.mode.set(Mode::Tiny);
            let text = "甲".repeat(130);
            let mut plan = staged(&h, &text);
            plan.append(
                (0..130)
                    .map(|i| span(i * 3, (i + 1) * 3, if i % 2 == 0 { "A" } else { "B" }))
                    .collect(),
            )
            .unwrap();
            plan.seal().unwrap();
            h.manager
                .start_plan("tiny-stage".into(), plan, PlanSessionOptions::default())
                .await
                .unwrap();
            h.manager.pause("tiny-stage").await.unwrap();
            h.drive(false, |h| {
                h.manager.plan_progress("tiny-stage").unwrap().chapter_ready
            })
            .await;
            h.manager.resume("tiny-stage").await.unwrap();
            h.drive(true, |h| h.ended("tiny-stage", EndReason::Completed))
                .await;
            assert_eq!(
                h.manager.plan_progress("tiny-stage").unwrap().played_end,
                text.len()
            );
        })
        .await;
}

#[tokio::test]
async fn staged_stop_waits_for_paused_reader_and_removes_storage_before_returning() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            h.backend.mode.set(Mode::Long);
            let root = tempfile::tempdir().unwrap();
            let mut plan = staged(&h, "甲");
            plan.append(vec![span(0, 3, "A")]).unwrap();
            plan.seal().unwrap();
            h.manager
                .start_plan(
                    "stop-reader".into(),
                    plan,
                    PlanSessionOptions {
                        staging: talechime_core::StagingOptions {
                            directory: Some(root.path().into()),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            h.manager.pause("stop-reader").await.unwrap();
            h.drive(false, |h| {
                h.manager
                    .plan_progress("stop-reader")
                    .unwrap()
                    .chapter_ready
                    && h.player.queued.get() >= Duration::from_secs(29)
            })
            .await;
            h.manager.stop().await.unwrap();
            assert_eq!(stage_count(root.path()), 0);
            assert_eq!(h.player.queued.get(), Duration::ZERO);
        })
        .await;
}

#[tokio::test]
async fn continuation_across_spans_resets_voice_style_seek_and_recovery_in_both_strategies() {
    tokio::task::LocalSet::new()
        .run_until(async {
            for policy in [PlaybackPolicy::Streaming, PlaybackPolicy::AfterChapterReady] {
                let mut h = Harness::new();
                h.backend.caps.borrow_mut().continuation = true;
                let text = "甲乙丙丁戊己";
                let caps = h.backend.capabilities();
                let mut plan = SpeechPlan::new(
                    h.plan(text).source().clone(),
                    VoiceSnapshot::new(
                        &caps.backend,
                        caps.model.as_deref(),
                        &caps,
                        caps.voices.clone(),
                    )
                    .unwrap(),
                    policy,
                );
                plan.append(vec![
                    span(0, 3, "A"),
                    span(3, 6, "A"),
                    span(6, 9, "B"),
                    SpeechSpan::new(TextRange { start: 9, end: 12 }, "B", Some("calm".into())),
                    SpeechSpan::new(TextRange { start: 12, end: 15 }, "B", Some("calm".into())),
                    span(15, 18, "B"),
                ])
                .unwrap();
                plan.seal().unwrap();
                let directory = tempfile::tempdir().unwrap();
                let options = PlanSessionOptions {
                    staging: talechime_core::StagingOptions {
                        directory: Some(directory.path().into()),
                        ..Default::default()
                    },
                    ..Default::default()
                };
                h.manager
                    .start_plan("before".into(), plan.clone(), options.clone())
                    .await
                    .unwrap();
                h.manager.pause("before").await.unwrap();
                h.drive(false, |h| {
                    h.manager.plan_progress("before").unwrap().generated_end == 18
                })
                .await;
                assert_eq!(
                    *h.backend.contexts.borrow(),
                    [None, Some("甲".into()), None, None, Some("丁".into()), None]
                );
                let count = h.backend.contexts.borrow().len();
                h.manager
                    .seek_plan("before", "after".into(), 12)
                    .await
                    .unwrap();
                h.manager.resume("after").await.unwrap();
                h.drive(true, |h| h.ended("after", EndReason::Completed))
                    .await;
                assert!(h.backend.contexts.borrow()[count].is_none());
                let count = h.backend.contexts.borrow().len();
                h.manager
                    .start_plan(
                        "resume".into(),
                        plan,
                        PlanSessionOptions {
                            resume_byte: Some(3),
                            ..options
                        },
                    )
                    .await
                    .unwrap();
                h.drive(true, |h| h.ended("resume", EndReason::Completed))
                    .await;
                assert!(h.backend.contexts.borrow()[count].is_none());
            }
        })
        .await;
}

#[tokio::test]
async fn native_speed_routes_to_generation_and_divides_out_of_the_sink() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            // Declare a native speed parameter on the prepared backend.
            h.backend.caps.borrow_mut().parameters = vec![tts_protocol::ParameterSpec {
                name: "speed".into(),
                kind: tts_protocol::ParamKind::Float { min: 0.5, max: 2.0 },
                default: tts_protocol::ParamValue::Float(1.0),
                description: String::new(),
            }];
            let mut plan = h.plan("甲乙");
            plan.append(vec![span(0, 3, "A"), span(3, 6, "A")]).unwrap();
            plan.seal().unwrap();
            h.manager
                .start_plan(
                    "native".into(),
                    plan,
                    PlanSessionOptions {
                        speed: 1.5,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            // The session routes the requested rate into generation; the sink stays 1.0.
            assert_eq!(h.player.settings.get(), (1.0, 1.0));
            // Runtime rate changes divide the native speed out instead of compounding.
            h.manager.configure_playback("native", 1.0, 2.0).unwrap();
            let (_, sink) = h.player.settings.get();
            assert!((sink - 2.0 / 1.5).abs() < 1e-6, "{sink}");
            h.manager
                .seek_plan("native", "resumed".into(), 0)
                .await
                .unwrap();
            let (_, sink) = h.player.settings.get();
            assert!((sink - 2.0 / 1.5).abs() < 1e-6, "seek lost speed: {sink}");
            h.drive(true, |h| h.ended("resumed", EndReason::Completed))
                .await;
            assert!(
                h.backend.calls.borrow().iter().all(|call| call.params
                    == [("speed".to_string(), tts_protocol::ParamValue::Float(1.5))])
            );
        })
        .await;
}

#[tokio::test]
async fn explicit_native_speed_wins_initially_and_survives_seek() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            h.backend.caps.borrow_mut().parameters = vec![tts_protocol::ParameterSpec {
                name: "speed".into(),
                kind: tts_protocol::ParamKind::Float { min: 0.5, max: 2.0 },
                default: tts_protocol::ParamValue::Float(1.0),
                description: String::new(),
            }];
            let mut plan = h.plan("甲乙");
            plan.append(vec![span(0, 6, "A")]).unwrap();
            plan.seal().unwrap();
            let mut params = talechime_core::GenerationParams::new();
            params.insert("speed", tts_protocol::ParamValue::Float(1.5));
            h.manager
                .start_plan(
                    "explicit".into(),
                    plan,
                    PlanSessionOptions {
                        speed: 2.0,
                        params,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            assert_eq!(h.player.settings.get(), (1.0, 1.0));
            h.manager
                .seek_plan("explicit", "resumed".into(), 0)
                .await
                .unwrap();
            assert_eq!(h.player.settings.get(), (1.0, 1.0));
            h.drive(true, |h| h.ended("resumed", EndReason::Completed))
                .await;
        })
        .await;
}

#[tokio::test]
async fn legacy_settings_divide_out_native_generation_speed() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            h.backend.caps.borrow_mut().parameters = vec![tts_protocol::ParameterSpec {
                name: "speed".into(),
                kind: tts_protocol::ParamKind::Float { min: 0.5, max: 2.0 },
                default: tts_protocol::ParamValue::Float(1.0),
                description: String::new(),
            }];
            let text = "甲乙";
            let source = h.plan(text).source().source().clone();
            let mut config = Config {
                backend: "fixture".into(),
                model: Some("shared".into()),
                voice: "A".into(),
                speed: 1.5,
                ..Default::default()
            };
            h.manager
                .start(
                    "legacy".into(),
                    talechime_core::session::StartRequest {
                        source,
                        text: text.into(),
                        text_hash: text_hash(text),
                        resume_byte: None,
                        restore_checkpoint: false,
                    },
                    &config,
                )
                .await
                .unwrap();
            assert_eq!(h.player.settings.get(), (1.0, 1.0));
            config.volume = 0.5;
            assert!(
                !h.manager
                    .update_settings("unused".into(), &config)
                    .await
                    .unwrap()
            );
            assert_eq!(h.player.settings.get(), (0.5, 1.0));
            config.speed = 2.0;
            h.manager
                .update_settings("unused".into(), &config)
                .await
                .unwrap();
            assert!((h.player.settings.get().1 - 2.0 / 1.5).abs() < 1e-6);
            h.drive(true, |h| h.ended("legacy", EndReason::Completed))
                .await;
        })
        .await;
}

#[tokio::test]
async fn backends_without_native_speed_keep_playback_routing() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let mut h = Harness::new();
            let mut plan = h.plan("甲乙");
            plan.append(vec![span(0, 3, "A"), span(3, 6, "A")]).unwrap();
            plan.seal().unwrap();
            h.manager
                .start_plan(
                    "playback".into(),
                    plan,
                    PlanSessionOptions {
                        speed: 1.5,
                        ..Default::default()
                    },
                )
                .await
                .unwrap();
            assert_eq!(h.player.settings.get(), (1.0, 1.5));
            h.manager.configure_playback("playback", 1.0, 2.0).unwrap();
            assert_eq!(h.player.settings.get(), (1.0, 2.0));
            h.drive(true, |h| h.ended("playback", EndReason::Completed))
                .await;
            assert!(
                h.backend
                    .calls
                    .borrow()
                    .iter()
                    .all(|call| call.params.is_empty())
            );
        })
        .await;
}
