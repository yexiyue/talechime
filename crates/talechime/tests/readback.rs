#[path = "library/fixture.rs"]
#[allow(dead_code)]
mod support;
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    rc::Rc,
    sync::Arc,
    time::Duration,
};
use support::{Fixture, Mode, Player};
use talechime::*;
enum Answer {
    Text(&'static str),
    Error,
    Pending,
}
struct Asr {
    family: &'static str,
    answers: RefCell<VecDeque<Answer>>,
    calls: Cell<usize>,
    cancelled: Rc<Cell<bool>>,
}
impl Asr {
    fn new(family: &'static str, answers: Vec<Answer>) -> Rc<Self> {
        Rc::new(Self {
            family,
            answers: RefCell::new(answers.into()),
            calls: Cell::new(0),
            cancelled: Default::default(),
        })
    }
}
impl Recognizer for Asr {
    fn identity(&self) -> RecognizerIdentity {
        RecognizerIdentity {
            family: self.family.into(),
            model: "fake".into(),
            revision: "pinned".into(),
            implementation: "fake-v1".into(),
        }
    }
    fn transcribe(&self, _: Arc<ReadbackAudio>) -> Recognition<'_> {
        Box::pin(async move {
            self.calls.set(self.calls.get() + 1);
            struct Cancel(Rc<Cell<bool>>);
            impl Drop for Cancel {
                fn drop(&mut self) {
                    self.0.set(true);
                }
            }
            let _guard = Cancel(self.cancelled.clone());
            let answer = self
                .answers
                .borrow_mut()
                .pop_front()
                .unwrap_or(Answer::Text("甲乙"));
            match answer {
                Answer::Text(s) => Ok(s.into()),
                Answer::Error => Err(VerificationError::Recognition("test failure".into())),
                Answer::Pending => std::future::pending().await,
            }
        })
    }
}
fn options(policy: VerificationPolicy) -> VerificationOptions {
    VerificationOptions {
        policy,
        timeout_ms: 20,
        ..Default::default()
    }
}
fn engine(a: Rc<Asr>, b: Rc<Asr>) -> (Engine, Rc<Fixture>) {
    let backend = Rc::new(Fixture::default());
    let mut engine = Engine::from_backend(backend.clone());
    engine
        .set_verifier(Rc::new(Verifier::new(a, b).unwrap()))
        .unwrap();
    (engine, backend)
}
async fn drain(stream: &mut PcmStream) -> (Vec<VerificationReport>, usize, bool) {
    let mut reports = vec![];
    let mut samples = 0;
    let mut error = false;
    while let Some(item) = stream.next().await {
        match item {
            Ok(SynthesisItem::Verification(report)) => reports.push(report),
            Ok(SynthesisItem::Audio(audio)) => samples += audio.pcm().samples.len(),
            Err(_) => error = true,
        }
    }
    (reports, samples, error)
}
#[tokio::test]
async fn gate_replaces_only_failed_attempt_before_delivering_pcm_and_keeps_voice_style() {
    run_local(async {
        let a = Asr::new("a", vec![Answer::Text("甲"), Answer::Text("甲乙")]);
        let b = Asr::new("b", vec![Answer::Text("甲")]);
        let (engine, backend) = engine(a.clone(), b.clone());
        backend.mode.set(Mode::Changing);
        let mut stream = engine
            .synthesize_verified(
                "甲乙",
                "B",
                Some("轻声".into()),
                options(VerificationPolicy::Gate {
                    max_retries: 1,
                    strict_suspect: false,
                }),
            )
            .unwrap();
        // Both attempts' reports precede the accepted PCM.
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            SynthesisItem::Verification(VerificationReport {
                verdict: VerificationVerdict::ConfirmedError,
                attempt: 0,
                ..
            })
        ));
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            SynthesisItem::Verification(VerificationReport {
                verdict: VerificationVerdict::Passed,
                attempt: 1,
                ..
            })
        ));
        let (_, samples, error) = drain(&mut stream).await;
        assert!(!error);
        assert_eq!(samples, 800);
        assert_eq!(stream.state(), SynthesisState::Completed);
        assert_eq!(backend.calls.borrow().len(), 2);
        assert!(
            backend
                .calls
                .borrow()
                .iter()
                .all(|(text, voice, style)| text == "甲乙"
                    && voice == "B"
                    && style.as_deref() == Some("轻声"))
        );
        assert_eq!(a.calls.get(), 2);
        assert_eq!(b.calls.get(), 1);
    })
    .await
}
#[tokio::test]
async fn exhaustion_and_unverified_gate_deliver_no_pcm_but_report_only_continues() {
    run_local(async {
        for (policy, answer, expected_error, verdict, calls) in [
            (
                VerificationPolicy::Gate {
                    max_retries: 1,
                    strict_suspect: false,
                },
                Answer::Text("甲"),
                true,
                VerificationVerdict::ConfirmedError,
                2,
            ),
            (
                VerificationPolicy::Gate {
                    max_retries: 1,
                    strict_suspect: false,
                },
                Answer::Error,
                true,
                VerificationVerdict::Unverified,
                1,
            ),
            (
                VerificationPolicy::ReportOnly,
                Answer::Error,
                false,
                VerificationVerdict::Unverified,
                1,
            ),
        ] {
            let a = Asr::new(
                "a",
                match answer {
                    Answer::Text(s) => vec![Answer::Text(s), Answer::Text(s)],
                    Answer::Error => vec![Answer::Error],
                    Answer::Pending => vec![],
                },
            );
            let (engine, backend) = engine(
                a,
                Asr::new("b", vec![Answer::Text("甲"), Answer::Text("甲")]),
            );
            let mut stream = engine
                .synthesize_verified("甲乙", "A", None, options(policy))
                .unwrap();
            let (reports, samples, error) = drain(&mut stream).await;
            assert_eq!(error, expected_error);
            assert_eq!(samples == 0, expected_error);
            assert_eq!(reports.last().unwrap().verdict, verdict);
            assert_eq!(backend.calls.borrow().len(), calls);
        }
    })
    .await
}
#[tokio::test]
async fn same_family_rejected_and_suspect_default_delivery_vs_strict_gate() {
    run_local(async {
        assert!(Verifier::new(Asr::new("a", vec![]), Asr::new("a", vec![])).is_err());
        for strict_suspect in [false, true] {
            let (engine, backend) = engine(
                Asr::new("a", vec![Answer::Text("丙乙")]),
                Asr::new("b", vec![Answer::Text("丙乙")]),
            );
            let mut stream = engine
                .synthesize_verified(
                    "甲乙",
                    "A",
                    None,
                    options(VerificationPolicy::Gate {
                        max_retries: 1,
                        strict_suspect,
                    }),
                )
                .unwrap();
            let (reports, samples, error) = drain(&mut stream).await;
            assert_eq!(reports[0].verdict, VerificationVerdict::Suspect);
            assert_eq!(error, strict_suspect);
            assert_eq!(samples == 0, strict_suspect);
            assert_eq!(backend.calls.borrow().len(), 1);
        }
    })
    .await
}
#[tokio::test]
async fn timeout_and_explicit_cancel_drop_recognition_and_engine_is_reusable() {
    run_local(async {
        let a = Asr::new("a", vec![Answer::Pending]);
        let (mut engine, _) = engine(a.clone(), Asr::new("b", vec![]));
        let mut stream = engine
            .synthesize_verified(
                "甲乙",
                "A",
                None,
                options(VerificationPolicy::Gate {
                    max_retries: 0,
                    strict_suspect: false,
                }),
            )
            .unwrap();
        let (reports, samples, error) = drain(&mut stream).await;
        assert!(error);
        assert_eq!(samples, 0);
        assert_eq!(reports[0].verdict, VerificationVerdict::Unverified);
        assert!(a.cancelled.get());
        a.cancelled.set(false);
        a.answers.borrow_mut().push_back(Answer::Pending);
        let mut stream = engine
            .synthesize_verified(
                "甲乙",
                "A",
                None,
                VerificationOptions {
                    timeout_ms: 10_000,
                    ..options(VerificationPolicy::ReportOnly)
                },
            )
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(20), stream.next())
                .await
                .is_err()
        );
        stream.cancel().await;
        assert!(a.cancelled.get());
        assert!(engine.synthesize_pcm("甲乙", "A", None, 4096).await.is_ok());
        engine.close().await.unwrap();
    })
    .await
}
#[tokio::test]
async fn bounds_and_missing_preparation_fail_before_delivery() {
    run_local(async {
        let backend = Rc::new(Fixture::default());
        let plain = Engine::from_backend(backend);
        assert!(
            plain
                .synthesize_verified("甲乙", "A", None, options(VerificationPolicy::ReportOnly))
                .is_err()
        );
        let (engine, backend) = engine(Asr::new("a", vec![]), Asr::new("b", vec![]));
        backend.mode.set(Mode::Long);
        let mut stream = engine
            .synthesize_verified("甲乙", "A", None, options(VerificationPolicy::ReportOnly))
            .unwrap();
        let (reports, samples, error) = drain(&mut stream).await;
        assert!(error);
        assert!(reports.is_empty());
        assert_eq!(samples, 0);
    })
    .await
}
#[tokio::test]
async fn after_chapter_error_never_reaches_player_or_chapter_ready() {
    run_local(async {
        let (engine, _) = engine(
            Asr::new("a", vec![Answer::Text("甲乙"), Answer::Text("甲")]),
            Asr::new("b", vec![Answer::Text("甲")]),
        );
        let player = Rc::new(Player::default());
        let mut listening = engine
            .listen_with(player.clone(), ListeningOptions::default())
            .unwrap();
        let source = SourceSnapshot::new(
            SourceId {
                namespace: "test".into(),
                book: "".into(),
                chapter: "".into(),
            },
            "甲乙甲乙",
            &text_hash("甲乙甲乙"),
        )
        .unwrap();
        let mut plan = engine
            .plan(
                source,
                vec!["A".into(), "B".into()],
                PlaybackPolicy::AfterChapterReady,
            )
            .unwrap();
        plan.append(vec![
            SpeechSpan::new(TextRange { start: 0, end: 6 }, "A", None),
            SpeechSpan::new(TextRange { start: 6, end: 12 }, "B", None),
        ])
        .unwrap();
        plan.seal().unwrap();
        let handle = listening.control();
        handle
            .start(
                "gate",
                plan,
                PlanSessionOptions {
                    verification: options(VerificationPolicy::Gate {
                        max_retries: 0,
                        strict_suspect: false,
                    }),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let mut reports = Vec::new();
        let mut error = false;
        loop {
            let event = tokio::time::timeout(Duration::from_secs(2), listening.recv())
                .await
                .unwrap()
                .unwrap();
            match event.event {
                Event::Verification(r) => {
                    reports.push(r.verdict);
                }
                Event::Error(e) => {
                    error = true;
                    assert_eq!(e.code, "verification_failed")
                }
                Event::SegmentStarted { .. } => panic!("rejected audio played"),
                Event::SessionEnded { reason, .. } => {
                    assert_eq!(reason, EndReason::Failed);
                    break;
                }
                _ => {}
            }
        }
        assert_eq!(
            reports,
            vec![
                VerificationVerdict::Passed,
                VerificationVerdict::ConfirmedError
            ]
        );
        assert!(error);
        assert_eq!(player.queued.get(), Duration::ZERO);
        assert!(!handle.progress("gate").await.unwrap().chapter_ready);
        listening.close().await.unwrap();
    })
    .await
}

#[tokio::test]
async fn standalone_cache_binds_actual_audio_source_voice_and_style_and_does_not_cache_errors() {
    let primary = Asr::new("a", vec![Answer::Text("甲乙")]);
    let reviewer = Asr::new("b", vec![]);
    let verifier = Verifier::new(primary.clone(), reviewer.clone()).unwrap();
    let source = SourceSnapshot::new(
        SourceId {
            namespace: "test".into(),
            book: "".into(),
            chapter: "".into(),
        },
        "甲乙",
        &text_hash("甲乙"),
    )
    .unwrap();
    let mut pcm = Pcm {
        samples: vec![0.1; 800],
        sample_rate: 1000,
        channels: 1,
    };
    for (i, voice, style, hit) in [
        (0, "A", None, false),
        (1, "A", None, true),
        (2, "B", None, false),
        (3, "B", Some("轻声"), false),
    ] {
        let report = verifier
            .report(
                ReadbackRequest {
                    source: &source,
                    range: TextRange { start: 0, end: 6 },
                    spoken_text: "甲乙",
                    backend: "fake",
                    model: Some("fixed"),
                    voice,
                    style,
                    attempt: i,
                },
                &pcm,
                &VerificationOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(report.cache_hit, hit);
        assert_eq!(report.attempt, i);
    }
    assert_eq!(primary.calls.get(), 3);
    assert_eq!(reviewer.calls.get(), 0);
    pcm.samples[0] = 0.2;
    let report = verifier
        .report(
            ReadbackRequest {
                source: &source,
                range: TextRange { start: 0, end: 6 },
                spoken_text: "甲乙",
                backend: "fake",
                model: Some("fixed"),
                voice: "A",
                style: None,
                attempt: 0,
            },
            &pcm,
            &VerificationOptions::default(),
        )
        .await
        .unwrap();
    assert!(!report.cache_hit);
    assert_eq!(primary.calls.get(), 4);
    pcm.samples[1] = 0.3;
    primary.answers.borrow_mut().push_back(Answer::Error);
    for verdict in [VerificationVerdict::Unverified, VerificationVerdict::Passed] {
        let report = verifier
            .report(
                ReadbackRequest {
                    source: &source,
                    range: TextRange { start: 0, end: 6 },
                    spoken_text: "甲乙",
                    backend: "fake",
                    model: Some("fixed"),
                    voice: "A",
                    style: None,
                    attempt: 0,
                },
                &pcm,
                &VerificationOptions::default(),
            )
            .await
            .unwrap();
        assert!(!report.cache_hit);
        assert_eq!(report.verdict, verdict);
    }
}

/// A native request may remain in cleanup after its async future ends.
struct TrackedAsr {
    inner: Rc<Asr>,
    receipts: RefCell<Vec<tokio::sync::watch::Sender<bool>>>,
}
impl Recognizer for TrackedAsr {
    fn identity(&self) -> RecognizerIdentity {
        self.inner.identity()
    }
    fn transcribe(&self, audio: Arc<ReadbackAudio>) -> Recognition<'_> {
        self.inner.transcribe(audio)
    }
    fn request(&self, audio: Arc<ReadbackAudio>) -> RecognitionRequest<'_> {
        let (done, completion) = tokio::sync::watch::channel(false);
        self.receipts.borrow_mut().push(done);
        RecognitionRequest {
            future: self.transcribe(audio),
            completion: Some(completion),
        }
    }
}

async fn outstanding_report(verifier: &Verifier) {
    let source = SourceSnapshot::new(
        SourceId {
            namespace: "test".into(),
            book: "".into(),
            chapter: "external".into(),
        },
        "甲乙",
        &text_hash("甲乙"),
    )
    .unwrap();
    verifier
        .report(
            ReadbackRequest {
                source: &source,
                range: TextRange { start: 0, end: 6 },
                spoken_text: "甲乙",
                backend: "external",
                model: None,
                voice: "external",
                style: None,
                attempt: 0,
            },
            &Pcm {
                samples: vec![0.2; 800],
                sample_rate: 1000,
                channels: 1,
            },
            &VerificationOptions::default(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn decimal_readback_does_not_pass_when_fractional_zeros_change_the_value() {
    let primary = Asr::new(
        "a",
        vec![Answer::Text("一点零五"), Answer::Text("一点零五")],
    );
    let reviewer = Asr::new("b", vec![Answer::Text("一点零五")]);
    let verifier = Verifier::new(primary.clone(), reviewer.clone()).unwrap();
    let pcm = Pcm {
        samples: vec![0.1; 800],
        sample_rate: 1000,
        channels: 1,
    };
    for (text, verdict) in [
        ("1.5", VerificationVerdict::Suspect),
        ("1.05", VerificationVerdict::Passed),
    ] {
        let source = SourceSnapshot::new(
            SourceId {
                namespace: "test".into(),
                book: "".into(),
                chapter: "decimal".into(),
            },
            text,
            &text_hash(text),
        )
        .unwrap();
        let report = verifier
            .report(
                ReadbackRequest {
                    source: &source,
                    range: TextRange {
                        start: 0,
                        end: text.len(),
                    },
                    spoken_text: text,
                    backend: "external",
                    model: None,
                    voice: "external",
                    style: None,
                    attempt: 0,
                },
                &pcm,
                &VerificationOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(report.verdict, verdict);
    }
    assert_eq!(primary.calls.get(), 2);
    assert_eq!(reviewer.calls.get(), 1);
}

#[tokio::test]
async fn disabled_verification_and_engine_close_ignore_unrelated_native_work() {
    run_local(async {
        let primary = Rc::new(TrackedAsr {
            inner: Asr::new("a", vec![]),
            receipts: Default::default(),
        });
        let verifier = Rc::new(Verifier::new(primary.clone(), Asr::new("b", vec![])).unwrap());
        outstanding_report(&verifier).await;
        let mut engine = Engine::from_backend(Rc::new(Fixture::default()));
        engine.set_verifier(verifier.clone()).unwrap();
        for _ in 0..2 {
            let mut stream = engine.synthesize("甲乙", "A", None).unwrap();
            let (reports, samples, error) = drain(&mut stream).await;
            assert!(reports.is_empty() && samples > 0 && !error);
            assert_eq!(stream.state(), SynthesisState::Completed);
            assert!(stream.cancellation().is_finished());
        }
        tokio::time::timeout(Duration::from_millis(100), engine.close())
            .await
            .unwrap()
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), verifier.settled())
                .await
                .is_err()
        );
        primary.receipts.borrow()[0].send_replace(true);
        verifier.settled().await;
    })
    .await;
}

#[tokio::test]
async fn cancellation_waits_for_its_native_receipt_and_not_shared_reports() {
    run_local(async {
        let inner = Asr::new("a", vec![Answer::Text("甲乙"), Answer::Pending]);
        let primary = Rc::new(TrackedAsr {
            inner: inner.clone(),
            receipts: Default::default(),
        });
        let verifier = Rc::new(Verifier::new(primary.clone(), Asr::new("b", vec![])).unwrap());
        outstanding_report(&verifier).await;
        let mut engine = Engine::from_backend(Rc::new(Fixture::default()));
        engine.set_verifier(verifier.clone()).unwrap();
        let mut stream = engine
            .synthesize_verified(
                "甲乙",
                "A",
                None,
                VerificationOptions {
                    policy: VerificationPolicy::ReportOnly,
                    ..Default::default()
                },
            )
            .unwrap();
        while primary.receipts.borrow().len() < 2 {
            tokio::task::yield_now().await;
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(10), stream.cancel())
                .await
                .is_err()
        );
        assert!(inner.cancelled.get());
        primary.receipts.borrow()[1].send_replace(true);
        tokio::time::timeout(Duration::from_millis(100), stream.cancel())
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_millis(100), engine.close())
            .await
            .unwrap()
            .unwrap();
        assert!(!*primary.receipts.borrow()[0].borrow());
        primary.receipts.borrow()[0].send_replace(true);
        verifier.settled().await;
    })
    .await;
}

#[tokio::test]
async fn listening_close_waits_only_for_its_own_native_request() {
    run_local(async {
        let primary = Rc::new(TrackedAsr {
            inner: Asr::new("a", vec![Answer::Text("甲乙"), Answer::Pending]),
            receipts: Default::default(),
        });
        let verifier = Rc::new(Verifier::new(primary.clone(), Asr::new("b", vec![])).unwrap());
        outstanding_report(&verifier).await;
        let mut engine = Engine::from_backend(Rc::new(Fixture::default()));
        engine.set_verifier(verifier.clone()).unwrap();
        let mut listening = engine
            .listen_with(Rc::new(Player::default()), ListeningOptions::default())
            .unwrap();
        let plan = engine
            .single_voice_plan(
                SourceSnapshot::new(
                    SourceId {
                        namespace: "test".into(),
                        book: "".into(),
                        chapter: "listen".into(),
                    },
                    "甲乙",
                    &text_hash("甲乙"),
                )
                .unwrap(),
                "A",
                None,
                PlaybackPolicy::Streaming,
            )
            .unwrap();
        listening
            .control()
            .start(
                "listening",
                plan,
                PlanSessionOptions {
                    verification: VerificationOptions {
                        policy: VerificationPolicy::ReportOnly,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        while primary.receipts.borrow().len() < 2 {
            tokio::task::yield_now().await;
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(10), listening.close())
                .await
                .is_err()
        );
        primary.receipts.borrow()[1].send_replace(true);
        tokio::time::timeout(Duration::from_millis(100), listening.close())
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(Duration::from_millis(100), engine.close())
            .await
            .unwrap()
            .unwrap();
        assert!(!*primary.receipts.borrow()[0].borrow());
        primary.receipts.borrow()[0].send_replace(true);
    })
    .await;
}
