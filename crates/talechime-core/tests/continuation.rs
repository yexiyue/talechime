use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    rc::Rc,
    sync::Arc,
};
use talechime_core::{
    SpeechContext, SynthesisOptions, SynthesisState, SynthesisStream,
    backend::{AudioChunk, Backend, BackendError, Pcm, Segmentation, Streaming},
    text::TextSegment,
    verification::*,
};
use tokio::sync::mpsc;
use tts_protocol::Capabilities;

type Call = (String, Option<(String, f32)>);

#[derive(Default)]
struct Fixture {
    calls: RefCell<Vec<Call>>,
    missing_end: Cell<bool>,
    reject_context: Cell<bool>,
    seconds: Cell<usize>,
}
impl Backend for Fixture {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            backend: "fixture".into(),
            model: None,
            model_name: String::new(),
            voices: vec!["a".into(), "b".into()],
            default_voice: "a".into(),
            voice_names: Default::default(),
            compiled_devices: vec![],
            native_streaming: true,
            style: true,
            cloning: false,
            pronunciation: false,
            continuation: true,
        }
    }
    fn stream<'a>(&'a self, text: &'a str, voice: &'a str) -> Streaming<'a> {
        self.stream_with_context(text, voice, None, None)
    }
    fn stream_with_context<'a>(
        &'a self,
        text: &'a str,
        _voice: &'a str,
        _style: Option<&'a str>,
        context: Option<&'a SpeechContext>,
    ) -> Streaming<'a> {
        Box::pin(async move {
            let previous = context.map(|c| (c.text().to_owned(), c.pcm().samples[0]));
            self.calls.borrow_mut().push((text.into(), previous));
            if context.is_some() && self.reject_context.get() {
                return Err(BackendError::Synthesis("reference encoding failed".into()));
            }
            let value = self.calls.borrow().len() as f32 / 100.0;
            let count = self.seconds.get().max(1) * 1000;
            let (tx, rx) = mpsc::channel(2);
            tx.try_send(Ok(AudioChunk::Pcm(Pcm {
                samples: vec![value; count],
                sample_rate: 1000,
                channels: 1,
            })))
            .unwrap();
            if !self.missing_end.get() {
                tx.try_send(Ok(AudioChunk::End)).unwrap();
            }
            Ok(rx)
        })
    }
    fn segments<'a>(&'a self, text: &'a str) -> Segmentation<'a> {
        Box::pin(async move {
            let start = text
                .char_indices()
                .find(|(_, c)| !c.is_whitespace())
                .map_or(text.len(), |(i, _)| i);
            if start == text.len() {
                return Ok(vec![]);
            }
            let end = text[start..]
                .find('。')
                .map_or(text.len(), |i| start + i + 3);
            Ok(vec![TextSegment {
                text: text[start..end].into(),
                start,
                end,
            }])
        })
    }
}
async fn drain(stream: &mut SynthesisStream) -> Result<(), BackendError> {
    while let Some(item) = stream.recv().await {
        item.map_err(|e| BackendError::Synthesis(e.to_string()))?;
    }
    assert_eq!(stream.state(), SynthesisState::Completed);
    Ok(())
}
fn start(backend: Rc<Fixture>, text: &str, continuation: bool) -> SynthesisStream {
    SynthesisStream::start_with_options(
        backend,
        text,
        "a",
        None,
        None,
        SynthesisOptions {
            continuation,
            ..Default::default()
        },
    )
    .unwrap()
}
#[tokio::test]
async fn defaults_soft_lines_hard_paragraphs_and_independent_calls() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let backend = Rc::new(Fixture::default());
            let mut stream =
                SynthesisStream::start(backend.clone(), "甲。\n乙。\n\n丙。丁。", "a", None)
                    .unwrap();
            drain(&mut stream).await.unwrap();
            {
                let calls = backend.calls.borrow();
                assert!(calls[0].1.is_none());
                assert_eq!(calls[1].1, Some(("甲。".into(), 0.01)));
                assert!(calls[2].1.is_none());
                assert_eq!(calls[3].1, Some(("丙。".into(), 0.03)));
            }
            let mut next = start(backend.clone(), "戊。己。", true);
            drain(&mut next).await.unwrap();
            assert!(backend.calls.borrow()[4].1.is_none());
            let mut off = start(backend.clone(), "庚。辛。", false);
            drain(&mut off).await.unwrap();
            assert!(backend.calls.borrow()[6..].iter().all(|c| c.1.is_none()));
        })
        .await;
}
#[tokio::test]
async fn concurrent_executions_and_cancelled_or_disconnected_streams_are_isolated() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let backend = Rc::new(Fixture::default());
            let mut a = start(backend.clone(), "甲。乙。", true);
            let mut b = start(backend.clone(), "丙。丁。", true);
            let (a, b) = tokio::join!(drain(&mut a), drain(&mut b));
            a.unwrap();
            b.unwrap();
            for (text, prior) in backend.calls.borrow().iter() {
                match text.as_str() {
                    "甲。" | "丙。" => assert!(prior.is_none()),
                    "乙。" => assert_eq!(prior.as_ref().unwrap().0, "甲。"),
                    "丁。" => assert_eq!(prior.as_ref().unwrap().0, "丙。"),
                    _ => unreachable!(),
                }
            }
            let mut cancelled = start(backend.clone(), "戊。己。", true);
            cancelled.recv().await.unwrap().unwrap();
            cancelled.cancel().await;
            backend.missing_end.set(true);
            let mut broken = start(backend.clone(), "庚。辛。", true);
            assert!(drain(&mut broken).await.is_err());
            assert_eq!(broken.state(), SynthesisState::Failed);
            backend.missing_end.set(false);
            let mut fresh = start(backend.clone(), "壬。", true);
            drain(&mut fresh).await.unwrap();
            assert!(backend.calls.borrow().last().unwrap().1.is_none());
        })
        .await;
}
#[tokio::test]
async fn overlong_reference_resets_without_interrupting_delivery() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let backend = Rc::new(Fixture::default());
            backend.seconds.set(16);
            let mut stream = start(backend.clone(), "甲。乙。", true);
            drain(&mut stream).await.unwrap();
            assert!(backend.calls.borrow().iter().all(|c| c.1.is_none()));
        })
        .await;
}
struct Asr {
    family: &'static str,
    answers: RefCell<VecDeque<&'static str>>,
}
impl Recognizer for Asr {
    fn identity(&self) -> RecognizerIdentity {
        RecognizerIdentity {
            family: self.family.into(),
            model: "test".into(),
            revision: "test".into(),
            implementation: "test".into(),
        }
    }
    fn transcribe(&self, _: Arc<ReadbackAudio>) -> Recognition<'_> {
        Box::pin(async move {
            {
                let answer = self.answers.borrow_mut().pop_front().unwrap_or("阳光。");
                Ok(answer.into())
            }
        })
    }
}
fn verifier(main: Vec<&'static str>, second: Vec<&'static str>) -> Rc<Verifier> {
    Rc::new(
        Verifier::new(
            Rc::new(Asr {
                family: "a",
                answers: RefCell::new(main.into()),
            }),
            Rc::new(Asr {
                family: "b",
                answers: RefCell::new(second.into()),
            }),
        )
        .unwrap(),
    )
}
#[tokio::test]
async fn gate_retries_snapshot_previous_and_report_only_errors_clear_it() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let backend = Rc::new(Fixture::default());
            let options = SynthesisOptions {
                verification: VerificationOptions {
                    policy: VerificationPolicy::Gate {
                        max_retries: 1,
                        strict_suspect: false,
                    },
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut stream = SynthesisStream::start_with_options(
                backend.clone(),
                "春风吹过树林。阳光照亮河面。小鸟飞向远方。",
                "a",
                None,
                Some(verifier(
                    vec![
                        "春风吹过树林。",
                        "阳光。",
                        "阳光照亮河面。",
                        "小鸟飞向远方。",
                    ],
                    vec!["阳光。"],
                )),
                options,
            )
            .unwrap();
            drain(&mut stream).await.unwrap();
            {
                let calls = backend.calls.borrow();
                assert_eq!(calls.len(), 4);
                assert_eq!(calls[1].1, calls[2].1);
                assert_eq!(calls[3].1.as_ref().unwrap().0, "阳光照亮河面。");
                assert_eq!(calls[3].1.as_ref().unwrap().1, 0.03);
            }
            backend.calls.borrow_mut().clear();
            let mut stream = SynthesisStream::start_with_options(
                backend.clone(),
                "春风吹过树林。阳光照亮河面。小鸟飞向远方。",
                "a",
                None,
                Some(verifier(
                    vec!["春风吹过树林。", "阳光。", "小鸟飞向远方。"],
                    vec!["阳光。"],
                )),
                SynthesisOptions {
                    verification: VerificationOptions {
                        policy: VerificationPolicy::ReportOnly,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            drain(&mut stream).await.unwrap();
            assert!(backend.calls.borrow()[2].1.is_none());
        })
        .await;
}

#[tokio::test]
async fn prefix_failure_terminates_without_independent_fallback() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let backend = Rc::new(Fixture::default());
            backend.reject_context.set(true);
            let mut stream = start(backend.clone(), "甲。乙。丙。", true);
            assert!(drain(&mut stream).await.is_err());
            assert_eq!(stream.state(), SynthesisState::Failed);
            assert_eq!(backend.calls.borrow().len(), 2);
            assert!(backend.calls.borrow()[1].1.is_some());
        })
        .await;
}
