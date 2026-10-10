use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    rc::Rc,
    sync::Arc,
};
use talechime_core::{
    SynthesisOptions, SynthesisState, SynthesisStream,
    backend::{AudioChunk, Backend, BackendError, Pcm, Segmentation, Streaming},
    text::TextSegment,
    verification::*,
};
use tokio::sync::mpsc;
use tts_protocol::Capabilities;

#[derive(Debug, PartialEq)]
struct Call {
    text: String,
    previous: Option<(String, f32)>,
    seed: u64,
    params: Vec<(String, tts_protocol::ParamValue)>,
}

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
            parameters: vec![tts_protocol::ParameterSpec {
                name: "gain".into(),
                kind: tts_protocol::ParamKind::Int { min: 0, max: 10 },
                default: tts_protocol::ParamValue::Int(3),
                description: "fixture gain".into(),
            }],
        }
    }
    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            let previous = request
                .context
                .map(|c| (c.text().to_owned(), c.pcm().samples[0]));
            self.calls.borrow_mut().push(Call {
                text: request.text.into(),
                previous,
                seed: request.seed,
                params: request
                    .params
                    .iter()
                    .map(|(name, value)| (name.to_owned(), value.clone()))
                    .collect(),
            });
            if request.context.is_some() && self.reject_context.get() {
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
                assert!(calls[0].previous.is_none());
                assert_eq!(calls[1].previous, Some(("甲。".into(), 0.01)));
                assert!(calls[2].previous.is_none());
                assert_eq!(calls[3].previous, Some(("丙。".into(), 0.03)));
            }
            let mut next = start(backend.clone(), "戊。己。", true);
            drain(&mut next).await.unwrap();
            assert!(backend.calls.borrow()[4].previous.is_none());
            let mut off = start(backend.clone(), "庚。辛。", false);
            drain(&mut off).await.unwrap();
            assert!(
                backend.calls.borrow()[6..]
                    .iter()
                    .all(|c| c.previous.is_none())
            );
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
            for call in backend.calls.borrow().iter() {
                match call.text.as_str() {
                    "甲。" | "丙。" => assert!(call.previous.is_none()),
                    "乙。" => assert_eq!(call.previous.as_ref().unwrap().0, "甲。"),
                    "丁。" => assert_eq!(call.previous.as_ref().unwrap().0, "丙。"),
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
            assert!(backend.calls.borrow().last().unwrap().previous.is_none());
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
            assert!(backend.calls.borrow().iter().all(|c| c.previous.is_none()));
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
                assert_eq!(calls[1].previous, calls[2].previous);
                // The gated retry resamples: identical context, different seed.
                assert_ne!(calls[1].seed, calls[2].seed);
                assert_ne!(calls[0].seed, calls[1].seed);
                assert_eq!(calls[3].previous.as_ref().unwrap().0, "阳光照亮河面。");
                assert_eq!(calls[3].previous.as_ref().unwrap().1, 0.03);
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
            assert!(backend.calls.borrow()[2].previous.is_none());
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
            assert!(backend.calls.borrow()[1].previous.is_some());
        })
        .await;
}

#[tokio::test]
async fn pinned_seeds_reproduce_and_params_reach_the_backend() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let backend = Rc::new(Fixture::default());
            let text = "甲。乙。丙。";
            let options = |seed: u64, params: talechime_core::GenerationParams| SynthesisOptions {
                params,
                seed: talechime_core::SeedPolicy::Pinned(seed),
                ..Default::default()
            };
            let mut params = talechime_core::GenerationParams::new();
            params.insert("gain", tts_protocol::ParamValue::Int(5));
            let mut first = SynthesisStream::start_with_options(
                backend.clone(),
                text,
                "a",
                None,
                None,
                options(9, params.clone()),
            )
            .unwrap();
            drain(&mut first).await.unwrap();
            let initial: Vec<u64> = backend
                .calls
                .borrow()
                .iter()
                .map(|call| call.seed)
                .collect();
            assert!(initial.windows(2).all(|pair| pair[0] != pair[1]));
            assert!(backend.calls.borrow().iter().all(
                |call| call.params == [("gain".to_string(), tts_protocol::ParamValue::Int(5))]
            ));

            // Same pin and same plan reproduce the exact seed sequence.
            let mut repeat = SynthesisStream::start_with_options(
                backend.clone(),
                text,
                "a",
                None,
                None,
                options(9, params.clone()),
            )
            .unwrap();
            drain(&mut repeat).await.unwrap();
            let replay: Vec<u64> = backend.calls.borrow()[initial.len()..]
                .iter()
                .map(|call| call.seed)
                .collect();
            assert_eq!(initial, replay);

            // A different pin and an out-of-range value are both explicit errors.
            let mut other = SynthesisStream::start_with_options(
                backend.clone(),
                text,
                "a",
                None,
                None,
                options(10, params),
            )
            .unwrap();
            drain(&mut other).await.unwrap();
            let diverged: Vec<u64> = backend.calls.borrow()[initial.len() * 2..]
                .iter()
                .map(|call| call.seed)
                .collect();
            assert_ne!(initial, diverged);
            let mut unknown = talechime_core::GenerationParams::new();
            unknown.insert("temperature", tts_protocol::ParamValue::Float(0.5));
            assert!(
                SynthesisStream::start_with_options(
                    backend.clone(),
                    text,
                    "a",
                    None,
                    None,
                    options(9, unknown),
                )
                .is_err()
            );
            let mut out_of_range = talechime_core::GenerationParams::new();
            out_of_range.insert("gain", tts_protocol::ParamValue::Int(99));
            assert!(
                SynthesisStream::start_with_options(
                    backend.clone(),
                    text,
                    "a",
                    None,
                    None,
                    options(9, out_of_range),
                )
                .is_err()
            );
        })
        .await;
}
