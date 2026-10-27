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
use talechime_protocol::Capabilities;
use tokio::sync::mpsc;

#[derive(Debug, PartialEq)]
struct Call {
    text: String,
    seed: u64,
    params: Vec<(String, talechime_protocol::ParamValue)>,
}

#[derive(Default)]
struct Fixture {
    calls: RefCell<Vec<Call>>,
    estimates: RefCell<Vec<(String, String, f64)>>,
    missing_end: Cell<bool>,
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
            parameters: vec![talechime_protocol::ParameterSpec {
                name: "gain".into(),
                kind: talechime_protocol::ParamKind::Int { min: 0, max: 10 },
                default: talechime_protocol::ParamValue::Int(3),
                description: "fixture gain".into(),
            }],
        }
    }
    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            self.calls.borrow_mut().push(Call {
                text: request.text.into(),
                seed: request.seed,
                params: request
                    .params
                    .iter()
                    .map(|(name, value)| (name.to_owned(), value.clone()))
                    .collect(),
            });
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
    fn next_segment<'a>(
        &'a self,
        request: talechime_core::backend::SegmentationRequest<'a>,
    ) -> Segmentation<'a> {
        let text = request.text;
        self.estimates.borrow_mut().push((
            text.into(),
            request.voice.into(),
            request.duration.seconds("甲乙丙丁"),
        ));
        Box::pin(async move {
            let start = text
                .char_indices()
                .find(|(_, c)| !c.is_whitespace())
                .map_or(text.len(), |(i, _)| i);
            if start == text.len() {
                return Ok(None);
            }
            let end = text[start..]
                .find('。')
                .map_or(text.len(), |i| start + i + 3);
            Ok(Some(TextSegment {
                text: text[start..end].into(),
                start,
                end,
            }))
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
fn start(backend: Rc<Fixture>, text: &str) -> SynthesisStream {
    SynthesisStream::start_with_options(backend, text, "a", None, None, SynthesisOptions::default())
        .unwrap()
}

#[tokio::test]
async fn concurrent_executions_and_cancelled_or_disconnected_streams_are_isolated() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let backend = Rc::new(Fixture::default());
            let mut a = start(backend.clone(), "甲。乙。");
            let mut b = start(backend.clone(), "丙。丁。");
            let (a, b) = tokio::join!(drain(&mut a), drain(&mut b));
            a.unwrap();
            b.unwrap();
            let mut cancelled = start(backend.clone(), "戊。己。");
            cancelled.recv().await.unwrap().unwrap();
            cancelled.cancel().await;
            backend.missing_end.set(true);
            let mut broken = start(backend.clone(), "庚。辛。");
            assert!(drain(&mut broken).await.is_err());
            assert_eq!(broken.state(), SynthesisState::Failed);
            backend.missing_end.set(false);
            let mut fresh = start(backend.clone(), "壬。");
            drain(&mut fresh).await.unwrap();
            assert_eq!(backend.calls.borrow().last().unwrap().text, "壬。");
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
async fn gate_retries_only_the_rejected_text_and_report_only_preserves_delivery() {
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
                assert_eq!(calls[1].text, calls[2].text);
                // The gated retry resamples the same text with a different seed.
                assert_ne!(calls[1].seed, calls[2].seed);
                assert_ne!(calls[0].seed, calls[1].seed);
                assert_eq!(calls[3].text, "小鸟飞向远方。");
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
            assert_eq!(backend.calls.borrow()[2].text, "小鸟飞向远方。");
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
            params.insert("gain", talechime_protocol::ParamValue::Int(5));
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
            assert!(backend.calls.borrow().iter().all(|call| call.params
                == [("gain".to_string(), talechime_protocol::ParamValue::Int(5))]));

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
            unknown.insert("temperature", talechime_protocol::ParamValue::Float(0.5));
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
            out_of_range.insert("gain", talechime_protocol::ParamValue::Int(99));
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

#[tokio::test]
async fn duration_learning_is_local_to_interleaved_executions_and_resets_after_failure() {
    tokio::task::LocalSet::new()
        .run_until(async {
            let backend = Rc::new(Fixture::default());
            let mut first = start(backend.clone(), "甲。乙。");
            let mut second = start(backend.clone(), "丙。丁。");
            let (a, b) = tokio::join!(drain(&mut first), drain(&mut second));
            a.unwrap();
            b.unwrap();
            for (text, _, estimate) in backend.estimates.borrow().iter() {
                match text.as_str() {
                    "甲。乙。" | "丙。丁。" => assert_eq!(*estimate, 1.0),
                    "乙。" | "丁。" => assert!(*estimate > 1.0),
                    _ => unreachable!(),
                }
            }
            let mut cancelled = start(backend.clone(), "戊。己。");
            cancelled.recv().await.unwrap().unwrap();
            cancelled.cancel().await;
            backend.missing_end.set(true);
            let mut broken = start(backend.clone(), "庚。辛。");
            assert!(drain(&mut broken).await.is_err());
            backend.missing_end.set(false);
            let mut fresh = start(backend.clone(), "壬。癸。");
            drain(&mut fresh).await.unwrap();
            assert_eq!(
                backend
                    .estimates
                    .borrow()
                    .iter()
                    .find(|(text, _, _)| text == "壬。癸。")
                    .unwrap()
                    .2,
                1.0
            );
        })
        .await;
}
#[tokio::test]
async fn explicit_segmentation_request_is_stable_without_prior_voice_selection() {
    let backend = Fixture::default();
    let estimate = talechime_core::text::duration::DurationEstimator::default();
    let request = talechime_core::backend::SegmentationRequest {
        text: "甲。乙。",
        voice: "b",
        duration: &estimate,
    };
    let first = backend.next_segment(request).await.unwrap().unwrap();
    let second = backend.next_segment(request).await.unwrap().unwrap();
    assert_eq!(
        (first.text, first.start, first.end),
        (second.text, second.start, second.end)
    );
}
