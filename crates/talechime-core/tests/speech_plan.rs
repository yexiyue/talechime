use talechime_core::{
    PlanError, PlanState, PlaybackPolicy, SourceSnapshot, SpeechPlan, SpeechSpan, VoiceSnapshot,
};
use tts_protocol::{Capabilities, SourceId, TextRange, text_hash};

fn capabilities(style: bool) -> Capabilities {
    Capabilities {
        backend: "fixture".into(),
        model: Some("test-model".into()),
        model_name: "Offline fixture".into(),
        compiled_devices: vec![],
        default_voice: "A".into(),
        voice_names: Default::default(),
        voices: vec!["A".into(), "B".into(), "unselected".into()],
        native_streaming: true,
        style,
        cloning: false,
        pronunciation: false,
        continuation: false,
    }
}

fn source(text: &str) -> SourceSnapshot {
    SourceSnapshot::new(
        SourceId {
            namespace: "test".into(),
            book: "book".into(),
            chapter: "chapter".into(),
        },
        text,
        &text_hash(text),
    )
    .unwrap()
}

fn voices(style: bool) -> VoiceSnapshot {
    VoiceSnapshot::new(
        "fixture",
        Some("test-model"),
        &capabilities(style),
        vec!["A".into(), "B".into()],
    )
    .unwrap()
}

fn plan(text: &str) -> SpeechPlan {
    SpeechPlan::new(source(text), voices(true), PlaybackPolicy::Streaming)
}

fn span(start: usize, end: usize, voice: &str) -> SpeechSpan {
    SpeechSpan::new(TextRange { start, end }, voice, None)
}

#[test]
fn snapshot_preserves_exact_text_and_rejects_other_digest() {
    let text = "\u{feff}中\r\n🙂e\u{301}\r \t";
    let snapshot = source(text);
    assert_eq!(snapshot.text(), text);
    assert_eq!(snapshot.hash(), text_hash(text));
    assert_eq!(snapshot.source().chapter, "chapter");
    assert!(matches!(
        SourceSnapshot::new(
            snapshot.source().clone(),
            text,
            &text_hash(&text.replace('\r', ""))
        ),
        Err(PlanError::DigestMismatch)
    ));
}

#[test]
fn whole_and_incremental_a_b_a_plans_are_equivalent() {
    let text = "中🙂文";
    let batch = vec![span(0, 3, "A"), span(3, 7, "B"), span(7, 10, "A")];
    let whole = SpeechPlan::complete(
        source(text),
        voices(true),
        PlaybackPolicy::Streaming,
        batch.clone(),
    )
    .unwrap();
    let mut incremental = plan(text);
    for assignment in batch {
        incremental.append(vec![assignment]).unwrap();
    }
    incremental.seal().unwrap();
    assert_eq!(incremental, whole);
    assert_eq!(
        whole
            .spans()
            .iter()
            .map(SpeechSpan::voice)
            .collect::<Vec<_>>(),
        ["A", "B", "A"]
    );
    assert_eq!(whole.accepted_end(), text.len());
}

#[test]
fn single_voice_uses_the_complete_plan_path() {
    for text in ["", " \r\n\t", "中文🙂"] {
        let spans = if text.is_empty() {
            vec![]
        } else {
            vec![span(0, text.len(), "A")]
        };
        let expected = SpeechPlan::complete(
            source(text),
            voices(true),
            PlaybackPolicy::AfterChapterReady,
            spans,
        )
        .unwrap();
        let actual = SpeechPlan::single_voice(
            source(text),
            voices(true),
            PlaybackPolicy::AfterChapterReady,
            "A",
            None,
        )
        .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(actual.state(), PlanState::Sealed);
        assert_eq!(actual.playback_policy(), PlaybackPolicy::AfterChapterReady);
    }
}

#[test]
fn a_bad_later_span_leaves_the_entire_batch_unaccepted() {
    let mut plan = plan("abcdef");
    plan.append(vec![span(0, 1, "A")]).unwrap();
    let before = plan.clone();
    assert!(matches!(
        plan.append(vec![span(1, 3, "B"), span(3, 6, "unselected")]),
        Err(PlanError::VoiceNotSelected(_))
    ));
    assert_eq!(plan, before);
    plan.append(vec![span(1, 6, "B")]).unwrap();
    plan.seal().unwrap();
}

#[test]
fn invalid_utf8_empty_reversed_and_out_of_bounds_ranges_are_rejected() {
    for range in [
        TextRange { start: 0, end: 1 },
        TextRange { start: 0, end: 4 },
        TextRange { start: 3, end: 3 },
        TextRange { start: 3, end: 0 },
        TextRange {
            start: 0,
            end: usize::MAX,
        },
    ] {
        let mut plan = plan("中🙂");
        let before = plan.clone();
        assert!(matches!(
            plan.append(vec![SpeechSpan::new(range, "A", None)]),
            Err(PlanError::InvalidRange(_))
        ));
        assert_eq!(plan, before);
    }
    let mut plan = plan("中🙂");
    plan.append(vec![span(0, 3, "A")]).unwrap();
    assert!(matches!(
        plan.append(vec![span(4, 7, "A")]),
        Err(PlanError::InvalidRange(_))
    ));
}

#[test]
fn gaps_overlaps_duplicates_and_empty_batches_do_not_mutate_prefix() {
    let mut plan = plan("abcdef");
    plan.append(vec![span(0, 2, "A")]).unwrap();
    let before = plan.clone();
    for assignment in [span(3, 6, "A"), span(1, 6, "A"), span(0, 2, "A")] {
        assert!(matches!(
            plan.append(vec![assignment]),
            Err(PlanError::NonContiguous { expected: 2, .. })
        ));
        assert_eq!(plan, before);
    }
    assert!(matches!(plan.append(vec![]), Err(PlanError::EmptyBatch)));
    assert_eq!(plan, before);
}

#[test]
fn seal_requires_coverage_and_closes_only_input() {
    let mut plan = plan("abc");
    assert!(matches!(
        plan.seal(),
        Err(PlanError::Incomplete {
            accepted_end: 0,
            source_len: 3
        })
    ));
    assert_eq!(plan.state(), PlanState::Open);
    plan.append(vec![span(0, 3, "A")]).unwrap();
    assert_eq!(plan.state(), PlanState::Open);
    plan.seal().unwrap();
    let before = plan.clone();
    assert!(matches!(
        plan.seal(),
        Err(PlanError::NotOpen(PlanState::Sealed))
    ));
    assert!(matches!(
        plan.append(vec![]),
        Err(PlanError::NotOpen(PlanState::Sealed))
    ));
    assert!(matches!(
        plan.fail(),
        Err(PlanError::NotOpen(PlanState::Sealed))
    ));
    assert_eq!(plan, before);
}

#[test]
fn explicit_failure_retains_prefix_and_rejects_further_mutation() {
    let mut plan = plan("abc");
    plan.append(vec![span(0, 1, "A")]).unwrap();
    plan.fail().unwrap();
    let before = plan.clone();
    assert_eq!(plan.accepted_end(), 1);
    assert_eq!(plan.spans().len(), 1);
    assert!(matches!(
        plan.append(vec![span(1, 3, "B")]),
        Err(PlanError::NotOpen(PlanState::Failed))
    ));
    assert!(matches!(
        plan.seal(),
        Err(PlanError::NotOpen(PlanState::Failed))
    ));
    assert!(matches!(
        plan.fail(),
        Err(PlanError::NotOpen(PlanState::Failed))
    ));
    assert_eq!(plan, before);
}

#[test]
fn snapshot_requires_exact_model_identity_and_available_unique_voices() {
    let caps = capabilities(false);
    for (backend, model) in [
        ("other", Some("test-model")),
        ("fixture", Some("other")),
        ("fixture", None),
    ] {
        assert!(matches!(
            VoiceSnapshot::new(backend, model, &caps, vec!["A".into()]),
            Err(PlanError::ModelMismatch)
        ));
    }
    assert!(matches!(
        VoiceSnapshot::new("fixture", Some("test-model"), &caps, vec![]),
        Err(PlanError::EmptyVoices)
    ));
    assert!(matches!(
        VoiceSnapshot::new("fixture", Some("test-model"), &caps, vec!["missing".into()]),
        Err(PlanError::Settings(_))
    ));
    assert!(matches!(
        VoiceSnapshot::new(
            "fixture",
            Some("test-model"),
            &caps,
            vec!["A".into(), "A".into()]
        ),
        Err(PlanError::DuplicateVoice(_))
    ));
}

#[test]
fn legacy_none_model_is_retained_and_metadata_is_owned() {
    let mut caps = capabilities(false);
    caps.model = None;
    let snapshot = VoiceSnapshot::new("fixture", None, &caps, vec!["A".into()]).unwrap();
    caps.voices.clear();
    assert_eq!(snapshot.backend(), "fixture");
    assert_eq!(snapshot.model(), None);
    assert_eq!(snapshot.voices().collect::<Vec<_>>(), ["A"]);
    SpeechPlan::single_voice(source("中"), snapshot, PlaybackPolicy::Streaming, "A", None).unwrap();
}

#[test]
fn styles_use_existing_configuration_rules_and_invalid_style_is_atomic() {
    for style in [String::new(), " \t".into(), "中".repeat(201)] {
        let mut plan = plan("abc");
        let before = plan.clone();
        assert!(matches!(
            plan.append(vec![SpeechSpan::new(
                TextRange { start: 0, end: 3 },
                "A",
                Some(style)
            )]),
            Err(PlanError::Settings(_))
        ));
        assert_eq!(plan, before);
    }
    let mut plan = plan("abc");
    plan.append(vec![SpeechSpan::new(
        TextRange { start: 0, end: 3 },
        "A",
        Some("中".repeat(200)),
    )])
    .unwrap();
    assert_eq!(plan.spans()[0].style(), Some("中".repeat(200).as_str()));
    let mut unsupported = SpeechPlan::new(source("abc"), voices(false), PlaybackPolicy::Streaming);
    assert!(matches!(
        unsupported.append(vec![SpeechSpan::new(
            TextRange { start: 0, end: 3 },
            "A",
            Some("calm".into())
        )]),
        Err(PlanError::Settings(_))
    ));
    unsupported.append(vec![span(0, 3, "A")]).unwrap();
}

#[test]
fn empty_chapter_still_validates_requested_single_voice_and_style() {
    assert!(matches!(
        SpeechPlan::single_voice(
            source(""),
            voices(false),
            PlaybackPolicy::Streaming,
            "unselected",
            None
        ),
        Err(PlanError::VoiceNotSelected(_))
    ));
    assert!(matches!(
        SpeechPlan::single_voice(
            source(""),
            voices(false),
            PlaybackPolicy::Streaming,
            "A",
            Some("calm".into())
        ),
        Err(PlanError::Settings(_))
    ));
    let mut empty = plan("");
    empty.seal().unwrap();
    assert!(empty.spans().is_empty());
}

#[test]
fn complete_constructor_rejects_incomplete_source_coverage() {
    assert!(matches!(
        SpeechPlan::complete(
            source("abc"),
            voices(true),
            PlaybackPolicy::Streaming,
            vec![span(0, 2, "A")]
        ),
        Err(PlanError::Incomplete {
            accepted_end: 2,
            source_len: 3
        })
    ));
}

#[test]
fn resume_requires_a_utf8_boundary_within_accepted_input() {
    let mut plan = plan("甲🙂乙");
    plan.append(vec![span(0, 3, "A")]).unwrap();
    for byte in [0, 3] {
        plan.validate_resume_byte(byte).unwrap();
    }
    for byte in [1, 4, 7, 11] {
        assert!(matches!(
            plan.validate_resume_byte(byte),
            Err(PlanError::InvalidResume { .. })
        ));
    }
    plan.append(vec![span(3, 10, "B")]).unwrap();
    plan.seal().unwrap();
    for byte in [0, 3, 7, 10] {
        plan.validate_resume_byte(byte).unwrap();
    }
    assert!(plan.validate_resume_byte(5).is_err());
}
