#[path = "library/fixture.rs"]
mod support;
use std::{rc::Rc, time::Duration};
use support::*;
use talechime::*;

#[tokio::test]
async fn direct_synthesis_validates_styles_and_returns_original_ranges() {
    run_local(async {
        let backend = Rc::new(Fixture::default());
        let mut engine = Engine::from_backend(backend.clone());
        assert!(engine.synthesize("甲", "missing", None).is_err());
        assert!(engine.synthesize("甲", "A", Some(" ".into())).is_err());
        assert!(backend.calls.borrow().is_empty());
        let mut stream = engine
            .synthesize("甲😀\n乙", "B", Some("轻声".into()))
            .unwrap();
        assert!(matches!(
            engine.synthesize("另", "A", None),
            Err(EngineError::Busy)
        ));
        assert!(matches!(engine.close().await, Err(EngineError::Busy)));
        let mut ranges = vec![];
        while let Some(block) = stream.recv().await {
            let block = block.unwrap();
            assert!(block.pcm().duration_ms().unwrap() > 0);
            ranges.push(block.range());
        }
        assert_eq!(stream.state(), SynthesisState::Completed);
        assert!(ranges.iter().all(|r| r.is_valid("甲😀\n乙")));
        assert!(
            backend
                .calls
                .borrow()
                .iter()
                .all(|(_, voice, style)| voice == "B" && style.as_deref() == Some("轻声"))
        );
        engine.close().await.unwrap();
        assert!(matches!(engine.capabilities(), Err(EngineError::Closed)));
    })
    .await;
}
#[tokio::test]
async fn bounded_collect_discards_overflow_and_engine_can_be_reused() {
    run_local(async {
        let engine = Engine::from_backend(Rc::new(Fixture::default()));
        assert!(matches!(
            engine.synthesize_pcm("甲", "A", None, 100).await,
            Err(EngineError::CollectionLimit(100))
        ));
        let pcm = engine.synthesize_pcm("乙", "B", None, 4096).await.unwrap();
        assert_eq!(pcm.samples.len(), 800);
        assert!(matches!(
            engine.synthesize_pcm("甲", "A", None, 0).await,
            Err(EngineError::InvalidOptions(_))
        ));
    })
    .await;
}
#[tokio::test]
async fn empty_layout_completes_without_inference_or_fake_audio() {
    run_local(async {
        let backend = Rc::new(Fixture::default());
        let engine = Engine::from_backend(backend.clone());
        for text in ["", " \n\t"] {
            let mut stream = engine.synthesize(text, "A", None).unwrap();
            assert!(stream.recv().await.is_none());
            assert_eq!(stream.state(), SynthesisState::Completed);
        }
        assert!(backend.calls.borrow().is_empty());
        assert!(matches!(
            engine.synthesize_pcm("", "A", None, 1024).await,
            Err(EngineError::EmptyAudio)
        ));
    })
    .await;
}
#[tokio::test]
async fn incomplete_silent_or_changed_format_streams_fail_explicitly() {
    run_local(async {
        let backend = Rc::new(Fixture::default());
        let engine = Engine::from_backend(backend.clone());
        for mode in [Mode::Disconnect, Mode::Silent, Mode::Format] {
            backend.mode.set(mode);
            let mut stream = engine.synthesize("甲", "A", None).unwrap();
            let mut failed = false;
            while let Some(block) = stream.recv().await {
                if block.is_err() {
                    failed = true;
                }
            }
            assert!(failed);
            assert_eq!(stream.state(), SynthesisState::Failed);
        }
    })
    .await;
}
#[tokio::test]
async fn retained_pcm_applies_budget_and_cross_thread_cancel_releases_execution() {
    run_local(async {
        let backend = Rc::new(Fixture::default());
        backend.mode.set(Mode::Long);
        let engine = Engine::from_backend(backend.clone());
        let mut stream = engine.synthesize("甲", "A", None).unwrap();
        let mut retained = vec![];
        for _ in 0..75 {
            retained.push(stream.recv().await.unwrap().unwrap());
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(20), stream.recv())
                .await
                .is_err()
        );
        assert!(backend.chunks.get() < 85);
        let cancel = stream.cancellation();
        std::thread::spawn(move || cancel.cancel()).join().unwrap();
        assert!(stream.recv().await.is_none());
        assert_eq!(stream.state(), SynthesisState::Cancelled);
        drop(retained);
        backend.mode.set(Mode::Normal);
        assert!(engine.synthesize_pcm("乙", "B", None, 4096).await.is_ok());
    })
    .await;
}
#[tokio::test]
async fn dropped_stream_is_awaited_by_engine_close() {
    run_local(async {
        let backend = Rc::new(Fixture::default());
        backend.mode.set(Mode::Long);
        let mut engine = Engine::from_backend(backend);
        let stream = engine.synthesize("甲", "A", None).unwrap();
        let completed = stream.cancellation();
        drop(stream);
        engine.close().await.unwrap();
        assert!(completed.is_finished());
    })
    .await;
}
#[tokio::test]
async fn optional_checkpoints_reject_restore_without_creating_files_or_replacing_execution() {
    run_local(async {
        let engine = Engine::from_backend(Rc::new(Fixture::default()));
        let player = Rc::new(Player::default());
        let mut listening = engine
            .listen_with(player, ListeningOptions::default())
            .unwrap();
        let handle = listening.control();
        handle
            .start(
                "open",
                plan(&engine, "甲", PlaybackPolicy::Streaming),
                PlanSessionOptions::default(),
            )
            .await
            .unwrap();
        assert!(
            handle
                .start(
                    "restore",
                    plan(&engine, "甲", PlaybackPolicy::Streaming),
                    PlanSessionOptions {
                        restore_checkpoint: true,
                        ..Default::default()
                    }
                )
                .await
                .is_err()
        );
        assert_eq!(handle.progress("open").await.unwrap().accepted_end, 0);
        listening.close().await.unwrap();
        assert!(matches!(
            handle.progress("open").await,
            Err(EngineError::Closed)
        ));
    })
    .await;
}
#[tokio::test]
async fn listening_a_b_a_uses_acknowledged_controls_and_no_checkpoint_path() {
    run_local(async {
        let backend = Rc::new(Fixture::default());
        let engine = Engine::from_backend(backend.clone());
        let mut listening = engine
            .listen_with(Rc::new(Player::default()), ListeningOptions::default())
            .unwrap();
        let handle = listening.control();
        let mut chapter = plan(&engine, "甲乙丙", PlaybackPolicy::Streaming);
        chapter
            .append(vec![SpeechSpan::new(
                TextRange { start: 0, end: 3 },
                "A",
                None,
            )])
            .unwrap();
        handle
            .start("roles", chapter, PlanSessionOptions::default())
            .await
            .unwrap();
        handle
            .append(
                "roles",
                vec![
                    SpeechSpan::new(TextRange { start: 3, end: 6 }, "B", Some("柔和".into())),
                    SpeechSpan::new(TextRange { start: 6, end: 9 }, "A", None),
                ],
            )
            .await
            .unwrap();
        handle.seal("roles").await.unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while let Some(event) = listening.recv().await {
                if matches!(
                    event.event,
                    Event::SessionEnded {
                        reason: EndReason::Completed,
                        ..
                    }
                ) {
                    break;
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(handle.progress("roles").await.unwrap().played_end, 9);
        assert_eq!(
            backend
                .calls
                .borrow()
                .iter()
                .map(|(_, v, _)| v.as_str())
                .collect::<Vec<_>>(),
            ["A", "B", "A"]
        );
        listening.close().await.unwrap();
        assert!(engine.synthesize_pcm("复用", "A", None, 4096).await.is_ok());
    })
    .await;
}
#[tokio::test]
async fn close_with_full_event_queue_cancels_without_observer_deadlock() {
    run_local(async {
        let engine = Engine::from_backend(Rc::new(Fixture::default()));
        let mut listening = engine
            .listen_with(
                Rc::new(Player::default()),
                ListeningOptions {
                    event_capacity: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        let handle = listening.control();
        let mut chapter = plan(&engine, "甲", PlaybackPolicy::Streaming);
        chapter
            .append(vec![SpeechSpan::new(
                TextRange { start: 0, end: 3 },
                "A",
                None,
            )])
            .unwrap();
        chapter.seal().unwrap();
        handle
            .start("full", chapter, PlanSessionOptions::default())
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
        tokio::time::timeout(Duration::from_secs(3), listening.close())
            .await
            .unwrap()
            .unwrap();
        assert!(engine.synthesize_pcm("复用", "B", None, 4096).await.is_ok());
        assert!(matches!(handle.stop().await, Err(EngineError::Closed)));
    })
    .await;
}
#[tokio::test]
async fn dropping_event_owner_requests_cleanup_and_does_not_release_engine_early() {
    run_local(async {
        let engine = Engine::from_backend(Rc::new(Fixture::default()));
        let listening = engine
            .listen_with(Rc::new(Player::default()), ListeningOptions::default())
            .unwrap();
        let handle = listening.control();
        drop(listening);
        assert!(matches!(
            engine.synthesize("忙", "A", None),
            Err(EngineError::Busy)
        ));
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                match engine.synthesize("复用", "A", None) {
                    Ok(mut stream) => {
                        stream.cancel().await;
                        break;
                    }
                    Err(EngineError::Busy) => tokio::task::yield_now().await,
                    Err(error) => panic!("{error}"),
                }
            }
        })
        .await
        .unwrap();
        assert!(matches!(handle.stop().await, Err(EngineError::Closed)));
    })
    .await;
}
#[tokio::test(flavor = "multi_thread")]
async fn send_controls_run_from_another_runtime_thread() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<ListeningHandle>();
    send_sync::<CancellationHandle>();
    run_local(async {
        let engine = Engine::from_backend(Rc::new(Fixture::default()));
        let mut listening = engine
            .listen_with(Rc::new(Player::default()), ListeningOptions::default())
            .unwrap();
        let handle = listening.control();
        let chapter = plan(&engine, "甲", PlaybackPolicy::Streaming);
        tokio::spawn(async move {
            handle
                .start("remote", chapter, PlanSessionOptions::default())
                .await
                .unwrap();
            handle
                .append(
                    "remote",
                    vec![SpeechSpan::new(TextRange { start: 0, end: 3 }, "A", None)],
                )
                .await
                .unwrap();
            handle.seal("remote").await.unwrap();
        })
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            while let Some(event) = listening.recv().await {
                if matches!(
                    event.event,
                    Event::SessionEnded {
                        reason: EndReason::Completed,
                        ..
                    }
                ) {
                    break;
                }
            }
        })
        .await
        .unwrap();
        listening.close().await.unwrap();
    })
    .await;
}
#[tokio::test]
async fn facade_staged_playback_cleans_explicit_paths_and_checkpoint_is_optional() {
    run_local(async {
        let root = tempfile::tempdir().unwrap();
        let backend = Rc::new(Fixture::default());
        backend.mode.set(Mode::Long);
        let engine = Engine::from_backend(backend);
        let mut listening = engine
            .listen_with(
                Rc::new(Player::default()),
                ListeningOptions {
                    checkpoints: Some(root.path().join("checkpoints")),
                    ..Default::default()
                },
            )
            .unwrap();
        let handle = listening.control();
        let mut chapter = plan(&engine, "甲", PlaybackPolicy::AfterChapterReady);
        chapter
            .append(vec![SpeechSpan::new(
                TextRange { start: 0, end: 3 },
                "A",
                None,
            )])
            .unwrap();
        chapter.seal().unwrap();
        handle
            .start(
                "staged",
                chapter,
                PlanSessionOptions {
                    staging: StagingOptions {
                        directory: Some(root.path().into()),
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        handle.pause("staged").await.unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if handle.progress("staged").await.unwrap().chapter_ready {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(handle.progress("staged").await.unwrap().played_end, 0);
        listening.close().await.unwrap();
        assert_eq!(
            std::fs::read_dir(root.path().join("talechime-staging-v1"))
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| e.file_type().unwrap().is_dir())
                .count(),
            0
        );
        assert!(root.path().join("checkpoints").exists());
    })
    .await;
}
#[tokio::test]
async fn invalid_preparation_and_assembly_are_read_only() {
    run_local(async {
        let root = tempfile::tempdir().unwrap();
        assert!(matches!(
            Engine::prepare(ModelOptions::new("not-compiled", root.path()), |_| {}).await,
            Err(EngineError::Prepare(_))
        ));
        let mut options = ModelOptions::new("moss", root.path());
        options.device = Device::Auto;
        assert!(matches!(
            Engine::prepare(options, |_| {}).await,
            Err(EngineError::InvalidOptions(_))
        ));
        let engine = Engine::from_backend(Rc::new(Fixture::default()));
        assert!(
            engine
                .listen_with(
                    Rc::new(Player::default()),
                    ListeningOptions {
                        event_capacity: 0,
                        ..Default::default()
                    }
                )
                .is_err()
        );
        let (events, _receiver) = tokio::sync::mpsc::channel(1);
        let mut session = engine
            .listen_with_events(
                Rc::new(Player::default()),
                ListeningOptions {
                    event_capacity: 0,
                    ..Default::default()
                },
                events,
            )
            .unwrap();
        session.close().await.unwrap();
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    })
    .await;
}

#[tokio::test]
async fn collect_rejects_format_changes_between_valid_segments() {
    run_local(async {
        let backend = Rc::new(Fixture::default());
        backend.mode.set(Mode::SegmentFormat);
        let engine = Engine::from_backend(backend);
        assert!(matches!(
            engine.synthesize_pcm("甲\n乙", "A", None, 100_000).await,
            Err(EngineError::CollectionFormat)
        ));
    })
    .await;
}
#[tokio::test]
async fn oversized_model_block_fails_before_boundary_processing() {
    run_local(async {
        let backend = Rc::new(Fixture::default());
        backend.mode.set(Mode::Oversized);
        let engine = Engine::from_backend(backend);
        let mut stream = engine.synthesize("甲", "A", None).unwrap();
        assert!(stream.recv().await.unwrap().is_err());
        assert_eq!(stream.state(), SynthesisState::Failed);
    })
    .await;
}

#[tokio::test]
async fn acknowledged_controls_preserve_seek_pause_and_failure_identity() {
    run_local(async {
        let engine = Engine::from_backend(Rc::new(Fixture::default()));
        let mut listening = engine
            .listen_with(Rc::new(Player::default()), ListeningOptions::default())
            .unwrap();
        let handle = listening.control();
        handle
            .start(
                "old",
                plan(&engine, "甲乙", PlaybackPolicy::Streaming),
                PlanSessionOptions::default(),
            )
            .await
            .unwrap();
        handle
            .append(
                "old",
                vec![SpeechSpan::new(TextRange { start: 0, end: 3 }, "A", None)],
            )
            .await
            .unwrap();
        handle.pause("old").await.unwrap();
        assert_eq!(
            handle.status().await.unwrap(),
            (Some("old".into()), SessionState::Paused)
        );
        assert!(
            handle
                .configure_playback("old", 1.0, f32::NAN)
                .await
                .is_err()
        );
        handle.configure_playback("old", 0.5, 1.5).await.unwrap();
        handle.seek("old", "new", 0).await.unwrap();
        assert_eq!(
            handle.status().await.unwrap(),
            (Some("new".into()), SessionState::Paused)
        );
        assert!(
            handle
                .append(
                    "old",
                    vec![SpeechSpan::new(TextRange { start: 3, end: 6 }, "B", None)]
                )
                .await
                .is_err()
        );
        handle.resume("new").await.unwrap();
        handle.fail_input("new", "analysis failed").await.unwrap();
        assert_eq!(
            handle.progress("new").await.unwrap().input_state,
            PlanState::Failed
        );
        let mut failure = false;
        while let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_millis(20), listening.recv()).await
        {
            if event.session_id == "new"
                && matches!(
                    event.event,
                    Event::SessionEnded {
                        reason: EndReason::Failed,
                        ..
                    }
                )
            {
                failure = true;
                break;
            }
        }
        assert!(failure);
        handle.stop().await.unwrap();
        listening.close().await.unwrap();
        listening.close().await.unwrap();
    })
    .await;
}

#[tokio::test]
async fn engine_plan_helpers_use_prepared_identity_and_validate_empty_voice_and_style() {
    run_local(async {
        let backend = Rc::new(support::Fixture::default());
        let mut engine = Engine::from_backend(backend);
        let source = SourceSnapshot::new(
            SourceId {
                namespace: "test".into(),
                book: "book".into(),
                chapter: "one".into(),
            },
            "中🙂",
            &text_hash("中🙂"),
        )
        .unwrap();
        let mut incremental = engine
            .plan(source.clone(), vec!["A".into()], PlaybackPolicy::Streaming)
            .unwrap();
        incremental
            .append(vec![SpeechSpan::new(
                TextRange { start: 0, end: 7 },
                "A",
                None,
            )])
            .unwrap();
        incremental.seal().unwrap();
        assert_eq!(
            incremental,
            engine
                .single_voice_plan(source, "A", None, PlaybackPolicy::Streaming)
                .unwrap()
        );
        let empty = SourceSnapshot::new(
            SourceId {
                namespace: "test".into(),
                book: "book".into(),
                chapter: "empty".into(),
            },
            "",
            &text_hash(""),
        )
        .unwrap();
        assert!(
            engine
                .single_voice_plan(empty.clone(), "unknown", None, PlaybackPolicy::Streaming)
                .is_err()
        );
        assert!(
            engine
                .single_voice_plan(empty, "A", Some(" ".into()), PlaybackPolicy::Streaming)
                .is_err()
        );
        engine.close().await.unwrap();
    })
    .await;
}
