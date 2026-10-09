//! Same-model A/B/A listening with facade assembly and a deterministic output.
#[path = "shared/fixture.rs"]
mod support;
use std::rc::Rc;
use talechime::*;
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_local(async {
        let mut engine = Engine::from_backend(Rc::new(support::Fixture::default()));
        let mut listening = engine.listen_with(
            Rc::new(support::Player::default()),
            ListeningOptions::default(),
        )?;
        let handle = listening.control();
        let mut chapter = support::plan(&engine, "甲乙丙", PlaybackPolicy::Streaming);
        chapter.append(vec![SpeechSpan::new(
            TextRange { start: 0, end: 3 },
            "A",
            None,
        )])?;
        handle
            .start("demo", chapter, PlanSessionOptions::default())
            .await?;
        let producer = tokio::task::spawn_local(async move {
            handle
                .append(
                    "demo",
                    vec![
                        SpeechSpan::new(TextRange { start: 3, end: 6 }, "B", None),
                        SpeechSpan::new(TextRange { start: 6, end: 9 }, "A", None),
                    ],
                )
                .await?;
            handle.seal("demo").await
        });
        while let Some(event) = listening.recv().await {
            match event.event {
                Event::SegmentStarted { range, .. } => {
                    println!("play {}..{}", range.start, range.end)
                }
                Event::SessionEnded { reason, .. } => {
                    println!("terminal: {reason:?}");
                    assert_eq!(reason, EndReason::Completed);
                    break;
                }
                _ => {}
            }
        }
        producer.await??;
        listening.close().await?;
        engine.close().await?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })
    .await
}
