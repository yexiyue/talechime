//! Single-voice listening follows the same plan construction and facade controls.
#[path = "shared/fixture.rs"]
mod support;
use std::rc::Rc;
use talechime::*;
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_local(async {
        let mut engine = Engine::from_backend(Rc::new(support::Fixture::default()));
        let source = support::plan(
            &engine,
            "单音色使用相同的计划路径。",
            PlaybackPolicy::Streaming,
        );
        let chapter = engine.single_voice_plan(
            source.source().clone(),
            "A",
            None,
            PlaybackPolicy::Streaming,
        )?;
        let mut listening = engine.listen_with(
            Rc::new(support::Player::default()),
            ListeningOptions::default(),
        )?;
        listening
            .control()
            .start("single", chapter, PlanSessionOptions::default())
            .await?;
        while let Some(event) = listening.recv().await {
            if let Event::SessionEnded { reason, .. } = event.event {
                println!("single voice: {reason:?}");
                assert_eq!(reason, EndReason::Completed);
                break;
            }
        }
        listening.close().await?;
        engine.close().await?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })
    .await
}
