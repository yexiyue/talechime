//! Consume model-free PCM directly; no player, filesystem, CLI config or checkpoints.
#[path = "shared/fixture.rs"]
mod support;
use std::rc::Rc;
use talechime::{Engine, SynthesisState, run_local};
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_local(async {
        let mut engine = Engine::from_backend(Rc::new(support::Fixture::default()));
        let mut stream = engine.synthesize("你好，世界。", "A", Some("轻声".into()))?;
        let mut samples = 0;
        while let Some(block) = stream.recv().await {
            let block = block?;
            samples += block.pcm().samples.len();
            println!(
                "{}..{}: {} samples",
                block.range().start,
                block.range().end,
                block.pcm().samples.len()
            );
        }
        assert_eq!(stream.state(), SynthesisState::Completed);
        let collected = engine.synthesize_pcm("再见。", "B", None, 4096).await?;
        println!(
            "stream={samples}, bounded collection={}",
            collected.samples.len()
        );
        engine.close().await?;
        Ok::<_, Box<dyn std::error::Error>>(())
    })
    .await
}
