//! Explicit real-model preparation. Running this example may download weights.
//! Usage: `cargo run -p talechime --example prepare_model -- BACKEND RESOURCE_DIR [MODEL]`
use talechime::*;
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let backend = args
        .next()
        .ok_or("provide BACKEND and RESOURCE_DIR; this example may download weights")?;
    let resources = args.next().ok_or("provide an explicit RESOURCE_DIR")?;
    let mut options = ModelOptions::new(backend, resources);
    options.model = args.next();
    run_local(async {
        let mut engine = Engine::prepare(options, |progress| eprintln!("{progress:?}")).await?;
        let capabilities = engine.capabilities()?;
        let mut stream = engine.synthesize(
            "你好，这是直接合成接口。",
            &capabilities.default_voice,
            None,
        )?;
        let mut samples = 0;
        while let Some(block) = stream.recv().await {
            samples += block?.pcm().samples.len();
        }
        if stream.state() != SynthesisState::Completed {
            return Err("generation did not complete".into());
        }
        engine.close().await?;
        println!("generated {samples} samples without a player or checkpoints");
        Ok::<_, Box<dyn std::error::Error>>(())
    })
    .await
}
