//! The host explicitly owns the thread/runtime. The library creates neither.
#[path = "shared/fixture.rs"]
mod support;
use talechime::*;
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let thread = std::thread::Builder::new()
        .name("host-speech".into())
        .spawn(
            || -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()?;
                runtime
                    .block_on(run_local(async {
                        // Rc backend/Engine are constructed and released on this host thread.
                        let mut engine =
                            Engine::from_backend(std::rc::Rc::new(support::Fixture::default()));
                        let pcm = engine
                            .synthesize_pcm("宿主拥有执行线程。", "A", None, 4096)
                            .await?;
                        engine.close().await?;
                        Ok::<_, EngineError>(pcm.samples.len())
                    }))
                    .map_err(Into::into)
            },
        )?;
    let samples = thread
        .join()
        .map_err(|_| std::io::Error::other("host speech thread panicked"))??;
    println!("host thread generated {samples} samples");
    Ok(())
}
