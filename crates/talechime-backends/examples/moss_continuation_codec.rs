//! Explicit real-weight codec warmup comparison, without Python or user audio.
use talechime_backends::moss::MossBackend;
use tts_core::{SpeechContext, backend::Pcm};
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(args.len() == 2, "verified Nano directory");
    let backend = MossBackend::load(args[1].clone().into()).await?;
    let text = "晚风吹过树林，远处的灯光照亮了回家的小路。";
    let mut stream = backend.stream_seeded(text, "Weiguo", Some(42)).await?;
    let mut audio = Pcm {
        samples: vec![],
        sample_rate: 48000,
        channels: 2,
    };
    let mut ended = false;
    while let Some(chunk) = stream.recv().await {
        match chunk? {
            tts_core::backend::AudioChunk::Pcm(pcm) => audio.samples.extend(pcm.samples),
            tts_core::backend::AudioChunk::End => {
                ended = true;
                break;
            }
        }
    }
    anyhow::ensure!(ended, "missing End");
    let context = SpeechContext::new(text, audio)?;
    let report = backend.check_continuation_codec(&context).await?;
    println!("{}", serde_json::to_string(&report)?);
    anyhow::ensure!(
        report.max_abs_difference < 1e-4,
        "codec warmup differs from full-prefix control"
    );
    Ok(())
}
