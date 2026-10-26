//! Explicit Candle Nano diagnostic; resources must already exist locally.
use std::{path::PathBuf, time::Instant};
use talechime_core::backend::AudioChunk;
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 4,
        "moss_probe RESOURCE_ROOT TEXT_FILE OUTPUT.wav"
    );
    let text = std::fs::read_to_string(&args[2])?;
    let backend = talechime_backends::moss::nano::NanoBackend::load_on(
        PathBuf::from(&args[1]),
        talechime_protocol::Device::Cpu,
    )
    .await?;
    let start = Instant::now();
    let mut stream = backend
        .stream_seeded_with_context(&text, "Weiguo", 42, None)
        .await?;
    let mut wav = hound::WavWriter::create(
        &args[3],
        hound::WavSpec {
            channels: 2,
            sample_rate: 48000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    let mut count = 0usize;
    let mut first = None;
    let mut ended = false;
    while let Some(chunk) = stream.recv().await {
        match chunk? {
            AudioChunk::Pcm(pcm) => {
                pcm.duration_ms()?;
                first.get_or_insert(start.elapsed().as_secs_f64());
                count += pcm.samples.len();
                for sample in pcm.samples {
                    wav.write_sample(sample)?;
                }
            }
            AudioChunk::End => {
                ended = true;
                break;
            }
        }
    }
    anyhow::ensure!(ended && count > 0, "missing audio or completion");
    wav.finalize()?;
    println!(
        "{}",
        serde_json::json!({"model":"nano-candle","samples":count,"audio_seconds":count as f64 / 96000.0,"first_pcm_seconds":first,"generation_seconds":start.elapsed().as_secs_f64(),"eos":true})
    );
    Ok(())
}
