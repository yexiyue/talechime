//! Export a bounded fixed-reference preview without opening a playback device.
use std::{path::PathBuf, time::Instant};
use talechime::{Device, Engine, ModelOptions, SeedPolicy, SynthesisOptions, run_local};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    anyhow::ensure!(
        args.len() == 5,
        "usage: voice_candidates RESOURCES LIBRARY VOICE TEXT_FILE OUTPUT_WAV"
    );
    let text = std::fs::read_to_string(&args[3])?;
    let mut options = ModelOptions::new("voxcpm", PathBuf::from(&args[0]));
    options.device = Device::Metal;
    options.voices = Some(PathBuf::from(&args[1]));
    run_local(async {
        let start = Instant::now();
        let mut engine = Engine::prepare(options, |_| {}).await?;
        let initial_voice_present = engine.capabilities()?.voices.contains(&format!("custom:{}", args[2]));
        let generated = engine.synthesize_pcm_with_options(
            text, &args[2], None, 64 * 1024 * 1024,
            SynthesisOptions { seed: SeedPolicy::Pinned(7), ..Default::default() },
        ).await;
        engine.close().await?;
        let pcm = generated?;
        anyhow::ensure!(!pcm.samples.is_empty() && pcm.samples.iter().all(|v| v.is_finite()), "invalid PCM");
        let mut wav = hound::WavWriter::create(&args[4], hound::WavSpec {
            channels: pcm.channels, sample_rate: pcm.sample_rate,
            bits_per_sample: 32, sample_format: hound::SampleFormat::Float,
        })?;
        for sample in &pcm.samples { wav.write_sample(*sample)?; }
        wav.finalize()?;
        println!("{}", serde_json::json!({"seconds":pcm.duration_ms()? as f64 / 1000., "elapsed_seconds":start.elapsed().as_secs_f64(), "seed":7, "device":"metal", "backend":"voxcpm", "model":"2b-q8_0", "initial_voice_present":initial_voice_present}));
        Ok(())
    }).await
}
