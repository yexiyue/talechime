//! Real-model verification probe (development only; requires converted
//! checkpoints — see `tools/tts/indextts_export_safetensors.py`).
//!
//! ```sh
//! cargo run --release -p indextts --example generate -- \
//!     --dir <converted-dir> --reference <spk.wav> --text "你好，世界。" \
//!     --lang zh --out out.wav [--seed 7] [--device cuda]
//! ```
//!
//! Writes the synthesized 22.05 kHz 16-bit PCM WAV plus a sidecar JSON with
//! per-stage timings, mirroring how the voxcpm probe reports rounds.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Parser)]
struct Args {
    /// Converted checkpoint directory (config.json + safetensors).
    #[arg(long)]
    dir: PathBuf,
    /// Speaker reference WAV (any sample rate; mono or stereo averaged).
    #[arg(long)]
    reference: PathBuf,
    /// Text to synthesize.
    #[arg(long)]
    text: String,
    /// Language tag: zh / en / ja / ...
    #[arg(long, default_value = "zh")]
    lang: String,
    /// Output WAV path.
    #[arg(long, default_value = "indextts-out.wav")]
    out: PathBuf,
    #[arg(long)]
    seed: Option<u64>,
    /// Sampling temperature etc. via GenerateConfig defaults.
    #[arg(long, default_value = "1.0")]
    duration_factor: f32,
    /// Optional separate emotion reference WAV.
    #[arg(long)]
    emo_reference: Option<PathBuf>,
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = match args.device.as_str() {
        "cpu" => candle_core::Device::Cpu,
        "cuda" => candle_core::Device::cuda_if_available(0)?,
        other => bail!("unsupported device {other}"),
    };

    let t0 = Instant::now();
    let mut model = indextts::Model::load(&args.dir, &device)
        .with_context(|| format!("load model from {}", args.dir.display()))?;
    let load_secs = t0.elapsed().as_secs_f64();

    let (pcm, sr) = indextts::dsp::load_wav_mono(&args.reference)?;
    let t1 = Instant::now();
    let reference = indextts::Reference::from_pcm(&pcm, sr)?;
    model.prepare_reference(&reference)?;
    if let Some(emo_path) = &args.emo_reference {
        let (epcm, esr) = indextts::dsp::load_wav_mono(emo_path)?;
        model.prepare_emo_reference(&indextts::Reference::from_pcm(&epcm, esr)?)?;
    }
    let ref_secs = t1.elapsed().as_secs_f64();

    let options = indextts::Options {
        seed: args.seed,
        duration_factor: args.duration_factor,
        ..Default::default()
    };
    let t2 = Instant::now();
    let outcome = model.generate(&args.text, &args.lang, &options)?;
    let gen_secs = t2.elapsed().as_secs_f64();
    let audio_secs = outcome.samples.len() as f64 / outcome.sample_rate as f64;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: outcome.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&args.out, spec)
        .with_context(|| format!("create {}", args.out.display()))?;
    for s in &outcome.samples {
        let v = s.clamp(-32767.0, 32767.0).round() as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;

    let report = serde_json::json!({
        "load_secs": load_secs,
        "reference_secs": ref_secs,
        "generate_secs": gen_secs,
        "audio_secs": audio_secs,
        "rtf": gen_secs / audio_secs.max(1e-6),
        "segments": outcome.segments,
        "sample_rate": outcome.sample_rate,
        "seed": args.seed,
    });
    let sidecar = args.out.with_extension("json");
    std::fs::write(&sidecar, serde_json::to_string_pretty(&report)?)?;
    println!("wrote {} and {}", args.out.display(), sidecar.display());
    println!("{report}");
    Ok(())
}
