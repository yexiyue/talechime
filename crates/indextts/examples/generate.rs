//! Real-model verification probe (development only; requires converted
//! checkpoints — see the retired IndexTTS exporter in Git revision `e591c94`).
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

use anyhow::{Context, Result};
use clap::Parser;

#[path = "support/probe.rs"]
mod probe;

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
    #[arg(long)]
    no_text_normalization: bool,
    /// Experiment: decode semantic codes with a CPU codec copy.
    #[arg(long)]
    decode_on_cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = probe::device(&args.device)?;

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
        text_normalization: !args.no_text_normalization,
        decode_on_cpu: args.decode_on_cpu,
        ..Default::default()
    };
    let t2 = Instant::now();
    let outcome = model.generate(&args.text, &args.lang, &options)?;
    let gen_secs = t2.elapsed().as_secs_f64();
    let audio_secs = outcome.samples.len() as f64 / outcome.sample_rate as f64;

    probe::write_wav(&args.out, &outcome.samples, outcome.sample_rate)?;

    let report = serde_json::json!({
        "load_secs": load_secs,
        "reference_secs": ref_secs,
        "generate_secs": gen_secs,
        "audio_secs": audio_secs,
        "rtf": gen_secs / audio_secs.max(1e-6),
        "segments": outcome.segments,
        "sample_rate": outcome.sample_rate,
        "seed": args.seed,
        "device": args.device,
        "text_normalization": !args.no_text_normalization,
        "decode_on_cpu": args.decode_on_cpu,
    });
    let sidecar = args.out.with_extension("json");
    std::fs::write(&sidecar, serde_json::to_string_pretty(&report)?)?;
    println!("wrote {} and {}", args.out.display(), sidecar.display());
    println!("{report}");
    Ok(())
}
