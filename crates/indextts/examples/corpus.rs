//! Corpus acceptance probe (development only; requires converted
//! checkpoints). Synthesizes every non-empty line of a UTF-8 corpus file
//! with one fixed reference voice, `--rounds` times, writing per-segment
//! WAVs plus an index.json with per-segment timings — mirroring the voxcpm
//! acceptance probe layout.
//!
//! ```sh
//! cargo run --release -p indextts --features cuda --example corpus -- \
//!     --dir <converted-dir> --reference <spk.wav> \
//!     --corpus tools/tts/acceptance-corpus.txt \
//!     --out-dir target/tts-integration/indextts --rounds 2 --seed 7 --device cuda
//! ```

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    dir: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    #[arg(long, default_value = "1")]
    rounds: usize,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for s in samples {
        writer.write_sample(s.clamp(-32767.0, 32767.0).round() as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = match args.device.as_str() {
        "cpu" => candle_core::Device::Cpu,
        "cuda" => candle_core::Device::cuda_if_available(0)?,
        other => bail!("unsupported device {other}"),
    };

    let segments: Vec<String> = BufReader::new(
        std::fs::File::open(&args.corpus)
            .with_context(|| format!("open corpus {}", args.corpus.display()))?,
    )
    .lines()
    .map_while(std::result::Result::ok)
    .map(|l| l.trim().to_string())
    .filter(|l| !l.is_empty())
    .collect();
    anyhow::ensure!(!segments.is_empty(), "corpus is empty");

    let t0 = Instant::now();
    let mut model = indextts::Model::load(&args.dir, &device)
        .with_context(|| format!("load model from {}", args.dir.display()))?;
    let load_secs = t0.elapsed().as_secs_f64();

    let (pcm, sr) = indextts::dsp::load_wav_mono(&args.reference)?;
    let reference = indextts::Reference::from_pcm(&pcm, sr)?;
    model.prepare_reference(&reference)?;

    std::fs::create_dir_all(&args.out_dir)?;
    let mut index = serde_json::Map::new();
    index.insert("load_secs".to_string(), serde_json::json!(load_secs));
    index.insert("seed".to_string(), serde_json::json!(args.seed));
    index.insert("device".to_string(), serde_json::json!(args.device));

    let mut total_audio = 0f64;
    let mut total_gen = 0f64;
    let mut round_entries = Vec::new();
    for round in 0..args.rounds {
        for (seg_idx, text) in segments.iter().enumerate() {
            let options = indextts::Options {
                seed: args.seed.map(|s| s + round as u64 * 1000 + seg_idx as u64),
                ..Default::default()
            };
            let t = Instant::now();
            let generated = model.generate(text, "zh", &options);
            let gen_secs = t.elapsed().as_secs_f64();
            // Record and continue: acceptance probes report per-segment
            // failures instead of aborting the round.
            let Ok(outcome) = generated else {
                let err = generated.err().map(|e| e.to_string()).unwrap_or_default();
                println!("round {round} seg {seg_idx}: FAILED: {err}");
                round_entries.push(serde_json::json!({
                    "round": round, "segment": seg_idx, "error": err,
                    "chars": text.chars().count(),
                }));
                continue;
            };
            let audio_secs = outcome.samples.len() as f64 / outcome.sample_rate as f64;
            let wav_name = format!("round{round:02}-seg{seg_idx:02}.wav");
            write_wav(
                &args.out_dir.join(&wav_name),
                &outcome.samples,
                outcome.sample_rate,
            )
            .with_context(|| format!("write {wav_name}"))?;
            // Objective audio stats for quick cross-run comparison.
            let rms = (outcome.samples.iter().map(|v| v * v).sum::<f32>()
                / outcome.samples.len().max(1) as f32)
                .sqrt();
            let zc = outcome
                .samples
                .windows(2)
                .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
                .count() as f64
                / outcome.samples.len().max(1) as f64;
            round_entries.push(serde_json::json!({
                "round": round, "segment": seg_idx, "wav": wav_name,
                "chars": text.chars().count(),
                "segments_synthesized": outcome.segments,
                "audio_secs": audio_secs, "generate_secs": gen_secs,
                "rtf": gen_secs / audio_secs.max(1e-6),
                "rms": rms, "zero_crossing_rate": zc,
            }));
            total_audio += audio_secs;
            total_gen += gen_secs;
            println!(
                "round {round} seg {seg_idx}: {audio_secs:.2}s audio, rtf {:.2}",
                gen_secs / audio_secs.max(1e-6)
            );
        }
    }
    index.insert("rounds".to_string(), serde_json::json!(round_entries));
    index.insert(
        "summary".to_string(),
        serde_json::json!({
            "total_audio_secs": total_audio,
            "total_generate_secs": total_gen,
            "overall_rtf": total_gen / total_audio.max(1e-6),
        }),
    );
    let index_path = args.out_dir.join("index.json");
    std::fs::write(
        &index_path,
        serde_json::to_string_pretty(&serde_json::Value::Object(index))?,
    )?;
    println!("wrote {}", index_path.display());
    Ok(())
}
