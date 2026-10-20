//! Decode fixed generated codes with full, independent, and left-context chunks.
//! Development-only audition; does not change the streaming implementation.
use anyhow::{Context, ensure};
use candle_core::DType;
use serde_json::json;
use std::{fs, path::Path, time::Instant};
use talechime_qwen3_tts::{AudioBuffer, Qwen3TTS, VoiceClonePrompt};

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut benchmark_rounds = 1usize;
    if args.len() == 7 && args[5] == "--benchmark" {
        benchmark_rounds = args.remove(6).parse().context("benchmark rounds")?;
        args.remove(5);
        ensure!(benchmark_rounds > 0, "benchmark rounds must be positive");
    }
    ensure!(
        args.len() == 5,
        "usage: MODEL PROMPT CODES DEVICE OUTPUT_DIR [--benchmark N]"
    );
    let _profiling = talechime_qwen3_tts::profiling::init();
    let model = Qwen3TTS::from_pretrained(&args[0], talechime_qwen3_tts::parse_device(&args[3])?)?;
    let prompt = VoiceClonePrompt::load(Path::new(&args[1]), model.device())?;
    let generated: Vec<Vec<u32>> = serde_json::from_slice(&fs::read(&args[2])?)?;
    ensure!(!generated.is_empty(), "empty generated codes");
    let mut combined = prompt
        .ref_codes
        .context("ICL reference codes required")?
        .to_dtype(DType::U32)?
        .to_vec2::<u32>()?;
    let reference_len = combined.len();
    combined.extend(generated.iter().cloned());
    let directory = Path::new(&args[4]);
    fs::create_dir_all(directory)?;
    let mut records = Vec::new();
    for (name, layout) in [
        (
            "full",
            ChunkLayout {
                chunk_frames: generated.len(),
                left_context_frames: 0,
            },
        ),
        (
            "independent",
            ChunkLayout {
                chunk_frames: 10,
                left_context_frames: 0,
            },
        ),
        (
            "context25",
            ChunkLayout {
                chunk_frames: 10,
                left_context_frames: 25,
            },
        ),
    ] {
        model.device().synchronize()?;
        let samples = decode_chunked(
            &model,
            &combined,
            generated.len(),
            reference_len,
            layout,
            true,
        )?;
        model.device().synchronize()?;

        let mut decode_milliseconds = Vec::new();
        for _ in 0..benchmark_rounds {
            model.device().synchronize()?;
            let started = Instant::now();
            decode_chunked(
                &model,
                &combined,
                generated.len(),
                reference_len,
                layout,
                false,
            )?;
            model.device().synchronize()?;
            decode_milliseconds.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        decode_milliseconds.sort_by(|a, b| a.total_cmp(b));
        let seconds = decode_milliseconds.iter().sum::<f64>() / 1000.0;
        ensure!(samples.len() == generated.len() * 1920, "incomplete output");
        ensure!(samples.iter().all(|v| v.is_finite()), "non-finite audio");
        let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(directory.join(format!("{name}.f32le")), bytes)?;
        AudioBuffer::new(samples, 24000).save(directory.join(format!("{name}.wav")))?;
        records.push(json!({"name": name, "decode_seconds": seconds,
            "chunk_frames": layout.chunk_frames,
            "left_context_frames": layout.left_context_frames,
            "frames": generated.len(), "reference_frames": reference_len,
            "benchmark_rounds": benchmark_rounds, "decode_ms_samples": decode_milliseconds}));
    }
    fs::write(
        directory.join("summary.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
struct ChunkLayout {
    chunk_frames: usize,
    left_context_frames: usize,
}

fn decode_chunked(
    model: &Qwen3TTS,
    combined: &[Vec<u32>],
    generated_frames: usize,
    reference_len: usize,
    layout: ChunkLayout,
    collect_samples: bool,
) -> anyhow::Result<Vec<f32>> {
    let mut samples = Vec::new();
    for start in (0..generated_frames).step_by(layout.chunk_frames) {
        let end = (start + layout.chunk_frames).min(generated_frames);
        let absolute_start = reference_len + start;
        let context_start = if start == 0 {
            0
        } else {
            absolute_start.saturating_sub(layout.left_context_frames)
        };
        let frames = &combined[context_start..reference_len + end];
        let audio = model.decode_codes(frames)?;
        ensure!(audio.sample_rate == 24000, "unexpected sample rate");
        ensure!(
            audio.len() == frames.len() * 1920,
            "unexpected samples per frame"
        );
        if collect_samples {
            let cut = (absolute_start - context_start) * 1920;
            samples.extend_from_slice(&audio.samples[cut..]);
        }
    }
    Ok(samples)
}
