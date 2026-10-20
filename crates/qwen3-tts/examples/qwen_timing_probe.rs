//! Real-model probe. Arguments: `MODEL PROMPT TEXT DEVICE OUTPUT_DIR [ROUNDS] [CHUNK_FRAMES]`.
//! Round zero warms both paths; WAV/raw F32 exports happen outside timing.
use anyhow::{Context, ensure};
use candle_core::{DType, Device, Tensor};
use serde_json::{Value, json};
use std::{fs, path::Path, time::Instant};
use talechime_qwen3_tts::{AudioBuffer, Language, Qwen3TTS, SynthesisOptions, VoiceClonePrompt};

const LOGIT_LENGTH: usize = 3072;
const READBACK_SAMPLE_COUNT: usize = 32;

fn save_audio(directory: &Path, name: &str, audio: &AudioBuffer) -> anyhow::Result<()> {
    ensure!(!audio.samples.is_empty(), "empty audio: {name}");
    ensure!(
        audio.samples.iter().all(|v| v.is_finite()),
        "non-finite audio: {name}"
    );
    let bytes: Vec<u8> = audio.samples.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(directory.join(format!("{name}.f32le")), bytes)?;
    audio.save(directory.join(format!("{name}.wav")))?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        (5..=7).contains(&args.len()),
        "usage: MODEL PROMPT TEXT DEVICE OUTPUT_DIR [ROUNDS] [CHUNK_FRAMES]"
    );
    let rounds: usize = args.get(5).map_or(Ok(3), |s| s.parse()).context("rounds")?;
    let chunk_frames: usize = args
        .get(6)
        .map_or(Ok(10), |s| s.parse())
        .context("chunk frames")?;
    ensure!(
        rounds > 0 && chunk_frames > 0,
        "rounds and chunk frames must be positive"
    );
    let directory = Path::new(&args[4]);
    fs::create_dir_all(directory)?;
    let _profiling = talechime_qwen3_tts::profiling::init();
    let model = Qwen3TTS::from_pretrained(&args[0], talechime_qwen3_tts::parse_device(&args[3])?)?;
    let prompt = VoiceClonePrompt::load(Path::new(&args[1]), model.device())?;
    let options = SynthesisOptions {
        seed: Some(42),
        chunk_frames,
        ..Default::default()
    };
    let mut records = Vec::new();
    let device = model.device();
    let mut transfer_records = Vec::new();
    let readback_logits = device
        .is_metal()
        .then(|| logits_fixture(device, LOGIT_LENGTH))
        .transpose()?;
    for round in 0..=rounds {
        model.device().synchronize()?;
        let started = Instant::now();
        let (full, codes) = model.synthesize_voice_clone_debug(
            &args[2],
            &prompt,
            Language::Chinese,
            Some(options.clone()),
        )?;
        model.device().synchronize()?;
        let full_seconds = started.elapsed().as_secs_f64();
        ensure!(
            codes.len() < options.max_length,
            "full generation hit frame limit; EOS unverified"
        );
        save_audio(directory, &format!("full-{round}"), &full)?;
        fs::write(
            directory.join(format!("codes-{round}.json")),
            serde_json::to_vec(&codes)?,
        )?;

        model.device().synchronize()?;
        let started = Instant::now();
        let mut stream = model.synthesize_voice_clone_streaming(
            &args[2],
            &prompt,
            Language::Chinese,
            options.clone(),
        )?;
        let mut chunks = Vec::new();
        let mut samples = Vec::new();
        while let Some(audio) = stream.next_chunk()? {
            ensure!(audio.sample_rate == 24000, "unexpected stream sample rate");
            chunks.push(
                json!({"seconds": started.elapsed().as_secs_f64(), "samples": audio.samples.len()}),
            );
            samples.extend(audio.samples);
        }
        model.device().synchronize()?;
        let stream_seconds = started.elapsed().as_secs_f64();
        ensure!(stream.is_done(), "generation ended without EOS");
        let frames = stream.frames_generated();
        let sample_count = samples.len();
        ensure!(
            frames < options.max_length,
            "stream generation hit frame limit; EOS unverified"
        );
        ensure!(frames == codes.len(), "full/stream frame counts differ");
        ensure!(
            sample_count == full.samples.len(),
            "full/stream sample counts differ"
        );
        ensure!(
            sample_count == frames * 1920,
            "chunk sample coverage differs from generated frames"
        );
        save_audio(
            directory,
            &format!("stream-{round}"),
            &AudioBuffer::new(samples, 24000),
        )?;
        let record = json!({"round": round, "warmup": round == 0, "seed": 42,
            "device": args[3], "chunk_frames": chunk_frames,
            "full_seconds": full_seconds, "full_samples": full.samples.len(), "full_frames": codes.len(),
            "stream_seconds": stream_seconds, "stream_samples": sample_count, "stream_frames": frames,
            "eos": stream.is_done(), "finite": true, "chunks": chunks});
        println!("{record}");
        records.push(record);

        // Isolate a committed readback from model kernels and queue submission.
        // F32 is the production dtype after `sample`; BF16 checks transfer size.
        if let Some(logits) = &readback_logits {
            transfer_records.extend(measure_readback(device, round, logits)?);
        }
    }
    fs::write(
        directory.join("rounds.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    fs::write(
        directory.join("transfer.json"),
        serde_json::to_vec_pretty(&transfer_records)?,
    )?;
    Ok(())
}

fn readback_milliseconds(device: &Device, logits: &Tensor, dtype: DType) -> anyhow::Result<f64> {
    device.synchronize()?;
    let started = Instant::now();
    logits.to_dtype(dtype)?.to_device(&Device::Cpu)?;
    device.synchronize()?;
    Ok(started.elapsed().as_secs_f64() * 1000.0)
}

fn measure_readback(device: &Device, round: usize, logits: &Tensor) -> anyhow::Result<Vec<Value>> {
    let mut records = Vec::new();
    for (dtype_name, dtype) in [("f32", DType::F32), ("bf16", DType::BF16)] {
        let mut samples = Vec::new();
        for _ in 0..READBACK_SAMPLE_COUNT {
            samples.push(readback_milliseconds(device, logits, dtype)?);
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        let record = json!({
            "round": round,
            "dtype": dtype_name,
            "shape": [1, LOGIT_LENGTH],
            "samples": samples,
            "min_ms": samples[0],
            "p50_ms": samples[samples.len() / 2],
            "mean_ms": samples.iter().sum::<f64>() / samples.len() as f64,
            "max_ms": *samples.last().expect("non-empty"),
        });
        println!("{record}");
        records.push(record);
    }
    Ok(records)
}

fn logits_fixture(device: &Device, vocab: usize) -> anyhow::Result<Tensor> {
    let values: Vec<f32> = (0..vocab)
        .map(|index| (index as f32 * 0.37).sin() * 12.0)
        .collect();
    Ok(Tensor::from_vec(values, (1, vocab), device)?)
}
