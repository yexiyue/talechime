//! Real-model probe. Arguments: `MODEL PROMPT TEXT DEVICE OUTPUT_DIR [ROUNDS] [CHUNK_FRAMES]`.
//! Round zero warms both paths; WAV/raw F32 exports happen outside timing.
use anyhow::{Context, ensure};
use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, SynthesisOptions, VoiceClonePrompt};
use serde_json::json;
use std::{fs, path::Path, time::Instant};

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
    let _profiling = qwen3_tts::profiling::init();
    let model = Qwen3TTS::from_pretrained(&args[0], qwen3_tts::parse_device(&args[3])?)?;
    let prompt = VoiceClonePrompt::load(Path::new(&args[1]), model.device())?;
    let options = SynthesisOptions {
        seed: Some(42),
        chunk_frames,
        ..Default::default()
    };
    let mut records = Vec::new();
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
    }
    fs::write(
        directory.join("rounds.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    Ok(())
}
