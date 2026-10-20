//! Generate full utterances with explicit seeds/temperature for listening checks.
use anyhow::{Context, ensure};
use serde_json::json;
use std::{fs, path::Path};
use talechime_qwen3_tts::{Language, Qwen3TTS, SynthesisOptions, VoiceClonePrompt};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    ensure!(
        args.len() >= 8,
        "usage: MODEL PROMPT TEXT DEVICE OUTPUT_DIR TEMPERATURE MAX_FRAMES SEED..."
    );
    let temperature: f64 = args[5].parse().context("temperature")?;
    ensure!(
        temperature.is_finite() && temperature >= 0.0,
        "invalid temperature"
    );
    let max_length: usize = args[6].parse().context("max frames")?;
    ensure!(max_length > 0, "max frames must be positive");
    let seeds = args[7..]
        .iter()
        .map(|seed| seed.parse::<u64>().context("seed"))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let directory = Path::new(&args[4]);
    fs::create_dir_all(directory)?;
    let model = Qwen3TTS::from_pretrained(&args[0], talechime_qwen3_tts::parse_device(&args[3])?)?;
    let prompt = VoiceClonePrompt::load(Path::new(&args[1]), model.device())?;
    let mut records = Vec::new();
    for seed in seeds {
        let options = SynthesisOptions {
            seed: Some(seed),
            temperature,
            max_length,
            ..Default::default()
        };
        let (audio, codes) = model.synthesize_voice_clone_debug(
            &args[2],
            &prompt,
            Language::Chinese,
            Some(options.clone()),
        )?;
        ensure!(!codes.is_empty(), "empty generated codes");
        ensure!(
            !audio.samples.is_empty() && audio.samples.iter().all(|v| v.is_finite()),
            "invalid audio"
        );
        let name = format!("seed-{seed}");
        let raw: Vec<u8> = audio.samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(directory.join(format!("{name}.f32le")), raw)?;
        audio.save(directory.join(format!("{name}.wav")))?;
        fs::write(
            directory.join(format!("{name}-codes.json")),
            serde_json::to_vec(&codes)?,
        )?;
        records.push(json!({"seed": seed, "temperature": temperature,
            "top_k": options.top_k, "top_p": options.top_p,
            "repetition_penalty": options.repetition_penalty,
            "max_frames": max_length, "hit_frame_limit": codes.len() == max_length,
            "frames": codes.len(), "samples": audio.samples.len(), "sample_rate": audio.sample_rate}));
    }
    fs::write(
        directory.join("summary.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    Ok(())
}
