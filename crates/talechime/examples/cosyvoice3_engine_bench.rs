//! Explicit real-model performance probe through the Talechime Engine:
//! prepare (model load on the owner thread), per-chunk streaming arrivals
//! for repeated same-seed requests, and one collect-all request. Measures
//! the full backend path (voice reference re-extraction included), not
//! just the compute library. Never runs in tests.
//!
//! Usage: `cosyvoice3_engine_bench RESOURCE_ROOT VOICE cpu|metal
//!        [--rounds N] [--seed S] [--text TEXT] [--out FILE] [--label NAME]`
use std::time::Instant;

use anyhow::{Context, bail};
use serde_json::json;
use talechime::*;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let mut rounds = 6usize;
    let mut seed = 42u64;
    let mut text = String::new();
    let mut out: Option<std::path::PathBuf> = None;
    let mut label = String::new();
    let mut pos = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--rounds" => rounds = it.next().context("--rounds N")?.parse()?,
            "--seed" => seed = it.next().context("--seed S")?.parse()?,
            "--text" => text = it.next().context("--text TEXT")?,
            "--out" => out = Some(std::path::PathBuf::from(it.next().context("--out FILE")?)),
            "--label" => label = it.next().context("--label NAME")?,
            _ => pos.push(a),
        }
    }
    if pos.len() != 3 {
        bail!(
            "usage: cosyvoice3_engine_bench RESOURCE_ROOT VOICE cpu|metal \
             [--rounds N] [--seed S] [--text TEXT] [--out FILE] [--label NAME]"
        );
    }
    if text.is_empty() {
        text = "今天天气真不错，我们一起去公园散步吧。顺便买点水果，晚上回家做沙拉，再来一壶热茶。"
            .into();
    }
    let process_start = Instant::now();
    let mut report = json!({
        "impl": "rust-talechime-engine",
        "label": label,
        "rounds": rounds,
        "seed": seed,
        "voice": pos[1],
    });

    let result = run_local(async {
        let mut options = ModelOptions::new("cosyvoice3", &pos[0]);
        options.device = match pos[2].as_str() {
            "cpu" => Device::Cpu,
            "metal" => Device::Metal,
            "cuda" => Device::Cuda,
            other => bail!("unsupported device {other}"),
        };
        options.model = Some("0.5b".into());
        let t = Instant::now();
        let mut engine = Engine::prepare(options, |_| {}).await?;
        let prepare_seconds = t.elapsed().as_secs_f64();

        let options = SynthesisOptions {
            seed: SeedPolicy::Pinned(seed),
            ..Default::default()
        };

        // Warmup request (untimed) so later rounds measure the hot path.
        {
            let mut stream =
                engine.synthesize_with_options(text.as_str(), &pos[1], None, options.clone())?;
            while let Some(block) = stream.recv().await {
                block?;
            }
            anyhow::ensure!(stream.state() == SynthesisState::Completed);
        }

        let mut stream_rounds = Vec::new();
        for i in 0..rounds {
            let entry = (async {
                let request_start = Instant::now();
                let mut stream = engine.synthesize_with_options(
                    text.as_str(),
                    &pos[1],
                    None,
                    options.clone(),
                )?;
                let mut chunks = Vec::new();
                let mut samples = 0usize;
                while let Some(block) = stream.recv().await {
                    let block = block?;
                    let pcm = block.pcm();
                    anyhow::ensure!(pcm.sample_rate == 24000 && pcm.channels == 1);
                    anyhow::ensure!(
                        !pcm.samples.is_empty() && pcm.samples.iter().all(|v| v.is_finite()),
                        "engine produced invalid PCM"
                    );
                    chunks.push(json!({
                        "index": chunks.len(),
                        "samples": pcm.samples.len(),
                        "audio_seconds": pcm.samples.len() as f64 / 24000.0,
                        "arrival_since_request": request_start.elapsed().as_secs_f64(),
                    }));
                    samples += pcm.samples.len();
                }
                let state = stream.state();
                let total = request_start.elapsed().as_secs_f64();
                let first = chunks.first().context("no PCM chunk arrived")?;
                let audio = samples as f64 / 24000.0;
                Ok::<serde_json::Value, anyhow::Error>(json!({
                    "first_pcm_seconds": first["arrival_since_request"],
                    "chunks": chunks,
                    "chunk_count": chunks.len(),
                    "total_seconds": total,
                    "pcm_samples": samples,
                    "audio_seconds": audio,
                    "rtf": total / audio,
                    "state": format!("{state:?}"),
                    "completed": state == SynthesisState::Completed,
                }))
            })
            .await;
            let mut entry = match entry {
                Ok(v) => v,
                Err(e) => json!({ "error": e.to_string() }),
            };
            entry["round"] = i.into();
            stream_rounds.push(entry);
        }

        // One collect-all request (the non-streaming facade path).
        let t = Instant::now();
        let pcm = engine
            .synthesize_pcm(text.as_str(), &pos[1], None, 8 * 1024 * 1024)
            .await?;
        let collect_all_seconds = t.elapsed().as_secs_f64();
        anyhow::ensure!(pcm.sample_rate == 24000 && !pcm.samples.is_empty());
        engine.close().await?;

        Ok::<serde_json::Value, anyhow::Error>(json!({
            "prepare_seconds": prepare_seconds,
            "ready_since_process_start": process_start.elapsed().as_secs_f64(),
            "stream_rounds": stream_rounds,
            "collect_all": {
                "seconds": collect_all_seconds,
                "pcm_samples": pcm.samples.len(),
                "audio_seconds": pcm.samples.len() as f64 / 24000.0,
                "rtf": collect_all_seconds / (pcm.samples.len() as f64 / 24000.0),
            },
        }))
    })
    .await;

    let payload = result?;
    if let Some(v) = payload.as_object() {
        for (k, val) in v {
            report[k.clone()] = val.clone();
        }
    }
    report["wall_since_process_start"] = process_start.elapsed().as_secs_f64().into();
    let text_json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &out {
        std::fs::write(path, &text_json)?;
    }
    println!("{text_json}");
    Ok(())
}
