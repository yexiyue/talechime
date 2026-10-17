//! Fair local performance benchmark for the Candle CosyVoice3 port.
//!
//! Timing boundaries match `tools/tts/cosyvoice3_bench.py` (official
//! PyTorch) so both sides measure the same spans. No diagnostic hooks and
//! no fixture export happen inside any timed region; every number is a
//! wall-clock span around public library calls on resident models.
//!
//! Usage:
//!   `bench <model-dir> <case-dir> <cpu|metal> <cold|warm|stream|components>`
//!   `      [--rounds N] [--threads N] [--seed S] [--out FILE] [--label NAME]`
//!   `      [--reextract] [--wav-out FILE]`
//!
//! `<case-dir>` is an official-reference fixture directory produced by
//! `tools/tts/cosyvoice3_reference.py` (metadata.json, frontend/llm/flow
//! safetensors, official prompt wav). `--reextract` re-runs the ORT
//! frontend on the prompt wav each round; without it the fixture's
//! pre-extracted prompt features are reused. File I/O (JSON, WAV) stays
//! outside all timed regions.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cosyvoice3::config::SamplingConfig;
use cosyvoice3::flow::{CausalMaskedDiffWithDiT, DiT};
use cosyvoice3::llm::CosyVoice3Lm;
use cosyvoice3::pipeline::{CosyVoice3, PromptFeatures, extract_prompt_features};
use cosyvoice3::vocoder::CausalHiFtGenerator;
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    model: PathBuf,
    case: PathBuf,
    device: String,
    mode: String,
    rounds: usize,
    threads: usize,
    seed: u64,
    out: Option<PathBuf>,
    label: String,
    reextract: bool,
    wav_out: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut pos = Vec::new();
    let mut rounds = 6usize;
    let mut threads = 8usize;
    let mut seed = 42u64;
    let mut out = None;
    let mut label = String::new();
    let mut reextract = false;
    let mut wav_out = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--rounds" => rounds = it.next().context("--rounds N")?.parse()?,
            "--threads" => threads = it.next().context("--threads N")?.parse()?,
            "--seed" => seed = it.next().context("--seed S")?.parse()?,
            "--out" => out = Some(PathBuf::from(it.next().context("--out FILE")?)),
            "--label" => label = it.next().context("--label NAME")?,
            "--wav-out" => wav_out = Some(PathBuf::from(it.next().context("--wav-out FILE")?)),
            "--reextract" => reextract = true,
            _ => pos.push(a),
        }
    }
    if pos.len() != 4 {
        bail!(
            "usage: bench <model-dir> <case-dir> <cpu|metal> <cold|warm|stream|components> \
             [--rounds N] [--threads N] [--seed S] [--out FILE] [--label NAME] [--reextract] [--wav-out FILE]"
        );
    }
    Ok(Args {
        model: PathBuf::from(&pos[0]),
        case: PathBuf::from(&pos[1]),
        device: pos[2].clone(),
        mode: pos[3].clone(),
        rounds,
        threads,
        seed,
        out,
        label,
        reextract,
        wav_out,
    })
}

fn pick_device(name: &str) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "metal" => Device::new_metal(0).map_err(anyhow::Error::from),
        other => bail!("unsupported device {other}"),
    }
}

/// Metal command buffers are asynchronous; flush them before stopping a
/// stage timer. CPU needs no barrier.
fn finish_stage(dev: &Device) -> Result<()> {
    if matches!(dev, Device::Metal(_)) {
        dev.synchronize()?;
    }
    Ok(())
}

/// Progress markers for the matrix runner's RSS timeline correlation.
/// Stderr writes stay outside every timed region.
fn mark(event: &str) {
    let mut stderr = std::io::stderr();
    let _ = writeln!(stderr, "BENCH_{event}");
    let _ = stderr.flush();
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn load_case(dir: &Path) -> Result<(String, String, PromptFeatures)> {
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("metadata.json"))?)?;
    let text = meta["text"].as_str().context("missing text")?.to_owned();
    let prompt_text = meta["prompt_text"]
        .as_str()
        .context("missing prompt_text")?
        .to_owned();
    let f = candle_core::safetensors::load(dir.join("frontend.safetensors"), &Device::Cpu)?;
    let prompt = PromptFeatures {
        text: prompt_text.clone(),
        speech_tokens: f["prompt_speech_token"]
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1()?,
        mel: f["prompt_speech_feat"].flatten_all()?.to_vec1()?,
        mel_frames: f["prompt_speech_feat"].dim(1)?,
        embedding: f["prompt_embedding"].flatten_all()?.to_vec1()?,
    };
    Ok((text, prompt_text, prompt))
}

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.map(|s| s as f32 / 2f32.powi(i32::from(spec.bits_per_sample) - 1)))
            .collect::<Result<_, _>>()?,
    };
    let mono = samples
        .chunks_exact(spec.channels.max(1) as usize)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect();
    Ok((mono, spec.sample_rate))
}

fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for s in samples {
        writer.write_sample(*s)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Per-round reference prompt: freshly extracted features, or the shared
/// fixture features (cloned once outside stage timers) when `--reextract`
/// is off.
struct RoundPrompt {
    extract_seconds: Option<f64>,
    features: PromptFeatures,
}

fn round_prompt(
    frontend: &mut Option<cosyvoice3::frontend::ort_models::FrontendOnnx>,
    wav: Option<(&[f32], u32)>,
    prompt_text: &str,
    fixture: &PromptFeatures,
) -> Result<RoundPrompt> {
    match (frontend.as_mut(), wav) {
        (Some(f), Some((samples, sr))) => {
            let t = Instant::now();
            let features = extract_prompt_features(f, samples, sr, prompt_text)?;
            Ok(RoundPrompt {
                extract_seconds: Some(t.elapsed().as_secs_f64()),
                features,
            })
        }
        _ => Ok(RoundPrompt {
            extract_seconds: None,
            features: PromptFeatures {
                text: fixture.text.clone(),
                speech_tokens: fixture.speech_tokens.clone(),
                mel: fixture.mel.clone(),
                mel_frames: fixture.mel_frames,
                embedding: fixture.embedding.clone(),
            },
        }),
    }
}

/// Resident staged-execution models (public component constructors, the
/// same objects `CosyVoice3` owns; kept separate so timers can sit around
/// each stage without touching library internals).
struct Staged {
    llm: CosyVoice3Lm,
    flow: CausalMaskedDiffWithDiT,
    hift: CausalHiFtGenerator,
    cfg: cosyvoice3::CosyVoice3Config,
}

impl Staged {
    fn load(model: &Path, dev: &Device) -> Result<Self> {
        let cfg = cosyvoice3::CosyVoice3Config::from_model_dir(model)?;
        let vb = |name: &str| -> Result<VarBuilder<'static>> {
            Ok(unsafe {
                VarBuilder::from_mmaped_safetensors(&[model.join(name)], DType::F32, dev)?
            })
        };
        let llm = CosyVoice3Lm::new(&cfg.llm, vb("llm.safetensors")?)?;
        let flow_vb = vb("flow.safetensors")?;
        let dit = DiT::new(cfg.flow.dit.clone(), flow_vb.pp("dit"))?;
        let flow = CausalMaskedDiffWithDiT::new(&cfg.flow, 192, dit, flow_vb)?;
        let hift = CausalHiFtGenerator::new(cfg.hift.clone(), 0, vb("hift.safetensors")?)?;
        Ok(Self {
            llm,
            flow,
            hift,
            cfg,
        })
    }

    /// One round of stage-attributed inference: LLM prefill, LLM decode,
    /// Flow, HiFT — the same public calls the pipeline makes, with timers
    /// around each stage and a device sync before stopping each timer.
    #[allow(clippy::too_many_arguments)]
    fn round(
        &mut self,
        text_token: &Tensor,
        prompt_text_token: &Tensor,
        llm_prompt: &Tensor,
        flow_prompt: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        seed: u64,
    ) -> Result<serde_json::Value> {
        let sampling = SamplingConfig {
            seed: Some(seed),
            ..SamplingConfig::default()
        };
        let t = Instant::now();
        let mut state =
            self.llm
                .start_incremental(text_token, prompt_text_token, llm_prompt, &sampling)?;
        finish_stage(self.flow.device())?;
        let prefill = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut tokens = Vec::new();
        while let Some(token) = self.llm.next_token(&mut state)? {
            tokens.push(token);
        }
        finish_stage(self.flow.device())?;
        let decode = t.elapsed().as_secs_f64();
        let hit_limit = state.hit_limit();

        let token_t = Tensor::from_slice(&tokens, (1, tokens.len()), prompt_feat.device())?;
        let t = Instant::now();
        let mel = self.flow.inference(
            &token_t,
            flow_prompt,
            prompt_feat,
            embedding,
            self.cfg.flow.cfm.n_timesteps,
            false,
            true,
        )?;
        finish_stage(self.flow.device())?;
        let flow_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let pcm = self.hift.inference(&mel, true)?;
        finish_stage(self.flow.device())?;
        let pcm = pcm.to_dtype(DType::F32)?.squeeze(0)?.to_vec1::<f32>()?;
        let hift_s = t.elapsed().as_secs_f64();

        Ok(json!({
            "llm_prefill_seconds": prefill,
            "llm_decode_seconds": decode,
            "llm_tokens": tokens.len(),
            "llm_hit_limit": hit_limit,
            "flow_seconds": flow_s,
            "hift_seconds": hift_s,
            "stage_pcm_samples": pcm.len(),
            "stage_pcm_valid": !pcm.is_empty() && pcm.iter().all(|v| v.is_finite()),
        }))
    }
}

/// Fixed-workload component timing: identical tokens/prompt/mel tensors for
/// both implementations, loaded from the case's official fixtures.
fn components(args: &Args, dev: &Device, report: &mut serde_json::Value) -> Result<()> {
    let load = |path: PathBuf| -> Result<std::collections::HashMap<String, Tensor>> {
        let mut out = std::collections::HashMap::new();
        for (k, t) in candle_core::safetensors::load(path, &Device::Cpu)? {
            // Metal lacks integer dtype casts; convert while still on CPU.
            let t = if matches!(t.dtype(), DType::I32 | DType::I64) {
                t.to_dtype(DType::U32)?
            } else {
                t
            };
            out.insert(k, t.to_device(dev)?);
        }
        Ok(out)
    };
    let flow_fx = load(args.case.join("flow.safetensors"))?;
    let hift_fx = load(args.case.join("hift.safetensors"))?;

    let cfg = cosyvoice3::CosyVoice3Config::from_model_dir(&args.model)?;
    let flow_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[args.model.join("flow.safetensors")],
            DType::F32,
            dev,
        )?
    };
    let dit = DiT::new(cfg.flow.dit.clone(), flow_vb.pp("dit"))?;
    let flow = CausalMaskedDiffWithDiT::new(&cfg.flow, 192, dit, flow_vb)?;
    let hift_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[args.model.join("hift.safetensors")],
            DType::F32,
            dev,
        )?
    };
    let hift = CausalHiFtGenerator::new(cfg.hift.clone(), 0, hift_vb)?;

    let token = flow_fx["token"].to_dtype(DType::U32)?;
    let prompt_token = flow_fx["prompt_token"].to_dtype(DType::U32)?;
    let prompt_feat = flow_fx["prompt_feat"].clone();
    let embedding = flow_fx["embedding"].clone();
    let mel = hift_fx["mel"].clone();
    anyhow::ensure!(token.dim(1)? > 0, "empty component token fixture");

    // Warmup outside timing.
    {
        let m = flow.inference(
            &token,
            &prompt_token,
            &prompt_feat,
            &embedding,
            cfg.flow.cfm.n_timesteps,
            false,
            true,
        )?;
        hift.inference(&m, true)?;
    }

    let mut flow_rounds = Vec::new();
    for _ in 0..args.rounds {
        let t = Instant::now();
        let m = flow.inference(
            &token,
            &prompt_token,
            &prompt_feat,
            &embedding,
            cfg.flow.cfm.n_timesteps,
            false,
            true,
        )?;
        finish_stage(dev)?;
        flow_rounds.push(json!({
            "seconds": t.elapsed().as_secs_f64(),
            "mel_frames": m.dim(2)?,
        }));
    }
    let mut hift_rounds = Vec::new();
    for _ in 0..args.rounds {
        let t = Instant::now();
        let pcm = hift.inference(&mel, true)?;
        finish_stage(dev)?;
        let v = pcm.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        anyhow::ensure!(
            v.iter().all(|x| x.is_finite()),
            "component hift produced non-finite pcm"
        );
        hift_rounds.push(json!({
            "seconds": t.elapsed().as_secs_f64(),
            "pcm_samples": v.len(),
        }));
    }
    report["components"] = json!({
        "tokens": token.dim(1)?,
        "prompt_tokens": prompt_token.dim(1)?,
        "mel_frames_fixture": mel.dim(2)?,
        "flow": flow_rounds,
        "hift": hift_rounds,
    });
    Ok(())
}

#[cfg(feature = "ort")]
fn main() -> Result<()> {
    let args = parse_args()?;
    let dev = pick_device(&args.device)?;
    let process_start = Instant::now();

    let mut report = json!({
        "impl": "rust-candle",
        "candle": "0.11.0",
        "ort": "rc.13",
        "label": args.label,
        "device": args.device,
        "mode": args.mode,
        "rounds": args.rounds,
        "threads": args.threads,
        "ort_frontend_threads": args.threads,
        "seed": args.seed,
        "reextract": args.reextract,
        "started_unix_ms": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?.as_millis() as u64,
    });

    let (text, prompt_text, prompt) = load_case(&args.case)?;
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(args.case.join("metadata.json"))?)?;
    let prompt_wav = meta["prompt_wav"]
        .as_str()
        .filter(|p| Path::new(p).exists())
        .map(PathBuf::from);

    // Model load: process start -> resident models.
    let t = Instant::now();
    let mut cv = CosyVoice3::load(&args.model, &dev, DType::F32, 0)?;
    finish_stage(&dev)?;
    let pipeline_load = t.elapsed().as_secs_f64();
    report["pipeline_load_seconds"] = pipeline_load.into();

    let mut frontend = None;
    if args.reextract {
        let t = Instant::now();
        let f = cosyvoice3::frontend::ort_models::FrontendOnnx::load(&args.model, args.threads)?;
        frontend = Some(f);
        report["frontend_load_seconds"] = t.elapsed().as_secs_f64().into();
    }

    mark(&format!("READY {} {}", unix_ms(), args.mode));

    match args.mode.as_str() {
        "cold" => {
            // Cold start: load -> first complete request from raw wav+text.
            anyhow::ensure!(args.reextract, "cold mode requires --reextract");
            let wav = prompt_wav.context("case metadata has no prompt_wav")?;
            let (samples, sr) = read_wav_mono(&wav)?;
            let ready_since_process_start = process_start.elapsed().as_secs_f64();
            let sampling = SamplingConfig {
                seed: Some(args.seed),
                ..SamplingConfig::default()
            };
            let request = round_prompt(&mut frontend, Some((&samples, sr)), &prompt_text, &prompt)?;
            let extract = request.extract_seconds.context("reextract off")?;
            let t = Instant::now();
            let pcm = cv.synthesize(
                &text,
                &request.features,
                cosyvoice3::Mode::ZeroShot,
                None,
                &sampling,
                1.0,
            )?;
            finish_stage(&dev)?;
            let synth = t.elapsed().as_secs_f64();
            report["cold"] = json!({
                "frontend_load_seconds": report["frontend_load_seconds"].clone(),
                "ready_since_process_start": ready_since_process_start,
                "first_extract_seconds": extract,
                "first_synthesize_seconds": synth,
                "first_request_total_seconds": extract + synth,
                "pcm_samples": pcm.len(),
                "pcm_valid": !pcm.is_empty() && pcm.iter().all(|v| v.is_finite()),
                "audio_seconds": pcm.len() as f64 / 24000.0,
            });
        }
        "warm" => {
            let llm_fx =
                candle_core::safetensors::load(args.case.join("llm.safetensors"), &Device::Cpu)?;
            let to_dev = |name: &str| -> Result<Tensor> {
                let t = llm_fx[name].clone();
                if matches!(t.dtype(), DType::I32 | DType::I64) {
                    Ok(t.to_dtype(DType::U32)?.to_device(&dev)?)
                } else {
                    Ok(t.to_device(&dev)?)
                }
            };
            let text_token = to_dev("text_token")?;
            let prompt_text_token = to_dev("prompt_text_token")?;
            let front = candle_core::safetensors::load(
                args.case.join("frontend.safetensors"),
                &Device::Cpu,
            )?;
            let front_to_dev = |name: &str| -> Result<Tensor> {
                let t = front[name].clone();
                if matches!(t.dtype(), DType::I32 | DType::I64) {
                    Ok(t.to_dtype(DType::U32)?.to_device(&dev)?)
                } else {
                    Ok(t.to_device(&dev)?)
                }
            };
            let llm_prompt = front_to_dev("prompt_speech_token")?;
            let flow_prompt = front_to_dev("prompt_speech_token")?;
            let prompt_feat = front["prompt_speech_feat"].to_device(&dev)?;
            let embedding = front["prompt_embedding"].to_device(&dev)?;

            let t = Instant::now();
            let mut staged = Staged::load(&args.model, &dev)?;
            let staged_load = t.elapsed().as_secs_f64();
            report["staged_load_seconds"] = staged_load.into();
            report["ready_since_process_start"] = process_start.elapsed().as_secs_f64().into();

            let sampling = SamplingConfig {
                seed: Some(args.seed),
                ..SamplingConfig::default()
            };
            let (samples, sr) = prompt_wav
                .as_ref()
                .map(|w| read_wav_mono(w))
                .transpose()?
                .unwrap_or((Vec::new(), 0));
            let wav_args = if args.reextract && !samples.is_empty() {
                Some((samples.as_slice(), sr))
            } else {
                None
            };

            // Warmup both paths (untimed).
            staged.round(
                &text_token,
                &prompt_text_token,
                &llm_prompt,
                &flow_prompt,
                &prompt_feat,
                &embedding,
                args.seed,
            )?;
            let warm = round_prompt(&mut frontend, wav_args, &prompt_text, &prompt)?;
            cv.synthesize(
                &text,
                &warm.features,
                cosyvoice3::Mode::ZeroShot,
                None,
                &sampling,
                1.0,
            )?;

            let mut rounds = Vec::new();
            for i in 0..args.rounds {
                mark(&format!("ROUND_START {i} {}", unix_ms()));
                let round = (|| -> Result<serde_json::Value> {
                    let request = round_prompt(&mut frontend, wav_args, &prompt_text, &prompt)?;
                    let stages = staged.round(
                        &text_token,
                        &prompt_text_token,
                        &llm_prompt,
                        &flow_prompt,
                        &prompt_feat,
                        &embedding,
                        args.seed,
                    )?;
                    // Production-path total on the resident pipeline with
                    // features ready (same inputs, timed separately).
                    let t = Instant::now();
                    let pcm = cv.synthesize(
                        &text,
                        &request.features,
                        cosyvoice3::Mode::ZeroShot,
                        None,
                        &sampling,
                        1.0,
                    )?;
                    finish_stage(&dev)?;
                    let synthesize_s = t.elapsed().as_secs_f64();
                    let audio = pcm.len() as f64 / 24000.0;
                    if let (Some(path), true) = (&args.wav_out, i == 0) {
                        write_wav(path, &pcm)?;
                    }
                    let mut v = stages;
                    v["extract_seconds"] = request.extract_seconds.into();
                    v["synthesize_seconds"] = synthesize_s.into();
                    v["pcm_samples"] = pcm.len().into();
                    v["audio_seconds"] = audio.into();
                    v["rtf"] = (synthesize_s / audio).into();
                    v["pcm_valid"] = (!pcm.is_empty() && pcm.iter().all(|x| x.is_finite())).into();
                    Ok(v)
                })()
                .unwrap_or_else(|e| json!({ "error": e.to_string() }));
                let mut round = round;
                round["round"] = i.into();
                rounds.push(round);
                mark(&format!("ROUND_END {i} {}", unix_ms()));
            }
            report["warm_rounds"] = rounds.into();
        }
        "stream" => {
            let sampling = SamplingConfig {
                seed: Some(args.seed),
                ..SamplingConfig::default()
            };
            let (samples, sr) = prompt_wav
                .as_ref()
                .map(|w| read_wav_mono(w))
                .transpose()?
                .unwrap_or((Vec::new(), 0));
            let wav_args = if args.reextract && !samples.is_empty() {
                Some((samples.as_slice(), sr))
            } else {
                None
            };

            // Warmup session (untimed).
            {
                let warm = round_prompt(&mut frontend, wav_args, &prompt_text, &prompt)?;
                let mut session = cv.start_stream(&text, &warm.features, &sampling)?;
                while session.next_pcm()?.is_some() {}
            }

            let mut rounds = Vec::new();
            for i in 0..args.rounds {
                mark(&format!("ROUND_START {i} {}", unix_ms()));
                let outcome = (|| -> Result<serde_json::Value> {
                    let request_start = Instant::now();
                    let request = round_prompt(&mut frontend, wav_args, &prompt_text, &prompt)?;
                    let session_timer = Instant::now();
                    let mut session = cv.start_stream(&text, &request.features, &sampling)?;
                    let session_start = session_timer.elapsed().as_secs_f64();

                    let mut chunks = Vec::new();
                    let mut all = Vec::new();
                    let mut hit_limit = false;
                    let mut got_final = false;
                    loop {
                        let call = Instant::now();
                        let chunk = session.next_pcm()?;
                        let gen_s = call.elapsed().as_secs_f64();
                        let Some(chunk) = chunk else { break };
                        chunks.push(json!({
                            "index": chunks.len(),
                            "samples": chunk.samples.len(),
                            "audio_seconds": chunk.samples.len() as f64 / 24000.0,
                            "gen_seconds": gen_s,
                            "arrival_since_request": request_start.elapsed().as_secs_f64(),
                            "arrival_since_session": session_timer.elapsed().as_secs_f64(),
                            "final": chunk.final_chunk,
                        }));
                        hit_limit |= chunk.hit_token_limit;
                        got_final |= chunk.final_chunk;
                        all.extend_from_slice(&chunk.samples);
                    }
                    let total = request_start.elapsed().as_secs_f64();
                    let first = chunks.first().context("stream produced no chunks")?;
                    if let (Some(path), true) = (&args.wav_out, i == 0) {
                        write_wav(path, &all)?;
                    }
                    Ok(json!({
                        "extract_seconds": request.extract_seconds,
                        "session_start_seconds": session_start,
                        "first_pcm_since_request": first["arrival_since_request"],
                        "first_pcm_since_session": first["arrival_since_session"],
                        "chunks": chunks,
                        "chunk_count": chunks.len(),
                        "total_seconds": total,
                        "pcm_samples": all.len(),
                        "audio_seconds": all.len() as f64 / 24000.0,
                        "rtf": total / (all.len() as f64 / 24000.0),
                        "completed": got_final,
                        "hit_token_limit": hit_limit,
                        "pcm_valid": !all.is_empty() && all.iter().all(|v| v.is_finite()),
                    }))
                })();
                let mut round = match outcome {
                    Ok(v) => v,
                    Err(e) => json!({ "error": e.to_string() }),
                };
                round["round"] = i.into();
                rounds.push(round);
                mark(&format!("ROUND_END {i} {}", unix_ms()));
            }
            report["stream_rounds"] = rounds.into();
        }
        "components" => {
            components(&args, &dev, &mut report)?;
        }
        other => bail!("unknown mode {other}"),
    }

    report["wall_since_process_start"] = process_start.elapsed().as_secs_f64().into();

    let text_json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.out {
        std::fs::write(path, &text_json)?;
    }
    println!("{text_json}");
    let mut stderr = std::io::stderr();
    writeln!(
        stderr,
        "bench done: {} {} {}",
        args.label, args.device, args.mode
    )?;
    Ok(())
}
