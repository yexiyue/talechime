//! Chain-B streaming demo: real LLM streaming with timing, sequential
//! request isolation, and cancellation/lifecycle behavior on a Tokio
//! LocalSet with a bounded channel.
//!
//! Usage: `stream_demo <model-dir> <fixture-dir> [metal] [scenario]`
//!   scenario: full (default) | lifecycle
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cosyvoice3::{
    CausalHiFtGenerator, CausalMaskedDiffWithDiT, CosyVoice3Config, CosyVoice3Lm, DiT,
    SamplingConfig, StreamingTts,
};
use std::path::Path;
use std::time::{Duration, Instant};

fn load_weights(path: &Path, dev: &Device) -> Result<VarBuilder<'static>> {
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, dev)? })
}

#[derive(Clone)]
struct Parts {
    cfg: CosyVoice3Config,
    text_token: Tensor,
    prompt_text_token: Tensor,
    llm_prompt: Tensor,
    flow_prompt: Tensor,
    prompt_feat: Tensor,
    embedding: Tensor,
}

fn load_parts(model: &Path, dir: &Path, dev: &Device) -> Result<Parts> {
    let cfg = CosyVoice3Config::from_model_dir(model)?;
    let load = |p: std::path::PathBuf| -> Result<std::collections::HashMap<String, Tensor>> {
        let mut out = std::collections::HashMap::new();
        for (k, t) in candle_core::safetensors::load(p, &Device::Cpu)? {
            // Metal has no integer dtype casts: convert while still on CPU.
            let t = if matches!(t.dtype(), DType::I32 | DType::I64) {
                t.to_dtype(DType::U32)?
            } else {
                t
            };
            out.insert(k, t.to_device(dev)?);
        }
        Ok(out)
    };
    let llm_fx = load(dir.join("llm.safetensors"))?;
    let front = load(dir.join("frontend.safetensors"))?;
    Ok(Parts {
        cfg,
        text_token: llm_fx["text_token"].to_dtype(DType::U32)?,
        prompt_text_token: llm_fx["prompt_text_token"].to_dtype(DType::U32)?,
        llm_prompt: front["prompt_speech_token"].to_dtype(DType::U32)?,
        flow_prompt: front["prompt_speech_token"].to_dtype(DType::U32)?,
        prompt_feat: front["prompt_speech_feat"].clone(),
        embedding: front["prompt_embedding"].clone(),
    })
}

fn build_session(
    model: &Path,
    p: &Parts,
    dev: &Device,
    seed: u64,
) -> Result<StreamingTts<'static>> {
    let llm = CosyVoice3Lm::new(
        &p.cfg.llm,
        load_weights(&model.join("llm.safetensors"), dev)?,
    )?;
    let vb = load_weights(&model.join("flow.safetensors"), dev)?;
    let dit = DiT::new(p.cfg.flow.dit.clone(), vb.pp("dit"))?;
    let flow = CausalMaskedDiffWithDiT::new(&p.cfg.flow, 192, dit, vb)?;
    let hift = CausalHiFtGenerator::new(
        p.cfg.hift.clone(),
        0,
        load_weights(&model.join("hift.safetensors"), dev)?,
    )?;
    let sampling = SamplingConfig {
        seed: Some(seed),
        ..SamplingConfig::default()
    };
    Ok(StreamingTts::new(
        llm,
        flow,
        hift,
        &p.text_token,
        &p.prompt_text_token,
        &p.llm_prompt,
        &p.flow_prompt,
        &p.prompt_feat,
        &p.embedding,
        &sampling,
    )?)
}

type ChunkTiming = (usize, f64, f64);

fn run_full(
    model: &Path,
    p: &Parts,
    dev: &Device,
    label: &str,
    out_wav: Option<&Path>,
) -> Result<Vec<f32>> {
    let mut session = build_session(model, p, dev, 42)?;
    let t0 = Instant::now();
    let mut all = Vec::new();
    let mut per_chunk: Vec<ChunkTiming> = Vec::new();
    let mut finished = false;
    while let Some(chunk) = session.next_pcm()? {
        anyhow::ensure!(
            !chunk.hit_token_limit,
            "{label}: LLM exhausted its token budget"
        );
        per_chunk.push((
            chunk.samples.len(),
            t0.elapsed().as_secs_f64(),
            all.len() as f64 / 24000.,
        ));
        all.extend_from_slice(&chunk.samples);
        finished = chunk.final_chunk;
    }
    anyhow::ensure!(
        finished && !all.is_empty(),
        "{label}: missing final chunk or audio"
    );
    if let Some(path) = out_wav {
        let mut writer = hound::WavWriter::create(
            path,
            hound::WavSpec {
                channels: 1,
                sample_rate: 24000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?;
        for sample in &all {
            writer.write_sample(*sample)?;
        }
        writer.finalize()?;
    }
    println!(
        "[{label}] chunks={} first_chunk_latency={:.3}s total={:.3}s audio={:.3}s hit_token_limit=false",
        per_chunk.len(),
        per_chunk.first().map(|c| c.1).unwrap_or(0.),
        t0.elapsed().as_secs_f64(),
        all.len() as f64 / 24000.
    );
    for (i, (n, at, _)) in per_chunk.iter().enumerate() {
        println!("  chunk {i}: {n} samples at t={at:.3}s");
    }
    Ok(all)
}

#[derive(Debug, PartialEq, Eq)]
enum ProducerExit {
    Completed(usize),
    ConsumerGone(usize),
}

/// Reserve queue capacity before any inference. Dropping a receiver before
/// the first poll avoids even model construction; errors never become EOF.
async fn produce_chunks(
    tx: tokio::sync::mpsc::Sender<cosyvoice3::StreamChunk>,
    mut next: impl FnMut() -> Result<Option<cosyvoice3::StreamChunk>>,
) -> Result<ProducerExit> {
    let mut sent = 0;
    loop {
        let permit = match tx.reserve().await {
            Ok(permit) => permit,
            Err(_) => return Ok(ProducerExit::ConsumerGone(sent)),
        };
        let chunk = next()?.context("stream ended without a final chunk")?;
        anyhow::ensure!(!chunk.hit_token_limit, "LLM exhausted its token budget");
        if tx.is_closed() {
            return Ok(ProducerExit::ConsumerGone(sent));
        }
        let finished = chunk.final_chunk;
        permit.send(chunk);
        sent += 1;
        if finished {
            return Ok(ProducerExit::Completed(sent));
        }
        // Synchronous inference cannot be preempted on this LocalSet.
        // Yield between blocks so the consumer can observe/cancel them.
        tokio::task::yield_now().await;
    }
}

fn spawn_producer(
    model: std::path::PathBuf,
    parts: std::rc::Rc<Parts>,
    dev: std::rc::Rc<Device>,
    tx: tokio::sync::mpsc::Sender<cosyvoice3::StreamChunk>,
    seed: u64,
) -> tokio::task::JoinHandle<Result<ProducerExit>> {
    tokio::task::spawn_local(async move {
        let mut session = None;
        produce_chunks(tx, || {
            if session.is_none() {
                session = Some(build_session(&model, &parts, &dev, seed)?);
            }
            session
                .as_mut()
                .context("missing initialized session")?
                .next_pcm()
                .map_err(Into::into)
        })
        .await
    })
}

async fn lifecycle(model: &Path, parts: Parts, dev: Device) -> Result<()> {
    let parts = std::rc::Rc::new(parts);
    let dev = std::rc::Rc::new(dev);
    let (tx, mut rx) = tokio::sync::mpsc::channel(2);
    let producer = spawn_producer(model.to_owned(), parts.clone(), dev.clone(), tx, 7);
    let first = rx.recv().await.context("no first chunk arrived")?;
    anyhow::ensure!(
        !first.final_chunk && !first.samples.is_empty(),
        "cancellation scenario needs a multi-block stream"
    );
    drop(rx);
    let exit = producer.await??;
    anyhow::ensure!(
        matches!(exit, ProducerExit::ConsumerGone(_)),
        "consumer cancellation was not observed: {exit:?}"
    );
    println!(
        "[cancel-after-first] received {} samples; {exit:?}",
        first.samples.len()
    );

    let (tx, rx) = tokio::sync::mpsc::channel(2);
    drop(rx);
    let producer = spawn_producer(model.to_owned(), parts.clone(), dev.clone(), tx, 8);
    anyhow::ensure!(
        producer.await?? == ProducerExit::ConsumerGone(0),
        "preclosed stream generated output"
    );
    println!("[cancel-before-first] no session construction or inference");

    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let producer = spawn_producer(model.to_owned(), parts, dev, tx, 9);
    let mut received = 0;
    let mut finished = false;
    while let Some(chunk) = rx.recv().await {
        anyhow::ensure!(!finished, "chunk received after completion");
        finished = chunk.final_chunk;
        received += 1;
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
    let exit = producer.await??;
    anyhow::ensure!(
        finished && received > 0 && exit == ProducerExit::Completed(received),
        "slow-consumer stream did not complete: {exit:?}"
    );
    println!("[slow-consumer] {received} chunks, capacity 1, completion verified");
    Ok(())
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let model = Path::new(
        args.get(1)
            .context("usage: stream_demo <model> <fixture> [metal] [full|lifecycle]")?,
    );
    let dir = Path::new(args.get(2).context("missing fixture")?);
    let scenario = args.get(4).map(String::as_str).unwrap_or("full");
    let dev = match args.get(3).map(String::as_str) {
        Some("metal") => Device::new_metal(0)?,
        _ => Device::Cpu,
    };
    let parts = load_parts(model, dir, &dev)?;
    if scenario == "lifecycle" {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?
            .block_on(tokio::task::LocalSet::new().run_until(lifecycle(model, parts, dev)))?;
    } else {
        anyhow::ensure!(scenario == "full", "unknown scenario: {scenario}");
        let first = run_full(
            model,
            &parts,
            &dev,
            "request-1",
            Some(&dir.join("rust-stream.wav")),
        )?;
        let second = run_full(model, &parts, &dev, "request-2", None)?;
        anyhow::ensure!(
            first == second,
            "same-seed requests differ: cache/state pollution"
        );
        println!("[isolation] same-seed sequential requests bit-identical");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cosyvoice3::StreamChunk;

    fn chunk(final_chunk: bool, hit_token_limit: bool) -> StreamChunk {
        StreamChunk {
            samples: vec![0.1],
            final_chunk,
            hit_token_limit,
            tokens_generated: 1,
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn closed_consumer_never_invokes_inference() -> Result<()> {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);
        let exit = produce_chunks(tx, || panic!("must not construct or advance a session")).await?;
        assert_eq!(exit, ProducerExit::ConsumerGone(0));
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn inference_errors_and_length_limits_are_not_successful_completion() {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        assert!(
            produce_chunks(tx, || anyhow::bail!("injected inference failure"))
                .await
                .is_err()
        );
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        assert!(
            produce_chunks(tx, || Ok(Some(chunk(true, true))))
                .await
                .is_err()
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn queue_capacity_blocks_the_next_inference_step() -> Result<()> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let calls = std::cell::Cell::new(0);
        let producer = produce_chunks(tx, || {
            calls.set(calls.get() + 1);
            Ok(Some(chunk(calls.get() == 2, false)))
        });
        tokio::pin!(producer);
        tokio::select! {
            biased;
            result = &mut producer => panic!("producer finished before consuming first chunk: {result:?}"),
            _ = tokio::task::yield_now() => {}
        }
        assert_eq!(calls.get(), 1);
        assert!(rx.recv().await.is_some());
        assert_eq!(producer.await?, ProducerExit::Completed(2));
        assert_eq!(calls.get(), 2);
        assert!(rx.recv().await.unwrap().final_chunk);
        Ok(())
    }
}
