//! Teacher-forced real-weight comparison against cosyvoice3_reference.py.
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cosyvoice3::{
    CausalHiFtGenerator, CausalMaskedDiffWithDiT, CosyVoice3Config, CosyVoice3Lm, DiT,
};
use std::{path::Path, time::Instant};
mod common;
use common::compare;

fn load_weights(path: &Path, dev: &Device) -> Result<VarBuilder<'static>> {
    // The converted files are immutable throughout this probe.
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, dev)? })
}
fn wav(path: &Path, pcm: &Tensor) -> Result<()> {
    let samples = pcm
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut w = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for s in samples {
        w.write_sample(s)?;
    }
    w.finalize()?;
    Ok(())
}
fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let model = Path::new(
        args.get(1)
            .context("usage: compare_model <model-dir> <fixture-dir> [metal]")?,
    );
    let dir = Path::new(args.get(2).context("missing fixture-dir")?);
    let dev = match args.get(3).map(String::as_str) {
        Some("metal") => Device::new_metal(0)?,
        _ => Device::Cpu,
    };
    let cfg = CosyVoice3Config::from_model_dir(model)?;
    let load = |name: &str| -> Result<std::collections::HashMap<String, Tensor>> {
        candle_core::safetensors::load(dir.join(format!("{name}.safetensors")), &Device::Cpu)?
            .into_iter()
            .map(|(k, t)| {
                let t = if matches!(t.dtype(), DType::I32 | DType::I64) {
                    t.to_dtype(DType::U32)?
                } else {
                    t
                };
                Ok((k, t.to_device(&dev)?))
            })
            .collect()
    };
    let mut failures = vec![];
    macro_rules! check {
        ($name:expr,$a:expr,$b:expr,$tol:expr) => {
            if let Err(e) = compare(
                $name,
                &$a?.to_device(&Device::Cpu)?,
                &$b.to_device(&Device::Cpu)?,
                $tol,
            ) {
                failures.push(e.to_string());
            }
        };
    }
    let f = load("llm")?;
    let t = Instant::now();
    let mut llm = CosyVoice3Lm::new(
        &cfg.llm,
        load_weights(&model.join("llm.safetensors"), &dev)?,
    )?;
    let front = load("frontend")?;
    let logits = llm.prefill_logits(
        &f["text_token"].to_dtype(DType::U32)?,
        &f["prompt_text_token"].to_dtype(DType::U32)?,
        &front["prompt_speech_token"].to_dtype(DType::U32)?,
    )?;
    check!(
        "real_first_logits",
        logits.unsqueeze(0),
        f["first_logits"],
        3e-3
    );
    println!("LLM load+prefill_seconds={:.3}", t.elapsed().as_secs_f64());
    drop(llm);
    let f = load("flow")?;
    let t = Instant::now();
    let vb = load_weights(&model.join("flow.safetensors"), &dev)?;
    if let Some(expected) = f.get("time_embed") {
        use candle_core::Module;
        let time = cosyvoice3::flow::embeddings::TimestepEmbedding::new(
            cfg.flow.dit.dim,
            vb.pp("dit.time_embed"),
        )?
        .forward(&f["t"])?;
        check!(
            "real_time_embed",
            Ok::<_, candle_core::Error>(time),
            expected,
            1e-4
        );
        let embed = cosyvoice3::flow::embeddings::InputEmbedding::new(
            80,
            80,
            cfg.flow.dit.dim,
            vb.pp("dit.input_embed"),
        )?;
        check!(
            "real_input_embed",
            embed.forward(
                &f["x"].transpose(1, 2)?,
                &f["cond"].transpose(1, 2)?,
                &f["mu"].transpose(1, 2)?,
                &f["spks"]
            ),
            f["input_embed"],
            1e-4
        );
    }
    let dit = DiT::new(cfg.flow.dit.clone(), vb.pp("dit"))?;
    check!(
        "real_estimator",
        dit.forward(
            &f["x"], &f["mask"], &f["mu"], &f["t"], &f["spks"], &f["cond"], false
        ),
        f["out"],
        3e-3
    );
    let mut flow = CausalMaskedDiffWithDiT::new(&cfg.flow, 192, dit, vb)?;
    if let Some(noise) = f.get("noise") {
        flow.cfm_mut().set_noise(noise.clone());
    }
    let mel = flow.inference(
        &f["token"].to_dtype(DType::U32)?,
        &f["prompt_token"].to_dtype(DType::U32)?,
        &f["prompt_feat"],
        &f["embedding"],
        cfg.flow.cfm.n_timesteps,
        false,
        true,
    )?;
    check!(
        "real_flow_mel",
        Ok::<_, candle_core::Error>(mel.clone()),
        f["mel"],
        1e-2
    );
    // Shared-token mel for the Python-side HiFT amplification probe.
    candle_core::safetensors::save(
        &std::iter::once(("mel".to_string(), mel.to_device(&Device::Cpu)?)).collect(),
        dir.join("rust_mel.safetensors"),
    )?;
    println!("flow load+solve_seconds={:.3}", t.elapsed().as_secs_f64());
    drop(flow);
    let f = load("hift")?;
    let t = Instant::now();
    let mut hift = CausalHiFtGenerator::new(
        cfg.hift,
        0,
        load_weights(&model.join("hift.safetensors"), &dev)?,
    )?;
    hift.set_source_noise(f["rand_ini"].clone(), f["source_noise"].clone())?;
    let pcm = hift.inference(&f["mel"], true)?;
    check!(
        "real_hift_pcm",
        Ok::<_, candle_core::Error>(pcm.clone()),
        f["pcm"],
        3e-3
    );
    wav(&dir.join("rust-shared-mel.wav"), &pcm)?;
    let pcm = hift.inference(&mel, true)?;
    check!(
        "real_tokens_to_pcm",
        Ok::<_, candle_core::Error>(pcm.clone()),
        f["pcm"],
        1e-2
    );
    wav(&dir.join("rust-shared-tokens.wav"), &pcm)?;
    println!(
        "HiFT load+two decodes_seconds={:.3}",
        t.elapsed().as_secs_f64()
    );
    anyhow::ensure!(
        failures.is_empty(),
        "{} comparison(s) failed: {failures:?}",
        failures.len()
    );
    Ok(())
}
