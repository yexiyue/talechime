//! Compare independent official fixtures; no implicit model downloads.
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use cosyvoice3::frontend::audio::{KaldiFbank, MatchaMel, WhisperMel, resample};
use cosyvoice3::{CosyVoice3Lm, DiT, DiTConfig, LlmConfig};
use std::path::Path;

mod common;
use candle_nn::VarBuilder;
use common::compare;
pub fn weights(path: &Path, dtype: DType) -> Result<VarBuilder<'static>> {
    Ok(VarBuilder::from_tensors(
        candle_core::safetensors::load(path, &Device::Cpu)?,
        dtype,
        &Device::Cpu,
    ))
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .context("usage: compare <component-fixture-dir>")?;
    let dir = Path::new(&dir);
    let f = candle_core::safetensors::load(dir.join("components.safetensors"), &Device::Cpu)?;
    let mut failures = Vec::new();
    macro_rules! check {
        ($name:expr, $actual:expr, $expected:expr, $tol:expr) => {
            match $actual {
                Ok(a) => {
                    if let Err(e) = compare($name, &a, $expected, $tol) {
                        failures.push(e.to_string());
                    }
                }
                Err(e) => {
                    println!("{}: ERROR {e}", $name);
                    failures.push(format!("{}: {e}", $name));
                }
            }
        };
    }
    check!(
        "whisper",
        WhisperMel::new().forward(&f["audio"]),
        &f["whisper"],
        2e-4
    );
    check!(
        "fbank",
        KaldiFbank::new().forward(&f["audio"]),
        &f["fbank"],
        2e-3
    );
    check!(
        "resample",
        resample(&f["audio"], 16000, 24000),
        &f["resampled"],
        2e-6
    );
    check!(
        "matcha",
        MatchaMel::new().forward(&f["resampled"]),
        &f["matcha"],
        2e-4
    );
    for n in [1, 7, 15, 16, 17, 31, 32, 33, 1200000] {
        let name = format!("randn_{n}");
        check!(
            &name,
            Tensor::from_vec(
                cosyvoice3::rng::TorchCpuRng::new(0).randn_vec(n),
                n,
                &Device::Cpu
            ),
            &f[&name],
            2e-6
        );
    }
    let mut cfg = LlmConfig {
        llm_input_size: 56,
        llm_output_size: 56,
        ..LlmConfig::default()
    };
    cfg.qwen2.hidden_size = 56;
    cfg.qwen2.num_hidden_layers = 2;
    cfg.qwen2.intermediate_size = 112;
    cfg.qwen2.vocab_size = 64;
    cfg.qwen2.max_position_embeddings = 128;
    let mut llm = CosyVoice3Lm::new(
        &cfg,
        weights(&dir.join("small-llm.safetensors"), DType::F32)?,
    )?;
    let u32_token = |name: &str| -> candle_core::Result<Tensor> { f[name].to_dtype(DType::U32) };
    check!(
        "first_logits",
        llm.prefill_logits(
            &u32_token("text_token")?,
            &u32_token("prompt_text_token")?,
            &u32_token("prompt_speech_token")?
        )
        .and_then(|x| x.unsqueeze(0)),
        &f["first_logits"],
        2e-5
    );
    check!(
        "second_logits",
        llm.decode_logits(11, 10).and_then(|x| x.unsqueeze(0)),
        &f["second_logits"],
        2e-5
    );
    let cfg = DiTConfig {
        dim: 128,
        depth: 2,
        heads: 2,
        ff_mult: 2,
        ..DiTConfig::default()
    };
    let dit = DiT::new(
        cfg,
        weights(&dir.join("small-dit.safetensors"), DType::F32)?,
    )?;
    for (name, streaming) in [("dit", false), ("dit_stream", true)] {
        check!(
            name,
            dit.forward(
                &f["x"], &f["mask"], &f["mu"], &f["t"], &f["spks"], &f["cond"], streaming
            ),
            &f[name],
            2e-4
        );
    }
    let f0 = cosyvoice3::vocoder::f0_predictor::CausalConvRnnF0Predictor::new(
        80,
        16,
        weights(&dir.join("small-f0.safetensors"), DType::F64)?,
    )?;
    check!("f0", f0.forward(&f["f0_mel"], true), &f["f0"], 2e-6);
    anyhow::ensure!(
        failures.is_empty(),
        "{} comparison(s) failed: {failures:?}",
        failures.len()
    );
    Ok(())
}
