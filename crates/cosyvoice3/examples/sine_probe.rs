//! Compare the actual SineGen2 implementation with official source fixtures.
//! Usage: `sine_probe <model-dir> <fixture-dir>`
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use std::path::Path;
use talechime_cosyvoice3::vocoder::source_module::SineGen2;
mod common;

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let model = Path::new(args.get(1).context("usage: sine_probe <model> <fixture>")?);
    let dir = Path::new(args.get(2).context("missing fixture")?);
    let dev = Device::Cpu;
    let hf = candle_core::safetensors::load(dir.join("hift.safetensors"), &dev)?;
    let stages = candle_core::safetensors::load(dir.join("official_stages.safetensors"), &dev)?;
    let reference = candle_core::safetensors::load(dir.join("sinegen_stages.safetensors"), &dev)?;
    let mut sine = SineGen2::new(24000, 480, 8, 0.1, 0.003, 10., 0, &dev)?;
    sine.set_noise(hf["rand_ini"].clone(), hf["source_noise"].clone())?;
    let f0 = stages["f0"].to_dtype(DType::F32)?;
    let t = f0.dim(1)?;
    let indices = Tensor::from_vec(
        (0..t * 480).map(|i| (i / 480) as u32).collect(),
        t * 480,
        &dev,
    )?;
    let f0_up = f0.unsqueeze(2)?.index_select(&indices, 1)?;
    common::compare("f0_up", &f0_up, &reference["f0_up"], 0.)?;
    // Exercise the production phase/cumsum path, rather than duplicating
    // its former algorithm in a probe that never called SineGen2.
    let (waves, _) = sine.forward(&f0_up)?;
    common::compare("sine_waves", &waves, &reference["sine_waves"], 1e-5)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model.join("hift.safetensors")], DType::F32, &dev)?
    };
    let merged = candle_nn::linear(9, 1, vb.pp("m_source.l_linear"))?
        .forward(&waves)?
        .tanh()?;
    common::compare("merged", &merged, &reference["merged"], 1e-5)?;
    Ok(())
}
