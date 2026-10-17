//! F0 predictor parity + sensitivity probe.
//!
//! Usage: f0_probe <model-dir> <fixture-dir-with-official_stages>
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use cosyvoice3::CosyVoice3Config;
use cosyvoice3::vocoder::f0_predictor::CausalConvRnnF0Predictor;
use std::path::Path;

fn stats(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    anyhow::ensure!(
        a.dims() == b.dims() && a.elem_count() > 0,
        "shape mismatch or empty comparison"
    );
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max = 0f32;
    let mut se = 0f64;
    for (a, b) in a.iter().zip(&b) {
        anyhow::ensure!(
            a.is_finite() && b.is_finite(),
            "non-finite comparison input"
        );
        let error = f64::from(*a) - f64::from(*b);
        max = max.max(error.abs() as f32);
        se += error * error;
    }
    Ok((max, (se / a.len() as f64).sqrt() as f32))
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let model = Path::new(args.get(1).context("usage: f0_probe <model> <fixture>")?);
    let dir = Path::new(args.get(2).context("missing fixture")?);

    let cfg = CosyVoice3Config::from_model_dir(model)?;
    let in_ch = cfg.hift.in_channels;
    let hf: std::collections::HashMap<String, Tensor> =
        candle_core::safetensors::load(dir.join("hift.safetensors"), &Device::Cpu)?
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();
    let st: std::collections::HashMap<String, Tensor> =
        candle_core::safetensors::load(dir.join("official_stages.safetensors"), &Device::Cpu)?
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>();

    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[model.join("hift.safetensors")],
            DType::F64,
            &Device::Cpu,
        )?
    };
    let pred = CausalConvRnnF0Predictor::new(in_ch, 512, vb.pp("f0_predictor"))?;
    let f0 = pred.forward(&hf["mel"], true)?;

    let (max, rmse) = stats(&f0, &st["f0"])?;
    println!("f0_vs_official: max_abs={max:.6e} rmse={rmse:.3e}");

    let mel_p = hf["mel"].affine(1.0 + 1e-7, 0.0)?;
    let f0_p = pred.forward(&mel_p, true)?;
    let (pmax, prmse) = stats(&f0, &f0_p)?;
    println!("f0_sensitivity_to_1e-7_mel: max_abs={pmax:.6e} rmse={prmse:.3e}");

    let f0_2 = pred.forward(&hf["mel"], true)?;
    let (dmax, _) = stats(&f0, &f0_2)?;
    println!("f0_rerun_self_diff: max_abs={dmax:.6e}");

    // conv0 f64 cross-check against torch
    if let Ok(c) = candle_core::safetensors::load(dir.join("conv0_f64.safetensors"), &Device::Cpu) {
        use candle_core::Module;
        use cosyvoice3::vocoder::causal_conv::{CausalConv1d, CausalType};
        let conv = CausalConv1d::new(
            in_ch,
            512,
            4,
            1,
            CausalType::Right,
            vb.pp("f0_predictor.condnet.0"),
        )?;
        let y = conv.forward(&hf["mel"].to_dtype(DType::F64)?)?;
        let (cmax, crmse) = stats(&y, &c["conv0_out"])?;
        println!("conv0_f64_vs_torch: max_abs={cmax:.6e} rmse={crmse:.3e}");
    }
    Ok(())
}
