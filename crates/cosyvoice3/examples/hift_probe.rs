//! Stage-level HiFT parity probe against official_stages.safetensors.
//!
//! Usage: hift_probe <model-dir> <fixture-dir> [metal]
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cosyvoice3::{CausalHiFtGenerator, CosyVoice3Config};
use std::path::Path;

fn stats(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    anyhow::ensure!(
        a.dims() == b.dims() && a.elem_count() > 0,
        "shape mismatch or empty comparison"
    );
    let a = a.to_dtype(DType::F32)?.flatten_all()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?;
    let n = a.elem_count();
    let av = a.to_vec1::<f32>()?;
    let bv = b.to_vec1::<f32>()?;
    let mut max = 0f32;
    let mut se = 0f64;
    for i in 0..n {
        anyhow::ensure!(
            av[i].is_finite() && bv[i].is_finite(),
            "non-finite comparison input"
        );
        let d = (av[i] - bv[i]).abs();
        max = max.max(d);
        se += d as f64 * d as f64;
    }
    Ok((max, (se / n as f64).sqrt() as f32))
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let model = Path::new(
        args.get(1)
            .context("usage: hift_probe <model> <fixture> [metal]")?,
    );
    let dir = Path::new(args.get(2).context("missing fixture")?);
    let dev = match args.get(3).map(String::as_str) {
        Some("metal") => Device::new_metal(0)?,
        _ => Device::Cpu,
    };

    let cfg = CosyVoice3Config::from_model_dir(model)?;
    let load_dev = |p: std::path::PathBuf| -> Result<std::collections::HashMap<String, Tensor>> {
        candle_core::safetensors::load(p, &Device::Cpu)?
            .into_iter()
            .map(|(k, t)| Ok((k, t.to_device(&dev)?)))
            .collect()
    };
    let hf = load_dev(dir.join("hift.safetensors"))?;
    let st = load_dev(dir.join("official_stages.safetensors"))?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model.join("hift.safetensors")], DType::F32, &dev)?
    };
    let mut hift = CausalHiFtGenerator::new(cfg.hift, 0, vb)?;
    hift.set_source_noise(hf["rand_ini"].clone(), hf["source_noise"].clone())?;
    let (pcm, trace) = hift.inference_traced(&hf["mel"], true)?;

    // si_N in the trace is the resblock output; official dump captured the
    // same module. x_stageN has no official counterpart here beyond si/pcm.
    for (name, t) in &trace {
        if let Some(b) = st.get(name) {
            let (max, rmse) = stats(t, b)?;
            println!("{name}: max_abs={max:.6e} rmse={rmse:.3e}");
        } else {
            println!("{name}: (no official counterpart, shape {:?})", t.dims());
        }
    }
    let (max, rmse) = stats(&pcm, &st["pcm"])?;
    println!("pcm: max_abs={max:.6e} rmse={rmse:.3e}");
    Ok(())
}
