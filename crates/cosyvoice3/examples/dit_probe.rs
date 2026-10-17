//! Layer-level DiT parity probe against cosyvoice3_dit_probe.py exports.
//!
//! Usage: `dit_probe <model-dir> <fixture-dir> <probe-dir> [metal] [f64]`
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use cosyvoice3::{CosyVoice3Config, DiT};
use std::path::Path;

fn load_weights(path: &Path, dtype: DType, dev: &Device) -> Result<VarBuilder<'static>> {
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, dev)? })
}

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
        se += (av[i] - bv[i]) as f64 * (av[i] - bv[i]) as f64;
    }
    Ok((max, (se / n as f64).sqrt() as f32))
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let model = Path::new(
        args.get(1)
            .context("usage: dit_probe <model> <fixture> <probe> [metal] [f64|detail <block>]")?,
    );
    let dir = Path::new(args.get(2).context("missing fixture")?);
    let probe = Path::new(args.get(3).context("missing probe dir")?);
    let dev = match args.get(4).map(String::as_str) {
        Some("metal") => Device::new_metal(0)?,
        _ => Device::Cpu,
    };
    let cfg = CosyVoice3Config::from_model_dir(model)?;
    let f: std::collections::HashMap<String, Tensor> =
        candle_core::safetensors::load(dir.join("flow.safetensors"), &Device::Cpu)?
            .into_iter()
            .map(|(k, t)| Ok((k, t.to_device(&dev)?)))
            .collect::<Result<_>>()?;

    if args.get(5).map(String::as_str) == Some("detail") {
        let idx: usize = args.get(6).context("missing block index")?.parse()?;
        let d: std::collections::HashMap<String, Tensor> =
            candle_core::safetensors::load(probe.join("ditdetail.safetensors"), &Device::Cpu)?
                .into_iter()
                .map(|(k, t)| Ok((k, t.to_device(&dev)?)))
                .collect::<Result<_>>()?;
        let b: std::collections::HashMap<String, Tensor> =
            candle_core::safetensors::load(probe.join("ditblocks.safetensors"), &Device::Cpu)?
                .into_iter()
                .map(|(k, t)| Ok((k, t.to_device(&dev)?)))
                .collect::<Result<_>>()?;
        let dit = DiT::new(
            cfg.flow.dit.clone(),
            load_weights(&model.join("flow.safetensors"), DType::F32, &dev)?.pp("dit"),
        )?;
        // Official input to block `idx`: output of block idx-1 (or the
        // input embedding for idx == 0), so this block runs in isolation.
        let x_in = if idx == 0 {
            f.get("input_embed")
                .context("missing official input_embed fixture")?
                .clone()
        } else {
            b[&format!("block_{:02}", idx - 1)].clone()
        };
        let seq = x_in.dim(1)?;
        let det = dit.block_detail(idx, &x_in, &f["time_embed"], &f["mask"], seq)?;
        let pairs = [
            ("norm", &det.norm),
            ("gate_msa", &det.gate_msa),
            ("shift_mlp", &det.shift_mlp),
            ("scale_mlp", &det.scale_mlp),
            ("gate_mlp", &det.gate_mlp),
            ("attn", &det.attn),
            ("after_attn", &det.after_attn),
            ("ff", &det.ff),
            ("out", &det.out),
        ];
        for (name, t) in pairs {
            let key = format!("b{idx:02}_{name}");
            let expected = d.get(&key).with_context(|| {
                format!("missing {key}; regenerate cosyvoice3_dit_probe.py exports")
            })?;
            let (max, rmse) = stats(t, expected)?;
            println!("b{idx:02}_{name}: max_abs={max:.6e} rmse={rmse:.3e}");
        }
        return Ok(());
    }

    let dtype = if args.get(5).map(String::as_str) == Some("f64") {
        DType::F64
    } else {
        DType::F32
    };
    let f = f
        .into_iter()
        .map(|(k, t)| Ok((k, t.to_dtype(dtype)?)))
        .collect::<Result<std::collections::HashMap<_, _>>>()?;
    let p: std::collections::HashMap<String, Tensor> =
        candle_core::safetensors::load(probe.join("ditblocks.safetensors"), &Device::Cpu)?
            .into_iter()
            .map(|(k, t)| Ok((k, t.to_device(&dev)?.to_dtype(dtype)?)))
            .collect::<Result<_>>()?;

    let dit = DiT::new(
        cfg.flow.dit.clone(),
        load_weights(&model.join("flow.safetensors"), dtype, &dev)?.pp("dit"),
    )?;
    let (out, trace) = dit.forward_traced(
        &f["x"], &f["mask"], &f["mu"], &f["t"], &f["spks"], &f["cond"], false,
    )?;
    for (i, t) in trace.iter().enumerate() {
        let name = if i + 1 == trace.len() {
            "norm_out".to_string()
        } else {
            format!("block_{i:02}")
        };
        let (max, rmse) = stats(t, &p[&name])?;
        let mag = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let tmax = mag.iter().fold(0f32, |m, v| m.max(v.abs()));
        println!("{name}: max_abs={max:.6e} rmse={rmse:.3e} act_max={tmax:.2}");
    }
    let (max, rmse) = stats(&out, &p["final"])?;
    println!("final: max_abs={max:.6e} rmse={rmse:.3e}");
    if dtype == DType::F64
        && let Ok(anchor) =
            candle_core::safetensors::load(probe.join("estimator_f64.safetensors"), &Device::Cpu)
    {
        let a = anchor["out"].to_device(&dev)?.to_dtype(DType::F64)?;
        let (max, rmse) = stats(&out, &a)?;
        println!("final_vs_official_f64: max_abs={max:.6e} rmse={rmse:.3e}");
    }
    Ok(())
}
