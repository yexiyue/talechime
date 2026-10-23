//! Offline real-weight CPU MLP merge screen; no model downloads.
#[path = "cpu_mlp_probe/silu_mul.rs"]
mod silu_mul;

use std::{hint::black_box, time::Instant};

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

fn compare(reference: &Tensor, candidate: &Tensor) -> Result<(usize, f32)> {
    let a = reference.flatten_all()?.to_vec1::<f32>()?;
    let b = candidate.flatten_all()?.to_vec1::<f32>()?;
    let differing = a
        .iter()
        .zip(&b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max_abs = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    Ok((differing, max_abs))
}

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        candle_core::bail!("usage: cpu_mlp_probe MODEL_SAFETENSORS [SEQ LAYER]");
    };
    let seq = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(157usize);
    let layer = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0usize);
    if seq == 0 || layer >= 28 {
        candle_core::bail!("require SEQ > 0 and LAYER < 28");
    }
    // This explicit local model must remain immutable for the lifetime of the mapping.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu)? }
        .pp(format!("llm.layers.{layer}.mlp"));
    let gate = Linear::new(vb.get((3072, 1024), "gate_proj.weight")?, None);
    let up = Linear::new(vb.get((3072, 1024), "up_proj.weight")?, None);
    let down = Linear::new(vb.get((1024, 3072), "down_proj.weight")?, None);
    let start = Instant::now();
    let merged = Tensor::cat(&[gate.weight(), up.weight()], 0)?;
    let merge_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let packed = merged.t()?.contiguous()?;
    println!(
        "{}",
        serde_json::json!({"kind":"prepare","seq":seq,"layer":layer,
        "merge_seconds":merge_seconds,"pack_seconds":start.elapsed().as_secs_f64(),
        "bytes_per_extra_layout":6144*1024*4})
    );
    let merged = Linear::new(merged, None);
    let mut state = 42u64;
    let xs = Tensor::from_vec(
        (0..2 * seq * 1024)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
            })
            .collect::<Vec<_>>(),
        (2, seq, 1024),
        &Device::Cpu,
    )?;
    let project = |id| -> Result<(Tensor, Tensor)> {
        if id == 0 || id == 4 {
            return Ok((gate.forward(&xs)?, up.forward(&xs)?));
        }
        let all = if id == 3 {
            xs.reshape((2 * seq, 1024))?
                .matmul(&packed)?
                .reshape((2, seq, 6144))?
        } else {
            merged.forward(&xs)?
        };
        let g = all.narrow(D::Minus1, 0, 3072)?;
        let u = all.narrow(D::Minus1, 3072, 3072)?;
        if id == 1 {
            Ok((g, u))
        } else {
            Ok((g.contiguous()?, u.contiguous()?))
        }
    };
    let run = |id| -> Result<Tensor> {
        let hidden = if id == 0 {
            // Match production order: gate projection, SiLU, up projection, multiply.
            let lhs = gate.forward(&xs)?.silu()?;
            (lhs * up.forward(&xs)?)?
        } else {
            let (gate, up) = project(id)?;
            if id == 4 {
                silu_mul::forward(&gate, &up)?
            } else {
                (gate.silu()? * up)?
            }
        };
        // Include the down projection and a residual consumer at the real shape.
        down.forward(&hidden)? + &xs
    };
    let reference = run(0)?;
    let (g_ref, u_ref) = project(0)?;
    for id in 1..5 {
        let (g, u) = project(id)?;
        let (differing, max_abs) = compare(&reference, &run(id)?)?;
        println!(
            "{}",
            serde_json::json!({"kind":"parity","seq":seq,"layer":layer,
            "candidate":id,"gate_differing":compare(&g_ref,&g)?.0,
            "up_differing":compare(&u_ref,&u)?.0,"mlp_differing":differing,"max_abs":max_abs})
        );
    }
    for id in [0, 1, 2, 3, 4, 4, 3, 2, 1, 0] {
        for _ in 0..3 {
            black_box(run(id)?);
        }
        let start = Instant::now();
        for _ in 0..20 {
            black_box(run(id)?);
        }
        println!(
            "{}",
            serde_json::json!({"kind":"timing","seq":seq,"layer":layer,
            "candidate":id,"microseconds":start.elapsed().as_secs_f64()*1e6/20.})
        );
    }
    Ok(())
}
