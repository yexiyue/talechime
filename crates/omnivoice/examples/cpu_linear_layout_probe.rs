//! CPU weight-layout screen using local layer-0 weights and synthetic inputs.
use std::{hint::black_box, time::Instant};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        candle_core::bail!("usage: cpu_linear_layout_probe MODEL_SAFETENSORS [ROWS]");
    };
    let rows = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(314);
    if rows == 0 {
        candle_core::bail!("ROWS must be positive");
    }
    // The local immutable model file stays mapped for the lifetime of vb.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu)? };
    for (name, input, output) in [
        ("self_attn.q_proj", 1024, 2048),
        ("self_attn.k_proj", 1024, 1024),
        ("self_attn.v_proj", 1024, 1024),
        ("self_attn.o_proj", 2048, 1024),
        ("mlp.gate_proj", 1024, 3072),
        ("mlp.up_proj", 1024, 3072),
        ("mlp.down_proj", 3072, 1024),
    ] {
        let weight = vb.get((output, input), &format!("llm.layers.0.{name}.weight"))?;
        let transposed = weight.t()?;
        let start = Instant::now();
        let packed = transposed.contiguous()?;
        let pack_seconds = start.elapsed().as_secs_f64();
        let mut state = 42u64;
        let x = Tensor::from_vec(
            (0..rows * input)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / (1u32 << 24) as f32 - 0.5
                })
                .collect::<Vec<_>>(),
            (rows, input),
            &Device::Cpu,
        )?;
        let reference = x.matmul(&transposed)?.flatten_all()?.to_vec1::<f32>()?;
        let candidate = x.matmul(&packed)?.flatten_all()?.to_vec1::<f32>()?;
        let differing = reference
            .iter()
            .zip(&candidate)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max_abs = reference
            .iter()
            .zip(&candidate)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "{}",
            serde_json::json!({"kind":"parity","name":name,"rows":rows,"differing":differing,
                "elements":reference.len(),"max_abs":max_abs,"pack_seconds":pack_seconds})
        );
        for id in [0, 1, 1, 0] {
            let w = if id == 0 { &transposed } else { &packed };
            for _ in 0..3 {
                black_box(x.matmul(w)?);
            }
            let start = Instant::now();
            for _ in 0..20 {
                black_box(x.matmul(w)?);
            }
            println!(
                "{}",
                serde_json::json!({"kind":"timing","name":name,"rows":rows,"candidate":id,
                    "microseconds":start.elapsed().as_secs_f64()*1e6/20.0})
            );
        }
    }
    Ok(())
}
