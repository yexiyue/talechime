//! Local real-weight K/V projection screen, including the production wrapper.
#[path = "../src/stage0_cpu_linear.rs"]
mod stage0_cpu_linear;

use std::{collections::HashMap, hint::black_box, time::Instant};

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};
use stage0_cpu_linear::CpuProjection;

fn bits(t: Tensor) -> Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        candle_core::bail!("usage: cpu_projection_probe MODEL_SAFETENSORS [SEQ]");
    };
    let seq = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(157);
    if seq == 0 {
        candle_core::bail!("SEQ must be positive");
    }
    let device = Device::Cpu;
    // The explicitly supplied local model remains immutable while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device)? };
    let kw = vb.get((1024, 1024), "llm.layers.0.self_attn.k_proj.weight")?;
    let vw = vb.get((1024, 1024), "llm.layers.0.self_attn.v_proj.weight")?;
    let kb = Linear::new(kw.clone(), None);
    let vb_ref = Linear::new(vw.clone(), None);
    let local = VarBuilder::from_tensors(
        HashMap::from([("k.weight".into(), kw), ("v.weight".into(), vw)]),
        DType::F32,
        &device,
    );
    let start = Instant::now();
    let kc = CpuProjection::load(1024, 1024, false, local.pp("k"))?;
    let vc = CpuProjection::load(1024, 1024, false, local.pp("v"))?;
    println!(
        "{}",
        serde_json::json!({"kind":"prepare","seq":seq,"seconds":start.elapsed().as_secs_f64(),"additional_weight_bytes":if candle_core::utils::has_accelerate() {2*1024*1024*4} else {0}})
    );
    let mut state = 42u64;
    let xs = Tensor::from_vec(
        (0..2 * seq * 1024)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect::<Vec<_>>(),
        (2, seq, 1024),
        &device,
    )?;
    let run = |candidate, consume| -> Result<(Tensor, Tensor)> {
        let k = if candidate {
            kc.forward(&xs)?
        } else {
            kb.forward(&xs)?
        };
        let v = if candidate {
            vc.forward(&xs)?
        } else {
            vb_ref.forward(&xs)?
        };
        if consume {
            // Head-major materialization is a simplified downstream consumer,
            // not a captured model trajectory or the complete attention layer.
            Ok((
                k.reshape((2, seq, 8, 128))?.transpose(1, 2)?.contiguous()?,
                v.reshape((2, seq, 8, 128))?.transpose(1, 2)?.contiguous()?,
            ))
        } else {
            Ok((k, v))
        }
    };
    for consume in [false, true] {
        let (a, b) = run(false, consume)?;
        let (c, d) = run(true, consume)?;
        assert_eq!(bits(a)?, bits(c)?);
        assert_eq!(bits(b)?, bits(d)?);
        println!(
            "{}",
            serde_json::json!({"kind":"parity","seq":seq,"consume":consume,"differing":0})
        );
        for candidate in [false, true, true, false] {
            for _ in 0..3 {
                black_box(run(candidate, consume)?);
            }
            let start = Instant::now();
            for _ in 0..30 {
                black_box(run(candidate, consume)?);
            }
            println!(
                "{}",
                serde_json::json!({"kind":"timing","seq":seq,"consume":consume,"candidate":candidate,"microseconds":start.elapsed().as_secs_f64()*1e6/30.0})
            );
        }
    }
    Ok(())
}
