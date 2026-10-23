//! Real-weight CPU output-head slicing screen; reads only an explicit local model.
use std::{hint::black_box, time::Instant};

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

fn main() -> Result<()> {
    let Some(path) = std::env::args().nth(1) else {
        candle_core::bail!("usage: cpu_head_slice_probe MODEL_SAFETENSORS [SEQ TARGET]");
    };
    let seq = std::env::args()
        .nth(2)
        .and_then(|x| x.parse().ok())
        .unwrap_or(157usize);
    let target = std::env::args()
        .nth(3)
        .and_then(|x| x.parse().ok())
        .unwrap_or(130usize);
    if target == 0 || target > seq {
        candle_core::bail!("require 0 < TARGET <= SEQ");
    }
    // The explicitly supplied model remains immutable while mapped.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu)? };
    let head = Linear::new(vb.get((8200, 1024), "audio_heads.weight")?, None);
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
        &Device::Cpu,
    )?;
    let run = |candidate| -> Result<(Tensor, Tensor)> {
        if candidate {
            let selected = Tensor::cat(
                &[
                    xs.i((0..1, seq - target..seq, ..))?,
                    xs.i((1..2, 0..target, ..))?,
                ],
                0,
            )?;
            let logits = head
                .forward(&selected)?
                .reshape((2, target, 8, 1025))?
                .permute((0, 2, 1, 3))?;
            Ok((logits.narrow(0, 0, 1)?, logits.narrow(0, 1, 1)?))
        } else {
            let full = head
                .forward(&xs)?
                .reshape((2, seq, 8, 1025))?
                .permute((0, 2, 1, 3))?;
            Ok((
                full.i((0..1, .., seq - target..seq, ..))?,
                full.i((1..2, .., 0..target, ..))?,
            ))
        }
    };
    let flatten = |(c, u): (Tensor, Tensor)| -> Result<Vec<f32>> {
        let mut values = c.flatten_all()?.to_vec1::<f32>()?;
        values.extend(u.flatten_all()?.to_vec1::<f32>()?);
        Ok(values)
    };
    let reference = flatten(run(false)?)?;
    let candidate = flatten(run(true)?)?;
    let differing = reference
        .iter()
        .zip(&candidate)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "{}",
        serde_json::json!({"kind":"parity","seq":seq,"target":target,"elements":reference.len(),"differing":differing})
    );
    for candidate in [false, true, true, false] {
        for _ in 0..3 {
            black_box(run(candidate)?);
        }
        let start = Instant::now();
        for _ in 0..20 {
            black_box(run(candidate)?);
        }
        println!(
            "{}",
            serde_json::json!({"kind":"timing","seq":seq,"target":target,"candidate":candidate,"microseconds":start.elapsed().as_secs_f64()*1e6/20.})
        );
    }
    Ok(())
}
