//! Offline CPU attention candidates at OmniVoice shapes; no model downloads.
use std::{hint::black_box, time::Instant};

use candle_core::{Device, Result, Tensor};
use candle_nn::attention::{AttnMask, flash_attn};

fn manual(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, copy: bool) -> Result<Tensor> {
    let k = candle_transformers::utils::repeat_kv(k.clone(), 2)?;
    let v = candle_transformers::utils::repeat_kv(v.clone(), 2)?;
    let kt = k.transpose(2, 3)?;
    let kt = if copy { kt.contiguous()? } else { kt };
    let scores = q.matmul(&kt)?.affine(1.0 / 128f64.sqrt(), 0.0)?;
    candle_nn::ops::softmax_last_dim(&scores.broadcast_add(mask)?)?.matmul(&v)
}

fn grouped(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = q.dims4()?;
    let kv = k.dim(1)?;
    let q = q.reshape((b, kv, (h / kv) * s, d))?;
    let scores = q
        .matmul(&k.transpose(2, 3)?)?
        .affine(1.0 / (d as f64).sqrt(), 0.0)?
        .reshape((b, h, s, s))?
        .broadcast_add(mask)?;
    candle_nn::ops::softmax_last_dim(&scores)?
        .reshape((b, kv, (h / kv) * s, s))?
        .matmul(&v.contiguous()?)?
        .reshape((b, h, s, d))
}

fn flash(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let mut rows = Vec::new();
    for i in 0..q.dim(0)? {
        rows.push(flash_attn::<f32>(
            &q.narrow(0, i, 1)?.transpose(1, 2)?,
            &k.narrow(0, i, 1)?.transpose(1, 2)?,
            &v.narrow(0, i, 1)?.transpose(1, 2)?,
            1.0 / 128f32.sqrt(),
            AttnMask::Mask(mask.narrow(0, i, 1)?),
            None,
            None,
        )?);
    }
    Tensor::cat(&rows, 0)
}

fn data(shape: (usize, usize, usize, usize), phase: f32) -> Result<Tensor> {
    let n = shape.0 * shape.1 * shape.2 * shape.3;
    let mut state = u64::from(phase.to_bits()) + 1;
    Tensor::from_vec(
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.6
            })
            .collect::<Vec<_>>(),
        shape,
        &Device::Cpu,
    )
}

fn main() -> Result<()> {
    let s = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(157);
    let q = data((2, 16, s, 128), 0.1)?;
    let k = data((2, 8, s, 128), 0.2)?;
    let v = data((2, s, 8, 128), 0.3)?.transpose(1, 2)?;
    let mask = Tensor::from_vec(
        (0..2 * s * s)
            .map(|i| {
                if i % s >= s - s / 8 {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect::<Vec<_>>(),
        (2, 1, s, s),
        &Device::Cpu,
    )?;
    let run = |id| match id {
        0 => manual(&q, &k, &v, &mask, true),
        1 => manual(&q, &k, &v, &mask, false),
        2 => grouped(&q, &k, &v, &mask),
        _ => flash(&q, &k, &v, &mask),
    };
    let reference = run(0)?.flatten_all()?.to_vec1::<f32>()?;
    for id in 0..4 {
        let output = run(id)?.flatten_all()?.to_vec1::<f32>()?;
        let differing = reference
            .iter()
            .zip(&output)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max_abs = reference
            .iter()
            .zip(&output)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "{}",
            serde_json::json!({"kind":"parity", "seq":s, "candidate":id,
                "elements":reference.len(),"differing":differing,"max_abs":max_abs})
        );
    }
    for id in [0, 1, 2, 3, 3, 2, 1, 0] {
        for _ in 0..3 {
            black_box(run(id)?);
        }
        let start = Instant::now();
        for _ in 0..20 {
            black_box(run(id)?);
        }
        println!(
            "{}",
            serde_json::json!({"kind":"timing", "seq":s,"candidate":id,
                "microseconds":start.elapsed().as_secs_f64()*1e6/20.0})
        );
    }
    Ok(())
}
