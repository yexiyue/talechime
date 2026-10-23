//! Offline Metal probes for OmniVoice greedy scoring and logits conversion.
use std::{error::Error, time::Instant};

use candle_core::{D, DType, Device, IndexOp, Tensor};

fn select(log_probs: &Tensor, gather: bool) -> candle_core::Result<(Tensor, Tensor)> {
    let ids = log_probs.argmax(D::Minus1)?;
    let scores = if gather {
        log_probs
            .gather(&ids.unsqueeze(D::Minus1)?, D::Minus1)?
            .squeeze(D::Minus1)?
    } else {
        log_probs.max(D::Minus1)?
    };
    Ok((ids.to_dtype(DType::I64)?, scores))
}

fn score(c: &Tensor, u: &Tensor, gather: bool) -> candle_core::Result<(Tensor, Tensor)> {
    let c = candle_nn::ops::log_softmax(c, D::Minus1)?;
    let u = candle_nn::ops::log_softmax(u, D::Minus1)?;
    let guided = (&c + ((&c - &u)? * 2.0)?)?;
    let probs = candle_nn::ops::log_softmax(&guided, D::Minus1)?;
    let width = probs.dim(D::Minus1)?;
    let prefix = probs.narrow(D::Minus1, 0, width - 1)?;
    let mask = Tensor::full(f32::NEG_INFINITY, (1, 8, 130, 1), probs.device())?;
    select(&Tensor::cat(&[prefix, mask], D::Minus1)?, gather)
}

fn bits(t: &Tensor) -> candle_core::Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn check(a: &(Tensor, Tensor), b: &(Tensor, Tensor)) -> Result<(), Box<dyn Error>> {
    if a.0.flatten_all()?.to_vec1::<i64>()? != b.0.flatten_all()?.to_vec1::<i64>()?
        || bits(&a.1)? != bits(&b.1)?
    {
        return Err("selection differs".into());
    }
    Ok(())
}

fn bench(
    device: &Device,
    label: &str,
    mut run: impl FnMut(usize) -> candle_core::Result<(Tensor, Tensor)>,
    variants: &[usize],
) -> Result<(), Box<dyn Error>> {
    let baseline = run(0)?;
    for &variant in variants {
        check(&baseline, &run(variant)?)?;
        for _ in 0..20 {
            let _ = run(variant)?;
        }
    }
    device.synchronize()?;
    for &variant in variants {
        let start = Instant::now();
        for _ in 0..100 {
            let _ = run(variant)?;
        }
        device.synchronize()?;
        println!(
            "{label} variant={variant} us={:.3}",
            start.elapsed().as_secs_f64() * 1e4
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::new_metal(0)?;
    for width in [1024, 1025, 4097] {
        let logits = Tensor::rand(-20f32, -1f32, (1, 8, 130, width), &device)?;
        bench(
            &device,
            &format!("select width={width}"),
            |v| select(&logits, v == 1),
            &[0, 1, 1, 0],
        )?;
    }
    let logits = Tensor::rand(-5f32, 5f32, (2, 8, 157, 1025), &device)?.to_dtype(DType::F16)?;
    let cast = |variant| -> candle_core::Result<(Tensor, Tensor)> {
        let batch = if variant == 2 {
            logits.to_dtype(DType::F32)?
        } else {
            logits.clone()
        };
        let c = batch.i((0..1, .., 27..157, ..))?;
        let u = batch.i((1..2, .., 0..130, ..))?;
        if variant == 1 {
            Ok((
                c.contiguous()?.to_dtype(DType::F32)?,
                u.contiguous()?.to_dtype(DType::F32)?,
            ))
        } else {
            Ok((c.to_dtype(DType::F32)?, u.to_dtype(DType::F32)?))
        }
    };
    let (c, u) = cast(0)?;
    bench(
        &device,
        "full scoring",
        |v| score(&c, &u, v == 1),
        &[0, 1, 1, 0],
    )?;
    bench(
        &device,
        "cast plus scoring",
        |v| {
            let (c, u) = cast(v)?;
            score(&c, &u, false)
        },
        &[0, 1, 2, 2, 1, 0],
    )?;
    Ok(())
}
