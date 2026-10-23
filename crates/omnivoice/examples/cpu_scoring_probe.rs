//! Offline CPU probe for greedy confidence selection; no model downloads.
use std::{error::Error, time::Instant};

use candle_core::{D, DType, Device, Tensor};

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

fn data(width: usize, shift: f32) -> candle_core::Result<Tensor> {
    let mut state = 42u64;
    Tensor::from_vec(
        (0..8 * 130 * width)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u32 << 24) as f32 * 10.0 + shift
            })
            .collect::<Vec<_>>(),
        (1, 8, 130, width),
        &Device::Cpu,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::Cpu;
    for width in [1024, 1025, 4097] {
        let logits = data(width, -20.)?;
        bench(
            &device,
            &format!("select width={width}"),
            |v| select(&logits, v == 1),
            &[0, 1, 1, 0],
        )?;
    }
    let c = data(1025, -5.)?;
    let u = c.sin()?;
    bench(
        &device,
        "full scoring",
        |v| score(&c, &u, v == 1),
        &[0, 1, 1, 0],
    )?;
    Ok(())
}
