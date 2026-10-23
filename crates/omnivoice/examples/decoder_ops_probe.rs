//! Real-weight decoder operator probes; diagnostic timings are not additive.
use std::{error::Error, path::PathBuf, time::Instant};

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};
use talechime_omnivoice::stage1_decoder::Stage1DecoderBundle;

fn bits(t: &Tensor) -> candle_core::Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn snake(xs: &Tensor, alpha: &Tensor, inverse: Option<&Tensor>) -> candle_core::Result<Tensor> {
    let sin = alpha.broadcast_mul(xs)?.sin()?;
    let sin = (&sin * &sin)?;
    let inverse = match inverse {
        Some(v) => v.clone(),
        None => (alpha + 1e-9)?.recip()?,
    };
    xs + inverse.broadcast_mul(&sin)?
}

fn timing(
    device: &Device,
    mut run: impl FnMut() -> candle_core::Result<Tensor>,
) -> candle_core::Result<f64> {
    for _ in 0..3 {
        let _ = run()?;
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..10 {
        let _ = run()?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e5)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: decoder_ops_probe MODEL FRAMES".into());
    }
    let frames: usize = args[1].parse()?;
    if !(1..=1024).contains(&frames) {
        return Err("frames must be in 1..=1024".into());
    }
    let device = Device::new_metal(0)?;
    let bundle = Stage1DecoderBundle::from_model_root(PathBuf::from(&args[0]))?;
    let cfg = bundle.model_config();
    // SAFETY: the pinned local model file is immutable during this probe.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[bundle.weights_path()], DType::F32, &device)?
    };
    let mut channels = cfg.decoder_hidden_dim;
    let mut len = frames;
    for (block, stride) in cfg.decoder_strides.iter().enumerate() {
        let alpha = vb
            .pp(format!("acoustic_decoder.block.{block}.snake1"))
            .get((1, channels, 1), "alpha")?;
        let inverse = (&alpha + 1e-9)?.recip()?;
        let xs = Tensor::rand(-0.5f32, 0.5f32, (1, channels, len), &device)?;
        let equal =
            bits(&snake(&xs, &alpha, None)?)? == bits(&snake(&xs, &alpha, Some(&inverse))?)?;
        if !equal {
            return Err("Snake cache changes bits".into());
        }
        for cached in [false, true, true, false] {
            let us = timing(&device, || snake(&xs, &alpha, cached.then_some(&inverse)))?;
            println!(
                "{}",
                serde_json::json!({"op":"snake", "block":block,"channels":channels,"len":len,"cached":cached,"us":us,"equal":equal})
            );
        }
        channels /= 2;
        len *= stride;
        for (unit, dilation) in [(1, 1), (2, 3), (3, 9)] {
            let base = vb.pp(format!("acoustic_decoder.block.{block}.res_unit{unit}"));
            let alpha = base.pp("snake2").get((1, channels, 1), "alpha")?;
            let inverse = (&alpha + 1e-9)?.recip()?;
            let xs = Tensor::rand(-0.5f32, 0.5f32, (1, channels, len), &device)?;
            for kernel in [1, 7] {
                let layer = base.pp(if kernel == 1 { "conv2" } else { "conv1" });
                let weight = layer.get((channels, channels, kernel), "weight")?;
                let bias = layer.get(channels, "bias")?;
                let config = Conv1dConfig {
                    dilation: if kernel == 1 { 1 } else { dilation },
                    padding: if kernel == 1 { 0 } else { 3 * dilation },
                    ..Default::default()
                };
                let conv = Conv1d::new(weight.clone(), Some(bias.clone()), config);
                let baseline = || conv.forward(&xs);
                let original_us = timing(&device, baseline)?;
                if kernel == 7 {
                    println!(
                        "{}",
                        serde_json::json!({"op":"conv7","block":block,"unit":unit,"channels":channels,"len":len,"dilation":dilation,"us":original_us})
                    );
                    continue;
                }
                let direct = || -> candle_core::Result<Tensor> {
                    let x = xs.transpose(1, 2)?.contiguous()?;
                    let w = weight.squeeze(2)?.t()?.unsqueeze(0)?;
                    x.matmul(&w)?
                        .transpose(1, 2)?
                        .contiguous()?
                        .broadcast_add(&bias.reshape((1, channels, 1))?)
                };
                let run = |variant| -> candle_core::Result<Tensor> {
                    match variant {
                        0 => baseline(),
                        1 => direct(),
                        _ => weight
                            .squeeze(2)?
                            .unsqueeze(0)?
                            .matmul(&xs)?
                            .broadcast_add(&bias.reshape((1, channels, 1))?),
                    }
                };
                let reference = bits(&baseline()?)?;
                for variant in [0, 1, 2, 2, 1, 0] {
                    let actual = bits(&run(variant)?)?;
                    let differing = reference
                        .iter()
                        .zip(&actual)
                        .filter(|(a, b)| a != b)
                        .count();
                    let us = timing(&device, || run(variant))?;
                    let chain_us = timing(&device, || {
                        let y = run(variant)?;
                        snake(&(y + &xs)?, &alpha, Some(&inverse))
                    })?;
                    println!(
                        "{}",
                        serde_json::json!({"op":"conv1","block":block,"unit":unit,"channels":channels,"len":len,"variant":variant,"equal":differing==0,"differing":differing,"us":us,"chain_us":chain_us})
                    );
                }
                let layout_us = timing(&device, || xs.transpose(1, 2)?.contiguous())?;
                println!(
                    "{}",
                    serde_json::json!({"op":"layout","block":block,"unit":unit,"channels":channels,"len":len,"us":layout_us})
                );
            }
        }
    }
    Ok(())
}
