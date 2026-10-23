//! Real decoder weights with deterministic synthetic activations, not a production profile.
//! Compare explicit window layouts including padding, materialization, bias and Snake.
use std::{error::Error, path::PathBuf, time::Instant};

use candle_core::{DType, Device, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};
use talechime_omnivoice::stage1_decoder::Stage1DecoderBundle;

fn timing(
    device: &Device,
    mut run: impl FnMut() -> candle_core::Result<Tensor>,
) -> candle_core::Result<f64> {
    for _ in 0..2 {
        let _ = run()?;
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..4 {
        let _ = run()?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 250.0)
}

fn values(t: &Tensor) -> candle_core::Result<Vec<f32>> {
    t.flatten_all()?.to_vec1::<f32>()
}

fn snake(xs: &Tensor, alpha: &Tensor) -> candle_core::Result<Tensor> {
    let sin = alpha.broadcast_mul(xs)?.sin()?;
    xs + (alpha + 1e-9)?.recip()?.broadcast_mul(&(&sin * &sin)?)?
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: conv7_layout_probe MODEL FRAMES".into());
    }
    let frames: usize = args[1].parse()?;
    if !(1..=256).contains(&frames) {
        return Err("frames must be in 1..=256".into());
    }
    let device = Device::new_metal(0)?;
    let bundle = Stage1DecoderBundle::from_model_root(PathBuf::from(&args[0]))?;
    // SAFETY: only the existing immutable local model is mapped.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[bundle.weights_path()], DType::F32, &device)?
    };
    let mut channels = bundle.model_config().decoder_hidden_dim;
    let mut len = frames;
    for (block, stride) in bundle.model_config().decoder_strides.iter().enumerate() {
        channels /= 2;
        len *= stride;
        let data: Vec<_> = (0..channels * len)
            .map(|i| ((i % 257) as f32 - 128.0) / 256.0)
            .collect();
        let xs = Tensor::from_vec(data, (1, channels, len), &device)?;
        for (unit, dilation) in [(1, 1), (2, 3), (3, 9)] {
            let base = vb.pp(format!("acoustic_decoder.block.{block}.res_unit{unit}"));
            let weight = base.pp("conv1").get((channels, channels, 7), "weight")?;
            let bias = base
                .pp("conv1")
                .get(channels, "bias")?
                .reshape((1, channels, 1))?;
            let alpha = base.pp("snake2").get((1, channels, 1), "alpha")?;
            let conv = Conv1d::new(
                weight.clone(),
                Some(bias.flatten_all()?),
                Conv1dConfig {
                    padding: 3 * dilation,
                    dilation,
                    ..Default::default()
                },
            );
            // unfold creates views; contiguous is the actual window materialization.
            // Both orders preserve the contraction index order [input_channel, tap].
            let columns = |ncl: bool| -> candle_core::Result<Tensor> {
                let windows = xs
                    .pad_with_zeros(2, 3 * dilation, 3 * dilation)?
                    .unfold(2, 6 * dilation + 1, 1)?
                    .unfold(3, 1, dilation)?
                    .squeeze(4)?;
                if ncl {
                    windows
                        .permute((0, 1, 3, 2))?
                        .contiguous()?
                        .reshape((1, channels * 7, len))
                } else {
                    windows
                        .permute((0, 2, 1, 3))?
                        .contiguous()?
                        .reshape((1, len, channels * 7))
                }
            };
            let w = weight.reshape((1, channels, channels * 7))?;
            let run = |variant| -> candle_core::Result<Tensor> {
                match variant {
                    0 => conv.forward(&xs),
                    1 => columns(false)?
                        .matmul(&w.t()?)?
                        .transpose(1, 2)?
                        .contiguous()?
                        .broadcast_add(&bias),
                    _ => w.matmul(&columns(true)?)?.broadcast_add(&bias),
                }
            };
            let reference = values(&run(0)?)?;
            for variant in [0, 1, 2, 2, 1, 0] {
                let actual = values(&run(variant)?)?;
                let differing = reference
                    .iter()
                    .zip(&actual)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let max_abs = reference
                    .iter()
                    .zip(&actual)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                let ms = timing(&device, || run(variant))?;
                let consumed_ms = timing(&device, || snake(&run(variant)?, &alpha))?;
                println!(
                    "{}",
                    serde_json::json!({"op":"full","block":block,"unit":unit,"channels":channels,"len":len,"variant":variant,"ms":ms,"consumed_ms":consumed_ms,"differing":differing,"max_abs":max_abs})
                );
            }
            // Isolated component costs are diagnostic and must not be added together.
            for ncl in [false, true] {
                let col = columns(ncl)?;
                let gemm = || {
                    if ncl {
                        w.matmul(&col)
                    } else {
                        col.matmul(&w.t()?)
                    }
                };
                let product = gemm()?;
                let materialize_ms = timing(&device, || columns(ncl))?;
                let gemm_ms = timing(&device, gemm)?;
                let output_ms = timing(&device, || {
                    let y = if ncl {
                        product.clone()
                    } else {
                        product.transpose(1, 2)?.contiguous()?
                    };
                    y.broadcast_add(&bias)
                })?;
                println!(
                    "{}",
                    serde_json::json!({"op":"components","block":block,"unit":unit,"ncl":ncl,"expanded_bytes":channels * len * 7 * 4,"materialize_ms":materialize_ms,"gemm_ms":gemm_ms,"output_ms":output_ms})
                );
            }
        }
    }
    Ok(())
}
