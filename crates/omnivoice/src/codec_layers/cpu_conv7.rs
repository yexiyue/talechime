//! Offline window-layout attribution with real decoder inputs; no runtime switch.
use std::{hint::black_box, time::Instant};

use super::*;
use crate::cpu_attribution as io;

fn columns(xs: &Tensor, ncl: bool) -> CandleResult<Tensor> {
    let (batch, channels, len) = xs.dims3()?;
    let windows = xs.pad_with_zeros(2, 3, 3)?.unfold(2, 7, 1)?;
    if ncl {
        windows
            .permute((0, 1, 3, 2))?
            .contiguous()?
            .reshape((batch, channels * 7, len))
    } else {
        windows
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((batch, len, channels * 7))
    }
}

fn snake(xs: &Tensor, alpha: &Tensor) -> CandleResult<Tensor> {
    let sin = alpha.broadcast_mul(xs)?.sin()?;
    xs + (alpha + 1e-9)?.recip()?.broadcast_mul(&(&sin * &sin)?)?
}

fn difference(expected: &Tensor, actual: &Tensor) -> CandleResult<serde_json::Value> {
    assert_eq!(expected.dims(), actual.dims());
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let differing = expected
        .iter()
        .zip(&actual)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max_abs = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    Ok(serde_json::json!({"differing":differing,"elements":actual.len(),"max_abs":max_abs}))
}

#[test]
fn cpu_decoder_conv7_preserves_bits_and_fallbacks() -> Result<()> {
    let device = &Device::Cpu;
    for channels in [32, 64, 128, 256, 512] {
        let weight = Tensor::rand(-0.1f32, 0.1, (channels + 1, channels, 7), device)?
            .narrow(0, 1, channels)?;
        for bias in [None, Some(Tensor::rand(-0.1f32, 0.1, channels, device)?)] {
            let conv = Conv1d::new(
                weight.clone(),
                bias,
                Conv1dConfig {
                    padding: 3,
                    ..Default::default()
                },
            );
            for batch in [1, 2] {
                for len in [1, 7, 8, 9, 13, 16, 24, 56, 64, 128, 130, 256, 1040] {
                    for strided in [false, true] {
                        let xs = if strided {
                            Tensor::rand(-0.1f32, 0.1, (batch, len, channels), device)?
                                .transpose(1, 2)?
                        } else {
                            Tensor::rand(-0.1f32, 0.1, (batch + 1, channels, len), device)?
                                .narrow(0, 1, batch)?
                        };
                        let reference = conv.forward(&xs)?;
                        for _ in 0..2 {
                            io::equal(&reference, &forward_dense_decoder_conv7(&conv, &xs)?)?;
                        }
                    }
                }
            }
        }
    }
    for (kernel, cfg) in [
        (1, Conv1dConfig::default()),
        (
            7,
            Conv1dConfig {
                stride: 2,
                padding: 3,
                ..Default::default()
            },
        ),
        (
            7,
            Conv1dConfig {
                padding: 0,
                ..Default::default()
            },
        ),
        (
            7,
            Conv1dConfig {
                dilation: 3,
                padding: 9,
                ..Default::default()
            },
        ),
        (
            7,
            Conv1dConfig {
                groups: 2,
                padding: 3,
                ..Default::default()
            },
        ),
    ] {
        let conv = Conv1d::new(
            Tensor::rand(-0.1f32, 0.1, (32, 32 / cfg.groups, kernel), device)?,
            None,
            cfg,
        );
        let xs = Tensor::rand(-0.1f32, 0.1, (2, 32, 32), device)?;
        io::equal(
            &conv.forward(&xs)?,
            &forward_dense_decoder_conv7(&conv, &xs)?,
        )?;
    }
    let conv = Conv1d::new(
        Tensor::rand(-0.1f32, 0.1, (32, 32, 7), device)?.transpose(0, 1)?,
        None,
        Conv1dConfig {
            padding: 3,
            ..Default::default()
        },
    );
    let xs = Tensor::rand(-0.1f32, 0.1, (1, 32, 32), device)?;
    io::equal(
        &conv.forward(&xs)?,
        &forward_dense_decoder_conv7(&conv, &xs)?,
    )?;
    for (input, output) in [(3, 5), (8, 8), (32, 64), (64, 32)] {
        let conv = Conv1d::new(
            Tensor::rand(-0.1f32, 0.1, (output, input, 7), device)?,
            None,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        );
        let xs = Tensor::rand(-0.1f32, 0.1, (2, input, 32), device)?;
        io::equal(
            &conv.forward(&xs)?,
            &forward_dense_decoder_conv7(&conv, &xs)?,
        )?;
    }
    let conv = Conv1d::new(
        Tensor::ones((32, 32, 7), DType::F16, device)?,
        None,
        Conv1dConfig {
            padding: 3,
            ..Default::default()
        },
    );
    let xs = Tensor::ones((1, 32, 32), DType::F16, device)?;
    if candle_core::utils::has_accelerate() {
        assert_eq!(
            conv.forward(&xs).unwrap_err().to_string(),
            forward_dense_decoder_conv7(&conv, &xs)
                .unwrap_err()
                .to_string()
        );
    } else {
        io::equal(
            &conv.forward(&xs)?.to_dtype(DType::F32)?,
            &forward_dense_decoder_conv7(&conv, &xs)?.to_dtype(DType::F32)?,
        )?;
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_conv7_keeps_original_path() -> Result<()> {
    let device = &Device::new_metal(0)?;
    let conv = Conv1d::new(
        Tensor::rand(-0.1f32, 0.1, (32, 32, 7), device)?,
        None,
        Conv1dConfig {
            padding: 3,
            ..Default::default()
        },
    );
    let xs = Tensor::rand(-0.1f32, 0.1, (1, 32, 32), device)?;
    io::equal(
        &conv.forward(&xs)?,
        &forward_dense_decoder_conv7(&conv, &xs)?,
    )
}

#[test]
#[ignore = "requires immutable local weights and real decoder activation archives"]
fn compare_real_conv7_layouts() -> Result<()> {
    assert!(candle_core::utils::has_accelerate());
    let bundle = crate::stage1_decoder::Stage1DecoderBundle::from_model_root(io::model()?)?;
    let ws = CodecWeightSource::from_weight_path(bundle.weights_path(), DType::F32, &Device::Cpu)?;
    let mut channels = bundle.model_config().decoder_hidden_dim;
    for block in 0..bundle.model_config().decoder_strides.len() {
        channels /= 2;
        let base = format!("decoder-frames-130-block-{block}-res_unit1");
        let capture = io::load(&format!("{base}-conv1"))?;
        let residual = io::load(&base)?;
        let xs = &capture["input"];
        let weights = ws.pp(format!("acoustic_decoder.block.{block}.res_unit1"));
        let conv = weights
            .pp("conv1")
            .load_conv1d_flexible(channels, channels, 7, 1, 3)?;
        let pointwise = weights
            .pp("conv2")
            .load_conv1d_flexible(channels, channels, 1, 1, 0)?;
        let alpha = weights.pp("snake2").get((1, channels, 1), "alpha")?;
        let CodecConv1dInner::Dense(layer) = &conv.inner else {
            panic!("dense fixture required")
        };
        let (batch, _, len) = xs.dims3()?;
        let w = layer
            .weight()
            .reshape((1, channels, channels * 7))?
            .broadcast_as((batch, channels, channels * 7))?;
        let bias = layer
            .bias()
            .expect("decoder bias")
            .reshape((1, channels, 1))?;
        let run_conv = |variant| -> CandleResult<Tensor> {
            match variant {
                0 => layer.forward(xs),
                1 => columns(xs, false)?
                    .matmul(&w.t()?)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .broadcast_add(&bias),
                _ => conv.forward_decoder_conv7(xs),
            }
        };
        let consume = |output: &Tensor| -> CandleResult<Tensor> {
            pointwise.forward_decoder_pointwise(&snake(output, &alpha)?)? + &residual["input"]
        };
        let reference = run_conv(0)?;
        io::equal(&reference, &capture["output"])?;
        let expected = consume(&reference)?;
        io::equal(&expected, &residual["output"])?;
        for variant in 0..3 {
            for repeat in 0..2 {
                let actual = run_conv(variant)?;
                println!(
                    "CPU_CONV7 {}",
                    serde_json::json!({"kind":"numeric","case":base,
                    "variant":variant,"repeat":repeat,"shape":xs.dims(),
                    "conv":difference(&reference,&actual)?,"consumer":difference(&expected,&consume(&actual)?)?})
                );
            }
        }
        for boundary in ["conv_bias", "snake_pointwise_residual"] {
            let run = |variant| -> CandleResult<Tensor> {
                let output = run_conv(variant)?;
                if boundary == "conv_bias" {
                    Ok(output)
                } else {
                    consume(&output)
                }
            };
            for variant in 0..3 {
                black_box(run(variant)?);
            }
            for group in 0..3 {
                for candidate in if group % 2 == 0 { [1, 2] } else { [2, 1] } {
                    for variant in [0, candidate, candidate, 0] {
                        let started = Instant::now();
                        for _ in 0..3 {
                            black_box(run(variant)?);
                        }
                        let us = started.elapsed().as_secs_f64() * 1e6 / 3.;
                        println!(
                            "CPU_CONV7 {}",
                            serde_json::json!({"kind":"timing","case":base,
                            "boundary":boundary,"candidate":candidate,"variant":variant,"group":group,"microseconds":us})
                        );
                    }
                }
            }
        }
        // Isolated components overlap with the full path and are diagnostic only.
        for ncl in [false, true] {
            let col = columns(xs, ncl)?;
            let gemm = || {
                if ncl {
                    w.matmul(&col)
                } else {
                    col.matmul(&w.t()?)
                }
            };
            let product = gemm()?;
            for component in ["windows", "gemm", "output_bias"] {
                let run = || -> CandleResult<Tensor> {
                    match component {
                        "windows" => columns(xs, ncl),
                        "gemm" => gemm(),
                        _ => {
                            let output = if ncl {
                                product.clone()
                            } else {
                                product.transpose(1, 2)?.contiguous()?
                            };
                            output.broadcast_add(&bias)
                        }
                    }
                };
                black_box(run()?);
                for group in 0..5 {
                    let started = Instant::now();
                    black_box(run()?);
                    let us = started.elapsed().as_secs_f64() * 1e6;
                    println!(
                        "CPU_CONV7 {}",
                        serde_json::json!({"kind":"component","case":base,
                        "ncl":ncl,"component":component,"group":group,"microseconds":us,
                        "expanded_bytes":batch*channels*len*7*4})
                    );
                }
            }
        }
    }
    Ok(())
}
