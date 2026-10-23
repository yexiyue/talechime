use super::*;

fn pointwise_cases(device: &Device) -> CandleResult<()> {
    let bits = |t: Tensor| -> CandleResult<Vec<u32>> {
        Ok(t.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    for (input, output) in [(3, 5), (8, 8), (16, 16), (32, 32), (32, 64), (64, 32)] {
        let weight = Tensor::rand(-0.1f32, 0.1f32, (output, input, 1), device)?;
        for bias in [None, Some(Tensor::rand(-0.1f32, 0.1f32, output, device)?)] {
            let conv = Conv1d::new(weight.clone(), bias, Conv1dConfig::default());
            let wrapper = CodecConv1d {
                inner: CodecConv1dInner::Dense(conv.clone()),
            };
            for batch in [1, 2] {
                for frames in [1, 7, 130, 179] {
                    for strided in [false, true] {
                        let xs = if strided {
                            Tensor::rand(-0.1f32, 0.1f32, (batch, frames, input), device)?
                                .transpose(1, 2)?
                        } else {
                            Tensor::rand(-0.1f32, 0.1f32, (batch, input, frames), device)?
                        };
                        let expected = bits(conv.forward(&xs)?)?;
                        assert_eq!(bits(wrapper.forward(&xs)?)?, expected);
                        for _ in 0..2 {
                            assert_eq!(
                                bits(wrapper.forward_decoder_pointwise(&xs)?)?,
                                expected,
                                "input={input} output={output} batch={batch} frames={frames} strided={strided}"
                            );
                        }
                    }
                }
            }
        }
    }
    for (kernel, config) in [
        (
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        ),
        (
            7,
            Conv1dConfig {
                padding: 9,
                dilation: 3,
                ..Default::default()
            },
        ),
        (
            1,
            Conv1dConfig {
                stride: 2,
                ..Default::default()
            },
        ),
        (
            1,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
        ),
        (
            1,
            Conv1dConfig {
                groups: 2,
                ..Default::default()
            },
        ),
    ] {
        let weight = Tensor::rand(-0.1f32, 0.1f32, (4, 4 / config.groups, kernel), device)?;
        let conv = Conv1d::new(weight, None, config);
        let xs = Tensor::rand(-0.1f32, 0.1f32, (2, 4, 23), device)?;
        assert_eq!(
            bits(forward_dense_decoder_pointwise(&conv, &xs)?)?,
            bits(conv.forward(&xs)?)?
        );
    }
    let conv = Conv1d::new(
        Tensor::ones((4, 4, 1), DType::F16, device)?,
        None,
        Conv1dConfig::default(),
    );
    let xs = Tensor::ones((1, 4, 7), DType::F16, device)?;
    if device.is_cpu() && candle_core::utils::has_accelerate() {
        let expected = conv.forward(&xs).unwrap_err().to_string();
        assert!(expected.contains("does not support f16 matmul"));
        assert_eq!(
            forward_dense_decoder_pointwise(&conv, &xs)
                .unwrap_err()
                .to_string(),
            expected
        );
    } else {
        assert_eq!(
            bits(forward_dense_decoder_pointwise(&conv, &xs)?)?,
            bits(conv.forward(&xs)?)?
        );
    }
    Ok(())
}

#[test]
fn cpu_decoder_pointwise_matches_original_bits_and_fallbacks() -> CandleResult<()> {
    pointwise_cases(&Device::Cpu)
}

#[test]
fn cpu_decoder_pointwise_preserves_bits_for_decoder_channels_and_layouts() -> CandleResult<()> {
    let device = &Device::Cpu;
    let bits = |t: Tensor| -> CandleResult<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    for channels in [32, 64, 128, 256, 512] {
        let weight = Tensor::rand(-0.1f32, 0.1, (channels + 1, channels, 1), device)?
            .narrow(0, 1, channels)?;
        for bias in [None, Some(Tensor::rand(-0.1f32, 0.1, channels, device)?)] {
            let conv = Conv1d::new(weight.clone(), bias, Conv1dConfig::default());
            for batch in [1, 2] {
                for len in [
                    1, 7, 8, 9, 13, 16, 24, 32, 40, 56, 64, 72, 96, 128, 130, 160, 179, 256, 280,
                    480, 520, 960, 1040,
                ] {
                    for strided in [false, true] {
                        let xs = if strided {
                            Tensor::rand(-0.1f32, 0.1, (batch, len, channels), device)?
                                .transpose(1, 2)?
                        } else {
                            Tensor::rand(-0.1f32, 0.1, (batch + 1, channels, len), device)?
                                .narrow(0, 1, batch)?
                        };
                        let reference = bits(conv.forward(&xs)?)?;
                        for _ in 0..2 {
                            let actual = bits(forward_dense_decoder_pointwise(&conv, &xs)?)?;
                            assert_eq!(
                                actual
                                    .iter()
                                    .zip(&reference)
                                    .filter(|(a, b)| a != b)
                                    .count(),
                                0,
                                "channels={channels} batch={batch} len={len} strided={strided}"
                            );
                        }
                    }
                }
            }
        }
        let xs = Tensor::rand(-0.1f32, 0.1, (1, channels, 32), device)?;
        let strided_weight =
            Tensor::rand(-0.1f32, 0.1, (channels, channels, 1), device)?.transpose(0, 1)?;
        let conv = Conv1d::new(strided_weight, None, Conv1dConfig::default());
        assert_eq!(
            bits(forward_dense_decoder_pointwise(&conv, &xs)?)?,
            bits(conv.forward(&xs)?)?
        );
    }
    Ok(())
}

#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn metal_decoder_pointwise_matches_original_bits_and_fallbacks() -> CandleResult<()> {
    pointwise_cases(&Device::new_metal(0)?)
}

#[test]
fn cpu_transposed_convolution_keeps_original_path() -> CandleResult<()> {
    let device = Device::Cpu;
    let xs = Tensor::from_vec(vec![0.1f32; 8], (1, 2, 4), &device)?;
    let weights = Tensor::from_vec(vec![0.2f32; 6], (2, 1, 3), &device)?;
    let config = ConvTranspose1dConfig {
        padding: 1,
        stride: 2,
        ..Default::default()
    };
    let layer = ConvTranspose1d::new(weights, None, config);
    let expected = layer.forward(&xs)?.flatten_all()?.to_vec1::<f32>()?;
    let actual = forward_dense_conv_transpose1d(&layer, &xs)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual, expected);
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn metal_transposed_convolution_crop_matches_host_reference() -> CandleResult<()> {
    let device = Device::new_metal(0)?;
    let (in_channels, out_channels) = (3, 2);
    for batch in [1, 2] {
        for frames in [1, 7, 130] {
            for stride in [2usize, 3, 4, 5, 8] {
                for with_bias in [false, true] {
                    let kernel = 2 * stride;
                    let config = ConvTranspose1dConfig {
                        stride,
                        padding: stride.div_ceil(2),
                        output_padding: stride % 2,
                        ..Default::default()
                    };
                    let values = (0..batch * in_channels * frames)
                        .map(|i| (i as f32 * 0.13).sin() * 0.2)
                        .collect::<Vec<_>>();
                    let weights = (0..in_channels * out_channels * kernel)
                        .map(|i| (i as f32 * 0.17).cos() * 0.1)
                        .collect::<Vec<_>>();
                    let bias = if with_bias {
                        Some(Tensor::from_vec(vec![0.03f32, -0.02], 2, &device)?)
                    } else {
                        None
                    };
                    let xs =
                        Tensor::from_vec(values.clone(), (batch, in_channels, frames), &device)?;
                    let layer = ConvTranspose1d::new(
                        Tensor::from_vec(
                            weights.clone(),
                            (in_channels, out_channels, kernel),
                            &device,
                        )?,
                        bias,
                        config,
                    );
                    let out_len =
                        (frames - 1) * stride + kernel - 2 * config.padding + config.output_padding;
                    let mut reference = vec![0f64; batch * out_channels * out_len];
                    if with_bias {
                        for (i, value) in reference.iter_mut().enumerate() {
                            *value = f64::from([0.03f32, -0.02][(i / out_len) % 2]);
                        }
                    }
                    for b in 0..batch {
                        for c in 0..in_channels {
                            for t in 0..frames {
                                let value = values[(b * in_channels + c) * frames + t];
                                for o in 0..out_channels {
                                    for k in 0..kernel {
                                        let position = t * stride + k;
                                        if position >= config.padding {
                                            let position = position - config.padding;
                                            if position < out_len {
                                                reference[(b * out_channels + o) * out_len
                                                    + position] += f64::from(value)
                                                    * f64::from(
                                                        weights
                                                            [(c * out_channels + o) * kernel + k],
                                                    );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let mut actuals = vec![forward_dense_conv_transpose1d(&layer, &xs)?];
                    // Candle's original debug length formula subtracts padding
                    // before adding the kernel width and underflows for one frame.
                    if (frames - 1) * stride >= 2 * config.padding {
                        actuals.push(layer.forward(&xs)?);
                    }
                    for tensor in actuals {
                        assert_eq!(tensor.dims(), &[batch, out_channels, out_len]);
                        let actual = tensor.flatten_all()?.to_vec1::<f32>()?;
                        for (index, (&actual, &expected)) in
                            actual.iter().zip(&reference).enumerate()
                        {
                            assert!(
                                actual.is_finite() && (f64::from(actual) - expected).abs() < 2e-5,
                                "batch={batch} frames={frames} stride={stride} bias={with_bias} index={index}: actual={actual} expected={expected}"
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
