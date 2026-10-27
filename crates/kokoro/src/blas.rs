//! macOS CPU affine GEMM. PyTorch initializes the destination with bias and
//! calls BLAS with beta=1; adding bias after a beta=0 GEMM rounds differently.
use candle_core::{Result, Storage, Tensor};
use std::ffi::{c_char, c_int};

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn sgemm_(
        ta: *const c_char,
        tb: *const c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const f32,
        a: *const f32,
        lda: *const c_int,
        b: *const f32,
        ldb: *const c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: *const c_int,
    );
}

fn gemm(a: &[f32], b: &[f32], c: &mut [f32], dims: [usize; 3], transpose_a: bool) -> Result<()> {
    let [m, n, k] = dims;
    if m == 0 || n == 0 || k == 0 || a.len() != m * k || b.len() != k * n || c.len() != m * n {
        candle_core::bail!("invalid affine GEMM dimensions");
    }
    let m = c_int::try_from(m).map_err(candle_core::Error::wrap)?;
    let n = c_int::try_from(n).map_err(candle_core::Error::wrap)?;
    let k = c_int::try_from(k).map_err(candle_core::Error::wrap)?;
    let ta = if transpose_a { b'T' } else { b'N' } as c_char;
    let tb = b'N' as c_char;
    let lda = if transpose_a { k } else { m };
    // SAFETY: all column-major matrix extents and leading dimensions are
    // checked above; source and destination allocations do not overlap.
    unsafe {
        sgemm_(
            &ta,
            &tb,
            &m,
            &n,
            &k,
            &1.0,
            a.as_ptr(),
            &lda,
            b.as_ptr(),
            &k,
            &1.0,
            c.as_mut_ptr(),
            &m,
        );
    }
    Ok(())
}

pub(crate) fn linear(x: &Tensor, w: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let input = x.dim(x.rank() - 1)?;
    let (output, columns) = w.dims2()?;
    if input == 0 || columns != input || bias.dims() != [output] {
        candle_core::bail!("invalid affine linear shape");
    }
    let rows = x.elem_count() / input;
    with_affine_inputs(x, w, bias, |values, weights, bias| {
        let mut result = bias.repeat(rows);
        gemm(weights, values, &mut result, [output, rows, input], true)?;
        let mut shape = x.dims().to_vec();
        *shape.last_mut().unwrap() = output;
        Tensor::from_vec(result, shape, x.device())
    })
}

fn with_affine_inputs<T>(
    x: &Tensor,
    weights: &Tensor,
    bias: &Tensor,
    f: impl FnOnce(&[f32], &[f32], &[f32]) -> Result<T>,
) -> Result<T> {
    with_f32_slice(x, |x| {
        with_f32_slice(weights, |weights| {
            with_f32_slice(bias, |bias| f(x, weights, bias))
        })
    })
}

fn with_f32_slice<T>(tensor: &Tensor, f: impl FnOnce(&[f32]) -> Result<T>) -> Result<T> {
    let (storage, layout) = tensor.storage_and_layout();
    if let Storage::Cpu(cpu) = &*storage
        && let Some((start, end)) = layout.contiguous_offsets()
    {
        // The lease stays on the owner thread until the synchronous BLAS call
        // returns. Strided inputs retain the previous materialization path.
        return f(&cpu.as_slice::<f32>()?[start..end]);
    }
    drop(storage);
    f(&tensor.flatten_all()?.to_vec1::<f32>()?)
}

pub(crate) fn conv(
    x: &Tensor,
    w: &Tensor,
    bias: &Tensor,
    stride: usize,
    dilation: usize,
    scratch: &mut Vec<f32>,
) -> Result<Tensor> {
    let (batch, input, steps) = x.dims3()?;
    let (output, columns, kernel) = w.dims3()?;
    if input == 0
        || steps == 0
        || kernel == 0
        || stride == 0
        || dilation == 0
        || input != columns
        || bias.dims() != [output]
    {
        candle_core::bail!("invalid convolution affine shape");
    }
    with_affine_inputs(x, w, bias, |values, weights, bias| {
        let padding = (kernel - 1) * dilation / 2;
        let output_steps = (steps + 2 * padding)
            .checked_sub(dilation * (kernel - 1) + 1)
            .ok_or_else(|| candle_core::Error::Msg("convolution has no output positions".into()))?
            / stride
            + 1;
        let mut result = Vec::with_capacity(batch * output * output_steps);
        for _ in 0..batch {
            for &value in bias {
                result.extend(std::iter::repeat_n(value, output_steps));
            }
        }
        let pointwise = kernel == 1 && stride == 1;
        let required = if pointwise {
            0
        } else {
            input * kernel * output_steps
        };
        // This model-owned buffer contains no request state. Every used
        // element is overwritten by window(), including both padding tails.
        // Retain the high-water length to avoid clearing on shape switches.
        if scratch.len() < required {
            scratch.reserve_exact(required - scratch.len());
            scratch.resize(required, 0.0);
        }
        let scratch = &mut scratch[..required];
        for b in 0..batch {
            let values = &values[b * input * steps..(b + 1) * input * steps];
            let columns = if pointwise {
                // Identical column order and GEMM geometry; the input already is
                // the complete 1x1 window matrix.
                values
            } else {
                for (c, channel) in values.chunks_exact(steps).enumerate() {
                    for (k, row) in scratch
                        [c * kernel * output_steps..(c + 1) * kernel * output_steps]
                        .chunks_exact_mut(output_steps)
                        .enumerate()
                    {
                        window(channel, row, k * dilation, padding, stride);
                    }
                }
                &*scratch
            };
            gemm(
                columns,
                weights,
                &mut result[b * output * output_steps..(b + 1) * output * output_steps],
                [output_steps, output, input * kernel],
                false,
            )?;
        }
        Tensor::from_vec(result, (batch, output, output_steps), x.device())
    })
}

/// Fill one tap's time row without changing the contraction or padding order.
fn window(input: &[f32], output: &mut [f32], offset: usize, padding: usize, stride: usize) {
    if stride == 1 {
        let destination = padding.saturating_sub(offset).min(output.len());
        let source = offset.saturating_sub(padding).min(input.len());
        let count = (output.len() - destination).min(input.len() - source);
        output[..destination].fill(0.0);
        output[destination..destination + count].copy_from_slice(&input[source..source + count]);
        output[destination + count..].fill(0.0);
    } else {
        for (t, value) in output.iter_mut().enumerate() {
            let position = t * stride + offset;
            *value = if position >= padding && position - padding < input.len() {
                input[position - padding]
            } else {
                0.0
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_linear_matches_materialized_gemm_with_views_and_offsets() -> Result<()> {
        let device = &candle_core::Device::Cpu;
        for (rows, input, output) in [(1, 7, 13), (13, 17, 19), (1, 256, 1024)] {
            let x = Tensor::from_vec(
                (0..(rows + 1) * input)
                    .map(|i| ((i * 17 % 101) as f32 - 50.0) / 37.0)
                    .collect::<Vec<_>>(),
                (rows + 1, input),
                device,
            )?
            .narrow(0, 1, rows)?;
            let w = Tensor::from_vec(
                (0..input * output)
                    .map(|i| ((i * 13 % 103) as f32 - 51.0) / 41.0)
                    .collect::<Vec<_>>(),
                (output, input),
                device,
            )?;
            let bias = Tensor::from_vec(
                (0..output).map(|i| i as f32 / 19.0).collect::<Vec<_>>(),
                output,
                device,
            )?;
            let mut expected = bias.to_vec1::<f32>()?.repeat(rows);
            gemm(
                &w.flatten_all()?.to_vec1::<f32>()?,
                &x.flatten_all()?.to_vec1::<f32>()?,
                &mut expected,
                [output, rows, input],
                true,
            )?;
            for input in [&x, &x.t()?.contiguous()?.t()?] {
                for weight in [&w, &w.t()?.contiguous()?.t()?] {
                    let actual = linear(input, weight, &bias)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert!(
                        actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn empty_convolution_output_is_rejected_before_window_allocation() -> Result<()> {
        let device = &candle_core::Device::Cpu;
        let x = Tensor::ones((1, 1, 1), candle_core::DType::F32, device)?;
        let w = Tensor::ones((1, 1, 2), candle_core::DType::F32, device)?;
        let bias = Tensor::zeros(1, candle_core::DType::F32, device)?;
        for stride in [1, 2] {
            assert!(conv(&x, &w, &bias, stride, 1, &mut Vec::new()).is_err());
        }
        Ok(())
    }

    #[test]
    fn windows_match_scalar_indexing_including_padding_bits() {
        let input = [0.0, -0.0, 1.5, -2.0, f32::from_bits(1), f32::INFINITY];
        for steps in 1..=input.len() {
            for output_steps in [1, 3, 8, 13] {
                for offset in 0..=12 {
                    for padding in 0..=12 {
                        for stride in [1, 2, 3] {
                            let mut actual = vec![f32::NAN; output_steps];
                            window(&input[..steps], &mut actual, offset, padding, stride);
                            for (t, value) in actual.into_iter().enumerate() {
                                let position = t * stride + offset;
                                let expected = if position >= padding && position - padding < steps
                                {
                                    input[position - padding]
                                } else {
                                    0.0
                                };
                                assert_eq!(value.to_bits(), expected.to_bits());
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn convolution_matches_original_columns_and_gemm_bitwise() -> Result<()> {
        let mut scratch = Vec::new();
        for (batch, input, output, steps, kernel, stride, dilation) in [
            (1, 3, 5, 13, 1, 1, 1),
            (2, 3, 5, 17, 1, 1, 5),
            (2, 3, 5, 17, 3, 1, 1),
            (2, 3, 5, 13, 7, 1, 3),
            (1, 2, 3, 1, 11, 1, 5),
            (2, 3, 5, 13, 3, 2, 1),
            (1, 2, 3, 17, 4, 2, 3),
            (1, 32, 64, 257, 11, 1, 5),
        ] {
            let values: Vec<_> = (0..batch * input * steps)
                .map(|i| ((i * 17 % 101) as f32 - 50.0) / 37.0)
                .collect();
            let weights: Vec<_> = (0..output * input * kernel)
                .map(|i| ((i * 13 % 103) as f32 - 51.0) / 41.0)
                .collect();
            let bias: Vec<_> = (0..output).map(|i| i as f32 / 19.0).collect();
            let x = Tensor::from_slice(&values, (batch, input, steps), &candle_core::Device::Cpu)?;
            let w = Tensor::from_slice(&weights, (output, input, kernel), x.device())?;
            let b = Tensor::from_slice(&bias, output, x.device())?;
            let actual = conv(&x, &w, &b, stride, dilation, &mut scratch)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let padding = (kernel - 1) * dilation / 2;
            let output_steps = (steps + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
            let mut expected = Vec::new();
            for _ in 0..batch {
                for &bias in &bias {
                    expected.extend(std::iter::repeat_n(bias, output_steps));
                }
            }
            for b in 0..batch {
                let mut columns = vec![0.0; input * kernel * output_steps];
                for c in 0..input {
                    for k in 0..kernel {
                        for t in 0..output_steps {
                            let position = t * stride + k * dilation;
                            if position >= padding && position - padding < steps {
                                columns[(c * kernel + k) * output_steps + t] =
                                    values[(b * input + c) * steps + position - padding];
                            }
                        }
                    }
                }
                gemm(
                    &columns,
                    &weights,
                    &mut expected[b * output * output_steps..(b + 1) * output * output_steps],
                    [output_steps, output, input * kernel],
                    false,
                )?;
            }
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            scratch.fill(f32::NAN);
            let strided_x = x.transpose(1, 2)?.contiguous()?.transpose(1, 2)?;
            let strided_w = w.transpose(0, 1)?.contiguous()?.transpose(0, 1)?;
            let strided = conv(&strided_x, &strided_w, &b, stride, dilation, &mut scratch)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(
                strided
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        Ok(())
    }
}
