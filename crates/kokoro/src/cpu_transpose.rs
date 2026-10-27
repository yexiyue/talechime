//! CPU transposed convolution, preserving Candle dot kernels and tap order.
use candle_core::cpu::kernels::VecOps;
use candle_core::{Result, Tensor};
use rayon::prelude::*;

/// Schedule independent output rows together, retaining Candle's tap order and
/// exact vector-dot kernel rather than changing the floating-point reduction.
pub(crate) fn transpose(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> Result<Tensor> {
    let (batch, input, steps) = x.dims3()?;
    let (weight_input, output, kernel) = w.dims3()?;
    if input == 0
        || output == 0
        || steps == 0
        || kernel == 0
        || stride == 0
        || weight_input != input
    {
        candle_core::bail!("invalid transposed convolution shape");
    }
    let output_steps = (steps - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(kernel))
        .and_then(|n| n.checked_add(output_padding))
        .and_then(|n| padding.checked_mul(2).and_then(|p| n.checked_sub(p)))
        .filter(|&n| n > 0)
        .ok_or_else(|| candle_core::Error::Msg("invalid transposed convolution extent".into()))?;
    let values = x.flatten_all()?.to_vec1::<f32>()?;
    let weights = w.flatten_all()?.to_vec1::<f32>()?;
    let mut packed = vec![0.0; values.len()];
    for (index, row) in packed.chunks_exact_mut(input).enumerate() {
        let b = index / steps;
        let t = index % steps;
        for (c, value) in row.iter_mut().enumerate() {
            *value = values[(b * input + c) * steps + t];
        }
    }
    let mut result = vec![0.0; batch * output * output_steps];
    let apply = |(index, row): (usize, &mut [f32])| {
        let b = index / output;
        let c = index % output;
        let mut tap = vec![0.0; input];
        for k in 0..kernel {
            for (i, value) in tap.iter_mut().enumerate() {
                *value = weights[(i * output + c) * kernel + k];
            }
            let start = padding.saturating_sub(k).div_ceil(stride).min(steps);
            let end = (padding + output_steps)
                .saturating_sub(k)
                .div_ceil(stride)
                .min(steps);
            for t in start..end {
                let source = &packed[(b * steps + t) * input..(b * steps + t + 1) * input];
                let mut dot = 0.0;
                // SAFETY: both slices have exactly `input` elements, and the
                // result points to a live, disjoint scalar on this worker.
                unsafe { <f32 as VecOps>::vec_dot(source.as_ptr(), tap.as_ptr(), &mut dot, input) };
                row[t * stride + k - padding] += dot;
            }
        }
    };
    if result.len() >= 64 * 1024 && output > 1 {
        result
            .par_chunks_mut(output_steps)
            .enumerate()
            .for_each(apply);
    } else {
        result.chunks_mut(output_steps).enumerate().for_each(apply);
    }
    Tensor::from_vec(result, (batch, output, output_steps), x.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transposed_rows_match_candle_dot_and_accumulation_bits() -> Result<()> {
        for (batch, input, output, steps, kernel, stride, padding, tail) in [
            (1, 1, 1, 1, 3, 2, 0, 0),
            (2, 3, 5, 13, 3, 2, 1, 1),
            (2, 7, 3, 17, 7, 3, 2, 0),
            (1, 32, 16, 257, 20, 10, 5, 0),
            (1, 16, 8, 513, 12, 6, 3, 0),
            (2, 17, 16, 257, 20, 10, 5, 0),
        ] {
            let device = &candle_core::Device::Cpu;
            let x = Tensor::from_vec(
                (0..batch * input * steps)
                    .map(|i| ((i * 17 % 101) as f32 - 50.0) / 31.0)
                    .collect::<Vec<_>>(),
                (batch, input, steps),
                device,
            )?;
            let w = Tensor::from_vec(
                (0..(input + 1) * output * kernel)
                    .map(|i| ((i * 13 % 103) as f32 - 51.0) / 37.0)
                    .collect::<Vec<_>>(),
                (input + 1, output, kernel),
                device,
            )?
            .narrow(0, 1, input)?;
            let expected = x
                .conv_transpose1d(&w, padding, tail, stride, 1, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for input in [&x, &x.transpose(1, 2)?.contiguous()?.transpose(1, 2)?] {
                for weight in [&w, &w.transpose(0, 1)?.contiguous()?.transpose(0, 1)?] {
                    let actual = transpose(input, weight, stride, padding, tail)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert_eq!(actual.len(), expected.len());
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
}
