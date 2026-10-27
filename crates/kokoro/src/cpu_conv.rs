//! Parallel window construction with Candle's original GEMM layout and reduction.
use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

pub(crate) fn forward(x: &Tensor, w: &Tensor, stride: usize, dilation: usize) -> Result<Tensor> {
    let (batch, input, steps) = x.dims3()?;
    let (output, columns, kernel) = w.dims3()?;
    if input == 0 || steps == 0 || kernel == 0 || stride == 0 || dilation == 0 || input != columns {
        candle_core::bail!("invalid CPU convolution shape");
    }
    let padding = (kernel - 1) * dilation / 2;
    let output_steps = (steps + 2 * padding)
        .checked_sub(dilation * (kernel - 1) + 1)
        .ok_or_else(|| candle_core::Error::Msg("convolution has no output positions".into()))?
        / stride
        + 1;
    let columns = x.apply_op1_no_bwd(&Columns {
        kernel,
        stride,
        dilation,
        padding,
        output_steps,
    })?;
    // Preserve [batch, time, channel*tap] x [batch, channel*tap, output].
    // Bias remains a separate addition in the caller, as in Candle conv1d.
    let weights = w
        .reshape((1, output, input * kernel))?
        .transpose(1, 2)?
        .broadcast_as((batch, input * kernel, output))?;
    let result = columns.matmul(&weights)?;
    if output == 1 || output_steps == 1 {
        result.transpose(1, 2)?.contiguous()
    } else {
        result.apply_op1_no_bwd(&ChannelFirst)
    }
}

/// Materialize [batch, time, channel] as [batch, channel, time] in cache tiles.
/// Each worker owns whole output channels; no arithmetic or reduction changes.
struct ChannelFirst;

impl CustomOp1 for ChannelFirst {
    fn name(&self) -> &'static str {
        "kokoro-cpu-channel-first"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (batch, steps, channels) = layout.shape().dims3()?;
        let source = storage.as_slice::<f32>()?;
        let strides = layout.stride();
        let mut output = vec![0.0; batch * channels * steps];
        let group_channels = channels.div_ceil(rayon::current_num_threads()).clamp(1, 16);
        // Groups never cross a batch boundary, including a partial final group.
        let transpose = |(b, output): (usize, &mut [f32])| {
            output
                .par_chunks_mut(group_channels * steps)
                .enumerate()
                .for_each(|(group, output)| {
                    for time in (0..steps).step_by(64) {
                        let end = (time + 64).min(steps);
                        for (channel, row) in output.chunks_mut(steps).enumerate() {
                            let base = layout.start_offset()
                                + b * strides[0]
                                + (group * group_channels + channel) * strides[2];
                            for (offset, value) in row[time..end].iter_mut().enumerate() {
                                *value = source[base + (time + offset) * strides[1]];
                            }
                        }
                    }
                });
        };
        if output.len() >= 64 * 1024 && steps > 0 && channels > 0 {
            output
                .par_chunks_mut(channels * steps)
                .enumerate()
                .for_each(transpose);
        } else {
            for b in 0..batch {
                for c in 0..channels {
                    for t in 0..steps {
                        output[(b * channels + c) * steps + t] = source[layout.start_offset()
                            + b * strides[0]
                            + t * strides[1]
                            + c * strides[2]];
                    }
                }
            }
        }
        Ok((CpuStorage::F32(output), (batch, channels, steps).into()))
    }
}

struct Columns {
    kernel: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    output_steps: usize,
}

impl CustomOp1 for Columns {
    fn name(&self) -> &'static str {
        "kokoro-cpu-columns"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (batch, channels, steps) = layout.shape().dims3()?;
        let values = storage.as_slice::<f32>()?;
        let strides = layout.stride();
        let width = channels * self.kernel;
        let mut output = vec![0.0; batch * self.output_steps * width];
        let fill = |(index, row): (usize, &mut [f32])| {
            let batch = index / self.output_steps;
            let time = index % self.output_steps;
            let base = layout.start_offset() + batch * strides[0];
            for (channel, taps) in row.chunks_exact_mut(self.kernel).enumerate() {
                for (tap, value) in taps.iter_mut().enumerate() {
                    let position = time * self.stride + tap * self.dilation;
                    if position >= self.padding && position - self.padding < steps {
                        *value = values
                            [base + channel * strides[1] + (position - self.padding) * strides[2]];
                    }
                }
            }
        };
        if output.len() >= 64 * 1024 && self.output_steps > 1 {
            output.par_chunks_mut(width).enumerate().for_each(fill);
        } else {
            output.chunks_mut(width).enumerate().for_each(fill);
        }
        Ok((
            CpuStorage::F32(output),
            (batch, self.output_steps, width).into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn channel_first_tiles_preserve_strided_bits_and_partial_groups() -> Result<()> {
        for (batch, steps, channels) in [(1, 7, 3), (2, 1025, 35), (2, 2049, 17)] {
            let x = Tensor::from_vec(
                (0..(batch + 1) * steps * channels)
                    .map(|i| [0.0, -0.0, f32::INFINITY, f32::NAN, 1.25][i % 5])
                    .collect::<Vec<_>>(),
                (batch + 1, steps, channels),
                &Device::Cpu,
            )?
            .narrow(0, 1, batch)?;
            for x in [&x, &x.transpose(1, 2)?.contiguous()?.transpose(1, 2)?] {
                let expected = x
                    .transpose(1, 2)?
                    .contiguous()?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let actual = x
                    .apply_op1_no_bwd(&ChannelFirst)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
        Ok(())
    }

    #[test]
    fn windows_match_candle_convolution_bits_with_strides_and_offsets() -> Result<()> {
        for (batch, input, output, steps, kernel, stride, dilation) in [
            (1, 3, 5, 13, 3, 1, 1),
            (2, 3, 5, 17, 7, 1, 3),
            (1, 2, 3, 1, 11, 1, 5),
            (2, 3, 5, 13, 3, 2, 1),
            (1, 2, 3, 17, 4, 2, 3),
            (1, 32, 64, 257, 11, 1, 5),
        ] {
            let x = Tensor::from_vec(
                (0..(batch + 1) * input * steps)
                    .map(|i| ((i * 17 % 101) as f32 - 50.0) / 37.0)
                    .collect::<Vec<_>>(),
                (batch + 1, input, steps),
                &Device::Cpu,
            )?
            .narrow(0, 1, batch)?;
            let w = Tensor::from_vec(
                (0..output * input * kernel)
                    .map(|i| ((i * 13 % 103) as f32 - 51.0) / 41.0)
                    .collect::<Vec<_>>(),
                (output, input, kernel),
                &Device::Cpu,
            )?;
            let expected = x
                .contiguous()?
                .conv1d(&w, (kernel - 1) * dilation / 2, stride, dilation, 1)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for x in [&x, &x.transpose(1, 2)?.contiguous()?.transpose(1, 2)?] {
                for w in [&w, &w.transpose(0, 1)?.contiguous()?.transpose(0, 1)?] {
                    let actual = forward(x, w, stride, dilation)?
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
