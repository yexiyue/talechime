//! PyTorch-layout inference operations used by the official Kokoro architecture.
use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::Module;
use rayon::prelude::*;
use std::collections::HashMap;

pub(crate) struct Weights {
    pub(crate) tensors: HashMap<String, Tensor>,
    // CPU im2col storage belongs to this model and is released with it. A mutex
    // preserves the underlying model's auto traits; inference stays on its owner.
    #[cfg(target_os = "macos")]
    conv_scratch: std::sync::Mutex<Vec<f32>>,
}

fn linear(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    #[cfg(target_os = "macos")]
    if x.device().is_cpu()
        && let Some(bias) = bias
    {
        return crate::blas::linear(x, w, bias);
    }
    candle_nn::Linear::new(w.clone(), bias.cloned()).forward(x)
}

impl Weights {
    pub fn new(tensors: HashMap<String, Tensor>) -> Self {
        Self {
            tensors,
            #[cfg(target_os = "macos")]
            conv_scratch: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing Kokoro tensor: {name}")))
    }

    pub fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        linear(
            x,
            self.get(&format!("{name}.weight"))?,
            self.tensors.get(&format!("{name}.bias")),
        )
    }

    pub fn conv(&self, x: &Tensor, name: &str, stride: usize, dilation: usize) -> Result<Tensor> {
        let w = self.get(&format!("{name}.weight"))?;
        #[cfg(target_os = "macos")]
        if x.device().is_cpu()
            && let Some(bias) = self.tensors.get(&format!("{name}.bias"))
        {
            let mut scratch = self.conv_scratch.lock().map_err(|_| {
                candle_core::Error::Msg("Kokoro convolution scratch lock poisoned".into())
            })?;
            return crate::blas::conv(x, w, bias, stride, dilation, &mut scratch);
        }
        let padding = (w.dim(2)? - 1) * dilation / 2;
        let y = if w.dim(2)? == 1 && stride == 1 {
            // PyTorch's pointwise convolution is a matrix product in channel-
            // first layout. Keep that layout instead of an im2col reduction.
            let (batch, _, _) = x.dims3()?;
            let (output, input, _) = w.dims3()?;
            w.squeeze(2)?
                .unsqueeze(0)?
                .broadcast_as((batch, output, input))?
                .contiguous()?
                .matmul(&x.contiguous()?)?
        } else {
            x.contiguous()?.conv1d(w, padding, stride, dilation, 1)?
        };
        self.bias_conv(y, name)
    }

    pub fn transpose(
        &self,
        x: &Tensor,
        name: &str,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<Tensor> {
        let w = self.get(&format!("{name}.weight"))?;
        #[cfg(target_os = "macos")]
        if groups == 1
            && padding > 0
            && x.device().is_cpu()
            && w.device().is_cpu()
            && x.dtype() == DType::F32
            && w.dtype() == DType::F32
        {
            return self.bias_conv(
                crate::blas::transpose(x, w, stride, padding, output_padding)?,
                name,
            );
        }
        let y = x
            .contiguous()?
            .conv_transpose1d(w, padding, output_padding, stride, 1, groups)?;
        self.bias_conv(y, name)
    }

    fn bias_conv(&self, x: Tensor, name: &str) -> Result<Tensor> {
        match self.tensors.get(&format!("{name}.bias")) {
            Some(bias) => x.broadcast_add(&bias.reshape((1, (), 1))?),
            None => Ok(x),
        }
    }

    pub fn layer_norm(&self, x: &Tensor, name: &str, eps: f64) -> Result<Tensor> {
        affine_norm(
            x,
            self.get(&format!("{name}.weight"))?,
            self.get(&format!("{name}.bias"))?,
            eps,
        )
    }

    pub fn adain(&self, x: &Tensor, style: &Tensor, name: &str) -> Result<Tensor> {
        if x.device().is_cpu()
            && style.device().is_cpu()
            && x.dtype() == DType::F32
            && style.dtype() == DType::F32
            && style.rank() == 2
        {
            let (batch, channels, _) = x.dims3()?;
            let params = self.linear(style, &format!("{name}.fc"))?;
            let (styles, width) = params.dims2()?;
            if width != channels * 2 || (styles != 1 && styles != batch) {
                candle_core::bail!("invalid adaptive normalization style shape");
            }
            let params = params.flatten_all()?.to_vec1::<f32>()?;
            let values = self.instance_norm_values(x, name, Some(&params))?;
            return Tensor::from_vec(values, x.shape(), x.device());
        }
        let normalized = self.instance_norm(x, name)?;
        let params = self.linear(style, &format!("{name}.fc"))?.unsqueeze(2)?;
        let channels = x.dim(1)?;
        normalized
            .broadcast_mul(&(params.narrow(1, 0, channels)? + 1.0)?)?
            .broadcast_add(&params.narrow(1, channels, channels)?)
    }

    fn instance_norm(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        if x.device().is_cpu() {
            return Tensor::from_vec(
                self.instance_norm_values(x, name, None)?,
                x.shape(),
                x.device(),
            );
        }
        let weight = self.get(&format!("{name}.norm.weight"))?;
        let bias = self.get(&format!("{name}.norm.bias"))?;
        normalize(x, 2, 1e-5)?
            .broadcast_mul(&weight.reshape((1, (), 1))?)?
            .broadcast_add(&bias.reshape((1, (), 1))?)
    }

    fn instance_norm_values(
        &self,
        x: &Tensor,
        name: &str,
        style: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let weight = self.get(&format!("{name}.norm.weight"))?;
        let bias = self.get(&format!("{name}.norm.bias"))?;
        // Match PyTorch CPU batch_norm's contiguous InstanceNorm path: double
        // statistic sums, float mean/variance storage, and affine transform.
        let (_, channels, steps) = x.dims3()?;
        if weight.dims() != [channels] || bias.dims() != [channels] {
            candle_core::bail!("invalid instance normalization shape");
        }
        let mut values = x.flatten_all()?.to_vec1::<f32>()?;
        let weight = weight.to_vec1::<f32>()?;
        let bias = bias.to_vec1::<f32>()?;
        instance_norm_rows(&mut values, steps, &weight, &bias, style);
        Ok(values)
    }

    /// Single unpadded segment, PyTorch's i/f/g/o gate order, both directions.
    pub fn bilstm(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let (batch, steps, _) = x.dims3()?;
        let mut directions = Vec::with_capacity(2);
        for suffix in ["", "_reverse"] {
            let ih = self.get(&format!("{name}.weight_ih_l0{suffix}"))?;
            let hh = self.get(&format!("{name}.weight_hh_l0{suffix}"))?;
            let bi = self.get(&format!("{name}.bias_ih_l0{suffix}"))?;
            let bh = self.get(&format!("{name}.bias_hh_l0{suffix}"))?;
            let hidden = hh.dim(1)?;
            let input = linear(x, ih, Some(bi))?;
            let mut h = Tensor::zeros((batch, hidden), DType::F32, x.device())?;
            let mut c = h.clone();
            let mut states = Vec::with_capacity(steps);
            for step in 0..steps {
                let t = if suffix.is_empty() {
                    step
                } else {
                    steps - 1 - step
                };
                let recurrent = linear(&h, hh, Some(bh))?;
                let gates = (input.i((.., t, ..))? + recurrent)?.chunk(4, 1)?;
                let i = candle_nn::ops::sigmoid(&gates[0])?;
                let f = candle_nn::ops::sigmoid(&gates[1])?;
                let g = gates[2].tanh()?;
                let o = candle_nn::ops::sigmoid(&gates[3])?;
                c = ((f * &c)? + (i * g)?)?;
                h = (o * c.tanh()?)?;
                states.push(h.clone());
            }
            if !suffix.is_empty() {
                states.reverse();
            }
            directions.push(Tensor::stack(&states, 1)?);
        }
        Tensor::cat(&directions, 2)
    }
}

fn instance_norm_rows(
    values: &mut [f32],
    steps: usize,
    weight: &[f32],
    bias: &[f32],
    style: Option<&[f32]>,
) {
    if values.is_empty() {
        return;
    }
    let apply = |(index, row): (usize, &mut [f32])| {
        let channel = index % weight.len();
        let mean = (row.iter().map(|&v| f64::from(v)).sum::<f64>() / steps as f64) as f32;
        let variance_sum = row
            .iter()
            .map(|&v| {
                let d = v - mean;
                f64::from(d * d)
            })
            .sum::<f64>() as f32;
        let inv_std = (1.0 / (f64::from(variance_sum / steps as f32) + 1e-5).sqrt()) as f32;
        let alpha = inv_std * weight[channel];
        let beta = (-mean).mul_add(alpha, bias[channel]);
        instance_affine(row, alpha, beta, index, weight.len(), style);
    };
    // Interleave independent accumulators to expose instruction-level
    // parallelism. Each lane still visits its own row strictly left-to-right.
    // This does not split a reduction or combine partial sums.
    if values.len() >= 64 * 1024 && weight.len() > 1 {
        values
            .par_chunks_mut(steps * 4)
            .enumerate()
            .for_each(|(group, rows)| {
                if rows.len() != steps * 4 {
                    for (lane, row) in rows.chunks_mut(steps).enumerate() {
                        apply((group * 4 + lane, row));
                    }
                    return;
                }
                let mut sum = [0.0f64; 4];
                for t in 0..steps {
                    for lane in 0..4 {
                        sum[lane] += f64::from(rows[lane * steps + t]);
                    }
                }
                let mean = sum.map(|v| (v / steps as f64) as f32);
                let mut variance_sum = [0.0f64; 4];
                for t in 0..steps {
                    for lane in 0..4 {
                        let d = rows[lane * steps + t] - mean[lane];
                        variance_sum[lane] += f64::from(d * d);
                    }
                }
                for (lane, row) in rows.chunks_mut(steps).enumerate() {
                    let index = group * 4 + lane;
                    let channel = index % weight.len();
                    let variance = variance_sum[lane] as f32 / steps as f32;
                    let inv_std = (1.0 / (f64::from(variance) + 1e-5).sqrt()) as f32;
                    let alpha = inv_std * weight[channel];
                    let beta = (-mean[lane]).mul_add(alpha, bias[channel]);
                    instance_affine(row, alpha, beta, index, weight.len(), style);
                }
            });
    } else {
        values.chunks_mut(steps).enumerate().for_each(apply);
    }
}

fn instance_affine(
    row: &mut [f32],
    alpha: f32,
    beta: f32,
    index: usize,
    channels: usize,
    style: Option<&[f32]>,
) {
    let channel = index % channels;
    if let Some(params) = style {
        let batch = if params.len() == channels * 2 {
            0
        } else {
            index / channels
        };
        let gamma = params[batch * channels * 2 + channel] + 1.0;
        let style_bias = params[batch * channels * 2 + channels + channel];
        for value in row {
            // Fuse traversal, not coefficients: each operation retains its
            // original F32 rounding after the learned normalization affine.
            let normalized = *value * alpha + beta;
            let scaled = normalized * gamma;
            *value = scaled + style_bias;
        }
    } else {
        for value in row {
            *value = *value * alpha + beta;
        }
    }
}

pub(crate) fn normalize(x: &Tensor, dim: usize, eps: f64) -> Result<Tensor> {
    if x.device().is_cpu() && dim == x.rank() - 1 {
        let width = x.dim(dim)?;
        let mut values = x.flatten_all()?.to_vec1::<f32>()?;
        for row in values.chunks_mut(width) {
            let (mean, variance) = crate::moments::row(row);
            let inv = 1.0 / (variance + eps as f32).sqrt();
            for value in row {
                *value = (*value - mean) * inv;
            }
        }
        return Tensor::from_vec(values, x.shape(), x.device());
    }
    let centered = x.broadcast_sub(&x.mean_keepdim(dim)?)?;
    centered.broadcast_div(&(centered.sqr()?.mean_keepdim(dim)? + eps)?.sqrt()?)
}

pub(crate) fn leaky(x: &Tensor, slope: f64) -> Result<Tensor> {
    x.maximum(&(x * slope)?)
}

pub(crate) fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    if x.device().is_cpu()
        && alpha.device().is_cpu()
        && x.dtype() == DType::F32
        && alpha.dtype() == DType::F32
        && let Ok((batch, channels, steps)) = x.dims3()
        && batch > 0
        && channels > 0
        && steps > 0
        && alpha.dims() == [1, channels, 1]
        && x.is_contiguous()
    {
        use candle_core::Storage;
        use candle_core::op::{Sin, UnaryOpT};
        let (storage, layout) = x.storage_and_layout();
        let Storage::Cpu(cpu) = &*storage else {
            candle_core::bail!("Snake requires CPU storage");
        };
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("Snake requires contiguous CPU input".into()))?;
        let values = &cpu.as_slice::<f32>()?[start..end];
        let alpha = alpha.flatten_all()?.to_vec1::<f32>()?;
        let mut scaled = vec![0.0; values.len()];
        let multiply = |(index, row): (usize, &mut [f32])| {
            let a = alpha[index % channels];
            for (value, &input) in row
                .iter_mut()
                .zip(&values[index * steps..(index + 1) * steps])
            {
                *value = input * a;
            }
        };
        if values.len() >= 64 * 1024 {
            scaled.par_chunks_mut(steps).enumerate().for_each(multiply);
        } else {
            scaled.chunks_mut(steps).enumerate().for_each(multiply);
        }
        let mut sine = vec![0.0; values.len()];
        // Tensor::sin uses this same vector kernel on the contiguous product.
        // Scalar sin can differ from Accelerate even when the formula matches.
        // Keep the vector primitive and aligned groups. Parallelize only large
        // arrays; complete-output tests cover short final chunks as well.
        if scaled.len() >= 4 * 64 * 1024 {
            sine.par_chunks_mut(64 * 1024)
                .zip(scaled.par_chunks(64 * 1024))
                .for_each(|(output, input)| Sin::f32_vec(input, output));
        } else {
            Sin::f32_vec(&scaled, &mut sine);
        }
        let finish = |(index, row): (usize, &mut [f32])| {
            let a = alpha[index % channels];
            for ((value, &input), &s) in row
                .iter_mut()
                .zip(&values[index * steps..(index + 1) * steps])
                .zip(&sine[index * steps..(index + 1) * steps])
            {
                let square = s * s;
                let correction = square / a;
                *value = input + correction;
            }
        };
        if values.len() >= 64 * 1024 {
            scaled.par_chunks_mut(steps).enumerate().for_each(finish);
        } else {
            scaled.chunks_mut(steps).enumerate().for_each(finish);
        }
        return Tensor::from_vec(scaled, x.shape(), x.device());
    }
    x + x.broadcast_mul(alpha)?.sin()?.sqr()?.broadcast_div(alpha)?
}

pub(crate) fn nearest_double(x: &Tensor) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    x.unsqueeze(3)?
        .broadcast_as((b, c, t, 2))?
        .contiguous()?
        .reshape((b, c, t * 2))
}

pub(crate) fn affine_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    if !x.device().is_cpu() {
        return candle_nn::LayerNorm::new(weight.clone(), bias.clone(), eps).forward(x);
    }
    let width = x.dim(x.rank() - 1)?;
    let mut values = x.flatten_all()?.to_vec1::<f32>()?;
    let weight = weight.to_vec1::<f32>()?;
    let bias = bias.to_vec1::<f32>()?;
    if width == 0 || weight.len() != width || bias.len() != width {
        candle_core::bail!("invalid affine normalization shape");
    }
    for row in values.chunks_mut(width) {
        let (mean, variance) = crate::moments::row(row);
        let inv = 1.0 / (variance + eps as f32).sqrt();
        for (i, value) in row.iter_mut().enumerate() {
            *value = ((*value - mean) * inv) * weight[i] + bias[i];
        }
    }
    Tensor::from_vec(values, x.shape(), x.device())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_affine_matches_separate_tensor_ops_for_shared_and_batched_styles() -> Result<()> {
        let device = &candle_core::Device::Cpu;
        for (batch, channels, steps, styles) in [(1, 3, 13, 1), (2, 3, 17, 1), (2, 8, 8193, 2)] {
            let x = Tensor::from_vec(
                (0..batch * channels * steps)
                    .map(|i| ((i * 17 % 101) as f32 - 50.0) / 37.0)
                    .collect::<Vec<_>>(),
                (batch, channels, steps),
                device,
            )?;
            let style = Tensor::from_vec(
                (0..styles * 4).map(|i| i as f32 / 13.0).collect::<Vec<_>>(),
                (styles, 4),
                device,
            )?;
            let weights = Weights::new(HashMap::from([
                (
                    "x.norm.weight".into(),
                    Tensor::from_vec(
                        (0..channels)
                            .map(|i| 0.7 + i as f32 / 19.0)
                            .collect::<Vec<_>>(),
                        channels,
                        device,
                    )?,
                ),
                (
                    "x.norm.bias".into(),
                    Tensor::from_vec(
                        (0..channels)
                            .map(|i| -0.3 + i as f32 / 23.0)
                            .collect::<Vec<_>>(),
                        channels,
                        device,
                    )?,
                ),
                (
                    "x.fc.weight".into(),
                    Tensor::from_vec(
                        (0..channels * 2 * 4)
                            .map(|i| ((i * 13 % 103) as f32 - 51.0) / 41.0)
                            .collect::<Vec<_>>(),
                        (channels * 2, 4),
                        device,
                    )?,
                ),
                (
                    "x.fc.bias".into(),
                    Tensor::from_vec(
                        (0..channels * 2)
                            .map(|i| i as f32 / 29.0)
                            .collect::<Vec<_>>(),
                        channels * 2,
                        device,
                    )?,
                ),
            ]));
            let normalized = weights.instance_norm(&x, "x")?;
            let params = weights.linear(&style, "x.fc")?.unsqueeze(2)?;
            let expected = normalized
                .broadcast_mul(&(params.narrow(1, 0, channels)? + 1.0)?)?
                .broadcast_add(&params.narrow(1, channels, channels)?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = weights
                .adain(&x, &style, "x")?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        Ok(())
    }

    #[test]
    fn snake_matches_tensor_chain_bitwise_for_broadcast_and_strided_inputs() -> Result<()> {
        for (batch, channels, steps) in [
            (1, 1, 1),
            (2, 3, 13),
            (1, 16, 513),
            (2, 8, 8193),
            (1, 1, 262144),
            (1, 1, 262145),
            (2, 8, 32769),
        ] {
            let values: Vec<_> = (0..batch * channels * steps)
                .map(|i| ((i * 17 % 503) as f32 - 251.0) / 37.0)
                .collect();
            let alpha: Vec<_> = (0..channels).map(|i| 0.1 + i as f32 / 7.0).collect();
            let x = Tensor::from_vec(values, (batch, channels, steps), &candle_core::Device::Cpu)?;
            let alpha = Tensor::from_vec(alpha, (1, channels, 1), x.device())?;
            let expected = (&x
                + x.broadcast_mul(&alpha)?
                    .sin()?
                    .sqr()?
                    .broadcast_div(&alpha)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            for input in [&x, &x.transpose(1, 2)?.contiguous()?.transpose(1, 2)?] {
                let actual = snake(input, &alpha)?.flatten_all()?.to_vec1::<f32>()?;
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
        Ok(())
    }

    #[test]
    fn independent_rows_keep_original_reductions_across_thread_counts() {
        for channels in [1, 3, 8] {
            for steps in [1, 17, 1024, 16385] {
                let input: Vec<_> = (0..2 * channels * steps)
                    .map(|i| ((i * 13 % 251) as f32 - 125.0) / 73.0)
                    .collect();
                let weight: Vec<_> = (0..channels).map(|c| 0.7 + c as f32 / 13.0).collect();
                let bias: Vec<_> = (0..channels).map(|c| -0.3 + c as f32 / 19.0).collect();
                let mut expected = input.clone();
                for b in 0..2 {
                    for c in 0..channels {
                        let row = &mut expected
                            [(b * channels + c) * steps..(b * channels + c + 1) * steps];
                        let mean =
                            (row.iter().map(|&v| f64::from(v)).sum::<f64>() / steps as f64) as f32;
                        let variance_sum = row
                            .iter()
                            .map(|&v| {
                                let d = v - mean;
                                f64::from(d * d)
                            })
                            .sum::<f64>() as f32;
                        let inv_std =
                            (1.0 / (f64::from(variance_sum / steps as f32) + 1e-5).sqrt()) as f32;
                        let alpha = inv_std * weight[c];
                        let beta = (-mean).mul_add(alpha, bias[c]);
                        for value in row {
                            *value = *value * alpha + beta;
                        }
                    }
                }
                for threads in [1, 4] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    let mut actual = input.clone();
                    pool.install(|| instance_norm_rows(&mut actual, steps, &weight, &bias, None));
                    assert!(
                        actual
                            .iter()
                            .zip(&expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
            }
        }
    }

    #[test]
    fn instance_affine_matches_independent_pytorch_rows() -> Result<()> {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/moments.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let floats = |name: &str| -> Vec<f32> {
                case[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            };
            let input = floats("input");
            let expected = floats("instance");
            let device = &candle_core::Device::Cpu;
            let weights = Weights::new(HashMap::from([
                (
                    "x.norm.weight".into(),
                    Tensor::from_slice(&[0.7f32], 1, device)?,
                ),
                (
                    "x.norm.bias".into(),
                    Tensor::from_slice(&[-0.3f32], 1, device)?,
                ),
            ]));
            let tensor = Tensor::from_slice(&input, (1, 1, input.len()), device)?;
            let actual = weights
                .instance_norm(&tensor, "x")?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(max < 1e-6, "instance normalization error {max}");
        }
        Ok(())
    }
}
