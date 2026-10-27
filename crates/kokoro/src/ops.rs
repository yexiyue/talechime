//! PyTorch-layout inference operations used by the official Kokoro architecture.
use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::Module;
use std::collections::HashMap;

pub(crate) struct Weights(pub HashMap<String, Tensor>);

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
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.0
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing Kokoro tensor: {name}")))
    }

    pub fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        linear(
            x,
            self.get(&format!("{name}.weight"))?,
            self.0.get(&format!("{name}.bias")),
        )
    }

    pub fn conv(&self, x: &Tensor, name: &str, stride: usize, dilation: usize) -> Result<Tensor> {
        let w = self.get(&format!("{name}.weight"))?;
        #[cfg(target_os = "macos")]
        if x.device().is_cpu()
            && let Some(bias) = self.0.get(&format!("{name}.bias"))
        {
            return crate::blas::conv(x, w, bias, stride, dilation);
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
        let y = x
            .contiguous()?
            .conv_transpose1d(w, padding, output_padding, stride, 1, groups)?;
        self.bias_conv(y, name)
    }

    fn bias_conv(&self, x: Tensor, name: &str) -> Result<Tensor> {
        match self.0.get(&format!("{name}.bias")) {
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
        let normalized = self.instance_norm(x, name)?;
        let params = self.linear(style, &format!("{name}.fc"))?.unsqueeze(2)?;
        let channels = x.dim(1)?;
        normalized
            .broadcast_mul(&(params.narrow(1, 0, channels)? + 1.0)?)?
            .broadcast_add(&params.narrow(1, channels, channels)?)
    }

    fn instance_norm(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let weight = self.get(&format!("{name}.norm.weight"))?;
        let bias = self.get(&format!("{name}.norm.bias"))?;
        if !x.device().is_cpu() {
            return normalize(x, 2, 1e-5)?
                .broadcast_mul(&weight.reshape((1, (), 1))?)?
                .broadcast_add(&bias.reshape((1, (), 1))?);
        }
        // Match PyTorch CPU batch_norm's contiguous InstanceNorm path: double
        // statistic sums, float mean/variance storage, and affine transform.
        let (batch, channels, steps) = x.dims3()?;
        let mut values = x.flatten_all()?.to_vec1::<f32>()?;
        let weight = weight.to_vec1::<f32>()?;
        let bias = bias.to_vec1::<f32>()?;
        for b in 0..batch {
            for c in 0..channels {
                let row = &mut values[(b * channels + c) * steps..(b * channels + c + 1) * steps];
                let mean = (row.iter().map(|&v| f64::from(v)).sum::<f64>() / steps as f64) as f32;
                let variance_sum = row
                    .iter()
                    .map(|&v| {
                        let d = v - mean;
                        f64::from(d * d)
                    })
                    .sum::<f64>() as f32;
                let inv_std = (1.0 / (f64::from(variance_sum / steps as f32) + 1e-5).sqrt()) as f32;
                let alpha = inv_std * weight[c];
                let beta = (-mean).mul_add(alpha, bias[c]);
                for value in row {
                    // PyTorch's NEON vector kernel performs multiply and add
                    // separately; fusing this affine transform changes F0.
                    *value = *value * alpha + beta;
                }
            }
        }
        Tensor::from_vec(values, (batch, channels, steps), x.device())
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
            let weights = Weights(HashMap::from([
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
