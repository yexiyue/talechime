//! Causal 1D convolutions from `cosyvoice/transformer/convolution.py`.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder, conv1d as nn_conv1d};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalType {
    /// Cache on the left (history only).
    Left,
    /// Cache on the right (look-ahead).
    Right,
}

/// `CausalConv1d`: stride-1 conv whose receptive padding comes entirely from
/// one side. `causal_padding = ((k*d - d)/2)*2 + (k+1)%2`.
#[derive(Debug, Clone)]
pub struct CausalConv1d {
    conv: Conv1d,
    causal_type: CausalType,
    causal_padding: usize,
}

impl CausalConv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        causal_type: CausalType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let causal_padding = ((kernel_size * dilation - dilation) / 2) * 2 + (kernel_size + 1) % 2;
        Ok(Self {
            conv: nn_conv1d(in_channels, out_channels, kernel_size, cfg, vb)?,
            causal_type,
            causal_padding,
        })
    }

    pub fn causal_padding(&self) -> usize {
        self.causal_padding
    }

    /// Streaming form: prepend/append an explicit cache instead of zeros.
    pub fn forward_with_cache(&self, x: &Tensor, cache: &Tensor) -> Result<Tensor> {
        debug_assert_eq!(cache.dim(2)?, self.causal_padding);
        let x = match self.causal_type {
            CausalType::Left => Tensor::cat(&[cache, x], 2)?,
            CausalType::Right => Tensor::cat(&[&x.clone(), cache], 2)?,
        };
        self.conv.forward(&x)
    }
}

impl Module for CausalConv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.causal_padding == 0 {
            return self.conv.forward(x);
        }
        let x = match self.causal_type {
            CausalType::Left => x.pad_with_zeros(2, self.causal_padding, 0)?,
            CausalType::Right => x.pad_with_zeros(2, 0, self.causal_padding)?,
        };
        self.conv.forward(&x)
    }
}

/// `CausalConv1dUpsample`: nearest upsample, left pad `k-1`, stride-1 conv.
#[derive(Debug)]
pub struct CausalConv1dUpsample {
    conv: Conv1d,
    scale: usize,
    kernel_size: usize,
}

impl CausalConv1dUpsample {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        scale: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        Ok(Self {
            conv: nn_conv1d(in_ch, out_ch, kernel_size, cfg, vb)?,
            scale,
            kernel_size,
        })
    }
}

impl Module for CausalConv1dUpsample {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = repeat_interleave_dim(x, 2, self.scale)?;
        let x = x.pad_with_zeros(2, self.kernel_size - 1, 0)?;
        self.conv.forward(&x)
    }
}

/// Repeat each slice along `dim` `repeats` times (nearest-neighbor
/// upsample), materializing one contiguous output: `unsqueeze ->
/// broadcast -> reshape`.
pub(crate) fn repeat_interleave_dim(x: &Tensor, dim: usize, repeats: usize) -> Result<Tensor> {
    let mut broadcast: Vec<usize> = x.dims().to_vec();
    broadcast.insert(dim + 1, repeats);
    let mut out = broadcast.clone();
    out[dim] *= repeats;
    out.remove(dim + 1);
    x.unsqueeze(dim + 1)?.broadcast_as(broadcast)?.reshape(out)
}

/// `ReflectionPad1d` via index gather (torch semantics: no edge repeat).
pub fn reflection_pad_1d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let t = x.dim(2)?;
    if t < 2 || t <= left || t <= right {
        candle_core::bail!("reflection_pad_1d: input too short ({t}) for pads ({left}, {right})");
    }
    let mut idx: Vec<u32> = Vec::with_capacity(left + t + right);
    for i in 0..left {
        idx.push((left - i) as u32);
    }
    for i in 0..t {
        idx.push(i as u32);
    }
    for i in 0..right {
        idx.push((t - 2 - i) as u32);
    }
    let idx = Tensor::from_vec(idx, left + t + right, x.device())?;
    x.index_select(&idx, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn reflection_pad_matches_torch() -> Result<()> {
        let device = Device::Cpu;
        // torch: F.pad([1,2,3,4], (2,0), 'reflect') == [3, 2, 1, 2, 3, 4]
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device)?;
        let y: Vec<f32> = reflection_pad_1d(&x, 2, 0)?.flatten_all()?.to_vec1()?;
        assert_eq!(y, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0]);
        // right pad by 1 mirrors t-2
        let y: Vec<f32> = reflection_pad_1d(&x, 0, 1)?.flatten_all()?.to_vec1()?;
        assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0, 3.0]);
        let _ = DType::F32;
        Ok(())
    }
}
