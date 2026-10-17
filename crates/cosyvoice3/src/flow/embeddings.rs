//! DiT embedding layers (time, modulation, input, rotary).
//!
//! Math mirrors `cosyvoice/flow/DiT/modules.py` plus the exact RoPE
//! convention of x-transformers 2.11.24 (`partial rotary` over the first
//! `rot_dim` channels with interleaved frequency duplication).

use candle_core::{D, DType, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder};

use crate::activations::{Mish, Swish};

/// A `Linear` applied as an explicit contiguous 2-D matmul over `(B, T, C)`
/// inputs, with the transposed weight materialized once at load time.
/// Avoids `.t()?.contiguous()?` per call across the 22 DiT blocks × 10 CFM steps.
#[derive(Debug, Clone)]
pub struct LinT2d {
    weight_t: Tensor,
    bias: Option<Tensor>,
    out_dim: usize,
}

impl LinT2d {
    pub fn from_linear(l: Linear) -> Result<Self> {
        let out_dim = l.weight().dim(0)?;
        Ok(Self {
            weight_t: l.weight().t()?.contiguous()?,
            bias: l.bias().cloned(),
            out_dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, f) = x.dims3()?;
        let mut y = x
            .contiguous()?
            .reshape((b * t, f))?
            .matmul(&self.weight_t)?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(bias)?;
        }
        y.reshape((b, t, self.out_dim))
    }
}

/// Sinusoidal timestep embedding (freq dim 256, scale 1000) + SiLU MLP.
#[derive(Debug)]
pub struct TimestepEmbedding {
    linear1: Linear,
    linear2: Linear,
    swish: Swish,
    inv_freqs: Tensor,
    dtype: candle_core::DType,
}

impl TimestepEmbedding {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let freq_dim = 256;
        let half = freq_dim / 2;
        let log_div = 10_000f64.ln() / (half as f64 - 1.0);
        let inv: Vec<f32> = (0..half)
            .map(|i| (-(i as f32) * log_div as f32).exp())
            .collect();
        let inv_freqs = Tensor::from_vec(inv, (1, half), vb.device())?;
        Ok(Self {
            linear1: candle_nn::linear(freq_dim, dim, vb.pp("time_mlp.0"))?,
            linear2: candle_nn::linear(dim, dim, vb.pp("time_mlp.2"))?,
            swish: Swish,
            inv_freqs,
            dtype: vb.dtype(),
        })
    }

    fn sinusoidal(&self, t: &Tensor) -> Result<Tensor> {
        let inv = if self.inv_freqs.device().same_device(t.device()) {
            self.inv_freqs.clone()
        } else {
            self.inv_freqs.to_device(t.device())?
        };
        let t = t.to_dtype(DType::F32)?.unsqueeze(1)?.affine(1000.0, 0.0)?;
        let freqs = t.broadcast_mul(&inv)?;
        let emb = Tensor::cat(&[&freqs.sin()?, &freqs.cos()?], D::Minus1)?;
        emb.to_dtype(self.dtype)
    }
}

impl Module for TimestepEmbedding {
    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let emb = self.sinusoidal(t)?;
        let emb = self.linear1.forward(&emb)?;
        let emb = self.swish.forward(&emb)?;
        self.linear2.forward(&emb)
    }
}

/// LayerNorm without affine parameters (elementwise_affine=False, eps 1e-6).
pub fn layer_norm_no_affine(x: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let std = (var + 1e-6)?.sqrt()?;
    xc.broadcast_div(&std)
}

/// AdaLayerNormZero: 6-way modulation from the timestep embedding.
#[derive(Debug)]
pub struct AdaLayerNormZero {
    linear: Linear,
    swish: Swish,
    dim: usize,
}

impl AdaLayerNormZero {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(dim, dim * 6, vb.pp("linear"))?,
            swish: Swish,
            dim,
        })
    }

    /// Returns `(modulated_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)` with
    /// each modulation tensor shaped `[B, D]`.
    pub fn forward(
        &self,
        x: &Tensor,
        emb: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let emb = self.linear.forward(&self.swish.forward(emb)?)?;
        let d = self.dim;
        let shift_msa = emb.narrow(D::Minus1, 0, d)?;
        let scale_msa = emb.narrow(D::Minus1, d, d)?;
        let gate_msa = emb.narrow(D::Minus1, 2 * d, d)?;
        let shift_mlp = emb.narrow(D::Minus1, 3 * d, d)?;
        let scale_mlp = emb.narrow(D::Minus1, 4 * d, d)?;
        let gate_mlp = emb.narrow(D::Minus1, 5 * d, d)?;

        let x = layer_norm_no_affine(x)?;
        let x = x.broadcast_mul(&(scale_msa.unsqueeze(1)? + 1.0)?)?;
        let x = x.broadcast_add(&shift_msa.unsqueeze(1)?)?;
        Ok((x, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }

    /// Hot-path variant taking an already Swish-activated `emb`, returning
    /// modulation tensors shaped `[B, 1, D]` ready for broadcast.
    pub(crate) fn forward_act(
        &self,
        x: &Tensor,
        act_emb: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let emb = self.linear.forward(act_emb)?.unsqueeze(1)?;
        let d = self.dim;
        let shift_msa = emb.narrow(D::Minus1, 0, d)?;
        let scale_msa = emb.narrow(D::Minus1, d, d)?;
        let gate_msa = emb.narrow(D::Minus1, 2 * d, d)?;
        let shift_mlp = emb.narrow(D::Minus1, 3 * d, d)?;
        let scale_mlp = emb.narrow(D::Minus1, 4 * d, d)?;
        let gate_mlp = emb.narrow(D::Minus1, 5 * d, d)?;

        let x = layer_norm_no_affine(x)?;
        let x = x.broadcast_mul(&(scale_msa + 1.0)?)?;
        let x = x.broadcast_add(&shift_msa)?;
        Ok((x, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }
}

/// AdaLayerNormZero for the final projection (scale/shift only).
#[derive(Debug)]
pub struct AdaLayerNormZeroFinal {
    linear: Linear,
    swish: Swish,
    dim: usize,
}

impl AdaLayerNormZeroFinal {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: candle_nn::linear(dim, dim * 2, vb.pp("linear"))?,
            swish: Swish,
            dim,
        })
    }

    pub fn forward(&self, x: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let act = self.swish.forward(emb)?;
        self.forward_act(x, &act)
    }

    pub(crate) fn forward_act(&self, x: &Tensor, act_emb: &Tensor) -> Result<Tensor> {
        let emb = self.linear.forward(act_emb)?.unsqueeze(1)?;
        let d = self.dim;
        let scale = emb.narrow(D::Minus1, 0, d)?;
        let shift = emb.narrow(D::Minus1, d, d)?;
        let x = layer_norm_no_affine(x)?;
        x.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)
    }
}

/// Causal conv position embedding (kernel 31, groups 16, Mish), residual.
#[derive(Debug)]
pub struct CausalConvPositionEmbedding {
    conv1: Conv1d,
    conv2: Conv1d,
    kernel_size: usize,
    mish: Mish,
}

impl CausalConvPositionEmbedding {
    pub fn new(dim: usize, kernel_size: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups,
            cudnn_fwd_algo: None,
        };
        // nn.Sequential indices: conv1.0 / conv2.0.
        Ok(Self {
            conv1: candle_nn::conv1d(dim, dim, kernel_size, cfg, vb.pp("conv1.0"))?,
            conv2: candle_nn::conv1d(dim, dim, kernel_size, cfg, vb.pp("conv2.0"))?,
            kernel_size,
            mish: Mish,
        })
    }
}

impl Module for CausalConvPositionEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xt = x.transpose(1, 2)?;
        let xt = xt.pad_with_zeros(2, self.kernel_size - 1, 0)?;
        let xt = self.mish.forward(&self.conv1.forward(&xt)?)?;
        let xt = xt.pad_with_zeros(2, self.kernel_size - 1, 0)?;
        let xt = self.conv2.forward(&xt)?;
        self.mish.forward(&xt.transpose(1, 2)?)
    }
}

/// Concatenates `[x, cond, mu, spks]`, projects, and adds causal conv
/// position information (residual).
#[derive(Debug)]
pub struct InputEmbedding {
    proj: LinT2d,
    conv_pos_embed: CausalConvPositionEmbedding,
    spk_dim: usize,
}

impl InputEmbedding {
    pub fn new(mel_dim: usize, spk_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let proj = LinT2d::from_linear(candle_nn::linear(
            mel_dim * 3 + spk_dim,
            out_dim,
            vb.pp("proj"),
        )?)?;
        Ok(Self {
            proj,
            conv_pos_embed: CausalConvPositionEmbedding::new(
                out_dim,
                31,
                16,
                vb.pp("conv_pos_embed"),
            )?,
            spk_dim,
        })
    }

    /// `x`, `cond`, `mu`: `[B, T, mel]`; `spks`: `[B, spk]`.
    pub fn forward(&self, x: &Tensor, cond: &Tensor, mu: &Tensor, spks: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let spks = spks.unsqueeze(1)?.broadcast_as((b, t, self.spk_dim))?;
        let combined = Tensor::cat(&[x, cond, mu, &spks], D::Minus1)?;
        self.forward_combined(&combined)
    }

    /// Project an already concatenated `[x, cond, mu, spks]` tensor `[B, T, 3*mel + spk]`.
    pub fn forward_combined(&self, combined: &Tensor) -> Result<Tensor> {
        let x = self.proj.forward(combined)?;
        let pos = self.conv_pos_embed.forward(&x)?;
        x + pos
    }
}

/// x-transformers `RotaryEmbedding` frequency table for `dim_head`.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    dim: usize,
    theta: f64,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, theta: f64) -> Self {
        Self { dim, theta }
    }

    /// `[1, seq_len, dim]` with interleaved duplication `[f0, f0, f1, f1, ..]`.
    pub fn forward(
        &self,
        seq_len: usize,
        device: &candle_core::Device,
        dtype: candle_core::DType,
    ) -> Result<Tensor> {
        let half = self.dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0 / self.theta.powf(2.0 * i as f64 / self.dim as f64) as f32)
            .collect();
        let inv = Tensor::from_vec(inv, half, device)?;
        let t = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(candle_core::DType::F32)?;
        let freqs = t.unsqueeze(1)?.broadcast_mul(&inv.unsqueeze(0)?)?;
        let freqs = freqs.unsqueeze(2)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.reshape((seq_len, self.dim))?;
        freqs.unsqueeze(0)?.to_dtype(dtype)
    }

    /// Precompute `(cos, sin)` tables `[1, seq_len, dim]` once for an entire solve.
    pub fn forward_cos_sin(
        &self,
        seq_len: usize,
        device: &candle_core::Device,
        dtype: candle_core::DType,
    ) -> Result<(Tensor, Tensor)> {
        let freqs = self.forward(seq_len, device, dtype)?;
        Ok((freqs.cos()?, freqs.sin()?))
    }
}

/// `rotate_half` over interleaved pairs: `[a0,a1,b0,b1] -> [-a1,a0,-b1,b0]`.
pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let mut shape = x.dims().to_vec();
    let last = shape.pop().unwrap();
    let half = last / 2;
    let mut pair_shape = shape.clone();
    pair_shape.push(half);
    pair_shape.push(2);
    let x = x.reshape(pair_shape)?;
    let x1 = x.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let x2 = x.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
    let rotated = Tensor::stack(&[&x2.neg()?, &x1], D::Minus1)?;
    let mut out = shape;
    out.push(last);
    rotated.reshape(out)
}

/// Partial rotary application on `[B, T, H*D]` tensors: only the first
/// `rot_dim` channels rotate (GPT-J style), the tail passes through.
pub fn apply_rotary_pos_emb(t: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let seq = t.dim(1)?;
    let freqs = if freqs.dim(1)? > seq {
        freqs.narrow(1, freqs.dim(1)? - seq, seq)?
    } else {
        freqs.clone()
    };
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    apply_rotary_cos_sin(t, &cos, &sin)
}

/// Partial rotary application using precomputed `cos` and `sin` `[1, seq, rot_dim]`.
pub fn apply_rotary_cos_sin(t: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let rot_dim = cos.dim(D::Minus1)?;
    let seq = t.dim(1)?;
    let dtype = t.dtype();
    let (cos, sin) = if cos.dim(1)? > seq {
        let start = cos.dim(1)? - seq;
        (cos.narrow(1, start, seq)?, sin.narrow(1, start, seq)?)
    } else {
        (cos.clone(), sin.clone())
    };
    let t_rot = t.narrow(D::Minus1, 0, rot_dim)?;
    let rotated = rotate_half(&t_rot)?;
    let t_out = (t_rot.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;
    let out = if t.dim(D::Minus1)? > rot_dim {
        Tensor::cat(
            &[
                &t_out,
                &t.narrow(D::Minus1, rot_dim, t.dim(D::Minus1)? - rot_dim)?,
            ],
            D::Minus1,
        )?
    } else {
        t_out
    };
    out.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn rope_partial_rotation_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(4, 10_000.0);
        let freqs = rope.forward(2, &device, DType::F32)?; // [1, 2, 4]
        let t = Tensor::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            (1, 2, 6),
            &device,
        )?;
        // rot_dim = 4: only the first 4 of 6 channels rotate; 5-6 pass.
        let out = apply_rotary_pos_emb(&t, &freqs)?;
        assert_eq!(out.dims(), &[1, 2, 6]);
        let flat: Vec<f32> = out.flatten_all()?.to_vec1()?;
        assert!((flat[4] - 5.0).abs() < 1e-6);
        assert!((flat[5] - 6.0).abs() < 1e-6);
        assert!((flat[10] - 11.0).abs() < 1e-6);
        assert!((flat[11] - 12.0).abs() < 1e-6);
        // freqs are 0 for position 0: position 0 output equals input.
        let inp: Vec<f32> = t.flatten_all()?.to_vec1()?;
        assert!((flat[0] - inp[0]).abs() < 1e-6);
        assert!((flat[1] - inp[1]).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn rotate_half_pairs() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], 4, &device)?;
        let y: Vec<f32> = rotate_half(&x)?.to_vec1()?;
        assert_eq!(y, vec![-2.0, 1.0, -4.0, 3.0]);
        Ok(())
    }
}
