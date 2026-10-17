//! Pre-lookahead layer (`cosyvoice/transformer/upsample_encoder.py`).
//!
//! Token-rate conv with `pre_lookahead_len` future context (right padding
//! or an explicit streaming context), a causal kernel-3 conv, residual, and
//! `token_mel_ratio` repeat upsampling to mel rate.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

use crate::activations::LeakyReLU;

#[derive(Debug)]
pub struct PreLookaheadLayer {
    conv1: Conv1d,
    conv2: Conv1d,
    lrelu: LeakyReLU,
    pre_lookahead_len: usize,
    token_mel_ratio: usize,
}

impl PreLookaheadLayer {
    /// `in_dim` (80) -> `hidden` (1024) -> `out_dim` (80).
    pub fn new(
        in_dim: usize,
        hidden: usize,
        out_dim: usize,
        pre_lookahead_len: usize,
        token_mel_ratio: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let no_pad = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        Ok(Self {
            conv1: candle_nn::conv1d(
                in_dim,
                hidden,
                pre_lookahead_len + 1,
                no_pad,
                vb.pp("conv1"),
            )?,
            conv2: candle_nn::conv1d(hidden, out_dim, 3, no_pad, vb.pp("conv2"))?,
            lrelu: LeakyReLU::new(0.01),
            pre_lookahead_len,
            token_mel_ratio,
        })
    }

    pub fn pre_lookahead_len(&self) -> usize {
        self.pre_lookahead_len
    }

    pub fn token_mel_ratio(&self) -> usize {
        self.token_mel_ratio
    }

    /// `x`: `[B, T, in_dim]` token embeddings; `context`: optional
    /// `[B, L, in_dim]` trailing tokens used as look-ahead during streamed
    /// (non-finalize) calls. Returns `[B, T * ratio, out_dim]`.
    pub fn forward(&self, x: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let input = x.clone();
        let xt = x.transpose(1, 2)?;

        let padded = match context {
            Some(ctx) => Tensor::cat(&[&xt, &ctx.transpose(1, 2)?], 2)?,
            None => {
                let (b, c, _) = xt.dims3()?;
                let pad = Tensor::zeros((b, c, self.pre_lookahead_len), xt.dtype(), xt.device())?;
                Tensor::cat(&[&xt, &pad], 2)?
            }
        };
        let h = self.lrelu.forward(&self.conv1.forward(&padded)?)?;

        let (b, c, _) = h.dims3()?;
        let pad = Tensor::zeros((b, c, 2), h.dtype(), h.device())?;
        let h = self.conv2.forward(&Tensor::cat(&[&pad, &h], 2)?)?;

        let out = (h.transpose(1, 2)? + input)?;
        repeat_interleave_time(&out, self.token_mel_ratio)
    }
}

/// `tensor.repeat_interleave(repeats, dim=1)` for `[B, T, C]`.
pub fn repeat_interleave_time(x: &Tensor, repeats: usize) -> Result<Tensor> {
    let (b, t, c) = x.dims3()?;
    let x = x.unsqueeze(2)?.broadcast_as((b, t, repeats, c))?;
    x.reshape((b, t * repeats, c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn repeat_interleave_doubles_each_frame() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 3, 2), &device)?;
        let y: Vec<f32> = repeat_interleave_time(&x, 2)?.flatten_all()?.to_vec1()?;
        assert_eq!(
            y,
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0]
        );
        let _ = DType::F32;
        Ok(())
    }
}
