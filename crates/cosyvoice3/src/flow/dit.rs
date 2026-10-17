//! Diffusion Transformer (DiT) flow estimator.
//!
//! Mirrors `cosyvoice/flow/DiT/dit.py::DiT.forward`: `[x, cond, mu, spks]`
//! input embedding, partial RoPE, `depth` DiT blocks with AdaLN-zero
//! modulation, and a final modulated projection. Weights live under
//! `dit.*` in the converted `flow.safetensors`.

use candle_core::{D, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::embeddings::{
    AdaLayerNormZero, AdaLayerNormZeroFinal, InputEmbedding, RotaryEmbedding, TimestepEmbedding,
    apply_rotary_pos_emb, layer_norm_no_affine,
};
use crate::config::DiTConfig;

/// FeedForward with tanh-approximate GELU (`approximate="tanh"`).
#[derive(Debug)]
pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    pub fn new(dim: usize, ff_mult: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = dim * ff_mult;
        // nn.Sequential layout: ff.0.0 / ff.2.
        Ok(Self {
            linear1: candle_nn::linear(dim, hidden, vb.pp("ff.0.0"))?,
            linear2: candle_nn::linear(hidden, dim, vb.pp("ff.2"))?,
        })
    }
}

impl candle_core::Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = gelu_tanh(&x)?;
        self.linear2.forward(&x)
    }
}

/// tanh-approximate GELU: 0.5x(1+tanh(√(2/π)(x+0.044715x³))).
fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    x.gelu()
}

#[derive(Debug)]
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(dim: usize, heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = heads * head_dim;
        Ok(Self {
            to_q: candle_nn::linear(dim, inner, vb.pp("to_q"))?,
            to_k: candle_nn::linear(dim, inner, vb.pp("to_k"))?,
            to_v: candle_nn::linear(dim, inner, vb.pp("to_v"))?,
            to_out: candle_nn::linear(inner, dim, vb.pp("to_out.0"))?,
            heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// `mask` is a bool `[1, 1, T, T]` keep-mask (true = attend).
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>, rope: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = apply_rotary_pos_emb(&self.to_q.forward(x)?, rope)?;
        let k = apply_rotary_pos_emb(&self.to_k.forward(x)?, rope)?;
        let v = self.to_v.forward(x)?;

        let shape = |tt: &Tensor| {
            tt.reshape((b, t, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let (q, k, v) = (shape(&q)?, shape(&k)?, shape(&v)?);

        let mut attn = q
            .matmul(&k.transpose(D::Minus2, D::Minus1)?)?
            .affine(self.scale, 0.0)?;
        if let Some(m) = mask {
            // Build the scalar on CPU: Metal has no F64 -> F32 cast.
            let neg = Tensor::new(f64::NEG_INFINITY, &Device::Cpu)?
                .to_dtype(attn.dtype())?
                .to_device(attn.device())?;
            attn = m
                .broadcast_as(attn.shape())?
                .where_cond(&attn, &neg.broadcast_as(attn.shape())?)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .reshape((b, t, self.heads * self.head_dim))?;
        self.to_out.forward(&out)
    }
}

#[derive(Debug)]
struct DiTBlock {
    attn_norm: AdaLayerNormZero,
    attn: Attention,
    ff: FeedForward,
}

impl DiTBlock {
    fn new(cfg: &DiTConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn_norm: AdaLayerNormZero::new(cfg.dim, vb.pp("attn_norm"))?,
            attn: Attention::new(cfg.dim, cfg.heads, cfg.dim_head, vb.pp("attn"))?,
            ff: FeedForward::new(cfg.dim, cfg.ff_mult, vb.pp("ff"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        mask: Option<&Tensor>,
        rope: &Tensor,
    ) -> Result<Tensor> {
        let (norm, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.attn_norm.forward(x, t)?;
        let attn = self.attn.forward(&norm, mask, rope)?;
        let gated = gate_msa.unsqueeze(1)?.broadcast_mul(&attn)?;
        let x = (x + gated)?;
        let ff_norm = layer_norm_no_affine(&x)?;
        let ff_in = ff_norm
            .broadcast_mul(&(scale_mlp.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_mlp.unsqueeze(1)?)?;
        let ff = self.ff.forward(&ff_in)?;
        let gated = gate_mlp.unsqueeze(1)?.broadcast_mul(&ff)?;
        x + gated
    }
}

#[derive(Debug)]
pub struct DiT {
    cfg: DiTConfig,
    time_embed: TimestepEmbedding,
    input_embed: InputEmbedding,
    rotary: RotaryEmbedding,
    blocks: Vec<DiTBlock>,
    norm_out: AdaLayerNormZeroFinal,
    proj_out: Linear,
}

impl DiT {
    pub fn new(cfg: DiTConfig, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(DiTBlock::new(
                &cfg,
                vb.pp(format!("transformer_blocks.{i}")),
            )?);
        }
        Ok(Self {
            time_embed: TimestepEmbedding::new(cfg.dim, vb.pp("time_embed"))?,
            input_embed: InputEmbedding::new(
                cfg.mel_dim,
                cfg.spk_dim,
                cfg.dim,
                vb.pp("input_embed"),
            )?,
            rotary: RotaryEmbedding::new(cfg.dim_head, 10_000.0),
            blocks,
            norm_out: AdaLayerNormZeroFinal::new(cfg.dim, vb.pp("norm_out"))?,
            proj_out: candle_nn::linear(cfg.dim, cfg.out_channels, vb.pp("proj_out"))?,
            cfg,
        })
    }

    /// Estimator forward.
    ///
    /// * `x`, `mu`, `cond`: `[B, 80, T]`; `mask`: `[B, 1, T]` (padding, all
    ///   ones for single request); `t`: `[B]`; `spks`: `[B, 80]`.
    ///
    /// In streaming mode an additional static-chunk attention mask
    /// (`j < (i / chunk + 1) * chunk`) is applied, matching
    /// `add_optional_chunk_mask` with `static_chunk_size = 50`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
    ) -> Result<Tensor> {
        self.forward_inner(x, mask, mu, t, spks, cond, streaming, None)
    }

    /// Estimator forward that also returns the activation after every
    /// transformer block plus the final norm, in order. Diagnostic probe
    /// for layer-by-layer parity against the official implementation;
    /// the values follow `forward` exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_traced(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut trace = Vec::with_capacity(self.blocks.len() + 1);
        let out = self.forward_inner(x, mask, mu, t, spks, cond, streaming, Some(&mut trace))?;
        Ok((out, trace))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        x: &Tensor,
        mask: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let seq = x.dim(2)?;
        let x = x.transpose(1, 2)?;
        let mu = mu.transpose(1, 2)?;
        let cond = cond.transpose(1, 2)?;

        let t_emb = self.time_embed.forward(t)?;
        let x = self.input_embed.forward(&x, &cond, &mu, spks)?;
        let rope = self.rotary.forward(seq, x.device(), x.dtype())?;

        let keep_mask = mask.to_dtype(candle_core::DType::U8)?.unsqueeze(1)?;
        let keep_mask = if streaming {
            chunk_attention_mask(seq, self.cfg.static_chunk_size, x.device())?
                .broadcast_mul(&keep_mask)?
        } else {
            keep_mask
        };

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, &t_emb, Some(&keep_mask), &rope)?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(x.clone());
            }
        }
        let x = self.norm_out.forward(&x, &t_emb)?;
        if let Some(trace) = trace {
            trace.push(x.clone());
        }
        self.proj_out.forward(&x)?.transpose(1, 2)
    }

    /// Run one transformer block on a caller-supplied activation and return
    /// its intermediates. Replacement-experiment probe: feeding the official
    /// block input isolates a single block from upstream divergence.
    pub fn block_detail(
        &self,
        idx: usize,
        x: &Tensor,
        t_emb: &Tensor,
        mask: &Tensor,
        seq: usize,
    ) -> Result<DiTBlockDetail> {
        let rope = self.rotary.forward(seq, x.device(), x.dtype())?;
        let keep_mask = mask.to_dtype(candle_core::DType::U8)?.unsqueeze(1)?;
        let block = &self.blocks[idx];
        let (norm, gate_msa, shift_mlp, scale_mlp, gate_mlp) = block.attn_norm.forward(x, t_emb)?;
        let attn = block.attn.forward(&norm, Some(&keep_mask), &rope)?;
        let after_attn = (x + gate_msa.unsqueeze(1)?.broadcast_mul(&attn)?)?;
        let ff_norm = layer_norm_no_affine(&after_attn)?;
        let ff_in = ff_norm
            .broadcast_mul(&(scale_mlp.unsqueeze(1)? + 1.0)?)?
            .broadcast_add(&shift_mlp.unsqueeze(1)?)?;
        let ff = block.ff.forward(&ff_in)?;
        let out = (after_attn.clone() + gate_mlp.unsqueeze(1)?.broadcast_mul(&ff)?)?;
        Ok(DiTBlockDetail {
            norm,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            attn,
            after_attn,
            ff,
            out,
        })
    }
}

/// Intermediates of one DiT block, see [`DiT::block_detail`].
pub struct DiTBlockDetail {
    pub norm: Tensor,
    pub gate_msa: Tensor,
    pub shift_mlp: Tensor,
    pub scale_mlp: Tensor,
    pub gate_mlp: Tensor,
    pub attn: Tensor,
    pub after_attn: Tensor,
    pub ff: Tensor,
    pub out: Tensor,
}

/// `[1, 1, T, T]` bool keep-mask for static-chunk streaming attention.
pub fn chunk_attention_mask(
    size: usize,
    chunk: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    if chunk == 0 {
        candle_core::bail!("chunk size must be positive");
    }
    let row: Vec<u8> = (0..size)
        .flat_map(|i| {
            let block_end = (i / chunk + 1) * chunk;
            (0..size).map(move |j| u8::from(j < block_end))
        })
        .collect();
    Tensor::from_vec(row, (1, 1, size, size), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn gelu_matches_pytorch_tanh_oracle() -> Result<()> {
        let x = Tensor::from_slice(&[-3f32, -1., 0., 1., 3.], 5, &Device::Cpu)?;
        let expected = [-0.003637433, -0.158808, 0., 0.841192, 2.9963627];
        for (a, b) in gelu_tanh(&x)?.to_vec1::<f32>()?.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
        Ok(())
    }

    #[test]
    fn chunk_mask_blocks_future_chunks() -> Result<()> {
        let m = chunk_attention_mask(6, 3, &Device::Cpu)?;
        let v: Vec<u8> = m.flatten_all()?.to_vec1()?;
        // position 2 may attend to 0..3; not 3..6.
        assert!(v[2 * 6] != 0 && v[2 * 6 + 2] != 0 && v[2 * 6 + 3] == 0);
        assert!(v[4 * 6 + 5] != 0);
        assert!(v[5] == 0);
        let _ = DType::F32;
        Ok(())
    }
}
