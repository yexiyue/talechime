//! Diffusion Transformer (DiT) flow estimator.
//!
//! Mirrors `cosyvoice/flow/DiT/dit.py::DiT.forward`: `[x, cond, mu, spks]`
//! input embedding, partial RoPE, `depth` DiT blocks with AdaLN-zero
//! modulation, and a final modulated projection. Weights live under
//! `dit.*` in the converted `flow.safetensors`.

use candle_core::{D, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::embeddings::{
    AdaLayerNormZero, AdaLayerNormZeroFinal, InputEmbedding, LinT2d, ModulatedInput,
    RotaryEmbedding, TimestepEmbedding, apply_rotary_cos_sin, layer_norm_no_affine,
    modulated_layer_norm,
};
use crate::config::DiTConfig;

/// FeedForward with tanh-approximate GELU (`approximate="tanh"`).
#[derive(Debug)]
pub struct FeedForward {
    linear1: LinT2d,
    linear2: LinT2d,
}

impl FeedForward {
    pub fn new(dim: usize, ff_mult: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = dim * ff_mult;
        // nn.Sequential layout: ff.0.0 / ff.2.
        Ok(Self {
            linear1: LinT2d::from_linear(candle_nn::linear(dim, hidden, vb.pp("ff.0.0"))?)?,
            linear2: LinT2d::from_linear(candle_nn::linear(hidden, dim, vb.pp("ff.2"))?)?,
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

/// RoPE tables for one solve, prepared once in `prepare_solve`.
///
/// `cos`/`sin` hold the pairwise-duplicated `[1, T, head_dim]` tables the
/// reference (CPU) chain broadcasts against. On Metal, `cos_i`/`sin_i` hold
/// the unique-frequency `[T, head_dim/2]` views the fused `rope_i` kernel
/// needs — only the first `head_dim` flat channels actually rotate, so the
/// fused chain applies `rope_i` to that slice alone.
struct RopeTables {
    cos: Tensor,
    sin: Tensor,
    cos_i: Option<Tensor>,
    sin_i: Option<Tensor>,
}

impl RopeTables {
    fn new(
        rotary: &RotaryEmbedding,
        seq: usize,
        device: &Device,
        dtype: candle_core::DType,
    ) -> Result<Self> {
        let (cos, sin) = rotary.forward_cos_sin(seq, device, dtype)?;
        if !device.is_metal() {
            return Ok(Self {
                cos,
                sin,
                cos_i: None,
                sin_i: None,
            });
        }
        let half = cos.dim(D::Minus1)? / 2;
        let uniq = |dup: &Tensor| -> Result<Tensor> {
            dup.reshape((1, seq, half, 2))?
                .narrow(D::Minus1, 0, 1)?
                .squeeze(D::Minus1)?
                .squeeze(0)?
                .contiguous()
        };
        Ok(Self {
            cos_i: Some(uniq(&cos)?),
            sin_i: Some(uniq(&sin)?),
            cos,
            sin,
        })
    }
}

#[derive(Debug)]
struct Attention {
    to_q: LinT2d,
    to_k: LinT2d,
    to_v: LinT2d,
    to_out: LinT2d,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new(dim: usize, heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = heads * head_dim;
        Ok(Self {
            to_q: LinT2d::from_linear(candle_nn::linear(dim, inner, vb.pp("to_q"))?)?,
            to_k: LinT2d::from_linear(candle_nn::linear(dim, inner, vb.pp("to_k"))?)?,
            to_v: LinT2d::from_linear(candle_nn::linear(dim, inner, vb.pp("to_v"))?)?,
            to_out: LinT2d::from_linear(candle_nn::linear(inner, dim, vb.pp("to_out.0"))?)?,
            heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// `mask_bias` is an optional additive attention mask with `0.0` at
    /// visible positions and `-inf` at masked positions. On Metal it must
    /// already be expanded to `[B, heads, T, T]` (see `prepare_solve`); the
    /// fused sdpa kernel rejects broadcastable shapes. Other devices take
    /// the `[B_m, 1, T_m, T]` broadcastable form.
    fn forward(
        &self,
        x: &Tensor,
        mask_bias: Option<&Tensor>,
        rope: &RopeTables,
        prof: Option<&mut DitProfile>,
    ) -> Result<Tensor> {
        let mut prof = prof;
        let sub = prof.as_ref().is_some_and(|p| p.enabled);
        let t_sub = sub.then(std::time::Instant::now);
        let (b, t, _) = x.dims3()?;
        let q = self.to_q.forward(x)?;
        let k = self.to_k.forward(x)?;
        let v = self.to_v.forward(x)?;
        if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t_sub) {
            p.sync_add_attn_part(x.device(), ts, false, true);
        }
        let t_sub = sub.then(std::time::Instant::now);

        let shape = |tt: &Tensor| {
            tt.reshape((b, t, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };

        let mask_ready = mask_bias.is_none_or(|m| m.dims() == [b, self.heads, t, t].as_slice());
        if let (true, Some(cos_i), Some(sin_i)) = (
            x.device().is_metal()
                && mask_ready
                && matches!(self.head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512),
            &rope.cos_i,
            &rope.sin_i,
        ) {
            // Fused chain: rotate only the real `head_dim`-wide slice with
            // the rope_i kernel (only the first flat channels rotate), then
            // attention through the fused sdpa kernel. Gated against the
            // reference chain in `tests::attention_stage_timing_metal`.
            // The rope+head-split fusion (one dispatch instead of
            // slice-copy/rope_i/cat/transpose per q,k) is preferred when the
            // layout allows; numerics are gated bit-exact against the
            // four-kernel chain.
            let rotate = |src: &Tensor| -> Result<Tensor> {
                let slice = src.narrow(D::Minus1, 0, self.head_dim)?.contiguous()?;
                let rot = candle_nn::rotary_emb::rope_i(&slice.unsqueeze(1)?, cos_i, sin_i)?;
                let rot = rot.squeeze(1)?;
                Tensor::cat(
                    &[
                        &rot,
                        &src.narrow(
                            D::Minus1,
                            self.head_dim,
                            src.dim(D::Minus1)? - self.head_dim,
                        )?,
                    ],
                    D::Minus1,
                )
            };
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let fused_rope = |src: &Tensor| -> Result<Tensor> {
                crate::flow::metal_fused::rope_headsplit(
                    src,
                    cos_i,
                    sin_i,
                    self.heads,
                    self.head_dim,
                )
            };
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let (q, k) = match (fused_rope(&q), fused_rope(&k)) {
                (Ok(q), Ok(k)) => (q, k),
                _ => (shape(&rotate(&q)?)?, shape(&rotate(&k)?)?),
            };
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            let (q, k) = (shape(&rotate(&q)?)?, shape(&rotate(&k)?)?);
            let v = shape(&v)?;
            if let (Some(p), Some(ts)) = (prof.as_deref_mut(), t_sub) {
                p.sync_add_attn_part(x.device(), ts, true, false);
            }
            let t_sub = sub.then(std::time::Instant::now);
            let out = candle_nn::ops::sdpa(&q, &k, &v, mask_bias, false, self.scale as f32, 1.0)?;
            let out = out.transpose(1, 2)?.contiguous()?.flatten(2, 3)?;
            let out = self.to_out.forward(&out)?;
            if let (Some(p), Some(ts)) = (prof, t_sub) {
                p.sync_add_attn_part(x.device(), ts, false, false);
            }
            return Ok(out);
        }

        // Reference chain (CPU and unexpanded-mask diagnostics).
        let q = apply_rotary_cos_sin(&q, &rope.cos, &rope.sin)?;
        let k = apply_rotary_cos_sin(&k, &rope.cos, &rope.sin)?;
        let (q, k, v) = (shape(&q)?, shape(&k)?, shape(&v)?);

        let mut attn = q
            .matmul(&k.transpose(D::Minus2, D::Minus1)?)?
            .affine(self.scale, 0.0)?;
        if let Some(bias) = mask_bias {
            attn = attn.broadcast_add(bias)?;
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

    /// One transformer block step. On Metal with `carry` (the previous
    /// block's fused tail normed input + this block's raw AdaLayerNorm
    /// embedding) and `emb_next` (the next modulator), the boundary layer
    /// norms are fused into the tail kernel: the block returns
    /// `(residual_carry, Some(normed_for_next))` where `normed_for_next`
    /// feeds the next block's attention (or, for the last block, `proj_out`
    /// directly). Otherwise the reference op chain runs and the second
    /// return is `None`.
    #[allow(clippy::too_many_arguments)]
    fn forward_act(
        &self,
        x: &Tensor,
        t_act: &Tensor,
        mask_bias: Option<&Tensor>,
        rope: &RopeTables,
        prof: Option<&mut DitProfile>,
        carry: Option<(Tensor, &Tensor)>,
        #[cfg_attr(
            not(all(feature = "metal", target_os = "macos")),
            allow(unused_variables)
        )]
        emb_next: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let mut prof = prof;
        let t_region = prof
            .as_ref()
            .is_some_and(|p| p.enabled)
            .then(std::time::Instant::now);
        // `carry` = (pre-fused normed input, this block's raw (B,1,6D)
        // AdaLayerNorm embedding): the affine already ran, so the boundary
        // layer norm is skipped (it lives in the previous tail kernel).
        #[cfg_attr(
            not(all(feature = "metal", target_os = "macos")),
            allow(unused_variables)
        )]
        let ModulatedInput {
            normalized: norm,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            raw: emb_raw,
        } = match carry {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Some((normalized, emb)) => ModulatedInput::from_raw(normalized, emb.clone())?,
            _ => self.attn_norm.forward_act(x, t_act)?,
        };
        if let (Some(p), Some(t)) = (prof.as_deref_mut(), t_region) {
            p.sync_add_adaln(x.device(), t);
        }
        let t_region = prof
            .as_ref()
            .is_some_and(|p| p.enabled)
            .then(std::time::Instant::now);
        let attn = self
            .attn
            .forward(&norm, mask_bias, rope, prof.as_deref_mut())?;
        if let (Some(p), Some(t)) = (prof.as_deref_mut(), t_region) {
            p.sync_add_attn(x.device(), t);
        }
        let t_region = prof
            .as_ref()
            .is_some_and(|p| p.enabled)
            .then(std::time::Instant::now);
        // Gate region + block tail: on Metal both run as fused kernels
        // (the gated residual/mod-LN pass writes the FFN input and the
        // residual base as two planes; the tail adds gate_mlp⊙ff and — when
        // the next modulator is available — the next block's normed input as
        // a second plane). Gated against the op-chain reference in `tests`.
        let reference_gate = || -> Result<(Tensor, Tensor)> {
            let gated = gate_msa.broadcast_mul(&attn)?;
            let x_res = (x + &gated)?;
            let ff_in = modulated_layer_norm(&x_res, &(&scale_mlp + 1.0)?, &shift_mlp)?;
            Ok((ff_in, x_res))
        };
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let (ff_in, x_res) = if super::metal_fused::supports_gate(x) {
            let both = crate::flow::metal_fused::gate_residual_modln(&attn, x, &emb_raw)?;
            (
                both.narrow(0, 0, 1)?.squeeze(0)?,
                both.narrow(0, 1, 1)?.squeeze(0)?,
            )
        } else {
            reference_gate()?
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let (ff_in, x_res) = reference_gate()?;
        if let (Some(p), Some(t)) = (prof.as_deref_mut(), t_region) {
            p.sync_add_gate(x.device(), t);
        }
        let t_region = prof
            .as_ref()
            .is_some_and(|p| p.enabled)
            .then(std::time::Instant::now);
        let ff = self.ff.forward(&ff_in)?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let (out, normed_next) = if super::metal_fused::supports_gate(x) {
            match emb_next {
                Some(emb_next) => {
                    // candle's cat occasionally returns the transpose-back
                    // layout here (strides (6D, 12D, 1)); the kernel requires
                    // row-major, so normalize (no-op when already packed).
                    let packed = Tensor::cat(&[&emb_raw, emb_next], 1)?.contiguous()?;
                    let tail = crate::flow::metal_fused::gate_tail_modln(&ff, &x_res, &packed)?;
                    let carry = tail.narrow(0, 0, 1)?.squeeze(0)?;
                    let normed = tail.narrow(0, 1, 1)?.squeeze(0)?;
                    (carry, Some(normed))
                }
                None => {
                    let gated = gate_mlp.broadcast_mul(&ff)?;
                    ((x_res + &gated)?, None)
                }
            }
        } else {
            let gated = gate_mlp.broadcast_mul(&ff)?;
            ((x_res + gated)?, None)
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let (out, normed_next) = {
            let gated = gate_mlp.broadcast_mul(&ff)?;
            ((x_res + &gated)?, None)
        };
        if let (Some(p), Some(t)) = (prof, t_region) {
            p.sync_add_ffn(out.device(), t);
        }
        Ok((out, normed_next))
    }
}

/// Wall-clock region attribution for one prepared DiT solve step, enabled by
/// `CV3_DIT_PROFILE=1`. Every region synchronizes the device before stopping
/// its timer: the Metal queue is async, so an unsynced region only measures
/// enqueue time and the first CPU readback absorbs the queue (the trap that
/// mis-attributed cv3 hift/istft and f0 earlier). Zero cost when disabled;
/// diagnostics only — never part of the timing contract.
#[derive(Debug, Default)]
struct DitProfile {
    enabled: bool,
    embed_ms: f64,
    adaln_ms: f64,
    attn_ms: f64,
    gate_ms: f64,
    ffn_ms: f64,
    out_ms: f64,
    /// Attn-region sub-split (Metal branch): qkv/to_out GEMMs, the rotate
    /// chain (slice+contiguous+rope_i+cat+transpose), sdpa + output glue.
    attn_gemm_ms: f64,
    attn_rot_ms: f64,
    attn_sdpa_ms: f64,
    step_wall_ms: f64,
}

impl DitProfile {
    fn from_env() -> Self {
        Self {
            enabled: std::env::var("CV3_DIT_PROFILE").is_ok_and(|v| v != "0"),
            ..Default::default()
        }
    }

    #[inline]
    fn ms(start: std::time::Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1e3
    }

    fn sync_add_adaln(&mut self, device: &Device, start: std::time::Instant) {
        device.synchronize().ok();
        self.adaln_ms += Self::ms(start);
    }

    fn sync_add_attn(&mut self, device: &Device, start: std::time::Instant) {
        device.synchronize().ok();
        self.attn_ms += Self::ms(start);
    }

    fn sync_add_gate(&mut self, device: &Device, start: std::time::Instant) {
        device.synchronize().ok();
        self.gate_ms += Self::ms(start);
    }

    fn sync_add_ffn(&mut self, device: &Device, start: std::time::Instant) {
        device.synchronize().ok();
        self.ffn_ms += Self::ms(start);
    }

    /// Sub-split accumulator for one attn region segment; syncs first so the
    /// async queue cannot leak work across buckets.
    fn sync_add_attn_part(
        &mut self,
        device: &Device,
        start: std::time::Instant,
        rot: bool,
        gemm: bool,
    ) {
        device.synchronize().ok();
        let ms = Self::ms(start);
        if rot {
            self.attn_rot_ms += ms;
        } else if gemm {
            self.attn_gemm_ms += ms;
        } else {
            self.attn_sdpa_ms += ms;
        }
    }

    fn step_line(&self, seq: usize) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "[cv3-dit] step seq {seq} wall {:.1} ms | embed {:.1} adaln {:.1} attn {:.1} (gemm {:.1} rot {:.1} sdpa {:.1}) gate {:.1} ffn {:.1} out {:.1} | blocks_sum {:.1} (dispatch gap {:.1})",
            self.step_wall_ms,
            self.embed_ms,
            self.adaln_ms,
            self.attn_ms,
            self.attn_gemm_ms,
            self.attn_rot_ms,
            self.attn_sdpa_ms,
            self.gate_ms,
            self.ffn_ms,
            self.out_ms,
            self.adaln_ms + self.attn_ms + self.gate_ms + self.ffn_ms,
            self.step_wall_ms
                - self.embed_ms
                - self.adaln_ms
                - self.attn_ms
                - self.gate_ms
                - self.ffn_ms
                - self.out_ms,
        );
    }
}

/// Step-invariant inputs prepared once for a multi-step CFM Euler solve.
pub(crate) struct PreparedDiT {
    /// Concatenated `[cond_t, mu_t, spks_t]` of shape `[B, T, 2*mel + spk]`.
    cond_mu_spks: Tensor,
    rope: RopeTables,
    /// Optional additive attention mask (`0.0` keep, `-inf` masked). `None`
    /// when `streaming == false` and the caller's mask is proven all-ones.
    mask_bias: Option<Tensor>,
    /// Metal-only: `mask_bias` expanded to `[B, heads, T, T]` once per solve
    /// for the fused sdpa kernel, which requires exact mask shapes.
    mask_bias_heads: Option<Tensor>,
}

#[derive(Debug)]
pub struct DiT {
    cfg: DiTConfig,
    time_embed: TimestepEmbedding,
    input_embed: InputEmbedding,
    rotary: RotaryEmbedding,
    blocks: Vec<DiTBlock>,
    norm_out: AdaLayerNormZeroFinal,
    proj_out: LinT2d,
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
            proj_out: LinT2d::from_linear(candle_nn::linear(
                cfg.dim,
                cfg.out_channels,
                vb.pp("proj_out"),
            )?)?,
            cfg,
        })
    }

    /// Precompute step-invariant tensors (`[cond, mu, spks]` tail, RoPE cos/sin,
    /// and attention mask) once before the CFM Euler loop.
    pub(crate) fn prepare_solve(
        &self,
        mask: &Tensor,
        mu: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
        mask_all_ones: bool,
    ) -> Result<PreparedDiT> {
        let (b, _, seq) = mu.dims3()?;
        let dtype = mu.dtype();
        let device = mu.device();
        let mu_t = mu.transpose(1, 2)?;
        let cond_t = cond.transpose(1, 2)?;
        let spks_t = spks
            .unsqueeze(1)?
            .broadcast_as((b, seq, self.cfg.spk_dim))?;
        let cond_mu_spks = Tensor::cat(&[&cond_t, &mu_t, &spks_t], D::Minus1)?;
        let rope = RopeTables::new(&self.rotary, seq, device, dtype)?;
        let mask_bias = if !streaming && mask_all_ones {
            None
        } else {
            let keep_mask = if mask_all_ones {
                chunk_attention_mask(seq, self.cfg.static_chunk_size, device)?
            } else {
                let keep = mask.to_dtype(candle_core::DType::U8)?.unsqueeze(1)?;
                if streaming {
                    chunk_attention_mask(seq, self.cfg.static_chunk_size, device)?
                        .broadcast_mul(&keep)?
                } else {
                    keep
                }
            };
            Some(keep_mask_to_additive(&keep_mask, dtype, device)?)
        };
        let mask_bias_heads = match (&mask_bias, device.is_metal()) {
            (Some(bias), true) => Some(
                bias.broadcast_as((b, self.cfg.heads, seq, seq))?
                    .contiguous()?,
            ),
            _ => None,
        };
        Ok(PreparedDiT {
            cond_mu_spks,
            rope,
            mask_bias,
            mask_bias_heads,
        })
    }

    pub(crate) fn forward_prepared(
        &self,
        x: &Tensor,
        t: &Tensor,
        prepared: &PreparedDiT,
        mut trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let mut prof = DitProfile::from_env();
        let t_step = if prof.enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let x_t = x.transpose(1, 2)?;
        let t_emb = self.time_embed.forward(t)?;
        let t_act = candle_nn::ops::silu(&t_emb)?;
        let combined = Tensor::cat(&[&x_t, &prepared.cond_mu_spks], D::Minus1)?;
        let mut x = self.input_embed.forward_combined(&combined)?;
        if let Some(t) = t_step {
            x.device().synchronize()?;
            prof.embed_ms += DitProfile::ms(t);
        }
        let mask_bias = if x.device().is_metal() {
            prepared
                .mask_bias_heads
                .as_ref()
                .or(prepared.mask_bias.as_ref())
        } else {
            prepared.mask_bias.as_ref()
        };
        // Boundary-LN fused chain (Metal): block i's tail kernel produces
        // block i+1's normed input, so each block needs the NEXT block's
        // raw AdaLayerNorm embedding (or, for the last block, the final
        // norm's — reordered into AdaLayerNorm channel order [shift, scale]
        // and padded to 6D).
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let fused_boundaries = super::metal_fused::supports_gate(&x);
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let emb_next_of = |i: usize| -> Result<Option<Tensor>> {
            if !fused_boundaries {
                return Ok(None);
            }
            if i + 1 < self.blocks.len() {
                Ok(Some(self.blocks[i + 1].attn_norm.mods(&t_act)?))
            } else {
                Ok(Some(self.norm_out.tail_modulation(&t_act)?))
            }
        };
        let n_blocks = self.blocks.len();
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let mut carry_norm: Option<(Tensor, Tensor)> = None; // (normed, emb_this)
        let mut final_normed: Option<Tensor> = None;
        for (i, block) in self.blocks.iter().enumerate() {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let emb_next = emb_next_of(i)?;
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let carry_owned = carry_norm.take();
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let (block_out, normed) = block.forward_act(
                &x,
                &t_act,
                mask_bias,
                &prepared.rope,
                Some(&mut prof),
                carry_owned.as_ref().map(|(n, e)| (n.clone(), e)),
                emb_next.as_ref(),
            )?;
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            let (block_out, normed) = block.forward_act(
                &x,
                &t_act,
                mask_bias,
                &prepared.rope,
                Some(&mut prof),
                None,
                None,
            )?;
            x = block_out;
            if let Some(trace) = trace.as_deref_mut() {
                trace.push(x.clone());
            }
            if let Some(normed) = normed {
                if i + 1 == n_blocks {
                    final_normed = Some(normed);
                } else {
                    #[cfg(all(feature = "metal", target_os = "macos"))]
                    {
                        carry_norm = Some((normed, emb_next.expect("fused chain provides emb")));
                    }
                }
            }
        }
        let t_region = if prof.enabled {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let x = if let Some(final_normed) = final_normed {
            final_normed
        } else {
            self.norm_out.forward_act(&x, &t_act)?
        };
        let out = self.proj_out.forward(&x)?.transpose(1, 2)?;
        if let (Some(t_start), Some(t_region)) = (t_step, t_region) {
            x.device().synchronize()?;
            prof.out_ms += DitProfile::ms(t_region);
            prof.step_wall_ms += DitProfile::ms(t_start);
            prof.step_line(prepared.cond_mu_spks.dim(1)?);
        }
        if let Some(trace) = trace {
            trace.push(x.clone());
        }
        Ok(out)
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
        trace: Option<&mut Vec<Tensor>>,
    ) -> Result<Tensor> {
        let prepared = self.prepare_solve(mask, mu, spks, cond, streaming, false)?;
        self.forward_prepared(x, t, &prepared, trace)
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
        let rope = RopeTables::new(&self.rotary, seq, x.device(), x.dtype())?;
        let keep_mask = mask.to_dtype(candle_core::DType::U8)?.unsqueeze(1)?;
        // Base (broadcastable) mask form: even on Metal this deliberately
        // exercises the reference chain so the probe stays comparable to the
        // official implementation.
        let mask_bias = keep_mask_to_additive(&keep_mask, x.dtype(), x.device())?;
        let block = &self.blocks[idx];
        let (norm, gate_msa, shift_mlp, scale_mlp, gate_mlp) = block.attn_norm.forward(x, t_emb)?;
        let attn = block.attn.forward(&norm, Some(&mask_bias), &rope, None)?;
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

/// Convert a U8 keep-mask (`1` = attend, `0` = mask) into an additive bias
/// tensor (`0.0` = attend, `-inf` = mask) once before the solver loop.
fn keep_mask_to_additive(
    keep_mask: &Tensor,
    dtype: candle_core::DType,
    device: &Device,
) -> Result<Tensor> {
    let shape = keep_mask.shape();
    let zeros = Tensor::zeros(shape, dtype, device)?;
    // Build the scalar on CPU: Metal has no F64 -> F32 cast.
    let neg = Tensor::new(f64::NEG_INFINITY, &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)?
        .broadcast_as(shape)?
        .contiguous()?;
    keep_mask.where_cond(&zeros, &neg)
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

    #[test]
    fn additive_mask_and_prepared_solve_match_where_cond() -> Result<()> {
        let dev = Device::Cpu;
        let keep = chunk_attention_mask(6, 3, &dev)?;
        let bias = keep_mask_to_additive(&keep, DType::F32, &dev)?;
        let scores = Tensor::arange(0f32, 36f32, &dev)?.reshape((1, 1, 6, 6))?;
        let neg = Tensor::new(f32::NEG_INFINITY, &dev)?.broadcast_as((1, 1, 6, 6))?;
        let via_where = candle_nn::ops::softmax_last_dim(&keep.where_cond(&scores, &neg)?)?;
        let via_add = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&bias)?)?;
        assert_eq!(
            via_where.flatten_all()?.to_vec1::<f32>()?,
            via_add.flatten_all()?.to_vec1::<f32>()?
        );

        let cfg = DiTConfig {
            dim: 32,
            depth: 2,
            heads: 2,
            dim_head: 16,
            ff_mult: 2,
            static_chunk_size: 3,
            ..DiTConfig::default()
        };
        let dit = DiT::new(cfg, VarBuilder::zeros(DType::F32, &dev))?;
        let x = Tensor::ones((2, 80, 6), DType::F32, &dev)?;
        let mask = Tensor::ones((2, 1, 6), DType::F32, &dev)?;
        let mu = Tensor::ones((2, 80, 6), DType::F32, &dev)?;
        let t = Tensor::from_slice(&[0.25f32, 0.25], 2, &dev)?;
        let spks = Tensor::ones((2, 80), DType::F32, &dev)?;
        let cond = Tensor::ones((2, 80, 6), DType::F32, &dev)?;
        for streaming in [false, true] {
            let direct = dit.forward(&x, &mask, &mu, &t, &spks, &cond, streaming)?;
            let prep = dit.prepare_solve(&mask, &mu, &spks, &cond, streaming, true)?;
            let prepared = dit.forward_prepared(&x, &t, &prep, None)?;
            assert_eq!(
                direct.flatten_all()?.to_vec1::<f32>()?,
                prepared.flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }

    fn err_vs(out: &Tensor, truth: &Tensor) -> Result<f64> {
        let max_abs = (out - truth)?.abs()?.max_all()?.to_scalar::<f32>()? as f64;
        let mag = truth.abs()?.max_all()?.to_scalar::<f32>()? as f64;
        Ok(max_abs / mag)
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device; no models"]
    fn padded_mask_and_small_geometry_metal() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let cfg = DiTConfig {
            dim: 32,
            depth: 2,
            heads: 2,
            dim_head: 16,
            ff_mult: 2,
            static_chunk_size: 3,
            ..DiTConfig::default()
        };
        let dit = DiT::new(cfg, VarBuilder::zeros(DType::F32, &dev))?;
        let x = Tensor::ones((2, 80, 6), DType::F32, &dev)?;
        let mask = Tensor::from_slice(&[1f32, 1., 1., 1., 0., 0.], (1, 1, 6), &dev)?
            .broadcast_as((2, 1, 6))?
            .contiguous()?;
        let t = Tensor::from_slice(&[0.25f32, 0.25], 2, &dev)?;
        let spks = Tensor::ones((2, 80), DType::F32, &dev)?;
        for streaming in [false, true] {
            let prep = dit.prepare_solve(&mask, &x, &spks, &x, streaming, false)?;
            let bias = prep.mask_bias_heads.as_ref().unwrap();
            assert_eq!(bias.dims(), &[2, 2, 6, 6]);
            let values = bias.flatten_all()?.to_vec1::<f32>()?;
            for row in values.as_chunks::<6>().0 {
                assert_eq!(row[0], 0.0);
                assert_eq!(row[4], f32::NEG_INFINITY);
                assert_eq!(row[5], f32::NEG_INFINITY);
            }
            // A configurable dim=32 model must use the reference gate chain,
            // rather than entering the fixed-width 1024 fusion and failing.
            let out = dit.forward_prepared(&x, &t, &prep, None)?;
            assert_eq!(out.dims(), &[2, 80, 6]);
            assert!(
                out.flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|v| *v == 0.)
            );
        }
        Ok(())
    }

    /// Numeric gate for the fused rope/head-split kernel: the rotation
    /// expression matches `rope_i`'s kernel exactly and passthrough channels
    /// are copies, so the fused single-dispatch result must be bit-identical
    /// to the production four-kernel chain (slice-copy/rope_i/cat/transpose).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore]
    fn fused_rope_headsplit_matches_chain_metal() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping, no metal device: {e}");
                return;
            }
        };
        let (b, heads, head_dim) = (2usize, 16usize, 64usize);
        let hidden = heads * head_dim;
        for t in [534usize, 128] {
            // Post-GEMM magnitudes: q/k entries are O(1) at sigma 1/32.
            let q = Tensor::randn(0f32, 1f32 / 32f32, (b, t, hidden), &device).unwrap();
            let cos = Tensor::randn(0f32, 1f32, (t, head_dim / 2), &device).unwrap();
            let sin = Tensor::randn(0f32, 1f32, (t, head_dim / 2), &device).unwrap();

            // Production chain (as in Attention::forward's fallback).
            let rotate = |src: &Tensor| -> Result<Tensor> {
                let slice = src.narrow(D::Minus1, 0, head_dim)?.contiguous()?;
                let rot = candle_nn::rotary_emb::rope_i(
                    &slice.unsqueeze(1)?,
                    &cos.contiguous()?,
                    &sin.contiguous()?,
                )?;
                let rot = rot.squeeze(1)?;
                Tensor::cat(
                    &[&rot, &src.narrow(D::Minus1, head_dim, hidden - head_dim)?],
                    D::Minus1,
                )
            };
            let shape = |tt: &Tensor| {
                tt.reshape((b, t, heads, head_dim))?
                    .transpose(1, 2)?
                    .contiguous()
            };
            let reference = shape(&rotate(&q).unwrap()).unwrap();
            let fused =
                crate::flow::metal_fused::rope_headsplit(&q, &cos, &sin, heads, head_dim).unwrap();
            let max_abs = (&fused - &reference)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            println!("rope-headsplit t={t} max_abs {max_abs:.3e}");
            assert_eq!(fused.dims(), &[b, heads, t, head_dim][..]);
            assert!(max_abs == 0.0, "rope-headsplit t={t} max_abs {max_abs:.3e}");
        }
    }

    /// Numeric gate for the fused gate-region/tail kernels against the
    /// production op chain (broadcast mul/add + per-branch modulated LN).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore]
    fn fused_gate_region_matches_chain_metal() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping, no metal device: {e}");
                return;
            }
        };
        let (b, t, d) = (2usize, 534usize, 1024usize);
        let attn = Tensor::randn(0f32, 1f32 / 16f32, (b, t, d), &device).unwrap();
        let x = (Tensor::randn(0f32, 1f32, (b, t, d), &device).unwrap()
            + Tensor::full(0.2f32, (b, t, d), &device).unwrap())
        .unwrap();
        let emb = Tensor::randn(0f32, 0.1f32, (b, 1, 6 * d), &device).unwrap();

        // Reference chain from the production op sequence.
        let gate_msa = emb.narrow(D::Minus1, 2 * d, d).unwrap();
        let shift_mlp = emb.narrow(D::Minus1, 3 * d, d).unwrap();
        let scale_mlp = emb.narrow(D::Minus1, 4 * d, d).unwrap();
        let gate_mlp = emb.narrow(D::Minus1, 5 * d, d).unwrap();
        let gated = gate_msa.broadcast_mul(&attn).unwrap();
        let x_res = (x.clone() + gated).unwrap();
        let ff_in_ref = crate::flow::embeddings::modulated_layer_norm(
            &x_res,
            &(scale_mlp + 1.0).unwrap(),
            &shift_mlp,
        )
        .unwrap();
        let ff = Tensor::randn(0f32, 1f32 / 8f32, (b, t, d), &device).unwrap();
        let out_ref = (x_res.clone() + gate_mlp.broadcast_mul(&ff).unwrap()).unwrap();

        // Tail v2: next block's modulation in AdaLayerNorm channel order
        // (shift @0D, scale @D), padded to 6D as production builds it.
        let emb_next = Tensor::randn(0f32, 0.1f32, (b, 1, 6 * d), &device).unwrap();
        let shift_next = emb_next.narrow(D::Minus1, 0, d).unwrap();
        let scale_next = emb_next.narrow(D::Minus1, d, d).unwrap();
        let normed_ref = crate::flow::embeddings::modulated_layer_norm(
            &out_ref,
            &(scale_next + 1.0).unwrap(),
            &shift_next,
        )
        .unwrap();

        let both = crate::flow::metal_fused::gate_residual_modln(&attn, &x, &emb).unwrap();
        let ff_in = both.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        let z = both.narrow(0, 1, 1).unwrap().squeeze(0).unwrap();
        let packed = Tensor::cat(&[&emb, &emb_next], 1).unwrap();
        let tail = crate::flow::metal_fused::gate_tail_modln(&ff, &z, &packed).unwrap();
        let out = tail.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        let normed = tail.narrow(0, 1, 1).unwrap().squeeze(0).unwrap();

        let rel_ff = err_vs(&ff_in, &ff_in_ref).unwrap();
        let rel_z = err_vs(&z, &x_res).unwrap();
        let rel_out = err_vs(&out, &out_ref).unwrap();
        let rel_normed = err_vs(&normed, &normed_ref).unwrap();
        println!(
            "gate-region rel: ff_in {rel_ff:.3e} z {rel_z:.3e} out {rel_out:.3e} normed {rel_normed:.3e}"
        );
        // z/out differ from the op chain only by Metal's fma contraction of
        // `x + g*attn` (one rounding instead of two); ff_in/normed add the
        // one-pass vs Welford variance difference. All are pure float
        // rounding.
        assert!(rel_z <= 1e-6, "z (residual plane) rel {rel_z:.3e}");
        assert!(rel_ff <= 1e-5, "ff_in rel {rel_ff:.3e}");
        assert!(rel_out <= 1e-6, "out rel {rel_out:.3e}");
        assert!(rel_normed <= 1e-5, "normed rel {rel_normed:.3e}");
    }

    /// Numeric gate for the fused modulated layer norm on Metal: the fused
    /// kernel computes the variance in one pass (`E[x^2] - mean^2`) while the
    /// reference chain is two-pass; at f32 with realistic activation
    /// magnitudes the difference must stay far below any semantic level.
    /// The CPU branch of `modulated_layer_norm` is the reference chain, so
    /// the CPU row doubles as a cross-check of the harness itself.
    #[test]
    #[ignore]
    fn modulated_ln_matches_reference_metal() {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let devices = [Device::Cpu, Device::new_metal(0).unwrap()];
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let devices = [Device::Cpu];
        for device in devices {
            for (t, sigma) in [(534usize, 1f32), (128, 8.0)] {
                let base = Tensor::randn(0f32, sigma, (2, t, 1024), &device).unwrap();
                let offset = Tensor::full(0.3f32 * sigma, (2, t, 1024), &device).unwrap();
                let x = (base + offset).unwrap();
                let scale = Tensor::randn(0f32, 0.5f32, (2, 1, 1024), &device).unwrap();
                let shift = Tensor::randn(0f32, 0.5f32, (2, 1, 1024), &device).unwrap();
                let fused = crate::flow::embeddings::modulated_layer_norm(
                    &x,
                    &(scale.clone() + 1.0).unwrap(),
                    &shift,
                )
                .unwrap();
                let norm = crate::flow::embeddings::layer_norm_no_affine(&x).unwrap();
                let reference = norm
                    .broadcast_mul(&(scale + 1.0).unwrap())
                    .unwrap()
                    .broadcast_add(&shift)
                    .unwrap();
                let rel = err_vs(&fused, &reference).unwrap();
                println!("modulated-ln t={t} sigma={sigma} rel {rel:.3e}");
                assert!(rel <= 1e-5, "t={t} sigma={sigma} rel {rel:.3e}");
            }
        }
    }

    /// Fixed-shape per-op probe of the production `Attention::forward` chain
    /// on Metal. Real model geometry (dim 1024 / heads 16 / head_dim 64, CFG
    /// batch 2) at streaming shapes T in {128, 512}; rope tables come from the
    /// real `RotaryEmbedding`, the mask from the production chunk helper.
    /// Numeric gates assert the candidate chains (slice+rope_i, sdpa) match
    /// the production chain within float rounding before any timing is
    /// trusted. Run with:
    /// `cargo test -p talechime-cosyvoice3 --features metal --lib flow::dit::tests::attention_stage_timing_metal -- --ignored --test-threads=1 --nocapture`
    #[test]
    #[ignore]
    fn attention_stage_timing_metal() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping, no metal device: {e}");
                return;
            }
        };
        let (b, heads, head_dim) = (2usize, 16usize, 64usize);
        let hidden = heads * head_dim;
        let ff_hidden = 2 * hidden;
        let scale = 1f64 / (head_dim as f64).sqrt();
        let chunk = 50usize;

        let stage = |name: &str, f: &mut dyn FnMut() -> Result<Tensor>| -> Tensor {
            for _ in 0..5 {
                std::mem::drop(f().unwrap());
            }
            device.synchronize().unwrap();
            let start = std::time::Instant::now();
            let mut last = None;
            for _ in 0..50 {
                last = Some(f().unwrap());
            }
            device.synchronize().unwrap();
            let ms = start.elapsed().as_secs_f64() / 50f64 * 1e3;
            println!("  {name:16} {ms:8.4} ms");
            last.unwrap()
        };

        for &t in &[128usize, 512usize] {
            println!("== T={t} B={b} H={heads} D={head_dim} ==");
            let mk = |rows: usize, cols: usize| -> (Tensor, Tensor) {
                (
                    Tensor::randn(0f32, 1f32, (rows, cols), &device).unwrap(),
                    Tensor::randn(0f32, 1f32, (rows,), &device).unwrap(),
                )
            };
            let (wq, bq) = mk(hidden, hidden);
            let (wk, bk) = mk(hidden, hidden);
            let (wv, bv) = mk(hidden, hidden);
            let (wo, bo) = mk(hidden, hidden);
            let (wf1, bf1) = mk(ff_hidden, hidden);
            let (wf2, bf2) = mk(hidden, ff_hidden);
            let lin = |w: Tensor, bias: Tensor| -> LinT2d {
                LinT2d::from_linear(candle_nn::Linear::new(w, Some(bias))).unwrap()
            };
            let lin_q = lin(wq.clone(), bq.clone());
            let lin_k = lin(wk.clone(), bk.clone());
            let lin_v = lin(wv.clone(), bv.clone());
            let lin_out = lin(wo.clone(), bo.clone());
            let lin_f1 = lin(wf1.clone(), bf1.clone());
            let lin_f2 = lin(wf2.clone(), bf2.clone());

            // sigma = 1/sqrt(hidden): with sigma=1 weights the projections
            // land at sigma~1 like the real post-AdaLN activations, keeping
            // attention scores O(1). Larger sigmas inflate scores to ~1e3
            // where ULP-level GEMM ordering flips near-tied softmax entries
            // and the relative gates measure float noise, not semantics.
            let x = Tensor::randn(0f32, 1f32 / 32f32, (b, t, hidden), &device).unwrap();
            let q0 = lin_q.forward(&x).unwrap();
            let k0 = lin_k.forward(&x).unwrap();
            let v0 = lin_v.forward(&x).unwrap();

            let rotary = RotaryEmbedding::new(head_dim, 10_000.0);
            let (cos_dup, sin_dup) = rotary.forward_cos_sin(t, &device, DType::F32).unwrap();
            // rope_i wants unique-frequency (T, head_dim/2) tables; the
            // production tables duplicate every angle pairwise.
            let half = head_dim / 2;
            let uniq = |dup: &Tensor| -> Tensor {
                dup.reshape((1, t, half, 2))
                    .unwrap()
                    .narrow(D::Minus1, 0, 1)
                    .unwrap()
                    .squeeze(D::Minus1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .contiguous()
                    .unwrap()
            };
            let cos_u = uniq(&cos_dup);
            let sin_u = uniq(&sin_dup);

            let shape = |tt: &Tensor| -> Result<Tensor> {
                tt.reshape((b, t, heads, head_dim))?
                    .transpose(1, 2)?
                    .contiguous()
            };
            // Candidate: rotate only the real 64-wide slice with the fused
            // rope_i kernel, reassemble with one cat.
            let rope_new = |src: &Tensor| -> Result<Tensor> {
                let slice = src.narrow(D::Minus1, 0, head_dim)?.contiguous()?;
                let rot = candle_nn::rotary_emb::rope_i(&slice.unsqueeze(1)?, &cos_u, &sin_u)?;
                let rot = rot.squeeze(1)?;
                Tensor::cat(
                    &[&rot, &src.narrow(D::Minus1, head_dim, hidden - head_dim)?],
                    D::Minus1,
                )
            };

            let _ = stage("qkv_gemm_x3", &mut || {
                let q = lin_q.forward(&x)?;
                let k = lin_k.forward(&x)?;
                let v = lin_v.forward(&x)?;
                Tensor::cat(&[&q, &k, &v], 0)
            });

            let q_rot_old = apply_rotary_cos_sin(&q0, &cos_dup, &sin_dup).unwrap();
            let k_rot_old = apply_rotary_cos_sin(&k0, &cos_dup, &sin_dup).unwrap();
            let _ = stage("rope_old_x2", &mut || {
                let q = apply_rotary_cos_sin(&q0, &cos_dup, &sin_dup)?;
                let k = apply_rotary_cos_sin(&k0, &cos_dup, &sin_dup)?;
                Tensor::cat(&[&q, &k], 0)
            });

            // Numeric gate: slice+rope_i reassembly vs the production chain.
            let rel = err_vs(&rope_new(&q0).unwrap(), &q_rot_old).unwrap();
            println!("  gate rope_new-vs-old rel {rel:.3e}");
            assert!(rel <= 1e-6, "rope_new-vs-old rel {rel:.3e}");

            let _ = stage("rope_new_x2", &mut || {
                let q = rope_new(&q0)?;
                let k = rope_new(&k0)?;
                Tensor::cat(&[&q, &k], 0)
            });

            let qh = shape(&q_rot_old).unwrap();
            let kh = shape(&k_rot_old).unwrap();
            let vh = shape(&v0).unwrap();
            let _ = stage("split_x3", &mut || {
                let q = shape(&q_rot_old)?;
                let k = shape(&k_rot_old)?;
                let v = shape(&v0)?;
                Tensor::cat(&[&q, &k, &v], 0)
            });
            let _ = stage("chain_old", &mut || {
                let q = shape(&apply_rotary_cos_sin(&q0, &cos_dup, &sin_dup)?)?;
                let k = shape(&apply_rotary_cos_sin(&k0, &cos_dup, &sin_dup)?)?;
                let v = shape(&v0)?;
                Tensor::cat(&[&q, &k, &v], 0)
            });

            let manual_attn = |mask: Option<&Tensor>| -> Result<Tensor> {
                let s = qh
                    .matmul(&kh.transpose(D::Minus2, D::Minus1)?)?
                    .affine(scale, 0.0)?;
                let s = match mask {
                    Some(m) => s.broadcast_add(m)?,
                    None => s,
                };
                let p = candle_nn::ops::softmax_last_dim(&s)?;
                p.matmul(&vh)
            };
            let _ = stage("attn_manual", &mut || manual_attn(None));

            let keep = chunk_attention_mask(t, chunk, &device).unwrap();
            let mask_bias = keep_mask_to_additive(&keep, DType::F32, &device).unwrap();
            let _ = stage("attn_manual_mask", &mut || manual_attn(Some(&mask_bias)));

            // sdpa gates vs manual before trusting its timings.
            let sdpa_plain = stage("attn_sdpa", &mut || {
                candle_nn::ops::sdpa(&qh, &kh, &vh, None, false, scale as f32, 1.0)
            });
            let rel = err_vs(&sdpa_plain, &manual_attn(None).unwrap()).unwrap();
            println!("  gate sdpa-vs-manual rel {rel:.3e}");
            // The two f32 paths accumulate the softmax denominator in
            // different orders; that rounding floor grows with T (512 terms
            // -> ~2e-6). Semantic errors land at >=1e-2.
            assert!(rel <= 3e-6, "sdpa-vs-manual rel {rel:.3e}");

            let mask_full = mask_bias
                .broadcast_as((b, heads, t, t))
                .unwrap()
                .contiguous()
                .unwrap();
            let sdpa_masked = stage("attn_sdpa_mask", &mut || {
                candle_nn::ops::sdpa(&qh, &kh, &vh, Some(&mask_full), false, scale as f32, 1.0)
            });
            let rel = err_vs(&sdpa_masked, &manual_attn(Some(&mask_bias)).unwrap()).unwrap();
            println!("  gate sdpa_masked-vs-manual rel {rel:.3e}");
            assert!(rel <= 1e-5, "sdpa_masked-vs-manual rel {rel:.3e}");

            let attn_out = sdpa_masked;
            let _ = stage("out_proj", &mut || {
                let o = attn_out.transpose(1, 2)?.contiguous()?.flatten(2, 3)?;
                lin_out.forward(&o)
            });
            let _ = stage("ffn", &mut || {
                let h = gelu_tanh(&lin_f1.forward(&x)?)?;
                lin_f2.forward(&h)
            });

            // Raw transpose-copy bandwidth: the split/out stages above are
            // (B,T,H,D)->(B,H,T,D) materializations; this isolates the copy
            // cost from everything else to size a fused-kernel opportunity.
            let _ = stage("tpose_copy", &mut || qh.transpose(1, 2)?.contiguous());

            // Full production Attention::forward in both modes: the base
            // (broadcastable) mask form selects the reference chain, the
            // expanded [B, heads, T, T] mask the fused Metal chain — exactly
            // the dispatch production code performs.
            let attn_mod = Attention {
                to_q: lin(wq, bq),
                to_k: lin(wk, bk),
                to_v: lin(wv, bv),
                to_out: lin(wo, bo),
                heads,
                head_dim,
                scale,
            };
            let rope_tables = RopeTables::new(&rotary, t, &device, DType::F32).unwrap();
            let old = attn_mod
                .forward(&x, Some(&mask_bias), &rope_tables, None)
                .unwrap();
            let _ = stage("attn_total_old", &mut || {
                attn_mod.forward(&x, Some(&mask_bias), &rope_tables, None)
            });
            let new = stage("attn_total_new", &mut || {
                attn_mod.forward(&x, Some(&mask_full), &rope_tables, None)
            });
            let rel = err_vs(&new, &old).unwrap();
            println!("  gate attn_total new-vs-old rel {rel:.3e}");
            assert!(rel <= 1e-5, "attn_total new-vs-old rel {rel:.3e}");
        }
    }
}
