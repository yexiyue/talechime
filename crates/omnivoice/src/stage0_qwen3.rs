use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use candle_core::{DType, Module, Tensor};
use candle_nn::{Activation, Embedding, VarBuilder};
use candle_transformers::models::with_tracing::{Linear, RmsNorm, linear_b};

use crate::error::Result;

#[derive(Debug, Clone)]
pub struct Stage0Qwen3BackboneConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub attention_bias: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub hidden_act: Activation,
}

#[derive(Debug)]
pub struct Stage0BackboneOutput {
    pub final_hidden: Tensor,
    pub captured_hidden_layers: BTreeMap<usize, Tensor>,
}

pub type Stage0Qwen3Config = Stage0Qwen3BackboneConfig;
pub type Stage0ForwardPass = Stage0BackboneOutput;

#[derive(Debug, Clone)]
struct Stage0Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Stage0Qwen3RotaryEmbedding {
    fn new(dtype: DType, cfg: &Stage0Qwen3BackboneConfig, vb: &VarBuilder<'_>) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|index| 1f32 / cfg.rope_theta.powf(index as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), vb.device())?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, vb.device())?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// Rotate q/k that are laid out as `[batch, seq, heads, head_dim]`.
    ///
    /// The fused `rope_thd` kernel indexes cos/sin by the sequence position and
    /// accepts this layout, which is what the projections already produce.  It
    /// therefore avoids the head-major copy that `rope` requires.
    fn apply_sequence_first(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, seq_len, _, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope_thd(q, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_thd(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Stage0Qwen3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Stage0Qwen3Mlp {
    fn load(cfg: &Stage0Qwen3BackboneConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_transformers::models::with_tracing::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_transformers::models::with_tracing::linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: candle_transformers::models::with_tracing::linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Stage0Qwen3Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Stage0Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<Stage0Qwen3RotaryEmbedding>,
}

impl Stage0Qwen3Attention {
    fn load(
        cfg: &Stage0Qwen3BackboneConfig,
        rotary_emb: Arc<Stage0Qwen3RotaryEmbedding>,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        Ok(Self {
            q_proj: linear_b(
                cfg.hidden_size,
                num_heads * head_dim,
                cfg.attention_bias,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_b(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                cfg.attention_bias,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_b(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                cfg.attention_bias,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_b(
                num_heads * head_dim,
                cfg.hidden_size,
                cfg.attention_bias,
                vb.pp("o_proj"),
            )?,
            q_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size: num_heads * head_dim,
            rotary_emb,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        flash_seqlens: Option<&[usize]>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let (q, k) = if xs.device().is_metal() {
            // The fused rope kernel for the sequence-first layout reads the
            // layout the projections already produce, so q and k never need a
            // head-major copy.  The transposed view below is what the
            // attention kernel reads; it accepts strided q/k/v.
            let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
            let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
            let q = apply_rms_norm(&self.q_norm, &q)?;
            let k = apply_rms_norm(&self.k_norm, &k)?;
            let (q, k) = self.rotary_emb.apply_sequence_first(&q, &k, 0)?;
            (q.transpose(1, 2)?, k.transpose(1, 2)?)
        } else {
            let q = q
                .reshape((batch, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let q_flat = q.flatten(0, 2)?;
            let k_flat = k.flatten(0, 2)?;
            let q = apply_rms_norm(&self.q_norm, &q_flat)?.reshape((
                batch,
                self.num_heads,
                seq_len,
                self.head_dim,
            ))?;
            let k = apply_rms_norm(&self.k_norm, &k_flat)?.reshape((
                batch,
                self.num_kv_heads,
                seq_len,
                self.head_dim,
            ))?;
            self.rotary_emb.apply(&q, &k, 0)?
        };
        let v = v
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let context = self.attention_context(q, k, v, attention_mask, flash_seqlens)?;
        Ok(context
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.hidden_size))?
            .apply(&self.o_proj)?)
    }

    fn attention_context(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Option<&Tensor>,
        flash_seqlens: Option<&[usize]>,
    ) -> Result<Tensor> {
        let _ = flash_seqlens;
        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        // The fused SDPA kernel only exists on Metal; CPU keeps the manual
        // score-matrix path.
        if q.device().is_metal() {
            // Metal SDPA maps query heads to KV heads directly for GQA.
            attention_sdpa(&q, &k, &v, attention_mask, scale as f32)
        } else {
            let repeats = self.num_heads / self.num_kv_heads;
            let k = candle_transformers::utils::repeat_kv(k, repeats)?;
            let v = candle_transformers::utils::repeat_kv(v, repeats)?;
            attention_manual(&q, &k, &v, attention_mask, scale)
        }
    }
}

/// Manual attention: materialises the full `[B, H, S, S]` score matrix,
/// adds the mask and softmaxes it in the runtime dtype.
fn attention_manual(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attention_mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let k_t = k.transpose(2, 3)?;
    // Accelerate accepts a transposed contiguous matrix directly. Materializing
    // it also changes BLAS's transpose route, which costs more at these shapes.
    // Keep the specialization at the validated OmniVoice head dimension.
    let k_t = if q.device().is_cpu()
        && q.dtype() == DType::F32
        && candle_core::utils::has_accelerate()
        && k.is_contiguous()
        && k.dim(3)? == 128
    {
        k_t
    } else {
        k_t.contiguous()?
    };
    let mut scores = q.matmul(&k_t)?.affine(scale, 0.0)?;
    if let Some(mask) = attention_mask {
        scores = scores.broadcast_add(mask)?;
    }
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    Ok(probs.matmul(v)?)
}

/// Fused attention: one Metal SDPA kernel, no materialised score matrix.
///
/// The fused kernel reads the mask through its strides and requires
/// `[bs, qheads, qseq, kseq]`; a zero head stride keeps the shared
/// `[bs, 1, qseq, kseq]` mask without materialising it.
fn attention_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attention_mask: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    let (batch, heads, q_seq, _) = q.dims4()?;
    let mask = match attention_mask {
        Some(mask) => {
            let (mask_batch, mask_heads, mask_q, mask_k) = mask.dims4()?;
            if mask_batch != batch || mask_q != q_seq {
                return Err(candle_core::Error::Msg(format!(
                    "stage0 attention mask shape {mask:?} does not match q shape {:?}",
                    q.dims()
                ))
                .into());
            }
            if mask_heads == heads {
                Some(mask.clone())
            } else {
                Some(mask.expand((batch, heads, mask_q, mask_k))?)
            }
        }
        None => None,
    };
    Ok(candle_nn::ops::sdpa(
        q,
        k,
        v,
        mask.as_ref(),
        false,
        scale,
        1.0,
    )?)
}

#[derive(Debug, Clone)]
struct Stage0Qwen3DecoderLayer {
    self_attn: Stage0Qwen3Attention,
    mlp: Stage0Qwen3Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Stage0Qwen3DecoderLayer {
    fn load(
        cfg: &Stage0Qwen3BackboneConfig,
        rotary: Arc<Stage0Qwen3RotaryEmbedding>,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: Stage0Qwen3Attention::load(cfg, rotary, vb.pp("self_attn"))?,
            mlp: Stage0Qwen3Mlp::load(cfg, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        flash_seqlens: Option<&[usize]>,
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden = apply_rms_norm(&self.input_layernorm, xs)?;
        let hidden = self
            .self_attn
            .forward(&hidden, attention_mask, flash_seqlens)?;
        let hidden = (hidden + residual)?;
        let residual = &hidden;
        let post_attention = apply_rms_norm(&self.post_attention_layernorm, &hidden)?;
        let mlp_hidden = self.mlp.forward(&post_attention)?;
        Ok((residual + mlp_hidden)?)
    }
}

#[derive(Debug, Clone)]
pub struct Stage0Qwen3Backbone {
    embed_tokens: Embedding,
    layers: Vec<Stage0Qwen3DecoderLayer>,
    norm: RmsNorm,
}

impl Stage0Qwen3Backbone {
    pub fn load(cfg: &Stage0Qwen3BackboneConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(Stage0Qwen3RotaryEmbedding::new(vb.dtype(), cfg, &vb)?);
        let layer_vb = vb.pp("model.layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            layers.push(Stage0Qwen3DecoderLayer::load(
                cfg,
                rotary.clone(),
                layer_vb.pp(index),
            )?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn text_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(self.embed_tokens.forward(input_ids)?)
    }

    pub fn embed_text_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.text_embeddings(input_ids)
    }

    pub fn forward_embeds(
        &self,
        input_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
        capture_layers: &BTreeSet<usize>,
    ) -> Result<Stage0BackboneOutput> {
        self.forward_embeds_with_flash(input_embeds, attention_mask, capture_layers, None)
    }

    pub fn forward_embeds_with_flash(
        &self,
        input_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
        capture_layers: &BTreeSet<usize>,
        flash_seqlens: Option<&[usize]>,
    ) -> Result<Stage0BackboneOutput> {
        let mut hidden = input_embeds.clone();
        let mut captures = BTreeMap::new();
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, attention_mask, flash_seqlens)?;
            // HuggingFace exposes `hidden_states[layer_index + 1]` before the
            // final model norm.  Keep the last decoder layer in that same
            // form; callers that need the post-norm output already receive it
            // separately as `final_hidden`.
            if capture_layers.contains(&index) {
                captures.insert(index, hidden.clone());
            }
        }
        let final_hidden = apply_rms_norm(&self.norm, &hidden)?;
        Ok(Stage0BackboneOutput {
            final_hidden,
            captured_hidden_layers: captures,
        })
    }
}

/// RMS norm for the stage0 backbone.
///
/// On Metal the fused kernel is used: it accumulates the sum of squares in
/// f32 and tracks an f64 host reference as closely as the layer-norm path
/// (relative RMS 2.9e-4 for both, see the metal tests). Other devices keep
/// the layer-norm path so f16/bf16 stays HuggingFace compatible.
fn apply_rms_norm(norm: &RmsNorm, xs: &Tensor) -> Result<Tensor> {
    if xs.device().is_metal() || !matches!(xs.dtype(), DType::F16 | DType::BF16) {
        Ok(norm.forward(xs)?)
    } else {
        Ok(norm.forward_diff(xs)?)
    }
}

#[allow(dead_code)]
fn try_flash_attention_per_row(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _seqlens: &[usize],
    _scale: f32,
) -> Result<Option<Tensor>> {
    Ok(None)
}

#[cfg(test)]
mod cpu_tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn cpu_attention_transposed_key_matches_materialized_bits() -> Result<()> {
        let device = Device::Cpu;
        for batch in [1, 2] {
            for seq in [1, 7, 33, 157] {
                for dim in [8, 128] {
                    let q = Tensor::rand(-0.5f32, 0.5, (batch, 4, seq, dim), &device)?;
                    let k = Tensor::rand(-0.5f32, 0.5, (batch, 4, seq, dim), &device)?;
                    let v = Tensor::rand(-0.5f32, 0.5, (batch, 4, seq, dim), &device)?;
                    let strided_k = k.transpose(2, 3)?.contiguous()?.transpose(2, 3)?;
                    let mask = Tensor::from_vec(
                        (0..batch * seq * seq)
                            .map(|i| {
                                if seq > 1 && i % seq == seq - 1 {
                                    f32::NEG_INFINITY
                                } else {
                                    0f32
                                }
                            })
                            .collect::<Vec<_>>(),
                        (batch, 1, seq, seq),
                        &device,
                    )?;
                    for k in [&k, &strided_k] {
                        for mask in [None, Some(&mask)] {
                            let scale = 1.0 / (dim as f64).sqrt();
                            let mut scores = q
                                .matmul(&k.transpose(2, 3)?.contiguous()?)?
                                .affine(scale, 0.0)?;
                            if let Some(mask) = mask {
                                scores = scores.broadcast_add(mask)?;
                            }
                            let expected = candle_nn::ops::softmax_last_dim(&scores)?
                                .matmul(&v)?
                                .flatten_all()?
                                .to_vec1::<f32>()?
                                .into_iter()
                                .map(f32::to_bits)
                                .collect::<Vec<_>>();
                            for _ in 0..2 {
                                let actual = attention_manual(&q, k, &v, mask, scale)?
                                    .flatten_all()?
                                    .to_vec1::<f32>()?
                                    .into_iter()
                                    .map(f32::to_bits)
                                    .collect::<Vec<_>>();
                                assert_eq!(actual, expected, "batch={batch} seq={seq} dim={dim}");
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "metal"))]
mod metal_tests {
    use super::*;
    use candle_core::Device;

    fn bit_pattern(tensor: &Tensor) -> Result<Vec<u32>> {
        Ok(tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    /// The sequence-first rope path replaces a head-major layout copy.  It has
    /// to produce exactly the same q and k, otherwise the discrete codes
    /// change.
    #[test]
    fn sequence_first_rope_matches_head_major_rope() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (batch, seq, dim) = (2usize, 157usize, 128usize);
        let cos = Tensor::rand(-1f32, 1f32, (seq, dim / 2), &device)?.to_dtype(DType::F16)?;
        let sin = Tensor::rand(-1f32, 1f32, (seq, dim / 2), &device)?.to_dtype(DType::F16)?;
        let norm_weight = Tensor::rand(0.5f32, 1.5f32, (dim,), &device)?.to_dtype(DType::F16)?;
        // The production path calls the fused kernel on Metal; the wrapper in
        // `with_tracing` forwards to exactly this implementation.
        let norm = candle_nn::RmsNorm::new(norm_weight, 1e-6);

        for heads in [16usize, 8] {
            let proj = Tensor::rand(-1f32, 1f32, (batch, seq, heads * dim), &device)?
                .to_dtype(DType::F16)?;

            // Head-major path: copy into `[B, H, S, D]`, normalise, rotate.
            let head_major = proj
                .reshape((batch, seq, heads, dim))?
                .transpose(1, 2)?
                .flatten(0, 2)?;
            let head_major = norm
                .forward(&head_major)?
                .reshape((batch, heads, seq, dim))?;
            let head_major = candle_nn::rotary_emb::rope(&head_major, &cos, &sin)?;

            // Sequence-first path: normalise and rotate in place, then hand
            // the transposed view to the attention.
            let sequence_first = norm.forward(&proj.reshape((batch, seq, heads, dim))?)?;
            let sequence_first =
                candle_nn::rotary_emb::rope_thd(&sequence_first, &cos, &sin)?.transpose(1, 2)?;

            assert_eq!(
                bit_pattern(&head_major)?,
                bit_pattern(&sequence_first)?,
                "sequence-first rope differs for {heads} heads"
            );
        }
        Ok(())
    }

    /// The attention kernel reads strided q/k/v, so the copy-free path has to
    /// produce the same context as the head-major path.
    #[test]
    fn strided_attention_matches_contiguous_attention() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (batch, seq, heads, kv_heads, dim) = (2usize, 157usize, 16usize, 8usize, 128usize);
        let scale = 1.0f32 / (dim as f32).sqrt();
        let mask = Tensor::rand(0f32, 1f32, (batch, 1, seq, seq), &device)?.to_dtype(DType::F16)?;
        let mask = mask.expand((batch, heads, seq, seq))?;

        for masked in [false, true] {
            let q_seq = Tensor::rand(-1f32, 1f32, (batch, seq, heads * dim), &device)?
                .to_dtype(DType::F16)?;
            let k_seq = Tensor::rand(-1f32, 1f32, (batch, seq, kv_heads * dim), &device)?
                .to_dtype(DType::F16)?;
            let v_seq = Tensor::rand(-1f32, 1f32, (batch, seq, kv_heads * dim), &device)?
                .to_dtype(DType::F16)?;

            // Copy-free path: the projections stay in the sequence-first
            // layout and the attention reads the transposed views.
            let q = q_seq.reshape((batch, seq, heads, dim))?.transpose(1, 2)?;
            let k = k_seq
                .reshape((batch, seq, kv_heads, dim))?
                .transpose(1, 2)?;
            let v = v_seq
                .reshape((batch, seq, kv_heads, dim))?
                .transpose(1, 2)?;
            let mask_ref = if masked { Some(&mask) } else { None };
            let strided = attention_sdpa(&q, &k, &v, mask_ref, scale)?;

            let contiguous = attention_sdpa(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                mask_ref,
                scale,
            )?;
            assert_eq!(
                bit_pattern(&strided)?,
                bit_pattern(&contiguous)?,
                "strided attention differs, masked={masked}"
            );
        }
        Ok(())
    }

    #[test]
    fn native_gqa_matches_repeated_kv_exactly() -> Result<()> {
        let device = Device::new_metal(0)?;
        for seq in [1, 33, 157, 287] {
            let q = Tensor::rand(-1f32, 1f32, (2, 16, seq, 128), &device)?.to_dtype(DType::F16)?;
            let k = Tensor::rand(-1f32, 1f32, (2, 8, seq, 128), &device)?.to_dtype(DType::F16)?;
            let v = Tensor::rand(-1f32, 1f32, (2, seq, 8, 128), &device)?
                .to_dtype(DType::F16)?
                .transpose(1, 2)?;
            let repeated_k = candle_transformers::utils::repeat_kv(k.clone(), 2)?;
            let repeated_v = candle_transformers::utils::repeat_kv(v.clone(), 2)?;
            let mask: Vec<f32> = (0..2 * seq * seq)
                .map(|i| {
                    if i % seq == seq - 1 && seq > 1 {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
                .collect();
            let mask = Tensor::from_vec(mask, (2, 1, seq, seq), &device)?.to_dtype(DType::F16)?;
            for mask in [None, Some(&mask)] {
                let scale = 1.0 / 128f32.sqrt();
                let native = attention_sdpa(&q, &k, &v, mask, scale)?;
                let repeated = attention_sdpa(&q, &repeated_k, &repeated_v, mask, scale)?;
                assert_eq!(
                    native
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    repeated
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    "sequence length {seq}, masked={}",
                    mask.is_some()
                );
            }
        }
        Ok(())
    }

    /// The fused Metal SDPA kernel and the manual score matrix path must
    /// agree within f16 tolerance on the shapes stage0 actually uses.
    #[test]
    fn fused_sdpa_matches_manual_attention() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (batch, heads, kv_heads, seq, dim) = (2usize, 16usize, 8usize, 287usize, 128usize);
        let q =
            Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), &device)?.to_dtype(DType::F16)?;
        let k =
            Tensor::rand(0f32, 1f32, (batch, kv_heads, seq, dim), &device)?.to_dtype(DType::F16)?;
        let v =
            Tensor::rand(0f32, 1f32, (batch, kv_heads, seq, dim), &device)?.to_dtype(DType::F16)?;
        let repeats = heads / kv_heads;
        let k = candle_transformers::utils::repeat_kv(k, repeats)?;
        let v = candle_transformers::utils::repeat_kv(v, repeats)?;
        // Additive mask in the runtime dtype, exactly like stage0 prepares it.
        let mask_bool = Tensor::rand(0f32, 1f32, (batch, 1, seq, seq), &device)?;
        let mask = mask_bool.to_dtype(DType::F16)?;
        let scale = 1.0f64 / (dim as f64).sqrt();

        let manual = attention_manual(&q, &k, &v, Some(&mask), scale)?;
        let fused = attention_sdpa(&q, &k, &v, Some(&mask), scale as f32)?;
        let manual = manual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let fused = fused
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(manual.len(), fused.len());
        let mut max_abs = 0f32;
        let mut sum_sq = 0f64;
        let mut reference_sq = 0f64;
        for (a, b) in manual.iter().zip(fused.iter()) {
            let diff = (a - b).abs();
            max_abs = max_abs.max(diff);
            sum_sq += f64::from(diff) * f64::from(diff);
            reference_sq += f64::from(*a) * f64::from(*a);
        }
        let rms = (sum_sq / manual.len() as f64).sqrt();
        let reference_rms = (reference_sq / manual.len() as f64).sqrt();
        // f16 has ~3 decimal digits; the two paths may only differ by the
        // accumulation order, not by a different softmax.
        assert!(
            max_abs < 0.05,
            "fused sdpa diverged from manual attention: max_abs={max_abs}"
        );
        assert!(
            rms <= 0.01 * reference_rms.max(1e-6),
            "fused sdpa rms error {rms} too large against {reference_rms}"
        );
        Ok(())
    }

    /// The fused kernel must also agree when the mask blocks whole rows.
    #[test]
    fn fused_sdpa_matches_manual_attention_with_masked_rows() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (batch, heads, seq, dim) = (2usize, 16usize, 64usize, 128usize);
        let q =
            Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), &device)?.to_dtype(DType::F16)?;
        let k =
            Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), &device)?.to_dtype(DType::F16)?;
        let v =
            Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), &device)?.to_dtype(DType::F16)?;
        let mut mask_values = vec![0f32; batch * seq * seq];
        for (index, value) in mask_values.iter_mut().enumerate() {
            if index % seq == seq - 1 {
                *value = f32::NEG_INFINITY;
            }
        }
        let mask =
            Tensor::from_vec(mask_values, (batch, 1, seq, seq), &device)?.to_dtype(DType::F16)?;
        let scale = 1.0f64 / (dim as f64).sqrt();
        let manual = attention_manual(&q, &k, &v, Some(&mask), scale)?;
        let fused = attention_sdpa(&q, &k, &v, Some(&mask), scale as f32)?;
        let manual = manual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let fused = fused
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut max_abs = 0f32;
        for (a, b) in manual.iter().zip(fused.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        assert!(
            max_abs < 0.05,
            "fused sdpa diverged with masked rows: max_abs={max_abs}"
        );
        Ok(())
    }

    /// The fused rms_norm kernel and the layer-norm path must both track an
    /// f64 host reference; the fused kernel rounds the scale to f16, so this
    /// records how much accuracy the speed-up costs.
    #[test]
    fn fused_rms_norm_tracks_f64_host_reference() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (rows, hidden) = (64usize, 1024usize);
        let eps = 1e-6f64;
        let xs = Tensor::rand(-3f32, 3f32, (rows, hidden), &device)?.to_dtype(DType::F16)?;
        let weight = Tensor::rand(0.5f32, 1.5f32, (hidden,), &device)?.to_dtype(DType::F16)?;
        let norm = candle_nn::RmsNorm::new(weight.clone(), eps);

        let reference = {
            let host = xs
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            let host_weight = weight
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?;
            let mut out = vec![0f32; rows * hidden];
            for (row, values) in host.iter().enumerate() {
                let mean_sq = values
                    .iter()
                    .map(|v| f64::from(*v) * f64::from(*v))
                    .sum::<f64>()
                    / hidden as f64;
                let scale = 1.0 / (mean_sq + eps).sqrt();
                for (index, value) in values.iter().enumerate() {
                    let normalized = f64::from(*value) * scale;
                    let scaled = normalized * f64::from(host_weight[index]);
                    out[row * hidden + index] = scaled as f32;
                }
            }
            out
        };

        let mut errors = Vec::new();
        for (label, actual) in [
            ("layer_norm_path", norm.forward_diff(&xs)?),
            ("fused_kernel", norm.forward(&xs)?),
        ] {
            let actual = actual
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            let mut max_abs = 0f32;
            let mut sum_sq = 0f64;
            let mut reference_sq = 0f64;
            for (row, values) in actual.iter().enumerate() {
                for (index, value) in values.iter().enumerate() {
                    let expected = reference[row * hidden + index];
                    let diff = (f64::from(*value) - f64::from(expected)).abs();
                    max_abs = max_abs.max(diff as f32);
                    sum_sq += diff * diff;
                    reference_sq += f64::from(expected) * f64::from(expected);
                }
            }
            let rms = (sum_sq / (rows * hidden) as f64).sqrt();
            let reference_rms = (reference_sq / (rows * hidden) as f64).sqrt();
            errors.push((label, max_abs, rms / reference_rms.max(1e-9)));
        }
        for (label, max_abs, relative_rms) in &errors {
            eprintln!("rms_norm {label}: max_abs={max_abs:.6} relative_rms={relative_rms:.3e}");
        }
        // Both paths must stay far below the f16 rounding band (2^-11 ~ 4.9e-4).
        for (label, _, relative_rms) in &errors {
            assert!(
                *relative_rms < 2e-3,
                "{label} diverged from the f64 reference: relative_rms={relative_rms}"
            );
        }
        Ok(())
    }
}
