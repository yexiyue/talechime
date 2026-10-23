use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use candle_core::{DType, Module, Tensor};
use candle_nn::{Activation, Embedding, VarBuilder};
use candle_transformers::models::with_tracing::{Linear, RmsNorm, linear_b};

use crate::error::Result;
use crate::stage0_cpu_linear::CpuProjection;

#[cfg(test)]
#[path = "stage0_qwen3/cpu_attribution.rs"]
mod cpu_attribution;

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
    gate_up_proj: Option<Linear>,
    down_proj: Linear,
    act_fn: Activation,
}

impl Stage0Qwen3Mlp {
    fn load(cfg: &Stage0Qwen3BackboneConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let gate =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let (gate_proj, up_proj, gate_up_proj) = if vb.device().is_cpu()
            && vb.dtype() == DType::F32
            && candle_core::utils::has_accelerate()
            && matches!(cfg.hidden_act, Activation::Silu)
            && (cfg.hidden_size, cfg.intermediate_size) == (1024, 3072)
            && !gate.weight().is_variable()
            && !up.weight().is_variable()
            && gate.weight().is_contiguous()
            && up.weight().is_contiguous()
        {
            let weight = Tensor::cat(&[gate.weight(), up.weight()], 0)?;
            // The fallback projections share this buffer; do not retain a second layout.
            (
                Linear::from_weights(weight.narrow(0, 0, cfg.intermediate_size)?, None),
                Linear::from_weights(
                    weight.narrow(0, cfg.intermediate_size, cfg.intermediate_size)?,
                    None,
                ),
                Some(Linear::from_weights(weight, None)),
            )
        } else {
            (
                Linear::from_weights(gate.weight().clone(), None),
                Linear::from_weights(up.weight().clone(), None),
                None,
            )
        };
        Ok(Self {
            gate_proj,
            up_proj,
            gate_up_proj,
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
        let (lhs, rhs) = if let Some(proj) = &self.gate_up_proj
            && xs.device().is_cpu()
            && xs.dtype() == DType::F32
            && xs.is_contiguous()
            && xs.rank() == 3
            && xs.dim(1)? >= 8
        {
            let both = xs.apply(proj)?;
            let intermediate = both.dim(candle_core::D::Minus1)? / 2;
            (
                both.narrow(candle_core::D::Minus1, 0, intermediate)?
                    .apply(&self.act_fn)?,
                both.narrow(candle_core::D::Minus1, intermediate, intermediate)?,
            )
        } else {
            (
                xs.apply(&self.gate_proj)?.apply(&self.act_fn)?,
                xs.apply(&self.up_proj)?,
            )
        };
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Stage0Qwen3Attention {
    q_proj: Linear,
    k_proj: CpuProjection,
    v_proj: CpuProjection,
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
            k_proj: CpuProjection::load(
                cfg.hidden_size,
                num_kv_heads * head_dim,
                cfg.attention_bias,
                vb.pp("k_proj"),
            )?,
            v_proj: CpuProjection::load(
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
            return attention_sdpa(&q, &k, &v, attention_mask, scale as f32);
        }
        if let Some(context) = cpu_grouped_attention(&q, &k, &v, attention_mask, scale)? {
            return Ok(context);
        }
        let repeats = self.num_heads / self.num_kv_heads;
        let k = candle_transformers::utils::repeat_kv(k, repeats)?;
        let v = candle_transformers::utils::repeat_kv(v, repeats)?;
        attention_manual(&q, &k, &v, attention_mask, scale)
    }
}

fn cpu_grouped_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Option<Tensor>> {
    if !q.device().is_cpu()
        || q.dtype() != DType::F32
        || !candle_core::utils::has_accelerate()
        || !q.is_contiguous()
        || !k.is_contiguous()
    {
        return Ok(None);
    }
    let (batch, heads, seq, dim) = q.dims4()?;
    let (kb, kv_heads, ks, kd) = k.dims4()?;
    // Small M can select a numerically different BLAS route. Keep those
    // shapes and configurations outside the validated 2:1 GQA path unchanged.
    if seq < 8
        || dim != 128
        || heads != 2 * kv_heads
        || (kb, ks, kd) != (batch, seq, dim)
        || v.dims() != k.dims()
    {
        return Ok(None);
    }
    // Consecutive query heads share a KV head. Folding them into the row
    // dimension avoids KV repetition and halves the number of BLAS calls.
    let grouped_q = q.reshape((batch, kv_heads, 2 * seq, dim))?;
    let mut scores = grouped_q
        .matmul(&k.transpose(2, 3)?)?
        .affine(scale, 0.0)?
        .reshape((batch, heads, seq, seq))?;
    if let Some(mask) = mask {
        scores = scores.broadcast_add(mask)?;
    }
    let context = candle_nn::ops::softmax_last_dim(&scores)?
        .reshape((batch, kv_heads, 2 * seq, seq))?
        .matmul(&v.contiguous()?)?
        .reshape((batch, heads, seq, dim))?;
    Ok(Some(context))
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
        #[cfg(test)]
        let trace_step = cpu_attribution::begin_forward();
        let mut hidden = input_embeds.clone();
        let mut captures = BTreeMap::new();
        for (index, layer) in self.layers.iter().enumerate() {
            #[cfg(test)]
            let trace_input = hidden.clone();
            hidden = layer.forward(&hidden, attention_mask, flash_seqlens)?;
            #[cfg(test)]
            cpu_attribution::capture_layer(
                trace_step,
                index,
                &trace_input,
                &hidden,
                attention_mask,
            )?;
            // HuggingFace exposes `hidden_states[layer_index + 1]` before the
            // final model norm.  Keep the last decoder layer in that same
            // form; callers that need the post-norm output already receive it
            // separately as `final_hidden`.
            if capture_layers.contains(&index) {
                captures.insert(index, hidden.clone());
            }
        }
        let final_hidden = apply_rms_norm(&self.norm, &hidden)?;
        #[cfg(test)]
        cpu_attribution::capture_head(trace_step, &final_hidden)?;
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

#[cfg(test)]
#[path = "stage0_qwen3/cpu_tests.rs"]
mod cpu_tests;

#[cfg(all(test, feature = "metal"))]
#[path = "stage0_qwen3/metal_tests.rs"]
mod metal_tests;
