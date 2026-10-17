//! Conformer encoder (`indextts/gpt/conformer_encoder.py` +
//! `gpt/conformer/{attention,embedding,subsampling}.py`) with `conv2d2`
//! input subsampling, used for the emotion conditioning path.
//!
//! Ported with the exact ESPnet-style wiring the upstream sources use:
//! `ConformerEncoder` defaults — `pos_enc_layer_type="rel_pos"` (so
//! `RelPositionMultiHeadedAttention`), `normalize_before=true`,
//! `concat_after=false`, macaron FF disabled, convolution module enabled
//! with `cnn_module_kernel=15`, SiLU activations and `LayerNorm(eps=1e-5)`.
//! Two deliberate upstream quirks are preserved verbatim:
//! - `RelPositionMultiHeadedAttention` in this fork comments out
//!   `rel_shift`, so `matrix_bd` uses the raw `q_with_bias_v @ p^T` bias
//!   against the plain `(1, T, D)` positional table (`pe[0:T]`, not the
//!   classic doubled ESPnet relative table). The table itself comes from
//!   the checkpoint's `embed.pos_enc.pe` buffer (`RelPositionalEncoding`
//!   `register_buffer`s it, so `load_state_dict` overwrites the analytic
//!   formula; the shipped 2.5 table is stored at low precision and must
//!   not be recomputed) — see [`ConformerEncoder::pos_encoding`].
//! - `Conv2dSubsampling2` slices the mask with `[:, :, 2::2]`, i.e.
//!   subsampled position `i` is valid when input frame `2i + 2` is valid;
//!   the subsampled length is `(T - 3) // 2 + 1 == (T - 1) / 2`.

use candle_core::{Device, Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder, conv1d,
    conv2d, layer_norm, linear, linear_no_bias,
};

/// Layer norm epsilon used by every norm in the encoder (explicit upstream).
const LN_EPS: f64 = 1e-5;

/// Convolution kernel of the conformer depthwise module (upstream default).
const CNN_MODULE_KERNEL: usize = 15;

pub struct ConformerEncoder {
    /// `embed.conv.0` — `Conv2d(1, output_size, 3, stride=2)` followed by
    /// ReLU (the activation itself has no weights).
    embed_conv: Conv2d,
    /// `embed.out` — `Linear(output_size * ((input_size - 1) / 2), output_size)`.
    embed_out: Linear,
    encoders: Vec<ConformerEncoderLayer>,
    /// `after_norm` — applied last because `normalize_before` is true.
    after_norm: LayerNorm,
    /// `embed.pos_enc.pe` — the positional table shipped in the checkpoint
    /// (`(1, 5000, output_size)` buffer; `RelPositionalEncoding` registers
    /// it, so `load_state_dict` overwrites the analytic formula with it).
    /// The 2.5 gpt.pth table is stored at low precision (multiples of
    /// 1/256, up to ~2e-3 off the exact sin/cos values), so it MUST be
    /// loaded verbatim or the relative-position attention bias diverges.
    /// `None` -> fall back to the analytic table (unit-test VarMaps).
    pos_table: Option<Tensor>,
    output_size: usize,
}

impl ConformerEncoder {
    /// Weight names verbatim under e.g. `emo_conditioning_encoder.*`.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VarBuilder,
        input_size: usize,
        output_size: usize,
        linear_units: usize,
        attention_heads: usize,
        num_blocks: usize,
        input_layer: &str,
    ) -> Result<Self> {
        if input_layer != "conv2d2" {
            candle_core::bail!(
                "unsupported conformer input_layer {input_layer:?} (only conv2d2 is shipped)"
            );
        }
        if !output_size.is_multiple_of(attention_heads) {
            candle_core::bail!(
                "output_size {output_size} not divisible by {attention_heads} heads"
            );
        }
        if input_size < 3 {
            candle_core::bail!("conv2d2 subsampling needs input_size >= 3 (got {input_size})");
        }
        let embed_vb = vb.pp("embed");
        let embed_conv = conv2d(
            1,
            output_size,
            3,
            Conv2dConfig {
                stride: 2,
                ..Default::default()
            },
            embed_vb.pp("conv").pp("0"),
        )?;
        let out_features = output_size * ((input_size - 1) / 2);
        let embed_out = linear(out_features, output_size, embed_vb.pp("out").pp("0"))?;

        let mut encoders = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            encoders.push(ConformerEncoderLayer::load(
                &vb.pp("encoders").pp(i.to_string()),
                output_size,
                linear_units,
                attention_heads,
            )?);
        }
        let after_norm = layer_norm(output_size, LN_EPS, vb.pp("after_norm"))?;
        let pos_table = vb.get((1, 5000, output_size), "embed.pos_enc.pe").ok(); // no checkpoint buffer -> analytic table
        Ok(Self {
            embed_conv,
            embed_out,
            encoders,
            after_norm,
            pos_table,
            output_size,
        })
    }

    /// `pe[:, :t_sub]` `(1, t_sub, output_size)`: the checkpoint's
    /// `embed.pos_enc.pe` buffer when present (the shipped table is a
    /// low-precision copy that differs from the exact formula), else the
    /// analytic sin/cos table.
    fn pos_encoding(&self, t_sub: usize, device: &Device) -> Result<Tensor> {
        match &self.pos_table {
            Some(t) => {
                let max_len = t.dim(1)?;
                if max_len < t_sub {
                    candle_core::bail!("positional table length {max_len} < {t_sub}");
                }
                Ok(t.narrow(1, 0, t_sub)?.contiguous()?)
            }
            None => rel_positional_encoding_table(t_sub, self.output_size, device)?
                .unsqueeze(0)?
                .contiguous(),
        }
    }

    /// `x_pad`: `(B, T, input_size)` padded; `ilens`: `(B,)` i64 true lengths.
    /// Returns `(encoder_out (B, T', output_size), masks (B, 1, T') bool)`
    /// following `ConformerEncoder.forward`.
    pub fn forward(&self, x_pad: &Tensor, ilens: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, t, _) = x_pad.dims3()?;
        if t < 3 {
            candle_core::bail!("conv2d2 subsampling needs T >= 3 (got {t})");
        }
        let lens = ilens
            .flatten_all()?
            .to_dtype(candle_core::DType::I64)?
            .to_vec1::<i64>()?;
        if lens.len() != b {
            candle_core::bail!("expected {b} lengths, got {}", lens.len());
        }

        // embed: (B, 1, T, F) -> conv2d stride 2 + ReLU -> (B, T', output_size)
        let x = x_pad.unsqueeze(1)?.to_dtype(candle_core::DType::F32)?;
        let x = self.embed_conv.forward(&x)?.relu()?;
        let (b, c, t_sub, f) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .reshape((b, t_sub, c * f))?
            .contiguous()?;
        let x = self.embed_out.forward(&x)?;

        // RelPositionalEncoding: x = x * sqrt(d_model); pos_emb = pe[:, :T'].
        let xscale = (self.output_size as f64).sqrt();
        let x = (x * xscale)?;
        let pos_emb = self.pos_encoding(t_sub, x.device())?;

        // Subsampled masks following `x_mask[:, :, 2::2]`: position i is
        // valid when input frame 2i + 2 is within the true length.
        let mask_vals: Vec<u8> = (0..b)
            .flat_map(|bi| {
                let len = lens[bi] as usize;
                (0..t_sub).map(move |i| u8::from(2 * i + 2 < len))
            })
            .collect();
        let masks = Tensor::from_vec(mask_vals, (b, 1, t_sub), x.device())?;

        let mut xs = x;
        for layer in &self.encoders {
            xs = layer.forward(&xs, &masks, &pos_emb)?;
        }
        let xs = self.after_norm.forward(&xs)?;
        Ok((xs, masks))
    }

    /// Test-only mirror of [`Self::forward`] that records the same stage
    /// tensors the Python hook dump (`emo_layers.json`) captures: the embed
    /// output (post `x * sqrt(d)`), each block's self-attn / conv-module /
    /// feed-forward sub-outputs and full block output, and the
    /// `after_norm` output.
    #[cfg(test)]
    pub(crate) fn forward_stage_dump(
        &self,
        x_pad: &Tensor,
        ilens: &Tensor,
    ) -> Result<(Vec<(String, Tensor)>, Tensor)> {
        let (_, _, _) = x_pad.dims3()?;
        let lens = ilens
            .flatten_all()?
            .to_dtype(candle_core::DType::I64)?
            .to_vec1::<i64>()?;
        let conv0 = self
            .embed_conv
            .forward(&x_pad.unsqueeze(1)?.to_dtype(candle_core::DType::F32)?)?;
        let x = conv0.relu()?;
        let (b, c, t_sub, f) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .reshape((b, t_sub, c * f))?
            .contiguous()?;
        let x_lin = self.embed_out.forward(&x)?;
        let x = (x_lin.clone() * (self.output_size as f64).sqrt())?;

        let pos_emb = self.pos_encoding(t_sub, x.device())?;

        let mask_vals: Vec<u8> = (0..b)
            .flat_map(|bi| {
                let len = lens[bi] as usize;
                (0..t_sub).map(move |i| u8::from(2 * i + 2 < len))
            })
            .collect();
        let masks = Tensor::from_vec(mask_vals, (b, 1, t_sub), x.device())?;

        let mut stages = vec![
            ("embed.conv0".to_string(), conv0),
            ("embed.out0".to_string(), x_lin),
            ("embed".to_string(), x.clone()),
            ("b0.pos_emb".to_string(), pos_emb.clone()),
        ];
        let mut xs = x;
        for (i, layer) in self.encoders.iter().enumerate() {
            if i == 0 {
                let h = layer.norm_mha.forward(&xs)?;
                stages.push(("b0.norm_mha".to_string(), h.clone()));
                stages.extend(
                    layer
                        .self_attn
                        .forward_stage_dump(&h, &masks, &pos_emb)?
                        .into_iter()
                        .map(|(n, t)| (format!("b0.att.{n}"), t)),
                );
            }
            let (attn, conv, ff, block) = layer.forward_stage_dump(&xs, &masks, &pos_emb)?;
            stages.push((format!("block{i}.self_attn"), attn));
            stages.push((format!("block{i}.conv_module"), conv));
            stages.push((format!("block{i}.feed_forward"), ff));
            stages.push((format!("block{i}"), block.clone()));
            xs = block;
        }
        let out = self.after_norm.forward(&xs)?;
        stages.push(("after_norm".to_string(), out.clone()));
        Ok((stages, masks))
    }
}

/// The `pe` table rows `[0, size)` of `PositionalEncoding` /
/// `RelPositionalEncoding` (`max_len` buffer trimmed to the needed rows):
/// `pe[pos, 2i] = sin(pos / 10000^(2i/d))`, `pe[pos, 2i+1] = cos(...)`.
fn rel_positional_encoding_table(size: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    let half = d_model / 2;
    let positions = Tensor::arange(0f32, size as f32, device)?
        .unsqueeze(1)?
        .contiguous()?;
    let div_term: Vec<f32> = (0..half)
        .map(|i| (((2 * i) as f64) * (-(10000f64).ln() / d_model as f64)).exp() as f32)
        .collect();
    let div_term = Tensor::from_vec(div_term, (1, half), device)?;
    let angles = positions.matmul(&div_term)?;
    // Interleave [sin, cos] per dimension pair.
    Tensor::stack(&[angles.sin()?, angles.cos()?], 2)?.reshape((size, d_model))
}

struct ConformerEncoderLayer {
    self_attn: RelPositionMultiHeadedAttention,
    /// `norm_ff` — feed-forward module norm.
    norm_ff: LayerNorm,
    /// `norm_mha` — self-attention module norm.
    norm_mha: LayerNorm,
    /// `norm_conv` — convolution module norm.
    norm_conv: LayerNorm,
    /// `norm_final` — final norm of the block (conv module present).
    norm_final: LayerNorm,
    feed_forward: PositionwiseFeedForward,
    conv_module: ConvolutionModule,
}

impl ConformerEncoderLayer {
    fn load(
        vb: &VarBuilder,
        size: usize,
        linear_units: usize,
        attention_heads: usize,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: RelPositionMultiHeadedAttention::load(
                &vb.pp("self_attn"),
                size,
                attention_heads,
            )?,
            norm_ff: layer_norm(size, LN_EPS, vb.pp("norm_ff"))?,
            norm_mha: layer_norm(size, LN_EPS, vb.pp("norm_mha"))?,
            norm_conv: layer_norm(size, LN_EPS, vb.pp("norm_conv"))?,
            norm_final: layer_norm(size, LN_EPS, vb.pp("norm_final"))?,
            feed_forward: PositionwiseFeedForward::load(
                &vb.pp("feed_forward"),
                size,
                linear_units,
            )?,
            conv_module: ConvolutionModule::load(&vb.pp("conv_module"), size)?,
        })
    }

    /// `mask` doubles as the attention key mask and the conv `mask_pad`
    /// (`(B, 1, T')`, 1 = valid); `pos_emb` is `(1, T', size)`.
    fn forward(&self, x: &Tensor, mask: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        // multi-headed self-attention (normalize_before)
        let residual = x.clone();
        let h = self.norm_mha.forward(x)?;
        let x_att = self.self_attn.forward(&h, mask, pos_emb)?;
        let mut x = (residual + x_att)?;

        // convolution module
        let residual = x.clone();
        let h = self.norm_conv.forward(&x)?;
        let h = self.conv_module.forward(&h, mask)?;
        x = (residual + h)?;

        // feed forward module (ff_scale = 1.0, macaron disabled)
        let residual = x.clone();
        let h = self.norm_ff.forward(&x)?;
        x = (residual + self.feed_forward.forward(&h)?)?;

        x = self.norm_final.forward(&x)?;
        Ok(x)
    }

    /// Test-only mirror of [`Self::forward`] that also returns the raw
    /// sub-module outputs (self-attn, conv module, feed-forward) matching
    /// the Python forward hooks. Returns `(attn, conv, ff, block_out)`.
    #[cfg(test)]
    fn forward_stage_dump(
        &self,
        x: &Tensor,
        mask: &Tensor,
        pos_emb: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // multi-headed self-attention (normalize_before)
        let residual = x.clone();
        let h = self.norm_mha.forward(x)?;
        let x_att = self.self_attn.forward(&h, mask, pos_emb)?;
        let mut x = (residual + &x_att)?;

        // convolution module
        let residual = x.clone();
        let h = self.norm_conv.forward(&x)?;
        let conv_out = self.conv_module.forward(&h, mask)?;
        x = (residual + &conv_out)?;

        // feed forward module (ff_scale = 1.0, macaron disabled)
        let residual = x.clone();
        let h = self.norm_ff.forward(&x)?;
        let ff_out = self.feed_forward.forward(&h)?;
        x = (residual + &ff_out)?;

        x = self.norm_final.forward(&x)?;
        Ok((x_att, conv_out, ff_out, x))
    }
}

/// `PositionwiseFeedForward` with SiLU activation.
struct PositionwiseFeedForward {
    w_1: Linear,
    w_2: Linear,
}

impl PositionwiseFeedForward {
    fn load(vb: &VarBuilder, idim: usize, hidden_units: usize) -> Result<Self> {
        Ok(Self {
            w_1: linear(idim, hidden_units, vb.pp("w_1"))?,
            w_2: linear(hidden_units, idim, vb.pp("w_2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.w_2.forward(&self.w_1.forward(xs)?.silu()?)
    }
}

/// `ConvolutionModule`: pointwise GLU -> depthwise conv -> LayerNorm + SiLU
/// -> pointwise, with padding masked to zero on input and output.
struct ConvolutionModule {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    pointwise_conv2: Conv1d,
    norm: LayerNorm,
}

impl ConvolutionModule {
    fn load(vb: &VarBuilder, channels: usize) -> Result<Self> {
        let padding = (CNN_MODULE_KERNEL - 1) / 2;
        Ok(Self {
            pointwise_conv1: conv1d(
                channels,
                2 * channels,
                1,
                Conv1dConfig::default(),
                vb.pp("pointwise_conv1"),
            )?,
            depthwise_conv: conv1d(
                channels,
                channels,
                CNN_MODULE_KERNEL,
                Conv1dConfig {
                    padding,
                    groups: channels,
                    ..Default::default()
                },
                vb.pp("depthwise_conv"),
            )?,
            pointwise_conv2: conv1d(
                channels,
                channels,
                1,
                Conv1dConfig::default(),
                vb.pp("pointwise_conv2"),
            )?,
            norm: layer_norm(channels, LN_EPS, vb.pp("norm"))?,
        })
    }

    /// `x`: `(B, T, C)`; `mask_pad`: `(B, 1, T)` with 1 = valid.
    fn forward(&self, x: &Tensor, mask_pad: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?.contiguous()?; // (B, C, T)
        let x = mask_pad_channels(&x, mask_pad)?;

        // GLU along the channel dimension.
        let halves = self.pointwise_conv1.forward(&x)?.chunk(2, 1)?;
        let x = (&halves[0] * candle_nn::ops::sigmoid(&halves[1])?)?;

        let x = self.depthwise_conv.forward(&x.contiguous()?)?;
        let x = self.norm.forward(&x.transpose(1, 2)?)?.silu()?;
        let x = self
            .pointwise_conv2
            .forward(&x.transpose(1, 2)?.contiguous()?)?;

        let x = mask_pad_channels(&x, mask_pad)?;
        x.transpose(1, 2)
    }
}

/// `x.masked_fill_(~mask_pad, 0.0)` for a channels-first `(B, C, T)`
/// tensor with a `(B, 1, T)` padding mask (1 = valid).
fn mask_pad_channels(x: &Tensor, mask_pad: &Tensor) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    let invalid = mask_pad
        .ones_like()?
        .sub(mask_pad)?
        .broadcast_as((b, c, t))?;
    let zeros = x.zeros_like()?;
    invalid.where_cond(&zeros, x)
}

/// `RelPositionMultiHeadedAttention` exactly as shipped in this fork:
/// the classic `rel_shift` of `matrix_bd` is commented out upstream, so the
/// position bias term is `q_with_bias_v @ p^T` against the unshifted
/// `pe[:, :T']` table. Non-causal; `pos_emb` is `(1, T', n_feat)`.
struct RelPositionMultiHeadedAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    h: usize,
    d_k: usize,
}

impl RelPositionMultiHeadedAttention {
    fn load(vb: &VarBuilder, n_feat: usize, n_head: usize) -> Result<Self> {
        let d_k = n_feat / n_head;
        let pos_bias_u = vb.get((n_head, d_k), "pos_bias_u")?;
        let pos_bias_v = vb.get((n_head, d_k), "pos_bias_v")?;
        Ok(Self {
            linear_q: linear(n_feat, n_feat, vb.pp("linear_q"))?,
            linear_k: linear(n_feat, n_feat, vb.pp("linear_k"))?,
            linear_v: linear(n_feat, n_feat, vb.pp("linear_v"))?,
            linear_out: linear(n_feat, n_feat, vb.pp("linear_out"))?,
            linear_pos: linear_no_bias(n_feat, n_feat, vb.pp("linear_pos"))?,
            pos_bias_u,
            pos_bias_v,
            h: n_head,
            d_k,
        })
    }

    /// `x`: `(B, T, n_feat)`; `mask`: `(B, 1, T')` with 1 = valid;
    /// `pos_emb`: `(1, T', n_feat)`.
    fn forward(&self, x: &Tensor, mask: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let t2 = pos_emb.dim(1)?;
        let q = self
            .linear_q
            .forward(x)?
            .reshape((b, t, self.h, self.d_k))?
            .transpose(1, 2)?;
        let k = self
            .linear_k
            .forward(x)?
            .reshape((b, t, self.h, self.d_k))?
            .transpose(1, 2)?;
        let v = self
            .linear_v
            .forward(x)?
            .reshape((b, t, self.h, self.d_k))?
            .transpose(1, 2)?;

        // q_with_bias_u / q_with_bias_v: (B, h, T, d_k) via a (B, T, h, d_k)
        // broadcast against the per-head (h, d_k) biases.
        let q_t = q.transpose(1, 2)?;
        let q_with_bias_u = q_t
            .broadcast_add(&self.pos_bias_u)?
            .transpose(1, 2)?
            .contiguous()?;
        let q_with_bias_v = q_t
            .broadcast_add(&self.pos_bias_v)?
            .transpose(1, 2)?
            .contiguous()?;

        let n_batch_pos = pos_emb.dim(0)?;
        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((n_batch_pos, t2, self.h, self.d_k))?
            .transpose(1, 2)?
            // torch matmul broadcasts the (1, h, T', d_k) position bias
            // against the (B, h, T, d_k) queries; candle does not.
            .broadcast_as((b, self.h, t2, self.d_k))?;

        // matrix_ac + matrix_bd (no rel_shift upstream) / sqrt(d_k)
        let matrix_ac = q_with_bias_u.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let matrix_bd = q_with_bias_v.matmul(&p.transpose(2, 3)?.contiguous()?)?;
        let scores = ((matrix_ac + matrix_bd)? * (1f64 / (self.d_k as f64).sqrt()))?;

        // forward_attention: -inf then softmax, then re-zero masked columns
        // (this also removes the NaN rows produced by fully padded queries,
        // exactly like torch's masked_fill after softmax).
        let invalid = mask
            .ones_like()?
            .sub(mask)?
            .unsqueeze(1)?
            .broadcast_as(scores.shape())?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?;
        let scores = invalid.where_cond(&neg_inf, &scores)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?;
        let zeros = attn.zeros_like()?;
        let attn = invalid.where_cond(&zeros, &attn)?;

        let out = attn.matmul(&v.contiguous()?)?;
        let out = out.transpose(1, 2)?.reshape((b, t, self.h * self.d_k))?;
        self.linear_out.forward(&out)
    }

    /// Test-only mirror of [`Self::forward`] recording the attention math
    /// internals (`q/k/v/p`, `matrix_ac/bd`, `scores`, `attn`, `out`).
    #[cfg(test)]
    fn forward_stage_dump(
        &self,
        x: &Tensor,
        mask: &Tensor,
        pos_emb: &Tensor,
    ) -> Result<Vec<(String, Tensor)>> {
        let (b, t, _) = x.dims3()?;
        let t2 = pos_emb.dim(1)?;
        let head = |name: &str, t: &Tensor, v: &mut Vec<(String, Tensor)>| {
            v.push((name.to_string(), t.clone()));
        };

        let q = self
            .linear_q
            .forward(x)?
            .reshape((b, t, self.h, self.d_k))?
            .transpose(1, 2)?;
        let k = self
            .linear_k
            .forward(x)?
            .reshape((b, t, self.h, self.d_k))?
            .transpose(1, 2)?;
        let v = self
            .linear_v
            .forward(x)?
            .reshape((b, t, self.h, self.d_k))?
            .transpose(1, 2)?;

        let q_t = q.transpose(1, 2)?;
        let q_u = q_t
            .broadcast_add(&self.pos_bias_u)?
            .transpose(1, 2)?
            .contiguous()?;
        let q_v = q_t
            .broadcast_add(&self.pos_bias_v)?
            .transpose(1, 2)?
            .contiguous()?;

        let n_batch_pos = pos_emb.dim(0)?;
        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((n_batch_pos, t2, self.h, self.d_k))?
            .transpose(1, 2)?
            .broadcast_as((b, self.h, t2, self.d_k))?;

        let matrix_ac = q_u.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let matrix_bd = q_v.matmul(&p.transpose(2, 3)?.contiguous()?)?;
        let scores = ((&matrix_ac + &matrix_bd)? * (1f64 / (self.d_k as f64).sqrt()))?;

        let invalid = mask
            .ones_like()?
            .sub(mask)?
            .unsqueeze(1)?
            .broadcast_as(scores.shape())?;
        let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?;
        let scores = invalid.where_cond(&neg_inf, &scores)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?;
        let zeros = attn.zeros_like()?;
        let attn = invalid.where_cond(&zeros, &attn)?;

        let mut stages = Vec::new();
        head("q", &q, &mut stages);
        head("k", &k, &mut stages);
        head("v", &v, &mut stages);
        head("p", &p, &mut stages);
        head("matrix_ac", &matrix_ac, &mut stages);
        head("matrix_bd", &matrix_bd, &mut stages);
        head("scores", &scores, &mut stages);
        head("attn", &attn, &mut stages);
        Ok(stages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{Init, VarBuilder, VarMap};

    /// VarMap builder that pre-creates every parameter with a small
    /// deterministic constant so loads succeed and the forward pass is
    /// non-degenerate without downloading anything.
    fn tiny_vb(
        input_size: usize,
        output_size: usize,
        linear_units: usize,
        heads: usize,
        blocks: usize,
    ) -> VarBuilder<'static> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let put = |vm: &VarMap, name: &str, shape: &[usize]| {
            vm.get(shape, name, Init::Const(0.1), candle_core::DType::F32, &dev)
                .unwrap();
        };
        let out_features = output_size * ((input_size - 1) / 2);
        put(&vm, "embed.conv.weight", &[output_size, 1, 3, 3]);
        put(&vm, "embed.conv.bias", &[output_size]);
        put(&vm, "embed.out.weight", &[output_size, out_features]);
        put(&vm, "embed.out.bias", &[output_size]);
        for i in 0..blocks {
            let p = format!("encoders.{i}");
            for ln in ["norm_ff", "norm_mha", "norm_conv", "norm_final"] {
                put(&vm, &format!("{p}.{ln}.weight"), &[output_size]);
                put(&vm, &format!("{p}.{ln}.bias"), &[output_size]);
            }
            for l in [
                "linear_q",
                "linear_k",
                "linear_v",
                "linear_out",
                "linear_pos",
            ] {
                put(
                    &vm,
                    &format!("{p}.self_attn.{l}.weight"),
                    &[output_size, output_size],
                );
                if l != "linear_pos" {
                    put(&vm, &format!("{p}.self_attn.{l}.bias"), &[output_size]);
                }
            }
            put(
                &vm,
                &format!("{p}.self_attn.pos_bias_u"),
                &[heads, output_size / heads],
            );
            put(
                &vm,
                &format!("{p}.self_attn.pos_bias_v"),
                &[heads, output_size / heads],
            );
            put(
                &vm,
                &format!("{p}.feed_forward.w_1.weight"),
                &[linear_units, output_size],
            );
            put(&vm, &format!("{p}.feed_forward.w_1.bias"), &[linear_units]);
            put(
                &vm,
                &format!("{p}.feed_forward.w_2.weight"),
                &[output_size, linear_units],
            );
            put(&vm, &format!("{p}.feed_forward.w_2.bias"), &[output_size]);
            put(
                &vm,
                &format!("{p}.conv_module.pointwise_conv1.weight"),
                &[2 * output_size, output_size, 1],
            );
            put(
                &vm,
                &format!("{p}.conv_module.pointwise_conv1.bias"),
                &[2 * output_size],
            );
            put(
                &vm,
                &format!("{p}.conv_module.depthwise_conv.weight"),
                &[output_size, 1, CNN_MODULE_KERNEL],
            );
            put(
                &vm,
                &format!("{p}.conv_module.depthwise_conv.bias"),
                &[output_size],
            );
            put(
                &vm,
                &format!("{p}.conv_module.pointwise_conv2.weight"),
                &[output_size, output_size, 1],
            );
            put(
                &vm,
                &format!("{p}.conv_module.pointwise_conv2.bias"),
                &[output_size],
            );
            put(&vm, &format!("{p}.conv_module.norm.weight"), &[output_size]);
            put(&vm, &format!("{p}.conv_module.norm.bias"), &[output_size]);
        }
        put(&vm, "after_norm.weight", &[output_size]);
        put(&vm, "after_norm.bias", &[output_size]);
        VarBuilder::from_varmap(&vm, candle_core::DType::F32, &dev)
    }

    #[test]
    fn conv2d2_shapes_and_mask() {
        let dev = Device::Cpu;
        let (input, output, heads, blocks) = (16, 8, 2, 2);
        let vb = tiny_vb(input, output, 32, heads, blocks);
        let enc =
            ConformerEncoder::load(&vb, input, output, 32, heads, blocks, "conv2d2").expect("load");

        let x = Tensor::randn(0f32, 1f32, (2, 21, input), &dev).unwrap();
        let ilens = Tensor::from_vec(vec![21i64, 15i64], (2,), &dev).unwrap();
        let (out, masks) = enc.forward(&x, &ilens).expect("forward");
        // Conv2dSubsampling2: T' = (T - 3) // 2 + 1 == (T - 1) / 2 = 10.
        assert_eq!(out.dims(), vec![2, 10, output]);
        assert_eq!(masks.dims(), vec![2, 1, 10]);
        let m = masks.to_vec3::<u8>().unwrap();
        // len=21 -> valid i with 2i+2 < 21 -> i <= 9 -> all 10 valid.
        assert!(m[0][0].iter().all(|&v| v == 1));
        // len=15 -> 2i+2 < 15 -> i <= 6 (2*6+2=14) -> first 7 valid.
        assert_eq!(&m[1][0][..7], &[1u8; 7]);
        assert!(m[1][0][7..].iter().all(|&v| v == 0));
        // fully masked softmax rows are cleaned, so the output stays finite
        let o = out.to_vec3::<f32>().unwrap();
        assert!(
            o.iter().flatten().flatten().all(|v| v.is_finite()),
            "output must be finite"
        );
    }

    #[test]
    fn conv2d2_output_length_formula() {
        let dev = Device::Cpu;
        let (input, output, heads, blocks) = (16, 8, 2, 1);
        let vb = tiny_vb(input, output, 32, heads, blocks);
        let enc =
            ConformerEncoder::load(&vb, input, output, 32, heads, blocks, "conv2d2").expect("load");
        // Exact Python formula floor((T - 3) / 2) + 1 (which equals
        // (T - 1) / 2) for a sweep of input lengths.
        for t in [4usize, 5, 6, 7, 8, 9, 12, 33] {
            let x = Tensor::randn(0f32, 1f32, (1, t, input), &dev).unwrap();
            let ilens = Tensor::from_vec(vec![t as i64], (1,), &dev).unwrap();
            let (out, masks) = enc.forward(&x, &ilens).expect("forward");
            let expected = (t - 3) / 2 + 1;
            assert_eq!(out.dims(), vec![1, expected, output], "T={t}");
            assert_eq!(masks.dims(), vec![1, 1, expected], "T={t}");
        }
    }

    #[test]
    fn padded_rows_do_not_leak_into_valid_region() {
        // The caller reads the encoder output rows [0, valid_len'); make
        // sure adding a padded batch element does not change the valid
        // element's rows (mask propagation through attention + conv).
        let dev = Device::Cpu;
        let (input, output, heads, blocks) = (16, 8, 2, 2);
        let vb = tiny_vb(input, output, 32, heads, blocks);
        let enc =
            ConformerEncoder::load(&vb, input, output, 32, heads, blocks, "conv2d2").expect("load");
        let a = Tensor::randn(0f32, 1f32, (1, 20, input), &dev).unwrap();
        let ilens_a = Tensor::from_vec(vec![20i64], (1,), &dev).unwrap();
        let (out_a, _) = enc.forward(&a, &ilens_a).unwrap();

        let b = Tensor::randn(0f32, 1f32, (1, 20, input), &dev).unwrap();
        let ab = Tensor::cat(&[&a, &b], 0).unwrap();
        let ilens = Tensor::from_vec(vec![20i64, 9], (2,), &dev).unwrap();
        let (out_ab, _) = enc.forward(&ab, &ilens).unwrap();

        let va = out_a.to_vec3::<f32>().unwrap();
        let vab = out_ab.to_vec3::<f32>().unwrap();
        let max_diff = va[0]
            .iter()
            .zip(vab[0].iter())
            .flat_map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(x, y)| (x - y).abs()))
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "batch mixing changed valid rows: {max_diff}"
        );
    }
}
