//! DiT estimator (`indextts/s2mel/modules/diffusion_transformer.py`) with
//! WaveNet final layer (`s2mel/modules/wavenet.py`).
//!
//! Shipped configuration: hidden 512, depth 13, heads 8 (head dim 64),
//! in_channels 80 (mel), content_dim 512 continuous condition, style
//! condition 192-d CAMPPlus (concatenated as features, `style_as_token=false`),
//! UViT skip connections (layers `i < depth/2` emit, `i > depth/2` receive),
//! long skip connection over the noisy mel, non-causal full attention, final
//! layer `wavenet` (8 layers, k5, dilation 1) plus a `FinalLayer` head.
//!
//! `setup_caches` upstream only preallocates attention KV buffers (with
//! `use_kv_cache=false` in this code path, i.e. nothing at all) and the RoPE
//! table / causal-mask scratch of `block_size` 16384; it does not affect
//! numerics, so this port computes the RoPE tables for the current sequence
//! length on the fly instead of caching them. `x_embedder`, `cond_embedder`
//! and `content_mask_embedder` exist in the checkpoint but are dead in the
//! inference forward and are not loaded.
//!
//! Upstream signature is `forward(x, prompt_x, x_lens, t, style, cond)`; the
//! fixed Rust signature drops `x_lens` (the shipped caller always passes the
//! full length, so the attention key mask is all-ones) and packs the prompt
//! mel into `x`.

use crate::config::{DiTConfig, WavenetConfig};
use crate::wavenet::{Wavenet, load_weight_norm_linear};
use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder, linear, linear_no_bias};

/// A `Linear` applied as an explicit contiguous 2-D matmul over `(B, T, C)`
/// inputs, with the transposed weight materialized once at load time.
/// Candle's CUDA matmul rejects the internally built strided views for some
/// 3-D lhs layouts; flattening by hand keeps every operand dense. The DiT
/// applies ~90 such projections per CFM step; doing `.t()?.contiguous()?`
/// per call re-copied megabytes of weights and added a dispatch each time.
/// The cached operand holds bit-identical
/// values, so every matmul result is unchanged.
pub(crate) struct LinT2d {
    weight_t: Tensor,
    bias: Option<Tensor>,
    out_dim: usize,
}

impl LinT2d {
    pub(crate) fn from_linear(l: Linear) -> Result<Self> {
        let out_dim = l.weight().dim(0)?;
        Ok(Self {
            weight_t: l.weight().t()?.contiguous()?,
            bias: l.bias().cloned(),
            out_dim,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
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

/// Shipped `args.style_encoder.dim` (CAMPPlus embedding size). `Dit::load`
/// must match the checkpoint's `cond_x_merge_linear` in-features
/// (`hidden + 2*in_channels + 192`).
pub(crate) const STYLE_DIM: usize = 192;

struct TimestepEmbedder {
    mlp0: Linear,
    mlp2: Linear,
    /// `exp(-ln(10000) * arange(half)/half)`, half = 256/2 (f32 buffer).
    freqs: Tensor,
}

impl TimestepEmbedder {
    fn load(vb: &VarBuilder, hidden_size: usize) -> Result<Self> {
        let mlp0 = linear(256, hidden_size, vb.pp("mlp.0"))?;
        let mlp2 = linear(hidden_size, hidden_size, vb.pp("mlp.2"))?;
        // Build the frequency buffer before moving the linears into Self.
        let freqs = {
            let device = mlp0.weight().device();
            let freqs: Vec<f32> = (0..128)
                .map(|i| (-(10000f64.ln()) * i as f64 / 128.0).exp() as f32)
                .collect();
            Tensor::from_vec(freqs, (128,), device)?
        };
        Ok(Self { mlp0, mlp2, freqs })
    }

    /// Sinusoidal embedding `(B,) -> (B, 256)` then MLP: exactly
    /// `cat([cos(scale*t ⊗ freqs), sin(scale*t ⊗ freqs)], -1)` with
    /// `scale = 1000` applied to `t` first, as upstream.
    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let args = t
            .to_dtype(DType::F32)?
            .affine(1000.0, 0.0)?
            .unsqueeze(1)? // (B, 1)
            .broadcast_mul(&self.freqs.unsqueeze(0)?)?; // (B, 128)
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], 1)?; // (B, 256)
        let emb = self.mlp0.forward(&emb)?.silu()?;
        self.mlp2.forward(&emb)
    }
}

/// `AdaptiveLayerNorm(x, c) = w ⊙ RMSNorm(x) + b` with `(w, b)` split from
/// `project_layer(c)`; RMS eps is the gpt_fast `norm_eps` 1e-5.
struct AdaptiveLayerNorm {
    project: LinT2d,
    norm_weight: Tensor,
}

impl AdaptiveLayerNorm {
    fn load(vb: &VarBuilder, dim: usize) -> Result<Self> {
        Ok(Self {
            project: LinT2d::from_linear(linear(dim, 2 * dim, vb.pp("project_layer"))?)?,
            norm_weight: vb.get((dim,), "norm.weight")?,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let dim = self.norm_weight.dim(0)?;
        let wb = self.project.forward(c)?; // (B, 1, 2*dim)
        let w = wb.narrow(2, 0, dim)?;
        let b = wb.narrow(2, dim, dim)?;
        let n = candle_nn::ops::rms_norm(&x.contiguous()?, &self.norm_weight, 1e-5)?;
        n.broadcast_mul(&w)?.broadcast_add(&b)
    }
}

struct DiTBlock {
    attention_norm: AdaptiveLayerNorm,
    wqkv: LinT2d,
    wo: LinT2d,
    ffn_norm: AdaptiveLayerNorm,
    w1: LinT2d,
    w3: LinT2d,
    w2: LinT2d,
    skip_in_linear: Option<LinT2d>,
}

pub struct Dit {
    blocks: Vec<DiTBlock>,
    norm: AdaptiveLayerNorm,
    t_embedder: TimestepEmbedder,
    t_embedder2: TimestepEmbedder,
    cond_projection: LinT2d,
    cond_x_merge_linear: LinT2d,
    skip_linear: Option<LinT2d>,
    conv1: Linear,
    conv2: Conv1d,
    res_projection: LinT2d,
    wavenet: Wavenet,
    final_linear: LinT2d,
    final_adaln: Linear,
    in_channels: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    long_skip_connection: bool,
    style_condition: bool,
    style_dim: usize,
}

/// `find_multiple` from gpt_fast: round `n` up to a multiple of `k`.
fn find_multiple(n: usize, k: usize) -> usize {
    if n.is_multiple_of(k) {
        n
    } else {
        n + k - n % k
    }
}

/// RoPE tables `(cos, sin)` of shape `(T, head_dim/2)` for
/// `candle_nn::rotary_emb::rope_i`, matching `precompute_freqs_cis(block_size,
/// head_dim, 10000)` restricted to the first `T` positions (f32 like the f32
/// checkpoint).
fn rope_tables(
    t_len: usize,
    head_dim: usize,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 10000f64.powf(-(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect();
    let inv = Tensor::from_vec(inv, (1, half), device)?;
    let pos = Tensor::arange(0f32, t_len as f32, device)?.reshape((t_len, 1))?;
    let angles = pos.matmul(&inv)?; // (T, half)
    let cos = angles.cos()?;
    let sin = angles.sin()?;
    Ok((cos, sin))
}

/// Preserve the separate rounded multiply/add chain used by the Windows CUDA
/// baseline. CUDA's fused rope_i kernel changes the full CFM/PCM output.
#[cfg(all(target_os = "windows", feature = "cuda"))]
fn apply_rope_unfused(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, t, h, d) = x.dims4()?;
    let half = d / 2;
    let xr = x.reshape((b, t, h, half, 2))?;
    let x0 = xr.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = xr.narrow(4, 1, 1)?.squeeze(4)?;
    let cos = cos.reshape((1, t, 1, half))?;
    let sin = sin.reshape((1, t, 1, half))?;
    let o0 = x0
        .broadcast_mul(&cos)?
        .broadcast_sub(&x1.broadcast_mul(&sin)?)?;
    let o1 = x1
        .broadcast_mul(&cos)?
        .broadcast_add(&x0.broadcast_mul(&sin)?)?;
    Tensor::stack(&[o0, o1], 4)?.flatten(3, 4)
}

impl Dit {
    /// Weight names under `models.cfm.estimator.*` in s2mel.pth.
    pub fn load(vb: &VarBuilder, dit_cfg: &DiTConfig, wavenet_cfg: &WavenetConfig) -> Result<Self> {
        if dit_cfg.time_as_token || dit_cfg.style_as_token {
            candle_core::bail!("token-style time/style conditioning is not shipped");
        }
        if dit_cfg.is_causal {
            candle_core::bail!("the shipped DiT is non-causal");
        }
        if dit_cfg.final_layer_type != "wavenet" {
            candle_core::bail!(
                "only the wavenet final layer is shipped (got {})",
                dit_cfg.final_layer_type
            );
        }
        // content_type only picks cond_embedder vs cond_projection upstream,
        // and forward() always uses cond_projection, so it is not consulted.
        if dit_cfg.zero_prompt_speech_token {
            candle_core::bail!("zero_prompt_speech_token is not shipped");
        }
        let hidden = dit_cfg.hidden_dim;
        if !hidden.is_multiple_of(dit_cfg.num_heads) {
            candle_core::bail!("hidden_dim must be divisible by num_heads");
        }
        // Upstream wires the DiT-hidden t1 into FinalLayer(wavenet.hidden_dim)
        // and shares the 512-d shapes; runnable checkpoints require equality.
        if wavenet_cfg.hidden_dim != hidden {
            candle_core::bail!(
                "wavenet.hidden_dim must equal DiT.hidden_dim ({} vs {hidden})",
                wavenet_cfg.hidden_dim
            );
        }
        let heads = dit_cfg.num_heads;
        let head_dim = hidden / heads;
        // gpt_fast ModelArgs.__post_init__: intermediate = find_multiple(
        // int(2 * (4 * dim) / 3), 256).
        let intermediate = find_multiple(2 * (4 * hidden) / 3, 256);
        let style_dim = if dit_cfg.style_condition {
            STYLE_DIM
        } else {
            0
        };

        let t_vb = vb.pp("t_embedder");
        let t_embedder = TimestepEmbedder::load(&t_vb, hidden)?;
        let t_embedder2 = TimestepEmbedder::load(&vb.pp("t_embedder2"), wavenet_cfg.hidden_dim)?;
        let cond_projection = LinT2d::from_linear(linear(
            dit_cfg.content_dim,
            hidden,
            vb.pp("cond_projection"),
        )?)?;
        let cond_x_merge_linear = LinT2d::from_linear(linear(
            hidden + 2 * dit_cfg.in_channels + style_dim,
            hidden,
            vb.pp("cond_x_merge_linear"),
        )?)?;

        let mut blocks = Vec::with_capacity(dit_cfg.depth);
        for i in 0..dit_cfg.depth {
            let lvb = vb.pp(format!("transformer.layers.{i}"));
            blocks.push(DiTBlock {
                attention_norm: AdaptiveLayerNorm::load(&lvb.pp("attention_norm"), hidden)?,
                wqkv: LinT2d::from_linear(linear_no_bias(
                    hidden,
                    3 * hidden,
                    lvb.pp("attention").pp("wqkv"),
                )?)?,
                wo: LinT2d::from_linear(linear_no_bias(
                    hidden,
                    hidden,
                    lvb.pp("attention").pp("wo"),
                )?)?,
                ffn_norm: AdaptiveLayerNorm::load(&lvb.pp("ffn_norm"), hidden)?,
                w1: LinT2d::from_linear(linear_no_bias(
                    hidden,
                    intermediate,
                    lvb.pp("feed_forward").pp("w1"),
                )?)?,
                w3: LinT2d::from_linear(linear_no_bias(
                    hidden,
                    intermediate,
                    lvb.pp("feed_forward").pp("w3"),
                )?)?,
                w2: LinT2d::from_linear(linear_no_bias(
                    intermediate,
                    hidden,
                    lvb.pp("feed_forward").pp("w2"),
                )?)?,
                skip_in_linear: if dit_cfg.uvit_skip_connection {
                    Some(LinT2d::from_linear(linear(
                        2 * hidden,
                        hidden,
                        lvb.pp("skip_in_linear"),
                    )?)?)
                } else {
                    None
                },
            });
        }
        let norm = AdaptiveLayerNorm::load(&vb.pp("transformer").pp("norm"), hidden)?;

        let skip_linear = if dit_cfg.long_skip_connection {
            Some(LinT2d::from_linear(linear(
                hidden + dit_cfg.in_channels,
                hidden,
                vb.pp("skip_linear"),
            )?)?)
        } else {
            None
        };
        let conv1 = linear(hidden, wavenet_cfg.hidden_dim, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv1d(
            wavenet_cfg.hidden_dim,
            dit_cfg.in_channels,
            1,
            Conv1dConfig::default(),
            vb.pp("conv2"),
        )?;
        let res_projection = LinT2d::from_linear(linear(
            hidden,
            wavenet_cfg.hidden_dim,
            vb.pp("res_projection"),
        )?)?;
        let wavenet = Wavenet::load(&vb.pp("wavenet"), wavenet_cfg)?;
        let final_linear = LinT2d::from_linear(load_weight_norm_linear(
            &vb.pp("final_layer").pp("linear"),
            wavenet_cfg.hidden_dim,
            wavenet_cfg.hidden_dim,
        )?)?;
        // FinalLayer.adaLN_modulation = Sequential(SiLU, Linear(h, 2h)).
        let final_adaln = linear(
            hidden,
            2 * hidden,
            vb.pp("final_layer").pp("adaLN_modulation").pp("1"),
        )?;

        Ok(Self {
            blocks,
            norm,
            t_embedder,
            t_embedder2,
            cond_projection,
            cond_x_merge_linear,
            skip_linear,
            conv1,
            conv2,
            res_projection,
            wavenet,
            final_linear,
            final_adaln,
            in_channels: dit_cfg.in_channels,
            hidden,
            heads,
            head_dim,
            long_skip_connection: dit_cfg.long_skip_connection,
            style_condition: dit_cfg.style_condition,
            style_dim,
        })
    }

    /// One estimator step. `x`: `(B, 2*in_channels, T)` = `cat([noisy_mel,
    /// prompt_mel], dim=1)` (upstream passes them as separate `x`/`prompt_x`
    /// arguments); `t`: timestep `(B,)` in [0, 1]; `cond`: `(B, T, 512)`
    /// length-regulated condition; `style`: `(B, 192)`; `drop_style`:
    /// upstream class-dropout branch, zeroing every merged feature beyond the
    /// noisy mel before `cond_x_merge_linear`. Returns `(B, in_channels, T)`.
    pub fn forward(
        &self,
        x: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        style: &Tensor,
        drop_style: bool,
    ) -> Result<Tensor> {
        let projected = self.project_condition(cond)?;
        self.forward_projected(x, t, &projected, style, drop_style)
    }

    /// The continuous condition is fixed for all steps of a CFM solve.
    pub(crate) fn project_condition(&self, cond: &Tensor) -> Result<Tensor> {
        self.cond_projection.forward(cond)
    }

    /// Same estimator with the condition projection supplied by the caller.
    /// The projected tensor belongs to one solve, never a cross-request cache.
    pub(crate) fn forward_projected(
        &self,
        x: &Tensor,
        t: &Tensor,
        projected: &Tensor,
        style: &Tensor,
        drop_style: bool,
    ) -> Result<Tensor> {
        let (b, channels, t_len) = x.dims3()?;
        let input_scope = crate::profile::scope(Some(x.device()), "dit.input");
        if channels != 2 * self.in_channels {
            candle_core::bail!(
                "DiT x must pack [noisy, prompt] mel halves ({} channels, got {channels})",
                2 * self.in_channels
            );
        }
        let noisy = x.narrow(1, 0, self.in_channels)?.transpose(1, 2)?; // (B, T, C_in)
        let prompt = x
            .narrow(1, self.in_channels, self.in_channels)?
            .transpose(1, 2)?;

        let t1 = self.t_embedder.forward(t)?; // (B, hidden)
        if projected.dims() != [b, t_len, self.hidden] {
            candle_core::bail!(
                "projected condition must be ({b}, {t_len}, {})",
                self.hidden
            );
        }
        let mut x_in = Tensor::cat(&[&noisy, &prompt, projected], 2)?; // (B, T, C+2C_in+H)
        if self.style_condition {
            let style_tok = style.to_dtype(DType::F32)?.unsqueeze(1)?.broadcast_as((
                b,
                t_len,
                self.style_dim,
            ))?;
            x_in = Tensor::cat(&[&x_in, &style_tok], 2)?;
        }
        if drop_style {
            // class_dropout: x_in[..., in_channels:] = 0 (kept noisy mel).
            let head = x_in.narrow(2, 0, self.in_channels)?;
            let tail = x_in
                .narrow(2, self.in_channels, x_in.dim(2)? - self.in_channels)?
                .zeros_like()?;
            x_in = Tensor::cat(&[&head, &tail], 2)?;
        }
        // LinT2d keeps the explicit contiguous 2-D matmul: candle's CUDA
        // path for a 3-D lhs against a 2-D weight internally builds a
        // strided view that its own matmul then rejects for this layout.
        let mut x = self.cond_x_merge_linear.forward(&x_in)?;
        drop(input_scope);

        // Transformer with UViT skip connections (non-causal, full attention).
        let (cos, sin) = {
            let _scope = crate::profile::scope(Some(x.device()), "dit.rope_tables");
            rope_tables(t_len, self.head_dim, x.device())?
        };
        let transformer_scope = crate::profile::scope(Some(x.device()), "dit.transformer");
        let c = t1.unsqueeze(1)?; // (B, 1, hidden)
        let half = self.blocks.len() / 2;
        let mut skip_stack: Vec<Tensor> = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let x_in_block = match (&block.skip_in_linear, i > half) {
                (Some(proj), true) => {
                    let skip = skip_stack.pop().ok_or_else(|| {
                        candle_core::Error::Msg("uvit skip stack underflow".into())
                    })?;
                    proj.forward(&Tensor::cat(&[&x, &skip], 2)?)?
                }
                _ => x.clone(),
            };
            x = self.forward_block(block, &x_in_block, &c, &cos, &sin)?;
            if block.skip_in_linear.is_some() && i < half {
                skip_stack.push(x.clone());
            }
        }
        let mut x_res = {
            let _g = crate::profile::scope(Some(x.device()), "dit.norm");
            self.norm.forward(&x, &c)?
        };
        drop(transformer_scope);
        let wavenet_scope = crate::profile::scope(Some(x.device()), "dit.wavenet");

        // Long skip over the noisy mel.
        if self.long_skip_connection {
            let _g = crate::profile::scope(Some(x.device()), "dit.long_skip");
            let skip_in = Tensor::cat(&[&x_res, &noisy], 2)?;
            x_res = self
                .skip_linear
                .as_ref()
                .expect("long_skip_connection implies skip_linear")
                .forward(&skip_in)?;
        }

        // WaveNet final layer with its own timestep embedding.
        let _g_wn = crate::profile::scope(Some(x.device()), "dit.wavenet.core");
        let h = self.conv1.forward(&x_res)?.transpose(1, 2)?.contiguous()?; // (B, wn_h, T)
        let t2 = self.t_embedder2.forward(t)?; // (B, wn_h)
        let wn = self.wavenet.forward(&h, &t2.unsqueeze(2)?)?; // (B, wn_h, T)
        drop(_g_wn);
        let mut x_final = wn
            .transpose(1, 2)?
            .broadcast_add(&self.res_projection.forward(&x_res)?)?;
        drop(wavenet_scope);
        let _final_scope = crate::profile::scope(Some(x.device()), "dit.final");

        // FinalLayer: affine-free LayerNorm (eps 1e-6) modulated by t1.
        let _g = crate::profile::scope(Some(x.device()), "dit.final_head");
        let shift_scale = self.final_adaln.forward(&t1.silu()?)?; // (B, 2*hidden)
        let shift = shift_scale.narrow(1, 0, self.hidden)?.unsqueeze(1)?;
        let scale = shift_scale
            .narrow(1, self.hidden, self.hidden)?
            .unsqueeze(1)?;
        let ones = Tensor::ones(self.hidden, DType::F32, x_final.device())?;
        let zeros = Tensor::zeros(self.hidden, DType::F32, x_final.device())?;
        let normed = candle_nn::ops::layer_norm(&x_final, &ones, &zeros, 1e-6)?;
        let modulated = normed
            .broadcast_mul(&scale.affine(1.0, 1.0)?)?
            .broadcast_add(&shift)?;
        x_final = self.final_linear.forward(&modulated)?;
        drop(_g);

        // conv2: Conv1d(wn_h -> in_channels, k1) over (B, wn_h, T).
        let _g = crate::profile::scope(Some(x.device()), "dit.conv2");
        let out = self
            .conv2
            .forward(&x_final.transpose(1, 2)?.contiguous()?)?;
        drop(_g);
        Ok(out)
    }

    /// Attention + SwiGLU feed-forward with pre-AdaptiveLayerNorm residuals,
    /// matching `TransformerBlock.forward` (non-causal, no cross attention).
    fn forward_block(
        &self,
        block: &DiTBlock,
        x: &Tensor,
        c: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let hidden = self.hidden;
        let (b, t_len, _) = x.dims3()?;
        let _g_adaln = crate::profile::scope(Some(x.device()), "dit.adaln");
        let attn_in = block.attention_norm.forward(x, c)?; // (B, T, H)
        drop(_g_adaln);
        let _g_qkv = crate::profile::scope(Some(x.device()), "dit.qkv");
        let qkv = block.wqkv.forward(&attn_in)?; // (B, T, 3H)
        drop(_g_qkv);
        // Split q/k/v into the head-major (B, H, T, D) layout the attention
        // wants and apply the interleaved RoPE to q/k. On Metal the whole
        // chain collapses into one hosted custom-op kernel reading qkv
        // directly (`metal_fused::qkv_split_rope`, bit-identical, gated in
        // tests); the fallback chain below covers CPU and unusual shapes.
        // The fused kernel replaces six strided copies (narrow/transpose ×3)
        // plus two rope_i dispatches per block per Euler step.
        let _g_split = crate::profile::scope(Some(x.device()), "dit.split");
        let split = |offset: usize| -> Result<Tensor> {
            qkv.narrow(2, offset, hidden)?
                .contiguous()?
                .reshape((b, t_len, self.heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let chain = || -> Result<(Tensor, Tensor, Tensor)> {
            let _g_rope = crate::profile::scope(Some(x.device()), "dit.rope");
            #[cfg(all(target_os = "windows", feature = "cuda"))]
            if x.device().is_cuda() {
                let rotate = |offset| {
                    let input = qkv.narrow(2, offset, hidden)?.contiguous()?.reshape((
                        b,
                        t_len,
                        self.heads,
                        self.head_dim,
                    ))?;
                    apply_rope_unfused(&input, cos, sin)?
                        .transpose(1, 2)?
                        .contiguous()
                };
                return Ok((rotate(0)?, rotate(hidden)?, split(2 * hidden)?));
            }
            let q = candle_nn::rotary_emb::rope_i(&split(0)?, cos, sin)?;
            let k = candle_nn::rotary_emb::rope_i(&split(hidden)?, cos, sin)?;
            drop(_g_rope);
            Ok((q, k, split(2 * hidden)?))
        };
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let (q, k, v) = if x.device().is_metal() {
            let fused = {
                let _g = crate::profile::scope(Some(x.device()), "dit.split_rope_fused");
                crate::metal_fused::qkv_split_rope(&qkv, cos, sin, self.heads, self.head_dim)
            };
            match fused {
                Ok(qkv3) => (
                    qkv3.narrow(0, 0, 1)?.squeeze(0)?,
                    qkv3.narrow(0, 1, 1)?.squeeze(0)?,
                    qkv3.narrow(0, 2, 1)?.squeeze(0)?,
                ),
                Err(_) => chain()?,
            }
        } else {
            chain()?
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let (q, k, v) = chain()?;
        drop(_g_split);
        // Full attention with the SDPA scale 1/sqrt(head_dim); the x_lens key
        // mask is all-ones for the shipped inference caller.
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let y = if x.device().is_metal()
            && matches!(self.head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512)
        {
            // candle_nn::ops::sdpa has no CPU implementation.
            let _g = crate::profile::scope(Some(x.device()), "dit.attn_sdpa");
            candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?
                .transpose(1, 2)?
                .contiguous()?
                .flatten(2, 3)?
        } else {
            let _g = crate::profile::scope(Some(x.device()), "dit.attn_scores");
            let kt = k.transpose(2, 3)?.contiguous()?;
            let scores = q.matmul(&kt)?.affine(scale, 0.0)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            probs
                .matmul(&v)?
                .transpose(1, 2)?
                .contiguous()?
                .flatten(2, 3)? // (B, T, H)
        };
        let _g_out = crate::profile::scope(Some(x.device()), "dit.attn_out");
        let h = x.broadcast_add(&block.wo.forward(&y)?)?;
        drop(_g_out);
        let _g_ffn = crate::profile::scope(Some(x.device()), "dit.ffn");
        let ffn_in = block.ffn_norm.forward(&h, c)?;
        let gated = block
            .w1
            .forward(&ffn_in)?
            .silu()?
            .broadcast_mul(&block.w3.forward(&ffn_in)?)?;
        let out = h.broadcast_add(&block.w2.forward(&gated)?);
        drop(_g_ffn);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DiTConfig, WavenetConfig};
    use candle_core::Device;
    use candle_nn::VarMap;

    /// The pre-optimization token-major RoPE: adjacent-element pairs `(x0, x1)`
    /// rotated by `(cos θ, sin θ)` on `(B, T, H, head_dim)` with broadcast
    /// `(1, T, 1, head_dim/2)` tables. Kept as the numeric reference for the
    /// fused `rope_i` switch.
    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, t, h, d) = x.dims4()?;
        let half = d / 2;
        let xr = x.reshape((b, t, h, half, 2))?;
        let x0 = xr.narrow(4, 0, 1)?.squeeze(4)?;
        let x1 = xr.narrow(4, 1, 1)?.squeeze(4)?;
        let o0 = x0
            .broadcast_mul(cos)?
            .broadcast_sub(&x1.broadcast_mul(sin)?)?;
        let o1 = x1
            .broadcast_mul(cos)?
            .broadcast_add(&x0.broadcast_mul(sin)?)?;
        Tensor::stack(&[o0, o1], 4)?.flatten(3, 4)
    }

    /// VarMap initializes linears with kaiming noise; reset everything the
    /// load touched to zero so the "zero-weight graph" reasoning holds.
    fn zero_all(varmap: &mut VarMap) {
        let names: Vec<String> = varmap.data().lock().unwrap().keys().cloned().collect();
        for name in names {
            let zeros = {
                let map = varmap.data().lock().unwrap();
                let var = &map[&name];
                Tensor::zeros(var.shape().dims(), DType::F32, var.device()).unwrap()
            };
            varmap.set_one(name, &zeros).unwrap();
        }
    }

    fn tiny_cfgs() -> (DiTConfig, WavenetConfig) {
        (
            DiTConfig {
                hidden_dim: 16,
                num_heads: 2,
                depth: 5,
                in_channels: 4,
                content_dim: 16,
                style_condition: true,
                final_layer_type: "wavenet".to_string(),
                is_causal: false,
                long_skip_connection: true,
                time_as_token: false,
                style_as_token: false,
                uvit_skip_connection: true,
                ..Default::default()
            },
            WavenetConfig {
                hidden_dim: 16,
                num_layers: 2,
                kernel_size: 3,
                dilation_rate: 1,
                p_dropout: 0.2,
                style_condition: true,
            },
        )
    }

    #[test]
    fn timestep_embedding_frequency_formula() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let emb = TimestepEmbedder::load(&vb, 16).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &device).unwrap();
        let out = emb.forward(&t).unwrap();
        assert_eq!(out.dims(), vec![1, 16]);
        // Zero MLP weights: output exactly zero, path exercised cleanly.
        assert!(
            out.to_vec2::<f32>()
                .unwrap()
                .iter()
                .flatten()
                .all(|&v| v == 0.0)
        );
    }

    #[test]
    fn forward_zero_weights_is_zero() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let (dit_cfg, wn_cfg) = tiny_cfgs();
        let model = Dit::load(&vb, &dit_cfg, &wn_cfg).unwrap();
        let b = 1;
        let t_len = 6;
        let x = Tensor::randn(0f32, 1f32, (b, 8, t_len), &device).unwrap();
        let t = Tensor::from_vec(vec![0.25f32], (b,), &device).unwrap();
        let cond = Tensor::randn(0f32, 1f32, (b, t_len, 16), &device).unwrap();
        let style = Tensor::randn(0f32, 1f32, (b, STYLE_DIM), &device).unwrap();
        let out = model.forward(&x, &t, &cond, &style, false).unwrap();
        assert_eq!(out.dims(), vec![b, 4, t_len]);
        // Every weight is zero: the whole graph collapses to conv2(0) = 0.
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
    }

    #[test]
    fn forward_constant_output_through_conv2_bias() {
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let (dit_cfg, wn_cfg) = tiny_cfgs();
        let model = Dit::load(&vb, &dit_cfg, &wn_cfg).unwrap();
        // Zero weights everywhere: everything reaching conv2 is zero, so the
        // output is exactly the conv2 bias broadcast over (B, 4, T).
        zero_all(&mut varmap);
        varmap
            .set_one("conv2.bias", Tensor::full(0.5f32, 4, &device).unwrap())
            .unwrap();
        let x = Tensor::randn(0f32, 1f32, (2, 8, 5), &device).unwrap();
        let t = Tensor::from_vec(vec![0.75f32, 0.75f32], (2,), &device).unwrap();
        let cond = Tensor::randn(0f32, 1f32, (2, 5, 16), &device).unwrap();
        let style = Tensor::randn(0f32, 1f32, (2, STYLE_DIM), &device).unwrap();
        let out = model.forward(&x, &t, &cond, &style, false).unwrap();
        assert_eq!(out.dims(), vec![2, 4, 5]);
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|&v| (v - 0.5).abs() < 1e-6)
        );
    }

    #[test]
    fn forward_rejects_unpacked_mel() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let (dit_cfg, wn_cfg) = tiny_cfgs();
        let model = Dit::load(&vb, &dit_cfg, &wn_cfg).unwrap();
        let x = Tensor::zeros((1, 4, 5), DType::F32, &device).unwrap();
        let t = Tensor::zeros((1,), DType::F32, &device).unwrap();
        let cond = Tensor::zeros((1, 5, 16), DType::F32, &device).unwrap();
        let style = Tensor::zeros((1, STYLE_DIM), DType::F32, &device).unwrap();
        assert!(model.forward(&x, &t, &cond, &style, false).is_err());
    }

    #[test]
    fn prepared_condition_matches_fresh_projection_across_lengths_and_steps() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let (dit_cfg, wn_cfg) = tiny_cfgs();
        let model = Dit::load(&vb, &dit_cfg, &wn_cfg).unwrap();
        for len in [5, 8, 5] {
            let cond = Tensor::randn(0f32, 1f32, (2, len, 16), &device).unwrap();
            let style = Tensor::randn(0f32, 1f32, (2, STYLE_DIM), &device).unwrap();
            let projected = model.project_condition(&cond).unwrap();
            for (time, drop_style) in [(0.0f32, false), (0.5, true)] {
                let x = Tensor::randn(0f32, 1f32, (2, 8, len), &device).unwrap();
                let t = Tensor::full(time, 2, &device).unwrap();
                let expected = model.forward(&x, &t, &cond, &style, drop_style).unwrap();
                let actual = model
                    .forward_projected(&x, &t, &projected, &style, drop_style)
                    .unwrap();
                assert_eq!(
                    actual.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
                );
            }
        }
    }

    #[test]
    fn rope_tables_match_gpt_fast() {
        let device = Device::Cpu;
        // freqs = 1 / 10000^(arange(0, 8, 2)/8) -> [1, 0.1778..]
        let (cos, sin) = rope_tables(3, 8, &device).unwrap();
        assert_eq!(cos.dims(), vec![3, 4]);
        let cos = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Position 0 rotates by angle 0; Accelerate's cosf may differ by 1 ULP.
        assert!(
            cos[0..4]
                .iter()
                .all(|value| (*value - 1.0f32).abs() <= f32::EPSILON)
        );
        let inv0 = 10000f64.powf(-2.0 / 8.0);
        // (1, 3, 1, 4) row-major: position 1, freq 1 -> index 4 + 1.
        assert!((cos[5] as f64 - inv0.cos()).abs() < 1e-6);
        let sin = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((sin[5] as f64 - inv0.sin()).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires converted checkpoints (INDEXTTS_CONVERTED_DIR)"]
    fn real_weights_forward() {
        let dir = match std::env::var("INDEXTTS_CONVERTED_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => return,
        };
        let device = Device::Cpu;
        let tensors =
            candle_core::safetensors::load(dir.join("s2mel.safetensors"), &device).unwrap();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let cfg = crate::config::S2MelConfig::default();
        let model = Dit::load(
            &vb.pp("models").pp("cfm").pp("estimator"),
            &cfg.dit,
            &cfg.wavenet,
        )
        .unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 160, 64), &device).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &device).unwrap();
        let cond = Tensor::randn(0f32, 1f32, (1, 64, 512), &device).unwrap();
        let style = Tensor::randn(0f32, 1f32, (1, STYLE_DIM), &device).unwrap();
        let out = model.forward(&x, &t, &cond, &style, false).unwrap();
        assert_eq!(out.dims(), vec![1, 80, 64]);
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    /// Metal probe at the real DiT attention shapes (B=2 CFG, 8 heads,
    /// head_dim 64, F32): the manual materialized path used by
    /// `forward_block` vs `candle_nn::ops::sdpa` (fused Metal kernel), at the
    /// target-only length (~492 frames) and the full solve length (~1024 =
    /// prompt + target). Numeric gate for the sdpa switch: both paths are
    /// compared against an f64 CPU ground truth; timing is diagnostic only.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device; timing probe, no models"]
    fn attention_sdpa_vs_manual_metal() {
        let device = Device::new_metal(0).unwrap();
        let (b, h, d) = (2usize, 8, 64);
        for t in [492usize, 1024] {
            let q = Tensor::randn(0f32, 1f32, (b, h, t, d), &device).unwrap();
            let k = Tensor::randn(0f32, 1f32, (b, h, t, d), &device).unwrap();
            let v = Tensor::randn(0f32, 1f32, (b, h, t, d), &device).unwrap();
            let scale = 1.0f32 / (d as f32).sqrt();
            let manual = |q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor> {
                let kt = k.transpose(2, 3)?.contiguous()?;
                let scores = q.matmul(&kt)?.affine(scale as f64, 0.0)?;
                let probs = candle_nn::ops::softmax_last_dim(&scores)?;
                probs
                    .matmul(v)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .flatten(2, 3)
            };
            let fused = |q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor> {
                candle_nn::ops::sdpa(q, k, v, None, false, scale, 1.0)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .flatten(2, 3)
            };
            let out_manual = manual(&q, &k, &v).unwrap();
            let out_fused = fused(&q, &k, &v).unwrap();
            let diff = (&out_manual - &out_fused).unwrap().abs().unwrap();
            let max_abs = diff.max_all().unwrap().to_scalar::<f32>().unwrap();

            // f64 ground truth on CPU: both paths must sit equally close to it
            // for the gap to count as reduction-order noise, not semantics.
            let q64 = q
                .to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F64)
                .unwrap();
            let k64 = k
                .to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F64)
                .unwrap();
            let v64 = v
                .to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F64)
                .unwrap();
            let truth = manual_f64(&q64, &k64, &v64, scale as f64);
            fn err_vs(out: &Tensor, truth: &Tensor) -> Result<f64> {
                let o = out.to_device(&Device::Cpu)?.to_dtype(DType::F64)?;
                (&o - truth)?.abs()?.max_all()?.to_scalar::<f64>()
            }
            let e_manual = err_vs(&out_manual, &truth).unwrap();
            let e_fused = err_vs(&out_fused, &truth).unwrap();
            println!(
                "T={t:4} max_abs {max_abs:.3e} | err-vs-f64 manual {e_manual:.3e} fused {e_fused:.3e}"
            );
            assert_eq!(out_manual.dims(), out_fused.dims().to_vec());
            assert!(
                e_fused <= e_manual.max(1e-5),
                "fused strays from f64 truth: {e_fused:.3e} vs manual {e_manual:.3e}"
            );

            let warmup = 5;
            let reps = 50;
            let time = |f: &mut dyn FnMut() -> Result<Tensor>| -> f64 {
                for _ in 0..warmup {
                    std::mem::drop(f().unwrap());
                }
                device.synchronize().unwrap();
                let start = std::time::Instant::now();
                for _ in 0..reps {
                    std::mem::drop(f().unwrap());
                }
                device.synchronize().unwrap();
                start.elapsed().as_secs_f64() / reps as f64 * 1e3
            };
            let manual_ms = time(&mut || manual(&q, &k, &v));
            let fused_ms = time(&mut || fused(&q, &k, &v));
            let ratio = manual_ms / fused_ms;
            println!(
                "T={t:4} manual {manual_ms:.3} ms | sdpa {fused_ms:.3} ms | ratio {ratio:.2}x"
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn manual_f64(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Tensor {
        let kt = k.transpose(2, 3).unwrap().contiguous().unwrap();
        let scores = q.matmul(&kt).unwrap().affine(scale, 0.0).unwrap();
        let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        probs
            .matmul(v)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .flatten(2, 3)
            .unwrap()
    }

    /// Differential check of `rope_i` vs the reference `apply_rope`: CPU
    /// rope_i must match apply_rope exactly, Metal (when built) within float
    /// rounding. If the big probe trips, this isolates whether the divergence
    /// is semantic or Metal-shape-specific.
    #[test]
    fn rope_i_matches_reference() {
        let (b, h, t, d) = (2usize, 2, 5, 8);
        let half = d / 2;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let devices = [Device::Cpu, Device::new_metal(0).unwrap()];
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let devices = [Device::Cpu];
        for device in devices {
            let (cos, sin) = rope_tables(t, d, &device).unwrap();
            let cos_b = cos.reshape((1, t, 1, half)).unwrap();
            let sin_b = sin.reshape((1, t, 1, half)).unwrap();
            let x = Tensor::randn(0f32, 1f32, (b, t, h, d), &device).unwrap();
            let want = apply_rope(&x, &cos_b, &sin_b)
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap();
            let got = candle_nn::rotary_emb::rope_i(
                &x.transpose(1, 2).unwrap().contiguous().unwrap(),
                &cos,
                &sin,
            )
            .unwrap();
            let max_abs = (&want - &got)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            println!("{:?}: rope_i vs apply_rope max_abs {max_abs:.3e}", device);
            assert!(max_abs <= 1e-5, "{:?}: max_abs {max_abs:.3e}", device);
        }
    }

    /// The shipped qkv split must carry identical data on CPU and Metal at
    /// the real solve shapes. The head-major layout requires the (h, d) split
    /// in token-major BEFORE the transpose — transposing the (B, T, H) view
    /// first leaves no d dimension to reshape into. Guards that invariant and
    /// the Metal strided-copy path.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device; no models"]
    fn split_orders_match_metal() {
        let (b, t) = (2usize, 1024);
        let (hidden, heads) = (512usize, 8);
        let head_dim = hidden / heads;
        for device in [Device::Cpu, Device::new_metal(0).unwrap()] {
            let qkv = Tensor::randn(0f32, 1f32, (b, t, 3 * hidden), &device).unwrap();
            let reference = |offset: usize| -> Result<Tensor> {
                let x = qkv
                    .narrow(2, offset, hidden)?
                    .contiguous()?
                    .reshape((b, t, heads, head_dim))?;
                x.flatten_all()
            };
            let split = |offset: usize| -> Result<Tensor> {
                qkv.narrow(2, offset, hidden)?
                    .contiguous()?
                    .reshape((b, t, heads, head_dim))?
                    .transpose(1, 2)?
                    .contiguous()
            };
            for offset in [0usize, hidden, 2 * hidden] {
                // Cross-check the token-major (b, t, h, d) reading against a
                // direct gather: element [b][t][h][d] = qkv[b][t][off + h*d + d2].
                let want = reference(offset).unwrap();
                let got = split(offset).unwrap();
                // Reorder the split back to token-major on CPU for comparison.
                let got_tm = got
                    .transpose(1, 2)
                    .unwrap()
                    .contiguous()
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_device(&Device::Cpu)
                    .unwrap();
                let want_cpu = want.to_device(&Device::Cpu).unwrap();
                let max_abs = (&want_cpu - &got_tm)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                println!("{:?} offset {offset}: max_abs {max_abs:.3e}", device);
                assert!(max_abs == 0.0, "{:?}: max_abs {max_abs:.3e}", device);
            }
        }
    }

    /// Numeric gate for the fused qkv split+rope kernel: the rotation
    /// expressions match candle's `ropei` kernel verbatim and the v plane is
    /// a pure copy, so the single-dispatch result must be bit-identical to
    /// the production split/rope_i chain (all heads rotate — IndexTTS ropes
    /// the whole hidden dim). Also checks that sdpa accepts the offset
    /// `narrow(0, i, 1)` plane views.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device; no models"]
    fn fused_qkv_split_rope_matches_chain_metal() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping, no metal device: {e}");
                return;
            }
        };
        let (b, heads, head_dim) = (2usize, 8, 64);
        let hidden = heads * head_dim;
        for t in [1024usize, 37] {
            let qkv = Tensor::randn(0f32, 1f32, (b, t, 3 * hidden), &device).unwrap();
            let cos = Tensor::randn(0f32, 1f32, (t, head_dim / 2), &device).unwrap();
            let sin = Tensor::randn(0f32, 1f32, (t, head_dim / 2), &device).unwrap();

            let split = |offset: usize| -> Result<Tensor> {
                qkv.narrow(2, offset, hidden)?
                    .contiguous()?
                    .reshape((b, t, heads, head_dim))?
                    .transpose(1, 2)?
                    .contiguous()
            };
            let q = candle_nn::rotary_emb::rope_i(&split(0).unwrap(), &cos, &sin).unwrap();
            let k = candle_nn::rotary_emb::rope_i(&split(hidden).unwrap(), &cos, &sin).unwrap();
            let v = split(2 * hidden).unwrap();

            let fused =
                crate::metal_fused::qkv_split_rope(&qkv, &cos, &sin, heads, head_dim).unwrap();
            assert_eq!(fused.dims(), &[3, b, heads, t, head_dim][..]);
            for (plane, expect) in [&q, &k, &v].iter().enumerate() {
                let got = fused.narrow(0, plane, 1).unwrap().squeeze(0).unwrap();
                let max_abs = (&got - *expect)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                println!("qkv-split-rope t={t} plane {plane}: max_abs {max_abs:.3e}");
                assert!(max_abs == 0.0, "plane {plane} t={t} max_abs {max_abs:.3e}");
            }

            // The production consumer: sdpa on the zero-copy plane views must
            // match sdpa on the contiguous chain outputs bit-for-bit.
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let fq = fused.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
            let fk = fused.narrow(0, 1, 1).unwrap().squeeze(0).unwrap();
            let fv = fused.narrow(0, 2, 1).unwrap().squeeze(0).unwrap();
            let from_views = candle_nn::ops::sdpa(&fq, &fk, &fv, None, false, scale, 1.0).unwrap();
            let from_chain = candle_nn::ops::sdpa(&q, &k, &v, None, false, scale, 1.0).unwrap();
            let max_abs = (&from_views - &from_chain)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            println!("qkv-split-rope t={t} sdpa: max_abs {max_abs:.3e}");
            assert!(max_abs == 0.0, "sdpa t={t} max_abs {max_abs:.3e}");
        }
    }

    /// Metal per-stage probe of `forward_block` at the real solve shapes
    /// (B=2 CFG, T≈1024 prompt+target, hidden 512, 8 heads, F32). Splits the
    /// block into its GEMMs and the q/k/v slicing+RoPE chain so the stage
    /// attribution behind any optimization candidate is measurable without
    /// the sync overhead of the INDEXTTS_PROFILE run-level spans.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device; timing probe, no models"]
    fn forward_block_stage_timing_metal() {
        let device = Device::new_metal(0).unwrap();
        let (b, t, hidden, heads) = (2usize, 1024, 512, 8);
        let head_dim = hidden / heads;
        let stage = |name: &str, f: &mut dyn FnMut() -> Result<Tensor>| {
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
            println!("{name:12} {ms:7.3} ms");
            last.unwrap()
        };

        let x = Tensor::randn(0f32, 1f32, (b, t, hidden), &device).unwrap();
        let w_qkv = Tensor::randn(0f32, 1f32, (hidden, 3 * hidden), &device).unwrap();
        let w_o = Tensor::randn(0f32, 1f32, (hidden, hidden), &device).unwrap();
        let w_ff = Tensor::randn(0f32, 1f32, (hidden, 4 * hidden), &device).unwrap();
        let w_ff2 = Tensor::randn(0f32, 1f32, (4 * hidden, hidden), &device).unwrap();
        let (cos, sin) = rope_tables(t, head_dim, &device).unwrap();
        let half = head_dim / 2;
        // The old token-major chain broadcast (1, T, 1, half) tables.
        let cos_b = cos.reshape((1, t, 1, half)).unwrap();
        let sin_b = sin.reshape((1, t, 1, half)).unwrap();

        let split = |qkv: &Tensor, offset: usize| -> Result<Tensor> {
            qkv.narrow(2, offset, hidden)?
                .contiguous()?
                .reshape((b, t, heads, head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };

        let qkv = stage("qkv_gemm", &mut || {
            x.flatten(0, 1)?.matmul(&w_qkv)?.reshape((b, t, 3 * hidden))
        });
        let q_old = stage("chain_old", &mut || {
            let q = qkv
                .narrow(2, 0, hidden)?
                .contiguous()?
                .reshape((b, t, heads, head_dim))?;
            let k = qkv
                .narrow(2, hidden, hidden)?
                .contiguous()?
                .reshape((b, t, heads, head_dim))?;
            let v = qkv
                .narrow(2, 2 * hidden, hidden)?
                .contiguous()?
                .reshape((b, t, heads, head_dim))?;
            let q = apply_rope(&q, &cos_b, &sin_b)?
                .transpose(1, 2)?
                .contiguous()?;
            let k = apply_rope(&k, &cos_b, &sin_b)?
                .transpose(1, 2)?
                .contiguous()?;
            let v = v.transpose(1, 2)?.contiguous()?;
            Tensor::cat(&[&q, &k, &v], 0) // keep all three alive
        })
        .narrow(0, 0, 3 * b)
        .unwrap();

        // Numeric gate: new split+rope_i chain vs the old chain on the same
        // qkv. qkv entries are ~N(0, 22) here (512-deep randn GEMM), so the
        // gate is relative: the fused kernel differs from the broadcast path
        // only by float rounding (2-3 ULP).
        let q_new = split(&qkv, 0).unwrap();
        let k_new = split(&qkv, hidden).unwrap();
        let v_new = split(&qkv, 2 * hidden).unwrap();
        let q_new = candle_nn::rotary_emb::rope_i(&q_new, &cos, &sin).unwrap();
        let k_new = candle_nn::rotary_emb::rope_i(&k_new, &cos, &sin).unwrap();
        let q_old_s = q_old.narrow(0, 0, b).unwrap();
        let k_old_s = q_old.narrow(0, b, b).unwrap();
        let v_old_s = q_old.narrow(0, 2 * b, b).unwrap();
        for (name, new, old) in [
            ("v", &v_new, &v_old_s),
            ("q", &q_new, &q_old_s),
            ("k", &k_new, &k_old_s),
        ] {
            let max_abs = (new - old)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let mag = old
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let rel = max_abs / mag;
            println!("{name} old-vs-new max_abs {max_abs:.3e} (|max| {mag:.1}, rel {rel:.3e})");
            assert!(rel <= 1e-6, "{name} old-vs-new rel {rel:.3e}");
        }

        let _ = stage("chain_new", &mut || {
            let q = candle_nn::rotary_emb::rope_i(&split(&qkv, 0)?, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope_i(&split(&qkv, hidden)?, &cos, &sin)?;
            let v = split(&qkv, 2 * hidden)?;
            Tensor::cat(&[&q, &k, &v], 0)
        });
        let qh = split(&qkv, 0).unwrap();
        let _ = stage("split_x3", &mut || {
            let q = split(&qkv, 0)?;
            let k = split(&qkv, hidden)?;
            let v = split(&qkv, 2 * hidden)?;
            Tensor::cat(&[&q, &k, &v], 0)
        });
        let _ = stage("rope_i_x2", &mut || {
            let qr = candle_nn::rotary_emb::rope_i(&qh, &cos, &sin)?;
            let kr = candle_nn::rotary_emb::rope_i(&qh, &cos, &sin)?;
            Tensor::cat(&[&qr, &kr], 0)
        });
        // The production Metal path since the split+rope fusion round: one
        // dispatch producing all three head-major planes. Bit-identical to
        // chain_new (gated in fused_qkv_split_rope_matches_chain_metal);
        // this stage is the region-level performance evidence.
        let _ = stage("chain_fused", &mut || {
            crate::metal_fused::qkv_split_rope(&qkv, &cos, &sin, heads, head_dim)
        });

        let q = q_new;
        let k = k_new;
        let v = v_new;
        let _ = stage("attn_manual", &mut || {
            let kt = k.transpose(2, 3)?.contiguous()?;
            let scores = q.matmul(&kt)?.affine(1.0 / (head_dim as f64).sqrt(), 0.0)?;
            candle_nn::ops::softmax_last_dim(&scores)?
                .matmul(&v)?
                .transpose(1, 2)?
                .contiguous()?
                .flatten(2, 3)
        });
        let _ = stage("attn_sdpa", &mut || {
            candle_nn::ops::sdpa(&q, &k, &v, None, false, 1.0 / (head_dim as f32).sqrt(), 1.0)?
                .transpose(1, 2)?
                .contiguous()?
                .flatten(2, 3)
        });
        let y = Tensor::randn(0f32, 1f32, (b, t, hidden), &device).unwrap();
        let _ = stage("wo_gemm", &mut || {
            y.flatten(0, 1)?.matmul(&w_o)?.reshape((b, t, hidden))
        });
        let _ = stage("ffn_gemm", &mut || {
            let a = y.flatten(0, 1)?.matmul(&w_ff)?.silu()?;
            let c = y.flatten(0, 1)?.matmul(&w_ff)?;
            let gated = a.broadcast_mul(&c)?;
            gated.matmul(&w_ff2)?.reshape((b, t, hidden))
        });
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "explicit real CUDA DiT stage diagnostic; requires local converted weights"]
    fn real_cuda_stage_diagnostic() -> Result<()> {
        assert!(
            crate::profile::enabled(),
            "set INDEXTTS_PROFILE=1 explicitly"
        );
        let root = std::env::var_os("INDEXTTS_CONVERTED_DIR").expect("local weights required");
        let root = std::path::Path::new(&root);
        let cfg = crate::config::Config::load(&root.join("config.json"))?;
        let device = Device::new_cuda(0)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[root.join("s2mel.safetensors")],
                DType::F32,
                &device,
            )?
        };
        let model = Dit::load(
            &vb.pp("models.cfm.estimator"),
            &cfg.s2mel.dit,
            &cfg.s2mel.wavenet,
        )?;
        for len in [64, 256] {
            let x = Tensor::full(0.125f32, (2, 160, len), &device)?;
            let cond = Tensor::full(0.25f32, (2, len, 512), &device)?;
            let style = Tensor::full(0.125f32, (2, STYLE_DIM), &device)?;
            let projected = model.project_condition(&cond)?;
            for round in 0..6 {
                let _report = crate::profile::report();
                let t = Tensor::full(round as f32 / 25.0, 2, &device)?;
                let output = model.forward_projected(&x, &t, &projected, &style, false)?;
                let values = output.flatten_all()?.to_vec1::<f32>()?;
                assert!(values.iter().all(|v| v.is_finite()));
                println!(
                    "{}",
                    serde_json::json!({"diagnostic":"real_dit_synthetic_inputs",
                    "length":len,"round":round,"warmup":round==0,"samples":values.len(),
                    "production_request":false})
                );
            }
        }
        Ok(())
    }
}
