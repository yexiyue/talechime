//! EnhancedCodec semantic codec (`indextts/codec/models.py`).
//!
//! Pipeline: optional Conv1d stride-2 downsample (+ GELU) → Vocos backbone
//! encoder (`indextts/codec/kmeans/vocos.py`, no AdaLayerNorm) + Linear →
//! factorized RVQ with L2-normalized codebook lookup (`quantizer_type="fvq"`,
//! 1 quantizer, codebook 8192×8, input 1024) → decoder backbone + Linear →
//! nearest-neighbor 2× upsample + Conv1d "up" projection.
//!
//! Weight names come verbatim from the Python state dict: `down.*`,
//! `encoder.*`, `decoder.*`, `quantizer.quantizers.{i}.*` (converted to
//! safetensors by the development export script; the checkpoint is stored
//! under key `model` in `codec.pth` and the export strips that prefix).
//! The factorized quantizer projections are torch `weight_norm` Conv1d
//! layers, so the state dict carries `weight_g`/`weight_v` pairs which are
//! folded here at load time (`w = g * v / ‖v‖₂` over the trailing dims).

use crate::config::SemanticCodecConfig;
use candle_core::{Result, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder, conv1d, embedding,
    layer_norm, linear,
};

/// LayerNorm epsilon used by every Vocos normalization (`eps=1e-6` upstream).
const VOCOS_LN_EPS: f64 = 1e-6;
/// `F.normalize` clamp floor used by the factorized codebook lookup.
const F_NORMALIZE_EPS: f64 = 1e-12;

/// CUDA workaround for Candle #3389, following PR #3531 at
/// `70bf01e7a56c8cc8d42649e78564d83930ab2cf7`. Keep the stock path on
/// CPU/Metal and outside the upstream launch/work thresholds.
struct DepthwiseConv1d {
    stock: Conv1d,
    /// Broadcast-ready `(1, C, K)` taps for eligible depthwise layers.
    taps: Option<Tensor>,
}

impl DepthwiseConv1d {
    fn new(stock: Conv1d) -> Result<Self> {
        let cfg = stock.config();
        let (channels, inputs_per_group, kernel) = stock.weight().dims3()?;
        let eligible = inputs_per_group == 1
            && cfg.groups == channels
            && cfg.stride == 1
            && cfg.dilation == 1
            && kernel > 0
            && 3 * cfg.groups * 2 >= (2 * kernel - 1) * 3;
        let taps = eligible
            .then(|| stock.weight().reshape((1, channels, kernel)))
            .transpose()?;
        Ok(Self { stock, taps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, length) = x.dims3()?;
        if let Some(taps) = &self.taps {
            let kernel = taps.dim(2)?;
            let padded_length = length + 2 * self.stock.config().padding;
            if let Some(output_length) = padded_length.checked_sub(kernel).map(|n| n + 1)
                && x.device().is_cuda()
                && channels == self.stock.config().groups
                && batch.saturating_mul(output_length).saturating_mul(kernel) <= 1 << 20
            {
                return self.forward_fast(x, taps, output_length);
            }
        }
        self.stock.forward(x)
    }

    fn forward_fast(&self, x: &Tensor, taps: &Tensor, output_length: usize) -> Result<Tensor> {
        let padding = self.stock.config().padding;
        let padded = if padding == 0 {
            x.clone()
        } else {
            x.pad_with_zeros(2, padding, padding)?
        };
        let mut out = padded
            .narrow(2, 0, output_length)?
            .broadcast_mul(&taps.narrow(2, 0, 1)?)?;
        for tap in 1..taps.dim(2)? {
            out = (out
                + padded
                    .narrow(2, tap, output_length)?
                    .broadcast_mul(&taps.narrow(2, tap, 1)?)?)?;
        }
        match self.stock.bias() {
            Some(bias) => out.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?),
            None => Ok(out),
        }
    }
}

/// Geometry of the codec, taken from [`SemanticCodecConfig`].
struct Dims {
    codebook_size: usize,
    hidden_size: usize,
    codebook_dim: usize,
    vocos_dim: usize,
    vocos_intermediate_dim: usize,
    vocos_num_layers: usize,
    num_quantizers: usize,
}

impl Dims {
    fn from_cfg(cfg: &SemanticCodecConfig) -> Result<Self> {
        if cfg.downsample_scale != 2 {
            candle_core::bail!(
                "semantic codec only implements downsample_scale=2 (got {})",
                cfg.downsample_scale
            );
        }
        if cfg.num_quantizers == 0 {
            candle_core::bail!("semantic codec requires at least one quantizer");
        }
        if cfg.vocos_num_layers == 0 {
            candle_core::bail!("semantic codec requires at least one Vocos layer");
        }
        if cfg.hidden_size == cfg.codebook_dim {
            candle_core::bail!(
                "factorized quantizer requires codebook_dim != hidden_size \
                 (the nn.Identity projection path is not shipped)"
            );
        }
        Ok(Self {
            codebook_size: cfg.codebook_size,
            hidden_size: cfg.hidden_size,
            codebook_dim: cfg.codebook_dim,
            vocos_dim: cfg.vocos_dim,
            vocos_intermediate_dim: cfg.vocos_intermediate_dim,
            vocos_num_layers: cfg.vocos_num_layers,
            num_quantizers: cfg.num_quantizers,
        })
    }
}

/// `torch.nn.utils.weight_norm` composition over all dims except dim 0:
/// `w = g * v / ‖v‖₂`.
fn fold_weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    v.broadcast_div(&norm)?.broadcast_mul(g)
}

/// Loads a kernel-1 Conv1d whose weight is either the verbatim weight-norm
/// pair `weight_g`/`weight_v` (folded here) or an already folded `weight`.
fn wn_conv1d(
    vb: &VarBuilder,
    name: &str,
    in_channels: usize,
    out_channels: usize,
) -> Result<Conv1d> {
    let vb = vb.pp(name);
    let pair = vb
        .get((out_channels, 1, 1), "weight_g")
        .and_then(|g| Ok((g, vb.get((out_channels, in_channels, 1), "weight_v")?)));
    let weight = match pair {
        Ok((g, v)) => fold_weight_norm(&g, &v)?,
        Err(_) => vb.get((out_channels, in_channels, 1), "weight")?,
    };
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(
        weight,
        Some(bias),
        Conv1dConfig {
            padding: 0,
            stride: 1,
            ..Default::default()
        },
    ))
}

/// Row-wise L2 normalization matching `F.normalize(x)` defaults (p=2, eps
/// 1e-12): `x / max(‖x‖₂, eps)`. Input is `(n, d)`.
fn l2_normalize_rows(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
    x.broadcast_div(&norm.clamp(F_NORMALIZE_EPS, f64::INFINITY)?)
}

/// Nearest-neighbor 2× upsampling along the time axis, matching
/// `F.interpolate(x, scale_factor=2, mode="nearest")` for `(B, C, T)`.
fn upsample_nearest_2x(x: &Tensor) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    x.unsqueeze(3)?
        .broadcast_as((b, c, t, 2))?
        .contiguous()?
        .reshape((b, c, t * 2))
}

/// ConvNeXt block (`vocos.py::ConvNeXtBlock`, non-conditional, with layer
/// scale). Input and output are `(B, C, T)`.
struct ConvNeXtBlock {
    dwconv: DepthwiseConv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn load(vb: VarBuilder, dim: usize, intermediate_dim: usize) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 3,
            groups: dim,
            ..Default::default()
        };
        let dwconv = DepthwiseConv1d::new(conv1d(dim, dim, 7, cfg, vb.pp("dwconv"))?)?;
        let norm = layer_norm(dim, VOCOS_LN_EPS, vb.pp("norm"))?;
        let pwconv1 = linear(dim, intermediate_dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(intermediate_dim, dim, vb.pp("pwconv2"))?;
        // Upstream always passes layer_scale_init_value = 1/num_layers > 0,
        // so `gamma` is always present in the checkpoint.
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.dwconv.forward(x)?;
        let x = x.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        let x = self.pwconv1.forward(&x)?;
        // nn.GELU() default (approximate="none") is the erf form.
        let x = x.gelu_erf()?;
        let x = self.pwconv2.forward(&x)?;
        let x = self.gamma.broadcast_mul(&x)?;
        let x = x.transpose(1, 2)?;
        residual + x
    }
}

/// Vocos backbone (`vocos.py::VocosBackbone`, `adanorm_num_embeddings=None`).
/// Input `(B, C, L)`, output `(B, L, dim)`.
struct VocosBackbone {
    embed: Conv1d,
    norm: LayerNorm,
    blocks: Vec<ConvNeXtBlock>,
    final_norm: LayerNorm,
}

impl VocosBackbone {
    fn load(
        vb: VarBuilder,
        input_channels: usize,
        dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let embed = conv1d(
            input_channels,
            dim,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("embed"),
        )?;
        let norm = layer_norm(dim, VOCOS_LN_EPS, vb.pp("norm"))?;
        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            blocks.push(ConvNeXtBlock::load(
                vb.pp(format!("convnext.{i}")),
                dim,
                intermediate_dim,
            )?);
        }
        let final_norm = layer_norm(dim, VOCOS_LN_EPS, vb.pp("final_layer_norm"))?;
        Ok(Self {
            embed,
            norm,
            blocks,
            final_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.embed.forward(x)?;
        let x = self.norm.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.final_norm.forward(&x.transpose(1, 2)?)
    }
}

/// `nn.Sequential(VocosBackbone, nn.Linear)` — the `encoder.0`/`encoder.1`
/// (and `decoder.*`) pairs of `EnhancedCodec`. Input `(B, hidden, T)`,
/// output `(B, T, hidden)`.
struct BackboneStack {
    backbone: VocosBackbone,
    proj: Linear,
}

impl BackboneStack {
    fn load(
        vb: VarBuilder,
        hidden_size: usize,
        vocos_dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
    ) -> Result<Self> {
        Ok(Self {
            backbone: VocosBackbone::load(
                vb.pp("0"),
                hidden_size,
                vocos_dim,
                intermediate_dim,
                num_layers,
            )?,
            proj: linear(vocos_dim, hidden_size, vb.pp("1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(&self.backbone.forward(x)?)
    }
}

/// Factorized vector quantizer
/// (`amphion_codec/quantize/factorized_vector_quantize.py`) in eval mode:
/// kernel-1 weight-normed Conv1d projections plus an L2-normalized
/// nearest-neighbor lookup whose output rows are the *raw* codebook entries.
struct FactorizedVQ {
    in_project: Conv1d,
    out_project: Conv1d,
    codebook: Embedding,
}

impl FactorizedVQ {
    fn load(
        vb: VarBuilder,
        input_dim: usize,
        codebook_size: usize,
        codebook_dim: usize,
    ) -> Result<Self> {
        let in_project = wn_conv1d(&vb, "in_project", input_dim, codebook_dim)?;
        let out_project = wn_conv1d(&vb, "out_project", codebook_dim, input_dim)?;
        let codebook = embedding(codebook_size, codebook_dim, vb.pp("codebook"))?;
        Ok(Self {
            in_project,
            out_project,
            codebook,
        })
    }

    /// `z`: `(B, D, T)`. Returns `(z_q (B, D, T), indices (B, T) u32)`.
    /// In eval mode the straight-through estimator is the identity, so this
    /// is just `out_project(codebook[nearest(z_e)])`.
    fn forward(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, _, t) = z.dims3()?;
        let codebook_dim = self.codebook.hidden_size();
        let z_e = self.in_project.forward(z)?;
        // einops "b d t -> (b t) d"
        let encodings = z_e
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * t, codebook_dim))?;
        // Distance and lookup operate on L2-normalized rows (cosine
        // distance); the quantized output uses the raw codebook rows.
        let encodings = l2_normalize_rows(&encodings)?;
        let codebook = l2_normalize_rows(self.codebook.embeddings())?;
        let e_sq = encodings.sqr()?.sum_keepdim(1)?;
        let c_sq = codebook.sqr()?.sum_keepdim(1)?.t()?;
        let cross = encodings.matmul(&codebook.t()?)?.affine(-2.0, 0.0)?;
        let dist = e_sq.broadcast_add(&cross)?.broadcast_add(&c_sq)?;
        // argmin(dist) == argmax(-dist); ties resolve to the lowest index,
        // matching torch.max on CPU.
        let indices = dist.affine(-1.0, 0.0)?.argmax(1)?.reshape((b, t))?;
        let z_q = self.codebook.forward(&indices)?.transpose(1, 2)?;
        let z_q = self.out_project.forward(&z_q)?;
        Ok((z_q, indices))
    }

    /// `vq2emb` with `out_proj=True`: codes `(B, T)` → `(B, D, T)`.
    fn vq2emb(&self, codes: &Tensor) -> Result<Tensor> {
        let emb = self.codebook.forward(codes)?.transpose(1, 2)?;
        self.out_project.forward(&emb)
    }
}

pub struct SemanticCodec {
    down: Conv1d,
    up: Conv1d,
    encoder: BackboneStack,
    decoder: BackboneStack,
    /// `ResidualVQ.quantizers`; the shipped configuration has exactly one.
    quantizers: Vec<FactorizedVQ>,
}

impl SemanticCodec {
    pub fn load(vb: &VarBuilder, cfg: &SemanticCodecConfig) -> Result<Self> {
        let dims = Dims::from_cfg(cfg)?;
        Self::build(vb, &dims)
    }

    fn build(vb: &VarBuilder, dims: &Dims) -> Result<Self> {
        let down = conv1d(
            dims.hidden_size,
            dims.hidden_size,
            3,
            Conv1dConfig {
                padding: 1,
                stride: 2,
                ..Default::default()
            },
            vb.clone().pp("down"),
        )?;
        let up = conv1d(
            dims.hidden_size,
            dims.hidden_size,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.clone().pp("up"),
        )?;
        let encoder = BackboneStack::load(
            vb.clone().pp("encoder"),
            dims.hidden_size,
            dims.vocos_dim,
            dims.vocos_intermediate_dim,
            dims.vocos_num_layers,
        )?;
        let decoder = BackboneStack::load(
            vb.clone().pp("decoder"),
            dims.hidden_size,
            dims.vocos_dim,
            dims.vocos_intermediate_dim,
            dims.vocos_num_layers,
        )?;
        let mut quantizers = Vec::with_capacity(dims.num_quantizers);
        for i in 0..dims.num_quantizers {
            quantizers.push(FactorizedVQ::load(
                vb.clone().pp(format!("quantizer.quantizers.{i}")),
                dims.hidden_size,
                dims.codebook_size,
                dims.codebook_dim,
            )?);
        }
        Ok(Self {
            down,
            up,
            encoder,
            decoder,
            quantizers,
        })
    }

    /// `x`: `(B, T, 1024)` continuous w2v-bert features.
    /// Returns `(codes (B, T/2) u32, quantized (B, 1024, T/2))` — the
    /// quantized tensor is channels-first (Python additionally transposes it
    /// to `(B, T/2, 1024)` before returning; this API keeps the layout the
    /// decoder consumes).
    ///
    /// Mirrors `EnhancedCodec.quantize`: down → gelu → encoder → FVQ.
    /// Odd `T` is NOT cropped: the Python `quantize` method only downsamples
    /// (unlike `forward`, which drops the last frame first), so the stride-2
    /// convolution emits `ceil(T/2)` frames.
    pub fn quantize(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // down → gelu → transpose: (B, T, D) -> (B, T', D)
        let x = self
            .down
            .forward(&x.transpose(1, 2)?)?
            .gelu_erf()?
            .transpose(1, 2)?;
        // encoder: backbone consumes (B, D, T'), returns (B, T', D).
        let mut residual = self.encoder.forward(&x.transpose(1, 2)?)?.transpose(1, 2)?;
        let mut quantized: Option<Tensor> = None;
        let mut indices = Vec::with_capacity(self.quantizers.len());
        for quantizer in &self.quantizers {
            let (z_q, idx) = quantizer.forward(&residual)?;
            indices.push(idx);
            quantized = match quantized {
                None => Some(z_q.clone()),
                Some(acc) => Some((acc + &z_q)?),
            };
            residual = (&residual - &z_q)?;
        }
        let quantized = quantized.expect("at least one quantizer (validated at load)");
        // `all_indices.squeeze(0)` for the single-quantizer configuration.
        let codes = if indices.len() == 1 {
            indices.pop().expect("one index tensor")
        } else {
            Tensor::stack(&indices, 0)?
        };
        Ok((codes, quantized))
    }

    /// `codes`: `(B, T)` semantic token ids.
    /// Returns the reconstructed continuous features `(B, 2T, 1024)`
    /// (vq2emb → decoder → nearest 2× upsample → `up` conv).
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let per_codebook: Vec<Tensor> = if codes.rank() == 2 {
            if self.quantizers.len() != 1 {
                candle_core::bail!(
                    "decoding (B, T) codes requires exactly one quantizer (got {})",
                    self.quantizers.len()
                );
            }
            vec![codes.clone()]
        } else if codes.rank() == 3 {
            if codes.dim(0)? != self.quantizers.len() {
                candle_core::bail!(
                    "codes lead dim {} does not match {} quantizers",
                    codes.dim(0)?,
                    self.quantizers.len()
                );
            }
            (0..self.quantizers.len())
                .map(|i| codes.narrow(0, i, 1)?.squeeze(0))
                .collect::<Result<Vec<_>>>()?
        } else {
            candle_core::bail!("codes must have rank 2 (B, T) or 3 (N, B, T)");
        };
        let mut quantized: Option<Tensor> = None;
        for (quantizer, codes) in self.quantizers.iter().zip(per_codebook) {
            let emb = quantizer.vq2emb(&codes)?;
            quantized = match quantized {
                None => Some(emb),
                Some(acc) => Some((acc + emb)?),
            };
        }
        let quantized = quantized.expect("at least one quantizer (validated at load)");
        // decoder: (B, D, T) -> (B, T, D), then upsample in channel-major.
        let x = self.decoder.forward(&quantized)?;
        let x = self
            .up
            .forward(&upsample_nearest_2x(&x.transpose(1, 2)?)?)?;
        x.transpose(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Var};
    use candle_nn::VarMap;

    /// Deterministic pseudo-random f32 in [-1, 1) from an LCG, so the
    /// fixed-input comparisons below are stable across runs and machines.
    fn lcg_fill(seed: u32, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((state >> 8) & 0xffff) as f32 / 32768.0 - 1.0
            })
            .collect()
    }

    /// Fixed-input cross-check of the depthwise fast path against the
    /// stock grouped `Conv1d`: shapes (incl. the shipped codec geometry),
    /// max abs/rel deviation, NaN/Inf absence and output lengths.
    #[test]
    fn depthwise_fast_path_matches_stock_conv() {
        check_depthwise_fast_path(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device; no models"]
    fn depthwise_fast_path_matches_stock_conv_cuda() {
        check_depthwise_fast_path(&Device::new_cuda(0).unwrap());
    }

    fn check_depthwise_fast_path(device: &Device) {
        // (batch, channels, length, kernel, padding, with bias)
        let cases: &[(usize, usize, usize, usize, usize, bool)] = &[
            (1, 384, 688, 7, 3, true),  // shipped decode geometry
            (1, 384, 1551, 7, 3, true), // long decode geometry
            (2, 16, 55, 7, 3, true),
            (1, 16, 3, 7, 3, true),   // boundary: l_out == l
            (1, 16, 6, 7, 3, true),   // boundary: l_out == l (even)
            (3, 24, 31, 5, 0, true),  // no padding
            (1, 24, 31, 5, 2, false), // no bias
        ];
        for &(b, c, l, k, padding, with_bias) in cases {
            let x =
                Tensor::from_vec(lcg_fill(0xA7 ^ c as u32, b * c * l), (b, c, l), device).unwrap();
            let flat_w = lcg_fill(0x5EED ^ k as u32, c * k);
            let weight = Tensor::from_vec(flat_w, (c, 1, k), device).unwrap();
            let bias = if with_bias {
                Some(Tensor::from_vec(lcg_fill(0xB1A5, c), (c,), device).unwrap())
            } else {
                None
            };
            let cfg = Conv1dConfig {
                padding,
                groups: c,
                ..Default::default()
            };
            let stock = Conv1d::new(weight.clone(), bias.clone(), cfg);
            let fast = DepthwiseConv1d::new(stock.clone()).unwrap();
            let want = stock.forward(&x).unwrap();
            let l_out = l + 2 * padding - (k - 1);
            let got = fast
                .forward_fast(&x, fast.taps.as_ref().expect("eligible"), l_out)
                .unwrap();
            let dispatched = fast.forward(&x).unwrap();
            let dispatched_values = dispatched.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let selected = if device.is_cuda() { &got } else { &want };
            assert_eq!(
                dispatched_values,
                selected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
            assert_eq!(got.dims(), vec![b, c, l_out], "shape for case {cases:?}");
            let w = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut max_abs = 0f64;
            let mut max_rel = 0f64;
            for (gv, wv) in g.iter().zip(&w) {
                assert!(gv.is_finite(), "NaN/Inf in fast path output");
                let a = (gv - wv).abs() as f64;
                max_abs = max_abs.max(a);
                max_rel = max_rel.max(a / wv.abs().max(1e-6) as f64);
            }
            // Summation-order deviation only (same class as candle's own
            // CPU-vs-CUDA conv difference). The absolute bound is the
            // meaningful criterion for O(1) inputs; the relative figure is
            // reported but not asserted because near-zero reference values
            // inflate it without indicating error.
            assert!(
                max_abs <= 1e-5,
                "case {cases:?}: max_abs {max_abs:.3e}, max_rel {max_rel:.3e}"
            );
            println!(
                "depthwise case (b={b},c={c},l={l},k={k},p={padding},bias={with_bias}): \
                 max_abs {max_abs:.3e}, max_rel {max_rel:.3e} over {} values",
                g.len()
            );
        }
    }

    /// Non-eligible configs (channel multiplier, stride, tiny groups) keep
    /// the stock path.
    #[test]
    fn depthwise_fast_path_rejects_ineligible_configs() {
        let device = Device::Cpu;
        let c = 8;
        let weight = Tensor::from_vec(lcg_fill(3, c * 7), (c, 1, 7), &device).unwrap();
        let grouped_cfg = Conv1dConfig {
            padding: 3,
            groups: 4,
            ..Default::default()
        };
        let grouped_not_depthwise = Conv1d::new(weight.clone(), None, grouped_cfg);
        assert!(
            DepthwiseConv1d::new(grouped_not_depthwise)
                .unwrap()
                .taps
                .is_none()
        );
        let strided_cfg = Conv1dConfig {
            padding: 3,
            groups: c,
            stride: 2,
            ..Default::default()
        };
        let strided = Conv1d::new(weight, None, strided_cfg);
        assert!(DepthwiseConv1d::new(strided).unwrap().taps.is_none());
    }

    fn tiny_dims() -> Dims {
        Dims {
            codebook_size: 8,
            hidden_size: 16,
            codebook_dim: 4,
            vocos_dim: 8,
            vocos_intermediate_dim: 12,
            vocos_num_layers: 2,
            num_quantizers: 1,
        }
    }

    /// Inserts a tensor into the VarMap *before* model construction so that
    /// load-time weight-norm folding sees the seeded values (the fold is
    /// computed once during `load`).
    fn seed_var(varmap: &VarMap, name: &str, t: &Tensor) -> Result<()> {
        varmap
            .data()
            .lock()
            .unwrap()
            .insert(name.to_string(), Var::from_tensor(t)?);
        Ok(())
    }

    /// Points the weight-normed quantizer projections at non-degenerate
    /// directions so the VarMap zero-init cannot produce 0/0 folds.
    fn seed_quantizer_wn(varmap: &VarMap, dev: &Device) -> Result<()> {
        let dims = tiny_dims();
        seed_var(
            varmap,
            "quantizer.quantizers.0.in_project.weight_v",
            &Tensor::full(0.02f32, (dims.codebook_dim, dims.hidden_size, 1), dev)?,
        )?;
        seed_var(
            varmap,
            "quantizer.quantizers.0.in_project.weight_g",
            &Tensor::ones((dims.codebook_dim, 1, 1), DType::F32, dev)?,
        )?;
        seed_var(
            varmap,
            "quantizer.quantizers.0.out_project.weight_v",
            &Tensor::full(0.02f32, (dims.hidden_size, dims.codebook_dim, 1), dev)?,
        )?;
        seed_var(
            varmap,
            "quantizer.quantizers.0.out_project.weight_g",
            &Tensor::ones((dims.hidden_size, 1, 1), DType::F32, dev)?,
        )?;
        Ok(())
    }

    fn build_tiny(varmap: &VarMap, dev: &Device) -> Result<SemanticCodec> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, dev);
        SemanticCodec::build(&vb, &tiny_dims())
    }

    #[test]
    fn quantize_and_decode_shapes_match_python_layout() -> Result<()> {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        seed_quantizer_wn(&varmap, &dev)?;
        let codec = build_tiny(&varmap, &dev)?;

        let x = Tensor::randn(0f32, 1f32, (1, 10, 16), &dev)?;
        let (codes, quantized) = codec.quantize(&x)?;
        assert_eq!(codes.dims(), &[1, 5]);
        assert_eq!(codes.dtype(), DType::U32);
        assert_eq!(quantized.dims(), &[1, 16, 5]);
        assert!(
            quantized
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite())
        );

        let rec = codec.decode(&codes)?;
        assert_eq!(rec.dims(), &[1, 10, 16]);
        assert!(
            rec.flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite())
        );
        Ok(())
    }

    /// Python `EnhancedCodec.quantize` (unlike `forward`) does not crop an
    /// odd length, so the stride-2 conv yields ceil(T/2) frames.
    #[test]
    fn quantize_odd_length_is_not_cropped() -> Result<()> {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        seed_quantizer_wn(&varmap, &dev)?;
        let codec = build_tiny(&varmap, &dev)?;

        let x = Tensor::randn(0f32, 1f32, (2, 11, 16), &dev)?;
        let (codes, quantized) = codec.quantize(&x)?;
        assert_eq!(codes.dims(), &[2, 6]);
        assert_eq!(quantized.dims(), &[2, 16, 6]);
        // decode doubles whatever length it is given.
        assert_eq!(codec.decode(&codes)?.dims(), &[2, 12, 16]);
        Ok(())
    }

    /// Hand-computed factorized-VQ lookup: projections are the identity on
    /// the first two channels; the codebook is (1,0), (0,1), (1,1). The
    /// lookup must use cosine distance on normalized rows while emitting the
    /// RAW codebook rows.
    #[test]
    fn fvq_lookup_matches_hand_computed_l2_math() -> Result<()> {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        // v = desired weight (unit rows), g = 1 => folded weight == desired.
        seed_var(
            &varmap,
            "fvq.in_project.weight_v",
            &Tensor::from_vec(vec![1f32, 0., 0., 0., 0., 1., 0., 0.], (2, 4, 1), &dev)?,
        )?;
        seed_var(
            &varmap,
            "fvq.in_project.weight_g",
            &Tensor::ones((2, 1, 1), DType::F32, &dev)?,
        )?;
        seed_var(
            &varmap,
            "fvq.out_project.weight_v",
            // Rows 2/3 keep a nonzero direction; their g scalars below zero
            // them out so the folded rows are exactly (0, 0) (a zero
            // `weight_v` row would fold to 0/0 = NaN).
            &Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 0., 1., 0.], (4, 2, 1), &dev)?,
        )?;
        seed_var(
            &varmap,
            "fvq.out_project.weight_g",
            &Tensor::from_vec(vec![1f32, 1., 0., 0.], (4, 1, 1), &dev)?,
        )?;
        seed_var(
            &varmap,
            "fvq.codebook.weight",
            &Tensor::from_vec(vec![1f32, 0., 0., 1., 1., 1.], (3, 2), &dev)?,
        )?;
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let fvq = FactorizedVQ::load(vb.pp("fvq"), 4, 3, 2)?;

        // Column t keeps only its first two channels after in_project.
        // normalized: (1,0), (0.7071, 0.7071), (0.0499, 0.9988).
        // Nearest codebook rows: 0, 2, 1.
        let z = Tensor::from_vec(
            vec![3f32, 1., 0.1, 0., 1., 2., 0., 0., 0., 0., 0., 0.],
            (1, 4, 3),
            &dev,
        )?;
        let (z_q, codes) = fvq.forward(&z)?;
        assert_eq!(codes.to_vec2::<u32>()?, &[[0, 2, 1]]);
        // out_project(cb row) per time step, with RAW (unnormalized) rows:
        // W·(1,0)=(1,0,0,0), W·(1,1)=(1,1,0,0), W·(0,1)=(0,1,0,0).
        let expected = vec![[[1f32, 1., 0.], [0., 1., 1.], [0., 0., 0.], [0., 0., 0.]]];
        let got = z_q.to_vec3::<f32>()?;
        for (got_b, exp_b) in got.iter().zip(expected.iter()) {
            for (got_r, exp_r) in got_b.iter().zip(exp_b.iter()) {
                for (g, e) in got_r.iter().zip(exp_r.iter()) {
                    assert!(
                        (g - e).abs() < 1e-6,
                        "quantized mismatch: got {got:?}, expected {expected:?}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn weight_norm_folding_composes_g_over_norm() -> Result<()> {
        let dev = Device::Cpu;
        let g = Tensor::from_vec(vec![10f32], (1, 1, 1), &dev)?;
        let v = Tensor::from_vec(vec![3f32, 4.], (1, 2, 1), &dev)?;
        let folded = fold_weight_norm(&g, &v)?;
        // 10 * (3,4)/5 = (6,8)
        assert_eq!(folded.flatten_all()?.to_vec1::<f32>()?, vec![6., 8.]);
        Ok(())
    }

    #[test]
    fn l2_normalize_matches_torch_f_normalize() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![3f32, 4., 0., 0.], (2, 2), &dev)?;
        let normalized = l2_normalize_rows(&x)?;
        assert_eq!(normalized.to_vec2::<f32>()?, &[[0.6f32, 0.8], [0., 0.]]);
        Ok(())
    }

    #[test]
    fn upsample_nearest_matches_torch_interpolate() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2., 3.], (1, 1, 3), &dev)?;
        let up = upsample_nearest_2x(&x)?;
        assert_eq!(up.to_vec3::<f32>()?, &[[[1f32, 1., 2., 2., 3., 3.]]]);
        Ok(())
    }

    #[test]
    #[ignore = "requires the converted checkpoint; set INDEXTTS_CONVERTED_DIR"]
    fn real_checkpoint_smoke() -> Result<()> {
        let Some(dir) = std::env::var_os("INDEXTTS_CONVERTED_DIR") else {
            return Ok(());
        };
        let path = std::path::Path::new(&dir).join("codec.safetensors");
        if !path.is_file() {
            return Ok(());
        }
        let dev = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &dev)? };
        let codec = SemanticCodec::load(&vb, &SemanticCodecConfig::default())?;
        let x = Tensor::randn(0f32, 1f32, (1, 40, 1024), &dev)?;
        let (codes, quantized) = codec.quantize(&x)?;
        assert_eq!(codes.dims(), &[1, 20]);
        assert_eq!(quantized.dims(), &[1, 1024, 20]);
        assert!(
            quantized
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite())
        );
        let rec = codec.decode(&codes)?;
        assert_eq!(rec.dims(), &[1, 40, 1024]);
        Ok(())
    }
}
