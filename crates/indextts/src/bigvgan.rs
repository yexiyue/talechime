//! BigVGAN v2 vocoder (`indextts/s2mel/modules/bigvgan/bigvgan.py`,
//! nvidia/bigvgan_v2_22khz_80band_256x).
//!
//! Torch path only (no CUDA fused kernels): snake activations, anti-aliased
//! activation via up/down sampling FIR, weight-norm convs folded at load
//! time (upstream calls `remove_weight_norm()` after loading).
//!
//! # Weight-norm folding
//!
//! `torch.nn.utils.weight_norm(conv, dim=0)` stores `weight_v` (the conv
//! weight shape) and `weight_g` of shape `(X, 1, 1)` where `X` is the size
//! of axis 0 — per output channel for `Conv1d` and per input channel for
//! `ConvTranspose1d`; the norm runs over all remaining axes. The effective
//! weight folded at load (what `remove_weight_norm` materializes) is
//! `w = weight_v * weight_g / ||weight_v||_2` with the norm over axes
//! `(1, 2)` (verified against torch 2.8 `remove_weight_norm`).
//!
//! The converted `bigvgan.safetensors` keeps the original `bigvgan`
//! checkpoint names; the state dict lives under a `generator.` prefix which
//! is stripped transparently when present.

use candle_core::{Device, Result, Tensor, conv::CudnnFwdAlgo};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};
use serde::Deserialize;
use std::f64::consts::PI;

/// Anti-aliased `Activation1d` defaults (ratio 2, kernel 12).
const AA_RATIO: usize = 2;
const AA_KERNEL: usize = 12;

#[derive(Debug, Deserialize)]
struct BigVganConfig {
    upsample_rates: Vec<usize>,
    upsample_kernel_sizes: Vec<usize>,
    upsample_initial_channel: usize,
    resblock_kernel_sizes: Vec<usize>,
    resblock_dilation_sizes: Vec<Vec<usize>>,
    resblock: String,
    num_mels: usize,
    activation: String,
    #[serde(default)]
    snake_logscale: bool,
    // Defaults mirror the `h.get(..., True)` fallbacks in bigvgan.py.
    #[serde(default = "default_true")]
    use_tanh_at_final: bool,
    #[serde(default = "default_true")]
    use_bias_at_final: bool,
}

fn default_true() -> bool {
    true
}

/// Modified Bessel function of the first kind, order 0 (power series; the
/// same scheme as `crate::dsp`).
fn i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..64 {
        term *= (x / (2.0 * k as f64)) * (x / (2.0 * k as f64));
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

/// `kaiser_sinc_filter1d` from `alias_free_activation/torch/filter.py`:
/// Kaiser-windowed sinc low-pass, normalized to unit sum. `torch.sinc` is
/// `sin(pi x) / (pi x)` with `sinc(0) = 1`.
fn kaiser_sinc_filter1d(cutoff: f64, half_width: f64, kernel_size: usize) -> Vec<f32> {
    let even = kernel_size.is_multiple_of(2);
    let half_size = kernel_size / 2;
    // Kaiser beta from the stop-band attenuation estimate.
    let delta_f = 4.0 * half_width;
    let a = 2.285 * (half_size as f64 - 1.0) * PI * delta_f + 7.95;
    let beta = if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.07886 * (a - 21.0)
    } else {
        0.0
    };
    let denom = i0(beta);
    let sinc = |x: f64| {
        if x == 0.0 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        }
    };
    let filter: Vec<f64> = (0..kernel_size)
        .map(|n| {
            // torch.kaiser_window(kernel_size, beta, periodic=False).
            let r = (2.0 * n as f64 / (kernel_size - 1) as f64) - 1.0;
            let window = i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom;
            // Even kernels sample the half-sample grid, odd ones the
            // symmetric grid around 0.
            let t = if even {
                (n as f64 - half_size as f64) + 0.5
            } else {
                n as f64 - half_size as f64
            };
            2.0 * cutoff * window * sinc(2.0 * cutoff * t)
        })
        .collect();
    let sum: f64 = filter.iter().sum();
    filter.iter().map(|v| (v / sum) as f32).collect()
}

/// Shared-channel FIR kernels, applied as ungrouped convolutions over
/// `(B*C, 1, L)` to avoid per-channel launches and per-tap tensor sweeps.
struct FirFilter {
    conv_direct: Tensor,
    /// Reversed odd/even taps for the two upsampling phases.
    conv_odd_rev: Tensor,
    conv_even_rev: Tensor,
}

impl FirFilter {
    fn new(taps: &[f32], device: &Device, dtype: candle_core::DType) -> Result<Self> {
        let kernel = |subset: &[f32], rev: bool| -> Result<Tensor> {
            let mut v: Vec<f32> = subset.to_vec();
            if rev {
                v.reverse();
            }
            Tensor::from_vec(v, (1, 1, subset.len()), device)?.to_dtype(dtype)
        };
        let even: Vec<f32> = (0..taps.len()).step_by(2).map(|i| taps[i]).collect();
        let odd: Vec<f32> = (1..taps.len()).step_by(2).map(|i| taps[i]).collect();
        Ok(Self {
            conv_direct: kernel(taps, false)?,
            conv_odd_rev: kernel(&odd, true)?,
            conv_even_rev: kernel(&even, true)?,
        })
    }
}

/// `out[m] = Σ_i kernel[i] · x[start + m - i]` for `m in 0..len`: one slice of a
/// full convolution, matching `np.convolve(x, kernel)[start .. start + len]`
/// (callers guarantee `start ≥ kernel.len() - 1` so no zero padding is
/// needed). `w` is the `(1, 1, K)` correlation kernel of the *reversed*
/// taps; the convolution is the batched `conv1d` over `(B*C, 1, L)`.
fn fir_conv_sum(x: &Tensor, w: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    let k = w.dim(2)?;
    if start + 1 < k {
        candle_core::bail!("fir slice start {start} too small for kernel {k}");
    }
    let (b, c, l) = x.dims3()?;
    // Auto selection exceeded the per-tap tolerance at long CUDA shapes.
    // Keep the FIR within the per-tap reference tolerance (see CUDA test).
    let conv = x.reshape((b * c, 1, l))?.conv1d_with_algo(
        w,
        0,
        1,
        1,
        1,
        Some(CudnnFwdAlgo::ImplicitGemm),
    )?;
    conv.narrow(2, start + 1 - k, len)?.reshape((b, c, len))
}

/// Strided (stride 2) depth-wise FIR over the last axis, matching
/// `LowPassFilter1d`: replicate padding `(k/2 - 1, k/2)` for the even
/// kernels used here, then `F.conv1d(x, filter, stride=2, groups=C)`:
/// `out[m] = Σ_k kernel[k] · x_pad[2m + k]` — one batched stride-2 `conv1d`.
fn fir_downsample(x: &Tensor, filter: &FirFilter) -> Result<Tensor> {
    let (b, c, l) = x.dims3()?;
    let k = filter.conv_direct.dim(2)?;
    let pad_left = k / 2 - usize::from(k.is_multiple_of(2));
    let pad_right = k / 2;
    let x = x.pad_with_same(2, pad_left, pad_right)?;
    let l = l + pad_left + pad_right;
    if l < k {
        candle_core::bail!("fir input length {l} shorter than kernel {k}");
    }
    let out_len = (l - k) / 2 + 1;
    let conv = x.reshape((b * c, 1, l))?.conv1d_with_algo(
        &filter.conv_direct,
        0,
        2,
        1,
        1,
        Some(CudnnFwdAlgo::ImplicitGemm),
    )?;
    debug_assert_eq!(conv.dims(), vec![b * c, 1, out_len]);
    conv.reshape((b, c, out_len))
}

/// Stride-2 up-FIR matching `ratio * F.conv_transpose1d(x, filter, stride=2,
/// groups=C)` followed by the `[pad_left : -pad_right]` crop in
/// `UpSample1d.forward` (output length is exactly `ratio * T`).
///
/// The transposed convolution decomposes into two stride-1 convolution
/// phases on the replicate-padded input `x'`: output index `2m` only sees
/// the odd filter taps (`y[2m] = Σ w[2i+1]·x'[m-i]`) and `2m+1` only the
/// even taps. The crop window starts at `pad_left` (odd, 15), so global even
/// outputs begin at phase index 7 and odd outputs at 8. Validated against
/// the torch `UpSample1d` module on multiple lengths.
fn fir_upsample(x: &Tensor, filter: &FirFilter) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    let pad = AA_KERNEL / AA_RATIO - 1; // 5
    let crop_left = pad * AA_RATIO + (AA_KERNEL - AA_RATIO) / 2; // 15
    let crop_right = pad * AA_RATIO + (AA_KERNEL - AA_RATIO).div_ceil(2); // 15
    // L_out of the uncropped conv_transpose; the crops leave exactly 2T.
    let l_out = (t + 2 * pad - 1) * AA_RATIO + AA_KERNEL;
    debug_assert_eq!(l_out - crop_left - crop_right, AA_RATIO * t);
    let xp = x.pad_with_same(2, pad, pad)?;
    let odd_phase = fir_conv_sum(&xp, &filter.conv_odd_rev, crop_left / 2, t)?;
    let even_phase = fir_conv_sum(&xp, &filter.conv_even_rev, crop_left.div_ceil(2), t)?;
    let y = Tensor::stack(&[odd_phase, even_phase], 3)?.reshape((b, c, AA_RATIO * t))?;
    y.affine(AA_RATIO as f64, 0.0)
}

/// `Snake`: `x + (1/(a+eps)) * sin^2(a x)`, `SnakeBeta`:
/// `x + (1/(b+eps)) * sin^2(a x)`; with `alpha_logscale` both parameters are
/// stored log-scale and exponentiated (`no_div_by_zero = 1e-9`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum SnakeKind {
    Snake,
    SnakeBeta,
}

struct Activation1d {
    logscale: bool,
    alpha: Tensor,        // (1, C, 1), raw storage scale
    beta: Option<Tensor>, // SnakeBeta only
}

impl Activation1d {
    fn load(vb: &VarBuilder, channels: usize, kind: SnakeKind, logscale: bool) -> Result<Self> {
        let alpha = vb
            .get((channels,), "act.alpha")?
            .reshape((1, channels, 1))?;
        let beta = if kind == SnakeKind::SnakeBeta {
            Some(vb.get((channels,), "act.beta")?.reshape((1, channels, 1))?)
        } else {
            None
        };
        Ok(Self {
            logscale,
            alpha,
            beta,
        })
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        let a = if self.logscale {
            self.alpha.exp()?
        } else {
            self.alpha.clone()
        };
        let s = x.broadcast_mul(&a)?.sin()?.sqr()?;
        let scale = match &self.beta {
            None => (a + 1e-9)?.recip()?,
            Some(b) => {
                let b = if self.logscale { b.exp()? } else { b.clone() };
                (b + 1e-9)?.recip()?
            }
        };
        x.broadcast_add(&s.broadcast_mul(&scale)?)
    }

    /// up-FIR -> snake -> down-FIR (`alias_free_activation/torch/act.py`).
    fn forward(&self, x: &Tensor, filter: &FirFilter) -> Result<Tensor> {
        let up = {
            let _g = crate::profile::scope(Some(x.device()), "bv.fir_up");
            fir_upsample(x, filter)?
        };
        let up = {
            let _g = crate::profile::scope(Some(x.device()), "bv.snake");
            self.apply_activation(&up)?
        };
        let _g = crate::profile::scope(Some(x.device()), "bv.fir_down");
        fir_downsample(&up, filter)
    }
}

/// AMPBlock (resblock "1" keeps the dilation-1 `convs2` pass, "2" does not).
struct AmpBlock {
    variant2: bool,
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    activations: Vec<Activation1d>,
}

impl AmpBlock {
    fn forward(&self, x: &Tensor, filter: &FirFilter) -> Result<Tensor> {
        let mut x = x.clone();
        let conv = |xt: &Tensor, conv: &Conv1d| -> Result<Tensor> {
            let _g = crate::profile::scope(Some(xt.device()), "bv.res_conv");
            conv.forward(xt)
        };
        if self.variant2 {
            for (c, act) in self.convs1.iter().zip(&self.activations) {
                let xt = act.forward(&x, filter)?;
                let xt = conv(&xt, c)?;
                x = (xt + x)?;
            }
        } else {
            // activations[::2] precede convs1, [1::2] precede convs2.
            let mut i = 0;
            while i + 1 < self.activations.len() {
                let a1 = &self.activations[i];
                let a2 = &self.activations[i + 1];
                let c1 = &self.convs1[i / 2];
                let c2 = &self.convs2[i / 2];
                let xt = a1.forward(&x, filter)?;
                let xt = conv(&xt, c1)?;
                let xt = a2.forward(&xt, filter)?;
                let xt = conv(&xt, c2)?;
                x = (xt + x)?;
                i += 2;
            }
        }
        Ok(x)
    }
}

/// Fold `weight_v`/`weight_g` into the effective conv weight (see the
/// module docs); `v_shape` is the full conv weight shape.
fn fold_weight_norm(
    vb: &VarBuilder,
    prefix: &str,
    v_shape: (usize, usize, usize),
) -> Result<Tensor> {
    let v = vb.get(v_shape, &format!("{prefix}.weight_v"))?;
    let g = vb.get((v_shape.0, 1, 1), &format!("{prefix}.weight_g"))?;
    let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    v.broadcast_mul(&g)?.broadcast_div(&norm)
}

fn load_conv1d_folded(
    vb: &VarBuilder,
    prefix: &str,
    (in_c, out_c): (usize, usize),
    kernel: usize,
    padding: usize,
    dilation: usize,
    bias: bool,
) -> Result<Conv1d> {
    let w = fold_weight_norm(vb, prefix, (out_c, in_c, kernel))?;
    let bias = if bias && vb.contains_tensor(&format!("{prefix}.bias")) {
        Some(vb.get((out_c,), &format!("{prefix}.bias"))?)
    } else {
        None
    };
    Ok(Conv1d::new(
        w,
        bias,
        Conv1dConfig {
            padding,
            dilation,
            ..Default::default()
        },
    ))
}

fn load_conv_transpose_folded(
    vb: &VarBuilder,
    prefix: &str,
    (in_c, out_c): (usize, usize),
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<ConvTranspose1d> {
    // torch layout: (in_channels, out_channels, kernel); weight norm runs
    // per input channel (dim 0).
    let w = fold_weight_norm(vb, prefix, (in_c, out_c, kernel))?;
    let bias = if vb.contains_tensor(&format!("{prefix}.bias")) {
        Some(vb.get((out_c,), &format!("{prefix}.bias"))?)
    } else {
        None
    };
    Ok(ConvTranspose1d::new(
        w,
        bias,
        ConvTranspose1dConfig {
            padding,
            stride,
            ..Default::default()
        },
    ))
}

/// `get_padding(kernel_size, dilation)` from bigvgan utils.py.
fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

pub struct BigVgan {
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    resblocks: Vec<AmpBlock>,
    activation_post: Activation1d,
    conv_post: Conv1d,
    num_kernels: usize,
    use_tanh_at_final: bool,
    /// Shared kaiser-sinc taps; the up- and down-sampling filters are
    /// identical for the (ratio 2, kernel 12) hyperparameters.
    filter: FirFilter,
}

impl BigVgan {
    /// `vb` over the converted `bigvgan_generator.pt` state dict;
    /// `config_json` is the raw contents of BigVGAN's `config.json`.
    pub fn load(vb: &VarBuilder, config_json: &str) -> Result<Self> {
        let cfg: BigVganConfig = serde_json::from_str(config_json)
            .map_err(|e| candle_core::Error::Msg(format!("parse bigvgan config.json: {e}")))?;
        let kind = match cfg.activation.as_str() {
            "snake" => SnakeKind::Snake,
            "snakebeta" => SnakeKind::SnakeBeta,
            other => candle_core::bail!("unsupported bigvgan activation {other:?}"),
        };
        if cfg.upsample_rates.len() != cfg.upsample_kernel_sizes.len() {
            candle_core::bail!("upsample_rates/upsample_kernel_sizes length mismatch");
        }
        // The exported checkpoint nests the state dict under "generator".
        let vb = if vb.contains_tensor("conv_pre.weight_v") {
            vb.clone()
        } else {
            vb.pp("generator")
        };

        let c0 = cfg.upsample_initial_channel;
        let conv_pre = load_conv1d_folded(&vb, "conv_pre", (cfg.num_mels, c0), 7, 3, 1, true)?;

        let variant2 = match cfg.resblock.as_str() {
            "1" => false,
            "2" => true,
            other => candle_core::bail!("unsupported bigvgan resblock {other:?}"),
        };

        let mut ups = Vec::with_capacity(cfg.upsample_rates.len());
        for (i, (&u, &k)) in cfg
            .upsample_rates
            .iter()
            .zip(&cfg.upsample_kernel_sizes)
            .enumerate()
        {
            let in_c = c0 >> i;
            let out_c = c0 >> (i + 1);
            ups.push(load_conv_transpose_folded(
                &vb,
                &format!("ups.{i}.0"),
                (in_c, out_c),
                k,
                u,
                (k - u) / 2,
            )?);
        }

        let num_kernels = cfg.resblock_kernel_sizes.len();
        let mut resblocks = Vec::with_capacity(ups.len() * num_kernels);
        for i in 0..ups.len() {
            let ch = c0 >> (i + 1);
            for (j, &k) in cfg.resblock_kernel_sizes.iter().enumerate() {
                let rb = vb.pp(format!("resblocks.{}", i * num_kernels + j));
                let dilations = cfg
                    .resblock_dilation_sizes
                    .get(j)
                    .ok_or_else(|| candle_core::Error::Msg("missing resblock dilations".into()))?;
                let mut convs1 = Vec::with_capacity(dilations.len());
                let mut convs2 = Vec::with_capacity(dilations.len());
                let mut activations =
                    Vec::with_capacity(dilations.len() * if variant2 { 1 } else { 2 });
                for (d_i, &d) in dilations.iter().enumerate() {
                    convs1.push(load_conv1d_folded(
                        &rb,
                        &format!("convs1.{d_i}"),
                        (ch, ch),
                        k,
                        get_padding(k, d),
                        d,
                        true,
                    )?);
                    if !variant2 {
                        convs2.push(load_conv1d_folded(
                            &rb,
                            &format!("convs2.{d_i}"),
                            (ch, ch),
                            k,
                            get_padding(k, 1),
                            1,
                            true,
                        )?);
                    }
                }
                let num_layers = dilations.len() * if variant2 { 1 } else { 2 };
                for a_i in 0..num_layers {
                    activations.push(Activation1d::load(
                        &rb.pp(format!("activations.{a_i}")),
                        ch,
                        kind,
                        cfg.snake_logscale,
                    )?);
                }
                resblocks.push(AmpBlock {
                    variant2,
                    convs1,
                    convs2,
                    activations,
                });
            }
        }

        let last_ch = c0 >> cfg.upsample_rates.len();
        let activation_post =
            Activation1d::load(&vb.pp("activation_post"), last_ch, kind, cfg.snake_logscale)?;
        // conv_post is built with bias=False when use_bias_at_final=false.
        let conv_post = load_conv1d_folded(
            &vb,
            "conv_post",
            (last_ch, 1),
            7,
            3,
            1,
            cfg.use_bias_at_final,
        )?;
        let fir_filter = FirFilter::new(
            &kaiser_sinc_filter1d(0.5 / AA_RATIO as f64, 0.6 / AA_RATIO as f64, AA_KERNEL),
            conv_pre.weight().device(),
            conv_pre.weight().dtype(),
        )?;

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            activation_post,
            conv_post,
            num_kernels,
            use_tanh_at_final: cfg.use_tanh_at_final,
            filter: fir_filter,
        })
    }

    /// `mel`: `(B, 80, T)`. Returns waveform `(B, 1, samples)`, f32.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        if mel.dim(2)? == 0 {
            candle_core::bail!("empty mel input");
        }
        let mut x = {
            let _g = crate::profile::scope(Some(mel.device()), "bv.pre");
            self.conv_pre.forward(mel)?
        };
        for (i, up) in self.ups.iter().enumerate() {
            x = {
                let _g = crate::profile::scope(Some(mel.device()), "bv.up_conv");
                up.forward(&x)?
            };
            let mut sum: Option<Tensor> = None;
            for j in 0..self.num_kernels {
                let y = self.resblocks[i * self.num_kernels + j].forward(&x, &self.filter)?;
                sum = Some(match sum {
                    Some(s) => (s + y)?,
                    None => y,
                });
            }
            x = sum
                .expect("resblocks")
                .affine(1.0 / self.num_kernels as f64, 0.0)?;
        }
        x = self.activation_post.forward(&x, &self.filter)?;
        x = {
            let _g = crate::profile::scope(Some(mel.device()), "bv.post");
            let x = self.conv_post.forward(&x)?;
            if self.use_tanh_at_final {
                x.tanh()?
            } else {
                // use_tanh_at_final=false in the shipped config: clamp to [-1, 1].
                x.clamp(-1.0, 1.0)?
            }
        };
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn kaiser_sinc_filter_matches_torch() {
        let f = kaiser_sinc_filter1d(0.25, 0.3, 12);
        // Oracle: kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, 12).
        let want = [
            0.0020289647f32,
            0.009389466,
            -0.025543459,
            -0.057657384,
            0.12857258,
            0.4432098,
        ];
        for (got, w) in f.iter().zip(want.iter()) {
            assert!(close(*got, *w, 1e-6), "tap {got} vs {w}");
        }
        // Symmetric, unit-sum (f32 normalization).
        for i in 0..6 {
            assert!(close(f[i], f[11 - i], 1e-7));
        }
        assert!(close(f.iter().sum::<f32>(), 1.0, 1e-6));
    }

    /// Per-tap reference implementations (the elementwise FIR before the
    /// batched-conv rewrite), kept for fixed-input cross-checking.
    fn fir_conv_sum_reference(
        x: &Tensor,
        kernel: &[f32],
        start: usize,
        len: usize,
    ) -> Result<Tensor> {
        let mut acc: Option<Tensor> = None;
        for (i, &w) in kernel.iter().enumerate() {
            let term = x.narrow(2, start - i, len)?.affine(w as f64, 0.0)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        Ok(acc.expect("non-empty kernel"))
    }

    fn fir_downsample_reference(x: &Tensor, kernel: &[f32]) -> Result<Tensor> {
        let (b, c, l) = x.dims3()?;
        let k = kernel.len();
        let pad_left = k / 2 - usize::from(k.is_multiple_of(2));
        let pad_right = k / 2;
        let x = x.pad_with_same(2, pad_left, pad_right)?;
        let l = l + pad_left + pad_right;
        let out_len = (l - k) / 2 + 1;
        let l2 = l - (l % 2);
        let pairs = x.narrow(2, 0, l2)?.reshape((b, c, l2 / 2, 2))?;
        let mut acc: Option<Tensor> = None;
        for (k_idx, &w) in kernel.iter().enumerate() {
            let part = pairs
                .narrow(2, k_idx / 2, out_len)?
                .narrow(3, k_idx % 2, 1)?
                .squeeze(3)?;
            let term = part.affine(w as f64, 0.0)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
        Ok(acc.expect("non-empty kernel"))
    }

    fn fir_upsample_reference(x: &Tensor, kernel: &[f32]) -> Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        let pad = AA_KERNEL / AA_RATIO - 1;
        let crop_left = pad * AA_RATIO + (AA_KERNEL - AA_RATIO) / 2;
        let even_taps: Vec<f32> = (0..AA_KERNEL).step_by(2).map(|i| kernel[i]).collect();
        let odd_taps: Vec<f32> = (1..AA_KERNEL).step_by(2).map(|i| kernel[i]).collect();
        let xp = x.pad_with_same(2, pad, pad)?;
        let odd_phase = fir_conv_sum_reference(&xp, &odd_taps, crop_left / 2, t)?;
        let even_phase = fir_conv_sum_reference(&xp, &even_taps, crop_left.div_ceil(2), t)?;
        let y = Tensor::stack(&[odd_phase, even_phase], 3)?.reshape((b, c, AA_RATIO * t))?;
        y.affine(AA_RATIO as f64, 0.0)
    }

    /// Fixed-input cross-check of the batched-conv FIR against the per-tap
    /// reference: shapes/lengths, NaN/Inf absence and the summation-order
    /// deviation, over generic, boundary-short and near-model shapes.
    #[test]
    fn fir_conv_batched_matches_per_tap_reference() {
        check_fir_conv(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device; no models"]
    fn fir_conv_batched_matches_per_tap_reference_cuda() {
        check_fir_conv(&Device::new_cuda(0).unwrap());
    }

    fn check_fir_conv(device: &Device) {
        let taps = kaiser_sinc_filter1d(0.25, 0.3, AA_KERNEL);
        let filter = FirFilter::new(&taps, device, candle_core::DType::F32).unwrap();
        let mut state = 0x00C0_FFEE_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((state >> 16) as f32 / 32768.0 - 1.0) * 2.0
        };
        for (b, c, t) in [
            (1, 1, 1),
            (1, 4, 2),
            (2, 9, 111),
            (1, 1, 6),
            (1, 4, 7),
            (2, 128, 641),
            (1, 32, 22048),
        ] {
            let values: Vec<f32> = (0..b * c * t).map(|_| next()).collect();
            let x = Tensor::from_vec(values, (b, c, t), device).unwrap();
            for (name, got, want) in [
                (
                    "upsample",
                    fir_upsample(&x, &filter).unwrap(),
                    fir_upsample_reference(&x, &taps).unwrap(),
                ),
                (
                    "downsample",
                    fir_downsample(&x, &filter).unwrap(),
                    fir_downsample_reference(&x, &taps).unwrap(),
                ),
                (
                    "roundtrip",
                    fir_downsample(&fir_upsample(&x, &filter).unwrap(), &filter).unwrap(),
                    fir_downsample_reference(&fir_upsample_reference(&x, &taps).unwrap(), &taps)
                        .unwrap(),
                ),
            ] {
                assert_eq!(got.dims(), want.dims(), "{name} shape for ({b},{c},{t})");
                let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let w = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let mut max_abs = 0f64;
                for (gv, wv) in g.iter().zip(&w) {
                    assert!(gv.is_finite(), "{name}: NaN/Inf at ({b},{c},{t})");
                    max_abs = max_abs.max((gv - wv).abs() as f64);
                }
                assert!(
                    max_abs <= 1e-5,
                    "{name} ({b},{c},{t}): max_abs {max_abs:.3e}"
                );
                println!(
                    "fir {name} ({b},{c},{t}): max_abs {max_abs:.3e} over {} values",
                    g.len()
                );
            }
        }
    }

    #[test]
    fn fir_upsample_downsample_match_torch_modules() {
        // Oracle from UpSample1d(2,12) / DownSample1d(2,12) on
        // x = [1/8 .. 7/8] (shape (1, 1, 7)).
        let up_want = [
            0.1152804f32,
            0.1458931,
            0.2140158,
            0.2829601,
            0.3448945,
            0.4056128,
            0.4693872,
            0.5306128,
            0.5943872,
            0.6551055,
            0.7170398,
            0.7859841,
            0.8541068,
            0.8847196,
        ];
        let down_want = [
            0.1294543f32,
            0.2459031,
            0.376443,
            0.5,
            0.6235569,
            0.7540967,
            0.8705455,
        ];
        let filter = FirFilter::new(
            &kaiser_sinc_filter1d(0.25, 0.3, 12),
            &Device::Cpu,
            candle_core::DType::F32,
        )
        .unwrap();
        let x = Tensor::new(
            &[[[0.125f32, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]]],
            &Device::Cpu,
        )
        .unwrap();
        let up = fir_upsample(&x, &filter).unwrap();
        assert_eq!(up.dims(), vec![1, 1, 14]);
        let up_v = up.to_vec3::<f32>().unwrap();
        for (got, w) in up_v[0][0].iter().zip(up_want.iter()) {
            assert!(close(*got, *w, 1e-5), "up {got} vs {w}");
        }
        let down = fir_downsample(&up, &filter).unwrap();
        assert_eq!(down.dims(), vec![1, 1, 7]);
        let down_v = down.to_vec3::<f32>().unwrap();
        for (got, w) in down_v[0][0].iter().zip(down_want.iter()) {
            assert!(close(*got, *w, 1e-5), "down {got} vs {w}");
        }
        // Odd lengths keep ceil semantics: T=13 -> 26 -> 13.
        let x13 = Tensor::new(
            &[[[
                0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65, 0.75, 0.85,
            ]]],
            &Device::Cpu,
        )
        .unwrap();
        let up13 = fir_upsample(&x13, &filter).unwrap();
        assert_eq!(up13.dims(), vec![1, 1, 26]);
        let down13 = fir_downsample(&up13, &filter).unwrap();
        assert_eq!(down13.dims(), vec![1, 1, 13]);
    }

    fn snake(kind: SnakeKind, logscale: bool, alpha: f64, beta: f64) -> Activation1d {
        let dev = &Device::Cpu;
        let a = Tensor::from_vec(vec![alpha as f32], (1, 1, 1), dev).unwrap();
        let b = if kind == SnakeKind::SnakeBeta {
            Some(Tensor::from_vec(vec![beta as f32], (1, 1, 1), dev).unwrap())
        } else {
            None
        };
        Activation1d {
            logscale,
            alpha: a,
            beta: b,
        }
    }

    #[test]
    fn snake_values_match_activations_py() {
        // Oracles from activations.py with in_features=1, alpha_logscale=True,
        // alpha=exp(0.3), inputs [-0.75, 0, 1.25, 2.5].
        let act = snake(SnakeKind::Snake, true, 0.3, 0.0);
        let x = Tensor::new(&[[[-0.75f32, 0.0, 1.25, 2.5]]], &Device::Cpu).unwrap();
        let y = act.apply_activation(&x).unwrap();
        let want = [-0.2171454f32, 0.0, 1.9808044, 2.5395138];
        let v = y.to_vec3::<f32>().unwrap();
        for (got, w) in v[0][0].iter().zip(want.iter()) {
            assert!(close(*got, *w, 1e-5), "snake {got} vs {w}");
        }
        // SnakeBeta with alpha=exp(0.2), beta=exp(-0.1).
        let act = snake(SnakeKind::SnakeBeta, true, 0.2, -0.1);
        let y = act.apply_activation(&x).unwrap();
        let want = [-0.0546573f32, 0.0, 2.3530285, 2.508553];
        let v = y.to_vec3::<f32>().unwrap();
        for (got, w) in v[0][0].iter().zip(want.iter()) {
            assert!(close(*got, *w, 1e-5), "snakebeta {got} vs {w}");
        }
    }

    const TEST_CONFIG: &str = r#"{
        "resblock": "1",
        "upsample_rates": [4,4,2,2,2,2],
        "upsample_kernel_sizes": [8,8,4,4,4,4],
        "upsample_initial_channel": 64,
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
        "use_tanh_at_final": false,
        "use_bias_at_final": false,
        "activation": "snakebeta",
        "snake_logscale": true,
        "num_mels": 80
    }"#;

    #[test]
    fn bigvgan_varmap_shape() {
        // VarMap-backed VarBuilder zero-initializes every tensor on first
        // access, exercising the full name map (including the generator.
        // prefix fallback) without the real checkpoint.
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let model = BigVgan::load(&vb, TEST_CONFIG).unwrap();
        let mel = Tensor::zeros((1, 80, 3), DType::F32, &Device::Cpu).unwrap();
        let out = model.forward(&mel).unwrap();
        // Product of upsample_rates = 256 samples per mel frame.
        assert_eq!(out.dims(), vec![1, 1, 3 * 256]);
        // The zero-initialized norms yield NaN, so only shapes are checked
        // here; numeric parity is covered by the ignored real-weights test.
    }

    #[test]
    #[ignore = "requires the converted checkpoint; set INDEXTTS_CONVERTED_DIR"]
    fn bigvgan_real_weights() {
        let Some(dir) = std::env::var_os("INDEXTTS_CONVERTED_DIR") else {
            eprintln!("INDEXTTS_CONVERTED_DIR not set; skipping");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let config = std::fs::read_to_string(dir.join("bigvgan.config.json")).unwrap();
        let path = dir.join("bigvgan.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu) }
            .unwrap();
        let model = BigVgan::load(&vb, &config).unwrap();
        // Deterministic mel: smooth ramp, no NaN-inducing extremes.
        let (mels, frames) = (80usize, 8usize);
        let data: Vec<f32> = (0..mels * frames)
            .map(|i| -4.0 + 8.0 * (i % (mels * frames)) as f32 / (mels * frames) as f32)
            .collect();
        let mel = Tensor::from_vec(data, (1, mels, frames), &Device::Cpu).unwrap();
        let out = model.forward(&mel).unwrap();
        assert_eq!(out.dims(), vec![1, 1, frames * 256]);
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
        assert!(v.iter().all(|x| x.abs() <= 1.0 + 1e-4));
        let rms = (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        assert!(rms > 1e-3, "output rms {rms}");
    }
}
