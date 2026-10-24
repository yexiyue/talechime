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
    /// `(2, half_k)` forward-order AA_RATIO-scaled phase taps `[odd; even]`
    /// for the Metal fused up-FIR+snake kernel (row 0 odd, row 1 even).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fused_phases: Option<Tensor>,
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
        // The upsample phases are pre-scaled by AA_RATIO so `fir_upsample`
        // needs no full-tensor affine afterwards. Scaling by a power of two
        // commutes with rounding, so this is bit-identical to scaling the
        // accumulated phases afterwards on every device path.
        let scale = |mut taps: Vec<f32>| {
            taps.iter_mut().for_each(|v| *v *= AA_RATIO as f32);
            taps
        };
        // Forward-order scaled phases for the fused up-FIR+snake kernel;
        // only the (AA_KERNEL=12, AA_RATIO=2) hyperparameters it is gated to.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let fused_phases = if taps.len() == AA_KERNEL && AA_RATIO == 2 {
            let mut rows = scale(odd.clone());
            rows.extend(scale(even.clone()));
            Some(Tensor::from_vec(rows, (2, AA_KERNEL / 2), device)?.to_dtype(dtype)?)
        } else {
            None
        };
        Ok(Self {
            conv_direct: kernel(taps, false)?,
            conv_odd_rev: kernel(&scale(odd), true)?,
            conv_even_rev: kernel(&scale(even), true)?,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            fused_phases,
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
    // The AA_RATIO factor is folded into the phase kernels (see FirFilter).
    Tensor::stack(&[odd_phase, even_phase], 3)?.reshape((b, c, AA_RATIO * t))
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
    /// `(2, C)` Metal preload `[a; 1/(beta+1e-9)]` for the fused
    /// up-FIR+snake kernel (same expressions as `apply_activation`).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fused_acts: Option<Tensor>,
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
        // Metal preload for the fused up-FIR+snake kernel: identical
        // expressions to `apply_activation`'s broadcast chain.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let fused_acts = if alpha.device().is_metal() {
            let a = if logscale {
                alpha.exp()?
            } else {
                alpha.clone()
            };
            let scale = match (&kind, &beta) {
                (_, Some(b)) => {
                    let b = if logscale { b.exp()? } else { b.clone() };
                    (b + 1e-9)?.recip()?
                }
                _ => (a.clone() + 1e-9)?.recip()?,
            };
            Some(
                Tensor::cat(&[a.reshape((channels,))?, scale.reshape((channels,))?], 0)?
                    .reshape((2, channels))?,
            )
        } else {
            None
        };
        Ok(Self {
            logscale,
            alpha,
            beta,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            fused_acts,
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
        // Metal fused path: one kernel for pad+up-FIR+snake (bit-composed
        // from the same tap order and snake expressions; gated at 1e-5
        // relative in the bigvgan tests), then the stride-2 down-FIR.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let up = if x.device().is_metal() {
            match (&filter.fused_phases, &self.fused_acts) {
                (Some(phases), Some(acts)) => {
                    let _g = crate::profile::scope(Some(x.device()), "bv.fir_up_snake");
                    crate::metal_fused::fir_up_snake(x, phases, acts, AA_KERNEL / AA_RATIO)?
                }
                _ => self.up_snake_chain(x, filter)?,
            }
        } else {
            self.up_snake_chain(x, filter)?
        };
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let up = self.up_snake_chain(x, filter)?;
        let _g = crate::profile::scope(Some(x.device()), "bv.fir_down");
        fir_downsample(&up, filter)
    }

    fn up_snake_chain(&self, x: &Tensor, filter: &FirFilter) -> Result<Tensor> {
        let up = {
            let _g = crate::profile::scope(Some(x.device()), "bv.fir_up");
            fir_upsample(x, filter)?
        };
        let _g = crate::profile::scope(Some(x.device()), "bv.snake");
        self.apply_activation(&up)
    }
}

/// A resblock conv plus its geometric parameters (`Conv1d` does not expose
/// its config), so the Metal GEMM-direct path can be dispatched without
/// re-deriving them.
struct ResConv {
    conv: Conv1d,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    padding: usize,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    dilation: usize,
}

impl ResConv {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _g = crate::profile::scope(Some(x.device()), "bv.res_conv");
        // candle's Metal conv1d pairs the (bit-identical) im2col kernel with
        // a strided-B batched matmul plus a transpose-back copy; the
        // GEMM-direct variant keeps the im2col and writes (c_out, l_out)
        // directly. Reduction order is the matmul's; gated at 1e-5 relative
        // in the bigvgan tests.
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let fused = if x.device().is_metal() {
            crate::metal_fused::conv1d_gemmdirect(
                x,
                self.conv.weight(),
                self.conv.bias(),
                self.padding,
                self.dilation,
            )
        } else {
            Err(candle_core::Error::Msg("cpu".into()))
        };
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            match fused {
                Ok(y) => Ok(y),
                Err(_) => self.conv.forward(x),
            }
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            self.conv.forward(x)
        }
    }
}

/// AMPBlock (resblock "1" keeps the dilation-1 `convs2` pass, "2" does not).
struct AmpBlock {
    variant2: bool,
    convs1: Vec<ResConv>,
    convs2: Vec<ResConv>,
    activations: Vec<Activation1d>,
}

impl AmpBlock {
    fn forward(&self, x: &Tensor, filter: &FirFilter) -> Result<Tensor> {
        let mut x = x.clone();
        if self.variant2 {
            for (c, act) in self.convs1.iter().zip(&self.activations) {
                let xt = act.forward(&x, filter)?;
                let xt = c.forward(&xt)?;
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
                let xt = c1.forward(&xt)?;
                let xt = a2.forward(&xt, filter)?;
                let xt = c2.forward(&xt)?;
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
    /// (stride, padding) per upsample level, for the Metal GEMM+gather path.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    ups_cfg: Vec<(usize, usize)>,
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
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let mut ups_cfg = Vec::with_capacity(cfg.upsample_rates.len());
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
            #[cfg(all(feature = "metal", target_os = "macos"))]
            ups_cfg.push((u, (k - u) / 2));
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
                    convs1.push(ResConv {
                        #[cfg(all(feature = "metal", target_os = "macos"))]
                        padding: get_padding(k, d),
                        #[cfg(all(feature = "metal", target_os = "macos"))]
                        dilation: d,
                        conv: load_conv1d_folded(
                            &rb,
                            &format!("convs1.{d_i}"),
                            (ch, ch),
                            k,
                            get_padding(k, d),
                            d,
                            true,
                        )?,
                    });
                    if !variant2 {
                        convs2.push(ResConv {
                            #[cfg(all(feature = "metal", target_os = "macos"))]
                            padding: get_padding(k, 1),
                            #[cfg(all(feature = "metal", target_os = "macos"))]
                            dilation: 1,
                            conv: load_conv1d_folded(
                                &rb,
                                &format!("convs2.{d_i}"),
                                (ch, ch),
                                k,
                                get_padding(k, 1),
                                1,
                                true,
                            )?,
                        });
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
            #[cfg(all(feature = "metal", target_os = "macos"))]
            ups_cfg,
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
                // The candle Metal conv_transpose1d falls into a naive
                // per-output scalar kernel whenever padding != 0 (the col2im
                // path is padding-0 only) — the shipped BigVGAN levels all
                // use padding (k-u)/2 > 0. On Metal prefer the GEMM+gather
                // factorization (`metal_fused::conv_transpose1d_fused`);
                // its c_in reduction order is the matmul's, gated at 1e-5
                // relative in the bigvgan tests.
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let fused = if mel.device().is_metal() {
                    let (stride, padding) = self.ups_cfg[i];
                    crate::metal_fused::conv_transpose1d_fused(
                        &x,
                        up.weight(),
                        up.bias(),
                        stride,
                        padding,
                    )
                } else {
                    Err(candle_core::Error::Msg("cpu".into()))
                };
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    match fused {
                        Ok(y) => y,
                        Err(_) => up.forward(&x)?,
                    }
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    up.forward(&x)?
                }
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

    /// Offline Metal timing probe: batched-conv FIR vs the per-tap reference on
    /// the real activation shapes of the six upsample stages (~492 mel frames).
    /// Diagnostic only; not part of any acceptance gate.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device; timing probe, no models"]
    fn fir_metal_conv_vs_tap_timing() {
        let device = Device::new_metal(0).unwrap();
        let taps = kaiser_sinc_filter1d(0.25, 0.3, AA_KERNEL);
        let filter = FirFilter::new(&taps, &device, candle_core::DType::F32).unwrap();
        let warmup = 5;
        let reps = 30;
        fn time(
            device: &Device,
            warmup: usize,
            reps: usize,
            mut f: impl FnMut() -> Result<Tensor>,
        ) -> Result<(f64, Tensor)> {
            for _ in 0..warmup {
                std::mem::drop(f()?);
            }
            device.synchronize()?;
            let start = std::time::Instant::now();
            let mut last = None;
            for _ in 0..reps {
                last = Some(f()?);
            }
            device.synchronize()?;
            Ok((
                start.elapsed().as_secs_f64() / reps as f64 * 1e3,
                last.unwrap(),
            ))
        }
        println!("shape            | conv ms | taps ms | speedup | max_abs");
        for (c, l) in [
            (1536usize, 492usize),
            (768, 1968),
            (384, 7872),
            (192, 15744),
            (96, 31488),
            (48, 62976),
        ] {
            let x = Tensor::randn(0f32, 1f32, (1, c, l), &device).unwrap();
            let (conv_ms, out_conv) =
                time(&device, warmup, reps, || fir_downsample(&x, &filter)).unwrap();
            let (tap_ms, out_tap) = time(&device, warmup, reps, || {
                fir_downsample_reference(&x, &taps)
            })
            .unwrap();
            let diff = (&out_conv - &out_tap).unwrap().abs().unwrap();
            let max_abs = diff
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .fold(0f32, f32::max);
            let ratio = conv_ms / tap_ms;
            println!(
                "{c:4}x{l:6} down | {conv_ms:7.2} | {tap_ms:7.2} | {ratio:6.2}x | {max_abs:.3e}"
            );
            let (conv_ms, out_conv) =
                time(&device, warmup, reps, || fir_upsample(&x, &filter)).unwrap();
            let (tap_ms, out_tap) =
                time(&device, warmup, reps, || fir_upsample_reference(&x, &taps)).unwrap();
            let diff = (&out_conv - &out_tap).unwrap().abs().unwrap();
            let max_abs = diff
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .fold(0f32, f32::max);
            let ratio = conv_ms / tap_ms;
            println!(
                "{c:4}x{l:6} up   | {conv_ms:7.2} | {tap_ms:7.2} | {ratio:6.2}x | {max_abs:.3e}"
            );
        }
    }

    /// Offline Metal timing probe of the AMP-block convs (`bv.res_conv`) and
    /// upsample transposed convs (`bv.up_conv`) on the real per-level shapes
    /// (T_mel ≈ 527, B=1), each against the raw contiguous 2-D GEMM of the
    /// equivalent im2col shape. Separates GEMM throughput from the
    /// im2col/strided-B/transpose plumbing cost. Diagnostic only.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device; timing probe, no models"]
    fn res_conv_stage_timing_metal() {
        let device = Device::new_metal(0).unwrap();
        fn time(device: &Device, mut f: impl FnMut() -> Result<Tensor>) -> Result<f64> {
            for _ in 0..3 {
                std::mem::drop(f()?);
            }
            device.synchronize()?;
            let start = std::time::Instant::now();
            let mut last = None;
            for _ in 0..15 {
                last = Some(f()?);
            }
            device.synchronize()?;
            std::mem::drop(last);
            Ok(start.elapsed().as_secs_f64() / 15f64 * 1e3)
        }
        // Levels after each upsample: (ch, L) with upsample_rates [4,4,2,2,2,2].
        let levels: [(usize, usize); 6] = [
            (768, 2108),
            (384, 8432),
            (192, 16864),
            (96, 33728),
            (48, 67456),
            (24, 134912),
        ];
        // resblock "1": convs1 (k, d) over dilations [1,3,5], convs2 d=1.
        println!("conv ch x L (k,d) | conv ms | gemm ms | conv TF | gemm TF");
        for (li, &(ch, l)) in levels.iter().enumerate() {
            let combos: &[(usize, usize)] = if li % 2 == 0 {
                &[
                    (3, 1),
                    (3, 3),
                    (3, 5),
                    (7, 1),
                    (7, 3),
                    (7, 5),
                    (11, 1),
                    (11, 3),
                    (11, 5),
                ]
            } else {
                &[(3, 1), (7, 1), (11, 1)]
            };
            for &(k, d) in combos {
                let x = Tensor::randn(0f32, 1f32, (1, ch, l), &device).unwrap();
                let w =
                    Tensor::randn(0f32, 1f32 / (ch as f32).sqrt(), (ch, ch, k), &device).unwrap();
                let padding = get_padding(k, d);
                let conv = candle_nn::Conv1d::new(
                    w.clone(),
                    None,
                    candle_nn::Conv1dConfig {
                        padding,
                        #[cfg(all(feature = "metal", target_os = "macos"))]
                        dilation: d,
                        ..Default::default()
                    },
                );
                let conv_ms = time(&device, || conv.forward(&x)).unwrap();
                // Equivalent im2col GEMM: (L, ch*k) @ (ch*k, ch), both
                // contiguous — the ceiling the plumbing should approach.
                let m = Tensor::randn(0f32, 1f32, (l, ch * k), &device).unwrap();
                let g =
                    Tensor::randn(0f32, 1f32 / (ch as f32).sqrt(), (ch * k, ch), &device).unwrap();
                let gemm_ms = time(&device, || m.matmul(&g)).unwrap();
                let tflops = 2.0 * l as f64 * ch as f64 * ch as f64 * k as f64 / conv_ms / 1e9;
                let gemm_tflops = 2.0 * l as f64 * ch as f64 * ch as f64 * k as f64 / gemm_ms / 1e9;
                println!(
                    "L{li} {ch:4}x{l:6} ({k:2},{d}) | {conv_ms:7.2} | {gemm_ms:7.2} | {tflops:6.2} | {gemm_tflops:6.2}"
                );
            }
        }
        println!("upconv in->out (k,s) L | conv ms");
        let ups: [(usize, usize, usize, usize, usize); 6] = [
            (1536, 768, 8, 4, 527),
            (768, 384, 8, 4, 2108),
            (384, 192, 4, 2, 8432),
            (192, 96, 4, 2, 16864),
            (96, 48, 4, 2, 33728),
            (48, 24, 4, 2, 67456),
        ];
        for (li, &(cin, cout, k, s, l)) in ups.iter().enumerate() {
            let x = Tensor::randn(0f32, 1f32, (1, cin, l), &device).unwrap();
            let w =
                Tensor::randn(0f32, 1f32 / (cin as f32).sqrt(), (cin, cout, k), &device).unwrap();
            let conv = candle_nn::ConvTranspose1d::new(
                w,
                None,
                candle_nn::ConvTranspose1dConfig {
                    padding: (k - s) / 2,
                    stride: s,
                    ..Default::default()
                },
            );
            let ms = time(&device, || conv.forward(&x)).unwrap();
            println!("L{li} {cin:4}->{cout:3} ({k},{s}) x{l:6} | {ms:7.2}");
        }
    }

    /// Numeric gate + region timing for the fused GEMM+gather
    /// conv_transpose1d against candle's `ConvTranspose1d` (whose Metal
    /// padding!=0 path is the naive per-output kernel) on the six real
    /// upsample shapes. Same tap order; only the c_in reduction order
    /// differs (matmul tiles), gated at 1e-5 relative.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device; no models"]
    fn convt_gather_matches_conv_transpose_metal() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping, no metal device: {e}");
                return;
            }
        };
        fn time(device: &Device, mut f: impl FnMut() -> Result<Tensor>) -> Result<f64> {
            for _ in 0..3 {
                std::mem::drop(f()?);
            }
            device.synchronize()?;
            let start = std::time::Instant::now();
            let mut last = None;
            for _ in 0..15 {
                last = Some(f()?);
            }
            device.synchronize()?;
            std::mem::drop(last);
            Ok(start.elapsed().as_secs_f64() / 15f64 * 1e3)
        }
        let ups: [(usize, usize, usize, usize, usize); 6] = [
            (1536, 768, 8, 4, 527),
            (768, 384, 8, 4, 2108),
            (384, 192, 4, 2, 8432),
            (192, 96, 4, 2, 16864),
            (96, 48, 4, 2, 33728),
            (48, 24, 4, 2, 67456),
        ];
        println!("convt c_in->c_out (k,s) x l_in | ref ms | fused ms | rel");
        for (cin, cout, k, s, l) in ups {
            let x = Tensor::randn(0f32, 1f32, (1, cin, l), &device).unwrap();
            let w =
                Tensor::randn(0f32, 1f32 / (cin as f32).sqrt(), (cin, cout, k), &device).unwrap();
            // The shipped levels all carry a bias; the gate must cover it.
            let bias = Tensor::randn(0f32, 1f32, (cout,), &device).unwrap();
            let padding = (k - s) / 2;
            let reference = candle_nn::ConvTranspose1d::new(
                w.clone(),
                Some(bias.clone()),
                candle_nn::ConvTranspose1dConfig {
                    padding,
                    stride: s,
                    ..Default::default()
                },
            );
            let out_ref = reference.forward(&x).unwrap();
            let fused = crate::metal_fused::conv_transpose1d_fused(&x, &w, Some(&bias), s, padding)
                .unwrap();
            assert_eq!(fused.dims(), out_ref.dims());
            let max_abs = (&fused - &out_ref)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let mag = out_ref
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let rel = max_abs / mag;
            let fused_ms = time(&device, || {
                crate::metal_fused::conv_transpose1d_fused(&x, &w, Some(&bias), s, padding)
            })
            .unwrap();
            let ref_ms2 = time(&device, || reference.forward(&x)).unwrap();
            println!(
                "convt {cin:4}->{cout:3} ({k},{s}) x{l:6} | {ref_ms2:7.2} | {fused_ms:7.2} | {rel:.3e}"
            );
            assert!(rel <= 1e-5, "{cin}->{cout} rel {rel:.3e}");
        }
    }

    /// CPU scalar simulation of the GEMM+gather conv_transpose
    /// factorization against candle's `ConvTranspose1d`, isolating the
    /// gather semantics from any Metal kernel issue.
    #[test]
    fn convt_gather_semantics_cpu() {
        let (cin, cout, k, s, p, l) = (3usize, 2usize, 4, 2, 1, 5);
        let l_out = (l - 1) * s - 2 * p + k;
        let x = Tensor::randn(0f32, 1f32, (1, cin, l), &Device::Cpu).unwrap();
        let w = Tensor::randn(0f32, 1f32, (cin, cout, k), &Device::Cpu).unwrap();
        let reference = candle_nn::ConvTranspose1d::new(
            w.clone(),
            None,
            candle_nn::ConvTranspose1dConfig {
                padding: p,
                stride: s,
                ..Default::default()
            },
        )
        .forward(&x)
        .unwrap();
        // col = x^T @ reshape(w, (cin, cout*k)), col[i, oc*k + kk].
        let xv = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let wv = w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut out = vec![0f32; cout * l_out];
        for oc in 0..cout {
            for ox in 0..l_out {
                let phase = (ox + p) % s;
                let base = (ox + p - phase) / s;
                let mut d = 0f32;
                let mut kk = phase;
                while kk < k {
                    let j = (kk - phase) / s;
                    let i = base as isize - j as isize;
                    if i >= 0 && (i as usize) < l {
                        // d += x[:, i] . w[:, oc, kk]
                        let mut acc = 0f32;
                        for c in 0..cin {
                            acc += xv[c * l + i as usize] * wv[(c * cout + oc) * k + kk];
                        }
                        d += acc;
                    }
                    kk += s;
                }
                out[oc * l_out + ox] = d;
            }
        }
        let got = Tensor::from_vec(out, (1, cout, l_out), &Device::Cpu).unwrap();
        let max_abs = (&got - &reference)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        println!("convt-gather cpu semantics max_abs {max_abs:.3e}");
        assert_eq!(got.dims(), reference.dims());
        assert!(max_abs <= 1e-5, "max_abs {max_abs:.3e}");
    }

    /// Numeric gate + region timing for the GEMM-direct conv1d against
    /// candle's `Conv1d` (im2col + strided-B batched matmul + transpose
    /// copy) on the real resblock shapes. The im2col buffer is candle's own
    /// kernel (bit-identical); only the GEMM reduction order differs,
    /// gated at 1e-5 relative.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device; no models"]
    fn conv_gemmdirect_matches_conv1d_metal() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping, no metal device: {e}");
                return;
            }
        };
        fn time(device: &Device, mut f: impl FnMut() -> Result<Tensor>) -> Result<f64> {
            for _ in 0..3 {
                std::mem::drop(f()?);
            }
            device.synchronize()?;
            let start = std::time::Instant::now();
            let mut last = None;
            for _ in 0..15 {
                last = Some(f()?);
            }
            device.synchronize()?;
            std::mem::drop(last);
            Ok(start.elapsed().as_secs_f64() / 15f64 * 1e3)
        }
        let levels: [(usize, usize); 6] = [
            (768, 2108),
            (384, 8432),
            (192, 16864),
            (96, 33728),
            (48, 67456),
            (24, 134912),
        ];
        println!("conv ch x l (k,d) | ref ms | fused ms | rel");
        for (li, &(ch, l)) in levels.iter().enumerate() {
            let combos: &[(usize, usize)] = if li % 2 == 0 {
                &[
                    (3, 1),
                    (3, 3),
                    (3, 5),
                    (7, 1),
                    (7, 3),
                    (7, 5),
                    (11, 1),
                    (11, 3),
                    (11, 5),
                ]
            } else {
                &[(3, 1), (7, 1), (11, 1)]
            };
            for &(k, d) in combos {
                let x = Tensor::randn(0f32, 1f32, (1, ch, l), &device).unwrap();
                let w =
                    Tensor::randn(0f32, 1f32 / (ch as f32).sqrt(), (ch, ch, k), &device).unwrap();
                // The shipped resblock convs all carry a bias.
                let bias = Tensor::randn(0f32, 1f32, (ch,), &device).unwrap();
                let padding = get_padding(k, d);
                let reference = candle_nn::Conv1d::new(
                    w.clone(),
                    Some(bias.clone()),
                    candle_nn::Conv1dConfig {
                        padding,
                        #[cfg(all(feature = "metal", target_os = "macos"))]
                        dilation: d,
                        ..Default::default()
                    },
                );
                let out_ref = reference.forward(&x).unwrap();
                let fused =
                    crate::metal_fused::conv1d_gemmdirect(&x, &w, Some(&bias), padding, d).unwrap();
                assert_eq!(fused.dims(), out_ref.dims());
                let max_abs = (&fused - &out_ref)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                let mag = out_ref
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                let rel = max_abs / mag;
                let fused_ms = time(&device, || {
                    crate::metal_fused::conv1d_gemmdirect(&x, &w, Some(&bias), padding, d)
                })
                .unwrap();
                let ref_ms = time(&device, || reference.forward(&x)).unwrap();
                println!(
                    "L{li} {ch:4}x{l:6} ({k:2},{d}) | {ref_ms:7.2} | {fused_ms:7.2} | {rel:.3e}"
                );
                assert!(rel <= 1e-5, "L{li} {ch} ({k},{d}) rel {rel:.3e}");
            }
        }
    }

    /// Numeric gate + region timing for the fused up-FIR+snake kernel
    /// against the `fir_upsample` + `apply_activation` chain on the real
    /// activation shapes (including odd T). Tap order and snake expressions
    /// match; differences are float rounding only, gated at 1e-5 relative.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device; no models"]
    fn fir_up_snake_matches_chain_metal() {
        let device = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping, no metal device: {e}");
                return;
            }
        };
        fn time(device: &Device, mut f: impl FnMut() -> Result<Tensor>) -> Result<f64> {
            for _ in 0..3 {
                std::mem::drop(f()?);
            }
            device.synchronize()?;
            let start = std::time::Instant::now();
            let mut last = None;
            for _ in 0..15 {
                last = Some(f()?);
            }
            device.synchronize()?;
            std::mem::drop(last);
            Ok(start.elapsed().as_secs_f64() / 15f64 * 1e3)
        }
        let taps = kaiser_sinc_filter1d(0.25, 0.3, AA_KERNEL);
        let filter = FirFilter::new(&taps, &device, candle_core::DType::F32).unwrap();
        let phases = filter.fused_phases.as_ref().unwrap();
        println!("shape (B,C,T) | chain ms | fused ms | rel");
        for (c, t) in [
            (768usize, 2108usize),
            (384, 8432),
            (192, 16864),
            (96, 33728),
            (48, 67456),
            (24, 134912),
            (48, 67455), // odd T
        ] {
            let x = Tensor::randn(0f32, 1f32, (1, c, t), &device).unwrap();
            let a = Tensor::randn(0f32, 0.1, (1, c, 1), &device)
                .unwrap()
                .exp()
                .unwrap();
            let b = Tensor::randn(0f32, 0.1, (1, c, 1), &device)
                .unwrap()
                .exp()
                .unwrap();
            let scale = (b + 1e-9).unwrap().recip().unwrap();
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let fused_acts = Tensor::cat(
                &[&a.reshape((c,)).unwrap(), &scale.reshape((c,)).unwrap()],
                0,
            )
            .unwrap()
            .reshape((2, c))
            .unwrap();
            let chain = |x: &Tensor| -> Result<Tensor> {
                let up = fir_upsample(x, &filter)?;
                // same expressions as apply_activation with these a/b
                let s = up.broadcast_mul(&a)?.sin()?.sqr()?;
                up.broadcast_add(&s.broadcast_mul(&scale)?)
            };
            let out_chain = chain(&x).unwrap();
            let fused =
                crate::metal_fused::fir_up_snake(&x, phases, &fused_acts, AA_KERNEL / AA_RATIO)
                    .unwrap();
            assert_eq!(fused.dims(), out_chain.dims());
            let max_abs = (&fused - &out_chain)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let mag = out_chain
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let rel = max_abs / mag;
            let fused_ms = time(&device, || {
                crate::metal_fused::fir_up_snake(&x, phases, &fused_acts, AA_KERNEL / AA_RATIO)
            })
            .unwrap();
            let chain_ms = time(&device, || chain(&x)).unwrap();
            println!("{c:4}x{t:6} | {chain_ms:7.2} | {fused_ms:7.2} | {rel:.3e}");
            assert!(rel <= 1e-5, "{c}x{t} rel {rel:.3e}");
        }
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
            #[cfg(all(feature = "metal", target_os = "macos"))]
            fused_acts: None,
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
