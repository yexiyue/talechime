//! WaveNet block used as the DiT final layer
//! (`indextts/s2mel/modules/wavenet.py`, class `WN`).
//!
//! Shipped configuration: 8 layers, kernel 5, dilation rate 1, hidden 512,
//! `gin_channels = wavenet.hidden_dim`, non-causal, inference-time (dropout
//! disabled). The conv layers are encodec `SConv1d`s with weight-norm
//! reparametrization (`in_layers.{i}.conv.conv.{weight_g,weight_v,bias}` in
//! the checkpoint); `SConv1d` ignores the padding argument its caller passes
//! and pads on its own: effective kernel `k`, stride 1 → symmetric
//! `(k-1)/2` **reflect** padding (`pad_mode='reflect'`).
//!
//! This module is internal to [`crate::dit`]; it is a separate file only to
//! mirror the upstream layout.

use crate::config::WavenetConfig;
use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, Module, VarBuilder};

pub struct Wavenet {
    hidden_channels: usize,
    n_layers: usize,
    /// Weight-normed dilated Conv1d k=`kernel_size`, hidden→2*hidden.
    in_layers: Vec<Conv1d>,
    /// Weight-normed 1×1 Conv1d; last layer hidden channels, others 2×hidden.
    res_skip_layers: Vec<Conv1d>,
    /// Weight-normed 1×1 Conv1d `gin → 2*hidden*n_layers` (time-broadcast).
    cond_layer: Conv1d,
}

/// `F.pad(x, (left, right), mode='reflect')` along the last dim (indices
/// `[left..1]` mirrored on the left, `[T-2..T-1-right]` on the right,
/// boundary not repeated).
fn reflect_pad1d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let t = x.dims()[x.dims().len() - 1];
    let max_pad = left.max(right);
    if t <= max_pad {
        // Upstream pad1d zero-extends first; only reachable for tiny inputs.
        candle_core::bail!("reflect_pad1d: length {t} too short for pad {max_pad}");
    }
    let mut idx: Vec<u32> = Vec::with_capacity(t + left + right);
    idx.extend((1..=left).rev().map(|i| i as u32));
    idx.extend((0..t as u32).collect::<Vec<_>>());
    idx.extend((t - right - 1..t - 1).rev().map(|i| i as u32));
    let index = Tensor::from_vec(idx, (t + left + right,), x.device())?;
    x.index_select(&index, x.dims().len() - 1)
}

/// Builds a candle `Conv1d` from PyTorch `weight_norm` parameters
/// (`weight_g`, `weight_v`, `bias`), folding `g * v / ||v||₂` over all axes
/// except the output-channel axis into a dense weight. The norm is clamped
/// to `1e-12` so all-zero `weight_v` (VarMap-initialized tests) fold to zero
/// instead of NaN; real checkpoints have norms of O(1) or larger.
fn load_weight_norm_conv1d(
    vb: &VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
) -> Result<Conv1d> {
    // torch stores `weight_g` with singleton trailing axes matching `v`
    // (e.g. `(out, 1, 1)` for Conv1d); flatten to per-channel scalars.
    let g = match vb.get((out_channels, 1, 1), "weight_g") {
        Ok(g) => g.flatten_all()?,
        Err(_) => vb.get((out_channels,), "weight_g")?,
    };
    let v = vb.get(
        (out_channels, in_channels / cfg.groups, kernel_size),
        "weight_v",
    )?;
    let bias = vb.get((out_channels,), "bias")?;
    let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?.maximum(1e-12f64)?;
    let mut shape = vec![1usize; v.rank()];
    shape[0] = out_channels;
    let weight = v.broadcast_div(&norm)?.broadcast_mul(&g.reshape(shape)?)?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

impl Wavenet {
    pub fn load(vb: &VarBuilder, cfg: &WavenetConfig) -> Result<Self> {
        if cfg.dilation_rate != 1 {
            // Shipped config uses dilation 1 everywhere; the general case
            // would need the same reflect padding with `dilation`.
            candle_core::bail!(
                "only dilation_rate=1 is shipped (got {})",
                cfg.dilation_rate
            );
        }
        if cfg.kernel_size.is_multiple_of(2) {
            candle_core::bail!("kernel_size must be odd (got {})", cfg.kernel_size);
        }
        let hidden = cfg.hidden_dim;
        let n_layers = cfg.num_layers;
        let mut in_layers = Vec::with_capacity(n_layers);
        let mut res_skip_layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            // dilation = dilation_rate ** i = 1; SConv1d symmetric reflect
            // padding (k - stride) / 2 = (k - 1) / 2 with stride 1; candle
            // conv stays unpadded, padding is applied in forward.
            let in_layer = load_weight_norm_conv1d(
                &vb.pp(format!("in_layers.{i}")).pp("conv").pp("conv"),
                hidden,
                2 * hidden,
                cfg.kernel_size,
                Conv1dConfig::default(),
            )?;
            // Last res/skip layer has plain hidden out channels.
            let res_skip_channels = if i < n_layers - 1 { 2 * hidden } else { hidden };
            let res_skip = load_weight_norm_conv1d(
                &vb.pp(format!("res_skip_layers.{i}")).pp("conv").pp("conv"),
                hidden,
                res_skip_channels,
                1,
                Conv1dConfig::default(),
            )?;
            in_layers.push(in_layer);
            res_skip_layers.push(res_skip);
        }
        let cond_layer = load_weight_norm_conv1d(
            &vb.pp("cond_layer").pp("conv").pp("conv"),
            hidden, // gin_channels = wavenet.hidden_dim in the DiT wiring
            2 * hidden * n_layers,
            1,
            Conv1dConfig::default(),
        )?;
        Ok(Self {
            hidden_channels: hidden,
            n_layers,
            in_layers,
            res_skip_layers,
            cond_layer,
        })
    }

    /// `x`: `(B, hidden, T)`; `style`: conditioning `(B, gin, T_g)` (upstream
    /// `g`; the DiT final layer passes `t_embedder2(t).unsqueeze(2)`, i.e.
    /// `(B, hidden, 1)` — the 192-d CAMPPlus style never enters WN directly).
    /// Upstream also multiplies by the `(B, 1, T)` sequence mask inside every
    /// layer; the shipped inference caller always passes full lengths (mask
    /// ≡ 1), so this port omits the no-op mask.
    pub fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let hidden = self.hidden_channels;
        // g: (B, 2*hidden*n_layers, T_g)
        let g = self.cond_layer.forward(style)?;
        let g_t = g.dims()[2];
        let mut x = x.clone();
        let mut output = x.zeros_like()?;
        for i in 0..self.n_layers {
            let k = self.in_layers[i].weight().dims()[2];
            let side = (k - 1) / 2;
            let padded = {
                let _g = crate::profile::scope(Some(x.device()), "wn.pad");
                reflect_pad1d(&x, side, side)?
            };
            let x_in = {
                let _g = crate::profile::scope(Some(x.device()), "wn.conv");
                self.in_layers[i].forward(&padded)?
            }; // (B, 2H, T)
            // g slice (B, 2H, T_g), broadcast over time.
            let g_l = if g_t == 1 {
                g.narrow(1, i * 2 * hidden, 2 * hidden)?
            } else {
                // Upstream only ever uses T_g == 1 (timestep embedding).
                candle_core::bail!("wavenet conditioning must have length 1 (got {g_t})");
            };
            let eager_activation = || -> Result<Tensor> {
                let _g = crate::profile::scope(Some(x.device()), "wn.act");
                let in_act = x_in.broadcast_add(&g_l)?;
                let t_act = in_act.narrow(1, 0, hidden)?.tanh()?;
                let s_act = candle_nn::ops::sigmoid(&in_act.narrow(1, hidden, hidden)?)?;
                t_act.mul(&s_act)
            };
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let acts = if x.device().is_metal() {
                let fused = {
                    let _g = crate::profile::scope(Some(x.device()), "wn.act");
                    crate::metal_fused::wn_gate_act(&x_in, &g, i * 2 * hidden)
                };
                fused.or_else(|_| eager_activation())?
            } else {
                eager_activation()?
            };
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            let acts = eager_activation()?;
            let res_skip_acts = {
                let _g = crate::profile::scope(Some(x.device()), "wn.res_skip");
                self.res_skip_layers[i].forward(&acts)?
            };
            let _g = crate::profile::scope(Some(x.device()), "wn.act");
            if i < self.n_layers - 1 {
                let eager_tail = || -> Result<(Tensor, Tensor)> {
                    let res = res_skip_acts.narrow(1, 0, hidden)?;
                    let skip = res_skip_acts.narrow(1, hidden, hidden)?;
                    Ok((x.add(&res)?, output.add(&skip)?))
                };
                #[cfg(all(feature = "metal", target_os = "macos"))]
                let tail = if x.device().is_metal() {
                    match crate::metal_fused::wn_res_skip_tail(&res_skip_acts, &x, &output) {
                        Ok(both) => (
                            both.narrow(0, 0, 1)?.squeeze(0)?,
                            both.narrow(0, 1, 1)?.squeeze(0)?,
                        ),
                        Err(_) => eager_tail()?,
                    }
                } else {
                    eager_tail()?
                };
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                let tail = eager_tail()?;
                (x, output) = tail;
            } else {
                output = output.add(&res_skip_acts)?;
            }
            drop(_g);
        }
        Ok(output)
    }
}

// Re-export for the weight-normed Linear used by dit.rs (final layer).
// candle `Linear` keeps the PyTorch `(out, in)` layout, so the folded weight
// is used as-is.
pub(crate) fn load_weight_norm_linear(
    vb: &VarBuilder,
    in_features: usize,
    out_features: usize,
) -> Result<Linear> {
    let g = match vb.get((out_features, 1), "weight_g") {
        Ok(g) => g.flatten_all()?,
        Err(_) => vb.get((out_features,), "weight_g")?,
    };
    let v = vb.get((out_features, in_features), "weight_v")?;
    let bias = vb.get((out_features,), "bias")?;
    let norm = v.sqr()?.sum_keepdim(1)?.sqrt()?.maximum(1e-12f64)?;
    let weight = v
        .broadcast_div(&norm)?
        .broadcast_mul(&g.reshape((out_features, 1))?)?;
    Ok(Linear::new(weight, Some(bias)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    /// VarMap initializes the weight-norm convs with kaiming noise; reset to
    /// zeros.
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

    fn tiny_cfg() -> WavenetConfig {
        WavenetConfig {
            hidden_dim: 8,
            num_layers: 2,
            kernel_size: 3,
            dilation_rate: 1,
            p_dropout: 0.2,
            style_condition: true,
        }
    }

    #[test]
    fn reflect_pad_matches_f_pad_reflect() {
        let device = Device::Cpu;
        let x = Tensor::from_vec((0i32..6).map(|i| i as f32).collect(), (6,), &device).unwrap();
        let padded = reflect_pad1d(&x, 2, 2).unwrap();
        // F.pad([0,1,2,3,4,5], (2,2), 'reflect') == [2,1,0,1,2,3,4,5,4,3]
        assert_eq!(
            padded.to_vec1::<f32>().unwrap(),
            vec![2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]
        );
        let x = x.reshape((1, 1, 6)).unwrap();
        let padded = reflect_pad1d(&x, 1, 1).unwrap();
        assert_eq!(padded.dims(), vec![1, 1, 8]);
        assert_eq!(
            padded.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0]
        );
    }

    #[test]
    fn forward_gating_math() {
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = Wavenet::load(&vb, &tiny_cfg()).unwrap();
        zero_all(&mut varmap);
        // All weights zero. in_layers.0 bias ones -> acts = tanh(1)*sigmoid(1)
        // everywhere; res_skip weights zero -> only their biases survive.
        varmap
            .set_one(
                "in_layers.0.conv.conv.bias",
                Tensor::ones(16, DType::F32, &device).unwrap(),
            )
            .unwrap();
        // res_skip_layers.0 bias: [res(8)=1, skip(8)=2]
        let rs0 = Tensor::cat(
            &[
                Tensor::ones(8, DType::F32, &device).unwrap(),
                Tensor::full(2f32, 8, &device).unwrap(),
            ],
            0,
        )
        .unwrap();
        varmap
            .set_one("res_skip_layers.0.conv.conv.bias", &rs0)
            .unwrap();
        varmap
            .set_one(
                "res_skip_layers.1.conv.conv.bias",
                Tensor::full(3f32, 8, &device).unwrap(),
            )
            .unwrap();
        let x = Tensor::zeros((1, 8, 10), DType::F32, &device).unwrap();
        let g = Tensor::zeros((1, 8, 1), DType::F32, &device).unwrap();
        let out = model.forward(&x, &g).unwrap();
        assert_eq!(out.dims(), vec![1, 8, 10]);
        // Layer 1 sees x = res_acts = 1 -> conv(x)=bias 0 -> acts 0 ->
        // res_skip_1(0) = bias 3. Output = skip_0 (2) + bias_1 (3) = 5.
        // The tanh/sigmoid path of layer 0 contributes nothing because the
        // res_skip weights are zero.
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|&v| (v - 5.0).abs() < 1e-5)
        );
    }

    #[test]
    fn forward_zero_weights_returns_zero() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Wavenet::load(&vb, &tiny_cfg()).unwrap();
        let x = Tensor::randn(0f32, 1f32, (2, 8, 7), &device).unwrap();
        let g = Tensor::randn(0f32, 1f32, (2, 8, 1), &device).unwrap();
        let out = model.forward(&x, &g).unwrap();
        assert_eq!(out.dims(), vec![2, 8, 7]);
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|&v| v == 0.0)
        );
    }
    /// Metal numeric gate for the fused wavenet kernels against the
    /// candle elementwise chain on the real solve shapes (B=2 CFG batch,
    /// hidden 512, odd T included). The tanh/sigmoid expressions are
    /// candle's verbatim, so the gate can demand bit-exactness; the tail
    /// add order matches the reference (x + res, output + skip).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device; no models"]
    fn wavenet_fused_acts_match_chain_metal() {
        let metal = match Device::new_metal(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skipping, no metal device: {e}");
                return;
            }
        };
        let (b, hidden, n_layers) = (2usize, 512usize, 8usize);
        for t in [1024usize, 37] {
            let two_h = 2 * hidden;
            let g_len = two_h * n_layers;
            let x_in = Tensor::randn(0f32, 1f32, (b, two_h, t), &metal).unwrap();
            let g = Tensor::randn(0f32, 1f32, (b, g_len, 1), &metal).unwrap();
            for layer in [0usize, 3, n_layers - 1] {
                let g_off = layer * two_h;
                let g_l = g.narrow(1, g_off, two_h).unwrap();
                let in_act = x_in.broadcast_add(&g_l).unwrap();
                let want = in_act
                    .narrow(1, 0, hidden)
                    .unwrap()
                    .tanh()
                    .unwrap()
                    .mul(
                        &candle_nn::ops::sigmoid(&in_act.narrow(1, hidden, hidden).unwrap())
                            .unwrap(),
                    )
                    .unwrap();
                let got = crate::metal_fused::wn_gate_act(&x_in, &g, g_off).unwrap();
                assert_eq!(got.dims(), want.dims());
                let max_abs = (got - &want)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                println!("wn-gate-act t={t} layer {layer}: max_abs {max_abs:.3e}");
                assert!(max_abs == 0.0, "t={t} layer {layer} max_abs {max_abs:.3e}");

                // Tail: rs -> (x + res, output + skip).
                let rs = Tensor::randn(0f32, 1f32, (b, two_h, t), &metal).unwrap();
                let x = Tensor::randn(0f32, 1f32, (b, hidden, t), &metal).unwrap();
                let output = Tensor::randn(0f32, 1f32, (b, hidden, t), &metal).unwrap();
                let both = crate::metal_fused::wn_res_skip_tail(&rs, &x, &output).unwrap();
                assert_eq!(both.dims(), &[2, b, hidden, t][..]);
                let want_x = x.add(&rs.narrow(1, 0, hidden).unwrap()).unwrap();
                let want_o = output.add(&rs.narrow(1, hidden, hidden).unwrap()).unwrap();
                let plane0 = both.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
                let plane1 = both.narrow(0, 1, 1).unwrap().squeeze(0).unwrap();
                for (plane, got, expect) in [(0usize, &plane0, &want_x), (1, &plane1, &want_o)] {
                    let max_abs = (got - expect)
                        .unwrap()
                        .abs()
                        .unwrap()
                        .max_all()
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap();
                    println!("wn-res-skip t={t} plane {plane}: max_abs {max_abs:.3e}");
                    assert!(max_abs == 0.0, "t={t} plane {plane} max_abs {max_abs:.3e}");
                }
            }
        }
    }
}
