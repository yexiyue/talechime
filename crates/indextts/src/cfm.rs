//! Conditional flow matching (`indextts/s2mel/modules/flow_matching.py`).
//!
//! Inference path of `CFM.inference` with `solver="euler"`, `n_timesteps=25`
//! and classifier-free-guidance rate 0.7 over the DiT estimator. This CFM
//! variant owns **no weights of its own** beyond the estimator: unlike the
//! Matcha-style CFMs there is no `mu` input/output projection and no mel
//! mean/std normalization — the `mu` argument of `inference()` *is* the
//! (already length-regulated) condition, and the reference mel enters only as
//! the `prompt` tensor that is pinned in front of the noise state. The only
//! other parameter is `sigma_min = 1e-6`, which the training loss uses; it
//! never appears on the euler inference path.
//!
//! Solver semantics (`solve_euler`): initialize `z ~ N(0,1)` over
//! `(B, 80, L)`, zero the prompt region of `z`, pin `prompt_x` (= reference
//! mel in front of an otherwise zero tensor), then for each of the
//! `n_timesteps` uniform euler steps from t=0 to t=1 compute the estimator
//! field on the doubled `[conditioned, null]` batch
//! (`prompt_x/style/mu` zeroed in the null half), combine with
//! `dphi = (1 + rate) * dphi_cond - rate * dphi_null`, integrate
//! `x += dt * dphi`, and re-zero the prompt region after every step.

use crate::config::S2MelConfig;
use crate::dit::Dit;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

pub struct Cfm {
    estimator: Dit,
    in_channels: usize,
}

/// Classifier-free-guidance combination
/// `(1 + rate) * dphi - rate * dphi_null`.
fn cfg_combine(dphi: &Tensor, dphi_null: &Tensor, rate: f64) -> Result<Tensor> {
    let guided = dphi.affine(1.0 + rate, 0.0)?;
    let null = dphi_null.affine(rate, 0.0)?;
    guided.sub(&null)
}

impl Cfm {
    /// Weight names under `models.cfm.*` in s2mel.pth — everything except the
    /// unused training loss lives under `models.cfm.estimator.*`.
    pub fn load(vb: &VarBuilder, cfg: &S2MelConfig) -> Result<Self> {
        match cfg.dit_type.as_deref() {
            Some("DiT") | None => {}
            Some(other) => candle_core::bail!("unknown dit_type {other:?} (only DiT is shipped)"),
        }
        let estimator = Dit::load(&vb.pp("estimator"), &cfg.dit, &cfg.wavenet)?;
        Ok(Self {
            estimator,
            in_channels: cfg.dit.in_channels,
        })
    }

    /// `condition`: `(B, L, 512)` length-regulated condition (prompt
    /// condition concatenated with the target condition by the caller);
    /// `condition_lengths`: mel frame counts (the shipped caller passes one
    /// full-length entry per sample — the estimator's attention mask is
    /// all-ones, so anything shorter is rejected); `ref_mel`:
    /// `(B, 80, T_ref)` prompt mel; `style`: `(B, 192)`; `n_timesteps`:
    /// euler steps (shipped 25); `cfg_rate`: guidance rate (shipped 0.7);
    /// temperature is fixed at the upstream default 1.0.
    ///
    /// Returns the FULL integrated mel `(B, 80, L)`; the caller slices off
    /// the first `T_ref` frames (`vc_target[:, :, ref_mel.size(-1):]`).
    pub fn inference(
        &self,
        condition: &Tensor,
        condition_lengths: &[usize],
        ref_mel: &Tensor,
        style: &Tensor,
        n_timesteps: usize,
        cfg_rate: f64,
    ) -> Result<Tensor> {
        if n_timesteps == 0 {
            candle_core::bail!("n_timesteps must be positive");
        }
        let (b, l, _) = condition.dims3()?;
        if condition_lengths.len() != b || condition_lengths.iter().any(|&len| len != l) {
            candle_core::bail!(
                "condition_lengths must equal the full sequence length {l} per sample"
            );
        }
        let (rb, channels, t_ref) = ref_mel.dims3()?;
        if rb != b || channels != self.in_channels || t_ref == 0 || t_ref > l {
            candle_core::bail!(
                "ref_mel must be (B, {}, T_ref) with 0 < T_ref <= {l}",
                self.in_channels
            );
        }
        let device = condition.device();
        // prompt_x: reference mel pinned in front of zeros.
        let prompt_tail = Tensor::zeros((b, channels, l - t_ref), ref_mel.dtype(), device)?;
        let prompt_x = Tensor::cat(&[ref_mel, &prompt_tail], 2)?;
        // z ~ N(0,1) * 1.0 with the prompt region zeroed (upstream zeroes x
        // in place before the loop; zeros never enter the ODE afterwards).
        let mut x = Tensor::randn(0f32, 1f32, (b, self.in_channels, l), device)?;
        x = zero_prompt_region(&x, t_ref)?;

        // These inputs do not change during integration. Keep them local to
        // this solve so different references and sequence lengths cannot leak.
        let (estimator_prompt, projected, estimator_style) = {
            let _g = crate::profile::scope(Some(device), "cfm.prepare");
            let (prompt, mu, style) = if cfg_rate > 0.0 {
                (
                    Tensor::cat(&[&prompt_x, &prompt_x.zeros_like()?], 0)?,
                    Tensor::cat(&[condition, &condition.zeros_like()?], 0)?,
                    Tensor::cat(&[style, &style.zeros_like()?], 0)?,
                )
            } else {
                (prompt_x, condition.clone(), style.clone())
            };
            (prompt, self.estimator.project_condition(&mu)?, style)
        };

        // t_span = linspace(0, 1, n + 1); dt per step from consecutive span
        // entries, t advanced by accumulation exactly like the solver loop.
        let t_span: Vec<f64> = (0..=n_timesteps)
            .map(|i| i as f64 / n_timesteps as f64)
            .collect();
        let mut t = 0f64;
        for step in 1..=n_timesteps {
            let dt = t_span[step] - t_span[step - 1];
            let t_vec = Tensor::full(t as f32, (b,), device)?;
            let dphi = if cfg_rate > 0.0 {
                // Doubled batch: [conditioned, null]. The null half zeroes
                // prompt_x, style AND mu (upstream `torch.cat` with zeros),
                // which is *not* the same as the class-dropout branch — the
                // estimator still applies cond_projection to the zero mu.
                let _g = crate::profile::scope(Some(device), "cfm.pack");
                let stacked_x = Tensor::cat(&[&x, &x], 0)?;
                let stacked_t = Tensor::cat(&[&t_vec, &t_vec], 0)?;
                let packed = Tensor::cat(&[&stacked_x, &estimator_prompt], 1)?;
                drop(_g);
                let _g = crate::profile::scope(Some(device), "cfm.estimator");
                let field = self.estimator.forward_projected(
                    &packed,
                    &stacked_t,
                    &projected,
                    &estimator_style,
                    false,
                )?;
                drop(_g);
                let _g = crate::profile::scope(Some(device), "cfm.combine");
                let chunks = field.chunk(2, 0)?;
                cfg_combine(&chunks[0], &chunks[1], cfg_rate)?
            } else {
                let packed = Tensor::cat(&[&x, &estimator_prompt], 1)?;
                let _g = crate::profile::scope(Some(device), "cfm.estimator");
                let field = self.estimator.forward_projected(
                    &packed,
                    &t_vec,
                    &projected,
                    &estimator_style,
                    false,
                )?;
                drop(_g);
                let _g = crate::profile::scope(Some(device), "cfm.combine");
                field
            };
            x = {
                let _g = crate::profile::scope(Some(device), "cfm.integrate");
                let x = x.broadcast_add(&dphi.affine(dt, 0.0)?)?;
                t += dt;
                zero_prompt_region(&x, t_ref)?
            };
        }
        Ok(x)
    }
}

/// `x[:, :, :prompt_len] = 0` — in-place zeroing of the solver state's prompt
/// region, expressed functionally.
fn zero_prompt_region(x: &Tensor, prompt_len: usize) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    if prompt_len >= t {
        candle_core::bail!("prompt_len {prompt_len} must be < sequence length {t}");
    }
    let head = Tensor::zeros((b, c, prompt_len), x.dtype(), x.device())?;
    Tensor::cat(&[&head, &x.narrow(2, prompt_len, t - prompt_len)?], 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DiTConfig, LengthRegulatorConfig, WavenetConfig};
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    /// VarMap initializes linears with kaiming noise; reset to zeros.
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

    fn tiny_cfg() -> S2MelConfig {
        S2MelConfig {
            dit: DiTConfig {
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
            wavenet: WavenetConfig {
                hidden_dim: 16,
                num_layers: 2,
                kernel_size: 3,
                dilation_rate: 1,
                p_dropout: 0.2,
                style_condition: true,
            },
            length_regulator: LengthRegulatorConfig::default(),
            ..Default::default()
        }
    }

    fn load_tiny(vb: &VarBuilder) -> Cfm {
        Cfm::load(vb, &tiny_cfg()).unwrap()
    }

    #[test]
    fn cfg_combine_formula() {
        let device = Device::Cpu;
        let dphi = Tensor::from_vec(vec![1.0f32, 2.0], (2,), &device).unwrap();
        let null = Tensor::from_vec(vec![3.0f32, 5.0], (2,), &device).unwrap();
        // (1 + 0.7) * [1, 2] - 0.7 * [3, 5] = [-0.4, -0.1]
        let out = cfg_combine(&dphi, &null, 0.7).unwrap();
        let v = out.to_vec1::<f32>().unwrap();
        assert!((v[0] - (-0.4)).abs() < 1e-6);
        assert!((v[1] - (-0.1)).abs() < 1e-6);
        // rate 0 is the identity.
        let out = cfg_combine(&dphi, &null, 0.0).unwrap();
        assert_eq!(out.to_vec1::<f32>().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn zero_estimator_preserves_noise_and_zeroes_prompt() {
        // All-zero DiT weights => estimator field is exactly 0, so the euler
        // loop integrates nothing: the output is the initial noise with the
        // prompt region re-zeroed after every step.
        let device = Device::Cpu;
        let cfm = load_tiny(&VarBuilder::zeros(DType::F32, &device));
        let l = 20;
        let t_ref = 7;
        let condition = Tensor::randn(0f32, 1f32, (1, l, 16), &device).unwrap();
        let ref_mel = Tensor::randn(0f32, 1f32, (1, 4, t_ref), &device).unwrap();
        let style = Tensor::randn(0f32, 1f32, (1, crate::dit::STYLE_DIM), &device).unwrap();
        let out = cfm
            .inference(&condition, &[l], &ref_mel, &style, 4, 0.7)
            .unwrap();
        assert_eq!(out.dims(), vec![1, 4, l]);
        // (B, C, T) -> (B, T, C) so frames[0] is indexed by time.
        let frames = out.transpose(1, 2).unwrap().to_vec3::<f32>().unwrap();
        for frame in frames[0].iter().take(t_ref) {
            assert!(frame.iter().all(|&v| v == 0.0));
        }
        let tail: Vec<f32> = frames[0][t_ref..].iter().flatten().copied().collect();
        assert!(tail.iter().all(|v| v.is_finite()));
        // tail == z ~ N(0,1): mean near 0, std near 1 (52 samples).
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        let var = tail.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / tail.len() as f32;
        assert!(mean.abs() < 0.7, "tail mean {mean}");
        assert!(var > 0.3 && var < 1.9, "tail var {var}");
    }

    #[test]
    fn constant_estimator_field_moves_state_by_exactly_one_dt_sum() {
        // conv2 bias => estimator(x) ≡ 0.25 for every branch and input, so
        // after n steps of dt = 1/n each: x = z + 0.25 (prompt zeroed). The
        // CFG weights (1+r)*0.25 - r*0.25 cancel to 0.25 as well.
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let cfm = load_tiny(&VarBuilder::from_varmap(&varmap, DType::F32, &device));
        zero_all(&mut varmap);
        varmap
            .set_one(
                "estimator.conv2.bias",
                Tensor::full(0.25f32, 4, &device).unwrap(),
            )
            .unwrap();
        let l = 24;
        let t_ref = 8;
        let condition = Tensor::zeros((1, l, 16), DType::F32, &device).unwrap();
        let ref_mel = Tensor::zeros((1, 4, t_ref), DType::F32, &device).unwrap();
        let style = Tensor::zeros((1, crate::dit::STYLE_DIM), DType::F32, &device).unwrap();
        let out = cfm
            .inference(&condition, &[l], &ref_mel, &style, 10, 0.7)
            .unwrap();
        let frames = out.transpose(1, 2).unwrap().to_vec3::<f32>().unwrap();
        assert!(frames[0].iter().take(t_ref).flatten().all(|&v| v == 0.0));
        // z recovered: out_tail - 0.25 must look like N(0,1).
        let tail: Vec<f32> = frames[0][t_ref..]
            .iter()
            .flatten()
            .map(|&v| v - 0.25)
            .collect();
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!(mean.abs() < 0.5, "shifted tail mean {mean}");
        assert!(tail.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rejects_partial_lengths_and_bad_prompt() {
        let device = Device::Cpu;
        let cfm = load_tiny(&VarBuilder::zeros(DType::F32, &device));
        let condition = Tensor::zeros((1, 12, 16), DType::F32, &device).unwrap();
        let ref_mel = Tensor::zeros((1, 4, 4), DType::F32, &device).unwrap();
        let style = Tensor::zeros((1, crate::dit::STYLE_DIM), DType::F32, &device).unwrap();
        assert!(
            cfm.inference(&condition, &[8], &ref_mel, &style, 2, 0.7)
                .is_err()
        );
        let long_ref = Tensor::zeros((1, 4, 13), DType::F32, &device).unwrap();
        assert!(
            cfm.inference(&condition, &[12], &long_ref, &style, 2, 0.7)
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires converted checkpoints (INDEXTTS_CONVERTED_DIR)"]
    fn real_weights_inference() {
        let dir = match std::env::var("INDEXTTS_CONVERTED_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => return,
        };
        let device = Device::Cpu;
        let tensors =
            candle_core::safetensors::load(dir.join("s2mel.safetensors"), &device).unwrap();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let cfg = crate::config::S2MelConfig::default();
        let cfm = Cfm::load(&vb.pp("models").pp("cfm"), &cfg).unwrap();
        let l = 96;
        let t_ref = 32;
        let condition = Tensor::randn(0f32, 1f32, (1, l, 512), &device).unwrap();
        let ref_mel = Tensor::randn(0f32, 1f32, (1, 80, t_ref), &device).unwrap();
        let style = Tensor::randn(0f32, 1f32, (1, crate::dit::STYLE_DIM), &device).unwrap();
        let out = cfm
            .inference(&condition, &[l], &ref_mel, &style, 25, 0.7)
            .unwrap();
        assert_eq!(out.dims(), vec![1, 80, l]);
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }
}
