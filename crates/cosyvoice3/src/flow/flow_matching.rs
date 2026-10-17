//! Causal conditional flow matching (`cosyvoice/flow/flow_matching.py`).
//!
//! Deterministic sampling: the initial noise comes from the seed-0
//! `torch.randn([1, 80, 50 * 300])` buffer baked into
//! `CausalConditionalCFM.__init__` (generic ATen CPU algorithm through
//! [`crate::rng::TorchCpuRng`]). CFG runs the estimator on a batch of two:
//! row 0 conditioned, row 1 zeroed (unconditional).

use std::sync::OnceLock;

use candle_core::{Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder};

use super::dit::DiT;
use super::pre_lookahead::PreLookaheadLayer;
use crate::config::{CfmConfig, FlowConfig};

/// Maximum sequence length of the fixed official CFM noise table.
pub(crate) const MAX_MEL_FRAMES: usize = 15_000;

/// Cached seed-0 noise table `[1, 80, 15000]`.
fn rand_noise_table(device: &candle_core::Device, dtype: candle_core::DType) -> Result<Tensor> {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    let data = TABLE.get_or_init(|| crate::rng::TorchCpuRng::new(0).randn_vec(80 * MAX_MEL_FRAMES));
    Tensor::from_slice(data, (1, 80, MAX_MEL_FRAMES), device)?.to_dtype(dtype)
}

#[derive(Debug)]
pub struct CausalConditionalCfm {
    estimator: DiT,
    cfg: CfmConfig,
    /// Custom noise override for fixture-driven comparison.
    noise_override: Option<Tensor>,
}

impl CausalConditionalCfm {
    pub fn new(estimator: DiT, cfg: CfmConfig) -> Self {
        Self {
            estimator,
            cfg,
            noise_override: None,
        }
    }

    pub fn with_noise(mut self, noise: Tensor) -> Self {
        self.noise_override = Some(noise);
        self
    }

    /// Use shared Python noise for a cross-platform numerical comparison.
    pub fn set_noise(&mut self, noise: Tensor) {
        self.noise_override = Some(noise);
    }

    /// `mu`: `[B, 80, T]`; `mask`: `[B, 1, T]`; `spks`: `[B, 80]`;
    /// `cond`: `[B, 80, T]`. `mask_all_ones` lets a caller whose mask is
    /// known to be all-ones skip building the attention bias. Returns the
    /// solved mel `[B, 80, T]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        mu: &Tensor,
        mask: &Tensor,
        n_timesteps: usize,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
        mask_all_ones: bool,
    ) -> Result<Tensor> {
        if mu.dim(0)? != 1 || n_timesteps == 0 {
            candle_core::bail!("CFM requires batch=1 and at least one solver step");
        }
        let seq = mu.dim(2)?;
        let dtype = mu.dtype();
        let device = mu.device();
        let z = match &self.noise_override {
            Some(n) => n.narrow(2, 0, seq)?.to_device(device)?.to_dtype(dtype)?,
            None => rand_noise_table(device, dtype)?.narrow(2, 0, seq)?,
        };
        let t_span = t_span(n_timesteps, self.cfg.t_scheduler.as_str(), device, dtype)?;
        self.solve_euler(z, &t_span, mu, mask, spks, cond, streaming, mask_all_ones)
    }

    #[allow(clippy::too_many_arguments)]
    fn solve_euler(
        &self,
        x: Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        mask: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
        streaming: bool,
        mask_all_ones: bool,
    ) -> Result<Tensor> {
        let n = t_span.dim(0)? - 1;
        let seq = x.dim(2)?;
        let dtype = x.dtype();
        let device = x.device().clone();
        let use_cfg = self.cfg.inference_cfg_rate > 0.0;

        let prepared = if use_cfg {
            // Batch of two: [conditioned, unconditioned].
            let mask_in = Tensor::cat(&[mask, mask], 0)?;
            let zeros_mu = Tensor::zeros((1, 80, seq), dtype, &device)?;
            let mu_in = Tensor::cat(&[mu, &zeros_mu], 0)?;
            let zeros_spks = Tensor::zeros((1, 80), dtype, &device)?;
            let spks_in = Tensor::cat(&[spks, &zeros_spks], 0)?;
            let zeros_cond = Tensor::zeros((1, 80, seq), dtype, &device)?;
            let cond_in = Tensor::cat(&[cond, &zeros_cond], 0)?;
            self.estimator.prepare_solve(
                &mask_in,
                &mu_in,
                &spks_in,
                &cond_in,
                streaming,
                mask_all_ones,
            )?
        } else {
            self.estimator
                .prepare_solve(mask, mu, spks, cond, streaming, mask_all_ones)?
        };

        let t_vals: Vec<f32> = t_span.to_dtype(candle_core::DType::F32)?.to_vec1()?;
        let mut x = x;
        let mut t: f32 = t_vals[0];
        let mut dt: f32 = t_vals[1] - t;

        for step in 1..=n {
            let dphi = if use_cfg {
                let x_in = Tensor::cat(&[&x, &x], 0)?;
                let t_in = Tensor::from_vec(vec![t, t], 2, &device)?.to_dtype(dtype)?;
                let out = self
                    .estimator
                    .forward_prepared(&x_in, &t_in, &prepared, None)?;
                let cond_part = out.narrow(0, 0, 1)?;
                let uncond_part = out.narrow(0, 1, 1)?;
                (cond_part.affine(1.0 + self.cfg.inference_cfg_rate, 0.0)?
                    - uncond_part.affine(self.cfg.inference_cfg_rate, 0.0)?)?
            } else {
                let t_in = Tensor::from_vec(vec![t], 1, &device)?.to_dtype(dtype)?;
                self.estimator
                    .forward_prepared(&x, &t_in, &prepared, None)?
            };
            x = (x + dphi.affine(dt as f64, 0.0)?)?;
            t += dt;
            if step < n {
                dt = t_vals[step + 1] - t;
            }
        }
        Ok(x)
    }
}

/// `linspace(0, 1, n+1)` with optional cosine schedule, in `dtype`.
fn t_span(
    n: usize,
    scheduler: &str,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let mut vals: Vec<f32> = (0..=n).map(|i| i as f32 / n as f32).collect();
    if scheduler == "cosine" {
        for v in vals.iter_mut() {
            // Match torch's float32 multiply/cos/subtract order.
            *v = 1.0 - (*v * 0.5 * std::f32::consts::PI).cos();
        }
    }
    Tensor::from_vec(vals, n + 1, device)?.to_dtype(dtype)
}

/// Complete flow decoder: token embedding, pre-lookahead upsampling, spk
/// projection, and CFM sampling (`CausalMaskedDiffWithDiT`).
#[derive(Debug)]
pub struct CausalMaskedDiffWithDiT {
    input_embedding: Embedding,
    spk_embed_affine_layer: Linear,
    pre_lookahead_layer: PreLookaheadLayer,
    cfm: CausalConditionalCfm,
    output_size: usize,
}

impl CausalMaskedDiffWithDiT {
    pub fn new(flow: &FlowConfig, spk_embed_dim: usize, dit: DiT, vb: VarBuilder) -> Result<Self> {
        // The yaml instantiates PreLookaheadLayer(80, 1024, pre_lookahead 3).
        let pre = PreLookaheadLayer::new(
            flow.input_size,
            1024,
            flow.output_size,
            flow.pre_lookahead_len,
            flow.token_mel_ratio,
            vb.pp("pre_lookahead_layer"),
        )?;
        Ok(Self {
            input_embedding: candle_nn::embedding(
                flow.vocab_size,
                flow.input_size,
                vb.pp("input_embedding"),
            )?,
            spk_embed_affine_layer: candle_nn::linear(
                spk_embed_dim,
                flow.output_size,
                vb.pp("spk_embed_affine_layer"),
            )?,
            pre_lookahead_layer: pre,
            cfm: CausalConditionalCfm::new(dit, flow.cfm.clone()),
            output_size: flow.output_size,
        })
    }

    pub fn cfm_mut(&mut self) -> &mut CausalConditionalCfm {
        &mut self.cfm
    }

    /// Device the flow weights live on.
    pub fn device(&self) -> &candle_core::Device {
        self.spk_embed_affine_layer.weight().device()
    }

    /// Solver step count from the yaml (`n_timesteps`).
    pub fn n_timesteps(&self) -> usize {
        self.cfm.cfg.n_timesteps
    }

    /// Mel frames per speech token (2 in the yaml).
    pub fn token_mel_ratio(&self) -> usize {
        self.pre_lookahead_layer.token_mel_ratio()
    }

    pub fn pre_lookahead_len(&self) -> usize {
        self.pre_lookahead_layer.pre_lookahead_len()
    }

    /// `tokens`: `[B, T]` u32 (generated); `prompt_tokens`: `[B, T']`;
    /// `prompt_feat`: `[B, T'_*ratio, 80]` prompt mel; `embedding`:
    /// `[B, 192]` raw speaker embedding. Returns mel `[B, 80, T*ratio]`
    /// for the generated part only.
    ///
    /// `finalize=false` feeds the trailing `pre_lookahead_len` tokens as
    /// look-ahead context only (official `token[:, :-pre_lookahead]` split),
    /// so the output covers `T - pre_lookahead_len` tokens; `finalize=true`
    /// processes every token with zero right-padding.
    #[allow(clippy::too_many_arguments)]
    pub fn inference(
        &self,
        tokens: &Tensor,
        prompt_tokens: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        n_timesteps: usize,
        streaming: bool,
        finalize: bool,
    ) -> Result<Tensor> {
        let batch = tokens.dim(0)?;
        if tokens.dim(1)? == 0 || (!finalize && tokens.dim(1)? <= self.pre_lookahead_len()) {
            candle_core::bail!("flow requires speech tokens beyond its non-final lookahead");
        }
        if batch != 1 || prompt_feat.dim(1)? != prompt_tokens.dim(1)? * 2 {
            candle_core::bail!("flow requires batch=1 and two prompt mel frames per token");
        }
        // F.normalize(embedding, dim=1) then Linear(192 -> 80).
        let norm = embedding
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?;
        let embedding = self
            .spk_embed_affine_layer
            .forward(&embedding.broadcast_div(&norm)?)?;

        let all_tokens = Tensor::cat(&[prompt_tokens, tokens], 1)?;
        // Official clamps negatives away; u32 inputs are already >= 0.
        let token_emb = self.input_embedding.forward(&all_tokens)?;

        let look = self.pre_lookahead_len();
        let mu = if finalize || all_tokens.dim(1)? <= look {
            self.pre_lookahead_layer.forward(&token_emb, None)?
        } else {
            let n = all_tokens.dim(1)?;
            let main = token_emb.narrow(1, 0, n - look)?;
            let ctx = token_emb.narrow(1, n - look, look)?;
            self.pre_lookahead_layer.forward(&main, Some(&ctx))?
        };
        let mu = mu.transpose(1, 2)?;

        let prompt_mel_len = prompt_feat.dim(1)?;
        let mel_len = mu.dim(2)?;
        let gen_len = mel_len.checked_sub(prompt_mel_len).ok_or_else(|| {
            candle_core::Error::Msg("prompt mel exceeds decoded sequence length".into())
        })?;

        let mask = Tensor::ones((batch, 1, mel_len), mu.dtype(), mu.device())?;
        let prompt = prompt_feat.transpose(1, 2)?;
        let pad = Tensor::zeros(
            (batch, self.output_size, gen_len),
            prompt.dtype(),
            prompt.device(),
        )?;
        let cond = Tensor::cat(&[&prompt, &pad], 2)?;

        let full = self
            .cfm
            .forward(&mu, &mask, n_timesteps, &embedding, &cond, streaming, true)?;
        full.narrow(2, prompt_mel_len, gen_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_t_span_matches_python() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let t = t_span(10, "cosine", &device, candle_core::DType::F32)?;
        let v: Vec<f32> = t.to_vec1()?;
        assert!((v[0]).abs() < 1e-7);
        assert!((v[10] - 1.0).abs() < 1e-6);
        // midpoint: 1 - cos(pi/4)
        assert!((v[5] - (1.0 - (std::f64::consts::FRAC_PI_2 * 0.5).cos()) as f32).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn seed0_noise_first_values_match_manual_normal_fill() -> Result<()> {
        // Cross-check the shared table against a direct TorchCpuRng draw.
        let device = candle_core::Device::Cpu;
        let table = rand_noise_table(&device, candle_core::DType::F32)?;
        let mut rng = crate::rng::TorchCpuRng::new(0);
        // Compare only full 16-blocks: the table size is block-aligned, so
        // its element 960 comes from a block transform while a 961-length
        // draw would use the scalar remainder path.
        let direct = rng.randn_vec(960);
        let flat: Vec<f32> = table.flatten_all()?.to_vec1()?;
        for i in 0..direct.len() {
            assert_eq!(flat[i], direct[i]);
        }
        Ok(())
    }
}
