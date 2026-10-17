//! PerceiverResampler (`indextts/gpt/perceiver.py`).
//!
//! Upstream constructs it for the emotion path as
//! `PerceiverResampler(1024, dim_context=<conformer output>, ff_mult=2,
//! heads=4, num_latents=1)` — depth defaults to 2, `dim_head` to 64 and the
//! cross-attention includes the latent queries themselves in the key/value
//! set (`cross_attn_include_queries=True`), so the attention mask covers
//! `[latents, context]`. Attended scores are masked with `-f32::MAX`
//! (torch `-finfo.max`) before the softmax, RMSNorm normalizes each row and
//! scales it by `sqrt(dim) * gamma`, and GEGLU splits the hidden dim with
//! an erf-based GELU on the gate half (`F.gelu(gate) * x` — gate is the
//! SECOND half).
//!
//! Weight names (upstream state dict): `proj_context.{weight,bias}` (only
//! when `dim_context != dim`, else Identity), `latents`,
//! `layers.{d}.0.to_{q,kv,out}.weight` (no biases),
//! `layers.{d}.1.{0,2}.weight/bias` (feed-forward Linears around GEGLU) and
//! `norm.gamma` (RMSNorm).

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder, linear_b};

/// Default `dim_head` of the lucidrains Attention module.
const DIM_HEAD: usize = 64;

/// Default `depth` of PerceiverResampler.
const DEPTH: usize = 2;

pub struct PerceiverResampler {
    /// `nn.Identity` when `dim_context == dim`.
    proj_context: Option<candle_nn::Linear>,
    latents: Tensor,
    layers: Vec<PerceiverLayer>,
    norm_gamma: Tensor,
    dim: usize,
}

struct PerceiverLayer {
    attn: ResamplerAttention,
    ff: ResamplerFeedForward,
}

struct ResamplerAttention {
    to_q: candle_nn::Linear,
    to_kv: candle_nn::Linear,
    to_out: candle_nn::Linear,
    heads: usize,
    scale: f64,
}

struct ResamplerFeedForward {
    /// `layers.d.1.0` — Linear(dim, dim_inner * 2) feeding GEGLU.
    net_in: candle_nn::Linear,
    /// `layers.d.1.2` — Linear(dim_inner, dim) after GEGLU.
    net_out: candle_nn::Linear,
}

impl PerceiverResampler {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VarBuilder,
        dim: usize,
        dim_context: usize,
        ff_mult: usize,
        heads: usize,
        num_latents: usize,
    ) -> Result<Self> {
        let proj_context = if dim_context != dim {
            Some(candle_nn::linear(dim_context, dim, vb.pp("proj_context"))?)
        } else {
            None
        };
        let latents = vb.get((num_latents, dim), "latents")?;
        if latents.dims() != [num_latents, dim] {
            candle_core::bail!(
                "latents shape {:?} != ({num_latents}, {dim})",
                latents.dims()
            );
        }
        let dim_inner = dim * ff_mult * 2 / 3;
        let mut layers = Vec::with_capacity(DEPTH);
        for d in 0..DEPTH {
            let layer_vb = vb.pp("layers").pp(d.to_string());
            layers.push(PerceiverLayer {
                attn: ResamplerAttention {
                    to_q: linear_b(dim, DIM_HEAD * heads, false, layer_vb.pp("0").pp("to_q"))?,
                    to_kv: linear_b(
                        dim,
                        2 * DIM_HEAD * heads,
                        false,
                        layer_vb.pp("0").pp("to_kv"),
                    )?,
                    to_out: linear_b(DIM_HEAD * heads, dim, false, layer_vb.pp("0").pp("to_out"))?,
                    heads,
                    scale: (DIM_HEAD as f64).powf(-0.5),
                },
                ff: ResamplerFeedForward {
                    // Upstream FeedForward uses plain `nn.Linear` — these
                    // carry biases (unlike the attention projections).
                    net_in: candle_nn::linear(dim, dim_inner * 2, layer_vb.pp("1").pp("0"))?,
                    net_out: candle_nn::linear(dim_inner, dim, layer_vb.pp("1").pp("2"))?,
                },
            });
        }
        let norm_gamma = vb.get((dim,), "norm.gamma")?;
        Ok(Self {
            proj_context,
            latents,
            layers,
            norm_gamma,
            dim,
        })
    }

    /// `x`: `(B, T, dim_context)`; `mask`: `(B, T)` u8/bool (true = valid).
    /// Returns `(B, num_latents, dim)`.
    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, _, _) = x.dims3()?;
        let x = match &self.proj_context {
            Some(proj) => proj.forward(x)?,
            None => x.clone(),
        };

        // latents (B, num_latents, dim); attention keys/values cover
        // [latents, context], so the attend mask is [1, valid-context].
        let num_latents = self.latents.dim(0)?;
        let mut latents = self
            .latents
            .unsqueeze(0)?
            .broadcast_as((b, num_latents, self.dim))?;
        let mask = mask.to_dtype(candle_core::DType::U8)?;
        let ones = Tensor::ones((b, 1), candle_core::DType::U8, x.device())?;
        let attend_mask = Tensor::cat(&[&ones, &mask], 1)?;

        for layer in &self.layers {
            let attn_out = layer.attn.forward(&latents, &x, &attend_mask)?;
            latents = (&latents + attn_out)?;
            let ff_out = layer.ff.forward(&latents)?;
            latents = (&latents + ff_out)?;
        }

        // RMSNorm: x / max(||x||, eps) * sqrt(dim) * gamma
        let normed = rms_norm_rows(&latents)?;
        let scale = (self.dim as f64).sqrt();
        let out = (normed * scale)?.broadcast_mul(&self.norm_gamma.unsqueeze(0)?.unsqueeze(0)?)?;
        out.contiguous()
    }

    /// Test-only mirror of [`Self::forward`] recording the per-layer
    /// attention / feed-forward sub-outputs (before their residual adds,
    /// like the Python forward hooks) plus the final normed output.
    #[cfg(test)]
    pub(crate) fn forward_stage_dump(
        &self,
        x: &Tensor,
        mask: &Tensor,
    ) -> Result<Vec<(String, Tensor)>> {
        let (b, _, _) = x.dims3()?;
        let x = match &self.proj_context {
            Some(proj) => proj.forward(x)?,
            None => x.clone(),
        };
        let num_latents = self.latents.dim(0)?;
        let mut latents = self
            .latents
            .unsqueeze(0)?
            .broadcast_as((b, num_latents, self.dim))?;
        let mask = mask.to_dtype(candle_core::DType::U8)?;
        let ones = Tensor::ones((b, 1), candle_core::DType::U8, x.device())?;
        let attend_mask = Tensor::cat(&[&ones, &mask], 1)?;

        let mut stages = Vec::new();
        stages.push(("per.latents0".to_string(), latents.clone()));
        stages.push(("per.ctx".to_string(), x.clone()));
        stages.push((
            "per.kv_in".to_string(),
            Tensor::cat(&[&latents, &x], 1)?.contiguous()?,
        ));
        for (d, layer) in self.layers.iter().enumerate() {
            let attn_out = if d == 0 {
                let (out, inner) = layer.attn.forward_stage_dump(&latents, &x, &attend_mask)?;
                stages.extend(inner);
                stages.push(("per.attn0".to_string(), out.clone()));
                out
            } else {
                let out = layer.attn.forward(&latents, &x, &attend_mask)?;
                stages.push((format!("per.attn{d}"), out.clone()));
                out
            };
            latents = (&latents + &attn_out)?;
            if d == 0 {
                stages.push(("per.post_attn0".to_string(), latents.clone()));
                // feed-forward internals (mirror the Python manual chain)
                let y = layer.ff.net_in.forward(&latents)?;
                stages.push(("per.ff0_y".to_string(), y.clone()));
                let halves = y.chunk(2, 2)?;
                stages.push(("per.ff0_x".to_string(), halves[0].clone()));
                stages.push(("per.ff0_gate".to_string(), halves[1].clone()));
                let gelu = halves[1].gelu_erf()?;
                stages.push(("per.ff0_gelu".to_string(), gelu.clone()));
                let geg = (&gelu * &halves[0])?;
                stages.push(("per.ff0_geglu".to_string(), geg.clone()));
                let ff_out = layer.ff.net_out.forward(&geg)?;
                stages.push(("per.ff0_out".to_string(), ff_out.clone()));
                latents = (&latents + &ff_out)?;
                stages.push(("per.post_ff0".to_string(), latents.clone()));
                continue;
            }
            let ff_out = layer.ff.forward(&latents)?;
            latents = (&latents + &ff_out)?;
            stages.push((format!("per.ff{d}"), ff_out));
        }
        let normed = rms_norm_rows(&latents)?;
        let scale = (self.dim as f64).sqrt();
        let out = (normed * scale)?.broadcast_mul(&self.norm_gamma.unsqueeze(0)?.unsqueeze(0)?)?;
        stages.push(("perceiver".to_string(), out.contiguous()?));
        Ok(stages)
    }
}

impl ResamplerAttention {
    /// `q_src`: `(B, N, dim)` latents; `context`: `(B, T, dim)` projected
    /// context; `mask`: `(B, 1 + T)` over `[latents, context]`.
    fn forward(&self, q_src: &Tensor, context: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let b = q_src.dim(0)?;
        let n = q_src.dim(1)?;
        let kv_in = Tensor::cat(&[q_src, context], 1)?.contiguous()?;
        let t_kv = kv_in.dim(1)?;

        let q = self.to_q.forward(q_src)?;
        let kv = self.to_kv.forward(&kv_in)?;
        let (k, v) = {
            let halves = kv.chunk(2, 2)?;
            (halves[0].clone(), halves[1].clone())
        };

        let q = q
            .reshape((b, n, self.heads, ()))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t_kv, self.heads, ()))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t_kv, self.heads, ()))?
            .transpose(1, 2)?
            .contiguous()?;

        let mut sim = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        sim = (sim * self.scale)?;
        // key-padding mask, -torch.finfo(float32).max like upstream Attend
        let invalid = mask
            .ones_like()?
            .sub(mask)?
            .unsqueeze(1)?
            .unsqueeze(1)?
            .broadcast_as(sim.shape())?;
        let neg_max = Tensor::full(f32::MIN, sim.shape(), sim.device())?;
        let sim = invalid.where_cond(&neg_max, &sim)?;
        let attn = candle_nn::ops::softmax_last_dim(&sim.contiguous()?)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, n, ()))?.contiguous()?;
        self.to_out.forward(&out)
    }

    /// Test-only mirror of [`Self::forward`] recording the attention math
    /// internals (`q/k/v`, `sim`, `attn`). Returns `(out, internals)`.
    #[cfg(test)]
    fn forward_stage_dump(
        &self,
        q_src: &Tensor,
        context: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let (out, q, k, v, sim, attn) = self.forward_parts(q_src, context, mask)?;
        let stages = vec![
            ("per.a0.q".to_string(), q),
            ("per.a0.k".to_string(), k),
            ("per.a0.v".to_string(), v),
            ("per.a0.sim".to_string(), sim),
            ("per.a0.attn".to_string(), attn),
        ];
        Ok((out, stages))
    }

    /// Shared computation of [`Self::forward`] exposing intermediates.
    #[cfg(test)]
    fn forward_parts(
        &self,
        q_src: &Tensor,
        context: &Tensor,
        mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let b = q_src.dim(0)?;
        let n = q_src.dim(1)?;
        let kv_in = Tensor::cat(&[q_src, context], 1)?.contiguous()?;
        let t_kv = kv_in.dim(1)?;

        let q = self.to_q.forward(q_src)?;
        let kv = self.to_kv.forward(&kv_in)?;
        let (k, v) = {
            let halves = kv.chunk(2, 2)?;
            (halves[0].clone(), halves[1].clone())
        };

        let q = q
            .reshape((b, n, self.heads, ()))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t_kv, self.heads, ()))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t_kv, self.heads, ()))?
            .transpose(1, 2)?
            .contiguous()?;

        let mut sim = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        sim = (sim * self.scale)?;
        let invalid = mask
            .ones_like()?
            .sub(mask)?
            .unsqueeze(1)?
            .unsqueeze(1)?
            .broadcast_as(sim.shape())?;
        let neg_max = Tensor::full(f32::MIN, sim.shape(), sim.device())?;
        let sim = invalid.where_cond(&neg_max, &sim)?;
        let attn = candle_nn::ops::softmax_last_dim(&sim.contiguous()?)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, n, ()))?.contiguous()?;
        let out = self.to_out.forward(&out)?;
        Ok((out, q, k, v, sim, attn))
    }
}

impl ResamplerFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let halves = self.net_in.forward(x)?.chunk(2, 2)?;
        // GEGLU: x, gate = chunk(2); F.gelu(gate) * x  (gate = second half)
        let gated = (halves[1].gelu_erf()? * &halves[0])?;
        self.net_out.forward(&gated)
    }
}

/// `F.normalize(x, dim=-1)`: divide each row by max(||row||_2, 1e-12).
fn rms_norm_rows(x: &Tensor) -> Result<Tensor> {
    let sq = x.powf(2.0)?;
    let denom = sq.sum_keepdim(2)?.sqrt()?.clamp(1e-12, f32::MAX as f64)?;
    x.broadcast_div(&denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{Init, VarBuilder, VarMap};

    fn tiny_vb(
        dim: usize,
        dim_context: usize,
        heads: usize,
        num_latents: usize,
    ) -> VarBuilder<'static> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let dim_inner = dim * 2 * 2 / 3; // ff_mult = 2 (single truncation like int(dim*mult*2/3))
        let put = |vm: &VarMap, name: &str, shape: &[usize]| {
            vm.get(shape, name, Init::Const(0.1), candle_core::DType::F32, &dev)
                .unwrap();
        };
        put(&vm, "proj_context.weight", &[dim, dim_context]);
        put(&vm, "proj_context.bias", &[dim]);
        put(&vm, "latents", &[num_latents, dim]);
        for d in 0..DEPTH {
            put(
                &vm,
                &format!("layers.{d}.0.to_q.weight"),
                &[heads * DIM_HEAD, dim],
            );
            put(
                &vm,
                &format!("layers.{d}.0.to_kv.weight"),
                &[2 * heads * DIM_HEAD, dim],
            );
            put(
                &vm,
                &format!("layers.{d}.0.to_out.weight"),
                &[dim, heads * DIM_HEAD],
            );
            put(
                &vm,
                &format!("layers.{d}.1.0.weight"),
                &[dim_inner * 2, dim],
            );
            put(&vm, &format!("layers.{d}.1.0.bias"), &[dim_inner * 2]);
            put(&vm, &format!("layers.{d}.1.2.weight"), &[dim, dim_inner]);
            put(&vm, &format!("layers.{d}.1.2.bias"), &[dim]);
        }
        put(&vm, "norm.gamma", &[dim]);
        VarBuilder::from_varmap(&vm, candle_core::DType::F32, &dev)
    }

    #[test]
    fn shapes_emo_configuration() {
        // Emotion resampler: dim=1024, dim_context=512, ff_mult=2, heads=4,
        // num_latents=1.
        let dev = Device::Cpu;
        let vb = tiny_vb(1024, 512, 4, 1);
        let model = PerceiverResampler::load(&vb, 1024, 512, 2, 4, 1).expect("load");
        // Constant 0.1 test weights make the 1024-dim FF grow ~x^2 per
        // layer, so feed a small-scale input to stay inside f32; real
        // trained weights sit in sane ranges.
        let x = (Tensor::randn(0f32, 1f32, (2, 9, 512), &dev).unwrap() * 0.01f64).unwrap();
        let mask_vals: Vec<u8> = [1u8, 1, 1, 1, 1, 0, 0, 0, 0]
            .into_iter()
            .cycle()
            .take(2 * 9)
            .collect();
        let mask = Tensor::from_vec(mask_vals, (2, 9), &dev).unwrap();
        let out = model.forward(&x, &mask).expect("forward");
        assert_eq!(out.dims(), vec![2, 1, 1024]);
        let o = out.to_vec3::<f32>().unwrap();
        assert!(o.iter().flatten().flatten().all(|v| v.is_finite()));
    }

    #[test]
    fn masked_context_rows_do_not_change_latent() {
        // Padded context rows must not influence the latent output.
        let dev = Device::Cpu;
        let vb = tiny_vb(64, 32, 2, 1);
        let model = PerceiverResampler::load(&vb, 64, 32, 2, 2, 1).expect("load");
        let x = Tensor::randn(0f32, 1f32, (1, 8, 32), &dev).unwrap();
        let mask = Tensor::from_vec(vec![1u8, 1, 1, 1, 1, 0, 0, 0], (1, 8), &dev).unwrap();
        let out_a = model.forward(&x, &mask).unwrap();

        // scramble the padded region — result must be identical
        let mut vals = x.to_vec3::<f32>().unwrap();
        for row in vals[0].iter_mut().skip(5) {
            for v in row.iter_mut() {
                *v = 77.0;
            }
        }
        let flat: Vec<f32> = vals.into_iter().flatten().flatten().collect();
        let x2 = Tensor::from_vec(flat, (1, 8, 32), &dev).unwrap();
        let out_b = model.forward(&x2, &mask).unwrap();
        let a = out_a.to_vec3::<f32>().unwrap();
        let b = out_b.to_vec3::<f32>().unwrap();
        let max_diff = a[0][0]
            .iter()
            .zip(b[0][0].iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "padded rows leaked into latent: {max_diff}"
        );
    }

    #[test]
    fn identity_projection_when_dims_match() {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let put = |vm: &VarMap, name: &str, shape: &[usize]| {
            vm.get(shape, name, Init::Const(0.1), candle_core::DType::F32, &dev)
                .unwrap();
        };
        let (dim, heads) = (32usize, 2);
        let dim_inner = dim * 2 * 2 / 3;
        put(&vm, "latents", &[1, dim]);
        for d in 0..DEPTH {
            put(
                &vm,
                &format!("layers.{d}.0.to_q.weight"),
                &[heads * DIM_HEAD, dim],
            );
            put(
                &vm,
                &format!("layers.{d}.0.to_kv.weight"),
                &[2 * heads * DIM_HEAD, dim],
            );
            put(
                &vm,
                &format!("layers.{d}.0.to_out.weight"),
                &[dim, heads * DIM_HEAD],
            );
            put(
                &vm,
                &format!("layers.{d}.1.0.weight"),
                &[dim_inner * 2, dim],
            );
            put(&vm, &format!("layers.{d}.1.0.bias"), &[dim_inner * 2]);
            put(&vm, &format!("layers.{d}.1.2.weight"), &[dim, dim_inner]);
            put(&vm, &format!("layers.{d}.1.2.bias"), &[dim]);
        }
        put(&vm, "norm.gamma", &[dim]);
        let vb = VarBuilder::from_varmap(&vm, candle_core::DType::F32, &dev);
        let model = PerceiverResampler::load(&vb, dim, dim, 2, heads, 1).expect("load");
        assert!(model.proj_context.is_none());
        let x = Tensor::randn(0f32, 1f32, (1, 5, dim), &dev).unwrap();
        let mask = Tensor::ones((1, 5), candle_core::DType::U8, &dev).unwrap();
        let out = model.forward(&x, &mask).expect("forward");
        assert_eq!(out.dims(), vec![1, 1, dim]);
    }
}
