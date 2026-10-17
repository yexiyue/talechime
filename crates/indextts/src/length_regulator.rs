//! InterpolateRegulator (`indextts/s2mel/modules/length_regulator.py`).
//!
//! Shipped 2.5 configuration: continuous input (`is_discrete=false`), no f0,
//! no VQ, single codebook, no quantizer dropout. Forward = `content_in_proj`
//! Linear(1024→512) → nearest-neighbor interpolation of the channel-major
//! tensor to `ylens.max()` → 4 × (Conv1d k3 p1 + GroupNorm(1, C) + Mish) →
//! Conv1d k1 → mask by sequence lengths. The discrete-token embedding, VQ,
//! f0 and multi-codebook branches of upstream are intentionally not ported.

use crate::config::LengthRegulatorConfig;
use crate::s2mel::sequence_mask_on;
use candle_core::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, GroupNorm, Linear, Module, VarBuilder, conv1d, group_norm};

pub struct InterpolateRegulator {
    content_in_proj: Linear,
    /// `(conv, group_norm)` per sampling ratio (shipped: 4 entries).
    blocks: Vec<(Conv1d, GroupNorm)>,
    /// Final `Conv1d(C, out_channels=C, 1)`.
    out_proj_w: Tensor,
    out_proj_b: Tensor,
}

/// Nearest-neighbor frame mapping used by `F.interpolate(mode='nearest')`:
/// output frame `i` copies input frame `floor(i * t_in / t_out)`.
fn nearest_indices(t_in: usize, t_out: usize) -> Vec<u32> {
    (0..t_out).map(|i| (i * t_in / t_out) as u32).collect()
}

impl InterpolateRegulator {
    /// Weight names under `models.length_regulator.*` in s2mel.pth
    /// (`content_in_proj.*` and `model.{0,3,6,9}`/`model.{1,4,7,10}`/
    /// `model.12` from the flattened `nn.Sequential`). The unused
    /// `embedding.*` / `mask_token` checkpoint entries are not requested.
    pub fn load(vb: &VarBuilder, cfg: &LengthRegulatorConfig) -> Result<Self> {
        if cfg.is_discrete {
            candle_core::bail!("the shipped length regulator is continuous (is_discrete=true)");
        }
        if cfg.vector_quantize {
            candle_core::bail!("vector_quantize is not part of the shipped checkpoint");
        }
        if cfg.f0_condition {
            candle_core::bail!("f0_condition is not part of the shipped checkpoint");
        }
        if cfg.n_codebooks != 1 {
            candle_core::bail!("only n_codebooks=1 is shipped (got {})", cfg.n_codebooks);
        }
        let content_in_proj =
            candle_nn::linear(cfg.in_channels, cfg.channels, vb.pp("content_in_proj"))?;
        // nn.Sequential layout: per sampling ratio [conv, group_norm, mish]
        // triples (indices 3k, 3k+1), then the final Conv1d at index 3*n.
        let mut blocks = Vec::with_capacity(cfg.sampling_ratios.len());
        for i in 0..cfg.sampling_ratios.len() {
            let conv = conv1d(
                cfg.channels,
                cfg.channels,
                3,
                Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp(format!("model.{}", 3 * i)),
            )?;
            // nn.GroupNorm(1, channels) with the PyTorch default eps.
            let norm = group_norm(1, cfg.channels, 1e-5, vb.pp(format!("model.{}", 3 * i + 1)))?;
            blocks.push((conv, norm));
        }
        // The final k=1 "conv" is applied as an explicit contiguous matmul:
        // candle's CUDA conv1d special-cases kernel size 1 into a matmul
        // that rejects the strided views produced upstream.
        let proj_idx = 3 * cfg.sampling_ratios.len();
        let out_proj_w = vb.get(
            (cfg.channels, cfg.channels, 1),
            &format!("model.{proj_idx}.weight"),
        )?;
        let out_proj_b = vb.get((cfg.channels,), &format!("model.{proj_idx}.bias"))?;
        Ok(Self {
            content_in_proj,
            blocks,
            out_proj_w,
            out_proj_b,
        })
    }

    fn out_proj_forward(&self, h: &Tensor) -> Result<Tensor> {
        // y[b,o,t] = sum_i w[o,i,0] * x[b,i,t] + b[o]  (k=1 conv == matmul)
        let (b, c, t) = h.dims3()?;
        let w = self.out_proj_w.squeeze(2)?.t()?.contiguous()?;
        h.transpose(1, 2)?
            .contiguous()?
            .reshape((b * t, c))?
            .matmul(&w)?
            .broadcast_add(&self.out_proj_b)?
            .reshape((b, t, c))?
            .transpose(1, 2)
    }

    /// `x`: `(B, T, 1024)` continuous semantic features; `ylens` per-sample
    /// target lengths. Returns `(B, ylens.max(), 512)` masked output
    /// (`out * sequence_mask(ylens)`; upstream also returns olens/None which
    /// the Rust caller contract drops).
    pub fn forward(&self, x: &Tensor, ylens: &[usize]) -> Result<Tensor> {
        let (b, t_in, c) = x.dims3()?;
        if c != self.content_in_proj.weight().dim(1)? {
            candle_core::bail!(
                "length regulator expects {c_in} input features (got {c})",
                c_in = self.content_in_proj.weight().dim(1)?
            );
        }
        if ylens.is_empty() || ylens.len() != b {
            candle_core::bail!("ylens must have one entry per batch element");
        }
        if t_in == 0 {
            candle_core::bail!("length regulator input must be non-empty");
        }
        let t_out = ylens.iter().copied().max().unwrap();
        // content_in_proj: (B, T_in, C)
        let h = self.content_in_proj.forward(x)?;
        // nearest-neighbor interpolation to t_out on the time axis.
        let index = Tensor::from_vec(nearest_indices(t_in, t_out), (t_out,), x.device())?;
        let mut h = h
            .transpose(1, 2)? // (B, C, T_in)
            .contiguous()? // index_select requires contiguous input
            .index_select(&index, 2)?; // (B, C, T_out)
        for (conv, norm) in &self.blocks {
            let y = conv.forward(&h)?;
            let y = norm.forward(&y)?;
            // nn.Mish(): x * tanh(softplus(x)) with softplus =
            // log1p(exp(x)) (torch's threshold-20 linear region only
            // differs where exp overflows f32, where tanh saturates to 1
            // and candle's formula degenerates to the same `x`).
            h = candle_nn::ops::mish(&y)?;
        }
        let h = self.out_proj_forward(&h)?.transpose(1, 2)?; // (B, T_out, C)
        let mask = sequence_mask_on(x.device(), ylens, t_out)?;
        h.broadcast_mul(&mask.unsqueeze(2)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    /// VarMap initializes convs/linears with kaiming noise; reset to zeros.
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

    fn tiny_cfg() -> LengthRegulatorConfig {
        LengthRegulatorConfig {
            channels: 8,
            is_discrete: false,
            in_channels: 16,
            content_codebook_size: 2048,
            sampling_ratios: vec![1, 1, 1, 1],
            vector_quantize: false,
            n_codebooks: 1,
            quantizer_dropout: 0.0,
            f0_condition: false,
            n_f0_bins: 512,
        }
    }

    #[test]
    fn nearest_indices_match_pytorch_nearest() {
        // floor(i * t_in / t_out): 5 -> 8 upsample.
        assert_eq!(nearest_indices(5, 8), vec![0, 0, 1, 1, 2, 3, 3, 4]);
        // 8 -> 5 downsample.
        assert_eq!(nearest_indices(8, 5), vec![0, 1, 3, 4, 6]);
        assert_eq!(nearest_indices(4, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn forward_shape_and_mask() {
        let device = Device::Cpu;
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = InterpolateRegulator::load(&vb, &tiny_cfg()).unwrap();
        zero_all(&mut varmap);
        // Non-zero output through the zero-weight stack: the final conv bias
        // makes the pre-mask output constant, so only the length mask matters.
        varmap
            .set_one(
                "model.12.bias",
                Tensor::ones(8, DType::F32, &device).unwrap(),
            )
            .unwrap();
        let x = Tensor::zeros((1, 5, 16), DType::F32, &device).unwrap();
        let out = model.forward(&x, &[8]).unwrap();
        assert_eq!(out.dims(), vec![1, 8, 8]);
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|&v| v == 1.0)
        );

        // Batch masking: second sample shorter -> trailing frames zeroed.
        let x = Tensor::zeros((2, 5, 16), DType::F32, &device).unwrap();
        let out = model.forward(&x, &[8, 3]).unwrap();
        assert_eq!(out.dims(), vec![2, 8, 8]);
        let rows = out.to_vec3::<f32>().unwrap();
        assert!(rows[0].iter().flatten().all(|&v| v == 1.0));
        assert!(
            rows[1]
                .iter()
                .flatten()
                .enumerate()
                .all(|(i, &v)| if i < 3 * 8 { v == 1.0 } else { v == 0.0 })
        );
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
        let model = InterpolateRegulator::load(
            &vb.pp("models").pp("length_regulator"),
            &cfg.length_regulator,
        )
        .unwrap();
        let x = Tensor::randn(0f32, 1f32, (1, 40, 1024), &device).unwrap();
        let out = model.forward(&x, &[69]).unwrap();
        assert_eq!(out.dims(), vec![1, 69, 512]);
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
    }

    /// Stage-by-stage mirror of [`InterpolateRegulator::forward`] against the
    /// Python per-submodule dump (`regulator_layers.json`, produced by
    /// `target/indextts-oracle/dump_regulator_layers.py` on the oracle
    /// input). Fails naming the first stage whose max abs diff exceeds the
    /// oracle tolerance, so regressions localize to one op.
    #[test]
    #[ignore = "requires INDEXTTS_CONVERTED_DIR, INDEXTTS_ORACLE_JSON, INDEXTTS_REGULATOR_LAYERS"]
    fn stage_layers_match_python_dump() {
        let (dir, oracle_path, layers_path) = match (
            std::env::var("INDEXTTS_CONVERTED_DIR").ok(),
            std::env::var("INDEXTTS_ORACLE_JSON").ok(),
            std::env::var("INDEXTTS_REGULATOR_LAYERS").ok(),
        ) {
            (Some(d), Some(o), Some(l)) => (d, o, l),
            _ => return,
        };
        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(
            std::path::Path::new(&dir).join("s2mel.safetensors"),
            &device,
        )
        .unwrap();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let cfg = crate::config::S2MelConfig::default();
        let model = InterpolateRegulator::load(
            &vb.pp("models").pp("length_regulator"),
            &cfg.length_regulator,
        )
        .unwrap();

        let oracle: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(oracle_path).unwrap()).unwrap();
        let layers: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(layers_path).unwrap()).unwrap();
        let layer = |name: &str| -> Vec<f32> {
            layers[name]["data"]
                .as_array()
                .unwrap_or_else(|| panic!("{name} missing from layers dump"))
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };

        let input = &oracle["codec"]["input"];
        let data: Vec<f32> = input["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let shape: Vec<usize> = input["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let x = Tensor::from_vec(data, shape, &device).unwrap();
        let ylens: Vec<usize> = oracle["regulator"]["ylens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let (_, t_in, _) = x.dims3().unwrap();
        let t_out = *ylens.iter().max().unwrap();

        let mut diffs: Vec<(String, f64)> = Vec::new();
        let mut check = |name: &str, got: &Tensor| {
            let want = layer(name);
            let got = got
                .flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(got.len(), want.len(), "{name}: length mismatch");
            let max_abs = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (*g - *w) as f64)
                .fold(0f64, f64::max);
            println!("{name}: max_abs {max_abs:.3e} over {} values", want.len());
            diffs.push((name.to_string(), max_abs));
        };

        let h = model.content_in_proj.forward(&x).unwrap();
        check("content_in_proj", &h);
        let index = Tensor::from_vec(nearest_indices(t_in, t_out), (t_out,), &device).unwrap();
        let mut h = h
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .index_select(&index, 2)
            .unwrap();
        for (i, (conv, norm)) in model.blocks.iter().enumerate() {
            let y = conv.forward(&h).unwrap();
            check(&format!("model.{}", 3 * i), &y);
            let y = norm.forward(&y).unwrap();
            check(&format!("model.{}", 3 * i + 1), &y);
            h = candle_nn::ops::mish(&y).unwrap();
            check(&format!("model.{}", 3 * i + 2), &h);
        }
        let h = model.out_proj_forward(&h).unwrap().transpose(1, 2).unwrap();
        check("model.12", &h);
        let h = h.transpose(1, 2).unwrap();
        let mask = crate::s2mel::sequence_mask_on(&device, &ylens, t_out).unwrap();
        let out = h.broadcast_mul(&mask.unsqueeze(2).unwrap()).unwrap();
        check("_final_output", &out);

        if let Some((name, d)) = diffs.iter().find(|(_, d)| *d > 5e-4) {
            panic!("first divergent stage: {name} max_abs {d:.3e}");
        }
    }
}
