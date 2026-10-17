//! GPT2 backbone (`indextts/gpt/transformers_gpt2.py`, old-style HF copy).
//!
//! Key porting points: `c_attn`/`mlp.c_fc`/`mlp.c_proj` use HF `Conv1D`
//! where the stored weight is `(in, out)` and the math is `x @ W + b` —
//! NOT a torch Linear layout. Attention is eager with a `tril` causal bias
//! and `masked_bias = -1e4`; the additive attention mask is built from the
//! caller-provided 1/0 mask. `wpe`/`wte` are absent from the checkpoint
//! (positions are added externally by the caller's embeds).
//!
//! Masking details that matter for parity with the eager Python path
//! (`GPT2Attention._attn`): scores are scaled by `1/sqrt(head_dim)`, then
//! causal-masked positions are *replaced* with `torch.finfo(f32).min`
//! (bottom-right aligned tril over `[key_len - query_len, key_len)` rows)
//! and the caller mask is added as `(1 - m) * finfo.min`. In candle both
//! masks are additive; the only divergence is that a position masked by
//! both becomes `-inf` instead of `2 * f32::MIN`, which softmax treats
//! identically.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Module, VarBuilder};

/// GPT2 `layer_norm_epsilon` default.
const LN_EPS: f64 = 1e-5;

#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_positions: usize,
    pub vocab_size: usize,
}

impl Gpt2Config {
    pub fn for_indextts(model_dim: usize, layers: usize, heads: usize, max_seq: usize) -> Self {
        Self {
            n_embd: model_dim,
            n_layer: layers,
            n_head: heads,
            n_positions: max_seq,
            vocab_size: 8194,
        }
    }
}

/// Per-layer key/value cache with a running sequence offset.
pub struct Gpt2Cache {
    /// Keys `(B, H, D, T)` are already transposed for attention;
    /// values remain `(B, H, T, D)`. Only new keys need a transpose.
    kvs: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
}

impl Gpt2Cache {
    pub fn new(layers: usize) -> Self {
        Self {
            kvs: vec![None; layers],
            len: 0,
        }
    }

    /// Number of cached positions (cache offset for the next forward).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `GPT2LMHeadModel._reorder_cache` / `GPT2InferenceModel._reorder_cache`:
    /// `past_state.index_select(0, beam_idx)` on every cached key and value,
    /// used by the beam-sample loop in [`crate::generation`] to match cache
    /// rows to the selected beams.
    pub(crate) fn reorder_beams(&mut self, beam_idx: &Tensor) -> Result<()> {
        for slot in self.kvs.iter_mut().filter_map(|kv| kv.as_mut()) {
            let (k, v) = (
                slot.0.index_select(&beam_idx.to_dtype(DType::U32)?, 0)?,
                slot.1.index_select(&beam_idx.to_dtype(DType::U32)?, 0)?,
            );
            *slot = (k, v);
        }
        Ok(())
    }
}

/// HF `Conv1D`: stored weight is `(in, out)` and the forward is `x @ W + b`.
/// The matmul runs on a flattened `(b * s, in)` view because candle matmul
/// does not mix a 3D lhs with a 2D rhs.
fn conv1d(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (batch, seq, dim) = x.dims3()?;
    x.reshape((batch * seq, dim))?
        .matmul(w)?
        .broadcast_add(b)?
        .reshape((batch, seq, w.dims()[1]))
}

/// Additive causal mask `(1, 1, q_len, total)` of `0` (keep) / `f32::MIN`
/// (drop), bottom-right aligned as in `GPT2Attention._attn`:
/// `bias[:, :, key_length - query_length : key_length, :key_length]`.
fn causal_additive(offset: usize, q_len: usize, total: usize, device: &Device) -> Result<Tensor> {
    let mut mask = Vec::with_capacity(q_len * total);
    for i in 0..q_len {
        for j in 0..total {
            mask.push(if j <= offset + i { 0.0 } else { f32::MIN });
        }
    }
    Tensor::from_vec(mask, (q_len, total), device)?
        .unsqueeze(0)?
        .unsqueeze(0)
}

/// Additive padding mask `(b, 1, 1, total)` built from a 1/0 caller mask:
/// `(1.0 - m) * torch.finfo(f32).min`, i.e. `m * (-min) + min` in one
/// affine kernel (ones stay exactly 0, zeros exactly `f32::MIN`).
fn padding_additive(mask: &Tensor, b: usize, total: usize) -> Result<Tensor> {
    let m = mask.to_dtype(DType::F32)?;
    if m.dims() != [b, total] {
        candle_core::bail!(
            "attention mask shape {:?} does not match (batch {b}, total {total})",
            m.dims()
        );
    }
    m.affine(-(f32::MIN as f64), f32::MIN as f64)?
        .reshape((b, 1, 1, total))
}

struct Block {
    ln_1: candle_nn::LayerNorm,
    /// `attn.c_attn` weight `(n_embd, 3 * n_embd)` in Conv1D layout.
    c_attn_w: Tensor,
    c_attn_b: Tensor,
    attn_c_proj_w: Tensor,
    attn_c_proj_b: Tensor,
    ln_2: candle_nn::LayerNorm,
    c_fc_w: Tensor,
    c_fc_b: Tensor,
    mlp_c_proj_w: Tensor,
    mlp_c_proj_b: Tensor,
}

pub struct Gpt2Model {
    blocks: Vec<Block>,
    ln_f: candle_nn::LayerNorm,
    n_head: usize,
    head_dim: usize,
    n_embd: usize,
    n_positions: usize,
}

impl Gpt2Model {
    /// `vb` is rooted at the `gpt.` prefix of gpt.pth's state dict.
    pub fn load(vb: &VarBuilder, cfg: &Gpt2Config, device: &Device) -> Result<Self> {
        let _ = device;
        if !cfg.n_embd.is_multiple_of(cfg.n_head) {
            candle_core::bail!(
                "n_embd {} must be divisible by n_head {}",
                cfg.n_embd,
                cfg.n_head
            );
        }
        let head_dim = cfg.n_embd / cfg.n_head;
        let inner = 4 * cfg.n_embd;
        let (d, t, i) = (cfg.n_embd, 3 * cfg.n_embd, inner);
        let mut blocks = Vec::with_capacity(cfg.n_layer);
        for layer in 0..cfg.n_layer {
            let vb_l = vb.pp(format!("h.{layer}"));
            blocks.push(Block {
                ln_1: candle_nn::layer_norm(cfg.n_embd, LN_EPS, vb_l.pp("ln_1"))?,
                c_attn_w: vb_l.get((d, t), "attn.c_attn.weight")?,
                c_attn_b: vb_l.get(t, "attn.c_attn.bias")?,
                attn_c_proj_w: vb_l.get((d, d), "attn.c_proj.weight")?,
                attn_c_proj_b: vb_l.get(d, "attn.c_proj.bias")?,
                ln_2: candle_nn::layer_norm(cfg.n_embd, LN_EPS, vb_l.pp("ln_2"))?,
                c_fc_w: vb_l.get((d, i), "mlp.c_fc.weight")?,
                c_fc_b: vb_l.get(i, "mlp.c_fc.bias")?,
                mlp_c_proj_w: vb_l.get((i, d), "mlp.c_proj.weight")?,
                mlp_c_proj_b: vb_l.get(d, "mlp.c_proj.bias")?,
            });
        }
        Ok(Self {
            blocks,
            ln_f: candle_nn::layer_norm(cfg.n_embd, LN_EPS, vb.pp("ln_f"))?,
            n_head: cfg.n_head,
            head_dim,
            n_embd: cfg.n_embd,
            n_positions: cfg.n_positions,
        })
    }

    /// Number of transformer blocks (for [`Gpt2Cache::new`]).
    pub fn layer_count(&self) -> usize {
        self.blocks.len()
    }

    /// `inputs_embeds`: `(B, S, n_embd)`; `attention_mask`: `(B, total_len)`
    /// 1/0 covering cache offset + S. With an empty cache and S > 1 the
    /// causal mask applies; with a warm cache (S == 1) each query attends to
    /// everything. Appends to `cache` and returns the last hidden states
    /// `(B, S, n_embd)`.
    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
        cache: &mut Gpt2Cache,
    ) -> Result<Tensor> {
        let (b, s, dim) = inputs_embeds.dims3()?;
        if dim != self.n_embd {
            candle_core::bail!("expected embedding dim {}, got {dim}", self.n_embd);
        }
        let device = inputs_embeds.device();
        let offset = cache.len;
        let total = offset + s;
        if total > self.n_positions {
            candle_core::bail!(
                "sequence length {total} exceeds n_positions {}",
                self.n_positions
            );
        }
        if cache.kvs.len() != self.blocks.len() {
            candle_core::bail!(
                "cache holds {} layers, model has {}",
                cache.kvs.len(),
                self.blocks.len()
            );
        }
        // Combined additive attention bias. With a warm cache (`s == 1`)
        // every causal position is already visible (`j <= offset` for all
        // `j < offset + 1`), so the causal half is all zeros and only the
        // caller padding contributes — the previous per-layer
        // `(scores + causal) + padding` composition adds those zeros and
        // `f32::MIN` exactly as `scores + padding` does (a position masked
        // by both overflows to `-inf` on either formulation, and softmax
        // treats a lone `f32::MIN` identically).
        let additive = match attention_mask {
            Some(m) => {
                let padding = padding_additive(m, b, total)?;
                if s > 1 {
                    Some(causal_additive(offset, s, total, device)?.broadcast_add(&padding)?)
                } else {
                    Some(padding)
                }
            }
            None if s > 1 => Some(causal_additive(offset, s, total, device)?),
            None => None,
        };
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut x = inputs_embeds.clone();
        for (index, blk) in self.blocks.iter().enumerate() {
            // Self-attention.
            let _g = crate::profile::scope(None, "g2.ln_qkv");
            let h = blk.ln_1.forward(&x)?;
            let qkv = conv1d(&h, &blk.c_attn_w, &blk.c_attn_b)?;
            drop(_g);
            let _g = crate::profile::scope(None, "g2.split");
            let (q, k, v) = split_qkv_heads(&qkv, b, s, self.n_head, self.head_dim)?;
            drop(_g);
            let _g = crate::profile::scope(None, "g2.kvcache");
            let k = k.transpose(2, 3)?.contiguous()?;
            let (k, v) = match &cache.kvs[index] {
                Some((old_k, old_v)) => {
                    (Tensor::cat(&[old_k, &k], 3)?, Tensor::cat(&[old_v, &v], 2)?)
                }
                None => (k, v),
            };
            cache.kvs[index] = Some((k.clone(), v.clone()));
            drop(_g);
            let _g = crate::profile::scope(None, "g2.attn");
            let scores = q.matmul(&k)?.affine(scale, 0.0)?;
            let scores = match &additive {
                Some(a) => scores.broadcast_add(a)?,
                None => scores,
            };
            let attn = candle_nn::ops::softmax_last_dim(&scores)?.matmul(&v.contiguous()?)?;
            let attn = attn
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, self.n_embd))?;
            drop(_g);
            let _g = crate::profile::scope(None, "g2.proj_mlp");
            x = (&x + conv1d(&attn, &blk.attn_c_proj_w, &blk.attn_c_proj_b)?)?;
            // MLP (`gelu_new` == candle's tanh-approximation `gelu`).
            let h = blk.ln_2.forward(&x)?;
            let h = conv1d(&h, &blk.c_fc_w, &blk.c_fc_b)?.gelu()?;
            x = (&x + conv1d(&h, &blk.mlp_c_proj_w, &blk.mlp_c_proj_b)?)?;
        }
        cache.len = total;
        self.ln_f.forward(&x)
    }
}

/// `(b, s, 3*dim)` fused qkv -> per-head `(b, head, s, head_dim)` q/k/v.
/// The 5-D reshape is a view on the contiguous projection output, so each
/// head split materializes in a single gather copy (the previous
/// narrow-then-contiguous-then-reshape-then-contiguous path copied twice).
fn split_qkv_heads(
    qkv: &Tensor,
    b: usize,
    s: usize,
    heads: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let fused = qkv.reshape((b, s, 3, heads, head_dim))?;
    let part = |i: usize| -> Result<Tensor> {
        fused
            .narrow(2, i, 1)?
            .squeeze(2)?
            .transpose(1, 2)?
            .contiguous()
    };
    Ok((part(0)?, part(1)?, part(2)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn dev() -> Device {
        Device::Cpu
    }

    #[test]
    fn causal_mask_is_bottom_right_aligned() {
        // 4x4 full-sequence case: row i may attend to columns j <= i.
        let m = causal_additive(0, 4, 4, &dev()).unwrap();
        assert_eq!(m.dims(), vec![1, 1, 4, 4]);
        let v = m
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (i, row) in v.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                let expected = if j <= i { 0.0 } else { f32::MIN };
                assert_eq!(x, expected, "mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn causal_mask_warm_cache_attends_everything() {
        // offset 3, one new query: allowed columns 0..=3 (all of them).
        let m = causal_additive(3, 1, 4, &dev()).unwrap();
        let v = m
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(v, vec![vec![0.0, 0.0, 0.0, 0.0]]);
        // offset 3, two new queries: first is still blocked at column 3.
        let m = causal_additive(2, 2, 4, &dev()).unwrap();
        let v = m
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(
            v,
            vec![vec![0.0, 0.0, 0.0, f32::MIN], vec![0.0, 0.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn padding_mask_additive_values() {
        // (1 - m) * finfo.min: ones stay 0, zeros become f32::MIN.
        let mask = Tensor::from_vec(vec![0.0f32, 1.0, 1.0, 1.0], (1, 4), &dev()).unwrap();
        let add = padding_additive(&mask, 1, 4).unwrap();
        assert_eq!(add.dims(), vec![1, 1, 1, 4]);
        let v = add.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![f32::MIN, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn conv1d_uses_in_out_layout() {
        // x (1, 2, 3) @ w (3, 2) + b (2): hand computed.
        let device = dev();
        let x =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 2, 3), &device).unwrap();
        let w = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![0.5f32, -0.5], 2, &device).unwrap();
        let y = conv1d(&x, &w, &b)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        // row [1,2,3] @ w = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        assert_eq!(y[0], vec![22.5, 27.5]);
        // row [4,5,6] @ w = [4+15+30, 8+20+36] = [49, 64]
        assert_eq!(y[1], vec![49.5, 63.5]);
    }

    #[test]
    fn load_rejects_transposed_conv1d_weight() {
        // A torch-Linear-layout c_attn weight (3*n_embd, n_embd) must fail
        // the Conv1D (in, out) shape check.
        let device = dev();
        let d = 4;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let ones = |n: usize| Tensor::ones((n,), DType::F32, &device).unwrap();
        tensors.insert("h.0.ln_1.weight".into(), ones(d));
        tensors.insert("h.0.ln_1.bias".into(), ones(d));
        // Wrong direction: (3d, d) instead of (d, 3d).
        tensors.insert(
            "h.0.attn.c_attn.weight".into(),
            Tensor::ones((3 * d, d), DType::F32, &device).unwrap(),
        );
        tensors.insert("h.0.attn.c_attn.bias".into(), ones(3 * d));
        tensors.insert(
            "h.0.attn.c_proj.weight".into(),
            Tensor::ones((d, d), DType::F32, &device).unwrap(),
        );
        tensors.insert("h.0.attn.c_proj.bias".into(), ones(d));
        tensors.insert("h.0.ln_2.weight".into(), ones(d));
        tensors.insert("h.0.ln_2.bias".into(), ones(d));
        tensors.insert(
            "h.0.mlp.c_fc.weight".into(),
            Tensor::ones((d, 4 * d), DType::F32, &device).unwrap(),
        );
        tensors.insert("h.0.mlp.c_fc.bias".into(), ones(4 * d));
        tensors.insert(
            "h.0.mlp.c_proj.weight".into(),
            Tensor::ones((4 * d, d), DType::F32, &device).unwrap(),
        );
        tensors.insert("h.0.mlp.c_proj.bias".into(), ones(d));
        tensors.insert("ln_f.weight".into(), ones(d));
        tensors.insert("ln_f.bias".into(), ones(d));
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let cfg = Gpt2Config::for_indextts(d, 1, 1, 16);
        assert!(Gpt2Model::load(&vb, &cfg, &device).is_err());
    }

    /// VarMap-backed model with deterministic pseudo-random weights.
    fn tiny_model(layers: usize, dim: usize, heads: usize, max_seq: usize) -> Gpt2Model {
        let device = dev();
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = Gpt2Config::for_indextts(dim, layers, heads, max_seq);
        let model = Gpt2Model::load(&vb, &cfg, &device).expect("load tiny model");
        // Fill every variable with a deterministic LCG so activations are
        // non-constant (VarMap initializes to zeros).
        let mut state: u32 = 0x2545_F491;
        let mut next = move || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((state >> 16) as f32 / 32768.0 - 0.5) * 0.1
        };
        let data = varmap.data().lock().unwrap();
        for var in data.values() {
            let n = var.shape().elem_count();
            let values: Vec<f32> = (0..n).map(|_| next()).collect();
            let shape = var.shape().clone();
            var.set(&Tensor::from_vec(values, shape, &device).unwrap())
                .expect("fill var");
        }
        model
    }

    #[test]
    fn tiny_model_forward_shapes_and_cache_equivalence() {
        let device = dev();
        let model = tiny_model(2, 16, 2, 32);
        let (b, s, d) = (2, 5, 16);
        let embeds = Tensor::randn(0f32, 1f32, (b, s, d), &device).unwrap();
        let mask = Tensor::ones((b, s), DType::F32, &device).unwrap();

        // Full forward, fresh cache.
        let mut cache_full = Gpt2Cache::new(model.layer_count());
        let full = model
            .forward(&embeds, Some(&mask), &mut cache_full)
            .expect("full forward");
        assert_eq!(full.dims(), vec![b, s, d]);
        assert_eq!(cache_full.len(), s);

        // Chunked forward (3 + 2) with a warm cache on the second call.
        let mut cache_chunk = Gpt2Cache::new(model.layer_count());
        let first = model
            .forward(
                &embeds.narrow(1, 0, 3).unwrap(),
                Some(&mask.narrow(1, 0, 3).unwrap()),
                &mut cache_chunk,
            )
            .expect("chunk 1");
        assert_eq!(first.dims(), vec![b, 3, d]);
        let second = model
            .forward(
                &embeds.narrow(1, 3, 2).unwrap(),
                Some(&mask),
                &mut cache_chunk,
            )
            .expect("chunk 2");
        assert_eq!(second.dims(), vec![b, 2, d]);

        // Both chunks match the full forward within float32 rounding.
        let full_v = full.to_vec3::<f32>().unwrap();
        for (chunk, start, len) in [(&first, 0, 3), (&second, 3, 2)] {
            let chunk_v = chunk.to_vec3::<f32>().unwrap();
            for (bi, (row_c, row_f)) in chunk_v.iter().zip(&full_v).enumerate() {
                for (si, (c, f)) in row_c.iter().zip(&row_f[start..start + len]).enumerate() {
                    for (k, (cv, fv)) in c.iter().zip(f).enumerate() {
                        assert!(
                            (cv - fv).abs() < 1e-3,
                            "row {bi} pos {si} ch {k}: {cv} vs {fv}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tiny_model_left_padded_mask_stays_finite() {
        let device = dev();
        let model = tiny_model(1, 16, 2, 32);
        let embeds = Tensor::randn(0f32, 1f32, (1, 4, 16), &device).unwrap();
        let mask = Tensor::from_vec(vec![0.0f32, 0.0, 1.0, 1.0], (1, 4), &device).unwrap();
        let mut cache = Gpt2Cache::new(model.layer_count());
        let out = model.forward(&embeds, Some(&mask), &mut cache).unwrap();
        let v = out.to_vec3::<f32>().unwrap();
        assert!(v.iter().flatten().flatten().all(|x| x.is_finite()));
    }

    #[test]
    fn cache_reorder_beams_permutes_rows() {
        let device = dev();
        let mut cache = Gpt2Cache::new(1);
        cache.kvs[0] = Some((
            Tensor::arange(0f32, 6f32, &device)
                .unwrap()
                .reshape((3, 2))
                .unwrap(),
            Tensor::arange(6f32, 12f32, &device)
                .unwrap()
                .reshape((3, 2))
                .unwrap(),
        ));
        let beam_idx = Tensor::from_vec(vec![2u32, 0, 2], 3, &device).unwrap();
        cache.reorder_beams(&beam_idx).unwrap();
        let (k, v) = cache.kvs[0].as_ref().unwrap();
        assert_eq!(k.to_vec2::<f32>().unwrap()[0], vec![4.0, 5.0]);
        assert_eq!(k.to_vec2::<f32>().unwrap()[2], vec![4.0, 5.0]);
        assert_eq!(v.to_vec2::<f32>().unwrap()[1], vec![6.0, 7.0]);
    }

    #[test]
    fn transposed_key_cache_matches_full_attention_after_beam_reorder() {
        let device = dev();
        let model = tiny_model(2, 8, 2, 32);
        let prompt = Tensor::randn(0f32, 1f32, (2, 3, 8), &device).unwrap();
        let mut cache = Gpt2Cache::new(model.layer_count());
        model.forward(&prompt, None, &mut cache).unwrap();
        let (k, v) = cache.kvs[0].as_ref().unwrap();
        assert_eq!(k.dims(), [2, 2, 4, 3]);
        assert_eq!(v.dims(), [2, 2, 3, 4]);
        let beams = Tensor::new(&[1u32, 0, 1], &device).unwrap();
        cache.reorder_beams(&beams).unwrap();
        let next = Tensor::randn(0f32, 1f32, (3, 1, 8), &device).unwrap();
        let actual = model.forward(&next, None, &mut cache).unwrap();
        let full = Tensor::cat(&[&prompt.index_select(&beams, 0).unwrap(), &next], 1).unwrap();
        let expected = model
            .forward(&full, None, &mut Gpt2Cache::new(model.layer_count()))
            .unwrap()
            .narrow(1, 3, 1)
            .unwrap();
        for (got, want) in actual
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .zip(expected.flatten_all().unwrap().to_vec1::<f32>().unwrap())
        {
            assert!((got - want).abs() < 1e-3, "{got} vs {want}");
        }
    }
}
