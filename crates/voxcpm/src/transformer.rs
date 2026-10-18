//! MiniCPM4 shared by the global, residual and local networks.
use crate::attention::attention;
use crate::weights::{Linear, Weights};
use candle_core::{DType, Result, Tensor};

struct Layer {
    norm: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ffn_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
}
pub(crate) struct Transformer {
    layers: Vec<Layer>,
    norm: Tensor,
    rope: Option<Rope>,
    causal: bool,
    eps: f64,
}
pub(crate) struct Cache {
    kv: Vec<Option<(Tensor, Tensor)>>,
    pub position: usize,
    limit: usize,
}
impl Cache {
    pub fn new(layers: usize, limit: usize) -> Self {
        Self {
            kv: vec![None; layers],
            position: 0,
            limit,
        }
    }
}

fn norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    candle_nn::ops::rms_norm(&x.contiguous()?, w, eps as f32)
}
struct Rope {
    /// `cos` repeated as `[cos, cos]` so one multiply covers the rotated halves.
    cos: Tensor,
    /// `sin` split by the official signs as `[-sin, sin]`.
    sin: Tensor,
}
impl Rope {
    fn new(factors: &Tensor) -> Result<Self> {
        let freq: Vec<f32> = (0..64)
            .map(|i| 10000f64.powf(-((2 * i) as f64) / 128.0) as f32)
            .collect();
        let f = Tensor::from_vec(freq, (1, 64), factors.device())?
            .broadcast_div(&factors.to_dtype(DType::F32)?.reshape((1, 64))?)?;
        let positions = Tensor::arange(0u32, 4096u32, factors.device())?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?;
        let angles = positions.matmul(&f)?;
        let cos = angles.cos()?;
        let sin = angles.sin()?;
        Ok(Self {
            cos: Tensor::cat(&[&cos, &cos], 1)?,
            sin: Tensor::cat(&[sin.neg()?, sin.clone()], 1)?,
        })
    }
}
impl Transformer {
    pub fn load(
        w: &mut Weights,
        prefix: &str,
        count: usize,
        norm_name: &str,
        rope: Option<Tensor>,
        causal: bool,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(count);
        for i in 0..count {
            let p = format!("{prefix}blk.{i}");
            layers.push(Layer {
                norm: w.tensor(&format!("{p}.attn_norm.weight"))?,
                q: w.linear(&format!("{p}.attn_q"), false)?,
                k: w.linear(&format!("{p}.attn_k"), false)?,
                v: w.linear(&format!("{p}.attn_v"), false)?,
                o: w.linear(&format!("{p}.attn_output"), false)?,
                ffn_norm: w.tensor(&format!("{p}.ffn_norm.weight"))?,
                gate: w.linear(&format!("{p}.ffn_gate"), false)?,
                up: w.linear(&format!("{p}.ffn_up"), false)?,
                down: w.linear(&format!("{p}.ffn_down"), false)?,
            });
        }
        Ok(Self {
            layers,
            norm: w.tensor(norm_name)?,
            rope: rope.as_ref().map(Rope::new).transpose()?,
            causal,
            eps: 1e-5,
        })
    }
    pub fn cache(&self, limit: usize) -> Cache {
        Cache::new(self.layers.len(), limit)
    }
    pub fn forward(
        &self,
        input: &Tensor,
        mut cache: Option<&mut Cache>,
        cancel: &impl Fn() -> bool,
    ) -> Result<Tensor> {
        let (b, t, _) = input.dims3()?;
        let offset = cache.as_ref().map_or(0, |c| c.position);
        if cache.as_ref().is_some_and(|c| offset + t > c.limit) {
            candle_core::bail!("Transformer context limit exceeded");
        }
        let mut x = input.clone();
        for (index, l) in self.layers.iter().enumerate() {
            if cancel() {
                candle_core::bail!("generation cancelled");
            }
            let h = norm(&x, &l.norm, self.eps)?;
            let q = l.q.forward(&h)?.reshape((b, t, 16, 128))?.transpose(1, 2)?;
            let k = l.k.forward(&h)?.reshape((b, t, 2, 128))?.transpose(1, 2)?;
            let v = l.v.forward(&h)?.reshape((b, t, 2, 128))?.transpose(1, 2)?;
            let (q, k) = match &self.rope {
                Some(rope) => {
                    // One rotation for the stacked heads: same per-element arithmetic
                    // as rotating q and k separately, in fewer kernel launches.
                    let rotated = rotate(&Tensor::cat(&[&q, &k], 1)?, offset, rope)?;
                    (rotated.narrow(1, 0, 16)?, rotated.narrow(1, 16, 2)?)
                }
                None => (q, k),
            };
            let (k, v) = if let Some(c) = cache.as_mut() {
                let pair = match &c.kv[index] {
                    Some((oldk, oldv)) => {
                        (Tensor::cat(&[oldk, &k], 2)?, Tensor::cat(&[oldv, &v], 2)?)
                    }
                    None => (k, v),
                };
                c.kv[index] = Some(pair.clone());
                pair
            } else {
                (k, v)
            };
            let attended = attention(&q, &k, &v, self.causal, offset)?
                .transpose(1, 2)?
                .reshape((b, t, 2048))?;
            x = (x + l.o.forward(&attended)?)?;
            let h = norm(&x, &l.ffn_norm, self.eps)?;
            x = (x + l
                .down
                .forward(&(l.gate.forward(&h)?.silu()? * l.up.forward(&h)?)?)?)?;
        }
        if let Some(c) = cache {
            c.position += t;
        }
        norm(&x, &self.norm, self.eps)
    }
}

fn rotate(x: &Tensor, offset: usize, rope: &Rope) -> Result<Tensor> {
    let (_, _, t, d) = x.dims4()?;
    let cos = rope.cos.narrow(0, offset, t)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = rope.sin.narrow(0, offset, t)?.unsqueeze(0)?.unsqueeze(0)?;
    let f = x.to_dtype(DType::F32)?;
    let first = f.narrow(3, 0, d / 2)?;
    let second = f.narrow(3, d / 2, d / 2)?;
    // `f` already holds both halves, so the official `x * cos + rotate_half(x) * sin`
    // reduces to one multiply-add pair over the doubled tables.
    let swapped = Tensor::cat(&[&second, &first], 3)?;
    (f.broadcast_mul(&cos)? + swapped.broadcast_mul(&sin)?)?.to_dtype(x.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The fused rotation must reproduce `x*cos + rotate_half(x)*sin` bit for bit.
    #[test]
    fn fused_rotation_matches_per_head_formula() -> Result<()> {
        let device = Device::Cpu;
        let rope = Rope::new(&Tensor::from_vec(
            (0..64).map(|i| 1.0 + i as f32 * 0.05).collect::<Vec<_>>(),
            64,
            &device,
        )?)?;
        for dtype in [DType::F32, DType::F16] {
            let values: Vec<f32> = (0..18 * 3 * 128)
                .map(|i| ((i % 37) as f32 - 18.0) / 7.0)
                .collect();
            let x = Tensor::from_vec(values.clone(), (1, 18, 3, 128), &device)?.to_dtype(dtype)?;
            let fused = rotate(&x, 5, &rope)?;
            for (head, expected) in [
                (0usize, rotate_head_by_hand(&x.narrow(1, 0, 1)?, 5, &rope)?),
                (16, rotate_head_by_hand(&x.narrow(1, 16, 1)?, 5, &rope)?),
            ] {
                let actual = fused.narrow(1, head, 1)?;
                assert_eq!(
                    actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    expected
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    "dtype {dtype:?} head {head}"
                );
            }
        }
        Ok(())
    }

    fn rotate_head_by_hand(x: &Tensor, offset: usize, rope: &Rope) -> Result<Tensor> {
        let (_, _, t, d) = x.dims4()?;
        let cos = rope
            .cos
            .narrow(0, offset, t)?
            .narrow(1, 0, d / 2)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        // The stored table keeps the official sign split, so recover `sin` for the
        // first half before applying the unfused formula.
        let sin = rope
            .sin
            .narrow(0, offset, t)?
            .narrow(1, 0, d / 2)?
            .neg()?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let f = x.to_dtype(DType::F32)?;
        let first = f.narrow(3, 0, d / 2)?;
        let second = f.narrow(3, d / 2, d / 2)?;
        let left = (first.broadcast_mul(&cos)? - second.broadcast_mul(&sin)?)?;
        let right = (second.broadcast_mul(&cos)? + first.broadcast_mul(&sin)?)?;
        Tensor::cat(&[left, right], 3)?.to_dtype(x.dtype())
    }
}
