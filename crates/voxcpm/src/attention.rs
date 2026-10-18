//! MiniCPM4 reference attention; cache and RoPE ownership stay in Transformer.
use candle_core::{DType, Result, Tensor};

pub(crate) fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    offset: usize,
) -> Result<Tensor> {
    let (batch, heads, tokens, width) = q.dims4()?;
    let (kv_batch, kv_heads, length, kv_width) = k.dims4()?;
    if kv_heads == 0
        || !heads.is_multiple_of(kv_heads)
        || kv_batch != batch
        || kv_width != width
        || v.dims() != k.dims()
    {
        candle_core::bail!("incompatible grouped attention shapes");
    }
    let groups = heads / kv_heads;
    let repeat = |tensor: &Tensor| {
        tensor
            .unsqueeze(2)?
            .broadcast_as((batch, kv_heads, groups, length, width))?
            .reshape((batch, heads, length, width))
    };
    let k = repeat(k)?;
    let v = repeat(v)?;
    let mut scores = (q
        .contiguous()?
        .matmul(&k.transpose(2, 3)?.contiguous()?)?
        .to_dtype(DType::F32)?
        / (width as f64).sqrt())?;
    if causal && tokens > 1 {
        let mask: Vec<u8> = (0..tokens)
            .flat_map(|i| (0..length).map(move |j| u8::from(j > offset + i)))
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, tokens, length), q.device())?
            .broadcast_as(scores.shape())?;
        scores = mask.where_cond(
            &Tensor::full(f32::NEG_INFINITY, scores.shape(), q.device())?,
            &scores,
        )?;
    }
    candle_nn::ops::softmax_last_dim(&scores)?
        .to_dtype(q.dtype())?
        .matmul(&v.contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn head_mapping_and_cached_causal_mask() -> Result<()> {
        // Zero logits give a closed-form oracle: mean of visible values.
        // Distinct head/batch constants detect incorrect GQA ordering.
        for batch in [1, 2] {
            for (tokens, offset) in [(1, 0), (1, 5), (3, 0), (3, 5)] {
                let length = offset + tokens;
                let q = Tensor::zeros((batch, 16, tokens, 128), DType::F32, &Device::Cpu)?;
                let k = Tensor::zeros((batch, 2, length, 128), DType::F32, &Device::Cpu)?;
                let data: Vec<f32> = (0..batch)
                    .flat_map(|b| {
                        (0..2).flat_map(move |head| {
                            (0..length).flat_map(move |token| {
                                std::iter::repeat_n((b * 20 + head * 10 + token) as f32, 128)
                            })
                        })
                    })
                    .collect();
                let v = Tensor::from_vec(data, (batch, 2, length, 128), &Device::Cpu)?;
                for causal in [false, true] {
                    let output = attention(&q, &k, &v, causal, offset)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    for b in 0..batch {
                        for head in 0..16 {
                            for row in 0..tokens {
                                let last = if causal { offset + row } else { length - 1 };
                                let expected = (b * 20 + head / 8 * 10) as f32 + last as f32 / 2.0;
                                let start = ((b * 16 + head) * tokens + row) * 128;
                                for &value in &output[start..start + 128] {
                                    assert!((value - expected).abs() <= 1e-5);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
