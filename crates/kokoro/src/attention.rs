//! CPU SDPA's unnormalized softmax/value product and final row scaling.
//! Mirrors PyTorch 2.7.1 FlashAttentionKernel.cpp's single-key-block path;
//! Kokoro's context never exceeds its 512-key block.
use candle_core::{Result, Tensor};

pub(crate) fn forward(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let (batch, heads, length, width) = q.dims4()?;
    if length == 0 || length > 512 || q.dims() != k.dims() || q.dims() != v.dims() {
        candle_core::bail!("invalid Kokoro attention shape");
    }
    let block = if length >= 192 { 64 } else { 32 };
    let mut output = Vec::new();
    for start in (0..length).step_by(block) {
        let count = (length - start).min(block);
        let scores =
            (q.narrow(2, start, count)?.contiguous()?.matmul(&k.t()?)? / (width as f64).sqrt())?;
        let mut values = scores.flatten_all()?.to_vec1::<f32>()?;
        let mut inverse = Vec::with_capacity(batch * heads * count);
        for row in values.chunks_mut(length) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sums = [0.0f32; 4];
            let vector_length = length / 4 * 4;
            for (i, value) in row[..vector_length].iter_mut().enumerate() {
                *value = (*value - max).exp();
                sums[i % 4] += *value;
            }
            let mut sum = (sums[0] + sums[2]) + (sums[1] + sums[3]);
            for value in &mut row[vector_length..] {
                *value = (*value - max).exp();
                sum += *value;
            }
            inverse.push(1.0 / sum);
        }
        let values = Tensor::from_vec(values, (batch, heads, count, length), q.device())?;
        let inverse = Tensor::from_vec(inverse, (batch, heads, count, 1), q.device())?;
        output.push(values.matmul(v)?.broadcast_mul(&inverse)?);
    }
    Tensor::cat(&output, 2)
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_independent_default_pytorch_cpu_sdpa() -> candle_core::Result<()> {
        use candle_core::{Device, Tensor};
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/attention.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let shape: Vec<usize> = case["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let tensor = |name: &str| -> candle_core::Result<Tensor> {
                let values: Vec<f32> = case[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect();
                Tensor::from_vec(values, shape.clone(), &Device::Cpu)
            };
            let expected = tensor("out")?.flatten_all()?.to_vec1::<f32>()?;
            let actual = super::forward(&tensor("q")?, &tensor("k")?, &tensor("v")?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(max < 1e-6, "CPU SDPA {shape:?}: max error {max}");
        }
        Ok(())
    }
}
