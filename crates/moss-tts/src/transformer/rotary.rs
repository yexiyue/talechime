//! One immutable RoPE table per forward, retaining the original arithmetic shapes.
use candle_core::{DType, Device, Result, Tensor};

pub(crate) struct RotaryTables {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryTables {
    pub(crate) fn new(
        length: usize,
        dim: usize,
        offset: usize,
        theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let frequencies: Vec<f32> = (0..dim / 2)
            .map(|i| theta.powf(-((2 * i) as f64) / dim as f64) as f32)
            .collect();
        let positions = Tensor::arange(offset as u32, (offset + length) as u32, device)?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?;
        let angles = positions.matmul(&Tensor::from_vec(frequencies, (1, dim / 2), device)?)?;
        Ok(Self {
            cos: angles.cos()?.to_dtype(dtype)?,
            sin: angles.sin()?.to_dtype(dtype)?,
        })
    }

    pub(crate) fn apply(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::rotary_emb::rope(&x.contiguous()?, &self.cos, &self.sin)
    }

    pub(crate) fn apply_interleaved(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &self.cos, &self.sin)
    }
}
