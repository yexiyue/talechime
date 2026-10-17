//! Activations shared by the flow estimator and the vocoder.

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// `x + sin^2(alpha * x) / alpha` with a per-channel `alpha`
/// (`cosyvoice.transformer.activation.Snake`, BigVGAN heritage).
#[derive(Debug, Clone)]
pub struct Snake {
    alpha: Tensor,
}

impl Snake {
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get(channels, "alpha")?;
        Ok(Self { alpha })
    }
}

impl Module for Snake {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T]; alpha aligns to the channel axis.
        let alpha = self
            .alpha
            .reshape((1, (), 1))?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;
        let no_div_by_zero = 1e-9;
        let sin = x.broadcast_mul(&alpha)?.sin()?;
        let inv = (alpha + no_div_by_zero)?.recip()?;
        x + sin.sqr()?.broadcast_mul(&inv)?
    }
}

/// ELU with the PyTorch default alpha=1.
#[derive(Debug, Clone, Default)]
pub struct Elu;

impl Module for Elu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.elu(1.0)
    }
}

/// Swish / SiLU.
#[derive(Debug, Clone, Default)]
pub struct Swish;

impl Module for Swish {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::silu(x)
    }
}

/// Leaky ReLU with an explicit slope: the vocoder mixes 0.1 inside the
/// upsampling loop with the 0.01 default at the post stage.
#[derive(Debug, Clone)]
pub struct LeakyReLU {
    slope: f64,
}

impl LeakyReLU {
    pub fn new(slope: f64) -> Self {
        Self { slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl Module for LeakyReLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::leaky_relu(x, self.slope)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.tanh()
    }
}

/// Mish as used by the DiT conv position embedding.
#[derive(Debug, Clone, Default)]
pub struct Mish;

impl Module for Mish {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::mish(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn snake_formula() -> Result<()> {
        let device = Device::Cpu;
        let alpha = Tensor::full(2.0f32, 3, &device)?;
        let snake = Snake { alpha };
        let x = Tensor::new(0.5f32, &device)?.broadcast_as((1, 3, 2))?;
        let y: Vec<f32> = snake.forward(&x)?.flatten_all()?.to_vec1()?;
        let expected = 0.5 + (2.0f32 * 0.5).sin().powi(2) / 2.0;
        assert!(y.iter().all(|&o| (o - expected).abs() < 1e-6));
        Ok(())
    }

    #[test]
    fn leaky_relu_default_matches_torch() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[-2.0f32, 0.0, 3.0], 3, &device)?;
        let y: Vec<f32> = LeakyReLU::default().forward(&x)?.to_vec1()?;
        assert_eq!(y, vec![-0.02, 0.0, 3.0]);
        Ok(())
    }
}
