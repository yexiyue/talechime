//! ConvNeXt Block
//!
//! A ConvNeXt-style block used in the audio decoder upsampling stages.
//! Consists of depthwise conv, LayerNorm, pointwise convs with GELU, and residual.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{LayerNorm, LayerNormConfig, Linear, VarBuilder, layer_norm, linear};

use super::CausalConv1d;

/// Causal depthwise convolution.
///
/// Candle's grouped `conv1d` decomposes into one convolution per channel on
/// Metal (upstream issue #3389; PR #3531 is CUDA-only and still open). This
/// path computes the same mathematical convolution while replacing those tiny
/// launches with broadcast multiply/add operations over the complete tensor.
struct DepthwiseCausalConv1d {
    stock: CausalConv1d,
    /// Broadcast-ready `(1, channels, kernel)` weights.
    taps: Option<Tensor>,
}

impl DepthwiseCausalConv1d {
    fn new(stock: CausalConv1d) -> Result<Self> {
        let eligible = stock.is_depthwise();
        let taps = eligible
            .then(|| {
                stock
                    .weight()
                    .reshape((1, stock.channels(), stock.kernel_size()))
            })
            .transpose()?;
        Ok(Self { stock, taps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, channels, length) = x.dims3()?;
        if let Some(taps) = &self.taps
            && x.device().is_metal()
            && channels == self.stock.channels()
        {
            return self.forward_fast(x, taps, batch, channels, length);
        }
        self.stock.forward(x)
    }

    fn forward_fast(
        &self,
        x: &Tensor,
        taps: &Tensor,
        batch: usize,
        channels: usize,
        length: usize,
    ) -> Result<Tensor> {
        let kernel_size = taps.dim(2)?;
        let causal_padding = self.stock.causal_padding();
        let padded = if causal_padding == 0 {
            x.clone()
        } else {
            x.pad_with_zeros(2, causal_padding, 0)?
        };

        let windows: Vec<Tensor> = (0..kernel_size)
            .map(|tap| padded.narrow(2, tap, length))
            .collect::<std::result::Result<_, _>>()?;
        let windows =
            Tensor::stack(&windows, 3)?.reshape((batch * channels, length, kernel_size))?;
        let weights = taps
            .reshape((channels, kernel_size))?
            .broadcast_as((batch, channels, kernel_size))?
            .reshape((batch * channels, kernel_size, 1))?;
        let mut output = windows
            .matmul(&weights)?
            .reshape((batch, channels, length))?;
        if let Some(bias) = self.stock.bias() {
            output = output.broadcast_add(&bias.reshape((1, channels, 1))?)?;
        }
        Ok(output)
    }
}

/// ConvNeXt Block
///
/// Architecture:
/// 1. Depthwise causal conv (groups=dim, kernel=7)
/// 2. LayerNorm
/// 3. Pointwise conv 1 (dim -> 4*dim)
/// 4. GELU activation
/// 5. Pointwise conv 2 (4*dim -> dim)
/// 6. Gamma scaling
/// 7. Residual connection
pub struct ConvNeXtBlock {
    /// Depthwise causal convolution
    dwconv: DepthwiseCausalConv1d,
    /// Layer normalization
    norm: LayerNorm,
    /// Pointwise conv 1 (expansion)
    pwconv1: Linear,
    /// Pointwise conv 2 (projection)
    pwconv2: Linear,
    /// Learnable scale parameter
    gamma: Tensor,
}

impl ConvNeXtBlock {
    /// Create a new ConvNeXt block.
    ///
    /// # Arguments
    /// * `dim` - Number of channels
    /// * `vb` - Variable builder for loading weights
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Depthwise causal conv: [dim, 1, 7] with groups=dim
        let dwconv =
            DepthwiseCausalConv1d::new(CausalConv1d::new(dim, dim, 7, 1, vb.pp("dwconv.conv"))?)?;

        // LayerNorm
        let norm_config = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let norm = layer_norm(dim, norm_config, vb.pp("norm"))?;

        // Pointwise convs (implemented as Linear)
        let pwconv1 = linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(4 * dim, dim, vb.pp("pwconv2"))?;

        // Gamma scale
        let gamma = vb.get((dim,), "gamma")?;

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    /// Create from raw weight tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        dwconv_weight: Tensor,
        dwconv_bias: Option<Tensor>,
        norm_weight: Tensor,
        norm_bias: Tensor,
        pwconv1_weight: Tensor,
        pwconv1_bias: Tensor,
        pwconv2_weight: Tensor,
        pwconv2_bias: Tensor,
        gamma: Tensor,
    ) -> Result<Self> {
        // Depthwise conv with groups=dim (each channel has its own kernel)
        let dim = dwconv_weight.dim(0)?;
        let dwconv = DepthwiseCausalConv1d::new(CausalConv1d::from_weights_grouped(
            dwconv_weight,
            dwconv_bias,
            1,   // dilation
            dim, // groups = dim for depthwise
        )?)?;

        // LayerNorm
        let norm = LayerNorm::new(norm_weight, norm_bias, 1e-6);

        // Pointwise convs
        let pwconv1 = Linear::new(pwconv1_weight, Some(pwconv1_bias));
        let pwconv2 = Linear::new(pwconv2_weight, Some(pwconv2_bias));

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    /// Forward pass.
    ///
    /// Input shape: [batch, channels, seq_len]
    /// Output shape: [batch, channels, seq_len]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        // Depthwise causal conv
        let hidden = self.dwconv.forward(x)?;

        // Transpose: [B, C, T] -> [B, T, C]
        let hidden = hidden.transpose(1, 2)?;

        // LayerNorm
        let hidden = self.norm.forward(&hidden)?;

        // Pointwise conv 1 (expansion)
        let hidden = self.pwconv1.forward(&hidden)?;

        // GELU activation (exact erf-based, matches PyTorch nn.GELU())
        let hidden = hidden.gelu_erf()?;

        // Pointwise conv 2 (projection)
        let hidden = self.pwconv2.forward(&hidden)?;

        // Gamma scaling
        let hidden = hidden.broadcast_mul(&self.gamma)?;

        // Transpose back: [B, T, C] -> [B, C, T]
        let hidden = hidden.transpose(1, 2)?;

        // Residual connection
        let out = (residual + hidden)?;

        Ok(out)
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        ConvNeXtBlock::forward(self, x).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn lcg_fill(seed: u32, n: usize) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((state >> 8) & 0xffff) as f32 / 32768.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn depthwise_causal_fast_path_matches_stock_conv() {
        check_depthwise_causal_fast_path(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn depthwise_causal_fast_path_matches_stock_conv_metal() {
        check_depthwise_causal_fast_path(&Device::new_metal(0).unwrap());
    }

    fn check_depthwise_causal_fast_path(device: &Device) {
        let cases = [(1usize, 32usize, 25usize), (2, 16, 7), (1, 8, 1)];
        for (batch, channels, length) in cases {
            let input = Tensor::from_vec(
                lcg_fill(0xA7 ^ channels as u32, batch * channels * length),
                (batch, channels, length),
                device,
            )
            .unwrap();
            let weight =
                Tensor::from_vec(lcg_fill(0x5EED, channels * 7), (channels, 1, 7), device).unwrap();
            let bias = Tensor::from_vec(lcg_fill(0xB1A5, channels), (channels,), device).unwrap();
            let stock =
                CausalConv1d::from_weights_grouped(weight, Some(bias), 1, channels).unwrap();
            let fast = DepthwiseCausalConv1d::new(stock).unwrap();
            let expected = fast.stock.forward(&input).unwrap();
            let actual = fast.forward(&input).unwrap();
            assert_eq!(actual.dims(), expected.dims());

            let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(actual, expected, "case {cases:?}");
        }
    }

    #[test]
    fn test_convnext_block_construction() {
        // VarBuilder from VarMap creates zero-initialized tensors on demand
        // so we test with from_weights instead of new()
        let device = Device::Cpu;
        let dim = 16;

        let dwconv_w = Tensor::randn(0.0f32, 0.1, (dim, 1, 7), &device).unwrap();
        let dwconv_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();
        let norm_w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let norm_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();
        let pwconv1_w = Tensor::randn(0.0f32, 0.1, (4 * dim, dim), &device).unwrap();
        let pwconv1_b = Tensor::zeros((4 * dim,), DType::F32, &device).unwrap();
        let pwconv2_w = Tensor::randn(0.0f32, 0.1, (dim, 4 * dim), &device).unwrap();
        let pwconv2_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();
        let gamma = Tensor::ones((dim,), DType::F32, &device).unwrap();

        let block = ConvNeXtBlock::from_weights(
            dwconv_w,
            Some(dwconv_b),
            norm_w,
            norm_b,
            pwconv1_w,
            pwconv1_b,
            pwconv2_w,
            pwconv2_b,
            gamma,
        );
        assert!(block.is_ok());
    }

    #[test]
    fn test_convnext_block_shape() {
        let device = Device::Cpu;
        let dim = 32;

        // Create weights manually
        // dwconv: [dim, 1, 7] for depthwise conv with groups=dim
        let dwconv_w = Tensor::randn(0.0f32, 0.1, (dim, 1, 7), &device).unwrap();
        let dwconv_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();

        let norm_w = Tensor::ones((dim,), DType::F32, &device).unwrap();
        let norm_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();

        let pwconv1_w = Tensor::randn(0.0f32, 0.1, (4 * dim, dim), &device).unwrap();
        let pwconv1_b = Tensor::zeros((4 * dim,), DType::F32, &device).unwrap();

        let pwconv2_w = Tensor::randn(0.0f32, 0.1, (dim, 4 * dim), &device).unwrap();
        let pwconv2_b = Tensor::zeros((dim,), DType::F32, &device).unwrap();

        let gamma = Tensor::ones((dim,), DType::F32, &device).unwrap();

        let block = ConvNeXtBlock::from_weights(
            dwconv_w,
            Some(dwconv_b),
            norm_w,
            norm_b,
            pwconv1_w,
            pwconv1_b,
            pwconv2_w,
            pwconv2_b,
            gamma,
        )
        .unwrap();

        let input = Tensor::randn(0.0f32, 1.0, (2, dim, 10), &device).unwrap();
        let output = block.forward(&input).unwrap();

        // Output should have same shape as input
        assert_eq!(output.dims(), &[2, dim, 10]);
    }
}
