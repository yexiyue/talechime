//! Causal Transposed 1D Convolution
//!
//! A ConvTranspose1d that maintains causality by trimming output.
//! Used for upsampling in the audio decoder.
//!
//! Trims `kernel_size - stride` from the right, preserving the left origin.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{ConvTranspose1d, ConvTranspose1dConfig, VarBuilder, conv_transpose1d};

/// Causal Transposed 1D Convolution
///
/// Applies ConvTranspose1d and trims the output to match the official model.
/// Right-only trimming preserves exact `input * stride` output length when
/// `kernel_size >= stride`, matching the tokenizer geometry.
pub struct CausalTransConv1d {
    conv: ConvTranspose1d,
    /// Number of samples to trim from the right of output
    right_trim: usize,
}

impl CausalTransConv1d {
    /// Create a new causal transposed conv1d layer.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolving kernel
    /// * `stride` - Stride of the convolution (upsampling factor)
    /// * `vb` - Variable builder for loading weights
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let config = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };

        let conv = conv_transpose1d(in_channels, out_channels, kernel_size, config, vb)?;

        // Match official Qwen3-TTS/Mimi implementation:
        // Only trim from the right side for exact input * stride output
        // This maintains causality while preserving length
        let right_trim = kernel_size.saturating_sub(stride);

        Ok(Self { conv, right_trim })
    }

    /// Create from raw weight and bias tensors.
    ///
    /// Weight should have shape [in_channels, out_channels, kernel_size].
    pub fn from_weights(weight: Tensor, bias: Option<Tensor>, stride: usize) -> Result<Self> {
        let kernel_size = weight.dim(2)?;

        let config = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };

        let conv = ConvTranspose1d::new(weight, bias, config);

        // Match official Qwen3-TTS/Mimi implementation:
        // Only trim from the right side for exact input * stride output
        // This maintains causality while preserving length
        let right_trim = kernel_size.saturating_sub(stride);

        Ok(Self { conv, right_trim })
    }

    /// Input history required to reproduce the requested output history.
    ///
    /// Right-only cropping leaves the origin unchanged. At a stride-aligned
    /// boundary, the earliest dependency is `ceil((context + kernel - stride) / stride)`.
    pub(super) fn input_context(&self, output_context: usize) -> usize {
        output_context
            .saturating_add(self.right_trim)
            .div_ceil(self.stride())
    }

    /// Forward pass with causal output trimming.
    ///
    /// Input shape: [batch, in_channels, seq_len]
    /// Output shape: [batch, out_channels, seq_len * stride]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply transposed convolution
        let out = self.conv.forward(x)?;

        // Trim output for causality
        let out_len = out.dim(2)?;
        if self.right_trim > 0 {
            let end = out_len.saturating_sub(self.right_trim);
            Ok(out.narrow(2, 0, end)?)
        } else {
            Ok(out)
        }
    }

    /// Get the stride (upsampling factor)
    pub fn stride(&self) -> usize {
        self.conv.config().stride
    }
}

impl Module for CausalTransConv1d {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        CausalTransConv1d::forward(self, x).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn right_trim_context_reproduces_full_output_at_stride_boundary() -> Result<()> {
        let device = Device::Cpu;
        let conv = CausalTransConv1d::from_weights(
            Tensor::ones((1, 1, 6), DType::F32, &device)?,
            None,
            3,
        )?;
        let input = Tensor::ones((1, 1, 12), DType::F32, &device)?;
        let full = conv.forward(&input)?;
        let context = conv.input_context(1);
        assert_eq!(context, 2);
        for keep in [context, context - 1] {
            let replay = conv.forward(&input.narrow(2, 8 - keep, keep + 4)?)?;
            let actual = replay
                .narrow(2, keep * 3 - 1, 13)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = full.narrow(2, 23, 13)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(actual == expected, keep == context);
        }
        let nonoverlapping = CausalTransConv1d::from_weights(
            Tensor::ones((1, 1, 2), DType::F32, &device)?,
            None,
            2,
        )?;
        assert_eq!(nonoverlapping.input_context(0), 0);
        assert_eq!(nonoverlapping.input_context(3), 2);
        Ok(())
    }

    #[test]
    fn test_causal_trans_conv_shape() {
        let device = Device::Cpu;

        // Create random weights: [in_channels, out_channels, kernel_size]
        // Note: ConvTranspose1d weight is [in, out, kernel], not [out, in, kernel]
        // kernel=4, stride=2
        let weight = Tensor::randn(0.0f32, 0.1, (64, 32, 4), &device).unwrap();
        let bias = Tensor::randn(0.0f32, 0.1, (32,), &device).unwrap();

        let conv = CausalTransConv1d::from_weights(weight, Some(bias), 2).unwrap();

        // Official behavior: trim only from right side
        assert_eq!(conv.right_trim, 2);

        // Input: [batch=1, channels=64, seq=10]
        let input = Tensor::randn(0.0f32, 1.0, (1, 64, 10), &device).unwrap();
        let output = conv.forward(&input).unwrap();

        // Output: raw = (10-1)*2 + 4 = 22, trim 2 from right -> 20 = 10 * 2
        // This gives exact input * stride upsampling
        assert_eq!(output.dims(), &[1, 32, 20]);
    }

    #[test]
    fn test_causal_trans_conv_stride_equals_kernel() {
        // When stride == kernel_size, no trimming needed
        let device = Device::Cpu;

        let weight = Tensor::randn(0.0f32, 0.1, (32, 32, 2), &device).unwrap();
        let bias = Tensor::zeros((32,), DType::F32, &device).unwrap();

        let conv = CausalTransConv1d::from_weights(weight, Some(bias), 2).unwrap();
        // No trimming needed when kernel == stride
        assert_eq!(conv.right_trim, 0);

        let input = Tensor::randn(0.0f32, 1.0, (1, 32, 5), &device).unwrap();
        let output = conv.forward(&input).unwrap();

        // 5 * 2 = 10 (exact upsampling when kernel == stride)
        assert_eq!(output.dims(), &[1, 32, 10]);
    }

    #[test]
    fn test_causal_trans_conv_various_strides() {
        // Test with kernel = 2 * stride (typical decoder block configuration)
        let device = Device::Cpu;

        // (kernel, stride) -> expected output for input_len=4
        // Official formula: output = input * stride (exact upsampling with right-only trim)
        let test_cases = [
            (16, 8, 32), // 4*8 = 32
            (10, 5, 20), // 4*5 = 20
            (8, 4, 16),  // 4*4 = 16
            (6, 3, 12),  // 4*3 = 12
        ];

        for (kernel_size, stride, expected_len) in test_cases {
            let weight = Tensor::randn(0.0f32, 0.1, (16, 8, kernel_size), &device).unwrap();
            let bias = Tensor::zeros((8,), DType::F32, &device).unwrap();

            let conv = CausalTransConv1d::from_weights(weight, Some(bias), stride).unwrap();

            // Verify trimming: only trim from right side
            let expected_right_trim = kernel_size - stride;
            assert_eq!(conv.right_trim, expected_right_trim);

            let input = Tensor::randn(0.0f32, 1.0, (1, 16, 4), &device).unwrap();
            let output = conv.forward(&input).unwrap();

            assert_eq!(
                output.dims(),
                &[1, 8, expected_len],
                "Failed for kernel={}, stride={}",
                kernel_size,
                stride
            );
        }
    }
}
