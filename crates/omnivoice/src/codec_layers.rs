//! GGUF-friendly codec weights: keep Linear/Embedding packed; dequant Conv only per forward.

use std::{path::Path, sync::Arc};

use candle_core::{
    DType, Device, Module, Result as CandleResult, Tensor,
    quantized::{GgmlDType, QMatMul, QTensor},
};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, Linear, VarBuilder,
};
use candle_transformers::quantized_var_builder::VarBuilder as QuantizedVarBuilder;

use crate::error::Result;

#[cfg(test)]
#[path = "codec_layers/cpu_pointwise.rs"]
mod cpu_pointwise;

#[cfg(test)]
#[path = "codec_layers/cpu_conv7.rs"]
mod cpu_conv7;

/// Lookup table that stays in GGUF storage and gathers rows on demand.
///
/// Float GGML blobs (F32/F16/BF16) are materialized as a dense tensor once at
/// load: candle's vulkan/wgpu quantized stems reject float GGML dtypes
/// (`quantized dtype X is not supported`) on the quantized embedding/gather
/// path, while dequantizing a float GGML tensor is a copy/cast, not a weight
/// expansion (F32 GGML storage IS raw f32 bytes). Packed quants (Q8_0, Q4_K, ...)
/// keep the native quantized gather.
#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    weight: Option<Arc<QTensor>>,
    dense: Option<Tensor>,
    activation_dtype: DType,
    device: Device,
}

impl QuantizedEmbedding {
    pub fn new(weight: Arc<QTensor>, activation_dtype: DType, device: Device) -> Result<Self> {
        if matches!(
            weight.dtype(),
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        ) {
            let dense = weight.dequantize(&device)?;
            Self::from_dense_tensor(dense, activation_dtype)
        } else {
            Ok(Self {
                weight: Some(weight),
                dense: None,
                activation_dtype,
                device,
            })
        }
    }

    fn from_dense_tensor(dense: Tensor, activation_dtype: DType) -> Result<Self> {
        let device = dense.device().clone();
        Ok(Self {
            weight: None,
            dense: Some(dense.to_dtype(activation_dtype)?),
            activation_dtype,
            device,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden = match (&self.weight, &self.dense) {
            (Some(q), _) => {
                let w = q.dequantize(input_ids.device())?;
                let ids = input_ids.flatten_all()?.to_dtype(DType::U32)?;
                let mut dims = input_ids.dims().to_vec();
                dims.push(w.dim(candle_core::D::Minus1)?);
                w.index_select(&ids, 0)?.reshape(dims)?
            }
            (_, Some(w)) => {
                // Same shape semantics as QTensor::embedding / candle_nn Embedding:
                // flatten ids, gather rows, reshape to (..., hidden).
                let ids = input_ids.flatten_all()?.to_dtype(DType::U32)?;
                let mut out_dims = input_ids.dims().to_vec();
                out_dims.push(w.dim(candle_core::D::Minus1)?);
                w.index_select(&ids, 0)?.reshape(out_dims)?
            }
            _ => unreachable!("QuantizedEmbedding holds exactly one weight variant"),
        };
        hidden.to_dtype(self.activation_dtype).map_err(Into::into)
    }

    /// Full codebook materialization for VQ distance search (encode path only).
    pub fn embeddings(&self) -> Result<Tensor> {
        match (&self.weight, &self.dense) {
            (Some(q), _) => Ok(q
                .dequantize(&self.device)?
                .to_dtype(self.activation_dtype)?),
            (_, Some(w)) => Ok(w.clone()),
            _ => unreachable!("QuantizedEmbedding holds exactly one weight variant"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CodecEmbedding {
    inner: CodecEmbeddingInner,
}

#[derive(Debug, Clone)]
enum CodecEmbeddingInner {
    Dense(Embedding),
    Quantized(QuantizedEmbedding),
}

impl CodecEmbedding {
    pub fn embeddings(&self) -> Result<Tensor> {
        match &self.inner {
            CodecEmbeddingInner::Dense(layer) => Ok(layer.embeddings().clone()),
            CodecEmbeddingInner::Quantized(layer) => layer.embeddings(),
        }
    }
}

impl Module for CodecEmbedding {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            CodecEmbeddingInner::Dense(layer) => layer.forward(xs),
            CodecEmbeddingInner::Quantized(layer) => layer.forward(xs).map_err(codec_candle_err),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CodecLinear {
    inner: CodecLinearInner,
}

#[derive(Debug, Clone)]
enum CodecLinearInner {
    Dense(Linear),
    Quantized {
        weight: QMatMul,
        bias: Option<Tensor>,
    },
}

impl Module for CodecLinear {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            CodecLinearInner::Dense(layer) => layer.forward(xs),
            CodecLinearInner::Quantized { weight, bias } => {
                let xs = xs.contiguous()?;
                let mut out = xs.apply(weight)?;
                if let Some(bias) = bias {
                    out = out.broadcast_add(bias)?;
                }
                Ok(out)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CodecConv1d {
    inner: CodecConv1dInner,
}

#[derive(Debug, Clone)]
enum CodecConv1dInner {
    Dense(Conv1d),
    Lazy(LazyConv1dWeights),
    WeightNormLazy(LazyWeightNormConv1d),
}

impl CodecConv1d {
    pub(crate) fn forward_decoder_conv7(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            CodecConv1dInner::Dense(layer) => forward_dense_decoder_conv7(layer, xs),
            _ => self.forward(xs),
        }
    }

    // Opt in only from the decoder: reference-audio encoding keeps its original path.
    pub(crate) fn forward_decoder_pointwise(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            CodecConv1dInner::Dense(layer) => forward_dense_decoder_pointwise(layer, xs),
            _ => self.forward(xs),
        }
    }
}

fn forward_dense_decoder_conv7(layer: &Conv1d, xs: &Tensor) -> CandleResult<Tensor> {
    let cfg = layer.config();
    let (output, input, kernel) = layer.weight().dims3()?;
    if !candle_core::utils::has_accelerate()
        || !xs.device().is_cpu()
        || !layer.weight().device().is_cpu()
        || xs.dtype() != DType::F32
        || layer.weight().dtype() != DType::F32
        || !xs.is_contiguous()
        || !layer.weight().is_contiguous()
        || input != output
        || !matches!(input, 32 | 64 | 128 | 256 | 512)
        || kernel != 7
        || cfg.stride != 1
        || cfg.padding != 3
        || cfg.dilation != 1
        || cfg.groups != 1
    {
        return layer.forward(xs);
    }
    let (batch, channels, len) = xs.dims3()?;
    if batch == 0 || channels != input || len < 8 || len % 8 != 0 {
        return layer.forward(xs);
    }
    // Preserve the [input_channel, tap] contraction order and produce NCL directly.
    let columns = xs
        .pad_with_zeros(2, 3, 3)?
        .unfold(2, 7, 1)?
        .permute((0, 1, 3, 2))?
        .contiguous()?
        .reshape((batch, input * 7, len))?;
    let weight = layer
        .weight()
        .reshape((1, output, input * 7))?
        .broadcast_as((batch, output, input * 7))?;
    let result = weight.matmul(&columns)?;
    match layer.bias() {
        Some(bias) => result.broadcast_add(&bias.reshape((1, output, 1))?),
        None => Ok(result),
    }
}

fn forward_dense_decoder_pointwise(layer: &Conv1d, xs: &Tensor) -> CandleResult<Tensor> {
    let config = layer.config();
    let (out_channels, in_channels, kernel) = layer.weight().dims3()?;
    // Keep CPU BLAS geometry to the aligned square projections used by this decoder.
    let cpu_pointwise = xs.device().is_cpu()
        && candle_core::utils::has_accelerate()
        && layer.weight().device().is_cpu()
        && layer.weight().dtype() == DType::F32
        && out_channels == in_channels
        && matches!(in_channels, 32 | 64 | 128 | 256 | 512);
    if !(xs.device().is_metal() || cpu_pointwise)
        || xs.dtype() != DType::F32
        || kernel != 1
        || config.stride != 1
        || config.padding != 0
        || config.dilation != 1
        || config.groups != 1
        || !xs.is_contiguous()
        || !layer.weight().is_contiguous()
    {
        return layer.forward(xs);
    }
    let (batch, channels, len) = xs.dims3()?;
    if batch == 0
        || len == 0
        || (cpu_pointwise && (channels != in_channels || len < 8 || len % 8 != 0))
    {
        return layer.forward(xs);
    }
    // NCL output directly avoids both im2col and the output transpose copy.
    let weight = layer.weight().squeeze(2)?.unsqueeze(0)?.broadcast_as((
        batch,
        out_channels,
        in_channels,
    ))?;
    let output = weight.matmul(xs)?;
    match layer.bias() {
        Some(bias) => output.broadcast_add(&bias.reshape((1, out_channels, 1))?),
        None => Ok(output),
    }
}

#[derive(Debug, Clone)]
struct LazyConv1dWeights {
    weight: Arc<QTensor>,
    bias: Option<Tensor>,
    config: Conv1dConfig,
    activation_dtype: DType,
    device: Device,
}

#[derive(Debug, Clone)]
struct LazyWeightNormConv1d {
    g: Arc<QTensor>,
    v: Arc<QTensor>,
    bias: Option<Tensor>,
    config: Conv1dConfig,
    activation_dtype: DType,
    device: Device,
}

impl Module for CodecConv1d {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            CodecConv1dInner::Dense(layer) => layer.forward(xs),
            CodecConv1dInner::Lazy(lazy) => lazy.forward(xs).map_err(codec_candle_err),
            CodecConv1dInner::WeightNormLazy(lazy) => lazy.forward(xs).map_err(codec_candle_err),
        }
    }
}

impl LazyConv1dWeights {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let weight = self
            .weight
            .dequantize(&self.device)?
            .to_dtype(self.activation_dtype)?;
        Conv1d::new(weight, self.bias.clone(), self.config)
            .forward(xs)
            .map_err(Into::into)
    }
}

impl LazyWeightNormConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = self.activation_dtype;
        let g = self.g.dequantize(&self.device)?.to_dtype(dtype)?;
        let v = self.v.dequantize(&self.device)?.to_dtype(dtype)?;
        let norm = v.sqr()?.sum_keepdim((0, 1))?.sqrt()?;
        let scale = g.broadcast_div(&norm)?;
        let weight = v.broadcast_mul(&scale.broadcast_as(v.shape().dims())?)?;
        Conv1d::new(weight, self.bias.clone(), self.config)
            .forward(xs)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
pub struct CodecConvTranspose1d {
    inner: CodecConvTranspose1dInner,
}

#[derive(Debug, Clone)]
enum CodecConvTranspose1dInner {
    Dense(ConvTranspose1d),
    Lazy(LazyConvTranspose1dWeights),
}

#[derive(Debug, Clone)]
struct LazyConvTranspose1dWeights {
    weight: Arc<QTensor>,
    bias: Option<Tensor>,
    config: ConvTranspose1dConfig,
    activation_dtype: DType,
    device: Device,
}

impl Module for CodecConvTranspose1d {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            CodecConvTranspose1dInner::Dense(layer) => forward_dense_conv_transpose1d(layer, xs),
            CodecConvTranspose1dInner::Lazy(lazy) => lazy.forward(xs).map_err(codec_candle_err),
        }
    }
}

fn forward_dense_conv_transpose1d(layer: &ConvTranspose1d, xs: &Tensor) -> CandleResult<Tensor> {
    let config = layer.config();
    if !xs.device().is_metal()
        || xs.dtype() != DType::F32
        || !xs.is_contiguous()
        || !layer.weight().is_contiguous()
        || config.groups != 1
        || config.dilation != 1
        || config.padding == 0
        || config.output_padding > config.padding
    {
        return layer.forward(xs);
    }

    // Nonzero padding misses Candle's GEMM + col2im kernel. Compute the full
    // convolution, then crop; output_padding reduces the right margin.
    let full = xs.conv_transpose1d(layer.weight(), 0, 0, config.stride, 1, 1)?;
    let right_crop = config.padding - config.output_padding;
    let output_len = full
        .dim(2)?
        .checked_sub(config.padding)
        .and_then(|len| len.checked_sub(right_crop))
        .ok_or_else(|| {
            candle_core::Error::Msg("transposed convolution crop exceeds output".into())
        })?;
    let cropped = full.narrow(2, config.padding, output_len)?;
    match layer.bias() {
        None => Ok(cropped),
        Some(bias) => cropped.broadcast_add(&bias.reshape((1, bias.elem_count(), 1))?),
    }
}

impl LazyConvTranspose1dWeights {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let weight = self
            .weight
            .dequantize(&self.device)?
            .to_dtype(self.activation_dtype)?;
        ConvTranspose1d::new(weight, self.bias.clone(), self.config)
            .forward(xs)
            .map_err(Into::into)
    }
}

fn codec_candle_err(error: crate::error::OmniVoiceError) -> candle_core::Error {
    candle_core::Error::Msg(error.to_string())
}

#[derive(Clone)]
pub struct CodecWeightSource {
    inner: CodecWeightSourceInner,
}

#[derive(Clone)]
enum CodecWeightSourceInner {
    Dense(VarBuilder<'static>),
    Quantized {
        qvb: QuantizedVarBuilder,
        activation_dtype: DType,
        device: Device,
    },
}

impl CodecWeightSource {
    pub fn from_weight_path(path: &Path, activation_dtype: DType, device: &Device) -> Result<Self> {
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            let qvb = QuantizedVarBuilder::from_gguf(path, device)?;
            Ok(Self {
                inner: CodecWeightSourceInner::Quantized {
                    qvb,
                    activation_dtype,
                    device: device.clone(),
                },
            })
        } else {
            let paths = [path];
            // SAFETY: read-only mmap of immutable weight files for process lifetime.
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&paths, activation_dtype, device)? };
            Ok(Self {
                inner: CodecWeightSourceInner::Dense(vb),
            })
        }
    }

    pub fn pp(&self, suffix: impl ToString) -> Self {
        let suffix = suffix.to_string();
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => Self {
                inner: CodecWeightSourceInner::Dense(vb.pp(suffix.clone())),
            },
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => Self {
                inner: CodecWeightSourceInner::Quantized {
                    qvb: qvb.pp(suffix),
                    activation_dtype: *activation_dtype,
                    device: device.clone(),
                },
            },
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self.inner, CodecWeightSourceInner::Quantized { .. })
    }

    pub fn get(&self, shape: impl Into<candle_core::Shape>, name: &str) -> Result<Tensor> {
        let shape = shape.into();
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => vb.get(shape, name).map_err(Into::into),
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => qvb
                .get(shape, name)?
                .dequantize(device)?
                .to_dtype(*activation_dtype)
                .map_err(Into::into),
        }
    }

    pub fn load_linear(
        &self,
        in_dim: usize,
        out_dim: usize,
        with_bias: bool,
    ) -> Result<CodecLinear> {
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                let weight = vb.get((out_dim, in_dim), "weight")?;
                let bias = if with_bias {
                    Some(vb.get(out_dim, "bias")?)
                } else {
                    None
                };
                Ok(CodecLinear {
                    inner: CodecLinearInner::Dense(Linear::new(weight, bias)),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                let weight = keep_quantized_matmul(qvb.get((out_dim, in_dim), "weight")?)?;
                let bias = if with_bias {
                    Some(
                        qvb.get(out_dim, "bias")?
                            .dequantize(device)?
                            .to_dtype(*activation_dtype)?,
                    )
                } else {
                    None
                };
                Ok(CodecLinear {
                    inner: CodecLinearInner::Quantized { weight, bias },
                })
            }
        }
    }

    pub fn load_embedding(&self, vocab_size: usize, hidden_size: usize) -> Result<CodecEmbedding> {
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => Ok(CodecEmbedding {
                inner: CodecEmbeddingInner::Dense(Embedding::new(
                    vb.get((vocab_size, hidden_size), "embed")?,
                    hidden_size,
                )),
            }),
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => Ok(CodecEmbedding {
                inner: CodecEmbeddingInner::Quantized(QuantizedEmbedding::new(
                    qvb.get((vocab_size, hidden_size), "embed")?,
                    *activation_dtype,
                    device.clone(),
                )?),
            }),
        }
    }

    pub fn load_embedding_at(
        &self,
        vocab_size: usize,
        hidden_size: usize,
        name: &str,
    ) -> Result<CodecEmbedding> {
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => Ok(CodecEmbedding {
                inner: CodecEmbeddingInner::Dense(Embedding::new(
                    vb.get((vocab_size, hidden_size), name)?,
                    hidden_size,
                )),
            }),
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => Ok(CodecEmbedding {
                inner: CodecEmbeddingInner::Quantized(QuantizedEmbedding::new(
                    qvb.get((vocab_size, hidden_size), name)?,
                    *activation_dtype,
                    device.clone(),
                )?),
            }),
        }
    }

    pub fn load_linear_named(
        &self,
        in_dim: usize,
        out_dim: usize,
        with_bias: bool,
        weight_name: &str,
        bias_name: &str,
    ) -> Result<CodecLinear> {
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                let weight = vb.get((out_dim, in_dim), weight_name)?;
                let bias = if with_bias {
                    Some(vb.get(out_dim, bias_name)?)
                } else {
                    None
                };
                Ok(CodecLinear {
                    inner: CodecLinearInner::Dense(Linear::new(weight, bias)),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                let weight = keep_quantized_matmul(qvb.get((out_dim, in_dim), weight_name)?)?;
                let bias = if with_bias {
                    Some(
                        qvb.get(out_dim, bias_name)?
                            .dequantize(device)?
                            .to_dtype(*activation_dtype)?,
                    )
                } else {
                    None
                };
                Ok(CodecLinear {
                    inner: CodecLinearInner::Quantized { weight, bias },
                })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_conv1d(
        &self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        with_bias: bool,
        groups: usize,
    ) -> Result<CodecConv1d> {
        let config = Conv1dConfig {
            stride,
            padding,
            dilation,
            groups,
            ..Default::default()
        };
        let in_per_group = in_channels / groups;
        let weight_shape = (out_channels, in_per_group, kernel_size);
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                let weight = vb.get(weight_shape, "weight")?;
                let bias = if with_bias {
                    Some(vb.get(out_channels, "bias")?)
                } else {
                    None
                };
                Ok(CodecConv1d {
                    inner: CodecConv1dInner::Dense(Conv1d::new(weight, bias, config)),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                let weight = qvb.get(weight_shape, "weight")?;
                let bias = if with_bias {
                    Some(
                        qvb.get(out_channels, "bias")?
                            .dequantize(device)?
                            .to_dtype(*activation_dtype)?,
                    )
                } else {
                    None
                };
                Ok(CodecConv1d {
                    inner: CodecConv1dInner::Lazy(LazyConv1dWeights {
                        weight,
                        bias,
                        config,
                        activation_dtype: *activation_dtype,
                        device: device.clone(),
                    }),
                })
            }
        }
    }

    pub fn load_conv1d_flexible(
        &self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        padding: usize,
    ) -> Result<CodecConv1d> {
        let config = Conv1dConfig {
            dilation,
            padding,
            ..Default::default()
        };
        let expected_in = in_channels / config.groups;
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                let weight = match vb.get((out_channels, expected_in, kernel_size), "weight") {
                    Ok(weight) => weight,
                    Err(_) if out_channels == 1 => vb
                        .get((expected_in, kernel_size), "weight")?
                        .reshape((out_channels, expected_in, kernel_size))?,
                    Err(error) => return Err(error.into()),
                };
                let bias = vb.get(out_channels, "bias")?;
                Ok(CodecConv1d {
                    inner: CodecConv1dInner::Dense(Conv1d::new(weight, Some(bias), config)),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                if let Ok(weight) = qvb.get((out_channels, expected_in, kernel_size), "weight") {
                    let bias = Some(
                        qvb.get(out_channels, "bias")?
                            .dequantize(device)?
                            .to_dtype(*activation_dtype)?,
                    );
                    return Ok(CodecConv1d {
                        inner: CodecConv1dInner::Lazy(LazyConv1dWeights {
                            weight,
                            bias,
                            config,
                            activation_dtype: *activation_dtype,
                            device: device.clone(),
                        }),
                    });
                }
                if out_channels == 1 {
                    let weight = qvb
                        .get((expected_in, kernel_size), "weight")?
                        .dequantize(device)?
                        .to_dtype(*activation_dtype)?
                        .reshape((out_channels, expected_in, kernel_size))?;
                    let bias = vb_get_bias(qvb, out_channels, device, *activation_dtype)?;
                    return Ok(CodecConv1d {
                        inner: CodecConv1dInner::Dense(Conv1d::new(weight, bias, config)),
                    });
                }
                Err(candle_core::Error::msg("missing conv1d weight").into())
            }
        }
    }

    pub fn load_weight_norm_conv1d(
        &self,
        out_channels: usize,
        in_channels_per_group: usize,
        kernel_size: usize,
        padding: usize,
        groups: usize,
    ) -> Result<CodecConv1d> {
        let config = Conv1dConfig {
            padding,
            groups,
            ..Default::default()
        };
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                if let Ok(weight) =
                    vb.get((out_channels, in_channels_per_group, kernel_size), "weight")
                {
                    return Ok(CodecConv1d {
                        inner: CodecConv1dInner::Dense(Conv1d::new(
                            weight,
                            Some(vb.get(out_channels, "bias")?),
                            config,
                        )),
                    });
                }
                let g = vb.get((1, 1, kernel_size), "parametrizations.weight.original0")?;
                let v = vb.get(
                    (out_channels, in_channels_per_group, kernel_size),
                    "parametrizations.weight.original1",
                )?;
                let norm = v.sqr()?.sum_keepdim((0, 1))?.sqrt()?;
                let scale = g.broadcast_div(&norm)?;
                let weight = v.broadcast_mul(&scale.broadcast_as(v.shape().dims())?)?;
                Ok(CodecConv1d {
                    inner: CodecConv1dInner::Dense(Conv1d::new(
                        weight,
                        Some(vb.get(out_channels, "bias")?),
                        config,
                    )),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                if let Ok(weight) =
                    qvb.get((out_channels, in_channels_per_group, kernel_size), "weight")
                {
                    let bias = Some(
                        qvb.get(out_channels, "bias")?
                            .dequantize(device)?
                            .to_dtype(*activation_dtype)?,
                    );
                    return Ok(CodecConv1d {
                        inner: CodecConv1dInner::Lazy(LazyConv1dWeights {
                            weight,
                            bias,
                            config,
                            activation_dtype: *activation_dtype,
                            device: device.clone(),
                        }),
                    });
                }
                let g = qvb.get((1, 1, kernel_size), "parametrizations.weight.original0")?;
                let v = qvb.get(
                    (out_channels, in_channels_per_group, kernel_size),
                    "parametrizations.weight.original1",
                )?;
                let bias = Some(
                    qvb.get(out_channels, "bias")?
                        .dequantize(device)?
                        .to_dtype(*activation_dtype)?,
                );
                Ok(CodecConv1d {
                    inner: CodecConv1dInner::WeightNormLazy(LazyWeightNormConv1d {
                        g,
                        v,
                        bias,
                        config,
                        activation_dtype: *activation_dtype,
                        device: device.clone(),
                    }),
                })
            }
        }
    }

    pub fn load_conv_transpose1d(
        &self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> Result<CodecConvTranspose1d> {
        let config = ConvTranspose1dConfig {
            stride,
            padding,
            output_padding,
            ..Default::default()
        };
        match &self.inner {
            CodecWeightSourceInner::Dense(vb) => {
                let weight = vb.get(
                    (in_channels, out_channels / config.groups, kernel_size),
                    "weight",
                )?;
                let bias = vb.get(out_channels, "bias")?;
                Ok(CodecConvTranspose1d {
                    inner: CodecConvTranspose1dInner::Dense(ConvTranspose1d::new(
                        weight,
                        Some(bias),
                        config,
                    )),
                })
            }
            CodecWeightSourceInner::Quantized {
                qvb,
                activation_dtype,
                device,
            } => {
                let weight = qvb.get(
                    (in_channels, out_channels / config.groups, kernel_size),
                    "weight",
                )?;
                let bias = Some(
                    qvb.get(out_channels, "bias")?
                        .dequantize(device)?
                        .to_dtype(*activation_dtype)?,
                );
                Ok(CodecConvTranspose1d {
                    inner: CodecConvTranspose1dInner::Lazy(LazyConvTranspose1dWeights {
                        weight,
                        bias,
                        config,
                        activation_dtype: *activation_dtype,
                        device: device.clone(),
                    }),
                })
            }
        }
    }
}

fn keep_quantized_matmul(weight: Arc<QTensor>) -> Result<QMatMul> {
    // Use `dequantize(...)` + `to_dtype(F16)` rather than `dequantize_f16(...)`.
    // On CUDA, `QTensor::dequantize_f16` dispatches to a specialized kernel path
    // that only supports packed-quant source dtypes and bails with
    // "unsupported dtype for dequantize F32" for float sources
    // (candle quantized/cuda.rs dequantize_f16 has no float-source branch).
    // Plain `dequantize` handles F32/F16/BF16 sources on every backend:
    // cuda via the CPU-fallback path (quantized/cuda.rs:679-681), wgpu via the
    // raw-float device path (quantized/mod.rs:327-338), vulkan via the device
    // copy/cast path (vulkan_backend.rs:4324-4338), cpu natively.
    let dtype = weight.dtype();
    match dtype {
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
            let device = weight.device();
            let w = weight.dequantize(&device)?.to_dtype(DType::F16)?;
            Ok(QMatMul::TensorF16(w))
        }
        _ => Ok(QMatMul::QTensor(weight)),
    }
}

fn vb_get_bias(
    qvb: &QuantizedVarBuilder,
    out_channels: usize,
    device: &Device,
    activation_dtype: DType,
) -> Result<Option<Tensor>> {
    Ok(Some(
        qvb.get(out_channels, "bias")?
            .dequantize(device)?
            .to_dtype(activation_dtype)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pointwise_cases(device: &Device) -> CandleResult<()> {
        let bits = |t: Tensor| -> CandleResult<Vec<u32>> {
            Ok(t.to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        for (input, output) in [(3, 5), (8, 8), (16, 16), (32, 32), (32, 64), (64, 32)] {
            let weight = Tensor::rand(-0.1f32, 0.1f32, (output, input, 1), device)?;
            for bias in [None, Some(Tensor::rand(-0.1f32, 0.1f32, output, device)?)] {
                let conv = Conv1d::new(weight.clone(), bias, Conv1dConfig::default());
                let wrapper = CodecConv1d {
                    inner: CodecConv1dInner::Dense(conv.clone()),
                };
                for batch in [1, 2] {
                    for frames in [1, 7, 130, 179] {
                        for strided in [false, true] {
                            let xs = if strided {
                                Tensor::rand(-0.1f32, 0.1f32, (batch, frames, input), device)?
                                    .transpose(1, 2)?
                            } else {
                                Tensor::rand(-0.1f32, 0.1f32, (batch, input, frames), device)?
                            };
                            let expected = bits(conv.forward(&xs)?)?;
                            assert_eq!(bits(wrapper.forward(&xs)?)?, expected);
                            for _ in 0..2 {
                                assert_eq!(
                                    bits(wrapper.forward_decoder_pointwise(&xs)?)?,
                                    expected,
                                    "input={input} output={output} batch={batch} frames={frames} strided={strided}"
                                );
                            }
                        }
                    }
                }
            }
        }
        for (kernel, config) in [
            (
                7,
                Conv1dConfig {
                    padding: 3,
                    ..Default::default()
                },
            ),
            (
                7,
                Conv1dConfig {
                    padding: 9,
                    dilation: 3,
                    ..Default::default()
                },
            ),
            (
                1,
                Conv1dConfig {
                    stride: 2,
                    ..Default::default()
                },
            ),
            (
                1,
                Conv1dConfig {
                    padding: 1,
                    ..Default::default()
                },
            ),
            (
                1,
                Conv1dConfig {
                    groups: 2,
                    ..Default::default()
                },
            ),
        ] {
            let weight = Tensor::rand(-0.1f32, 0.1f32, (4, 4 / config.groups, kernel), device)?;
            let conv = Conv1d::new(weight, None, config);
            let xs = Tensor::rand(-0.1f32, 0.1f32, (2, 4, 23), device)?;
            assert_eq!(
                bits(forward_dense_decoder_pointwise(&conv, &xs)?)?,
                bits(conv.forward(&xs)?)?
            );
        }
        let conv = Conv1d::new(
            Tensor::ones((4, 4, 1), DType::F16, device)?,
            None,
            Conv1dConfig::default(),
        );
        let xs = Tensor::ones((1, 4, 7), DType::F16, device)?;
        if device.is_cpu() && candle_core::utils::has_accelerate() {
            let expected = conv.forward(&xs).unwrap_err().to_string();
            assert!(expected.contains("does not support f16 matmul"));
            assert_eq!(
                forward_dense_decoder_pointwise(&conv, &xs)
                    .unwrap_err()
                    .to_string(),
                expected
            );
        } else {
            assert_eq!(
                bits(forward_dense_decoder_pointwise(&conv, &xs)?)?,
                bits(conv.forward(&xs)?)?
            );
        }
        Ok(())
    }

    #[test]
    fn cpu_decoder_pointwise_matches_original_bits_and_fallbacks() -> CandleResult<()> {
        pointwise_cases(&Device::Cpu)
    }

    #[test]
    fn cpu_decoder_pointwise_preserves_bits_for_decoder_channels_and_layouts() -> CandleResult<()> {
        let device = &Device::Cpu;
        let bits = |t: Tensor| -> CandleResult<Vec<u32>> {
            Ok(t.flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        for channels in [32, 64, 128, 256, 512] {
            let weight = Tensor::rand(-0.1f32, 0.1, (channels + 1, channels, 1), device)?
                .narrow(0, 1, channels)?;
            for bias in [None, Some(Tensor::rand(-0.1f32, 0.1, channels, device)?)] {
                let conv = Conv1d::new(weight.clone(), bias, Conv1dConfig::default());
                for batch in [1, 2] {
                    for len in [
                        1, 7, 8, 9, 13, 16, 24, 32, 40, 56, 64, 72, 96, 128, 130, 160, 179, 256,
                        280, 480, 520, 960, 1040,
                    ] {
                        for strided in [false, true] {
                            let xs = if strided {
                                Tensor::rand(-0.1f32, 0.1, (batch, len, channels), device)?
                                    .transpose(1, 2)?
                            } else {
                                Tensor::rand(-0.1f32, 0.1, (batch + 1, channels, len), device)?
                                    .narrow(0, 1, batch)?
                            };
                            let reference = bits(conv.forward(&xs)?)?;
                            for _ in 0..2 {
                                let actual = bits(forward_dense_decoder_pointwise(&conv, &xs)?)?;
                                assert_eq!(
                                    actual
                                        .iter()
                                        .zip(&reference)
                                        .filter(|(a, b)| a != b)
                                        .count(),
                                    0,
                                    "channels={channels} batch={batch} len={len} strided={strided}"
                                );
                            }
                        }
                    }
                }
            }
            let xs = Tensor::rand(-0.1f32, 0.1, (1, channels, 32), device)?;
            let strided_weight =
                Tensor::rand(-0.1f32, 0.1, (channels, channels, 1), device)?.transpose(0, 1)?;
            let conv = Conv1d::new(strided_weight, None, Conv1dConfig::default());
            assert_eq!(
                bits(forward_dense_decoder_pointwise(&conv, &xs)?)?,
                bits(conv.forward(&xs)?)?
            );
        }
        Ok(())
    }

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn metal_decoder_pointwise_matches_original_bits_and_fallbacks() -> CandleResult<()> {
        pointwise_cases(&Device::new_metal(0)?)
    }

    #[test]
    fn cpu_transposed_convolution_keeps_original_path() -> CandleResult<()> {
        let device = Device::Cpu;
        let xs = Tensor::from_vec(vec![0.1f32; 8], (1, 2, 4), &device)?;
        let weights = Tensor::from_vec(vec![0.2f32; 6], (2, 1, 3), &device)?;
        let config = ConvTranspose1dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let layer = ConvTranspose1d::new(weights, None, config);
        let expected = layer.forward(&xs)?.flatten_all()?.to_vec1::<f32>()?;
        let actual = forward_dense_conv_transpose1d(&layer, &xs)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_transposed_convolution_crop_matches_host_reference() -> CandleResult<()> {
        let device = Device::new_metal(0)?;
        let (in_channels, out_channels) = (3, 2);
        for batch in [1, 2] {
            for frames in [1, 7, 130] {
                for stride in [2usize, 3, 4, 5, 8] {
                    for with_bias in [false, true] {
                        let kernel = 2 * stride;
                        let config = ConvTranspose1dConfig {
                            stride,
                            padding: stride.div_ceil(2),
                            output_padding: stride % 2,
                            ..Default::default()
                        };
                        let values = (0..batch * in_channels * frames)
                            .map(|i| (i as f32 * 0.13).sin() * 0.2)
                            .collect::<Vec<_>>();
                        let weights = (0..in_channels * out_channels * kernel)
                            .map(|i| (i as f32 * 0.17).cos() * 0.1)
                            .collect::<Vec<_>>();
                        let bias = if with_bias {
                            Some(Tensor::from_vec(vec![0.03f32, -0.02], 2, &device)?)
                        } else {
                            None
                        };
                        let xs = Tensor::from_vec(
                            values.clone(),
                            (batch, in_channels, frames),
                            &device,
                        )?;
                        let layer = ConvTranspose1d::new(
                            Tensor::from_vec(
                                weights.clone(),
                                (in_channels, out_channels, kernel),
                                &device,
                            )?,
                            bias,
                            config,
                        );
                        let out_len = (frames - 1) * stride + kernel - 2 * config.padding
                            + config.output_padding;
                        let mut reference = vec![0f64; batch * out_channels * out_len];
                        if with_bias {
                            for (i, value) in reference.iter_mut().enumerate() {
                                *value = f64::from([0.03f32, -0.02][(i / out_len) % 2]);
                            }
                        }
                        for b in 0..batch {
                            for c in 0..in_channels {
                                for t in 0..frames {
                                    let value = values[(b * in_channels + c) * frames + t];
                                    for o in 0..out_channels {
                                        for k in 0..kernel {
                                            let position = t * stride + k;
                                            if position >= config.padding {
                                                let position = position - config.padding;
                                                if position < out_len {
                                                    reference[(b * out_channels + o) * out_len
                                                        + position] += f64::from(value)
                                                        * f64::from(
                                                            weights[(c * out_channels + o)
                                                                * kernel
                                                                + k],
                                                        );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        let mut actuals = vec![forward_dense_conv_transpose1d(&layer, &xs)?];
                        // Candle's original debug length formula subtracts padding
                        // before adding the kernel width and underflows for one frame.
                        if (frames - 1) * stride >= 2 * config.padding {
                            actuals.push(layer.forward(&xs)?);
                        }
                        for tensor in actuals {
                            assert_eq!(tensor.dims(), &[batch, out_channels, out_len]);
                            let actual = tensor.flatten_all()?.to_vec1::<f32>()?;
                            for (index, (&actual, &expected)) in
                                actual.iter().zip(&reference).enumerate()
                            {
                                assert!(
                                    actual.is_finite()
                                        && (f64::from(actual) - expected).abs() < 2e-5,
                                    "batch={batch} frames={frames} stride={stride} bias={with_bias} index={index}: actual={actual} expected={expected}"
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
