use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

/// Keep the original layout for short/strided inputs; pretranspose only the
/// CPU K/V geometry whose local benefit justifies the extra weight storage.
#[derive(Debug, Clone)]
pub(crate) struct CpuProjection {
    original: Linear,
    weight_t: Option<Tensor>,
    span: tracing::Span,
}

impl CpuProjection {
    pub(crate) fn load(
        input: usize,
        output: usize,
        bias: bool,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        // Use Candle's constructor to preserve its initialization hints too.
        let original = candle_nn::linear_b(input, output, bias, vb)?;
        let weight = original.weight();
        let weight_t = if weight.device().is_cpu()
            && weight.dtype() == DType::F32
            && candle_core::utils::has_accelerate()
            && !weight.is_variable()
            && (input, output) == (1024, 1024)
        {
            Some(weight.t()?.contiguous()?)
        } else {
            None
        };
        Ok(Self {
            original,
            weight_t,
            span: tracing::span!(tracing::Level::TRACE, "linear"),
        })
    }
}

impl Module for CpuProjection {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let Some(weight_t) = &self.weight_t else {
            return self.original.forward(xs);
        };
        let (input, output) = weight_t.dims2()?;
        if !xs.is_contiguous()
            || !(2..=4).contains(&xs.rank())
            || xs.dims().last() != Some(&input)
            || xs.elem_count() / input < 8
        {
            return self.original.forward(xs);
        }
        let mut shape = xs.dims().to_vec();
        let last = shape.len() - 1;
        shape[last] = output;
        let result = xs.reshape(((), input))?.matmul(weight_t)?.reshape(shape)?;
        match self.original.bias() {
            Some(bias) => result.broadcast_add(bias),
            None => Ok(result),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    fn bits(t: Tensor) -> Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    #[test]
    fn cpu_projection_preserves_bits_for_short_strided_and_batched_inputs() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::rand(-0.1f32, 0.1, (1024, 1024), &device)?;
        for with_bias in [false, true] {
            let bias = with_bias
                .then(|| Tensor::rand(-0.1f32, 0.1, 1024, &device))
                .transpose()?;
            let mut tensors = HashMap::from([("weight".to_owned(), weight.clone())]);
            if let Some(bias) = &bias {
                tensors.insert("bias".to_owned(), bias.clone());
            }
            let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
            let layer = CpuProjection::load(1024, 1024, with_bias, vb)?;
            assert_eq!(
                layer.weight_t.is_some(),
                candle_core::utils::has_accelerate()
            );
            let reference = Linear::new(weight.clone(), bias);
            for batch in [1, 2] {
                for seq in [1, 2, 7, 8, 33, 157] {
                    for strided in [false, true] {
                        let xs = if strided {
                            Tensor::rand(-1f32, 1.0, (batch, 1024, seq), &device)?
                                .transpose(1, 2)?
                        } else {
                            Tensor::rand(-1f32, 1.0, (batch, seq, 1024), &device)?
                        };
                        let expected = bits(reference.forward(&xs)?)?;
                        for _ in 0..2 {
                            let actual = bits(layer.forward(&xs)?)?;
                            let differing =
                                actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
                            assert_eq!(
                                differing, 0,
                                "batch={batch} seq={seq} strided={strided} bias={with_bias}"
                            );
                        }
                    }
                }
            }
            for shape in [vec![8, 1024], vec![2, 2, 2, 1024]] {
                let xs = Tensor::rand(-1f32, 1.0, shape, &device)?;
                assert_eq!(bits(layer.forward(&xs)?)?, bits(reference.forward(&xs)?)?);
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_projection_keeps_other_shapes_on_the_original_route() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::rand(-1f32, 1.0, (16, 8), &device)?;
        let vb = VarBuilder::from_tensors(
            HashMap::from([("weight".into(), weight.clone())]),
            DType::F32,
            &device,
        );
        let layer = CpuProjection::load(8, 16, false, vb)?;
        assert!(layer.weight_t.is_none());
        let xs = Tensor::rand(-1f32, 1.0, (2, 7, 8), &device)?;
        assert_eq!(
            bits(layer.forward(&xs)?)?,
            bits(Linear::new(weight, None).forward(&xs)?)?
        );
        Ok(())
    }

    #[test]
    fn cpu_projection_does_not_pretranspose_f16() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::ones((1024, 1024), DType::F16, &device)?;
        let vb = VarBuilder::from_tensors(
            HashMap::from([("weight".into(), weight)]),
            DType::F16,
            &device,
        );
        let layer = CpuProjection::load(1024, 1024, false, vb)?;
        assert!(layer.weight_t.is_none());
        Ok(())
    }

    #[test]
    fn cpu_projection_keeps_variable_weights_live() -> Result<()> {
        let device = Device::Cpu;
        let mut variables = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&variables, DType::F32, &device);
        let layer = CpuProjection::load(1024, 1024, false, vb)?;
        assert!(layer.weight_t.is_none());
        variables.set_one("weight", Tensor::zeros((1024, 1024), DType::F32, &device)?)?;
        let xs = Tensor::ones((8, 1024), DType::F32, &device)?;
        assert!(
            layer
                .forward(&xs)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| *v == 0.)
        );
        Ok(())
    }

    #[test]
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn metal_projection_keeps_the_original_route_and_bits() -> Result<()> {
        let device = Device::new_metal(0)?;
        let weight = Tensor::rand(-0.1f32, 0.1, (1024, 1024), &device)?;
        let vb = VarBuilder::from_tensors(
            HashMap::from([("weight".into(), weight.clone())]),
            DType::F32,
            &device,
        );
        let layer = CpuProjection::load(1024, 1024, false, vb)?;
        assert!(layer.weight_t.is_none());
        let xs = Tensor::rand(-1f32, 1.0, (2, 8, 1024), &device)?;
        assert_eq!(
            bits(layer.forward(&xs)?)?,
            bits(Linear::new(weight, None).forward(&xs)?)?
        );
        Ok(())
    }
}
