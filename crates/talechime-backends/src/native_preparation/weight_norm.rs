//! Fixed reduction order for native conversion, independent of host SIMD width.
// Follows PyTorch v2.7.1 WeightNormKernel.cpp and vec/functional_base.h (BSD-3).
// The recipe uses the independently verified ARM64 reference order: four F32
// lanes with a pairwise finish, two F64 lanes. See licenses/PYTORCH-LICENSE.
use super::PreparationControl;
use candle_core::{DType, Device, Tensor};

pub(crate) fn fold(v: &Tensor, g: &Tensor, control: &PreparationControl) -> anyhow::Result<Tensor> {
    anyhow::ensure!(
        v.device().is_cpu() && g.device().is_cpu(),
        "weight conversion requires CPU tensors"
    );
    let rows = *v
        .dims()
        .first()
        .ok_or_else(|| anyhow::anyhow!("weight normalization requires a dimension"))?;
    anyhow::ensure!(
        rows > 0
            && v.elem_count().is_multiple_of(rows)
            && g.elem_count() == rows
            && v.dtype() == g.dtype(),
        "unsupported weight normalization layout"
    );
    let width = v.elem_count() / rows;
    anyhow::ensure!(width > 0, "empty weight normalization row");
    match v.dtype() {
        DType::F32 => {
            let values = v.flatten_all()?.to_vec1::<f32>()?;
            let scales = g.flatten_all()?.to_vec1::<f32>()?;
            let mut output = Vec::with_capacity(values.len());
            for (row, scale) in values.chunks_exact(width).zip(scales) {
                control.check()?;
                let mut lanes = [0f32; 4];
                for (index, value) in row.iter().enumerate() {
                    lanes[index % 4] += value * value;
                }
                let sum = if width < 4 {
                    lanes[..width].iter().copied().sum::<f32>()
                } else {
                    (lanes[0] + lanes[2]) + (lanes[1] + lanes[3])
                };
                let norm = sum.sqrt();
                anyhow::ensure!(
                    norm.is_finite() && norm > 0.0 && scale.is_finite(),
                    "invalid weight normalization values"
                );
                let factor = scale / norm;
                output.extend(row.iter().map(|value| value * factor));
            }
            Ok(Tensor::from_vec(output, v.dims(), &Device::Cpu)?)
        }
        DType::F64 => {
            let values = v.flatten_all()?.to_vec1::<f64>()?;
            let scales = g.flatten_all()?.to_vec1::<f64>()?;
            let mut output = Vec::with_capacity(values.len());
            for (row, scale) in values.chunks_exact(width).zip(scales) {
                control.check()?;
                let mut lanes = [0f64; 2];
                for (index, value) in row.iter().enumerate() {
                    lanes[index % 2] += value * value;
                }
                let norm = (lanes[0] + lanes[1]).sqrt();
                anyhow::ensure!(
                    norm.is_finite() && norm > 0.0 && scale.is_finite(),
                    "invalid weight normalization values"
                );
                let factor = scale / norm;
                output.extend(row.iter().map(|value| value * factor));
            }
            Ok(Tensor::from_vec(output, v.dims(), &Device::Cpu)?)
        }
        _ => anyhow::bail!("weight conversion requires F32 or F64"),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_preparation::fold_weight_norm as fold;
    #[test]
    fn folds_both_precisions_and_rejects_zero_norm_or_cancel() {
        let control = PreparationControl::default();
        let v = Tensor::new(&[[3f32, 4.]], &Device::Cpu).unwrap();
        let g = Tensor::new(&[10f32], &Device::Cpu).unwrap();
        assert_eq!(
            fold(&v, &g, &control).unwrap().to_vec2::<f32>().unwrap(),
            [vec![6., 8.]]
        );
        assert_eq!(
            fold(
                &v.to_dtype(DType::F64).unwrap(),
                &g.to_dtype(DType::F64).unwrap(),
                &control
            )
            .unwrap()
            .to_vec2::<f64>()
            .unwrap(),
            [vec![6., 8.]]
        );
        let zero = Tensor::zeros((1, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(fold(&zero, &g, &control).is_err());
        control.cancel();
        assert!(fold(&v, &g, &control).is_err());
    }
}
