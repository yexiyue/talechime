use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor, op::UnaryOpT};

struct SiluMul;

impl CustomOp2 for SiluMul {
    fn name(&self) -> &'static str {
        "omnivoice_cpu_silu_mul"
    }

    fn cpu_fwd(
        &self,
        gate: &CpuStorage,
        gl: &Layout,
        up: &CpuStorage,
        ul: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let Some((ga, gb)) = gl.contiguous_offsets() else {
            candle_core::bail!("gate must be contiguous");
        };
        let Some((ua, ub)) = ul.contiguous_offsets() else {
            candle_core::bail!("up must be contiguous");
        };
        let gate = &gate.as_slice::<f32>()?[ga..gb];
        let up = &up.as_slice::<f32>()?[ua..ub];
        if gl.shape() != ul.shape() {
            candle_core::bail!("activation shapes differ");
        }
        let mut output = vec![0f32; gate.len()];
        // Preserve Candle's feature-selected SiLU implementation and its rounding.
        candle_core::op::Silu::f32_vec(gate, &mut output);
        for (value, up) in output.iter_mut().zip(up) {
            *value *= up;
        }
        Ok((CpuStorage::F32(output), gl.shape().clone()))
    }
}

pub(crate) fn forward(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.device().is_cpu()
        && up.device().is_cpu()
        && gate.dtype() == DType::F32
        && up.dtype() == DType::F32
        && gate.is_contiguous()
        && up.is_contiguous()
        && gate.shape() == up.shape()
    {
        gate.apply_op2_no_bwd(up, &SiluMul)
    } else {
        gate.silu()? * up
    }
}
