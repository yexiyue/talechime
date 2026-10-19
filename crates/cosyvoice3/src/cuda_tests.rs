use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::{DiT, DiTConfig};

/// Exercises the real grouped-convolution/attention layouts without a checkpoint.
#[test]
#[ignore = "requires a CUDA device; run serially"]
fn dit_cuda_operator_smoke() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let cfg = DiTConfig {
        dim: 64,
        depth: 1,
        heads: 2,
        dim_head: 32,
        ..DiTConfig::default()
    };
    let dit = DiT::new(cfg, VarBuilder::zeros(DType::F32, &device))?;
    let x = Tensor::ones((2, 80, 52), DType::F32, &device)?;
    let mask = Tensor::ones((2, 1, 52), DType::F32, &device)?;
    let speaker = Tensor::zeros((2, 80), DType::F32, &device)?;
    let time = Tensor::from_slice(&[0.1f32, 0.1], 2, &device)?;
    for streaming in [false, true] {
        let output = dit.forward(&x, &mask, &x, &time, &speaker, &x, streaming)?;
        device.synchronize()?;
        assert_eq!(output.dims(), x.dims());
        assert!(
            output
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|v| v.is_finite())
        );
    }
    Ok(())
}
