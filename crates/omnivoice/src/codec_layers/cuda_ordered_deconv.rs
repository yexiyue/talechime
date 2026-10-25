//! Geometry-specialized F32 CUDA decoder kernel with unchanged FMA order.
use candle_core::{CustomOp2, DType, Layout, Result, Shape, Tensor};
use candle_nn::ConvTranspose1d;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ordered_deconv.ptx"));
const MODULE: &str = "talechime_omnivoice_ordered_deconv_20261024_v1";

#[derive(Clone, Copy)]
struct Ordered {
    cin: usize,
    cout: usize,
    stride: usize,
    function: &'static str,
}

#[cfg(test)]
thread_local! { static EAGER:std::cell::Cell<bool>=const{std::cell::Cell::new(false)}; }
#[cfg(test)]
pub(crate) fn with_eager<T>(run: impl FnOnce() -> T) -> T {
    struct Guard(bool);
    impl Drop for Guard {
        fn drop(&mut self) {
            EAGER.with(|s| s.set(self.0));
        }
    }
    let _guard = Guard(EAGER.with(|s| s.replace(true)));
    run()
}

pub(crate) fn forward(layer: &ConvTranspose1d, xs: &Tensor) -> Result<Option<Tensor>> {
    #[cfg(test)]
    if EAGER.with(std::cell::Cell::get) {
        return Ok(None);
    }
    let cfg = layer.config();
    if !xs.device().is_cuda()
        || xs.dtype() != DType::F32
        || layer.weight().dtype() != DType::F32
        || !xs.is_contiguous()
        || !layer.weight().is_contiguous()
        || !xs.device().same_device(layer.weight().device())
        || cfg.groups != 1
        || cfg.dilation != 1
    {
        return Ok(None);
    }
    let Ok((batch, cin, len)) = xs.dims3() else {
        return Ok(None);
    };
    let Ok((wcin, cout, kernel)) = layer.weight().dims3() else {
        return Ok(None);
    };
    let function = match (cin, cout, cfg.stride) {
        (1024, 512, 8) => "ordered_deconv_1024_512_8",
        (512, 256, 5) => "ordered_deconv_512_256_5",
        (256, 128, 4) => "ordered_deconv_256_128_4",
        (128, 64, 2) => "ordered_deconv_128_64_2",
        (64, 32, 3) => "ordered_deconv_64_32_3",
        _ => return Ok(None),
    };
    if batch != 1
        || len == 0
        || cin != wcin
        || kernel != 2 * cfg.stride
        || cfg.padding != cfg.stride.div_ceil(2)
        || cfg.output_padding != cfg.stride % 2
    {
        return Ok(None);
    }
    let Some(output_len) = len.checked_mul(cfg.stride) else {
        return Ok(None);
    };
    let Some(output_count) = output_len.checked_mul(cout) else {
        return Ok(None);
    };
    if output_count > i32::MAX as usize
        || xs.elem_count() > i32::MAX as usize
        || layer.weight().elem_count() > i32::MAX as usize
    {
        return Ok(None);
    }
    let candle_core::Device::Cuda(device) = xs.device() else {
        return Ok(None);
    };
    use cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
    if !device
        .cuda_stream()
        .context()
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .is_ok_and(|major| major >= 8)
    {
        return Ok(None);
    }
    let output = xs.apply_op2(
        layer.weight(),
        Ordered {
            cin,
            cout,
            stride: cfg.stride,
            function,
        },
    )?;
    Ok(Some(match layer.bias() {
        Some(bias) => output.broadcast_add(&bias.reshape((1, cout, 1))?)?,
        None => output,
    }))
}

impl CustomOp2 for Ordered {
    fn name(&self) -> &'static str {
        "omnivoice-ordered-deconv"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("ordered decoder is CUDA-only")
    }
    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        xl: &Layout,
        w: &candle_core::CudaStorage,
        wl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::{backend::BackendStorage, cuda_backend::WrapErr};
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        let (batch, cin, len) = xl.shape().dims3()?;
        if batch != 1
            || cin != self.cin
            || len == 0
            || wl.dims() != [self.cin, self.cout, 2 * self.stride]
        {
            candle_core::bail!("ordered decoder geometry changed")
        }
        let output_len = len
            .checked_mul(self.stride)
            .ok_or_else(|| candle_core::Error::Msg("decoder output length overflow".into()))?;
        let output_count = output_len
            .checked_mul(self.cout)
            .ok_or_else(|| candle_core::Error::Msg("decoder element count overflow".into()))?;
        if output_count > i32::MAX as usize
            || xl.shape().elem_count() > i32::MAX as usize
            || wl.shape().elem_count() > i32::MAX as usize
        {
            candle_core::bail!("ordered decoder index exceeds i32")
        }
        let (a, b) = xl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("contiguous decoder input required".into()))?;
        let xv = x.as_cuda_slice::<f32>()?.slice(a..b);
        let (a, b) = wl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("contiguous decoder weight required".into()))?;
        let wv = w.as_cuda_slice::<f32>()?.slice(a..b);
        let device = x.device().clone();
        // SAFETY: checked i32 dimensions and logical slices bound every read;
        // the kernel writes each of output_count elements exactly once.
        let mut out = unsafe { device.alloc::<f32>(output_count) }?;
        let function = device.get_or_load_custom_func(self.function, MODULE, PTX)?;
        let count = output_count as u32;
        let len = len as i32;
        let cfg = LaunchConfig {
            grid_dim: (count.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = function.builder();
        builder
            .arg(&mut out)
            .arg(&xv)
            .arg(&wv)
            .arg(&len)
            .arg(&count);
        // SAFETY: buffer slices, scalar ABI and launch size match ordered_deconv.cu.
        unsafe { builder.launch(cfg) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, device),
            (batch, self.cout, output_len).into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Module};
    use candle_nn::ConvTranspose1dConfig;
    #[test]
    fn cpu_and_unvalidated_geometry_keep_original_path() -> Result<()> {
        let x = Tensor::zeros((1, 32, 3), DType::F32, &Device::Cpu)?;
        let w = Tensor::zeros((32, 16, 4), DType::F32, &Device::Cpu)?;
        let layer = ConvTranspose1d::new(
            w,
            None,
            ConvTranspose1dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
        );
        assert!(forward(&layer, &x)?.is_none());
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA; synthetic tensors only, no weights or downloads"]
    fn ordered_kernel_edges_offsets_and_fallbacks() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for (cin, cout, stride) in [
            (1024, 512, 8),
            (512, 256, 5),
            (256, 128, 4),
            (128, 64, 2),
            (64, 32, 3),
        ] {
            let weights = (0..cin * cout * 2 * stride)
                .map(|i| ((i % 31) as f32 - 15.0) / 31.0)
                .collect::<Vec<_>>();
            let weight = Tensor::from_vec(weights, (cin, cout, 2 * stride), &device)?;
            let layer = ConvTranspose1d::new(
                weight,
                None,
                ConvTranspose1dConfig {
                    stride,
                    padding: stride.div_ceil(2),
                    output_padding: stride % 2,
                    ..Default::default()
                },
            );
            for len in [1usize, 2, 3, 7, 48, 49] {
                let base = [0.0f32, -0.0, 1e-40, -1e-40, 0.5, -0.5, 1.0, -1.0];
                let data = (0..2 * cin * len)
                    .map(|i| base[i % base.len()])
                    .collect::<Vec<_>>();
                // Nonzero storage offset, while preserving contiguous NCL.
                let input = Tensor::from_vec(data, (2, cin, len), &device)?.narrow(0, 1, 1)?;
                let expected = layer.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
                let actual = forward(&layer, &input)?
                    .expect("fixture must select ordered kernel")
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "geometry {cin}/{cout}/{stride}/{len}"
                );
                assert!(forward(&layer, &input.to_dtype(DType::F16)?)?.is_none());
                if len > 1 {
                    assert!(
                        forward(
                            &layer,
                            &input.transpose(1, 2)?.contiguous()?.transpose(1, 2)?
                        )?
                        .is_none()
                    );
                }
            }
        }
        Ok(())
    }
}
