//! Test-only polyphase GEMM screen, preserving tap-major K packing.
use super::*;
use candle_core::IndexOp;
use std::time::Instant;

fn relayout(
    layer: &ConvTranspose1d,
    xs: &Tensor,
    input: bool,
    weight: bool,
) -> CandleResult<Tensor> {
    let xs = if input {
        xs.transpose(1, 2)?.contiguous()?.transpose(1, 2)?
    } else {
        xs.clone()
    };
    let weight = if weight {
        layer
            .weight()
            .permute((1, 2, 0))?
            .contiguous()?
            .permute((2, 0, 1))?
    } else {
        layer.weight().clone()
    };
    let cfg = layer.config();
    let output = xs.conv_transpose1d(
        &weight,
        cfg.padding,
        cfg.output_padding,
        cfg.stride,
        cfg.dilation,
        cfg.groups,
    )?;
    match layer.bias() {
        Some(bias) => output.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1))?),
        None => Ok(output),
    }
}

fn phase_gemm(layer: &ConvTranspose1d, xs: &Tensor) -> CandleResult<Tensor> {
    let cfg = layer.config();
    let (batch, channels, len) = xs.dims3()?;
    let (cin, cout, kernel) = layer.weight().dims3()?;
    assert!(xs.device().is_cuda() && xs.dtype() == DType::F32);
    assert_eq!(cin, channels);
    assert_eq!(cfg.groups, 1);
    assert_eq!(cfg.dilation, 1);
    assert_eq!(kernel, 2 * cfg.stride);
    assert_eq!(cfg.padding, cfg.stride.div_ceil(2));
    assert_eq!(cfg.output_padding, cfg.stride % 2);
    // Every output phase has len elements for this decoder's geometry.
    let padded = xs.pad_with_zeros(2, 1, 1)?;
    let mut phases = Vec::with_capacity(cfg.stride);
    for phase in 0..cfg.stride {
        let first_tap = (phase + cfg.padding) % cfg.stride;
        let shift = (phase + cfg.padding) / cfg.stride;
        // Preserve the CUDA kernel's tap-major then channel-major contraction.
        let columns = Tensor::cat(
            &[
                padded.narrow(2, 1 + shift, len)?,
                padded.narrow(2, shift, len)?,
            ],
            1,
        )?;
        let weight = Tensor::cat(
            &[
                layer.weight().i((.., .., first_tap))?.t()?,
                layer.weight().i((.., .., first_tap + cfg.stride))?.t()?,
            ],
            1,
        )?
        .unsqueeze(0)?
        .broadcast_as((batch, cout, 2 * cin))?;
        phases.push(weight.matmul(&columns)?);
    }
    let output = Tensor::stack(&phases, 3)?.reshape((batch, cout, len * cfg.stride))?;
    match layer.bias() {
        Some(bias) => output.broadcast_add(&bias.reshape((1, cout, 1))?),
        None => Ok(output),
    }
}

pub(crate) fn compare(index: usize, conv: &CodecConvTranspose1d, xs: &Tensor) -> Result<usize> {
    let CodecConvTranspose1dInner::Dense(layer) = &conv.inner else {
        return Err(crate::OmniVoiceError::Unsupported(
            "dense deconv fixture required".into(),
        ));
    };
    let expected = layer.forward(xs)?;
    let candidate = phase_gemm(layer, xs)?;
    assert_eq!(expected.dims(), candidate.dims());
    let a = expected.flatten_all()?.to_vec1::<f32>()?;
    let b = candidate.flatten_all()?.to_vec1::<f32>()?;
    assert!(a.iter().chain(&b).all(|v| v.is_finite()));
    let diffs = a
        .iter()
        .zip(&b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max_abs = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let mae = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / a.len() as f64;
    for (name, pack_input, pack_weight) in [
        ("input", true, false),
        ("weight", false, true),
        ("both", true, true),
    ] {
        let output = relayout(layer, xs, pack_input, pack_weight)?;
        let actual = output.flatten_all()?.to_vec1::<f32>()?;
        let differences = a
            .iter()
            .zip(&actual)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        println!(
            "CUDA_DECONV_LAYOUT_BITS {}",
            serde_json::json!({"block":index,"layout":name,"bit_differences":differences})
        );
        assert_eq!(differences, 0, "same kernel and values must preserve bits");
        for group in 0..3 {
            for (slot, fast) in [false, true, true, false].into_iter().enumerate() {
                xs.device().synchronize()?;
                let start = Instant::now();
                for _ in 0..5 {
                    std::hint::black_box(if fast {
                        relayout(layer, xs, pack_input, pack_weight)?
                    } else {
                        layer.forward(xs)?
                    });
                }
                xs.device().synchronize()?;
                println!(
                    "CUDA_DECONV_LAYOUT_TIME {}",
                    serde_json::json!({"block":index,"layout":name,"group":group,"slot":slot,"packed":fast,"seconds_per_call":start.elapsed().as_secs_f64()/5.0,"scope":"deconv_bias_including_input_weight_relayout"})
                );
            }
        }
    }
    println!(
        "CUDA_PHASE_DECONV_BITS {}",
        serde_json::json!({"block":index,"input_shape":xs.dims(),"stride":layer.config().stride,"output_shape":candidate.dims(),"bit_differences":diffs,"max_abs":max_abs,"mae":mae})
    );
    for group in 0..3 {
        for (slot, fast) in [false, true, true, false].into_iter().enumerate() {
            xs.device().synchronize()?;
            let start = Instant::now();
            for _ in 0..5 {
                std::hint::black_box(if fast {
                    phase_gemm(layer, xs)?
                } else {
                    layer.forward(xs)?
                });
            }
            xs.device().synchronize()?;
            println!(
                "CUDA_PHASE_DECONV_TIME {}",
                serde_json::json!({"block":index,"group":group,"slot":slot,"phase_gemm":fast,"seconds_per_call":start.elapsed().as_secs_f64()/5.0,"scope":"deconv_bias_including_weight_and_input_packing"})
            );
        }
    }
    Ok(diffs)
}

// Same F32 FMA sequence as Candle's direct kernel; specialize only geometry
// and integer address arithmetic. This remains a test-only feasibility probe.
struct OrderedDeconv {
    cin: usize,
    cout: usize,
    stride: usize,
    ptx: std::sync::Arc<String>,
}
impl OrderedDeconv {
    fn new(cin: usize, cout: usize, stride: usize) -> CandleResult<Self> {
        type Key = (usize, usize, usize);
        type Cache = std::collections::HashMap<Key, std::sync::Arc<String>>;
        static CACHE: std::sync::OnceLock<std::sync::Mutex<Cache>> = std::sync::OnceLock::new();
        let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
        let key = (cin, cout, stride);
        let ptx = if let Some(ptx) = cache.get(&key) {
            ptx.clone()
        } else {
            let source = format!(
                r#"
extern "C" __global__ void ordered_deconv(float* out,const float* x,const float* w,int len,unsigned count) {{
    unsigned id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id>=count)return;
    const int out_len=len*{stride};
    const int ox=id%out_len;
    const int oc=(id/out_len)%{cout};
    const int batch=id/(out_len*{cout});
    const int tap0=(ox+{padding})%{stride};
    float d=0.0f;
    #pragma unroll
    for(int part=0;part<2;part++){{
        const int tap=tap0+part*{stride};
        const int ix=(ox+{padding}-tap)/{stride};
        if(ix<0||ix>=len)continue;
        #pragma unroll 4
        for(int ic=0;ic<{cin};ic++){{
            d=__fmaf_rn(x[(batch*{cin}+ic)*len+ix],w[(ic*{cout}+oc)*{kernel}+tap],d);
        }}
    }}
    out[id]=d;
}}
"#,
                padding = stride.div_ceil(2),
                kernel = 2 * stride
            );
            let started = Instant::now();
            let options = cudarc::nvrtc::CompileOptions {
                arch: Some("compute_80"),
                ftz: Some(false),
                fmad: Some(true),
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(source, options)
                .map_err(|e| candle_core::Error::Msg(format!("NVRTC: {e:?}")))?
                .to_src();
            println!(
                "CUDA_ORDERED_DECONV_COMPILE {}",
                serde_json::json!({"cin":cin,"cout":cout,"stride":stride,"seconds":started.elapsed().as_secs_f64(),"arch":"compute_80"})
            );
            let ptx = std::sync::Arc::new(ptx);
            cache.insert(key, ptx.clone());
            ptx
        };
        Ok(Self {
            cin,
            cout,
            stride,
            ptx,
        })
    }
}
impl candle_core::CustomOp2 for OrderedDeconv {
    fn name(&self) -> &'static str {
        "ordered-deconv-probe"
    }
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("CUDA-only probe")
    }
    fn cuda_fwd(
        &self,
        x: &candle_core::CudaStorage,
        xl: &candle_core::Layout,
        w: &candle_core::CudaStorage,
        wl: &candle_core::Layout,
    ) -> CandleResult<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::{
            WrapErr,
            cudarc::driver::{LaunchConfig, PushKernelArg},
        };
        let (batch, cin, len) = xl.shape().dims3()?;
        if cin != self.cin
            || wl.dims() != [self.cin, self.cout, 2 * self.stride]
            || len == 0
            || batch == 0
        {
            candle_core::bail!("ordered deconv geometry changed")
        }
        let shape = candle_core::Shape::from((batch, self.cout, len * self.stride));
        let count = u32::try_from(shape.elem_count())
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        if shape.elem_count() > i32::MAX as usize
            || xl.shape().elem_count() > i32::MAX as usize
            || wl.shape().elem_count() > i32::MAX as usize
        {
            candle_core::bail!("ordered deconv index exceeds i32")
        }
        let (a, b) = xl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("contiguous input required".into()))?;
        let xv = x.as_cuda_slice::<f32>()?.slice(a..b);
        let (a, b) = wl
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("contiguous weight required".into()))?;
        let wv = w.as_cuda_slice::<f32>()?.slice(a..b);
        let device = x.device().clone();
        // SAFETY: launch writes every element exactly once; all reads are
        // constrained by validated dimensions and contiguous logical slices.
        let mut out = unsafe { device.alloc::<f32>(shape.elem_count()) }?;
        let name = format!(
            "omni_ordered_deconv_probe_v1_{}_{}_{}",
            self.cin, self.cout, self.stride
        );
        let function =
            device.get_or_load_custom_func("ordered_deconv", &name, self.ptx.as_str())?;
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
        // SAFETY: scalar ABI and slice bounds match the compiled kernel above.
        unsafe { builder.launch(cfg) }.w()?;
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, device),
            shape,
        ))
    }
}

fn ordered(layer: &ConvTranspose1d, xs: &Tensor) -> CandleResult<Tensor> {
    let (cin, cout, kernel) = layer.weight().dims3()?;
    let cfg = layer.config();
    assert_eq!(kernel, 2 * cfg.stride);
    assert_eq!(cfg.padding, cfg.stride.div_ceil(2));
    assert_eq!(cfg.output_padding, cfg.stride % 2);
    assert_eq!(cfg.groups, 1);
    assert_eq!(cfg.dilation, 1);
    let output = xs.apply_op2(layer.weight(), OrderedDeconv::new(cin, cout, cfg.stride)?)?;
    match layer.bias() {
        Some(bias) => output.broadcast_add(&bias.reshape((1, cout, 1))?),
        None => Ok(output),
    }
}

pub(crate) fn compare_ordered(
    index: usize,
    conv: &CodecConvTranspose1d,
    xs: &Tensor,
) -> Result<usize> {
    let CodecConvTranspose1dInner::Dense(layer) = &conv.inner else {
        return Err(crate::OmniVoiceError::Unsupported(
            "dense decoder required".into(),
        ));
    };
    let expected = layer.forward(xs)?.flatten_all()?.to_vec1::<f32>()?;
    let actual = ordered(layer, xs)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(expected.len(), actual.len());
    let diffs = expected
        .iter()
        .zip(&actual)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let maximum = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "CUDA_ORDERED_DECONV_BITS {}",
        serde_json::json!({"block":index,"shape":xs.dims(),"bit_differences":diffs,"max_abs":maximum})
    );
    for group in 0..3 {
        for (slot, fast) in [false, true, true, false].into_iter().enumerate() {
            xs.device().synchronize()?;
            let started = Instant::now();
            for _ in 0..5 {
                std::hint::black_box(if fast {
                    ordered(layer, xs)?
                } else {
                    layer.forward(xs)?
                });
            }
            xs.device().synchronize()?;
            println!(
                "CUDA_ORDERED_DECONV_TIME {}",
                serde_json::json!({"block":index,"group":group,"slot":slot,"ordered":fast,"seconds_per_call":started.elapsed().as_secs_f64()/5.0,"scope":"deconv_and_bias_compile_excluded"})
            );
        }
    }
    Ok(diffs)
}
