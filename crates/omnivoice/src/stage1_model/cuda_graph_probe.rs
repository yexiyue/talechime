//! Real fixed-code CUDA decoder feasibility; no production routing.
use super::*;
use crate::stage0_model::{CudaTestReplay as Replay, cuda_test_is_capturing as is_capturing};
use candle_core::{DType, Device};
use std::{fs, time::Instant};

#[derive(serde::Deserialize)]
struct Codes {
    codebooks: usize,
    frames: usize,
    values: Vec<i64>,
}

fn values(tensor: &Tensor) -> Result<Vec<u32>> {
    Ok(tensor
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

#[test]
#[ignore = "requires explicit local CUDA model and a directory of actual code JSON files; serial release"]
fn fixed_decoder_graph_bits_and_cost() -> Result<()> {
    let env = |name| std::env::var(name).map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()));
    let bundle =
        crate::stage1_decoder::Stage1DecoderBundle::from_model_root(env("OMNI_CUDA_TRACE_MODEL")?)?;
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let weights = CodecWeightSource::from_weight_path(bundle.weights_path(), DType::F32, &device)?;
    let model = Stage1Model::load(&weights, bundle.model_config())?;
    let mut files = fs::read_dir(env("OMNI_CUDA_DECODER_CODES")?)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|p| p.extension().is_some_and(|v| v == "json"));
    files.sort();
    assert!(!files.is_empty());
    for path in files {
        let codes: Codes = serde_json::from_slice(&fs::read(&path)?)?;
        assert_eq!(codes.codebooks, 8);
        assert_eq!(codes.values.len(), codes.codebooks * codes.frames);
        assert!(codes.values.iter().all(|v| (0..1024).contains(v)));
        let input = Tensor::from_vec(
            codes.values.clone(),
            (1, codes.codebooks, codes.frames),
            &device,
        )?;
        let changed = Tensor::from_vec(
            codes
                .values
                .iter()
                .map(|v| (v + 1) % 1024)
                .collect::<Vec<_>>(),
            input.shape(),
            &device,
        )?;
        let _guard = cuda.enable_cuda_graph_htod_cache();
        device.synchronize()?;
        let started = Instant::now();
        let warm = model.decode_tensor(&input)?;
        let graph = Replay::record(cuda, &[&input], &warm, vec![], |args| {
            model
                .decode_tensor(&args[0])
                .map_err(|e| candle_core::Error::Msg(e.to_string()))
        })?;
        assert!(!is_capturing(cuda.cuda_stream().as_ref())?);
        device.synchronize()?;
        println!(
            "CUDA_DECODER_GRAPH_RECORD {}",
            serde_json::json!({"case":path.file_stem().unwrap().to_string_lossy(),"frames":codes.frames,"seconds":started.elapsed().as_secs_f64(),"scope":"warmup_record_instantiate_sync"})
        );
        for (call, input) in [&input, &changed, &input].into_iter().enumerate() {
            let expected = values(&model.decode_tensor(input)?)?;
            let actual = values(&graph.launch(&[input])?)?;
            assert_eq!(expected.len(), actual.len());
            let diff = expected.iter().zip(&actual).filter(|(a, b)| a != b).count();
            println!(
                "CUDA_DECODER_GRAPH_BITS {}",
                serde_json::json!({"case":path.file_stem().unwrap().to_string_lossy(),"frames":codes.frames,"call":call,"samples":actual.len(),"bit_differences":diff})
            );
            assert_eq!(diff, 0, "raw PCM must stay bit-exact including A/B/A");
        }
        for group in 0..5 {
            for (slot, fast) in [false, true, true, false].into_iter().enumerate() {
                device.synchronize()?;
                let started = Instant::now();
                for _ in 0..5 {
                    let out = if fast {
                        graph.launch(&[&input])?
                    } else {
                        model.decode_tensor(&input)?
                    };
                    std::hint::black_box(out.flatten_all()?.to_vec1::<f32>()?);
                }
                device.synchronize()?;
                println!(
                    "CUDA_DECODER_GRAPH_TIME {}",
                    serde_json::json!({"case":path.file_stem().unwrap().to_string_lossy(),"frames":codes.frames,"group":group,"slot":slot,"graph":fast,"seconds_per_call":started.elapsed().as_secs_f64()/5.0,"scope":"device_codes_through_raw_cpu_pcm"})
                );
            }
        }
        // Drop the graph and scratch buffers before their model/weights owner.
        drop(graph);
    }
    Ok(())
}

fn measure_stage(
    name: &str,
    input: &Tensor,
    export: Option<&std::path::Path>,
    mut run: impl FnMut(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let output = run(input)?;
    for group in 0..3 {
        input.device().synchronize()?;
        let start = Instant::now();
        for _ in 0..5 {
            std::hint::black_box(run(input)?);
        }
        input.device().synchronize()?;
        println!(
            "CUDA_DECODER_STAGE {}",
            serde_json::json!({"name":name,"input_shape":input.dims(),"output_shape":output.dims(),"group":group,"seconds_per_call":start.elapsed().as_secs_f64()/5.0,"scope":"synchronized_boundary_not_exclusive_kernel"})
        );
    }
    if let Some(root) = export {
        let tensors = std::collections::HashMap::from([
            (
                "input".to_owned(),
                input.to_device(&Device::Cpu)?.contiguous()?,
            ),
            (
                "output".to_owned(),
                output.to_device(&Device::Cpu)?.contiguous()?,
            ),
        ]);
        candle_core::safetensors::save(&tensors, root.join(format!("{name}.safetensors")))?;
    }
    Ok(output)
}

#[test]
#[ignore = "requires explicit local CUDA model and fixed codes; optional fresh activation directory"]
fn profile_fixed_decoder_boundaries() -> Result<()> {
    let env = |name| std::env::var(name).map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()));
    let bundle =
        crate::stage1_decoder::Stage1DecoderBundle::from_model_root(env("OMNI_CUDA_TRACE_MODEL")?)?;
    let device = Device::new_cuda(0)?;
    let weights = CodecWeightSource::from_weight_path(bundle.weights_path(), DType::F32, &device)?;
    let model = Stage1Model::load(&weights, bundle.model_config())?;
    let codes: Codes = serde_json::from_slice(&fs::read(env("OMNI_CUDA_DECODER_PROFILE_CODES")?)?)?;
    let tokens = Tensor::from_vec(codes.values, (1, codes.codebooks, codes.frames), &device)?;
    let raw = model.decode_tensor(&tokens)?;
    let trace = model.decode_tensor_with_trace(&tokens)?;
    assert_eq!(values(&raw)?, values(&trace.raw_waveform)?);
    let export = std::env::var_os("OMNI_CUDA_DECODER_ACTIVATIONS").map(std::path::PathBuf::from);
    if let Some(root) = &export {
        assert!(!root.exists(), "fresh activation directory");
        fs::create_dir_all(root)?;
    }
    let mut x = measure_stage("conv1", &trace.decoder_input, None, |x| {
        Ok(x.apply(&model.decoder.conv1)?)
    })?;
    for (index, block) in model.decoder.blocks.iter().enumerate() {
        let whole = measure_stage(&format!("block-{index}"), &x, None, |x| Ok(x.apply(block)?))?;
        let y = measure_stage(&format!("block-{index}-snake1"), &x, None, |x| {
            Ok(x.apply(&block.snake1)?)
        })?;
        x = measure_stage(&format!("block-{index}-deconv"), &y, None, |x| {
            Ok(x.apply(&block.conv_t1)?)
        })?;
        for (unit_index, unit) in [&block.res_unit1, &block.res_unit2, &block.res_unit3]
            .into_iter()
            .enumerate()
        {
            let base = format!("block-{index}-unit-{unit_index}");
            let unit_out = measure_stage(&base, &x, None, |x| Ok(x.apply(unit)?))?;
            let y = x.apply(&unit.snake1)?;
            let primitive_export = if unit_index == 0 {
                export.as_deref()
            } else {
                None
            };
            let y = measure_stage(&format!("{base}-conv7"), &y, primitive_export, |x| {
                Ok(unit.conv1.forward_decoder_conv7(x)?)
            })?;
            let y = y.apply(&unit.snake2)?;
            let y = measure_stage(&format!("{base}-pointwise"), &y, primitive_export, |x| {
                Ok(unit.conv2.forward_decoder_pointwise(x)?)
            })?;
            assert_eq!(values(&(y + &x)?)?, values(&unit_out)?);
            x = unit_out;
        }
        assert_eq!(values(&x)?, values(&whole)?);
    }
    let x = x.apply(&model.decoder.snake1)?;
    let output = measure_stage("conv2", &x, None, |x| Ok(x.apply(&model.decoder.conv2)?))?;
    assert_eq!(values(&output)?, values(&raw)?);
    Ok(())
}

#[test]
#[ignore = "requires explicit local CUDA model and real fixed codes; serial release"]
fn screen_real_polyphase_deconv() -> Result<()> {
    let env = |name| std::env::var(name).map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()));
    let bundle =
        crate::stage1_decoder::Stage1DecoderBundle::from_model_root(env("OMNI_CUDA_TRACE_MODEL")?)?;
    let device = Device::new_cuda(0)?;
    let weights = CodecWeightSource::from_weight_path(bundle.weights_path(), DType::F32, &device)?;
    let model = Stage1Model::load(&weights, bundle.model_config())?;
    let codes: Codes = serde_json::from_slice(&fs::read(env("OMNI_CUDA_DECODER_PROFILE_CODES")?)?)?;
    let tokens = Tensor::from_vec(codes.values, (1, codes.codebooks, codes.frames), &device)?;
    let trace = model.decode_tensor_with_trace(&tokens)?;
    let mut x = trace.decoder_input.apply(&model.decoder.conv1)?;
    let mut diffs = 0;
    for (index, block) in model.decoder.blocks.iter().enumerate() {
        let y = x.apply(&block.snake1)?;
        diffs += crate::codec_layers::cuda_deconv_probe::compare(index, &block.conv_t1, &y)?;
        x = x.apply(block)?;
    }
    println!(
        "CUDA_PHASE_DECONV_TOTAL {}",
        serde_json::json!({"bit_differences":diffs})
    );
    assert_eq!(
        diffs, 0,
        "only a bit-preserving candidate may enter production"
    );
    Ok(())
}

#[test]
#[ignore = "requires local CUDA model/codes and NVRTC; serial release"]
fn screen_real_ordered_deconv() -> Result<()> {
    let env = |name| std::env::var(name).map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()));
    let bundle =
        crate::stage1_decoder::Stage1DecoderBundle::from_model_root(env("OMNI_CUDA_TRACE_MODEL")?)?;
    let device = Device::new_cuda(0)?;
    let weights = CodecWeightSource::from_weight_path(bundle.weights_path(), DType::F32, &device)?;
    let model = Stage1Model::load(&weights, bundle.model_config())?;
    let codes: Codes = serde_json::from_slice(&fs::read(env("OMNI_CUDA_DECODER_PROFILE_CODES")?)?)?;
    let tokens = Tensor::from_vec(codes.values, (1, codes.codebooks, codes.frames), &device)?;
    let trace = model.decode_tensor_with_trace(&tokens)?;
    let mut x = trace.decoder_input.apply(&model.decoder.conv1)?;
    let mut diffs = 0;
    for (index, block) in model.decoder.blocks.iter().enumerate() {
        let y = x.apply(&block.snake1)?;
        diffs +=
            crate::codec_layers::cuda_deconv_probe::compare_ordered(index, &block.conv_t1, &y)?;
        x = x.apply(block)?;
    }
    assert_eq!(diffs, 0, "ordered kernel must preserve raw primitive bits");
    Ok(())
}

#[test]
#[ignore = "requires local CUDA model and directory of actual codes; serial release"]
fn production_ordered_decoder_keeps_raw_bits() -> Result<()> {
    let env = |name| std::env::var(name).map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()));
    let bundle =
        crate::stage1_decoder::Stage1DecoderBundle::from_model_root(env("OMNI_CUDA_TRACE_MODEL")?)?;
    let device = Device::new_cuda(0)?;
    let weights = CodecWeightSource::from_weight_path(bundle.weights_path(), DType::F32, &device)?;
    let model = Stage1Model::load(&weights, bundle.model_config())?;
    let mut files = fs::read_dir(env("OMNI_CUDA_DECODER_CODES")?)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.sort();
    for path in files {
        let codes: Codes = serde_json::from_slice(&fs::read(&path)?)?;
        let input = Tensor::from_vec(
            codes.values.clone(),
            (1, codes.codebooks, codes.frames),
            &device,
        )?;
        let changed = Tensor::from_vec(
            codes
                .values
                .iter()
                .map(|v| (v + 1) % 1024)
                .collect::<Vec<_>>(),
            input.shape(),
            &device,
        )?;
        for (call, x) in [&input, &changed, &input].into_iter().enumerate() {
            let expected =
                crate::codec_layers::cuda_ordered_deconv::with_eager(|| model.decode_tensor(x))?;
            let actual = model.decode_tensor(x)?;
            assert_eq!(
                values(&expected)?,
                values(&actual)?,
                "raw PCM {} call {call}",
                path.display()
            );
        }
    }
    Ok(())
}
