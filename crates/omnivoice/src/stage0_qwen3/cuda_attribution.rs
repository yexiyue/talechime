//! Opt-in real CUDA attention experiments, excluded from production.
use super::*;
use candle_core::Device;
use std::{cell::Cell, collections::HashMap, fs, path::PathBuf};

thread_local! { static CAPTURE: Cell<Option<(usize,usize)>> = const {Cell::new(None)}; }

pub(super) fn active() -> bool {
    CAPTURE.with(Cell::get).is_some()
}

fn root() -> Result<PathBuf> {
    std::env::var_os("OMNI_CUDA_ATTENTION_ROOT")
        .map(PathBuf::from)
        .ok_or_else(|| crate::OmniVoiceError::InvalidRequest("set OMNI_CUDA_ATTENTION_ROOT".into()))
}

pub(super) fn capture(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<()> {
    let case = CAPTURE.with(|state| {
        let (index, layers) = state.get()?;
        state.set(Some((index + 1, layers)));
        let (step, layer) = (index / layers, index % layers);
        ([0, 15, 31].contains(&step) && [0, (layers - 1) / 2, layers - 1].contains(&layer))
            .then_some((step, layer))
    });
    if let Some((step, layer)) = case {
        let mut entries = vec![("q", q), ("k", k), ("v", v)];
        if let Some(mask) = mask {
            entries.push(("mask", mask));
        }
        let mut metadata = serde_json::Map::new();
        let mut tensors = HashMap::new();
        for (name, tensor) in entries {
            metadata.insert(name.into(),serde_json::json!({"dims":tensor.dims(),"strides":tensor.stride(),"offset":tensor.layout().start_offset(),"dtype":format!("{:?}",tensor.dtype())}));
            tensors.insert(
                name.to_owned(),
                tensor.to_device(&Device::Cpu)?.contiguous()?,
            );
        }
        let path = root()?.join(format!("step-{step}-layer-{layer}.safetensors"));
        assert!(!path.exists(), "fresh capture required");
        candle_core::safetensors::save(&tensors, &path)?;
        fs::write(
            path.with_extension("json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
    }
    Ok(())
}

struct CaptureGuard;
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        CAPTURE.with(|state| state.set(None));
    }
}

#[test]
#[ignore = "requires local CUDA model/text and fresh output directory"]
fn capture_real_attention() -> Result<()> {
    let directory = root()?;
    assert!(!directory.exists(), "fresh capture directory required");
    fs::create_dir_all(directory)?;
    let model = std::env::var("OMNI_CUDA_TRACE_MODEL")
        .map_err(|e| crate::OmniVoiceError::InvalidRequest(e.to_string()))?;
    let text = fs::read_to_string(
        std::env::var("OMNI_CUDA_TRACE_TEXT")
            .map_err(|e| crate::OmniVoiceError::InvalidRequest(e.to_string()))?,
    )?;
    let layers = crate::stage0_model::Stage0Config::from_model_root(&model)?
        .llm_config
        .num_hidden_layers;
    assert!(layers > 0);
    let pipeline = crate::pipeline::Pipeline::from_options(
        crate::RuntimeOptions::new(model).with_device(crate::DeviceSpec::Cuda(0)),
    )?;
    use std::hash::{Hash, Hasher};
    let mut seed = std::hash::DefaultHasher::new();
    42u64.hash(&mut seed);
    0u64.hash(&mut seed);
    0u8.hash(&mut seed);
    pipeline.set_seed(seed.finish())?;
    CAPTURE.with(|state| state.set(Some((0, layers))));
    let _guard = CaptureGuard;
    pipeline.generate(
        &crate::contracts::GenerationRequest::new_text_only(text.trim()).with_language("zh"),
    )?;
    assert_eq!(
        CAPTURE.with(Cell::get),
        Some((32 * layers, layers)),
        "capture exactly one unchunked request"
    );
    Ok(())
}

fn grouped(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    Ok(cuda_grouped_attention(q, k, v, mask, scale)?
        .expect("captured input must use the production specialization"))
}

#[test]
#[ignore = "requires real CUDA device; no model weights"]
fn grouped_cuda_shape_and_mask_bits() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for seq in [8, 31, 63, 117, 128, 258, 256, 512] {
        let q = Tensor::arange(0u32, (2 * 16 * seq * 128) as u32, &device)?
            .to_dtype(DType::F32)?
            .sin()?
            .to_dtype(DType::F16)?
            .reshape((2, 16, seq, 128))?;
        let k = q.narrow(1, 0, 8)?.contiguous()?;
        let v = k.transpose(1, 2)?.contiguous()?.transpose(1, 2)?;
        let mask = Tensor::from_vec(
            (0..2 * seq)
                .map(|i| {
                    if i % seq == seq - 1 {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
                .collect::<Vec<_>>(),
            (2, 1, 1, seq),
            &device,
        )?
        .to_dtype(DType::F16)?;
        for mask in [None, Some(&mask)] {
            let old = attention_manual(
                &q,
                &candle_transformers::utils::repeat_kv(k.clone(), 2)?,
                &candle_transformers::utils::repeat_kv(v.clone(), 2)?,
                mask,
                1.0 / (128f64).sqrt(),
            )?;
            let new = grouped(&q, &k, &v, mask, 1.0 / (128f64).sqrt())?;
            let bits = |t: Tensor| -> Result<Vec<u32>> {
                Ok(t.to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(bits(old)?, bits(new)?, "seq={seq}");
        }
        assert!(
            cuda_grouped_attention(
                &q.to_dtype(DType::F32)?,
                &k.to_dtype(DType::F32)?,
                &v.to_dtype(DType::F32)?,
                None,
                1.
            )?
            .is_none()
        );
    }
    for seq in [7, 513] {
        let q = Tensor::zeros((1, 16, seq, 128), DType::F16, &device)?;
        let k = Tensor::zeros((1, 8, seq, 128), DType::F16, &device)?;
        assert!(cuda_grouped_attention(&q, &k, &k, None, 1.)?.is_none());
    }
    Ok(())
}

#[test]
#[ignore = "requires explicit local CUDA model and text; verifies cancellation reuse"]
fn grouped_generation_recovers_after_cancellation() -> Result<()> {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let model = std::env::var("OMNI_CUDA_TRACE_MODEL")
        .map_err(|e| crate::OmniVoiceError::InvalidRequest(e.to_string()))?;
    let text = fs::read_to_string(
        std::env::var("OMNI_CUDA_TRACE_TEXT")
            .map_err(|e| crate::OmniVoiceError::InvalidRequest(e.to_string()))?,
    )?;
    let pipeline = crate::pipeline::Pipeline::from_options(
        crate::RuntimeOptions::new(model).with_device(crate::DeviceSpec::Cuda(0)),
    )?;
    let request =
        crate::contracts::GenerationRequest::new_text_only(text.trim()).with_language("zh");
    pipeline.set_seed(42)?;
    let expected = pipeline.generate(&request)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    pipeline.set_cancellation_probe(Some(crate::stage0_model::CancellationProbe(Arc::new(
        move || counter.fetch_add(1, Ordering::Relaxed) >= 5,
    ))));
    pipeline.set_seed(42)?;
    let interrupted = pipeline.generate(&request);
    pipeline.set_cancellation_probe(None);
    assert!(
        matches!(interrupted, Err(crate::OmniVoiceError::InvalidRequest(message)) if message == "inference cancelled")
    );
    assert!(calls.load(Ordering::Relaxed) > 5);
    pipeline.set_seed(42)?;
    let reused = pipeline.generate(&request)?;
    assert_eq!(expected.len(), reused.len());
    for (old, new) in expected.iter().zip(&reused) {
        assert_eq!(old.sample_rate, new.sample_rate);
        assert_eq!(
            old.samples.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            new.samples.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires real captured CUDA Q/K/V; frozen release only"]
fn replay_real_grouped_attention() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let mut files = fs::read_dir(root()?)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|p| p.extension().is_some_and(|e| e == "safetensors"));
    files.sort();
    assert!(!files.is_empty());
    for path in files {
        let layout: serde_json::Value =
            serde_json::from_slice(&fs::read(path.with_extension("json"))?)?;
        let tensors = candle_core::safetensors::load(&path, &device)?;
        let (q, k) = (&tensors["q"], &tensors["k"]);
        let v = tensors["v"]
            .transpose(1, 2)?
            .contiguous()?
            .transpose(1, 2)?;
        assert_eq!(
            serde_json::json!(v.stride()),
            layout["v"]["strides"],
            "reconstruct production V view"
        );
        assert_eq!(serde_json::json!(q.stride()), layout["q"]["strides"]);
        assert_eq!(serde_json::json!(k.stride()), layout["k"]["strides"]);
        let mask = tensors.get("mask");
        if let Some(mask) = mask {
            assert_eq!(
                serde_json::json!(mask.stride()),
                layout["mask"]["strides"],
                "preserve mask layout"
            );
        }
        let (batch, heads, seq, dim) = q.dims4()?;
        let repeats = heads / k.dim(1)?;
        let scale = 1.0 / (dim as f64).sqrt();
        let run = |fast: bool| -> Result<Tensor> {
            let context = if fast {
                grouped(q, k, &v, mask, scale)?
            } else {
                let k = candle_transformers::utils::repeat_kv(k.clone(), repeats)?;
                let v = candle_transformers::utils::repeat_kv(v.clone(), repeats)?;
                attention_manual(q, &k, &v, mask, scale)?
            };
            Ok(context
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, seq, heads * dim))?)
        };
        let old = run(false)?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let new = run(true)?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mismatches = old
            .iter()
            .zip(&new)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let maximum = old
            .iter()
            .zip(&new)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        for group in 0..5 {
            for (slot, fast) in [false, true, true, false].into_iter().enumerate() {
                device.synchronize()?;
                let start = std::time::Instant::now();
                for _ in 0..10 {
                    std::hint::black_box(run(fast)?);
                }
                device.synchronize()?;
                println!(
                    "CUDA_REAL_GQA {}",
                    serde_json::json!({"case":path.file_stem().unwrap().to_string_lossy(),"q_shape":q.dims(),"bit_mismatches":mismatches,"max_abs":maximum,"group":group,"slot":slot,"candidate":fast,"seconds_per_call":start.elapsed().as_secs_f64()/10.0,"scope":"attention_and_output_layout_without_o_proj"})
                );
            }
        }
    }
    Ok(())
}
