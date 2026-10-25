//! Opt-in test-only capture/replay of actual CUDA scoring inputs.
use super::*;
use std::{cell::Cell, collections::HashMap, fs, path::PathBuf};

thread_local! { static STEP: Cell<Option<usize>> = const {Cell::new(None)}; }

pub(super) fn active() -> bool {
    STEP.with(Cell::get).is_some()
}

fn root() -> Result<PathBuf> {
    std::env::var_os("OMNI_CUDA_TRACE_ROOT")
        .map(PathBuf::from)
        .ok_or_else(|| OmniVoiceError::InvalidRequest("set OMNI_CUDA_TRACE_ROOT".into()))
}

pub(super) fn capture(c: &Tensor, u: &Tensor, probs: &Tensor) -> Result<()> {
    let step = STEP.with(|state| {
        let current = state.get()?;
        state.set(Some(current + 1));
        [0, 15, 31].contains(&current).then_some(current)
    });
    if let Some(step) = step {
        let layout: serde_json::Map<_,_> = [("c_logits",c),("u_logits",u),("log_probs",probs)].into_iter()
            .map(|(name,t)| (name.into(),serde_json::json!({"dims":t.dims(),"strides":t.stride(),"offset":t.layout().start_offset(),"dtype":format!("{:?}",t.dtype())}))).collect();
        let tensors: HashMap<_, _> = [("c_logits", c), ("u_logits", u), ("log_probs", probs)]
            .into_iter()
            .map(|(name, t)| Ok((name.to_owned(), t.to_device(&Device::Cpu)?.contiguous()?)))
            .collect::<Result<_>>()?;
        let path = root()?.join(format!("step-{step}.safetensors"));
        assert!(!path.exists(), "refuse to overwrite capture");
        candle_core::safetensors::save(&tensors, &path)?;
        fs::write(
            root()?.join(format!("step-{step}-layout.json")),
            serde_json::to_vec_pretty(&layout)?,
        )?;
    }
    Ok(())
}

struct CaptureGuard;
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        STEP.with(|state| state.set(None));
    }
}

#[test]
#[ignore = "requires explicit local CUDA model, text, and fresh capture directory"]
fn capture_real_scoring_inputs() -> Result<()> {
    let directory = root()?;
    assert!(!directory.exists(), "capture directory must be fresh");
    fs::create_dir_all(&directory)?;
    let model = std::env::var("OMNI_CUDA_TRACE_MODEL")
        .map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()))?;
    let text = fs::read_to_string(
        std::env::var("OMNI_CUDA_TRACE_TEXT")
            .map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()))?,
    )?;
    let pipeline = crate::pipeline::Pipeline::from_options(
        crate::RuntimeOptions::new(model).with_device(crate::DeviceSpec::Cuda(0)),
    )?;
    use std::hash::{Hash, Hasher};
    let mut seed = std::hash::DefaultHasher::new();
    42u64.hash(&mut seed);
    0u64.hash(&mut seed);
    0u8.hash(&mut seed);
    pipeline.set_seed(seed.finish())?;
    STEP.with(|state| state.set(Some(0)));
    let _guard = CaptureGuard;
    let audio = pipeline.generate(
        &crate::contracts::GenerationRequest::new_text_only(text.trim()).with_language("zh"),
    )?;
    assert_eq!(
        STEP.with(Cell::get),
        Some(32),
        "use one unchunked 32-step input"
    );
    assert!(
        audio
            .iter()
            .all(|a| !a.samples.is_empty() && a.samples.iter().all(|v| v.is_finite()))
    );
    Ok(())
}

#[test]
#[ignore = "requires previously captured real CUDA scoring inputs; use release"]
fn replay_real_confidence_boundary() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for step in [0, 15, 31] {
        let tensors = candle_core::safetensors::load(
            root()?.join(format!("step-{step}.safetensors")),
            &device,
        )?;
        let values = &tensors["log_probs"];
        let run = |fast: bool| -> Result<(Tensor, Tensor)> {
            let tokens = values.argmax(candle_core::D::Minus1)?;
            if fast {
                let confidence = values
                    .gather(
                        &tokens.unsqueeze(candle_core::D::Minus1)?,
                        candle_core::D::Minus1,
                    )?
                    .squeeze(candle_core::D::Minus1)?;
                Ok((tokens.to_dtype(DType::I64)?, confidence))
            } else {
                let tokens = tokens.to_dtype(DType::I64)?;
                Ok((tokens, values.max(candle_core::D::Minus1)?))
            }
        };
        let original = run(false)?;
        let candidate = run(true)?;
        assert_eq!(
            original.0.flatten_all()?.to_vec1::<i64>()?,
            candidate.0.flatten_all()?.to_vec1::<i64>()?
        );
        let bits = |t: Tensor| -> Result<Vec<u32>> {
            Ok(t.flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        assert_eq!(bits(original.1)?, bits(candidate.1)?);
        for group in 0..5 {
            for (slot, fast) in [false, true, true, false].into_iter().enumerate() {
                device.synchronize()?;
                let start = std::time::Instant::now();
                for _ in 0..20 {
                    std::hint::black_box(run(fast)?);
                }
                device.synchronize()?;
                println!(
                    "CUDA_REAL_CONFIDENCE {}",
                    serde_json::json!({"step":step,"shape":values.dims(),"group":group,"slot":slot,"candidate":fast,"seconds_per_call":start.elapsed().as_secs_f64()/20.0})
                );
            }
        }
    }
    Ok(())
}
