//! Opt-in real-input test of target-only audio head projection.
use super::*;
use std::{cell::RefCell, time::Instant};

#[derive(Default)]
struct State {
    enabled: bool,
    steps: usize,
    mismatches: usize,
}
thread_local! { static STATE: RefCell<State> = RefCell::new(State::default()); }

pub(super) fn active() -> bool {
    STATE.with_borrow(|s| s.enabled)
}

fn target_rows(hidden: &Tensor, conditional: usize, target: usize) -> Result<Tensor> {
    Ok(Tensor::cat(
        &[
            hidden.i((0..1, conditional - target..conditional, ..))?,
            hidden.i((1..2, 0..target, ..))?,
        ],
        0,
    )?)
}

fn select_logits(logits: &Tensor, conditional: usize, target: usize) -> Result<Tensor> {
    Ok(Tensor::cat(
        &[
            logits.i((0..1, .., conditional - target..conditional, ..))?,
            logits.i((1..2, .., 0..target, ..))?,
        ],
        0,
    )?)
}

pub(super) fn check(
    model: &Stage0Model,
    hidden: &Tensor,
    logits: &Tensor,
    lengths: Option<&[usize]>,
) -> Result<()> {
    if !active() {
        return Ok(());
    }
    let lengths = lengths.expect("real CFG lengths");
    assert_eq!(lengths.len(), 2);
    let (conditional, target) = (lengths[0], lengths[1]);
    let project = |input: &Tensor| -> Result<Tensor> {
        let (batch, seq, _) = input.dims3()?;
        Ok(model
            .audio_heads
            .forward(input)?
            .reshape((batch, seq, model.num_audio_codebook, model.audio_vocab_size))?
            .permute((0, 2, 1, 3))?)
    };
    let run = |fast| -> Result<Tensor> {
        if fast {
            project(&target_rows(hidden, conditional, target)?)
        } else {
            select_logits(&project(hidden)?, conditional, target)
        }
    };
    // The real consumer casts two target slices, without concatenating logits.
    let boundary = |fast| -> Result<(Tensor, Tensor)> {
        let (projected, start) = if fast {
            (project(&target_rows(hidden, conditional, target)?)?, 0)
        } else {
            (project(hidden)?, conditional - target)
        };
        Ok((
            projected
                .i((0..1, .., start..start + target, ..))?
                .to_dtype(DType::F32)?,
            projected
                .i((1..2, .., 0..target, ..))?
                .to_dtype(DType::F32)?,
        ))
    };
    let expected = select_logits(logits, conditional, target)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let actual = run(true)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(expected.len(), actual.len());
    let mismatches = expected
        .iter()
        .zip(&actual)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let maximum = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let step = STATE.with_borrow_mut(|s| {
        let n = s.steps;
        s.steps += 1;
        s.mismatches += mismatches;
        n
    });
    println!(
        "CUDA_TARGET_HEAD_BITS {}",
        serde_json::json!({"step":step,"hidden_shape":hidden.dims(),"conditional":conditional,"target":target,"mismatches":mismatches,"max_abs":maximum,"elements":actual.len()})
    );
    if [0, 15, 31].contains(&step) {
        for group in 0..3 {
            for (slot, fast) in [false, true, true, false].into_iter().enumerate() {
                hidden.device().synchronize()?;
                let start = Instant::now();
                for _ in 0..10 {
                    std::hint::black_box(boundary(fast)?);
                }
                hidden.device().synchronize()?;
                println!(
                    "CUDA_TARGET_HEAD_TIME {}",
                    serde_json::json!({"step":step,"group":group,"slot":slot,"target_only":fast,"seconds_per_call":start.elapsed().as_secs_f64()/10.0,"scope":"head_target_packing_and_scoring_casts_eager"})
                );
            }
        }
    }
    Ok(())
}

struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        STATE.with_borrow_mut(|s| *s = State::default());
    }
}

#[derive(serde::Deserialize)]
struct CachedPrompt {
    dims: (usize, usize),
    tokens: Vec<i64>,
    transcript: String,
    rms: Option<f32>,
}

#[test]
#[ignore = "requires explicit local CUDA model/text and optional frozen prompt; serial release"]
fn real_target_head_projection_bits_and_cost() -> Result<()> {
    let env = |name| std::env::var(name).map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()));
    let pipeline = crate::pipeline::Pipeline::from_options(
        crate::RuntimeOptions::new(env("OMNI_CUDA_TRACE_MODEL")?)
            .with_device(crate::DeviceSpec::Cuda(0)),
    )?;
    let text = fs::read_to_string(env("OMNI_CUDA_TRACE_TEXT")?)?;
    let mut request =
        crate::contracts::GenerationRequest::new_text_only(text.trim()).with_language("zh");
    if let Some(path) = std::env::var_os("OMNI_CUDA_HEAD_PROMPT") {
        let cached: CachedPrompt = serde_json::from_slice(&fs::read(path)?)?;
        request = request.with_voice_clone_prompt(crate::contracts::VoiceClonePrompt {
            ref_audio_tokens: I64Tensor2::new(cached.dims, cached.tokens)?,
            ref_text: cached.transcript,
            ref_rms: cached.rms,
        });
    }
    pipeline.set_seed(7330414270409262323)?;
    STATE.with_borrow_mut(|s| s.enabled = true);
    let _guard = Guard;
    let audio = pipeline.generate(&request)?;
    assert!(
        audio
            .iter()
            .all(|a| !a.samples.is_empty() && a.samples.iter().all(|v| v.is_finite()))
    );
    assert_eq!(STATE.with_borrow(|s| s.steps), 32);
    assert_eq!(
        STATE.with_borrow(|s| s.mismatches),
        0,
        "candidate must preserve every consumed logit"
    );
    Ok(())
}
