//! Test-only full-forward feasibility probe. No production routing or switches.
use super::*;
use std::cell::RefCell;
use std::time::Instant;

use super::cuda_forward_graph::replay;

#[derive(Default)]
struct State {
    enabled: bool,
    force_eager: bool,
    graph_records: usize,
    step: usize,
    graph: Option<replay::Replay>,
}
thread_local! { static STATE: RefCell<State> = RefCell::new(State::default()); }

pub(super) fn active() -> bool {
    STATE.with_borrow(|s| s.enabled || s.force_eager)
}

pub(super) fn recorded() {
    STATE.with_borrow_mut(|s| s.graph_records += 1);
}

fn forward(
    model: &Stage0Model,
    inputs: &[Tensor],
    lengths: &[usize],
) -> candle_core::Result<Tensor> {
    model
        .forward_with_text_cache(
            &inputs[0],
            &inputs[1],
            &inputs[2],
            &BTreeSet::new(),
            Some(&inputs[3]),
            Some(lengths),
        )
        .map(|out| out.logits)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))
}

pub(super) fn check(
    model: &Stage0Model,
    inputs: [&Tensor; 4],
    lengths: &[usize],
    expected: &Tensor,
) -> Result<()> {
    STATE.with_borrow_mut(|state| -> Result<()> {
        if !state.enabled { return Ok(()); }
        let Device::Cuda(cuda) = expected.device() else { unreachable!() };
        let _guard = cuda.enable_cuda_graph_htod_cache();
        if state.graph.is_none() {
            let start = Instant::now();
            let warm = forward(model, &inputs.iter().map(|t| (*t).clone()).collect::<Vec<_>>(), lengths)?;
            let graph = replay::Replay::record(cuda, &inputs, &warm, vec![], |args| forward(model,args,lengths))?;
            assert!(!replay::is_capturing(cuda.cuda_stream().as_ref())?);
            expected.device().synchronize()?;
            println!("CUDA_FORWARD_GRAPH_RECORD {}", serde_json::json!({"seconds":start.elapsed().as_secs_f64(),"input_shape":inputs[0].dims(),"output_shape":expected.dims(),"scope":"extra_warmup_record_instantiate_sync"}));
            state.graph=Some(graph);
        }
        let graph=state.graph.as_ref().unwrap();
        let actual=graph.launch(&inputs)?;
        let actual=actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let expected_values=expected.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mismatches=actual.iter().zip(&expected_values).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
        let maximum=actual.iter().zip(&expected_values).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
        println!("CUDA_FORWARD_GRAPH_BITS {}",serde_json::json!({"step":state.step,"mismatches":mismatches,"max_abs":maximum,"elements":actual.len()}));
        assert_eq!(mismatches,0,"graph must preserve all logits");
        if [0,15,31].contains(&state.step) {
            for group in 0..3 {
                for (slot, fast) in [false,true,true,false].into_iter().enumerate() {
                    expected.device().synchronize()?;
                    let start=Instant::now();
                    for _ in 0..3 {
                        if fast { std::hint::black_box(graph.launch(&inputs)?); }
                        else { std::hint::black_box(forward(model,&inputs.iter().map(|t|(*t).clone()).collect::<Vec<_>>(),lengths)?); }
                    }
                    expected.device().synchronize()?;
                    println!("CUDA_FORWARD_GRAPH_TIME {}",serde_json::json!({"step":state.step,"group":group,"slot":slot,"graph":fast,"seconds_per_call":start.elapsed().as_secs_f64()/3.0}));
                }
            }
        }
        state.step+=1;
        Ok(())
    })
}

struct Guard;
impl Drop for Guard {
    fn drop(&mut self) {
        STATE.with_borrow_mut(|state| *state = State::default());
    }
}

#[test]
#[ignore = "requires explicit local CUDA model/text; run serial release"]
fn real_forward_graph_matches_every_step() -> Result<()> {
    let model = std::env::var("OMNI_CUDA_TRACE_MODEL")
        .map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()))?;
    let text = fs::read_to_string(
        std::env::var("OMNI_CUDA_TRACE_TEXT")
            .map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()))?,
    )?;
    let pipeline = crate::pipeline::Pipeline::from_options(
        crate::RuntimeOptions::new(model).with_device(crate::DeviceSpec::Cuda(0)),
    )?;
    pipeline.set_seed(7330414270409262323)?;
    STATE.with_borrow_mut(|s| s.enabled = true);
    // The graph's captured weights belong to pipeline. Drop graph before pipeline,
    // including unwinding after any capture or numerical failure.
    let _guard = Guard;
    let audio = pipeline.generate(
        &crate::contracts::GenerationRequest::new_text_only(text.trim()).with_language("zh"),
    )?;
    assert_eq!(
        STATE.with_borrow(|s| s.step),
        32,
        "use a single unchunked request"
    );
    assert!(
        audio
            .iter()
            .all(|a| !a.samples.is_empty() && a.samples.iter().all(|v| v.is_finite()))
    );
    Ok(())
}

#[test]
#[ignore = "requires explicit local CUDA model and two texts; serial release"]
fn request_graph_matches_eager_and_recovers_after_cancel() -> Result<()> {
    let model = std::env::var("OMNI_CUDA_TRACE_MODEL")
        .map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()))?;
    let texts = ["OMNI_CUDA_TRACE_TEXT", "OMNI_CUDA_TRACE_TEXT_ALT"]
        .map(|name| {
            let path =
                std::env::var(name).map_err(|e| OmniVoiceError::InvalidRequest(e.to_string()))?;
            fs::read_to_string(path).map_err(Into::into)
        })
        .into_iter()
        .collect::<Result<Vec<String>>>()?;
    assert_ne!(texts[0], texts[1], "use distinct request shapes");
    let pipeline = crate::pipeline::Pipeline::from_options(
        crate::RuntimeOptions::new(model).with_device(crate::DeviceSpec::Cuda(0)),
    )?;
    let _guard = Guard;
    let generate = |index: usize| -> Result<Vec<u32>> {
        pipeline.set_seed(7330414270409262323)?;
        Ok(pipeline
            .generate(
                &crate::contracts::GenerationRequest::new_text_only(texts[index].trim())
                    .with_language("zh"),
            )?
            .iter()
            .flat_map(|a| a.samples.iter().map(|v| v.to_bits()))
            .collect())
    };
    STATE.with_borrow_mut(|s| s.force_eager = true);
    let eager = [generate(0)?, generate(1)?];
    STATE.with_borrow_mut(|s| s.force_eager = false);
    for index in [0, 1, 0] {
        assert_eq!(generate(index)?, eager[index], "request A/B/A PCM differs");
    }
    pipeline.set_cancellation_probe(Some(CancellationProbe(Arc::new(move || {
        // Frontend cancellation checks can change independently of denoise.
        // Cancel only after this request has actually recorded its first step.
        STATE.with_borrow(|s| s.graph_records >= 4)
    }))));
    let error = generate(0).unwrap_err();
    assert!(
        matches!(error, OmniVoiceError::InvalidRequest(ref message) if message == "inference cancelled")
    );
    pipeline.set_cancellation_probe(None);
    assert_eq!(generate(0)?, eager[0], "cancelled graph leaked state");
    assert_eq!(
        STATE.with_borrow(|s| s.graph_records),
        5,
        "test must execute production graphs rather than fallback"
    );
    Ok(())
}
