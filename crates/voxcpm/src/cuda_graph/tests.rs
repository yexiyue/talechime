//! Local CUDA qualification. These tests never download weights.
use super::*;
use crate::weights::Weights;
use std::path::PathBuf;

fn acoustic() -> Result<(Device, Acoustic)> {
    let root = PathBuf::from(std::env::var("VOXCPM_TEST_MODELS").expect("set VOXCPM_TEST_MODELS"));
    let device = Device::new_cuda(0)?;
    let mut base = Weights::open(&root.join("VoxCPM2-BaseLM-Q8_0.gguf"), &device, DType::F32)?;
    let rope = base.tensor("rope_factors_short.weight")?;
    let mut weights = Weights::open(&root.join("VoxCPM2-Acoustic-F16.gguf"), &device, DType::F16)?;
    Ok((device, Acoustic::load(&mut weights, &rope)?))
}

fn values(shape: &[usize], seed: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let n = shape.iter().product();
    let data: Vec<_> = (0..n)
        .map(|i| ((i * 17 + seed * 13) as f32 * 0.17).sin() * 0.3)
        .collect();
    Tensor::from_vec(data, shape, device)?.to_dtype(dtype)
}

fn identical(actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let bits = |t: &Tensor| -> Result<Vec<u32>> {
        Ok(t.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    assert_eq!(bits(actual)?, bits(expected)?);
    Ok(())
}

#[test]
#[ignore = "requires local GGUF weights and CUDA; run serially"]
fn real_model_replay_noise_schedule_and_lru() -> Result<()> {
    let (device, mut acoustic) = acoustic()?;
    let mut graphs = CfmGraphs::new(&device).expect("enable VOXCPM_CUDA_GRAPH");
    let mu = values(&[1, 2, 1024], 1, DType::F16, &device)?;
    let prefix = values(&[1, 64, 4], 2, DType::F16, &device)?;
    // Revisit schedules both before and after eviction. Different guidance bits
    // also specialize the graph without changing the acoustic schedule.
    for (steps, cfg) in [
        (10, 2.0),
        (12, 2.0),
        (10, 2.0),
        (10, 1.5),
        (6, 2.0),
        (8, 2.0),
        (10, 2.0),
    ] {
        for seed in 0..8 {
            let noise = values(&[1, 64, 4], seed, DType::F32, &device)?;
            let expected = acoustic.sample(&mu, &prefix, &noise, steps, cfg, &|| false)?;
            let actual =
                graphs.sample(&mut acoustic, &mu, &prefix, &noise, steps, cfg, &|| false)?;
            identical(&actual, &expected)?;
            assert!(graphs.active(), "unexpected graph fallback");
            assert!(graphs.captured.len() <= MAX_CAPTURED);
        }
    }
    // E1 was not retained as a runtime path; still qualify the reusable recorder.
    let input = values(&[1, 4, 64], 0, DType::F16, &device)?;
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let encoder = {
        let _guard = cuda.enable_cuda_graph_htod_cache();
        let warmup = acoustic.encode(&input, &|| false)?;
        Replay::record(cuda, &[&input], &warmup, vec![], |args| {
            acoustic.encode(&args[0], &|| false)
        })?
    };
    for seed in 0..8 {
        let input = values(&[1, 4, 64], seed, DType::F16, &device)?;
        let expected = acoustic.encode(&input, &|| false)?;
        let actual = encoder.launch(&[&input])?;
        identical(&actual, &expected)?;
    }
    let noise = values(&[1, 64, 4], 41, DType::F32, &device)?;
    // Cancellation after a real replay must not disable later requests.
    let polls = std::cell::Cell::new(0);
    let cancel = || {
        polls.set(polls.get() + 1);
        polls.get() > 1
    };
    assert!(
        graphs
            .sample(&mut acoustic, &mu, &prefix, &noise, 10, 2.0, &cancel)
            .is_err()
    );
    assert!(graphs.active());
    let expected = acoustic.sample(&mu, &prefix, &noise, 10, 2.0, &|| false)?;
    let fallback = graphs.finish(
        Err(candle_core::Error::Msg(
            "injected synchronous graph rejection".into(),
        )),
        &|| false,
        || acoustic.sample(&mu, &prefix, &noise, 10, 2.0, &|| false),
    )?;
    identical(&fallback, &expected)?;
    assert!(graphs.disabled);
    identical(
        &graphs.sample(&mut acoustic, &mu, &prefix, &noise, 10, 2.0, &|| false)?,
        &expected,
    )?;
    Ok(())
}

#[test]
#[ignore = "requires CUDA GPU; run serially"]
fn views_capture_cleanup_and_fallback() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let _guard = cuda.enable_cuda_graph_htod_cache();
    let input = values(&[3, 4], 0, DType::F32, &device)?;
    // Contiguous narrow retains a nonzero offset into a larger allocation.
    let view = input.narrow(0, 1, 1)?;
    let warmup = view.affine(2.0, 1.0)?;
    let replay = Replay::record(cuda, &[&view], &warmup, vec![], |args| {
        args[0].affine(2.0, 1.0)
    })?;
    identical(&replay.launch(&[&view])?, &warmup)?;
    identical(
        &replay.launch(&[&input.narrow(0, 2, 1)?])?,
        &input.narrow(0, 2, 1)?.affine(2.0, 1.0)?,
    )?;

    let failure = Replay::record(cuda, &[&view], &warmup, vec![], |args| {
        let _temporary = args[0].sqr()?;
        candle_core::bail!("injected recording failure")
    });
    assert!(failure.is_err());
    assert!(!crate::cuda_graph::replay::is_capturing(
        &cuda.cuda_stream()
    )?);
    identical(&view.affine(2.0, 1.0)?, &warmup)?;

    let mut graphs = CfmGraphs::new(&device).expect("enable graphs");
    let output = graphs.finish(
        Err(candle_core::Error::Msg("injected launch failure".into())),
        &|| false,
        || Ok(warmup.clone()),
    )?;
    identical(&output, &warmup)?;
    assert!(graphs.disabled && !graphs.active());
    assert!(
        graphs
            .finish(Ok(warmup), &|| true, || unreachable!())
            .is_err()
    );
    Ok(())
}
