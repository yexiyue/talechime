//! E3 qualification of the production dec_proj convolution, not full AudioVAE replay.
//!
//! Run explicitly with local VOXCPM_TEST_MODELS and --ignored --nocapture,
//! serially with other GPU work stopped. No weights are downloaded.
//!
//! Naively recording Codec::decode is unsafe: Conv histories and Upsample tails
//! are replaced by host-side assignments that do not run during graph replay.
//! They can also retain allocations made inside capture past end_capture, which
//! violates Replay's balanced-allocation requirement. Full decode needs explicit
//! persistent state buffers and state updates recorded into the graph first.

use super::decoder::DecoderGraph;
use crate::cuda_graph::replay::Replay;
use crate::{codec::Codec, weights::Weights};
use candle_core::{DType, Device, Result, Tensor};
use std::path::PathBuf;

fn frame(seed: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let values: Vec<f32> = (0..64 * 4)
        .map(|i| ((i * 17 + seed * 29) as f32 * 0.13).sin() * 0.25)
        .collect();
    Tensor::from_vec(values, (1, 64, 4), device)?.to_dtype(dtype)
}

fn bits(tensor: &Tensor) -> Result<Vec<u32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

#[test]
#[ignore = "requires local Acoustic-F16 GGUF and CUDA; run serially"]
fn full_decoder_replay_preserves_state_and_reset() -> Result<()> {
    decoder_state_gate(false)
}

#[test]
#[ignore = "requires local original AudioVAE F32 weights and CUDA; run serially"]
fn original_f32_decoder_replay_preserves_state_and_reset() -> Result<()> {
    decoder_state_gate(true)
}

fn decoder_state_gate(original: bool) -> Result<()> {
    let variable = if original {
        "VOXCPM_TEST_ORIGINAL_MODELS"
    } else {
        "VOXCPM_TEST_MODELS"
    };
    let root = PathBuf::from(std::env::var_os(variable).expect("local model directory required"));
    let device = Device::new_cuda(0)?;
    let dtype = if original { DType::F32 } else { DType::F16 };
    let mut weights = if original {
        Weights::native(&root.join("audiovae.pth"), &device, dtype)?
    } else {
        Weights::open(&root.join("VoxCPM2-Acoustic-F16.gguf"), &device, dtype)?
    };
    let mut eager = Codec::load(&mut weights)?;
    let mut captured = Codec::load(&mut weights)?;
    let mut graph = DecoderGraph::default();
    for request in 0..3 {
        eager.reset();
        captured.reset();
        graph.reset();
        for seed in 0..8 {
            let input = frame(seed + request * 8, &device, dtype)?;
            let expected = bits(&eager.decode(&input, &|| false)?)?;
            let output = graph.decode(&mut captured, &input, &|| false)?;
            assert_eq!(bits(&output)?, expected, "request {request} frame {seed}");
            assert!(graph.active(), "decoder capture fell back");
        }
    }
    // Cancellation before launch must not consume a decoder frame.
    let input = frame(99, &device, dtype)?;
    assert!(graph.decode(&mut captured, &input, &|| true).is_err());
    let expected = bits(&eager.decode(&input, &|| false)?)?;
    assert_eq!(
        bits(&graph.decode(&mut captured, &input, &|| false)?)?,
        expected
    );
    // Cancellation after replay discards PCM but may have advanced device state.
    // A new request must reset that state before any output is delivered.
    let polls = std::cell::Cell::new(0);
    let cancel = || {
        polls.set(polls.get() + 1);
        polls.get() > 1
    };
    assert!(graph.decode(&mut captured, &input, &cancel).is_err());
    assert_eq!(polls.get(), 2);
    assert!(graph.active(), "cancellation disabled a reusable graph");
    eager.reset();
    captured.reset();
    graph.reset();
    for seed in 100..108 {
        let input = frame(seed, &device, dtype)?;
        let expected = bits(&eager.decode(&input, &|| false)?)?;
        assert_eq!(
            bits(&graph.decode(&mut captured, &input, &|| false)?)?,
            expected
        );
        assert!(graph.active());
    }
    Ok(())
}

#[test]
#[ignore = "E3 feasibility only: requires local Acoustic-F16 GGUF and CUDA; run serially"]
fn decoder_conv1d_capture_feasibility() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("VOXCPM_TEST_MODELS")
            .expect("set VOXCPM_TEST_MODELS to local GGUF directory"),
    );
    let device = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &device else {
        unreachable!()
    };
    let mut weights = Weights::open(&root.join("VoxCPM2-Acoustic-F16.gguf"), &device, DType::F16)?;
    // Validate the real Codec's complete weight set without mutating its state.
    let _codec = Codec::load(&mut weights)?;
    let weight = weights.tensor("audio_vae.decoder.model.1")?;
    let bias = weights.tensor("audio_vae.decoder.model.1.bias")?;
    let (output_channels, input_channels, kernel) = weight.dims3()?;
    assert_eq!(input_channels, 64);
    assert_eq!(kernel, 1, "dec_proj must preserve the four-sample frame");
    assert_eq!(bias.dims(), &[output_channels]);
    let bias = bias.reshape((1, output_channels, 1))?;

    // Match Conv::forward for dec_proj: no history or padding, stride/dilation
    // one, and groups one. This exercises the dense CUDA convolution actually
    // used by decode, without claiming cuDNN specifically or full decoder safety.
    let convolution = |input: &Tensor| -> Result<Tensor> {
        input.conv1d(&weight, 0, 1, 1, 1)?.broadcast_add(&bias)
    };
    // Populate HTOD parameter caches and lazy convolution resources before
    // capture. Parameters are allocated outside capture and retained.
    let _guard = cuda.enable_cuda_graph_htod_cache();
    let initial = frame(0, &device, DType::F16)?;
    let warmup = convolution(&initial)?;
    assert_eq!(warmup.dims(), &[1, output_channels, 4]);
    // Reading the warmup also surfaces asynchronous eager convolution errors.
    let _ = bits(&warmup)?;
    let replay = match Replay::record(
        cuda,
        &[&initial],
        &warmup,
        vec![weight.clone(), bias.clone()],
        |inputs| convolution(&inputs[0]),
    ) {
        Ok(replay) => replay,
        Err(error) => {
            // Device/backend capture support is an experimental outcome, not a
            // universal assertion. Keep rejection visible as a failing ignored
            // qualification test rather than silently reporting a successful skip.
            candle_core::bail!(
                "E3 primitive capture unavailable on this configuration: {error}; \
                 full AudioVAE capture remains unqualified"
            );
        }
    };
    eprintln!("E3 dec_proj dense conv1d capture succeeded; checking eight replays");
    for seed in 1..=8 {
        let input = frame(seed, &device, DType::F16)?;
        // Materialize expected values before replay and actual values before the
        // next launch: Replay returns an alias of its reusable output buffer.
        let expected = bits(&convolution(&input)?)?;
        let actual = replay.launch(&[&input])?;
        assert_eq!(actual.dims(), &[1, output_channels, 4]);
        assert_eq!(bits(&actual)?, expected, "E3 replay differs at seed {seed}");
    }
    eprintln!("E3 primitive: 8/8 bitwise matches; full decoder remains unqualified");
    Ok(())
}
