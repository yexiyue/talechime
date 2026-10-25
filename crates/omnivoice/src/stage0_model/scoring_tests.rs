use super::*;

fn check_greedy_confidence(device: &Device) -> Result<()> {
    for width in [1024, 1025, 4097] {
        // Repeated maxima, different rows, and a mask token with the largest raw logit.
        let values: Vec<f32> = (0..2 * 8 * 7 * width)
            .map(|i| {
                if i % width == width - 1 {
                    20.0
                } else {
                    (i % 13) as f32 - 6.0
                }
            })
            .collect();
        let logits = Tensor::from_vec(values, (2, 8, 7, width), device)?;
        let u = (logits.sin()? * 0.5)?;
        for guidance in [0.0, 2.0] {
            let c_probs = candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;
            let old_probs = if guidance == 0.0 {
                c_probs
            } else {
                let u_probs = candle_nn::ops::log_softmax(&u, candle_core::D::Minus1)?;
                let guided = (&c_probs + ((&c_probs - &u_probs)? * guidance as f64)?)?;
                candle_nn::ops::log_softmax(&guided, candle_core::D::Minus1)?
            };
            let old_probs = mask_audio_token(&old_probs, width - 1)?;
            let (tokens, confidence) = predict_tokens_with_scoring_from_tensors(
                &logits,
                &u,
                guidance,
                0.0,
                width - 1,
                None,
            )?;
            assert_eq!(
                tokens.flatten_all()?.to_vec1::<i64>()?,
                old_probs
                    .argmax(candle_core::D::Minus1)?
                    .to_dtype(DType::I64)?
                    .flatten_all()?
                    .to_vec1::<i64>()?,
                "tokens width={width} guidance={guidance}"
            );
            let bits = |t: Tensor| -> Result<Vec<u32>> {
                Ok(t.flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(
                bits(confidence)?,
                bits(old_probs.max(candle_core::D::Minus1)?)?,
                "confidence width={width} guidance={guidance}"
            );
        }
    }
    Ok(())
}

#[test]
fn cpu_greedy_confidence_gather_matches_full_reduction() -> Result<()> {
    check_greedy_confidence(&Device::Cpu)
}

#[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
#[test]
#[ignore = "requires an actual CUDA device; no model downloads"]
fn cuda_greedy_confidence_gather_matches_full_reduction() -> Result<()> {
    check_greedy_confidence(&Device::new_cuda(0)?)
}

#[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
#[test]
#[ignore = "synthetic CUDA boundary timing; run a frozen release binary serially"]
fn cuda_greedy_confidence_boundary_timing() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for frames in [41, 130] {
        let values = Tensor::arange(0u32, (8 * frames * 1025) as u32, &device)?
            .to_dtype(DType::F32)?
            .sin()?
            .reshape((1, 8, frames, 1025))?;
        let original = || -> Result<(Tensor, Tensor)> {
            let tokens = values
                .argmax(candle_core::D::Minus1)?
                .to_dtype(DType::I64)?;
            Ok((tokens, values.max(candle_core::D::Minus1)?))
        };
        let candidate = || -> Result<(Tensor, Tensor)> {
            let tokens = values.argmax(candle_core::D::Minus1)?;
            let confidence = values
                .gather(
                    &tokens.unsqueeze(candle_core::D::Minus1)?,
                    candle_core::D::Minus1,
                )?
                .squeeze(candle_core::D::Minus1)?;
            Ok((tokens.to_dtype(DType::I64)?, confidence))
        };
        std::hint::black_box(original()?);
        std::hint::black_box(candidate()?);
        for group in 0..5 {
            for (slot, fast) in [false, true, true, false].into_iter().enumerate() {
                device.synchronize()?;
                let started = std::time::Instant::now();
                for _ in 0..20 {
                    std::hint::black_box(if fast { candidate()? } else { original()? });
                }
                device.synchronize()?;
                println!(
                    "CUDA_CONFIDENCE {}",
                    serde_json::json!({"frames":frames,"group":group,"slot":slot,"candidate":fast,"seconds_per_call":started.elapsed().as_secs_f64()/20.0,"input":"synthetic_f32_not_real_model"})
                );
            }
        }
    }
    Ok(())
}

#[test]
fn cpu_greedy_confidence_preserves_nonfinite_and_signed_zero_results() -> Result<()> {
    for values in [
        vec![0., -0., 1., 1., f32::NEG_INFINITY],
        vec![f32::from_bits(0x7fc00042), -1., 3., 3., 0.],
        vec![1., f32::INFINITY, -1., f32::NAN, 0.],
        vec![f32::NEG_INFINITY; 5],
    ] {
        let logits = Tensor::from_vec(values, (1, 1, 1, 5), &Device::Cpu)?;
        let probs = mask_audio_token(
            &candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?,
            4,
        )?;
        let (tokens, confidence) =
            predict_tokens_with_scoring_from_tensors(&logits, &logits, 0., 0., 4, None)?;
        assert_eq!(
            tokens.flatten_all()?.to_vec1::<i64>()?,
            probs
                .argmax(candle_core::D::Minus1)?
                .to_dtype(DType::I64)?
                .flatten_all()?
                .to_vec1::<i64>()?
        );
        assert_eq!(
            confidence
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            probs
                .max(candle_core::D::Minus1)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[test]
fn cpu_stochastic_confidence_remains_the_distribution_maximum() -> Result<()> {
    use rand::SeedableRng;

    let logits = Tensor::from_vec(
        (0..8 * 7 * 33)
            .map(|i| (i % 7) as f32 / 4.)
            .collect::<Vec<_>>(),
        (1, 8, 7, 33),
        &Device::Cpu,
    )?;
    let probs = mask_audio_token(
        &candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?,
        32,
    )?;
    let mut rng = StdRng::seed_from_u64(42);
    let (tokens, confidence) =
        predict_tokens_with_scoring_from_tensors(&logits, &logits, 0., 1., 32, Some(&mut rng))?;
    assert_ne!(
        tokens.flatten_all()?.to_vec1::<i64>()?,
        probs
            .argmax(candle_core::D::Minus1)?
            .to_dtype(DType::I64)?
            .flatten_all()?
            .to_vec1::<i64>()?
    );
    assert_eq!(
        confidence.flatten_all()?.to_vec1::<f32>()?,
        probs
            .max(candle_core::D::Minus1)?
            .flatten_all()?
            .to_vec1::<f32>()?
    );
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_greedy_confidence_gather_matches_full_reduction() -> Result<()> {
    check_greedy_confidence(&Device::new_metal(0)?)
}
