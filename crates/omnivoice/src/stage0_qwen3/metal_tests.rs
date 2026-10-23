use super::*;
use candle_core::Device;

fn bit_pattern(tensor: &Tensor) -> Result<Vec<u32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

/// The sequence-first rope path replaces a head-major layout copy.  It has
/// to produce exactly the same q and k, otherwise the discrete codes
/// change.
#[test]
fn sequence_first_rope_matches_head_major_rope() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (batch, seq, dim) = (2usize, 157usize, 128usize);
    let cos = Tensor::rand(-1f32, 1f32, (seq, dim / 2), &device)?.to_dtype(DType::F16)?;
    let sin = Tensor::rand(-1f32, 1f32, (seq, dim / 2), &device)?.to_dtype(DType::F16)?;
    let norm_weight = Tensor::rand(0.5f32, 1.5f32, (dim,), &device)?.to_dtype(DType::F16)?;
    // The production path calls the fused kernel on Metal; the wrapper in
    // `with_tracing` forwards to exactly this implementation.
    let norm = candle_nn::RmsNorm::new(norm_weight, 1e-6);

    for heads in [16usize, 8] {
        let proj =
            Tensor::rand(-1f32, 1f32, (batch, seq, heads * dim), &device)?.to_dtype(DType::F16)?;

        // Head-major path: copy into `[B, H, S, D]`, normalise, rotate.
        let head_major = proj
            .reshape((batch, seq, heads, dim))?
            .transpose(1, 2)?
            .flatten(0, 2)?;
        let head_major = norm
            .forward(&head_major)?
            .reshape((batch, heads, seq, dim))?;
        let head_major = candle_nn::rotary_emb::rope(&head_major, &cos, &sin)?;

        // Sequence-first path: normalise and rotate in place, then hand
        // the transposed view to the attention.
        let sequence_first = norm.forward(&proj.reshape((batch, seq, heads, dim))?)?;
        let sequence_first =
            candle_nn::rotary_emb::rope_thd(&sequence_first, &cos, &sin)?.transpose(1, 2)?;

        assert_eq!(
            bit_pattern(&head_major)?,
            bit_pattern(&sequence_first)?,
            "sequence-first rope differs for {heads} heads"
        );
    }
    Ok(())
}

/// The attention kernel reads strided q/k/v, so the copy-free path has to
/// produce the same context as the head-major path.
#[test]
fn strided_attention_matches_contiguous_attention() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (batch, seq, heads, kv_heads, dim) = (2usize, 157usize, 16usize, 8usize, 128usize);
    let scale = 1.0f32 / (dim as f32).sqrt();
    let mask = Tensor::rand(0f32, 1f32, (batch, 1, seq, seq), &device)?.to_dtype(DType::F16)?;
    let mask = mask.expand((batch, heads, seq, seq))?;

    for masked in [false, true] {
        let q_seq =
            Tensor::rand(-1f32, 1f32, (batch, seq, heads * dim), &device)?.to_dtype(DType::F16)?;
        let k_seq = Tensor::rand(-1f32, 1f32, (batch, seq, kv_heads * dim), &device)?
            .to_dtype(DType::F16)?;
        let v_seq = Tensor::rand(-1f32, 1f32, (batch, seq, kv_heads * dim), &device)?
            .to_dtype(DType::F16)?;

        // Copy-free path: the projections stay in the sequence-first
        // layout and the attention reads the transposed views.
        let q = q_seq.reshape((batch, seq, heads, dim))?.transpose(1, 2)?;
        let k = k_seq
            .reshape((batch, seq, kv_heads, dim))?
            .transpose(1, 2)?;
        let v = v_seq
            .reshape((batch, seq, kv_heads, dim))?
            .transpose(1, 2)?;
        let mask_ref = if masked { Some(&mask) } else { None };
        let strided = attention_sdpa(&q, &k, &v, mask_ref, scale)?;

        let contiguous = attention_sdpa(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            mask_ref,
            scale,
        )?;
        assert_eq!(
            bit_pattern(&strided)?,
            bit_pattern(&contiguous)?,
            "strided attention differs, masked={masked}"
        );
    }
    Ok(())
}

#[test]
fn native_gqa_matches_repeated_kv_exactly() -> Result<()> {
    let device = Device::new_metal(0)?;
    for seq in [1, 33, 157, 287] {
        let q = Tensor::rand(-1f32, 1f32, (2, 16, seq, 128), &device)?.to_dtype(DType::F16)?;
        let k = Tensor::rand(-1f32, 1f32, (2, 8, seq, 128), &device)?.to_dtype(DType::F16)?;
        let v = Tensor::rand(-1f32, 1f32, (2, seq, 8, 128), &device)?
            .to_dtype(DType::F16)?
            .transpose(1, 2)?;
        let repeated_k = candle_transformers::utils::repeat_kv(k.clone(), 2)?;
        let repeated_v = candle_transformers::utils::repeat_kv(v.clone(), 2)?;
        let mask: Vec<f32> = (0..2 * seq * seq)
            .map(|i| {
                if i % seq == seq - 1 && seq > 1 {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let mask = Tensor::from_vec(mask, (2, 1, seq, seq), &device)?.to_dtype(DType::F16)?;
        for mask in [None, Some(&mask)] {
            let scale = 1.0 / 128f32.sqrt();
            let native = attention_sdpa(&q, &k, &v, mask, scale)?;
            let repeated = attention_sdpa(&q, &repeated_k, &repeated_v, mask, scale)?;
            assert_eq!(
                native
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                repeated
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                "sequence length {seq}, masked={}",
                mask.is_some()
            );
        }
    }
    Ok(())
}

/// The fused Metal SDPA kernel and the manual score matrix path must
/// agree within f16 tolerance on the shapes stage0 actually uses.
#[test]
fn fused_sdpa_matches_manual_attention() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (batch, heads, kv_heads, seq, dim) = (2usize, 16usize, 8usize, 287usize, 128usize);
    let q = Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), &device)?.to_dtype(DType::F16)?;
    let k = Tensor::rand(0f32, 1f32, (batch, kv_heads, seq, dim), &device)?.to_dtype(DType::F16)?;
    let v = Tensor::rand(0f32, 1f32, (batch, kv_heads, seq, dim), &device)?.to_dtype(DType::F16)?;
    let repeats = heads / kv_heads;
    let k = candle_transformers::utils::repeat_kv(k, repeats)?;
    let v = candle_transformers::utils::repeat_kv(v, repeats)?;
    // Additive mask in the runtime dtype, exactly like stage0 prepares it.
    let mask_bool = Tensor::rand(0f32, 1f32, (batch, 1, seq, seq), &device)?;
    let mask = mask_bool.to_dtype(DType::F16)?;
    let scale = 1.0f64 / (dim as f64).sqrt();

    let manual = attention_manual(&q, &k, &v, Some(&mask), scale)?;
    let fused = attention_sdpa(&q, &k, &v, Some(&mask), scale as f32)?;
    let manual = manual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let fused = fused
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(manual.len(), fused.len());
    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    let mut reference_sq = 0f64;
    for (a, b) in manual.iter().zip(fused.iter()) {
        let diff = (a - b).abs();
        max_abs = max_abs.max(diff);
        sum_sq += f64::from(diff) * f64::from(diff);
        reference_sq += f64::from(*a) * f64::from(*a);
    }
    let rms = (sum_sq / manual.len() as f64).sqrt();
    let reference_rms = (reference_sq / manual.len() as f64).sqrt();
    // f16 has ~3 decimal digits; the two paths may only differ by the
    // accumulation order, not by a different softmax.
    assert!(
        max_abs < 0.05,
        "fused sdpa diverged from manual attention: max_abs={max_abs}"
    );
    assert!(
        rms <= 0.01 * reference_rms.max(1e-6),
        "fused sdpa rms error {rms} too large against {reference_rms}"
    );
    Ok(())
}

/// The fused kernel must also agree when the mask blocks whole rows.
#[test]
fn fused_sdpa_matches_manual_attention_with_masked_rows() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (batch, heads, seq, dim) = (2usize, 16usize, 64usize, 128usize);
    let q = Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), &device)?.to_dtype(DType::F16)?;
    let k = Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), &device)?.to_dtype(DType::F16)?;
    let v = Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), &device)?.to_dtype(DType::F16)?;
    let mut mask_values = vec![0f32; batch * seq * seq];
    for (index, value) in mask_values.iter_mut().enumerate() {
        if index % seq == seq - 1 {
            *value = f32::NEG_INFINITY;
        }
    }
    let mask =
        Tensor::from_vec(mask_values, (batch, 1, seq, seq), &device)?.to_dtype(DType::F16)?;
    let scale = 1.0f64 / (dim as f64).sqrt();
    let manual = attention_manual(&q, &k, &v, Some(&mask), scale)?;
    let fused = attention_sdpa(&q, &k, &v, Some(&mask), scale as f32)?;
    let manual = manual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let fused = fused
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut max_abs = 0f32;
    for (a, b) in manual.iter().zip(fused.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(
        max_abs < 0.05,
        "fused sdpa diverged with masked rows: max_abs={max_abs}"
    );
    Ok(())
}

/// The fused rms_norm kernel and the layer-norm path must both track an
/// f64 host reference; the fused kernel rounds the scale to f16, so this
/// records how much accuracy the speed-up costs.
#[test]
fn fused_rms_norm_tracks_f64_host_reference() -> Result<()> {
    let device = Device::new_metal(0)?;
    let (rows, hidden) = (64usize, 1024usize);
    let eps = 1e-6f64;
    let xs = Tensor::rand(-3f32, 3f32, (rows, hidden), &device)?.to_dtype(DType::F16)?;
    let weight = Tensor::rand(0.5f32, 1.5f32, (hidden,), &device)?.to_dtype(DType::F16)?;
    let norm = candle_nn::RmsNorm::new(weight.clone(), eps);

    let reference = {
        let host = xs
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let host_weight = weight
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let mut out = vec![0f32; rows * hidden];
        for (row, values) in host.iter().enumerate() {
            let mean_sq = values
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                / hidden as f64;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for (index, value) in values.iter().enumerate() {
                let normalized = f64::from(*value) * scale;
                let scaled = normalized * f64::from(host_weight[index]);
                out[row * hidden + index] = scaled as f32;
            }
        }
        out
    };

    let mut errors = Vec::new();
    for (label, actual) in [
        ("layer_norm_path", norm.forward_diff(&xs)?),
        ("fused_kernel", norm.forward(&xs)?),
    ] {
        let actual = actual
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let mut max_abs = 0f32;
        let mut sum_sq = 0f64;
        let mut reference_sq = 0f64;
        for (row, values) in actual.iter().enumerate() {
            for (index, value) in values.iter().enumerate() {
                let expected = reference[row * hidden + index];
                let diff = (f64::from(*value) - f64::from(expected)).abs();
                max_abs = max_abs.max(diff as f32);
                sum_sq += diff * diff;
                reference_sq += f64::from(expected) * f64::from(expected);
            }
        }
        let rms = (sum_sq / (rows * hidden) as f64).sqrt();
        let reference_rms = (reference_sq / (rows * hidden) as f64).sqrt();
        errors.push((label, max_abs, rms / reference_rms.max(1e-9)));
    }
    for (label, max_abs, relative_rms) in &errors {
        eprintln!("rms_norm {label}: max_abs={max_abs:.6} relative_rms={relative_rms:.3e}");
    }
    // Both paths must stay far below the f16 rounding band (2^-11 ~ 4.9e-4).
    for (label, _, relative_rms) in &errors {
        assert!(
            *relative_rms < 2e-3,
            "{label} diverged from the f64 reference: relative_rms={relative_rms}"
        );
    }
    Ok(())
}
