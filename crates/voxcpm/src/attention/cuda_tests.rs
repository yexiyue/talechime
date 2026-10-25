//! Isolated CUDA hypotheses. These helpers do not enter production inference.
use super::*;
use candle_core::Device;

struct Captured {
    seen: usize,
    inputs: Vec<(usize, Tensor, Tensor, Tensor)>,
}

thread_local! {
    static CAPTURE: std::cell::RefCell<Option<Captured>> = const { std::cell::RefCell::new(None) };
    static REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static PATCHES: std::cell::RefCell<Option<Vec<Vec<u32>>>> = const { std::cell::RefCell::new(None) };
}

pub(super) fn reference_enabled() -> bool {
    REFERENCE.with(std::cell::Cell::get)
}

pub(crate) fn capture_patch(patch: &Tensor) -> Result<()> {
    PATCHES.with(|state| {
        if let Some(patches) = state.borrow_mut().as_mut() {
            patches.push(
                patch
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect(),
            );
        }
        Ok(())
    })
}

fn reference_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    offset: usize,
) -> Result<Tensor> {
    attention_impl(q, k, v, causal, offset, false)
}

pub(super) fn capture(q: &Tensor, k: &Tensor, v: &Tensor) {
    if q.dims() != [1, 16, 1, 128] || !q.device().is_cuda() {
        return;
    }
    CAPTURE.with(|state| {
        if let Some(state) = state.borrow_mut().as_mut() {
            let index = state.seen;
            if [0, 27, 35, 576, 603, 611, 2304, 2331, 2339].contains(&index) {
                state.inputs.push((index, q.clone(), k.clone(), v.clone()));
            }
            state.seen += 1;
        }
    });
}

fn packed_key(k: &Tensor, heads: usize) -> Result<Tensor> {
    let (batch, kv_heads, length, width) = k.dims4()?;
    // A singleton sequence already avoids the original transpose copy. Preserve
    // its existing singleton strides and GEMM dispatch instead of specializing it.
    if length == 1 {
        return k
            .unsqueeze(2)?
            .broadcast_as((batch, kv_heads, heads / kv_heads, length, width))?
            .reshape((batch, heads, length, width))?
            .transpose(2, 3)?
            .contiguous();
    }
    k.transpose(2, 3)?
        .unsqueeze(2)?
        .broadcast_as((batch, kv_heads, heads / kv_heads, width, length))?
        .reshape((batch, heads, width, length))?
        .contiguous()
}

fn packed_single_query(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    // Match the baseline's test-only hook cost during local replay.
    capture(q, k, v);
    let (batch, heads, _, width) = q.dims4()?;
    let (_, kv_heads, length, _) = k.dims4()?;
    let scores = (q
        .contiguous()?
        .matmul(&packed_key(k, heads)?)?
        .to_dtype(DType::F32)?
        / (width as f64).sqrt())?;
    let v = v
        .unsqueeze(2)?
        .broadcast_as((batch, kv_heads, heads / kv_heads, length, width))?
        .reshape((batch, heads, length, width))?;
    candle_nn::ops::softmax_last_dim(&scores)?
        .to_dtype(q.dtype())?
        .matmul(&v.contiguous()?)
}

#[test]
#[ignore = "explicit CUDA copy-only component oracle; no model or downloads"]
fn packed_key_cuda_oracle() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut cases = Vec::new();
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for batch in [1, 2] {
            for length in [
                1, 2, 3, 7, 8, 16, 31, 32, 40, 64, 77, 127, 128, 191, 256, 512,
            ] {
                for view in [false, true] {
                    let (q, k, v) = fixture(batch, length, dtype, view, &device)?;
                    let old_key = k
                        .unsqueeze(2)?
                        .broadcast_as((batch, 2, 8, length, 128))?
                        .reshape((batch, 16, length, 128))?
                        .transpose(2, 3)?
                        .contiguous()?;
                    let new_key = packed_key(&k, 16)?;
                    assert_eq!(old_key.stride(), new_key.stride());
                    assert_eq!(
                        old_key
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                        new_key
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?
                    );
                    let old = reference_attention(&q, &k, &v, true, length - 1)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let new = packed_single_query(&q, &k, &v)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let runtime = attention(&q, &k, &v, true, length - 1)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert_eq!(
                        old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        runtime.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    let changed = old
                        .iter()
                        .zip(&new)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    cases.push(serde_json::json!({"dtype":format!("{dtype:?}"),"batch":batch,"length":length,"view":view,"changed":changed}));
                }
            }
        }
    }
    if let Some(path) = std::env::var_os("TALECHIME_VOX_PACKED_KEY_ORACLE") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(serde_json::to_string_pretty(&cases)?.as_bytes())?;
    }
    assert!(
        cases.iter().all(|c| c["changed"] == 0),
        "copy-only oracle differs"
    );
    eprintln!(
        "{} CUDA shapes preserve packed-key values/strides and attention bits",
        cases.len()
    );
    Ok(())
}

#[test]
#[ignore = "explicit CUDA synthetic cost screen; no model or downloads"]
fn packed_key_cuda_cost_screen() -> anyhow::Result<()> {
    use std::time::Instant;
    let device = Device::new_cuda(0)?;
    let mut rows = Vec::new();
    for dtype in [DType::F16, DType::BF16] {
        for length in [40, 128, 256] {
            let (q, k, v) = fixture(1, length, dtype, false, &device)?;
            for boundary in ["key", "attention_with_context_restore"] {
                let invoke = |candidate| -> Result<Tensor> {
                    if boundary == "key" {
                        if candidate {
                            packed_key(&k, 16)
                        } else {
                            k.unsqueeze(2)?
                                .broadcast_as((1, 2, 8, length, 128))?
                                .reshape((1, 16, length, 128))?
                                .transpose(2, 3)?
                                .contiguous()
                        }
                    } else {
                        let context = if candidate {
                            packed_single_query(&q, &k, &v)?
                        } else {
                            reference_attention(&q, &k, &v, true, length - 1)?
                        };
                        context.transpose(1, 2)?.reshape((1, 1, 2048))
                    }
                };
                for _ in 0..32 {
                    std::hint::black_box(invoke(false)?);
                    std::hint::black_box(invoke(true)?);
                }
                for (slot, arm) in "ABBAABBA".chars().enumerate() {
                    device.synchronize()?;
                    let started = Instant::now();
                    for _ in 0..512 {
                        std::hint::black_box(invoke(arm == 'B')?);
                    }
                    device.synchronize()?;
                    rows.push(serde_json::json!({"kind":"synthetic_cost_screen_not_model_performance","dtype":format!("{dtype:?}"),"length":length,"boundary":boundary,"slot":slot,"arm":arm.to_string(),"iterations":512,"seconds_per_call":started.elapsed().as_secs_f64()/512.}));
                }
            }
        }
    }
    if let Some(path) = std::env::var_os("TALECHIME_VOX_PACKED_KEY_SCREEN") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(serde_json::to_string_pretty(&rows)?.as_bytes())?;
    }
    eprintln!(
        "{} synthetic cost blocks; real-input and whole-model evidence still required",
        rows.len()
    );
    Ok(())
}

#[test]
#[ignore = "explicit local CUDA model/input capture and replay; no downloads"]
fn real_cuda_packed_key_replay() -> anyhow::Result<()> {
    use std::{path::PathBuf, time::Instant};
    let directory =
        PathBuf::from(std::env::var_os("TALECHIME_VOX_GQA_MODEL").expect("model required"));
    let variant = std::env::var("TALECHIME_VOX_GQA_VARIANT")?;
    let output =
        PathBuf::from(std::env::var_os("TALECHIME_VOX_GQA_OUTPUT").expect("output required"));
    let text = std::fs::read_to_string(
        std::env::var_os("TALECHIME_VOX_GQA_TEXT").expect("text required"),
    )?;
    let segmented = std::env::var_os("TALECHIME_VOX_GQA_SEGMENTS");
    let requests = if let Some(path) = &segmented {
        let plan: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        anyhow::ensure!(
            plan["corpus_sha256"] == talechime_protocol::text_hash(&text),
            "corpus hash mismatch"
        );
        let requests = plan["segments"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("segments required"))?
            .clone();
        anyhow::ensure!(!requests.is_empty() && requests.len() <= 16);
        for (call, request) in requests.iter().enumerate() {
            anyhow::ensure!(request["call_index"].as_u64() == Some(call as u64));
            let start = request["range"]["start"].as_u64().expect("start") as usize;
            let end = request["range"]["end"].as_u64().expect("end") as usize;
            let source = text
                .get(start..end)
                .ok_or_else(|| anyhow::anyhow!("invalid UTF-8 source range"))?;
            anyhow::ensure!(
                request["text"] == source
                    && request["text_sha256"] == talechime_protocol::text_hash(source)
            );
            anyhow::ensure!(
                request["actual_seed"].as_u64()
                    == Some(talechime_core::SeedPolicy::Pinned(42).resolve(call as u64, 0)),
                "seed mismatch"
            );
        }
        requests
    } else {
        vec![
            serde_json::json!({"call_index":0,"range":null,"text":text.trim(),"text_sha256":talechime_protocol::text_hash(text.trim()),"actual_seed":talechime_core::SeedPolicy::Pinned(42).resolve(0,0)}),
        ]
    };
    let device = Device::new_cuda(0)?;
    let mut model = match variant.as_str() {
        "q8" => crate::Model::load(&directory, &device)?,
        "bf16" => crate::Model::load_original(&directory, &device, DType::BF16, &|| false)?,
        _ => anyhow::bail!("unknown variant"),
    };
    let reference = if variant == "q8" {
        let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(
            std::env::var_os("TALECHIME_VOX_GQA_PROMPT").expect("prompt required"),
        )?)?;
        anyhow::ensure!(saved["identity"]["model"] == "2b-q8_0");
        anyhow::ensure!(
            saved["identity"]["revision"] == "169f64d8b98bbaab1761e4ca3a83e6af653456cc"
        );
        let frames = saved["frames"].as_u64().expect("frames") as usize;
        let values: Vec<f32> = serde_json::from_value(saved["features"].clone())?;
        anyhow::ensure!((1..=188).contains(&frames) && values.len() == frames * 4 * 64);
        Some(crate::Reference {
            features: Tensor::from_vec(values, (frames, 4, 64), &device)?.to_dtype(DType::F16)?,
            transcript: Some(
                saved["identity"]["transcript"]
                    .as_str()
                    .expect("transcript")
                    .into(),
            ),
        })
    } else {
        None
    };
    let full = std::env::var_os("TALECHIME_VOX_PACKED_FULL").is_some();
    if full {
        anyhow::ensure!(
            std::env::var("VOXCPM_CUDA_GRAPH").as_deref() == Ok("0"),
            "full comparison requires eager generation; cached reference graphs could hide the candidate"
        );
    }
    let mut expected = Vec::new();
    let mut captures = Vec::new();
    REFERENCE.with(|mode| mode.set(true));
    for request in &requests {
        let mut pcm_bits = Vec::new();
        let mut chunk_lengths = Vec::new();
        if full {
            PATCHES.with(|state| *state.borrow_mut() = Some(Vec::new()));
        }
        CAPTURE.with(|state| {
            *state.borrow_mut() = Some(Captured {
                seen: 0,
                inputs: Vec::new(),
            })
        });
        let mut samples = 0;
        let outcome = model.generate(
            request["text"].as_str().expect("text"),
            reference.as_ref(),
            &crate::Options {
                seed: request["actual_seed"].as_u64().expect("seed"),
                max_frames: 187,
                ..Default::default()
            },
            &|| false,
            |pcm| {
                if pcm.iter().any(|v| !v.is_finite()) {
                    candle_core::bail!("nonfinite PCM");
                }
                samples += pcm.len();
                if full {
                    pcm_bits.extend(pcm.iter().map(|v| v.to_bits()));
                    chunk_lengths.push(pcm.len());
                }
                Ok(())
            },
        )?;
        let captured = CAPTURE.with(|state| state.borrow_mut().take().expect("capture enabled"));
        anyhow::ensure!(outcome == crate::Outcome::Eos && samples > 0);
        anyhow::ensure!(
            (3..=9).contains(&captured.inputs.len()),
            "not all requested inputs captured"
        );
        captures.push((request.clone(), samples, captured));
        if full {
            let patches = PATCHES.with(|state| state.borrow_mut().take().expect("patch capture"));
            expected.push((pcm_bits, chunk_lengths, patches));
        }
    }
    REFERENCE.with(|mode| mode.set(false));
    let mut full_rows = Vec::new();
    for (request, (expected_pcm, expected_chunks, expected_patches)) in
        requests.iter().zip(&expected)
    {
        PATCHES.with(|state| *state.borrow_mut() = Some(Vec::new()));
        let mut pcm_bits = Vec::new();
        let mut chunk_lengths = Vec::new();
        let outcome = model.generate(
            request["text"].as_str().expect("text"),
            reference.as_ref(),
            &crate::Options {
                seed: request["actual_seed"].as_u64().expect("seed"),
                max_frames: 187,
                ..Default::default()
            },
            &|| false,
            |pcm| {
                pcm_bits.extend(pcm.iter().map(|v| v.to_bits()));
                chunk_lengths.push(pcm.len());
                Ok(())
            },
        )?;
        let patches = PATCHES.with(|state| state.borrow_mut().take().expect("patch capture"));
        anyhow::ensure!(outcome == crate::Outcome::Eos);
        anyhow::ensure!(&pcm_bits == expected_pcm, "full PCM bits differ");
        anyhow::ensure!(
            &chunk_lengths == expected_chunks,
            "PCM chunk boundaries differ"
        );
        anyhow::ensure!(&patches == expected_patches, "latent patch bits differ");
        full_rows.push(serde_json::json!({"request":request,"pcm_samples":pcm_bits.len(),"chunks":chunk_lengths,"latent_patches":patches.len(),"pcm_and_patches_bits_exact":true,"natural_eos":true}));
    }
    let mut rows = Vec::new();
    for (request, _, captured) in &captures {
        for (index, q, k, v) in &captured.inputs {
            let restore = |t: Tensor| t.transpose(1, 2)?.reshape((1, 1, 2048));
            let old = restore(reference_attention(q, k, v, true, k.dim(2)? - 1)?)?;
            let new = restore(packed_single_query(q, k, v)?)?;
            let bits = |t: Tensor| -> Result<Vec<u32>> {
                Ok(t.to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|v| v.to_bits())
                    .collect())
            };
            assert_eq!(bits(old)?, bits(new)?);
            for _ in 0..32 {
                std::hint::black_box(restore(reference_attention(q, k, v, true, k.dim(2)? - 1)?)?);
                std::hint::black_box(restore(packed_single_query(q, k, v)?)?);
            }
            let mut blocks = Vec::new();
            for (slot, arm) in "ABBAABBA".chars().enumerate() {
                device.synchronize()?;
                let start = Instant::now();
                for _ in 0..256 {
                    let context = if arm == 'A' {
                        reference_attention(q, k, v, true, k.dim(2)? - 1)?
                    } else {
                        packed_single_query(q, k, v)?
                    };
                    std::hint::black_box(restore(context)?);
                }
                device.synchronize()?;
                blocks.push(serde_json::json!({"slot":slot,"arm":arm.to_string(),"seconds_per_call":start.elapsed().as_secs_f64()/256.,"iterations":256}));
            }
            rows.push(serde_json::json!({"request":request,"ordinal":index,"dtype":format!("{:?}",q.dtype()),"q_shape":q.dims(),"k_shape":k.dims(),"q_strides":q.stride(),"k_strides":k.stride(),"v_strides":v.stride(),"context_bits_exact":true,"blocks":blocks}));
        }
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(serde_json::to_string_pretty(&serde_json::json!({"kind":"actual_input_replay_not_whole_model_performance","boundary":if segmented.is_some(){"frozen_production_ranges_and_seeds_same_model_owner"}else{"whole_text_single_model_call"},"variant":variant,"full_model_comparison":full_rows,"requests":captures.iter().map(|(request,samples,captured)|serde_json::json!({"request":request,"raw_model_pcm_samples":samples,"natural_eos":true,"captured_calls":captured.seen})).collect::<Vec<_>>(),"rows":rows}))?.as_bytes())?;
    Ok(())
}

fn fixture(
    batch: usize,
    length: usize,
    dtype: DType,
    view: bool,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let data = |count, salt| -> Vec<f32> {
        (0..count)
            .map(|i| (((i * 48271 + salt) % 65521) as f32 - 32760.) / 32768.)
            .collect()
    };
    let q = Tensor::from_vec(data(batch * 16 * 128, 37), (batch, 1, 16, 128), device)?
        .to_dtype(dtype)?
        .transpose(1, 2)?;
    let kv = |salt| -> Result<Tensor> {
        let t = Tensor::from_vec(
            data(batch * 2 * length * 128, salt),
            (batch, length, 2, 128),
            device,
        )?
        .to_dtype(dtype)?
        .transpose(1, 2)?;
        if view { Ok(t) } else { t.contiguous() }
    };
    Ok((q, kv(1709)?, kv(35023)?))
}
