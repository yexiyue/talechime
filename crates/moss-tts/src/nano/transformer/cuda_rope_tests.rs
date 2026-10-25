//! Interleaved Nano RoPE screening, independent from the Qwen-style layout.
use super::*;
use candle_core::{DType, Device, Result};
use std::cell::Cell;
use std::time::Instant;

thread_local! {
    static REFERENCE: Cell<bool> = const { Cell::new(false) };
    static TABLE_BUILDS: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn reference_enabled() -> bool {
    REFERENCE.with(Cell::get)
}

pub(super) fn shared_table_built() {
    TABLE_BUILDS.with(|value| value.set(value.get() + 1));
}

fn tables(x: &Tensor, offset: usize) -> Result<crate::transformer::RotaryTables> {
    let (_, _, length, dim) = x.dims4()?;
    crate::transformer::RotaryTables::new(length, dim, offset, 10_000., x.dtype(), x.device())
}

fn apply(x: &Tensor, table: &crate::transformer::RotaryTables) -> Result<Tensor> {
    table.apply_interleaved(x)
}

fn input(length: usize, device: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        (0..length * 768)
            .map(|i| ((i * 23 % 127) as f32 - 63.) / 64.)
            .collect(),
        (1, length, 12, 64),
        device,
    )?
    .transpose(1, 2)
}

fn bits(x: Tensor) -> Result<Vec<u32>> {
    Ok(x.flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

#[test]
#[ignore = "explicit Windows CUDA interleaved Nano RoPE screening; no downloads"]
fn real_cuda_nano_forward_rope_tables_match_original() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut checked = 0;
    for length in [1, 2, 12, 128, 226, 269] {
        for offset in [0, 1, 15, 226, 4096] {
            let q = input(length, &device)?;
            let k = (&q * 0.75)?;
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let q = q.to_dtype(dtype)?;
                let k = k.to_dtype(dtype)?;
                let table = tables(&q, offset)?;
                for x in [&q, &k] {
                    assert_eq!(
                        bits(crate::transformer::rope(x, offset, 10_000., true)?)?,
                        bits(apply(x, &table)?)?,
                        "length={length} offset={offset} dtype={dtype:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    eprintln!("{checked} interleaved Q/K CUDA outputs exact");
    for (layers, length) in [(12, 1), (12, 226), (1, 1)] {
        let q = input(length, &device)?;
        let k = (&q * 0.75)?;
        let run = |reuse| -> Result<()> {
            let table = if reuse { Some(tables(&q, 15)?) } else { None };
            for _ in 0..layers {
                for x in [&q, &k] {
                    let output = match &table {
                        Some(table) => apply(x, table)?,
                        None => crate::transformer::rope(x, 15, 10_000., true)?,
                    };
                    std::hint::black_box(output);
                }
            }
            Ok(())
        };
        for reuse in [false, true, false, true] {
            run(reuse)?;
        }
        device.synchronize()?;
        let mut rows = Vec::new();
        for reuse in [false, true, true, false, false, true, true, false] {
            device.synchronize()?;
            let started = Instant::now();
            for _ in 0..30 {
                run(reuse)?;
            }
            device.synchronize()?;
            rows.push(serde_json::json!({"reuse":reuse,"seconds_per_forward_rope":started.elapsed().as_secs_f64()/30.}));
        }
        eprintln!(
            "NANO_ROPE_TIMING {}",
            serde_json::json!({"layers":layers,"length":length,"dtype":"F32","rows":rows})
        );
    }
    Ok(())
}

#[test]
#[ignore = "explicit local Nano CUDA model, codec and prompt rows; no downloads"]
fn real_cuda_nano_shared_rope_preserves_codes_pcm_and_eos() -> anyhow::Result<()> {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            REFERENCE.with(|value| value.set(false));
        }
    }
    let directory = std::path::PathBuf::from(
        std::env::var_os("TALECHIME_NANO_PROFILE_MODEL")
            .ok_or_else(|| anyhow::anyhow!("explicit Nano model directory required"))?,
    );
    let rows = std::env::var_os("TALECHIME_NANO_PROFILE_ROWS")
        .ok_or_else(|| anyhow::anyhow!("explicit Nano prompt rows required"))?;
    let rows: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(rows)?)?;
    let device = Device::new_cuda(0)?;
    let mut model = crate::nano::Nano::load(&directory.join("tts"), &device)?;
    let mut codec = crate::codec::AudioCodec::load(&directory.join("codec"), &device, DType::F32)?;
    let mut run = |seed, reference| -> anyhow::Result<(Vec<Vec<u32>>, Vec<u32>)> {
        REFERENCE.with(|value| value.set(reference));
        TABLE_BUILDS.with(|value| value.set(0));
        let _reset = Reset;
        let mut codes = Vec::new();
        model.generate(&rows, 375, seed, &|| false, |frame| {
            codes.push(frame);
            Ok(())
        })?;
        anyhow::ensure!(!codes.is_empty());
        let builds = TABLE_BUILDS.with(Cell::get);
        anyhow::ensure!(
            if reference { builds == 0 } else { builds > 0 },
            "unexpected table path: reference={reference} builds={builds}"
        );
        codec.reset_decoder();
        let mut pcm = Vec::new();
        for chunk in codes.chunks(3) {
            let samples = codec.decode(chunk, &|| false)?;
            anyhow::ensure!(samples.iter().all(|sample| sample.is_finite()));
            pcm.extend(samples.into_iter().map(f32::to_bits));
        }
        Ok((codes, pcm))
    };
    for seed in [3_560_097_011, 42, 7] {
        let expected = run(seed, true)?;
        assert_eq!(run(seed, false)?, expected, "seed={seed}");
        assert_eq!(run(seed, true)?, expected, "reference reuse changed");
        eprintln!(
            "Nano seed={seed}: {} frames and {} raw PCM samples exact, natural EOS returned; shared path active",
            expected.0.len(),
            expected.1.len()
        );
    }
    Ok(())
}
