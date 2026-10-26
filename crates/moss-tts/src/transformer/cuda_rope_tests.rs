//! Forward-scoped reuse prototype; no model downloads or production switch.
use super::*;
use candle_core::Device;
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

fn tables(x: &Tensor, offset: usize, theta: f64) -> Result<RotaryTables> {
    let (_, _, length, dim) = x.dims4()?;
    RotaryTables::new(length, dim, offset, theta, x.dtype(), x.device())
}

fn shared(x: &Tensor, tables: &RotaryTables) -> Result<Tensor> {
    tables.apply(x)
}

fn bits(x: Tensor) -> Result<Vec<u32>> {
    Ok(x.flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

fn input(length: usize, heads: usize, dim: usize, device: &Device) -> Result<Tensor> {
    Tensor::from_vec(
        (0..length * heads * dim)
            .map(|i| ((i * 17 % 113) as f32 - 56.) / 57.)
            .collect(),
        (1, length, heads, dim),
        device,
    )?
    .transpose(1, 2)
}

#[test]
#[ignore = "explicit Windows CUDA RoPE numeric and timing prototype; no downloads"]
fn real_cuda_forward_scoped_rope_tables_match_original() -> anyhow::Result<()> {
    let device = Device::new_cuda(0)?;
    let mut checked = 0;
    for length in [1, 2, 12, 128, 256] {
        for offset in [0, 1, 15, 226, 4096] {
            for dim in [64, 128] {
                for dtype in [DType::F32, DType::F16, DType::BF16] {
                    let q = input(length, 16, dim, &device)?.to_dtype(dtype)?;
                    let k = input(length, 8, dim, &device)?.to_dtype(dtype)?;
                    let tables = tables(&q, offset, 1_000_000.)?;
                    for x in [&q, &k] {
                        assert_eq!(
                            bits(rope(x, offset, 1_000_000., false)?)?,
                            bits(shared(x, &tables)?)?,
                            "length={length} offset={offset} dim={dim} dtype={dtype:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    eprintln!("{checked} Q/K CUDA RoPE outputs bitwise identical");
    for length in [1, 226] {
        let q = input(length, 16, 128, &device)?.to_dtype(DType::BF16)?;
        let k = input(length, 8, 128, &device)?.to_dtype(DType::BF16)?;
        let run = |reuse: bool| -> Result<()> {
            let cached = if reuse {
                Some(tables(&q, 226, 1_000_000.)?)
            } else {
                None
            };
            for _ in 0..28 {
                for x in [&q, &k] {
                    let value = match &cached {
                        Some(tables) => shared(x, tables)?,
                        None => rope(x, 226, 1_000_000., false)?,
                    };
                    std::hint::black_box(value);
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
            for _ in 0..12 {
                run(reuse)?;
            }
            device.synchronize()?;
            rows.push(serde_json::json!({"reuse":reuse,"seconds_per_forward_rope":started.elapsed().as_secs_f64()/12.}));
        }
        eprintln!(
            "ROPE_TIMING {}",
            serde_json::json!({"length":length,"layers":28,"dtype":"BF16","rows":rows})
        );
    }
    Ok(())
}

#[test]
#[ignore = "explicit Realtime/VoiceDesign CUDA models and codec; no downloads"]
fn real_cuda_rope_reuse_preserves_model_codes_and_pcm() -> anyhow::Result<()> {
    struct ResetReference;
    impl Drop for ResetReference {
        fn drop(&mut self) {
            REFERENCE.with(|value| value.set(false));
        }
    }
    let root = std::path::PathBuf::from(
        std::env::var_os("TALECHIME_MOSS17_CUDA_ROOT")
            .ok_or_else(|| anyhow::anyhow!("explicit local MOSS resource root required"))?,
    );
    let device = Device::new_cuda(0)?;
    for (mode, revision) in [
        ("realtime-1.7b", "75682787d8e2fcc73faca37ba2931453ca9c4022"),
        (
            "voice-design-1.7b",
            "97521ec2b6f3ec5026ac1f5751f8fc302d82c2d4",
        ),
    ] {
        let mut model = crate::Model::load(
            mode,
            &root.join("moss/models").join(mode).join(revision),
            &device,
            DType::BF16,
        )?;
        let mut codec = crate::codec::AudioCodec::load(
            &root.join("moss/models/codec/3cd226ba2947efa357ef453bcad111b6eafba782"),
            &device,
            DType::F16,
        )?;
        for (seed, text) in [
            (42, "你好，欢迎使用叙铃。"),
            (7330414270409262323, "你好，欢迎使用叙铃。"),
            (42, "窗外下起了小雨，我们坐在灯下，继续讲述今天的故事。"),
        ] {
            let mut run = |reference| -> anyhow::Result<(Vec<Vec<u32>>, Vec<u32>)> {
                REFERENCE.with(|value| value.set(reference));
                TABLE_BUILDS.with(|value| value.set(0));
                let _reset = ResetReference;
                let mut request = crate::Generation::new(text);
                request.seed = seed;
                request.max_frames = if mode == "voice-design-1.7b" {
                    125
                } else {
                    375
                };
                request.instruction =
                    (mode == "voice-design-1.7b").then_some("年轻女性，声音清晰自然");
                let mut codes = Vec::new();
                model.generate(&request, &|| false, |frame| {
                    codes.push(frame.to_vec());
                    Ok(true)
                })?;
                let builds = TABLE_BUILDS.with(Cell::get);
                anyhow::ensure!(
                    if reference { builds == 0 } else { builds > 0 },
                    "unexpected shared table path: reference={reference} builds={builds}"
                );
                anyhow::ensure!(!codes.is_empty());
                codec.reset_decoder();
                let mut pcm = Vec::new();
                for frames in codes.chunks(5) {
                    let samples = codec.decode(frames, &|| false)?;
                    anyhow::ensure!(samples.iter().all(|sample| sample.is_finite()));
                    pcm.extend(samples.into_iter().map(f32::to_bits));
                }
                Ok((codes, pcm))
            };
            let expected = run(true)?;
            assert_eq!(run(false)?, expected, "{mode} seed={seed} text={text}");
            assert_eq!(run(true)?, expected, "reference state reuse changed");
            eprintln!(
                "{mode} seed={seed} text_bytes={} frames={} raw_pcm_samples={}: original/shared/original exact, complete generation returned",
                text.len(),
                expected.0.len(),
                expected.1.len()
            );
        }
    }
    Ok(())
}
