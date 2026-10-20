//! Ignored macOS-only replay qualification for a CPU AudioVAE candidate.
//!
//! Capture and replay are deliberately separate. Capture writes the real Metal
//! CFM frames once; plain and Accelerate CPU builds then replay exactly those
//! frames without allowing Cargo feature unification to affect the comparison.
use super::{GenerateRequest, Model, Options, Outcome};
use crate::{codec::Codec, weights::Weights};
use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const PATCH_SHAPE: (usize, usize, usize) = (1, 64, 4);
const PATCH_VALUES: usize = PATCH_SHAPE.0 * PATCH_SHAPE.1 * PATCH_SHAPE.2;

struct CapturedSegment {
    patches: Vec<Tensor>,
    pcm: Vec<f32>,
}

#[derive(Clone, Copy, Default)]
struct ReplayTiming {
    handoff: Duration,
    decode: Duration,
    pcm_ready: Duration,
    total: Duration,
    frames: usize,
}

impl ReplayTiming {
    fn json(self) -> serde_json::Value {
        let frames = self.frames.max(1) as f64;
        serde_json::json!({
            "frames": self.frames,
            "handoff_ms": self.handoff.as_secs_f64() * 1_000.0,
            "decode_ms": self.decode.as_secs_f64() * 1_000.0,
            "pcm_ready_ms": self.pcm_ready.as_secs_f64() * 1_000.0,
            "total_ms": self.total.as_secs_f64() * 1_000.0,
            "handoff_ms_per_frame": self.handoff.as_secs_f64() * 1_000.0 / frames,
            "decode_ms_per_frame": self.decode.as_secs_f64() * 1_000.0 / frames,
            "pcm_ready_ms_per_frame": self.pcm_ready.as_secs_f64() * 1_000.0 / frames,
            "total_ms_per_frame": self.total.as_secs_f64() * 1_000.0 / frames,
        })
    }
}

struct ReplayResult {
    timing: ReplayTiming,
    pcm: Vec<Vec<f32>>,
}

struct VariantRuns {
    measured: [ReplayResult; 2],
}

#[derive(Clone, Copy)]
enum DecodePath {
    MetalF16,
    MetalF32,
    CpuF32,
}

impl DecodePath {
    const fn name(self) -> &'static str {
        match self {
            Self::MetalF16 => "metal_f16",
            Self::MetalF32 => "metal_f32",
            Self::CpuF32 => "cpu_f32",
        }
    }

    fn stage(self, patch: &Tensor, target: &Device) -> candle_core::Result<Tensor> {
        match self {
            Self::MetalF16 => Ok(patch.clone()),
            Self::MetalF32 => patch.to_dtype(DType::F32),
            // This includes the Metal command completion and the F16-to-F32
            // conversion that a serial mixed-device candidate would require.
            Self::CpuF32 => patch.to_device(target)?.to_dtype(DType::F32),
        }
    }
}

fn required_path(name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(
        env::var(name).with_context(|| format!("set {name}"))?,
    ))
}

fn candle_io<T>(value: std::io::Result<T>) -> candle_core::Result<T> {
    value.map_err(|error| candle_core::Error::Msg(error.to_string()))
}

fn write_f32le(writer: &mut impl Write, values: &[f32]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    writer.write_all(&bytes)
}

fn read_f32le(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len().is_multiple_of(std::mem::size_of::<f32>()),
        "{} is not f32le data",
        path.display()
    );
    let values: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "{} contains non-finite values",
        path.display()
    );
    Ok(values)
}

fn load_reference(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(
        (1..=2).contains(&spec.channels) && (8_000..=192_000).contains(&spec.sample_rate),
        "reference WAV must be mono/stereo at 8..192 kHz"
    );
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2f32.powi(spec.bits_per_sample as i32 - 1);
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 / scale))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    ensure!(
        samples.len().is_multiple_of(spec.channels as usize)
            && samples.iter().all(|value| value.is_finite()),
        "reference WAV has invalid samples"
    );
    let mono: Vec<_> = samples
        .chunks_exact(spec.channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
        .collect();
    ensure!(
        mono.iter().any(|sample| sample.abs() > 0.0001),
        "reference WAV is silent"
    );
    if spec.sample_rate == 16_000 {
        return Ok(mono);
    }
    let ratio = 16_000.0 / spec.sample_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let expected = (mono.len() as f64 * ratio).round() as usize;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 1.0, params, mono.len(), 1)?;
    let delay = resampler.output_delay();
    let mut output = resampler.process(&[mono], None)?.remove(0);
    output.extend(resampler.process_partial::<Vec<f32>>(None, None)?.remove(0));
    let output: Vec<_> = output.into_iter().skip(delay).take(expected).collect();
    ensure!(output.len() == expected && output.iter().all(|value| value.is_finite()));
    Ok(output)
}

fn load_codec(model: &Path, device: &Device, dtype: DType) -> candle_core::Result<Codec> {
    let mut weights = Weights::open(&model.join("VoxCPM2-Acoustic-F16.gguf"), device, dtype)?;
    Codec::load(&mut weights)
}

fn peak_rss_bytes() -> Option<u64> {
    // Darwin reports ru_maxrss in bytes. It is a process high-water mark, so
    // snapshots before/after each load describe only this experiment process.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        (libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0).then_some(usage.ru_maxrss as u64)
    }
}

fn finite_pcm(decoded: Tensor, target: &Device) -> candle_core::Result<Vec<f32>> {
    let values = decoded
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if values.is_empty() || values.iter().any(|sample| !sample.is_finite()) {
        candle_core::bail!("decoder produced invalid PCM on {target:?}");
    }
    Ok(values)
}

fn replay(
    codec: &mut Codec,
    path: DecodePath,
    target: &Device,
    captured: &[CapturedSegment],
) -> candle_core::Result<ReplayResult> {
    let mut timing = ReplayTiming::default();
    let mut segments = Vec::with_capacity(captured.len());
    for segment in captured {
        codec.reset();
        let mut pcm = Vec::new();
        for patch in &segment.patches {
            let total = Instant::now();
            let staged = path.stage(patch, target)?;
            target.synchronize()?;
            timing.handoff += total.elapsed();

            let decode = Instant::now();
            let decoded = codec.decode(&staged, &|| false)?;
            target.synchronize()?;
            timing.decode += decode.elapsed();

            let ready = Instant::now();
            pcm.extend(finite_pcm(decoded, target)?);
            timing.pcm_ready += ready.elapsed();
            timing.total += total.elapsed();
            timing.frames += 1;
        }
        segments.push(pcm);
    }
    codec.reset();
    Ok(ReplayResult {
        timing,
        pcm: segments,
    })
}

fn run_variant(
    name: DecodePath,
    codec: &mut Codec,
    device: &Device,
    captured: &[CapturedSegment],
) -> candle_core::Result<(serde_json::Value, VariantRuns)> {
    let warmup = replay(codec, name, device, captured)?.timing;
    let first = replay(codec, name, device, captured)?;
    let second = replay(codec, name, device, captured)?;
    let report = serde_json::json!({
        "path": name.name(),
        "warmup": warmup.json(),
        "measured": [first.timing.json(), second.timing.json()],
    });
    Ok((
        report,
        VariantRuns {
            measured: [first, second],
        },
    ))
}

fn quality(reference: &[Vec<f32>], candidate: &[Vec<f32>]) -> serde_json::Value {
    let same_segments = reference.len() == candidate.len()
        && reference
            .iter()
            .zip(candidate)
            .all(|(a, b)| a.len() == b.len());
    let reference: Vec<_> = reference.iter().flatten().copied().collect();
    let candidate: Vec<_> = candidate.iter().flatten().copied().collect();
    let same_length = reference.len() == candidate.len();
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_sq = 0f64;
    let mut signal_sq = 0f64;
    let finite = reference
        .iter()
        .chain(&candidate)
        .all(|sample| sample.is_finite());
    let mut different = 0usize;
    for (left, right) in reference.iter().zip(&candidate) {
        if !left.is_finite() || !right.is_finite() {
            continue;
        }
        let diff = left - right;
        max_abs = max_abs.max(diff.abs());
        sum_abs += f64::from(diff.abs());
        sum_sq += f64::from(diff).powi(2);
        signal_sq += f64::from(*left).powi(2);
        different += usize::from(left.to_bits() != right.to_bits());
    }
    let count = reference.len().min(candidate.len()).max(1) as f64;
    let rmse = (sum_sq / count).sqrt();
    let snr_db = (rmse > 0.0).then(|| 10.0 * (signal_sq / sum_sq).log10());
    serde_json::json!({
        "reference_samples": reference.len(),
        "candidate_samples": candidate.len(),
        "same_length": same_length,
        "same_segment_lengths": same_segments,
        "finite": finite,
        "different_samples": different,
        "max_abs": max_abs,
        "mean_abs": sum_abs / count,
        "rmse": rmse,
        "snr_db": snr_db,
        "bit_identical": same_segments && finite && different == 0,
    })
}

#[test]
fn quality_preserves_segment_and_bit_boundaries() {
    let repartitioned = quality(&[vec![1.0], vec![2.0]], &[vec![1.0, 2.0]]);
    assert_eq!(repartitioned["same_length"], true);
    assert_eq!(repartitioned["bit_identical"], false);
    let signed_zero = quality(&[vec![0.0]], &[vec![-0.0]]);
    assert_eq!(signed_zero["max_abs"], 0.0);
    assert_eq!(signed_zero["bit_identical"], false);
    let nonfinite_tail = quality(&[vec![0.0]], &[vec![0.0, f32::NAN]]);
    assert_eq!(nonfinite_tail["finite"], false);
}

fn capture(
    model_directory: &Path,
    reference_path: &Path,
    transcript_path: &Path,
    corpus: &Path,
    output: &Path,
) -> Result<()> {
    ensure!(
        !output.join("capture-index.json").exists(),
        "capture directory already contains capture-index.json"
    );
    fs::create_dir_all(output)?;
    let metal = Device::new_metal(0)?;
    let cpu = Device::Cpu;
    let model_load = Instant::now();
    let mut model = Model::load(model_directory, &metal)?;
    metal.synchronize()?;
    let model_load_ms = model_load.elapsed().as_millis();
    let reference_load = Instant::now();
    let samples = load_reference(reference_path)?;
    let transcript = fs::read_to_string(transcript_path)?;
    let reference = model.reference(&samples, Some(transcript), &|| false)?;
    metal.synchronize()?;
    let reference_ms = reference_load.elapsed().as_millis();
    let text = fs::read_to_string(corpus)?;
    let segments = talechime_core::text::preprocess_text(&text, 180);
    ensure!(!segments.is_empty(), "corpus produced no spoken segments");

    let capture_started = Instant::now();
    let options = Options::default();
    let mut capture_index = Vec::with_capacity(segments.len());
    for (index, segment) in segments.iter().enumerate() {
        let patch_name = format!("patches-{index:03}.f32le");
        let pcm_name = format!("production-pcm-{index:03}.f32le");
        let mut patch_raw = BufWriter::new(File::create(output.join(&patch_name))?);
        let mut production_pcm = Vec::new();
        let mut patch_count = 0usize;
        let status = model.generate_inner(
            GenerateRequest {
                text: &segment.text,
                reference: Some(&reference),
                identity: None,
                options: &options,
            },
            &|| false,
            &mut |pcm| {
                production_pcm.extend_from_slice(pcm);
                Ok(())
            },
            &mut |patch| {
                if patch.dims3()? != PATCH_SHAPE {
                    candle_core::bail!("unexpected CFM patch shape");
                }
                let values = patch
                    .to_dtype(DType::F32)?
                    .to_device(&cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                candle_io(write_f32le(&mut patch_raw, &values))?;
                patch_count += 1;
                Ok(())
            },
        )?;
        model.codec.reset();
        ensure!(
            status == Outcome::Eos,
            "segment {} ended as {status:?}",
            index + 1
        );
        candle_io(patch_raw.flush())?;
        let mut pcm_raw = BufWriter::new(File::create(output.join(&pcm_name))?);
        write_f32le(&mut pcm_raw, &production_pcm)?;
        pcm_raw.flush()?;
        capture_index.push(serde_json::json!({
            "segment": index + 1,
            "text": segment.text,
            "patches": patch_count,
            "patch_shape": PATCH_SHAPE,
            "patch_file": patch_name,
            "production_pcm_file": pcm_name,
            "production_pcm_samples": production_pcm.len(),
            "outcome": "eos",
        }));
    }
    metal.synchronize()?;
    let report = serde_json::json!({
        "experiment": "voxcpm_macos_accelerate_codec_capture_v1",
        "model_directory": model_directory,
        "reference": reference_path,
        "corpus": corpus,
        "model_load_ms": model_load_ms,
        "reference_ms": reference_ms,
        "capture_ms": capture_started.elapsed().as_millis(),
        "segments": capture_index,
        "input": {"segments": segments.len()},
    });
    fs::write(
        output.join("capture-index.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn capture_segments(
    capture: &Path,
    metal: &Device,
) -> Result<(Vec<CapturedSegment>, serde_json::Value)> {
    let index: serde_json::Value =
        serde_json::from_slice(&fs::read(capture.join("capture-index.json"))?)?;
    let segments = index["segments"]
        .as_array()
        .context("capture index has no segment array")?;
    ensure!(!segments.is_empty(), "capture index has no segments");
    let mut captured = Vec::with_capacity(segments.len());
    for (index, segment) in segments.iter().enumerate() {
        ensure!(
            segment["patch_shape"].as_array() == Some(&vec![1.into(), 64.into(), 4.into()]),
            "segment {} has an incompatible patch shape",
            index + 1
        );
        ensure!(
            segment["outcome"].as_str() == Some("eos"),
            "segment {} did not end at EOS",
            index + 1
        );
        let patch_count = segment["patches"]
            .as_u64()
            .context("capture patch count missing")? as usize;
        let patch_file = segment["patch_file"]
            .as_str()
            .context("capture patch file missing")?;
        let values = read_f32le(&capture.join(patch_file))?;
        ensure!(
            values.len() == patch_count * PATCH_VALUES,
            "segment {} patch data length does not match capture index",
            index + 1
        );
        let patches = values
            .as_chunks::<PATCH_VALUES>()
            .0
            .iter()
            .map(|values| {
                Tensor::from_vec(values.to_vec(), PATCH_SHAPE, metal)?.to_dtype(DType::F16)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let pcm_file = segment["production_pcm_file"]
            .as_str()
            .context("capture production PCM file missing")?;
        let pcm = read_f32le(&capture.join(pcm_file))?;
        ensure!(
            pcm.len()
                == segment["production_pcm_samples"]
                    .as_u64()
                    .unwrap_or_default() as usize,
            "segment {} PCM length does not match capture index",
            index + 1
        );
        captured.push(CapturedSegment { patches, pcm });
    }
    Ok((captured, index))
}

fn replay_capture(model_directory: &Path, capture: &Path, output: &Path) -> Result<()> {
    ensure!(
        !output.join("summary.json").exists(),
        "output already contains summary.json"
    );
    fs::create_dir_all(output)?;
    let metal = Device::new_metal(0)?;
    let cpu = Device::Cpu;
    let (captured, capture_index) = capture_segments(capture, &metal)?;
    metal.synchronize()?;

    let initial_peak = peak_rss_bytes();
    let load = Instant::now();
    let mut metal_f16 = load_codec(model_directory, &metal, DType::F16)?;
    metal.synchronize()?;
    let metal_f16_load_ms = load.elapsed().as_millis();
    let mut metal_f32 = load_codec(model_directory, &metal, DType::F32)?;
    metal.synchronize()?;
    let metal_f32_load_ms = load.elapsed().as_millis() - metal_f16_load_ms;
    let mut cpu_f32 = load_codec(model_directory, &cpu, DType::F32)?;
    let cpu_f32_load_ms = load.elapsed().as_millis() - metal_f16_load_ms - metal_f32_load_ms;
    let loaded_peak = peak_rss_bytes();

    let (metal_f16_report, metal_f16) =
        run_variant(DecodePath::MetalF16, &mut metal_f16, &metal, &captured)?;
    let (metal_f32_report, metal_f32) =
        run_variant(DecodePath::MetalF32, &mut metal_f32, &metal, &captured)?;
    let (cpu_f32_report, cpu_f32) = run_variant(DecodePath::CpuF32, &mut cpu_f32, &cpu, &captured)?;
    let final_peak = peak_rss_bytes();
    let production_pcm: Vec<_> = captured.iter().map(|segment| segment.pcm.clone()).collect();
    let quality_rounds = |runs: &VariantRuns| {
        [
            quality(&production_pcm, &runs.measured[0].pcm),
            quality(&production_pcm, &runs.measured[1].pcm),
        ]
    };
    let report = serde_json::json!({
        "experiment": "voxcpm_macos_accelerate_codec_replay_v1",
        "accelerate_feature": cfg!(feature = "accelerate-experiment"),
        "cpu_path": if cfg!(feature = "accelerate-experiment") { "accelerate" } else { "plain" },
        "model_directory": model_directory,
        "capture_directory": capture,
        "capture": capture_index,
        "input": {"segments": captured.len(), "frames": captured.iter().map(|segment| segment.patches.len()).sum::<usize>()},
        "codec_load": {"metal_f16_ms": metal_f16_load_ms, "metal_f32_ms": metal_f32_load_ms, "cpu_f32_ms": cpu_f32_load_ms},
        "peak_rss_bytes": {"before_codecs": initial_peak, "after_codecs": loaded_peak, "after_replays": final_peak},
        "variants": {"metal_f16": metal_f16_report, "metal_f32": metal_f32_report, "cpu_f32": cpu_f32_report},
        "quality": {
            "production_metal_f16": quality_rounds(&metal_f16),
            "production_metal_f32": quality_rounds(&metal_f32),
            "production_cpu_f32": quality_rounds(&cpu_f32),
        },
        "thread_environment": {"RAYON_NUM_THREADS": env::var("RAYON_NUM_THREADS").ok(), "VECLIB_MAXIMUM_THREADS": env::var("VECLIB_MAXIMUM_THREADS").ok()},
    });
    fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[test]
#[ignore = "requires local VoxCPM GGUF weights, a reference WAV and Apple Metal"]
fn macos_accelerate_codec_experiment() -> Result<()> {
    let mode = env::var("VOXCPM_EXPERIMENT_MODE").unwrap_or_else(|_| "replay".to_owned());
    let model_directory = required_path("VOXCPM_EXPERIMENT_MODEL")?;
    let capture_directory = required_path("VOXCPM_EXPERIMENT_CAPTURE")?;
    match mode.as_str() {
        "capture" => capture(
            &model_directory,
            &required_path("VOXCPM_EXPERIMENT_REFERENCE")?,
            &required_path("VOXCPM_EXPERIMENT_TRANSCRIPT")?,
            &env::var("VOXCPM_EXPERIMENT_CORPUS")
                .map(PathBuf::from)
                .unwrap_or_else(|_| {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("tests/fixtures/acceptance-corpus.txt")
                }),
            &capture_directory,
        ),
        "replay" => replay_capture(
            &model_directory,
            &capture_directory,
            &required_path("VOXCPM_EXPERIMENT_OUT")?,
        ),
        _ => anyhow::bail!("VOXCPM_EXPERIMENT_MODE must be capture or replay"),
    }
}
