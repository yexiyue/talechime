//! Native Kokoro v1.1-zh inference, independently compared with official PyTorch.
//!
//! The acoustic model accepts one complete phoneme segment at a time. It does not
//! provide reference-audio cloning or native streaming. See `SOURCE.md` for the
//! exact official sources and development-only export procedure.
mod attention;
#[cfg(target_os = "macos")]
mod blas;
#[cfg(feature = "g2pw")]
pub mod context;
mod decoder;
mod encoder;
pub mod frontend;
mod moments;
mod ops;
#[cfg(test)]
mod reference_tests;
mod spectral;

use candle_core::{DType, Device, Result, Tensor};
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

/// Native PCM sample rate.
pub const SAMPLE_RATE: u32 = 24000;
/// Pinned official model and voice revision.
pub const MODEL_REVISION: &str = "01e7505bd6a7a2ac4975463114c3a7650a9f7218";
/// Pinned official voice file identities (metadata only; contains no weights).
pub const BUILTIN_VOICES_JSON: &str = include_str!("../assets/voices.json");

/// Native single-segment generation parameters and hard output budget.
#[derive(Clone, Copy, Debug)]
pub struct SynthesisOptions {
    /// Model duration scaling; positive and finite.
    pub speed: f32,
    /// Native excitation seed (does not identify a PyTorch RNG stream).
    pub seed: u64,
    /// Maximum output samples, checked before allocating expanded features/noise.
    pub max_samples: usize,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            speed: 1.0,
            seed: 7,
            max_samples: SAMPLE_RATE as usize * 60,
        }
    }
}

/// Configuration accompanying the pinned official Chinese weights.
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    /// Character-to-model-ID map. These are v1.1 tokens, not legacy IPA.
    pub vocab: HashMap<char, u32>,
    n_token: usize,
    hidden_dim: usize,
    style_dim: usize,
    n_layer: usize,
    max_dur: usize,
}

/// Explicit random inputs for reproducible cross-framework acoustic comparison.
pub struct ExcitationNoise {
    /// Uniform initial harmonic phases, shape `[1, 9]`.
    pub initial_phase: Tensor,
    /// Unit Gaussian noise, shape `[1, output_samples, 9]`.
    pub gaussian: Tensor,
}

impl ExcitationNoise {
    /// Generate native seeded noise. Equal seeds do not match PyTorch RNG streams.
    pub fn seeded(samples: usize, seed: u64, device: &Device) -> Result<Self> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let phase: Vec<f32> = (0..9).map(|_| rng.random()).collect();
        let count = samples
            .checked_mul(9)
            .ok_or_else(|| candle_core::Error::Msg("noise budget overflow".into()))?;
        let mut values = Vec::with_capacity(count);
        while values.len() < count {
            let radius = (-2.0 * rng.random_range(f32::MIN_POSITIVE..1.0).ln()).sqrt();
            let angle = rng.random::<f32>() * std::f32::consts::TAU;
            values.push(radius * angle.cos());
            if values.len() < count {
                values.push(radius * angle.sin());
            }
        }
        Ok(Self {
            initial_phase: Tensor::from_vec(phase, (1, 9), device)?,
            gaussian: Tensor::from_vec(values, (1, samples, 9), device)?,
        })
    }
}

/// Completed segment. Integer durations are model frame counts, not forced alignment.
pub struct Output {
    /// Mono F32 waveform at [`SAMPLE_RATE`].
    pub audio: Tensor,
    /// Predicted frame duration for each ID, including both boundary IDs.
    pub durations: Vec<u32>,
}

/// Optional intermediate tensors for independent official comparisons.
#[derive(Default)]
pub struct Trace(Option<BTreeMap<String, Tensor>>);

impl Trace {
    /// Enable tracing; ordinary synthesis retains no intermediates.
    pub fn enabled() -> Self {
        Self(Some(BTreeMap::new()))
    }
    /// Read recorded stages.
    pub fn tensors(&self) -> Option<&BTreeMap<String, Tensor>> {
        self.0.as_ref()
    }
    pub(crate) fn insert(&mut self, name: &str, value: &Tensor) {
        if let Some(values) = &mut self.0 {
            values.insert(name.into(), value.clone());
        }
    }
}

/// Official-layout weights and the single-segment acoustic network.
pub struct Model {
    weights: ops::Weights,
    encoder: ops::Weights,
    config: Config,
    device: Device,
}

impl Model {
    /// Load development-converted official F32 safetensors and its original config.
    /// Encoders/prosody stay on CPU; `device` selects convolutional decoder offload.
    pub fn load(weights: &Path, config: &Path, device: &Device) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(config).map_err(candle_core::Error::wrap)?)
                .map_err(candle_core::Error::wrap)?;
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../assets/config.json"))
                .map_err(candle_core::Error::wrap)?;
        if value != expected {
            candle_core::bail!(
                "unsupported Kokoro configuration; expected pinned official v1.1-zh"
            );
        }
        let config: Config = serde_json::from_value(value).map_err(candle_core::Error::wrap)?;
        if (
            config.n_token,
            config.hidden_dim,
            config.style_dim,
            config.n_layer,
            config.max_dur,
        ) != (178, 512, 128, 3, 50)
        {
            candle_core::bail!("unsupported Kokoro configuration; expected official v1.1-zh");
        }
        let values = candle_core::safetensors::load(weights, &Device::Cpu)?;
        if values.values().any(|t| t.dtype() != DType::F32) {
            candle_core::bail!("Kokoro currently requires original F32 weights");
        }
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();
        for (name, tensor) in values {
            if name.starts_with("decoder.") {
                decoder.insert(name, tensor.to_device(device)?);
            } else {
                encoder.insert(name, tensor);
            }
        }
        Ok(Self {
            weights: ops::Weights::new(decoder),
            encoder: ops::Weights::new(encoder),
            config,
            device: device.clone(),
        })
    }

    /// Map a complete official phoneme string without silently dropping unknown tokens.
    pub fn token_ids(&self, phonemes: &str) -> Result<Vec<u32>> {
        let mut ids = vec![0];
        for c in phonemes.chars() {
            ids.push(*self.config.vocab.get(&c).ok_or_else(|| {
                candle_core::Error::Msg(format!("unsupported Kokoro phoneme {c:?}"))
            })?);
        }
        ids.push(0);
        if ids.len() > 512 || ids.len() <= 2 {
            candle_core::bail!("Kokoro requires 1..=510 phoneme tokens");
        }
        Ok(ids)
    }

    /// Synthesize one unpadded segment. `max_samples` bounds duration expansion and noise.
    /// Explicit noise enables independent official comparison; otherwise use native seed.
    pub fn forward(
        &self,
        ids: &[u32],
        style: &Tensor,
        options: &SynthesisOptions,
        noise: Option<&ExcitationNoise>,
        trace: &mut Trace,
    ) -> Result<Output> {
        self.forward_cancellable(ids, style, options, noise, trace, &|| false)
    }

    /// Synthesize while checking consumer cancellation between acoustic stages.
    pub fn forward_cancellable(
        &self,
        ids: &[u32],
        style: &Tensor,
        options: &SynthesisOptions,
        noise: Option<&ExcitationNoise>,
        trace: &mut Trace,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Output> {
        let check = || {
            if cancelled() {
                Err(candle_core::Error::Msg("Kokoro synthesis cancelled".into()))
            } else {
                Ok(())
            }
        };
        check()?;
        let SynthesisOptions {
            speed,
            seed,
            max_samples,
        } = *options;
        if ids.len() < 3
            || ids.len() > 512
            || ids.first() != Some(&0)
            || ids.last() != Some(&0)
            || ids.iter().any(|&id| id >= 178)
        {
            candle_core::bail!("invalid Kokoro IDs or segment length");
        }
        if !speed.is_finite()
            || speed <= 0.0
            || style.dims() != [1, 256]
            || style.dtype() != DType::F32
        {
            candle_core::bail!("invalid Kokoro speed or style");
        }
        if style
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .any(|x| !x.is_finite())
        {
            candle_core::bail!("nonfinite Kokoro style");
        }
        if max_samples / 600 < ids.len() {
            candle_core::bail!("Kokoro sample budget is smaller than minimum duration");
        }
        // Preserve the official CPU F32 F0 trajectory across devices. Decoder
        // offload contains the heavy convolutional work; a fully GPU encoder's
        // small F0 roundoff is amplified by harmonic phase accumulation.
        let ids_tensor = Tensor::from_slice(ids, (1, ids.len()), &Device::Cpu)?;
        let style = style.to_device(&Device::Cpu)?;
        let prosody_style = style.narrow(1, 128, 128)?;
        let acoustic_style = style.narrow(1, 0, 128)?;
        let w = &self.encoder;
        let bert = w.albert(&ids_tensor)?;
        check()?;
        trace.insert("bert", &bert);
        let projected = w.linear(&bert, "bert_encoder")?;
        trace.insert("bert_encoder", &projected);
        let d = w.duration_encoder(&projected, &prosody_style)?;
        check()?;
        trace.insert("predictor.text_encoder", &d);
        let lstm = w.bilstm(&d, "predictor.lstm")?;
        trace.insert("predictor.lstm.0", &lstm);
        let logits = w.linear(&lstm, "predictor.duration_proj.linear_layer")?;
        trace.insert("predictor.duration_proj", &logits);
        let duration = (candle_nn::ops::sigmoid(&logits)?.sum(2)? / f64::from(speed))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let durations = bounded_durations(&duration, max_samples)?;
        check()?;
        let frames: usize = durations.iter().map(|&d| d as usize).sum();
        let indices: Vec<u32> = durations
            .iter()
            .enumerate()
            .flat_map(|(i, &count)| std::iter::repeat_n(i as u32, count as usize))
            .collect();
        let indices = Tensor::from_vec(indices, frames, &Device::Cpu)?;
        let expanded = d.transpose(1, 2)?.contiguous()?.index_select(&indices, 2)?;
        let (f0, n) = w.f0_noise(&expanded, &prosody_style, trace)?;
        check()?;
        trace.insert("predictor.F0_proj", &f0);
        trace.insert("predictor.N_proj", &n);
        let text = w.text_encode(&ids_tensor)?;
        check()?;
        trace.insert("text_encoder", &text);
        let asr = text.contiguous()?.index_select(&indices, 2)?;
        let native_noise;
        let noise = match noise {
            Some(noise) => noise,
            None => {
                native_noise = ExcitationNoise::seeded(frames * 600, seed, &self.device)?;
                &native_noise
            }
        };
        let audio = self.weights.decode(
            &asr.to_device(&self.device)?,
            &f0.to_device(&self.device)?,
            &n.to_device(&self.device)?,
            &acoustic_style.to_device(&self.device)?,
            noise,
            trace,
        )?;
        check()?;
        if audio.elem_count() > max_samples {
            candle_core::bail!("Kokoro output exceeded sample budget");
        }
        trace.insert("audio", &audio);
        Ok(Output { audio, durations })
    }
}

fn bounded_durations(values: &[f32], max_samples: usize) -> Result<Vec<u32>> {
    let mut total = 0usize;
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() || v < 0.0 || v > u32::MAX as f32 {
                candle_core::bail!("invalid Kokoro duration");
            }
            let duration = v.round_ties_even().max(1.0) as u32;
            total = total
                .checked_add(duration as usize)
                .ok_or_else(|| candle_core::Error::Msg("Kokoro duration overflow".into()))?;
            if total > max_samples / 600 {
                candle_core::bail!("Kokoro duration exceeds output sample budget");
            }
            Ok(duration)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn durations_preserve_ties_to_even_and_reject_excessive_output() -> Result<()> {
        assert_eq!(
            bounded_durations(&[0.0, 1.5, 2.5, 3.5], 6000)?,
            [1, 2, 2, 4]
        );
        assert!(bounded_durations(&[f32::NAN], 6000).is_err());
        assert!(bounded_durations(&[50.0], 600).is_err());
        Ok(())
    }

    #[test]
    fn invalid_inputs_fail_before_accessing_model_weights() -> Result<()> {
        let model = Model {
            weights: ops::Weights::new(HashMap::new()),
            encoder: ops::Weights::new(HashMap::new()),
            config: Config {
                vocab: [('ㄚ', 100)].into(),
                n_token: 178,
                hidden_dim: 512,
                style_dim: 128,
                n_layer: 3,
                max_dur: 50,
            },
            device: Device::Cpu,
        };
        let style = Tensor::zeros((1, 256), DType::F32, &Device::Cpu)?;
        for (ids, options, message) in [
            (vec![0, 178, 0], SynthesisOptions::default(), "IDs"),
            (
                vec![0, 100, 0],
                SynthesisOptions {
                    speed: f32::NAN,
                    ..Default::default()
                },
                "speed or style",
            ),
            (
                vec![0, 100, 0],
                SynthesisOptions {
                    max_samples: 100,
                    ..Default::default()
                },
                "sample budget",
            ),
        ] {
            let error = model
                .forward(&ids, &style, &options, None, &mut Trace::default())
                .err()
                .expect("must reject invalid input");
            assert!(error.to_string().contains(message), "{error}");
        }
        assert!(model.token_ids("ㄚ❓").is_err());
        assert!(model.token_ids("").is_err());
        assert!(model.token_ids(&"ㄚ".repeat(511)).is_err());
        assert_eq!(model.token_ids("ㄚ")?, [0, 100, 0]);
        Ok(())
    }
}
