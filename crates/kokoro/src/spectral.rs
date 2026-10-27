//! Official harmonic excitation and centered periodic-Hann STFT conventions.
use crate::ExcitationNoise;
use candle_core::{Device, Result, Tensor};
use rustfft::{FftPlanner, num_complex::Complex32};

const FFT: usize = 20;
const HOP: usize = 5;
const SCALE: usize = 300;

pub(crate) fn excitation(f0: &Tensor, noise: &ExcitationNoise) -> Result<Tensor> {
    let pitch = f0.flatten_all()?.to_vec1::<f32>()?;
    let length = pitch.len() * SCALE;
    if noise.gaussian.dims() != [1, length, 9] || noise.initial_phase.dims() != [1, 9] {
        candle_core::bail!("invalid Kokoro excitation noise shapes for {length} samples");
    }
    let normal = noise.gaussian.flatten_all()?.to_vec1::<f32>()?;
    let mut output = vec![0f32; length * 9];
    // align_corners=False linear downsampling selects within each nearest-upsampled
    // F0 block. The official initial-phase addition at sample 0 is outside those taps.
    for harmonic in 0..9 {
        let mut accumulator = 0f64;
        let phases: Vec<_> = pitch
            .iter()
            .map(|&f| {
                // F0 can be negative in unvoiced regions. PyTorch remainder is
                // Euclidean; Rust `%` would change the accumulated phase history.
                let radians = (f * (harmonic + 1) as f32 / 24000.0).rem_euclid(1.0);
                accumulator += f64::from(radians);
                accumulator as f32 * 2.0 * std::f32::consts::PI * SCALE as f32
            })
            .collect();
        for sample in 0..length {
            // PyTorch's float-input interpolate computes source coordinates in
            // float, including the reciprocal scale. F64 coordinates change the
            // harmonic phase enough to invalidate a full waveform comparison.
            let coordinate = (1.0 / SCALE as f32)
                .mul_add(sample as f32 + 0.5, -0.5)
                .max(0.0);
            let left = coordinate.floor();
            let fraction = coordinate - left;
            let a = left.min((pitch.len() - 1) as f32) as usize;
            let b = (left + 1.0).min((pitch.len() - 1) as f32) as usize;
            let phase = phases[a].mul_add(1.0 - fraction, phases[b] * fraction);
            let voiced = pitch[sample / SCALE] > 10.0;
            let amplitude = if voiced { 0.003 } else { 0.1 / 3.0 };
            output[sample * 9 + harmonic] = if voiced { phase.sin() * 0.1 } else { 0.0 }
                + amplitude * normal[sample * 9 + harmonic];
        }
    }
    Tensor::from_vec(output, (1, length, 9), f0.device())
}

fn window() -> Vec<f32> {
    (0..FFT)
        .map(|i| (1.0 - (2.0 * std::f32::consts::PI * i as f32 / FFT as f32).cos()) * 0.5)
        .collect()
}

pub(crate) fn stft(source: &Tensor) -> Result<(Tensor, Tensor)> {
    let x = source.to_vec1::<f32>()?;
    if x.len() <= FFT / 2 {
        candle_core::bail!("Kokoro excitation too short for reflection STFT");
    }
    let frames = x.len() / HOP + 1;
    let bins = FFT / 2 + 1;
    let mut mag = vec![0f32; bins * frames];
    let mut phase = mag.clone();
    let window = window();
    let fft = FftPlanner::new().plan_fft_forward(FFT);
    let mut block = vec![Complex32::default(); FFT];
    for frame in 0..frames {
        for (tap, value) in block.iter_mut().enumerate() {
            let index = (frame * HOP + tap) as isize - (FFT / 2) as isize;
            let index = if index < 0 {
                -index
            } else if index >= x.len() as isize {
                2 * x.len() as isize - 2 - index
            } else {
                index
            } as usize;
            *value = Complex32::new(x[index] * window[tap], 0.0);
        }
        fft.process(&mut block);
        for bin in 0..bins {
            mag[bin * frames + frame] = block[bin].norm();
            phase[bin * frames + frame] = block[bin].arg();
        }
    }
    Ok((
        Tensor::from_vec(mag, (1, bins, frames), source.device())?,
        Tensor::from_vec(phase, (1, bins, frames), source.device())?,
    ))
}

pub(crate) fn istft(magnitude: &Tensor, phase: &Tensor) -> Result<Tensor> {
    let (_, bins, frames) = magnitude.dims3()?;
    if bins != 11 || frames < 2 || magnitude.dims() != phase.dims() {
        candle_core::bail!("invalid Kokoro ISTFT shape");
    }
    let mag = magnitude.flatten_all()?.to_vec1::<f32>()?;
    let phase_values = phase.flatten_all()?.to_vec1::<f32>()?;
    let length = (frames - 1) * HOP + FFT;
    let mut result = vec![0f32; length];
    let mut divisor = result.clone();
    let window = window();
    let fft = FftPlanner::new().plan_fft_inverse(FFT);
    let mut block = vec![Complex32::default(); FFT];
    for frame in 0..frames {
        for bin in 0..bins {
            let p = phase_values[bin * frames + frame];
            block[bin] = Complex32::new(p.cos(), p.sin()) * mag[bin * frames + frame];
        }
        for bin in bins..FFT {
            block[bin] = block[FFT - bin].conj();
        }
        fft.process(&mut block);
        for (tap, value) in block.iter().enumerate() {
            let index = frame * HOP + tap;
            result[index] += value.re / FFT as f32 * window[tap];
            divisor[index] += window[tap] * window[tap];
        }
    }
    let samples = (frames - 1) * HOP;
    let audio: Vec<_> = (FFT / 2..FFT / 2 + samples)
        .map(|i| result[i] / divisor[i])
        .collect();
    if audio.iter().any(|x| !x.is_finite()) {
        candle_core::bail!("Kokoro vocoder produced nonfinite audio");
    }
    Tensor::from_vec(audio, samples, &Device::Cpu)?.to_device(magnitude.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Fixture {
        pitch: Vec<f32>,
        sample_indices: Vec<usize>,
        excitation_samples: Vec<Vec<f32>>,
        signal: Vec<f32>,
        magnitude: Vec<f32>,
        phase: Vec<f32>,
        reconstructed: Vec<f32>,
    }

    fn fixture() -> Fixture {
        serde_json::from_str(include_str!("../tests/fixtures/spectral.json")).unwrap()
    }

    #[test]
    fn excitation_matches_independent_official_interpolation_samples() -> Result<()> {
        let f = fixture();
        let samples = f.pitch.len() * 300;
        let noise = ExcitationNoise {
            initial_phase: Tensor::from_vec(
                (0..9).map(|i| i as f32 / 10.0).collect(),
                (1, 9),
                &Device::Cpu,
            )?,
            gaussian: Tensor::from_vec(
                [0.1f32, -0.7, 0.3]
                    .into_iter()
                    .cycle()
                    .take(samples * 9)
                    .collect(),
                (1, samples, 9),
                &Device::Cpu,
            )?,
        };
        let pitch = Tensor::from_vec(f.pitch.clone(), (1, 1, f.pitch.len()), &Device::Cpu)?;
        let actual = excitation(&pitch, &noise)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (sample, expected) in f.sample_indices.iter().zip(&f.excitation_samples) {
            for harmonic in 0..9 {
                let value = actual[sample * 9 + harmonic];
                assert!(
                    (value - expected[harmonic]).abs() < 2e-6,
                    "sample {sample} harmonic {harmonic}: {value} != {}",
                    expected[harmonic]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn spectra_match_official_centered_reflection_and_periodic_hann() -> Result<()> {
        let f = fixture();
        let signal = Tensor::from_vec(f.signal.clone(), f.signal.len(), &Device::Cpu)?;
        let (magnitude, phase) = stft(&signal)?;
        let actual = magnitude.flatten_all()?.to_vec1::<f32>()?;
        for (value, expected) in actual.iter().zip(&f.magnitude) {
            assert!(
                (value - expected).abs() < 2e-6,
                "magnitude {value} != {expected}"
            );
        }
        // Angles close to a spectral zero are ill-conditioned. Compare complex
        // bins so phase wrapping and numerical zeros do not create false failures.
        let phases = phase.flatten_all()?.to_vec1::<f32>()?;
        for ((value, angle), (expected, expected_angle)) in actual
            .iter()
            .zip(phases)
            .zip(f.magnitude.iter().zip(f.phase))
        {
            let a = Complex32::new(angle.cos(), angle.sin()) * *value;
            let b = Complex32::new(expected_angle.cos(), expected_angle.sin()) * *expected;
            assert!((a - b).norm() < 3e-6);
        }
        let restored = istft(&magnitude, &phase)?.to_vec1::<f32>()?;
        assert_eq!(restored.len(), f.reconstructed.len());
        for (actual, expected) in restored.iter().zip(f.reconstructed) {
            assert!((actual - expected).abs() < 2e-6);
        }
        Ok(())
    }
}
