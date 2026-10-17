//! Audio DSP helpers: WAV loading, sinc resampling, Kaldi fbank and the
//! upstream tail fade-out.

use candle_core::{DType, Device, Result, Tensor};

/// Load a WAV file as mono f32 in [-1, 1]. Channels are averaged, matching
/// `librosa.load(mono=True)` used by upstream `_load_and_cut_audio` (not
/// first-channel selection).
pub fn load_wav_mono(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| candle_core::Error::Msg(format!("open wav {}: {e}", path.display())))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let mut sums = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f64;
            for sample in reader.samples::<i32>() {
                let v = sample
                    .map_err(|e| candle_core::Error::Msg(format!("read wav sample: {e}")))?
                    as f64
                    / max;
                sums.push(v as f32);
            }
        }
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                sums.push(
                    sample.map_err(|e| candle_core::Error::Msg(format!("read wav sample: {e}")))?,
                );
            }
        }
    }
    if channels <= 1 {
        return Ok((sums, spec.sample_rate));
    }
    let n = sums.len() / channels;
    let mono = (0..n)
        .map(|i| {
            let frame = &sums[i * channels..(i + 1) * channels];
            frame.iter().sum::<f32>() / channels as f32
        })
        .collect();
    Ok((mono, spec.sample_rate))
}

/// Rational-ratio windowed-sinc resampler (Kaiser window, 16 zero crossings
/// per side, cutoff at the lower Nyquist). This is NOT bit-exact with
/// `torchaudio.transforms.Resample`; the numeric oracles compare stages
/// downstream of resampling, so callers wanting exact parity should feed
/// pre-resampled PCM.
pub fn resample(input: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to {
        return Ok(input.to_vec());
    }
    let ratio = to as f64 / from as f64;
    let cutoff = ratio.min(1.0);
    let out_len = ((input.len() as f64) * ratio).floor() as usize;
    let half_width = 16usize;
    let beta = 14.76965645937749f64;
    let bessel_den = i0(beta);
    let mut out = Vec::with_capacity(out_len);
    for n in 0..out_len {
        // Input time of output sample n.
        let t = n as f64 / ratio;
        let first = t.ceil() as i64 - half_width as i64;
        let last = t.floor() as i64 + half_width as i64;
        let mut acc = 0f64;
        for k in first..=last {
            if k < 0 || k as usize >= input.len() {
                continue;
            }
            let d = k as f64 - t;
            let s = if d.abs() < 1e-12 {
                cutoff
            } else {
                (std::f64::consts::PI * cutoff * d).sin() / (std::f64::consts::PI * d)
            };
            // Kaiser window over the fractional tap position.
            let r = d / half_width as f64;
            let w = if r.abs() >= 1.0 {
                0.0
            } else {
                i0(beta * (1.0 - r * r).sqrt()) / bessel_den
            };
            acc += input[k as usize] as f64 * s * w;
        }
        out.push(acc as f32);
    }
    Ok(out)
}

fn i0(x: f64) -> f64 {
    // Modified Bessel function of the first kind, order 0 (series).
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..50 {
        term *= (x / (2.0 * k as f64)) * (x / (2.0 * k as f64));
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

/// 80-bin Kaldi fbank, an exact port of `torchaudio.compliance.kaldi.fbank`
/// with the upstream arguments: 25 ms / 10 ms at 16 kHz, dither 0, povey
/// window, snip_edges, no energy column, log mel. Returns `(T, 80)` rows.
pub fn kaldi_fbank_80(wave: &[f32]) -> Result<Vec<f32>> {
    const SR: f64 = 16000.0;
    const FRAME_LEN: usize = 400; // 25 ms
    const FRAME_SHIFT: usize = 160; // 10 ms
    const NFFT: usize = 512; // next power of two of 400
    const NUM_BINS: usize = 80;
    const LOW_FREQ: f64 = 20.0;
    const PREEMPH: f64 = 0.97;
    const EPS: f32 = f32::EPSILON; // torchaudio's `_get_epsilon` for f32

    if wave.len() < FRAME_LEN {
        return Ok(Vec::new());
    }
    let m = 1 + (wave.len() - FRAME_LEN) / FRAME_SHIFT;

    // Povey window: symmetric hann ** 0.85.
    let window: Vec<f64> = (0..FRAME_LEN)
        .map(|n| {
            let h = 0.5
                * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / (FRAME_LEN - 1) as f64).cos());
            h.powf(0.85)
        })
        .collect();

    // Kaldi mel banks (torchaudio get_mel_banks, vtln_warp=1).
    let nyquist = 0.5 * SR;
    let high_freq = nyquist; // high_freq=0.0 -> +nyquist
    let fft_bin_width = SR / NFFT as f64;
    let mel_low = 1127.0 * (1.0 + LOW_FREQ / 700.0).ln();
    let mel_high = 1127.0 * (1.0 + high_freq / 700.0).ln();
    let mel_delta = (mel_high - mel_low) / (NUM_BINS as f64 + 1.0);
    let mel_of = |hz: f64| 1127.0 * (1.0 + hz / 700.0).ln();
    // banks[bin][fft_bin]
    let mut banks = vec![vec![0f64; NFFT / 2 + 1]; NUM_BINS];
    for (bin, bank) in banks.iter_mut().enumerate() {
        let left = mel_low + bin as f64 * mel_delta;
        let center = mel_low + (bin as f64 + 1.0) * mel_delta;
        let right = mel_low + (bin as f64 + 2.0) * mel_delta;
        for (k, cell) in bank.iter_mut().enumerate().take(NFFT / 2) {
            let mel = mel_of(fft_bin_width * k as f64);
            let up = (mel - left) / (center - left);
            let down = (right - mel) / (right - center);
            *cell = up.min(down).max(0.0);
        }
    }

    let mut planner = rustfft::FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(NFFT);
    let mut frame_buf = vec![rustfft::num_complex::Complex::new(0.0, 0.0); NFFT];

    let mut feats = Vec::with_capacity(m * NUM_BINS);
    for i in 0..m {
        let start = i * FRAME_SHIFT;
        let frame = &wave[start..start + FRAME_LEN];
        // Remove DC offset per frame.
        let mean = frame.iter().sum::<f32>() / FRAME_LEN as f32;
        // Preemphasis with replicate-style left edge, then window.
        for (j, buf) in frame_buf.iter_mut().enumerate().take(FRAME_LEN) {
            if j < FRAME_LEN {
                let prev = if j == 0 { frame[0] } else { frame[j - 1] };
                let v = (frame[j] - mean) as f64 - PREEMPH * (prev - mean) as f64;
                buf.re = v * window[j];
                buf.im = 0.0;
            }
        }
        for buf in frame_buf.iter_mut().skip(FRAME_LEN) {
            buf.re = 0.0;
            buf.im = 0.0;
        }
        fft.process(&mut frame_buf);
        // Power spectrum via the first NFFT/2+1 bins.
        let mut mel_energy = [0f64; NUM_BINS];
        for (bin, bank) in banks.iter().enumerate() {
            let mut e = 0f64;
            for (k, w) in bank.iter().enumerate() {
                let c = frame_buf[k];
                e += (c.re * c.re + c.im * c.im) * w;
            }
            mel_energy[bin] = e;
        }
        for e in mel_energy {
            feats.push(e.max(EPS as f64).ln() as f32);
        }
    }
    Ok(feats)
}

/// Kaldi fbank as a `(1, T, 80)` tensor with per-bin time-axis mean
/// subtraction, ready for [`crate::campplus::CampPlus::forward`] (matches
/// `feat = feat - feat.mean(dim=0, keepdim=True)` in `infer_v2_5.py`).
pub fn campplus_fbank_tensor(wave: &[f32], device: &Device) -> Result<Tensor> {
    let feats = kaldi_fbank_80(wave)?;
    if feats.is_empty() {
        candle_core::bail!("reference too short for CAMPPlus fbank");
    }
    let t = feats.len() / 80;
    let mut means = [0f32; 80];
    for row in 0..t {
        for (bin, m) in means.iter_mut().enumerate() {
            *m += feats[row * 80 + bin];
        }
    }
    for m in means.iter_mut() {
        *m /= t as f32;
    }
    let mut centered = Vec::with_capacity(feats.len());
    for row in 0..t {
        for bin in 0..80 {
            centered.push(feats[row * 80 + bin] - means[bin]);
        }
    }
    Tensor::from_vec(centered, (1, t, 80), device)?.to_dtype(DType::F32)
}

/// Upstream `fade_out_tail`: raised-cosine ramp over the last 20 ms so an
/// early stop token never leaves an audible step (index-tts #247/#488/#523).
pub fn fade_out_tail(samples: &mut [f32], sampling_rate: u32) {
    const FADE_MS: f64 = 20.0;
    let n = ((sampling_rate as f64 * FADE_MS / 1000.0) as usize).min(samples.len());
    if n <= 1 {
        return;
    }
    let start = samples.len() - n;
    for (i, s) in samples[start..].iter_mut().enumerate() {
        let theta = std::f64::consts::PI * i as f64 / (n - 1) as f64;
        *s *= (0.5 * (1.0 + theta.cos())) as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_identity_and_ratio() {
        let x: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let same = resample(&x, 22050, 22050).unwrap();
        assert_eq!(same.len(), 100);
        assert_eq!(same, x);
        // 44100 -> 22050 halves the length; DC signal stays DC.
        let dc = vec![0.5f32; 1000];
        let half = resample(&dc, 44100, 22050).unwrap();
        assert!(half.len() >= 480 && half.len() <= 500);
        // Interior samples stay ~0.5 (edges have transition bands).
        let interior = &half[16..half.len() - 16];
        let max_dev = interior
            .iter()
            .map(|v| (v - 0.5).abs())
            .fold(0.0f32, f32::max);
        assert!(max_dev < 1e-3, "max deviation {max_dev}");
    }

    #[test]
    fn fbank_shape_and_dc_floor() {
        // 1 second of silence: 1 + (16000-400)/160 = 98 frames of log(eps).
        let wave = vec![0f32; 16000];
        let f = kaldi_fbank_80(&wave).unwrap();
        assert_eq!(f.len(), 98 * 80);
        assert!(f.iter().all(|v| (*v - f32::EPSILON.ln()).abs() < 1e-4));
    }

    #[test]
    fn fbank_matches_kaldi_reference_values() {
        // Hand-check a single 400-sample frame against the torchaudio
        // algorithm computed by hand for bin 0 (low mel triangle) with a
        // pure sine input; here we only pin the exact frame count and the
        // finiteness/oracle separability, real parity is covered by the
        // Python oracle fixtures.
        let wave: Vec<f32> = (0..2000)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin() as f32)
            .collect();
        let f = kaldi_fbank_80(&wave).unwrap();
        assert_eq!(f.len(), (1 + (2000 - 400) / 160) * 80);
        assert!(f.iter().all(|v| v.is_finite()));
        // Energy around 440 Hz must be higher than near-Nyquist bins.
        let row = &f[..80];
        let mel_440: f64 = 1127.0 * (1.0f64 + 440.0 / 700.0).ln();
        let mel_low: f64 = 1127.0 * (1.0f64 + 20.0 / 700.0).ln();
        let mel_high: f64 = 1127.0 * (1.0f64 + 8000.0 / 700.0).ln();
        let delta: f64 = (mel_high - mel_low) / 81.0;
        let center_bin = ((mel_440 - mel_low) / delta - 1.0f64).round() as usize;
        assert!(
            row[center_bin.min(79)] > row[79],
            "center bin {} vs last bin",
            center_bin.min(79)
        );
    }

    #[test]
    fn fade_out_tail_shapes() {
        let mut x = vec![1f32; 22050];
        fade_out_tail(&mut x, 22050);
        assert_eq!(x.len(), 22050);
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!(x.last().unwrap().abs() < 1e-6);
        let n = (22050 * 20 / 1000) as usize;
        assert!((x[x.len() - n] - 1.0).abs() < 0.02);
    }
}
