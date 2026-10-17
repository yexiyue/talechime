//! Frontend audio features.
//!
//! Three extractors, each pinned to an upstream reference:
//!
//! * matcha mel (80 bins, 24 kHz, hop 480, `center=False` with reflect
//!   side pads) — the flow prompt feature, math from
//!   `matcha.utils.audio.mel_spectrogram` with librosa slaney filters.
//! * whisper log-mel (128 bins, 16 kHz, hop 160) — the FSQ tokenizer
//!   input, matching `whisper.log_mel_spectrogram` including the
//!   `max-8` clamp and `(x+4)/4` normalization.
//! * kaldi fbank (80 bins, 16 kHz, dither 0) — the campplus input,
//!   matching `torchaudio.compliance.kaldi.fbank`.

use candle_core::{DType, Device, Result, Tensor};
use std::f64::consts::PI;

fn fft_complex(input: &[f32]) -> Vec<f32> {
    let n = input.len() / 2;
    if n == 1 {
        return input.to_vec();
    }
    if n % 2 == 1 {
        return dft_complex(input);
    }
    let mut even = Vec::with_capacity(n);
    let mut odd = Vec::with_capacity(n);
    for i in 0..n / 2 {
        even.push(input[i * 4]);
        even.push(input[i * 4 + 1]);
        odd.push(input[i * 4 + 2]);
        odd.push(input[i * 4 + 3]);
    }
    let even_fft = fft_complex(&even);
    let odd_fft = fft_complex(&odd);
    let mut out = vec![0.0f32; n * 2];
    for k in 0..n / 2 {
        let angle = 2.0 * PI * k as f64 / n as f64;
        let c = angle.cos() as f32;
        let s = (-angle.sin()) as f32;
        let or = odd_fft[k * 2];
        let oi = odd_fft[k * 2 + 1];
        let tr = c * or - s * oi;
        let ti = c * oi + s * or;
        out[k * 2] = even_fft[k * 2] + tr;
        out[k * 2 + 1] = even_fft[k * 2 + 1] + ti;
        out[(k + n / 2) * 2] = even_fft[k * 2] - tr;
        out[(k + n / 2) * 2 + 1] = even_fft[k * 2 + 1] - ti;
    }
    out
}

fn dft_complex(input: &[f32]) -> Vec<f32> {
    let n = input.len() / 2;
    let mut out = vec![0.0f32; n * 2];
    for k in 0..n {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for j in 0..n {
            let angle = 2.0 * PI * k as f64 * j as f64 / n as f64;
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            re += input[j * 2] * c + input[j * 2 + 1] * s;
            im += -input[j * 2] * s + input[j * 2 + 1] * c;
        }
        out[k * 2] = re;
        out[k * 2 + 1] = im;
    }
    out
}

/// librosa `filters.mel` (slaney mel scale + slaney area norm).
fn librosa_mel_filters(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: Option<f64>,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let mut filters = vec![0.0f32; n_mels * n_freqs];
    let fmax = fmax.unwrap_or(sample_rate as f64 / 2.0);

    let min_log_hz = 1000.0f64;
    let min_log_mel = 15.0f64;
    let logstep = 6.4f64.ln() / 27.0;
    let hz_to_mel = |hz: f64| {
        if hz >= min_log_hz {
            min_log_mel + (hz / min_log_hz).ln() / logstep
        } else {
            3.0 * hz / 200.0
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel >= min_log_mel {
            min_log_hz * ((mel - min_log_mel) * logstep).exp()
        } else {
            200.0 * mel / 3.0
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let hz_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    for m in 0..n_mels {
        let (left, center, right) = (hz_points[m], hz_points[m + 1], hz_points[m + 2]);
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let w = if freq >= left && freq < center {
                (freq - left) / (center - left).max(1e-10)
            } else if freq >= center && freq <= right {
                (right - freq) / (right - center).max(1e-10)
            } else {
                0.0
            };
            filters[m * n_freqs + k] = w as f32;
        }
        let enorm = 2.0 / (right - left).max(1e-10);
        for k in 0..n_freqs {
            filters[m * n_freqs + k] *= enorm as f32;
        }
    }
    filters
}

fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos())) as f32)
        .collect()
}

/// torch `F.pad(x, (l, r), 'reflect')` on a raw sample buffer.
fn reflect_pad_samples(x: &[f32], left: usize, right: usize) -> Result<Vec<f32>> {
    let n = x.len();
    if n <= left || n <= right {
        candle_core::bail!("audio length {n} is too short for reflect padding ({left}, {right})");
    }
    let mut out = Vec::with_capacity(left + n + right);
    for i in 0..left {
        out.push(x[left - i]);
    }
    out.extend_from_slice(x);
    for i in 0..right {
        out.push(x[n - 2 - i]);
    }
    Ok(out)
}

/// Matcha-style mel for the flow prompt feature. Output `[1, 80, T]`.
#[derive(Debug, Clone)]
pub struct MatchaMel {
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    filters: Vec<f32>,
    window: Vec<f32>,
    device: Device,
}

impl MatchaMel {
    pub fn new() -> Self {
        Self {
            n_fft: 1920,
            hop: 480,
            n_mels: 80,
            filters: librosa_mel_filters(24_000, 1920, 80, 0.0, None),
            window: hann_periodic(1920),
            device: Device::Cpu,
        }
    }

    /// `audio`: `[T]` f32 at 24 kHz (mono, peak-scaled).
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        let samples: Vec<f32> = audio
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1()?;
        let pad = (self.n_fft - self.hop) / 2; // 720 each side, reflect
        let padded = reflect_pad_samples(&samples, pad, pad)?;
        let n_frames = if padded.len() >= self.n_fft {
            (padded.len() - self.n_fft) / self.hop + 1
        } else {
            0
        };
        let n_freqs = self.n_fft / 2 + 1;
        let mut mel = vec![0.0f32; self.n_mels * n_frames];
        for f in 0..n_frames {
            let start = f * self.hop;
            let mut input = vec![0.0f32; self.n_fft * 2];
            for i in 0..self.n_fft {
                input[i * 2] = padded[start + i] * self.window[i];
            }
            let spec = fft_complex(&input);
            let mut mag = vec![0.0f32; n_freqs];
            for k in 0..n_freqs {
                let re = spec[k * 2];
                let im = spec[k * 2 + 1];
                mag[k] = (re * re + im * im + 1e-9).sqrt();
            }
            for m in 0..self.n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_freqs {
                    sum += mag[k] * self.filters[m * n_freqs + k];
                }
                mel[m * n_frames + f] = sum.max(1e-5).ln();
            }
        }
        let t = Tensor::from_vec(mel, (self.n_mels, n_frames), &Device::Cpu)?;
        t.unsqueeze(0)?.to_device(&self.device)
    }
}

impl Default for MatchaMel {
    fn default() -> Self {
        Self::new()
    }
}

/// Whisper log-mel for the FSQ speech tokenizer. Output `[1, 128, T-1]`.
#[derive(Debug, Clone)]
pub struct WhisperMel {
    filters: Vec<f32>,
    window: Vec<f32>,
    device: Device,
}

impl WhisperMel {
    pub fn new() -> Self {
        Self {
            filters: librosa_mel_filters(16_000, 400, 128, 0.0, None),
            window: hann_periodic(400),
            device: Device::Cpu,
        }
    }

    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        let samples: Vec<f32> = audio
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1()?;
        let (n_fft, hop) = (400usize, 160usize);
        // torch.stft(center=True, pad_mode='reflect')
        let padded = reflect_pad_samples(&samples, n_fft / 2, n_fft / 2)?;
        let n_frames_all = if padded.len() >= n_fft {
            (padded.len() - n_fft) / hop + 1
        } else {
            0
        };
        // whisper drops the final frame
        let n_frames = n_frames_all.saturating_sub(1);
        let n_freqs = n_fft / 2 + 1;
        let n_mels = 128usize;
        let mut mel = vec![0.0f32; n_mels * n_frames];
        for f in 0..n_frames {
            let start = f * hop;
            let mut input = vec![0.0f32; n_fft * 2];
            for i in 0..n_fft {
                input[i * 2] = padded[start + i] * self.window[i];
            }
            let spec = fft_complex(&input);
            let mut power = vec![0.0f32; n_freqs];
            for k in 0..n_freqs {
                let re = spec[k * 2];
                let im = spec[k * 2 + 1];
                power[k] = re * re + im * im;
            }
            for m in 0..n_mels {
                let mut sum = 0.0f32;
                for (&p, &w) in power
                    .iter()
                    .zip(&self.filters[m * n_freqs..(m + 1) * n_freqs])
                {
                    sum += p * w;
                }
                mel[m * n_frames + f] = sum.max(1e-10).log10();
            }
        }
        // log_spec = maximum(log_spec, log_spec.max() - 8.0);
        // log_spec = (log_spec + 4.0) / 4.0
        if n_frames > 0 {
            let mut max = f32::NEG_INFINITY;
            for &v in mel.iter() {
                max = max.max(v);
            }
            let floor = max - 8.0;
            for v in mel.iter_mut() {
                *v = ((*v).max(floor) + 4.0) / 4.0;
            }
        }
        let t = Tensor::from_vec(mel, (n_mels, n_frames), &Device::Cpu)?;
        t.unsqueeze(0)?.to_device(&self.device)
    }
}

impl Default for WhisperMel {
    fn default() -> Self {
        Self::new()
    }
}

/// Kaldi fbank for campplus. Output `[T, 80]`.
#[derive(Debug, Clone)]
pub struct KaldiFbank {
    window_size: usize,
    hop: usize,
    padded_window: usize,
    preemphasis: f64,
    n_mels: usize,
    povey: Vec<f32>,
    filters: Vec<f32>,
    device: Device,
}

impl KaldiFbank {
    pub fn new() -> Self {
        let sample_rate = 16_000usize;
        let n_mels = 80usize;
        let window_size: usize = 400; // 25 ms
        let hop = 160; // 10 ms
        let padded_window = window_size.next_power_of_two(); // 512
        let povey: Vec<f32> = (0..window_size)
            .map(|i| {
                let hann = 0.5 - 0.5 * (2.0 * PI * i as f64 / (window_size - 1) as f64).cos();
                (hann.powf(0.85)) as f32
            })
            .collect();
        let filters = Self::htk_filters(sample_rate, padded_window, n_mels, 20.0, 8000.0);
        Self {
            window_size,
            hop,
            padded_window,
            preemphasis: 0.97,
            n_mels,
            povey,
            filters,
            device: Device::Cpu,
        }
    }

    fn htk_filters(
        sample_rate: usize,
        padded_window: usize,
        n_mels: usize,
        low: f64,
        high: f64,
    ) -> Vec<f32> {
        // torchaudio get_mel_banks: bins over n_fft/2, nyquist excluded.
        let n_bins = padded_window / 2;
        let mut filters = vec![0.0f32; n_mels * (n_bins + 1)];
        let hz_to_mel = |hz: f64| 1127.0 * (1.0 + hz / 700.0).ln();
        let mel_low = hz_to_mel(low);
        let mel_high = hz_to_mel(high);
        let delta = (mel_high - mel_low) / (n_mels + 1) as f64;
        let bin_width = sample_rate as f64 / padded_window as f64;
        for m in 0..n_mels {
            let left = mel_low + m as f64 * delta;
            let center = mel_low + (m + 1) as f64 * delta;
            let right = mel_low + (m + 2) as f64 * delta;
            for k in 0..n_bins {
                let k_mel = hz_to_mel(k as f64 * bin_width);
                let up = (k_mel - left) / (center - left);
                let down = (right - k_mel) / (right - center);
                filters[m * (n_bins + 1) + k] = up.min(down).max(0.0) as f32;
            }
        }
        filters
    }

    /// `audio`: `[T]` f32 at 16 kHz (mono, NOT mean-centered — campplus
    /// centers per utterance afterwards).
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, audio: &Tensor) -> Result<Tensor> {
        let samples: Vec<f32> = audio
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1()?;
        let n_frames = if samples.len() >= self.window_size {
            1 + (samples.len() - self.window_size) / self.hop
        } else {
            0
        };
        let n_bins = self.padded_window / 2 + 1;
        let filter_bins = self.padded_window / 2;
        let mut fbank = vec![0.0f32; n_frames * self.n_mels];
        for f in 0..n_frames {
            let start = f * self.hop;
            let mut frame: Vec<f32> = (0..self.window_size)
                .map(|i| samples.get(start + i).copied().unwrap_or(0.0))
                .collect();
            // remove_dc_offset
            let mean = frame.iter().sum::<f32>() / frame.len() as f32;
            for v in frame.iter_mut() {
                *v -= mean;
            }
            // pre-emphasis with replicate left edge
            let alpha = self.preemphasis as f32;
            let mut prev = frame[0];
            for v in frame.iter_mut() {
                let cur = *v;
                *v = cur - alpha * prev;
                prev = cur;
            }
            for (i, v) in frame.iter_mut().enumerate() {
                *v *= self.povey[i];
            }
            frame.resize(self.padded_window, 0.0);
            let mut input = vec![0.0f32; self.padded_window * 2];
            for (i, &v) in frame.iter().enumerate() {
                input[i * 2] = v;
            }
            let spec = fft_complex(&input);
            let mut power = vec![0.0f32; n_bins];
            for k in 0..n_bins {
                let re = spec[k * 2];
                let im = spec[k * 2 + 1];
                power[k] = re * re + im * im;
            }
            for m in 0..self.n_mels {
                let mut sum = 0.0f32;
                for k in 0..n_bins {
                    sum += power[k] * self.filters[m * (filter_bins + 1) + k];
                }
                fbank[f * self.n_mels + m] = sum.max(f32::EPSILON).ln();
            }
        }
        Tensor::from_vec(fbank, (n_frames, self.n_mels), &self.device)
    }
}

impl Default for KaldiFbank {
    fn default() -> Self {
        Self::new()
    }
}

/// torchaudio-style sinc resample (rational, width 6, rolloff 0.99).
#[allow(clippy::needless_range_loop)]
pub fn resample(audio: &Tensor, from_sr: usize, to_sr: usize) -> Result<Tensor> {
    if from_sr == 0 || to_sr == 0 {
        candle_core::bail!("sample rates must be positive");
    }
    if from_sr == to_sr {
        return Ok(audio.clone());
    }
    let samples: Vec<f32> = audio
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec1()?;
    let gcd = gcd(from_sr, to_sr);
    let orig = from_sr / gcd;
    let new = to_sr / gcd;
    let base = orig.min(new) as f64 * 0.99;
    let width = (6.0 * orig as f64 / base).ceil() as usize;

    let kernel_size = 2 * width + orig;
    let mut kernels: Vec<Vec<f64>> = Vec::with_capacity(new);
    for phase in 0..new {
        let mut kernel = Vec::with_capacity(kernel_size);
        let phase_offset = -(phase as f64) / new as f64;
        for k in 0..kernel_size {
            let idx = (k as i64 - width as i64) as f64 / orig as f64;
            let t = (phase_offset + idx) * base;
            let t = t.clamp(-6.0, 6.0);
            let window = (t * PI / 12.0).cos().powi(2);
            let t_pi = t * PI;
            let sinc = if t_pi.abs() < 1e-10 {
                1.0
            } else {
                t_pi.sin() / t_pi
            };
            kernel.push(sinc * window * (base / orig as f64));
        }
        kernels.push(kernel);
    }

    let padded_len = samples.len() + 2 * width + orig;
    let mut padded = vec![0.0f64; padded_len];
    for (i, &s) in samples.iter().enumerate() {
        padded[width + i] = s as f64;
    }
    let out_len = (samples.len() * new).div_ceil(orig);
    let mut out = Vec::with_capacity(out_len);
    for o in 0..out_len {
        let phase = o % new;
        let base_idx = (o / new) * orig;
        let mut sum = 0.0f64;
        for (k, &w) in kernels[phase].iter().enumerate() {
            let i = base_idx + k;
            if i < padded_len {
                sum += padded[i] * w;
            }
        }
        out.push(sum as f32);
    }
    let n = out.len();
    Tensor::from_vec(out, n, audio.device())
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_filters_peak_in_range() {
        let filters = librosa_mel_filters(24_000, 1920, 80, 0.0, None);
        assert_eq!(filters.len(), 80 * 961);
        // First and last filters should have non-zero mass.
        let first: f32 = filters[..961].iter().sum();
        let last: f32 = filters[79 * 961..].iter().sum();
        assert!(first > 0.0 && last > 0.0);
    }

    #[test]
    fn whisper_mel_normalization_applied() -> Result<()> {
        let device = Device::Cpu;
        // Silence: all-zero audio -> log10(clamp)=−10, then normalized.
        let audio = Tensor::zeros((1600,), DType::F32, &device)?;
        let mel = WhisperMel::new().forward(&audio)?;
        assert_eq!(mel.dims(), &[1, 128, 10]);
        let v: Vec<f32> = mel.flatten_all()?.to_vec1()?;
        // max log10(1e-10) = -10 -> (-10-10+... wait: all equal -> max=-10
        // floor=-18; value (-10+4)/4 = -1.5
        assert!(
            v.iter().all(|&x| (x - (-1.5)).abs() < 1e-4),
            "{:?}",
            &v[..4]
        );
        Ok(())
    }

    #[test]
    fn resample_doubles_length() -> Result<()> {
        let device = Device::Cpu;
        let audio = Tensor::from_slice(
            &(0..100).map(|i| (i as f32 * 0.1).sin()).collect::<Vec<_>>(),
            100,
            &device,
        )?;
        let up = resample(&audio, 8000, 16_000)?;
        assert_eq!(up.dim(0)?, 200);
        let down = resample(&up, 16_000, 8000)?;
        assert_eq!(down.dim(0)?, 100);
        Ok(())
    }
}
