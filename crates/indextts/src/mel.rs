//! Mel spectrogram (`indextts/s2mel/modules/audio.py`, `mel_spectrogram`).
//!
//! Torch-compatible STFT: Hann window (periodic), no center padding
//! (`center=false` is forced by upstream), reflect-style frame truncation
//! matching `torch.stft` boundary behavior, Slaney-style mel filterbank
//! (`librosa.filters.mel` defaults: `htk=false`, `norm="slaney"`, `fmax`
//! defaults to `sr/2`), magnitudes `sqrt(|stft|^2 + 1e-9)` and dynamic-range
//! `spectrogram = magnitude.clamp(min=1e-5).log()`.
//!
//! The audio.py wrapper additionally pads the waveform by
//! `(n_fft - hop_size) / 2` on both sides with `mode="reflect"` before the
//! STFT; that padding is reproduced here regardless of `center`.

use candle_core::{DType, Device, Result, Tensor};
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

/// Slaney-mel scale constants shared by `librosa.filters.mel`.
const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP; // 15.0

/// `ln(6.4) / 27` (not const because `f64::ln` is not a const fn).
fn log_step() -> f64 {
    6.4f64.ln() / 27.0
}

fn hz_to_mel(freq: f64) -> f64 {
    let mel = freq / F_SP;
    if freq >= MIN_LOG_HZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / log_step()
    } else {
        mel
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * log_step()).exp()
    } else {
        mel * F_SP
    }
}

/// Port of `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` with the
/// default `htk=False` scale and `norm="slaney"`. Returns the `(n_mels,
/// n_fft/2+1)` matrix flattened row-major.
fn slaney_mel_basis(
    sampling_rate: u32,
    n_fft: usize,
    num_mels: usize,
    fmin: f64,
    fmax: Option<f64>,
) -> Result<Vec<f32>> {
    let fmax = fmax.unwrap_or(sampling_rate as f64 / 2.0);
    if !(fmin >= 0.0 && fmax > fmin) {
        candle_core::bail!("invalid mel range: fmin {fmin} fmax {fmax}");
    }
    let n_bins = n_fft / 2 + 1;
    // fft_frequencies: linspace(0, sr/2, 1 + n_fft/2).
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sampling_rate as f64 / n_fft as f64)
        .collect();
    // mel_frequencies(n_mels + 2, fmin, fmax).
    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);
    let mel_f: Vec<f64> = (0..num_mels + 2)
        .map(|i| mel_to_hz(min_mel + (max_mel - min_mel) * i as f64 / (num_mels + 1) as f64))
        .collect();

    let mut basis = vec![0f32; num_mels * n_bins];
    for i in 0..num_mels {
        let fdiff_lower = mel_f[i + 1] - mel_f[i];
        let fdiff_upper = mel_f[i + 2] - mel_f[i + 1];
        // Slaney norm: unit area per triangle.
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - mel_f[i]) / fdiff_lower;
            let upper = (mel_f[i + 2] - f) / fdiff_upper;
            basis[i * n_bins + k] = (lower.min(upper).max(0.0) * enorm) as f32;
        }
    }
    Ok(basis)
}

/// Periodic Hann window, matching `torch.hann_window(win_size)` (the
/// `periodic=True` default): `0.5 - 0.5*cos(2*pi*i/win_size)`.
fn hann_window(win_size: usize) -> Vec<f32> {
    (0..win_size)
        .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / win_size as f64).cos()) as f32)
        .collect()
}

/// `torch.nn.functional.pad(x, (left, right), mode="reflect")` on one row:
/// the edge sample itself is never repeated.
fn reflect_pad(x: &[f32], left: usize, right: usize) -> Result<Vec<f32>> {
    if left == 0 && right == 0 {
        return Ok(x.to_vec());
    }
    let n = x.len();
    if left >= n || right >= n {
        candle_core::bail!(
            "reflect padding ({left}, {right}) must be smaller than the signal length {n}"
        );
    }
    let mut out = vec![0f32; n + left + right];
    out[left..left + n].copy_from_slice(x);
    // Left mirror never repeats the edge sample: out[i] = x[left - i].
    for i in 0..left {
        out[left - 1 - i] = x[i + 1];
    }
    // Right mirror: out[left + n + i] = x[n - 2 - i].
    for i in 0..right {
        out[left + n + i] = x[n - 2 - i];
    }
    Ok(out)
}

/// `torch.stft` with `center=False, onesided=True` magnitudes:
/// `sqrt(re^2 + im^2 + 1e-9)` per (frame, bin). Rows are `(bins, frames)`
/// column-major-in-frames, i.e. frame-major `[frame0_bins, frame1_bins, ..]`.
fn stft_magnitude(x: &[f32], window: &[f32], n_fft: usize, hop: usize) -> Result<Vec<f32>> {
    if x.len() < n_fft {
        candle_core::bail!(
            "stft input length {} is shorter than n_fft {n_fft}",
            x.len()
        );
    }
    let frames = 1 + (x.len() - n_fft) / hop;
    let n_bins = n_fft / 2 + 1;
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buf = vec![Complex::<f32>::default(); n_fft];
    let mut out = Vec::with_capacity(frames * n_bins);
    for t in 0..frames {
        for (i, slot) in buf.iter_mut().enumerate() {
            slot.re = x[t * hop + i] * window[i];
            slot.im = 0.0;
        }
        fft.process(&mut buf);
        for c in &buf[..n_bins] {
            out.push((c.re * c.re + c.im * c.im + 1e-9).sqrt());
        }
    }
    Ok(out)
}

/// `wav`: `(B, N)` f32 in [-1, 1]. Returns `(B, n_mels, frames)`.
#[allow(clippy::too_many_arguments)]
pub fn mel_spectrogram(
    wav: &Tensor,
    n_fft: usize,
    win_size: usize,
    hop_size: usize,
    num_mels: usize,
    sampling_rate: u32,
    fmin: f64,
    fmax: Option<f64>,
    center: bool,
) -> Result<Tensor> {
    if n_fft == 0 || hop_size == 0 || hop_size > n_fft {
        candle_core::bail!("invalid stft parameters: n_fft {n_fft}, hop {hop_size}");
    }
    if win_size == 0 || win_size > n_fft {
        candle_core::bail!("win_size {win_size} must be in 1..={n_fft}");
    }
    let (batch, _samples) = wav.dims2()?;
    let device = wav.device().clone();
    let rows = wav
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;

    // Window padded to n_fft (torch.stft zero-pads a shorter centered window).
    let mut window = vec![0f32; n_fft];
    let win_left = (n_fft - win_size) / 2;
    for (i, w) in hann_window(win_size).into_iter().enumerate() {
        window[win_left + i] = w;
    }

    // audio.py always pads (n_fft - hop) / 2 reflect on both sides; with
    // center=true torch.stft adds n_fft/2 more (pad_mode "reflect").
    let pre_pad = (n_fft - hop_size) / 2;
    let center_pad = if center { n_fft / 2 } else { 0 };
    let mut frames = 0usize;
    let mut spectra = Vec::with_capacity(batch);
    for row in &rows {
        let row = reflect_pad(row, pre_pad, pre_pad)?;
        let row = reflect_pad(&row, center_pad, center_pad)?;
        let mag = stft_magnitude(&row, &window, n_fft, hop_size)?;
        frames = mag.len() / (n_fft / 2 + 1);
        spectra.push(mag);
    }

    let n_bins = n_fft / 2 + 1;
    let mut flat = Vec::with_capacity(batch * frames * n_bins);
    for mag in &spectra {
        flat.extend_from_slice(mag);
    }
    // Magnitudes are frame-major; move to (B, bins, frames) for the matmul.
    let spec = Tensor::from_vec(flat, (batch, frames, n_bins), &Device::Cpu)?.transpose(1, 2)?;
    let basis = slaney_mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax)?;
    let basis = Tensor::from_vec(basis, (num_mels, n_bins), &Device::Cpu)?;
    // Per-batch 2-D matmuls: a stride-0 broadcast of the basis as a batched
    // matmul lhs is not reliable across candle backends.
    let mut mels = Vec::with_capacity(batch);
    for b in 0..batch {
        mels.push(basis.matmul(&spec.narrow(0, b, 1)?.squeeze(0)?)?);
    }
    let mel = Tensor::cat(&mels, 0)?.reshape((batch, num_mels, frames))?;
    // spectral_normalize_torch: log(clamp(min=1e-5)).
    let mel = mel.clamp(1e-5, f64::INFINITY)?.log()?;
    mel.to_device(&device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    #[test]
    fn hann_window_is_periodic() {
        let w = hann_window(8);
        let expect = [
            0.0f32,
            0.5 - 0.5 * (2.0 * PI / 8.0).cos() as f32,
            0.5,
            0.5 + 0.5 * (2.0 * PI / 8.0).cos() as f32,
            1.0,
        ];
        for (got, want) in w.iter().zip(expect.iter()) {
            assert!(close(*got, *want, 1e-6), "{w:?}");
        }
        // Symmetric around the 1.0 peak, excluding both endpoints.
        for i in 1..4 {
            assert!(close(w[i], w[8 - i], 1e-6));
        }
    }

    #[test]
    fn reflect_pad_matches_torch() {
        // torch: F.pad([a,b,c,d], (2,2), mode="reflect") == [c,b,a,b,c,d,c,b]
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = reflect_pad(&x, 2, 2).unwrap();
        assert_eq!(y, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
        // (n_fft - hop)/2 = 64 on a 256-sample signal.
        let long = vec![0.5f32; 256];
        let y = reflect_pad(&long, 64, 64).unwrap();
        assert_eq!(y.len(), 384);
        assert!(y.iter().all(|v| *v == 0.5));
        // torch errors when pad >= length; so do we.
        assert!(reflect_pad(&[1.0f32; 3], 3, 0).is_err());
    }

    #[test]
    fn mel_basis_matches_librosa() {
        // Oracles from librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80,
        // fmin=0, fmax=None).
        let m = slaney_mel_basis(22050, 1024, 80, 0.0, None).unwrap();
        let n_bins = 513;
        let row0_head = [0.0f32, 0.01276074, 0.02316559, 0.01040485, 0.0, 0.0];
        for (k, want) in row0_head.iter().enumerate() {
            assert!(close(m[k], *want, 1e-5), "row0[{k}] {} vs {want}", m[k]);
        }
        assert_eq!(m[39 * n_bins + 512], 0.0);
        let row79_tail = [
            0.00052574f32,
            0.00042059,
            0.00031545,
            0.0002103,
            0.00010515,
            0.0,
        ];
        for (k, want) in row79_tail.iter().enumerate() {
            let idx = 79 * n_bins + (n_bins - 6 + k);
            assert!(close(m[idx], *want, 1e-4), "row79 tail {k}");
        }
        for (pos, row) in [0usize, 40, 79].into_iter().enumerate() {
            let sum: f32 = m[row * n_bins..(row + 1) * n_bins].iter().sum();
            let want = [0.04633117f32, 0.04664324, 0.04643372][pos];
            assert!(close(sum, want, 1e-5), "row {row} sum {sum}");
        }
        assert!(m.iter().all(|v| *v >= 0.0));

        // Second config used by mel_spectrogram tests: sr=16000, n_fft=256.
        let m = slaney_mel_basis(16000, 256, 80, 0.0, None).unwrap();
        let n_bins = 129;
        // Row 0 only touches bin 1; row 79 spans bins 119..127.
        assert!(close(m[1], 0.00863771, 1e-5), "16000 row0 bin1 {}", m[1]);
        let want79 = [
            (119usize, 0.0003376f32),
            (120, 0.00106623),
            (121, 0.00179487),
            (122, 0.00252351),
            (127, 0.00070119),
        ];
        for (k, want) in want79 {
            let got = m[79 * n_bins + k];
            assert!(
                close(got, want, 1e-4),
                "16000 row79 bin{k}: {got} vs {want}"
            );
        }
        for (pos, row) in [0usize, 40, 79].into_iter().enumerate() {
            let sum: f32 = m[row * n_bins..(row + 1) * n_bins].iter().sum();
            let want = [0.00863771f32, 0.01592092, 0.01598622][pos];
            assert!(close(sum, want, 1e-4), "16000 row {row} sum {sum}");
        }
    }

    #[test]
    fn stft_magnitude_matches_naive_dft() {
        // 256-sample sine (8 cycles) padded like audio.py: 64 reflect both
        // sides -> 384 samples, hop 128, n_fft 256 -> 2 frames.
        let n = 256usize;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 8.0 * i as f64 / n as f64).sin() as f32)
            .collect();
        let x = reflect_pad(&x, 64, 64).unwrap();
        let window = hann_window(256);
        let mag = stft_magnitude(&x, &window, 256, 128).unwrap();
        assert_eq!(mag.len(), 2 * 129);

        // Naive DFT: X_k = sum_n w[n] x[n] e^{-2 pi i k n / N}.
        let naive = |frame: usize, k: usize| -> f32 {
            let off = frame * 128;
            let mut re = 0f64;
            let mut im = 0f64;
            for j in 0..256 {
                let v = (x[off + j] * window[j]) as f64;
                let ang = -2.0 * PI * k as f64 * j as f64 / 256.0;
                re += v * ang.cos();
                im += v * ang.sin();
            }
            ((re * re + im * im) as f32 + 1e-9).sqrt()
        };
        for (frame, k) in [(0usize, 0usize), (0, 8), (0, 16), (1, 100)] {
            let got = mag[frame * 129 + k];
            let want = naive(frame, k);
            assert!(
                close(got, want, 1e-4),
                "frame {frame} bin {k}: {got} vs {want}"
            );
        }
        // Oracle magnitudes from torch.stft on the same input.
        for (idx, want) in [
            (0usize, 5.157427f32),
            (8, 52.466904),
            (16, 1.776582),
            (129 + 100, 0.055881),
        ] {
            assert!(close(mag[idx], want, 1e-5), "mag[{idx}] {}", mag[idx]);
        }
    }

    #[test]
    fn stft_magnitude_cosine_high_bins() {
        // 256-sample cosine (16 cycles): frame 1 crosses the reflect pad and
        // carries real leakage at the top FFT bins; verified against an f64
        // naive DFT and torch.stft (f32).
        let n = 256usize;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 16.0 * i as f64 / n as f64).cos() as f32)
            .collect();
        let x = reflect_pad(&x, 64, 64).unwrap();
        let mag = stft_magnitude(&x, &hann_window(256), 256, 128).unwrap();
        for (k, want) in [(16usize, 62.14865f32), (123, 0.03914), (127, 0.03899)] {
            let got = mag[129 + k];
            assert!(close(got, want, 1e-3), "f1 bin {k}: {got} vs {want}");
        }
    }

    #[test]
    fn mel_spectrogram_matches_audio_py() {
        // Two-channel 256-sample input (sine + cosine), n_fft=256, hop=128,
        // sr=16000, fmax=None -> 4 frames of pre-pad reflect + 2 STFT frames.
        let n = 256usize;
        let a: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 8.0 * i as f64 / n as f64).sin() as f32)
            .collect();
        let b: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 16.0 * i as f64 / n as f64).cos() as f32)
            .collect();
        let mut both = Vec::with_capacity(2 * n);
        both.extend_from_slice(&a);
        both.extend_from_slice(&b);
        let wav = Tensor::from_vec(both, (2, n), &Device::Cpu).unwrap();
        let out = mel_spectrogram(&wav, 256, 256, 128, 80, 16000, 0.0, None, false).unwrap();
        assert_eq!(out.dims(), vec![2, 80, 2]);
        let v = out.to_vec3::<f32>().unwrap();
        // Oracles from indextts mel_spectrogram on the same input.
        let head4 = [
            -3.109924f32,
            -3.103957,
            -2.363778,
            -2.357811,
            -2.388526,
            -2.382322,
            -2.978343,
            -2.972138,
        ];
        for (i, want) in head4.iter().enumerate() {
            let mel = i / 2;
            let frame = i % 2;
            assert!(
                close(v[0][mel][frame], *want, 1e-4),
                "b0 m{mel} f{frame}: got {} want {want}",
                v[0][mel][frame]
            );
        }
        for (frame, want) in [(0usize, -11.512925f32), (1, -7.376298)] {
            assert!(
                close(v[1][79][frame], want, 1e-3),
                "b1 m79 f{frame}: got {} want {want}",
                v[1][79][frame]
            );
        }
    }

    #[test]
    fn mel_spectrogram_center_adds_half_window_reflect() {
        // 256 samples, n_fft=128, hop=64: pre-pad 32 -> 320; center pads 64
        // more -> 448; frames = 1 + (448 - 128)/64 = 6.
        let wav = Tensor::zeros((1, 256), DType::F32, &Device::Cpu).unwrap();
        let out = mel_spectrogram(&wav, 128, 128, 64, 80, 16000, 0.0, None, true).unwrap();
        assert_eq!(out.dims(), vec![1, 80, 6]);
        // Without center: 1 + (320 - 128)/64 = 4 frames.
        let out = mel_spectrogram(&wav, 128, 128, 64, 80, 16000, 0.0, None, false).unwrap();
        assert_eq!(out.dims(), vec![1, 80, 4]);
        // All-zero input lands on the 1e-5 clamp: log(1e-5).
        let v = out.to_vec3::<f32>().unwrap();
        assert!(v[0].iter().flatten().all(|x| close(*x, 1e-5f32.ln(), 1e-4)));
    }

    #[test]
    fn mel_spectrogram_rejects_short_reflect_pad() {
        // (1024 - 256)/2 = 384 > 256 samples: torch F.pad errors, so do we.
        let wav = Tensor::zeros((1, 256), DType::F32, &Device::Cpu).unwrap();
        assert!(mel_spectrogram(&wav, 1024, 1024, 256, 80, 22050, 0.0, None, false).is_err());
    }
}
