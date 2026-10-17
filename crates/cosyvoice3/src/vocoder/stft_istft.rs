//! 16-point STFT / iSTFT for the HiFT vocoder.
//!
//! Matrix-DFT implementations matching `torch.stft` /
//! `torch.istft` with `n_fft=16, hop=4, win=16`, hann window
//! (`fftbins=True`), `center=True` and `pad_mode='reflect'`.

use std::f64::consts::PI;
use std::sync::OnceLock;

use candle_core::{DType, Device, Result, Tensor};

const N_FFT: usize = 16;
const HOP: usize = 4;
const FREQ_BINS: usize = N_FFT / 2 + 1;

fn dft_tables() -> &'static (Vec<f32>, Vec<f32>) {
    static TABLES: OnceLock<(Vec<f32>, Vec<f32>)> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut re = vec![0.0f32; N_FFT * N_FFT];
        let mut im = vec![0.0f32; N_FFT * N_FFT];
        for k in 0..N_FFT {
            for m in 0..N_FFT {
                let angle = 2.0 * PI * (k * m) as f64 / N_FFT as f64;
                re[k * N_FFT + m] = angle.cos() as f32;
                im[k * N_FFT + m] = (-angle.sin()) as f32;
            }
        }
        (re, im)
    })
}

fn hann_periodic() -> Vec<f32> {
    (0..N_FFT)
        .map(|i| (0.5 * (1.0 - (2.0 * PI * i as f64 / N_FFT as f64).cos())) as f32)
        .collect()
}

/// STFT of a mono waveform `[B, samples]` -> `(real, imag)` each
/// `[B, 9, n_frames]`, centered with reflect padding.
#[derive(Debug, Clone)]
pub struct HiFTStft {
    dft_real: Tensor,
    dft_imag: Tensor,
    window: Tensor,
}

impl HiFTStft {
    pub fn new(device: &Device, dtype: DType) -> Result<Self> {
        let (re, im) = dft_tables();
        let dft_real = Tensor::from_slice(re, (N_FFT, N_FFT), device)?.to_dtype(dtype)?;
        let dft_imag = Tensor::from_slice(im, (N_FFT, N_FFT), device)?.to_dtype(dtype)?;
        let window = Tensor::from_vec(hann_periodic(), N_FFT, device)?.to_dtype(dtype)?;
        Ok(Self {
            dft_real,
            dft_imag,
            window,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = x
            .to_device(self.dft_real.device())?
            .to_dtype(self.dft_real.dtype())?;
        let (b, samples) = x.dims2()?;
        let pad = N_FFT / 2;
        let padded = super::causal_conv::reflection_pad_1d(&x.reshape((b, 1, samples))?, pad, pad)?
            .reshape((b, samples + 2 * pad))?;
        let total = padded.dim(1)?;
        let n_frames = (total - N_FFT) / HOP + 1;

        let frames: Vec<Tensor> = (0..n_frames)
            .map(|i| padded.narrow(1, i * HOP, N_FFT))
            .collect::<Result<Vec<_>>>()?;
        let frames = Tensor::stack(&frames, 1)?; // [B, F, 16]
        let window = self.window.unsqueeze(0)?.unsqueeze(0)?;
        let windowed = frames
            .broadcast_mul(&window.broadcast_as(frames.shape())?)?
            .contiguous()?;

        let re = windowed.broadcast_matmul(&self.dft_real)?;
        let im = windowed.broadcast_matmul(&self.dft_imag)?;
        Ok((
            re.narrow(2, 0, FREQ_BINS)?.transpose(1, 2)?,
            im.narrow(2, 0, FREQ_BINS)?.transpose(1, 2)?,
        ))
    }
}

/// iSTFT: magnitude `[B, 9, T]` (clamped to max 100, as `_istft`) and phase
/// `[B, 9, T]` -> waveform `[B, (T-1)*4]`, torch center=True alignment.
#[derive(Debug, Clone)]
pub struct HiFTiStft {
    idft_real: Tensor,
    idft_imag: Tensor,
    window: Tensor,
}

impl HiFTiStft {
    pub fn new(device: &Device, dtype: DType) -> Result<Self> {
        let mut re = vec![0.0f32; N_FFT * N_FFT];
        let mut im = vec![0.0f32; N_FFT * N_FFT];
        for k in 0..N_FFT {
            for m in 0..N_FFT {
                let angle = 2.0 * PI * (k * m) as f64 / N_FFT as f64;
                // IDFT: (1/N) exp(+2πikm/N)
                re[k * N_FFT + m] = (angle.cos() / N_FFT as f64) as f32;
                im[k * N_FFT + m] = (angle.sin() / N_FFT as f64) as f32;
            }
        }
        let idft_real = Tensor::from_slice(&re, (N_FFT, N_FFT), device)?.to_dtype(dtype)?;
        let idft_imag = Tensor::from_slice(&im, (N_FFT, N_FFT), device)?.to_dtype(dtype)?;
        let window = Tensor::from_vec(hann_periodic(), N_FFT, device)?.to_dtype(dtype)?;
        Ok(Self {
            idft_real,
            idft_imag,
            window,
        })
    }

    pub fn forward(&self, magnitude: &Tensor, phase: &Tensor) -> Result<Tensor> {
        let device = self.idft_real.device();
        let dtype = self.idft_real.dtype();
        let mag = magnitude
            .clamp(f32::MIN as f64, 100.0)?
            .to_device(device)?
            .to_dtype(dtype)?;
        let phase = phase.to_device(device)?.to_dtype(dtype)?;

        let real = (&mag * phase.cos()?)?;
        let imag = (&mag * phase.sin()?)?;

        // Conjugate-symmetric completion [B, 9, T] -> [B, 16, T].
        let real_full = complete_spectrum(&real, false)?;
        let imag_full = complete_spectrum(&imag, true)?;

        // [B, T, 16] frames; IDFT takes the real part of the complex matmul.
        let real_full = real_full.transpose(1, 2)?.contiguous()?;
        let imag_full = imag_full.transpose(1, 2)?.contiguous()?;
        let frames = (real_full.broadcast_matmul(&self.idft_real)?
            - imag_full.broadcast_matmul(&self.idft_imag)?)?;

        // Synthesis window + overlap-add + sum(window^2) normalization.
        let (b, n_frames, _) = frames.dims3()?;
        let window = self.window.unsqueeze(0)?.unsqueeze(0)?;
        let frames = frames.broadcast_mul(&window.broadcast_as(frames.shape())?)?;
        let frames: Vec<f32> = frames
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?;

        let win: Vec<f32> = self
            .window
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1()?;
        let ola_len = (n_frames - 1) * HOP + N_FFT;
        let mut out = vec![0.0f32; b * ola_len];
        let mut wsum = vec![0.0f32; ola_len];
        for i in 0..n_frames {
            let base = i * HOP;
            for j in 0..N_FFT {
                wsum[base + j] += win[j] * win[j];
            }
        }
        for v in wsum.iter_mut() {
            if *v < 1e-8 {
                *v = 1.0;
            }
        }
        for bi in 0..b {
            for i in 0..n_frames {
                let base = i * HOP;
                for j in 0..N_FFT {
                    out[bi * ola_len + base + j] += frames[bi * n_frames * N_FFT + i * N_FFT + j];
                }
            }
            for (i, v) in out[bi * ola_len..(bi + 1) * ola_len].iter_mut().enumerate() {
                *v /= wsum[i];
            }
        }

        // center=True: drop n_fft//2 from both ends.
        let trim = N_FFT / 2;
        let signal = Tensor::from_vec(out, (b, ola_len), &Device::Cpu)?;
        let signal = if ola_len > 2 * trim {
            signal.narrow(1, trim, ola_len - 2 * trim)?
        } else {
            signal
        };
        signal
            .to_device(magnitude.device())?
            .to_dtype(magnitude.dtype())
    }
}

/// Mirror bins 1..8 into 9..15; `conjugate` negates the mirror and zeroes
/// DC/Nyquist imaginary parts.
fn complete_spectrum(half: &Tensor, conjugate: bool) -> Result<Tensor> {
    let (b, _, t) = half.dims3()?;
    let data: Vec<f32> = half
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let mut full = vec![0.0f32; b * N_FFT * t];
    for bi in 0..b {
        for ti in 0..t {
            for f in 0..FREQ_BINS {
                let v = data[bi * FREQ_BINS * t + f * t + ti];
                let v = if conjugate && (f == 0 || f == FREQ_BINS - 1) {
                    0.0
                } else {
                    v
                };
                full[bi * N_FFT * t + f * t + ti] = v;
            }
            let sign = if conjugate { -1.0 } else { 1.0 };
            for i in 0..(N_FFT - FREQ_BINS) {
                let src = FREQ_BINS - 2 - i; // 7..1
                let dst = FREQ_BINS + i; // 9..15
                full[bi * N_FFT * t + dst * t + ti] =
                    sign * data[bi * FREQ_BINS * t + src * t + ti];
            }
        }
    }
    let out = Tensor::from_vec(full, (b, N_FFT, t), &Device::Cpu)?;
    out.to_device(half.device())?.to_dtype(half.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_istft_roundtrip_recovers_signal() -> Result<()> {
        let device = Device::Cpu;
        let stft = HiFTStft::new(&device, DType::F32)?;
        let istft = HiFTiStft::new(&device, DType::F32)?;
        // A smooth low-frequency signal reconstructs well under hann+COLA.
        let n = 200;
        let wav: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin() as f32 * 0.5)
            .collect();
        let x = Tensor::from_vec(wav.clone(), (1, n), &device)?;
        let (re, im) = stft.forward(&x)?;
        let mag = (re.sqr()? + im.sqr()?)?.sqrt()?;
        let im_v: Vec<f32> = im.flatten_all()?.to_vec1()?;
        let re_v: Vec<f32> = re.flatten_all()?.to_vec1()?;
        let phase_v: Vec<f32> = im_v
            .iter()
            .zip(re_v.iter())
            .map(|(&y, &x)| y.atan2(x))
            .collect();
        let phase = Tensor::from_vec(phase_v, im.shape(), im.device())?;
        let y = istft.forward(&mag, &phase)?;
        let yv: Vec<f32> = y.squeeze(0)?.to_vec1()?;
        // Compare the interior (boundary frames reflect-padded).
        let m = yv.len().min(n);
        let mut max_err = 0.0f32;
        for i in 0..m {
            max_err = max_err.max((yv[i] - wav[i]).abs());
        }
        assert!(max_err < 2e-2, "roundtrip error {max_err}");
        Ok(())
    }

    #[test]
    fn istft_output_length_matches_torch() -> Result<()> {
        let device = Device::Cpu;
        let istft = HiFTiStft::new(&device, DType::F32)?;
        let mag = Tensor::ones((1, FREQ_BINS, 10), DType::F32, &device)?;
        let phase = Tensor::zeros((1, FREQ_BINS, 10), DType::F32, &device)?;
        let y = istft.forward(&mag, &phase)?;
        assert_eq!(y.dims(), &[1, 36]); // (10-1)*4
        Ok(())
    }
}
