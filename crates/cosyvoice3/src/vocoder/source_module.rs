//! NSF source module: `SineGen2` + harmonic merge
//! (`cosyvoice/hifigan/generator.py`).
//!
//! Causal inference uses fixed pseudo-random initial phase and noise
//! buffers (`torch.rand` at module construction in the official code; not
//! part of the checkpoint). We generate them from an injectable seed so
//! runs are reproducible; fixture runs can override the buffers.

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::activations::Tanh;
use crate::flow::embeddings::LinT2d;
use crate::vocoder::causal_conv::repeat_interleave_dim;

/// 300 s at 24 kHz of pre-drawn noise, matching the official buffers.
pub const MAX_AUDIO_SAMPLES: usize = 300 * 24_000;

#[derive(Debug, Clone)]
pub struct SineGen2 {
    upsample_scale: usize,
    harmonic_num: usize,
    sine_amp: f64,
    noise_std: f64,
    voiced_threshold: f64,
    /// `[1, H]` fixed initial phase; fundamental stays 0.
    rand_ini: Tensor,
    /// `[1, MAX, H]` fixed noise in [0, 1).
    sine_waves_noise: Tensor,
    /// `[1, 1, H]` harmonic multipliers `1..=H`.
    harmonics: Tensor,
    /// Scalar `sampling_rate` tensor on device.
    rate: Tensor,
}

impl SineGen2 {
    /// Override buffers with shared official tensors for numerical comparisons.
    pub fn set_noise(&mut self, initial_phase: Tensor, noise: Tensor) -> Result<()> {
        if initial_phase.dims() != [1, self.harmonic_num + 1]
            || noise.dim(0)? != 1
            || noise.dim(2)? != self.harmonic_num + 1
        {
            candle_core::bail!("invalid SineGen2 phase/noise shape");
        }
        self.rand_ini = initial_phase;
        self.sine_waves_noise = noise;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sampling_rate: usize,
        upsample_scale: usize,
        harmonic_num: usize,
        sine_amp: f64,
        noise_std: f64,
        voiced_threshold: f64,
        seed: u64,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let mut rng = crate::rng::TorchCpuRng::new(seed);
        let h = harmonic_num + 1;
        let mut rand_ini = rng.rand_vec(h);
        rand_ini[0] = 0.0;
        let rand_ini = Tensor::from_vec(rand_ini, (1, h), device)?;
        let sine_waves_noise = Tensor::from_vec(
            rng.rand_vec(MAX_AUDIO_SAMPLES * h),
            (1, MAX_AUDIO_SAMPLES, h),
            device,
        )?;
        let harmonics_vec: Vec<f32> = (1..=h).map(|i| i as f32).collect();
        let harmonics = Tensor::from_vec(harmonics_vec, (1, 1, h), device)?;
        let rate = Tensor::new(sampling_rate as f64, &candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?
            .to_device(device)?;
        Ok(Self {
            upsample_scale,
            harmonic_num,
            sine_amp,
            noise_std,
            voiced_threshold,
            rand_ini,
            sine_waves_noise,
            harmonics,
            rate,
        })
    }

    /// `f0`: `[B, T, 1]` already upsampled to audio rate. Returns
    /// `(sine_waves [B, T, H], uv [B, T])`.
    pub fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor)> {
        let device = f0.device();
        let dtype = f0.dtype();
        let (b, t, _) = f0.dims3()?;
        let h = self.harmonic_num + 1;

        // harmonics: f0 * [1..=H]
        let harmonics = self.harmonics.to_device(device)?.to_dtype(dtype)?;
        let fn_mat = f0.broadcast_mul(&harmonics)?;

        // rad = (fn / sr) % 1, computed as x - floor(x).
        // Scalar `/` in Candle multiplies by a rounded reciprocal. The
        // official true division rounds differently; phase accumulation
        // amplifies that one-ULP difference across the harmonics.
        let rate = self.rate.to_device(device)?.to_dtype(dtype)?;
        let rad_raw = fn_mat.broadcast_div(&rate)?;
        let rad = (&rad_raw - rad_raw.floor()?)?;

        // fixed initial phase on the first frame
        let rand_ini = self
            .rand_ini
            .to_device(device)?
            .to_dtype(dtype)?
            .broadcast_as((b, 1, h))?;
        let first = (rad.narrow(1, 0, 1)? + &rand_ini)?;
        let rad = if t > 1 {
            let rest = rad.narrow(1, 1, t - 1)?;
            Tensor::cat(&[&first, &rest], 1)?
        } else {
            first
        };

        // Downsample linearly to frame rate, cumsum phase, then upsample
        // with nearest (causal) — matching `_f02sine`.
        if b != 1 || t == 0 || t % self.upsample_scale != 0 {
            candle_core::bail!(
                "SineGen2 requires batch=1 and a nonempty whole number of mel frames"
            );
        }
        let down_t = t / self.upsample_scale;
        let rad_down = interp_linear_time(&rad, down_t)?;
        let phase_down = cumsum_f32_torch(&rad_down)?.affine(2.0 * std::f64::consts::PI, 0.0)?;
        let phase_scaled = phase_down.affine(self.upsample_scale as f64, 0.0)?;
        let phase = interp_nearest_time(&phase_scaled, t)?;

        let sine = phase.sin()?.affine(self.sine_amp, 0.0)?;

        let uv = f0
            .squeeze(2)?
            .gt(self.voiced_threshold)?
            .to_dtype(dtype)?
            .unsqueeze(2)?;
        let one_minus_uv = uv.affine(-1.0, 1.0)?; // 1 - uv, [B, T, 1]
        let noise_amp = &(&uv.affine(self.noise_std, 0.0)?
            + &one_minus_uv.affine(self.sine_amp / 3.0, 0.0)?)?;
        let noise_slice = self
            .sine_waves_noise
            .narrow(1, 0, t)?
            .to_device(device)?
            .to_dtype(dtype)?;
        let noise = apply_noise_amplitude(noise_amp, &noise_slice, b, t, h)?;

        let sine = (sine.broadcast_mul(&uv)? + noise)?;
        let uv = uv.squeeze(2)?;
        Ok((sine, uv))
    }
}

/// `noise_slice * noise_amp` with an explicit broadcast only for `b > 1`
/// (the production shapes are `[1, T, H]` x `[B, T, 1]`).
fn apply_noise_amplitude(
    noise_amp: &Tensor,
    noise_slice: &Tensor,
    b: usize,
    t: usize,
    h: usize,
) -> Result<Tensor> {
    if b == 1 {
        noise_slice.broadcast_mul(noise_amp)
    } else {
        noise_amp
            .broadcast_as((b, t, h))?
            .broadcast_mul(&noise_slice.broadcast_as((b, t, h))?)
    }
}

/// Cumsum along dim 1 of `[B, T, C]` f32 with torch CPU semantics: the
/// accumulation runs in f64 and each prefix sum is emitted as f32. Candle's
/// sequential f32 `cumsum` drifts ~1e-4 over a few hundred frames, which the
/// ×480 phase upscale turns into ~5e-2 rad of pitch phase error.
fn cumsum_f32_torch(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != candle_core::DType::F32 || x.dims().len() != 3 {
        candle_core::bail!("cumsum_f32_torch expects a rank-3 f32 tensor");
    }
    let device = x.device().clone();
    let src = x
        .to_device(&candle_core::Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (b, t, c) = x.dims3()?;
    let mut out = vec![0f32; src.len()];
    for bi in 0..b {
        for ci in 0..c {
            let mut acc = 0f64;
            for ti in 0..t {
                acc += f64::from(src[(bi * t + ti) * c + ci]);
                out[(bi * t + ti) * c + ci] = acc as f32;
            }
        }
    }
    Tensor::from_vec(out, (b, t, c), &candle_core::Device::Cpu)?.to_device(&device)
}

/// `F.interpolate(x, scale_factor=s, mode='linear')` over the time axis of
/// `[B, T, C]` (align_corners=False, area-pixel mapping `(i+0.5)/s-0.5`).
/// Implemented on CPU f32 vectors for exact PyTorch parity.
pub fn interp_linear_time(x: &Tensor, out_t: usize) -> Result<Tensor> {
    let (b, t, c) = x.dims3()?;
    if t == 0 || out_t == 0 {
        candle_core::bail!("linear interpolation requires nonempty input and output");
    }
    let device = x.device();
    let dtype = x.dtype();
    let src: Vec<f32> = x
        .to_dtype(candle_core::DType::F32)?
        .to_device(&candle_core::Device::Cpu)?
        .flatten_all()?
        .to_vec1()?;
    let scale = t as f64 / out_t as f64;
    let mut out = vec![0.0f32; b * out_t * c];
    for bi in 0..b {
        for i in 0..out_t {
            let src_pos = (i as f64 + 0.5) * scale - 0.5;
            let f0 = src_pos.floor();
            let lambda = (src_pos - f0) as f32;
            let i0 = f0 as isize;
            for ci in 0..c {
                let v = if i0 < 0 {
                    src[bi * t * c + ci]
                } else if (i0 as usize) + 1 >= t {
                    src[bi * t * c + (t - 1) * c + ci]
                } else {
                    let a = src[bi * t * c + i0 as usize * c + ci];
                    let bb = src[bi * t * c + (i0 as usize + 1) * c + ci];
                    a + (bb - a) * lambda
                };
                out[bi * out_t * c + i * c + ci] = v;
            }
        }
    }
    Tensor::from_vec(out, (b, out_t, c), &candle_core::Device::Cpu)?
        .to_device(device)?
        .to_dtype(dtype)
}

/// Nearest upsample on the time axis of `[B, T, C]` with integer scale.
pub fn interp_nearest_time(x: &Tensor, out_t: usize) -> Result<Tensor> {
    let t = x.dim(1)?;
    if t > 0 && out_t.is_multiple_of(t) {
        return repeat_interleave_dim(x, 1, out_t / t);
    }
    let scale = t as f64 / out_t as f64;
    let indices: Vec<u32> = (0..out_t)
        .map(|i| ((i as f64 * scale).floor() as usize).min(t - 1) as u32)
        .collect();
    let indices = Tensor::from_vec(indices, out_t, x.device())?;
    x.index_select(&indices, 1)
}

#[derive(Debug)]
pub struct SourceModuleHnNSF {
    sine_gen: SineGen2,
    l_linear: LinT2d,
    tanh: Tanh,
}

impl SourceModuleHnNSF {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sampling_rate: usize,
        upsample_scale: usize,
        harmonic_num: usize,
        sine_amp: f64,
        add_noise_std: f64,
        voiced_threshold: f64,
        seed: u64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let sine_gen = SineGen2::new(
            sampling_rate,
            upsample_scale,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold,
            seed,
            vb.device(),
        )?;
        Ok(Self {
            sine_gen,
            l_linear: LinT2d::from_linear(candle_nn::linear(
                harmonic_num + 1,
                1,
                vb.pp("l_linear"),
            )?)?,
            tanh: Tanh,
        })
    }

    /// `f0`: `[B, T, 1]`. Returns `(sine_merge [B, T, 1], uv [B, T])`.
    pub fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor)> {
        let (sine_wavs, uv) = self.sine_gen.forward(f0)?;
        let merged = self.tanh.forward(&self.l_linear.forward(&sine_wavs)?)?;
        Ok((merged, uv))
    }

    pub fn set_noise(&mut self, initial_phase: Tensor, noise: Tensor) -> Result<()> {
        self.sine_gen.set_noise(initial_phase, noise)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn harmonic_source_matches_independent_official_torch_oracle() -> Result<()> {
        #[derive(serde::Deserialize)]
        struct Oracle {
            f0: Vec<f32>,
            waves: Vec<Vec<f32>>,
        }
        let oracle: Oracle =
            serde_json::from_str(include_str!("../../tests/fixtures/torch-sine-source.json"))
                .unwrap();
        let device = Device::Cpu;
        let frames = oracle.f0.len();
        let samples = frames * 480;
        let mut sine = SineGen2::new(24000, 480, 8, 0.1, 0.003, 10., 0, &device)?;
        sine.set_noise(
            Tensor::zeros((1, 9), DType::F32, &device)?,
            Tensor::zeros((1, samples, 9), DType::F32, &device)?,
        )?;
        let hz = oracle
            .f0
            .iter()
            .flat_map(|&hz| std::iter::repeat_n(hz, 480))
            .collect::<Vec<_>>();
        let f0 = Tensor::from_vec(hz, (1, samples, 1), &device)?;
        let (waves, _) = sine.forward(&f0)?;
        let indices = Tensor::from_vec(
            (0..frames).map(|i| (i * 480) as u32).collect(),
            frames,
            &device,
        )?;
        let actual = waves
            .index_select(&indices, 1)?
            .squeeze(0)?
            .to_vec2::<f32>()?;
        for (actual, expected) in actual.iter().flatten().zip(oracle.waves.iter().flatten()) {
            assert!((actual - expected).abs() < 3e-7, "{actual} != {expected}");
        }
        // HiFT replays each accumulated prefix from sample zero. Phase
        // must stay anchored there rather than restart at the emitted tail.
        for prefix_frames in 1..frames {
            let count = prefix_frames * 480;
            let (prefix, _) = sine.forward(&f0.narrow(1, 0, count)?)?;
            assert_eq!(
                prefix.flatten_all()?.to_vec1::<f32>()?,
                waves.narrow(1, 0, count)?.flatten_all()?.to_vec1::<f32>()?,
                "source phase changed for {prefix_frames} frames"
            );
        }
        Ok(())
    }

    #[test]
    fn cumsum_accumulates_in_f64_not_sequential_f32() -> Result<()> {
        // Regression for the SineGen2 phase drift: 0.1 summed 100k times
        // in f32 drifts ~1e-4; torch (and this helper) must not.
        let n = 100_000;
        let x = Tensor::full(0.1f32, (1, n, 1), &Device::Cpu)?;
        let out = cumsum_f32_torch(&x)?;
        let v: Vec<f32> = out.flatten_all()?.to_vec1()?;
        let mut acc = 0f64;
        let src = f64::from(0.1f32);
        for (i, got) in v.iter().enumerate() {
            acc += src;
            assert_eq!(*got, acc as f32, "prefix {i}");
        }
        // And the drift would be visible with plain f32 accumulation.
        let mut f32acc = 0f32;
        for _ in 0..n {
            f32acc += 0.1;
        }
        assert!((f32acc - v[n - 1]).abs() > 1e-5);
        let _ = DType::F32;
        Ok(())
    }

    #[test]
    fn linear_downsample_identity_when_scale_one() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::arange(0f32, 12.0, &device)?.reshape((1, 12, 1))?;
        let y = interp_linear_time(&x, 12)?;
        let v: Vec<f32> = y.flatten_all()?.to_vec1()?;
        for (i, &o) in v.iter().enumerate() {
            assert!((o - i as f32).abs() < 1e-5);
        }
        let _ = DType::F32;
        Ok(())
    }

    #[test]
    fn linear_downsample_samples_area_centers() -> Result<()> {
        let device = Device::Cpu;
        // 8 samples of a ramp downsampled to 1: src pos = 0.5*8-0.5 = 3.5
        // -> mean of x[3] and x[4].
        let x = Tensor::arange(0f32, 8.0, &device)?.reshape((1, 8, 1))?;
        let y: Vec<f32> = interp_linear_time(&x, 1)?.flatten_all()?.to_vec1()?;
        assert!((y[0] - 3.5).abs() < 1e-5);
        Ok(())
    }
}
