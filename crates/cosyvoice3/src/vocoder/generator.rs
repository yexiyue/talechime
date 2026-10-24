//! `CausalHiFTGenerator` — mel to 24 kHz waveform
//! (`cosyvoice/hifigan/generator.py`).
//!
//! Non-streaming (`finalize=true`) path first; the streamed chunk protocol
//! (conv_pre look-ahead context, hift mel cache) is a follow-up. Weight
//! names assume weight-norm-fused safetensors from the conversion script.

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::time::Instant;

use super::causal_conv::{
    CausalConv1d, CausalConv1dUpsample, CausalType, reflection_pad_1d, repeat_interleave_dim,
};
use super::f0_predictor::CausalConvRnnF0Predictor;
use super::source_module::SourceModuleHnNSF;
use super::stft_istft::{HiFTStft, HiFTiStft};
use crate::activations::{LeakyReLU, Snake};
use crate::config::HiftConfig;

/// Wall-clock stage attribution for one HiFT decode, enabled by
/// `CV3_HIFT_PROFILE=1`. The decode is sequential, so per-stage wall time
/// attributes without overlap. Zero cost when disabled. Diagnostics only —
/// never part of the timing contract.
#[derive(Debug, Default)]
struct HiftProfile {
    enabled: bool,
    f0_ms: f64,
    source_ms: f64,
    pre_ms: f64,
    up_ms: f64,
    sbranch_ms: f64,
    res_ms: f64,
    post_ms: f64,
    /// Finer split of `f0_ms`: condnet convs vs classifier.
    f0_cond_ms: f64,
    /// Finer split of `post_ms`: conv_post vs istft.
    post_conv_ms: f64,
    post_istft_ms: f64,
}

impl HiftProfile {
    fn from_env() -> Self {
        Self {
            enabled: std::env::var("CV3_HIFT_PROFILE").is_ok_and(|v| v != "0"),
            ..Default::default()
        }
    }

    #[inline]
    fn ms(start: Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1e3
    }

    fn line(&self, frames: usize) {
        if !self.enabled {
            return;
        }
        let total = self.f0_ms
            + self.source_ms
            + self.pre_ms
            + self.up_ms
            + self.sbranch_ms
            + self.res_ms
            + self.post_ms;
        eprintln!(
            "[cv3-hift] frames {frames} total {total:.0} ms | f0 {:.0} (cond {:.0} cls {:.0}) source {:.0} pre {:.0} up {:.0} sbranch {:.0} res {:.0} post {:.0} (conv {:.0} istft {:.0})",
            self.f0_ms,
            self.f0_cond_ms,
            self.f0_ms - self.f0_cond_ms,
            self.source_ms,
            self.pre_ms,
            self.up_ms,
            self.sbranch_ms,
            self.res_ms,
            self.post_ms,
            self.post_conv_ms,
            self.post_istft_ms,
        );
    }
}

#[derive(Debug)]
struct ResBlock {
    convs1: Vec<CausalConv1d>,
    convs2: Vec<CausalConv1d>,
    activations1: Vec<Snake>,
    activations2: Vec<Snake>,
}

impl ResBlock {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        let mut activations1 = Vec::with_capacity(dilations.len());
        let mut activations2 = Vec::with_capacity(dilations.len());
        for (i, &d) in dilations.iter().enumerate() {
            convs1.push(CausalConv1d::new(
                channels,
                channels,
                kernel_size,
                d,
                CausalType::Left,
                vb.pp(format!("convs1.{i}")),
            )?);
            convs2.push(CausalConv1d::new(
                channels,
                channels,
                kernel_size,
                1,
                CausalType::Left,
                vb.pp(format!("convs2.{i}")),
            )?);
            activations1.push(Snake::new(channels, vb.pp(format!("activations1.{i}")))?);
            activations2.push(Snake::new(channels, vb.pp(format!("activations2.{i}")))?);
        }
        Ok(Self {
            convs1,
            convs2,
            activations1,
            activations2,
        })
    }
}

impl Module for ResBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let xt = self.activations1[i].forward(&x)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.activations2[i].forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            x = (x + xt)?;
        }
        Ok(x)
    }
}

#[derive(Debug)]
pub struct CausalHiFtGenerator {
    cfg: HiftConfig,
    f0_predictor: CausalConvRnnF0Predictor,
    m_source: SourceModuleHnNSF,
    conv_pre: CausalConv1d,
    ups: Vec<CausalConv1dUpsample>,
    source_downs: Vec<SourceDown>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<Vec<ResBlock>>,
    conv_post: CausalConv1d,
    stft: HiFTStft,
    istft: HiFTiStft,
    lrelu: LeakyReLU,
}

/// Strided source downsample conv; `u == 1` collapses to a left-causal k1.
#[derive(Debug)]
struct SourceDown {
    conv: candle_nn::Conv1d,
    causal_padding: usize,
}

impl Module for SourceDown {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.causal_padding > 0 {
            x.pad_with_zeros(2, self.causal_padding, 0)?
        } else {
            x.clone()
        };
        self.conv.forward(&x)
    }
}

impl CausalHiFtGenerator {
    /// Share the official non-checkpoint source buffers during numerical validation.
    pub fn set_source_noise(&mut self, initial_phase: Tensor, noise: Tensor) -> Result<()> {
        self.m_source.set_noise(initial_phase, noise)
    }
    pub fn new(cfg: HiftConfig, seed: u64, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let upsample_scale: usize =
            cfg.upsample_rates.iter().product::<usize>() * cfg.istft_hop_len;

        let f0_predictor = CausalConvRnnF0Predictor::new(
            cfg.in_channels,
            512,
            vb.pp("f0_predictor")
                .set_device(candle_core::Device::Cpu)
                .set_dtype(candle_core::DType::F64),
        )?;
        let m_source = SourceModuleHnNSF::new(
            cfg.sampling_rate,
            upsample_scale,
            cfg.nb_harmonics,
            cfg.nsf_alpha,
            cfg.nsf_sigma,
            cfg.nsf_voiced_threshold,
            seed,
            vb.pp("m_source"),
        )?;

        let mut channel_sizes = vec![cfg.base_channels];
        for i in 0..cfg.upsample_rates.len() {
            channel_sizes.push(cfg.base_channels / (1 << (i + 1)));
        }

        let conv_pre = CausalConv1d::new(
            cfg.in_channels,
            cfg.base_channels,
            cfg.conv_pre_look_right + 1,
            1,
            CausalType::Right,
            vb.pp("conv_pre"),
        )?;

        let mut ups = Vec::with_capacity(cfg.upsample_rates.len());
        for (i, (&rate, &kernel)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            ups.push(CausalConv1dUpsample::new(
                channel_sizes[i],
                channel_sizes[i + 1],
                kernel,
                rate,
                vb.pp(format!("ups.{i}")),
            )?);
        }

        // downsample_rates = [1] + upsample_rates[::-1][:-1]; cumprod reversed.
        let mut downsample_rates = vec![1usize];
        let reversed: Vec<usize> = cfg.upsample_rates.iter().rev().copied().collect();
        if reversed.len() > 1 {
            downsample_rates.extend_from_slice(&reversed[..reversed.len() - 1]);
        }
        let mut cum: Vec<usize> = Vec::new();
        let mut acc = 1;
        for &r in &downsample_rates {
            acc *= r;
            cum.push(acc);
        }
        cum.reverse();

        let source_in = cfg.istft_n_fft + 2; // 18
        let mut source_downs = Vec::new();
        let mut source_resblocks = Vec::new();
        for (i, &u) in cum.iter().enumerate() {
            let out_ch = channel_sizes[i + 1];
            let (kernel, stride, causal_padding) =
                if u == 1 { (1, 1, 0) } else { (u * 2, u, u - 1) };
            let conv_cfg = candle_nn::Conv1dConfig {
                padding: 0,
                stride,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            source_downs.push(SourceDown {
                conv: candle_nn::conv1d(
                    source_in,
                    out_ch,
                    kernel,
                    conv_cfg,
                    vb.pp(format!("source_downs.{i}")),
                )?,
                causal_padding,
            });
            source_resblocks.push(ResBlock::new(
                out_ch,
                cfg.source_resblock_kernel_sizes[i],
                &cfg.source_resblock_dilation_sizes[i],
                vb.pp(format!("source_resblocks.{i}")),
            )?);
        }

        let num_kernels = cfg.resblock_kernel_sizes.len();
        let mut resblocks = Vec::new();
        for (i, _) in cfg.upsample_rates.iter().enumerate() {
            let ch = channel_sizes[i + 1];
            let mut stage = Vec::with_capacity(num_kernels);
            for (j, &kernel) in cfg.resblock_kernel_sizes.iter().enumerate() {
                stage.push(ResBlock::new(
                    ch,
                    kernel,
                    &cfg.resblock_dilation_sizes[j],
                    vb.pp(format!("resblocks.{}", i * num_kernels + j)),
                )?);
            }
            resblocks.push(stage);
        }

        let final_ch = *channel_sizes.last().unwrap();
        let conv_post = CausalConv1d::new(
            final_ch,
            cfg.istft_n_fft + 2,
            7,
            1,
            CausalType::Left,
            vb.pp("conv_post"),
        )?;

        Ok(Self {
            f0_predictor,
            m_source,
            conv_pre,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post,
            stft: HiFTStft::new(&device, dtype)?,
            istft: HiFTiStft::new(&device, dtype)?,
            lrelu: LeakyReLU::new(cfg.lrelu_slope),
            cfg,
        })
    }

    /// `mel`: `[B, 80, T]` -> waveform `[B, samples]`, clamped to
    /// `±audio_limit`. `finalize=false` trims the tail hop window as in the
    /// official streamed decode.
    pub fn inference(&self, mel: &Tensor, finalize: bool) -> Result<Tensor> {
        self.inference_inner(mel, finalize, None)
    }

    /// Inference that also records named stage tensors (f0, harmonic
    /// source, post-conv_pre mel branch, per-upstage fusion results) for
    /// parity probes. The computation is identical to [`Self::inference`].
    pub fn inference_traced(
        &self,
        mel: &Tensor,
        finalize: bool,
    ) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let mut trace = Vec::new();
        let wav = self.inference_inner(mel, finalize, Some(&mut trace))?;
        Ok((wav, trace))
    }

    fn inference_inner(
        &self,
        mel: &Tensor,
        finalize: bool,
        mut trace: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Tensor> {
        let (batch, channels, frames) = mel.dims3()?;
        if batch != 1
            || channels != self.cfg.in_channels
            || frames == 0
            || (!finalize
                && frames
                    <= self.f0_predictor.first_causal_padding() + self.cfg.conv_pre_look_right + 1)
        {
            candle_core::bail!(
                "HiFT requires batch=1, matching channels and nonempty mel (streaming must exceed lookahead plus the tail frame)"
            );
        }
        let dtype = mel.dtype();
        let prod_rates: usize = self.cfg.upsample_rates.iter().product();
        let upsample_scale: usize = prod_rates * self.cfg.istft_hop_len;
        let mut prof = HiftProfile::from_env();

        // 1. f0 in float64 (handled inside the predictor), then nearest
        //    upsample x upsample_scale: torch.nn.Upsample default mode.
        let t_stage = Instant::now();
        let f0 = self.f0_predictor.forward(mel, finalize)?; // [B, T]
        if let Some(trace) = trace.as_deref_mut() {
            trace.push(("f0".into(), f0.to_dtype(dtype)?));
        }
        let f0_up = repeat_interleave_dim(&f0.to_dtype(dtype)?.unsqueeze(2)?, 1, upsample_scale)?; // [B, T*scale, 1]
        if prof.enabled {
            mel.device().synchronize()?;
            prof.f0_ms += HiftProfile::ms(t_stage);
        }

        // 2. harmonic source + STFT of the source branch.
        let t_stage = Instant::now();
        let (source, _uv) = self.m_source.forward(&f0_up)?; // [B, Ts, 1]
        let source = source.squeeze(2)?.to_dtype(dtype)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.push(("source".into(), source.clone()));
        }
        let (mut s_re, mut s_im) = self.stft.forward(&source)?;
        if !finalize {
            // Trim is prod(upsample_rates) * conv_pre_look_right STFT frames.
            let trim = prod_rates * self.cfg.conv_pre_look_right;
            s_re = s_re.narrow(2, 0, s_re.dim(2)? - trim)?;
            s_im = s_im.narrow(2, 0, s_im.dim(2)? - trim)?;
        }
        let s_stft = Tensor::cat(&[&s_re, &s_im], 1)?;
        if prof.enabled {
            mel.device().synchronize()?;
            prof.source_ms += HiftProfile::ms(t_stage);
        }

        // 3. mel branch: conv_pre (look-right) then up stages. In streamed
        // mode Python trims the mel fed to `decode` by the f0 predictor's
        // first look-right padding before splitting the conv_pre context.
        let t_stage = Instant::now();
        let mut x = if finalize {
            self.conv_pre.forward(mel)?
        } else {
            let f0_trim = self.f0_predictor.first_causal_padding();
            let mel = mel.narrow(2, 0, mel.dim(2)?.saturating_sub(f0_trim))?;
            let look = self.cfg.conv_pre_look_right;
            let main = mel.narrow(2, 0, mel.dim(2)? - look)?;
            let context = mel.narrow(2, mel.dim(2)? - look, look)?;
            self.conv_pre.forward_with_cache(&main, &context)?
        };

        if let Some(trace) = trace.as_deref_mut() {
            trace.push(("x_pre".into(), x.clone()));
        }
        if prof.enabled {
            mel.device().synchronize()?;
            prof.pre_ms += HiftProfile::ms(t_stage);
        }
        for (i, up) in self.ups.iter().enumerate() {
            let t_stage = Instant::now();
            x = self.lrelu.forward(&x)?;
            x = up.forward(&x)?;
            if i == self.ups.len() - 1 {
                x = reflection_pad_1d(&x, 1, 0)?;
            }
            if prof.enabled {
                mel.device().synchronize()?;
                prof.up_ms += HiftProfile::ms(t_stage);
            }
            let t_stage = Instant::now();
            let si = self.source_downs[i].forward(&s_stft)?;
            let si = self.source_resblocks[i].forward(&si)?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.push((format!("si_{i}"), si.clone()));
            }
            if prof.enabled {
                mel.device().synchronize()?;
                prof.sbranch_ms += HiftProfile::ms(t_stage);
            }
            if x.dim(2)? != si.dim(2)? {
                candle_core::bail!(
                    "source fusion shape mismatch at stage {i}: {} vs {}",
                    x.dim(2)?,
                    si.dim(2)?
                );
            }
            x = (x + si)?;
            let t_stage = Instant::now();
            let mut acc: Option<Tensor> = None;
            for rb in &self.resblocks[i] {
                let y = rb.forward(&x)?;
                acc = Some(match acc {
                    None => y,
                    Some(a) => (a + y)?,
                });
            }
            x = (acc.unwrap() / self.resblocks[i].len() as f64)?;
            if prof.enabled {
                mel.device().synchronize()?;
                prof.res_ms += HiftProfile::ms(t_stage);
            }
            if let Some(trace) = trace.as_deref_mut() {
                trace.push((format!("x_stage{i}"), x.clone()));
            }
        }

        // 4. post: leaky_relu with the PyTorch default slope 0.01 (not
        //    lrelu_slope), then magnitude/phase split and iSTFT.
        let t_stage = Instant::now();
        let t_conv = Instant::now();
        let x = candle_nn::ops::leaky_relu(&x, 0.01)?;
        let x = self.conv_post.forward(&x)?;
        if prof.enabled {
            mel.device().synchronize()?;
            prof.post_conv_ms += HiftProfile::ms(t_conv);
        }
        let half = self.cfg.istft_n_fft / 2 + 1;
        let magnitude = x.narrow(1, 0, half)?.exp()?;
        let phase = x.narrow(1, half, half)?.sin()?;
        let t_istft = Instant::now();
        let wav = self.istft.forward(&magnitude, &phase)?;
        if prof.enabled {
            mel.device().synchronize()?;
            prof.post_istft_ms += HiftProfile::ms(t_istft);
            prof.post_ms += HiftProfile::ms(t_stage);
        }
        let wav = if !finalize {
            // Trim is prod(upsample_rates) * istft hop samples.
            let trim = prod_rates * self.cfg.istft_hop_len;
            wav.narrow(1, 0, wav.dim(1)?.saturating_sub(trim))?
        } else {
            wav
        };
        prof.line(frames);
        wav.clamp(-self.cfg.audio_limit, self.cfg.audio_limit)
    }
}
