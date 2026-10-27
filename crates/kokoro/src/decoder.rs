//! AdaIN residual decoder and iSTFTNet, following official PyTorch inference.
use crate::{
    ExcitationNoise, Trace,
    ops::{Weights, leaky, nearest_double, snake},
    spectral,
};
use candle_core::{Result, Tensor};

impl Weights {
    pub fn resblock(&self, x: &Tensor, style: &Tensor, p: &str, upsample: bool) -> Result<Tensor> {
        let mut shortcut = if upsample {
            nearest_double(x)?
        } else {
            x.clone()
        };
        if self.tensors.contains_key(&format!("{p}.conv1x1.weight")) {
            shortcut = self.conv(&shortcut, &format!("{p}.conv1x1"), 1, 1)?;
        }
        let mut y = leaky(&self.adain(x, style, &format!("{p}.norm1"))?, 0.2)?;
        if upsample {
            y = self.transpose(&y, &format!("{p}.pool"), 2, 1, 1, y.dim(1)?)?;
        }
        y = self.conv(&y, &format!("{p}.conv1"), 1, 1)?;
        y = leaky(&self.adain(&y, style, &format!("{p}.norm2"))?, 0.2)?;
        y = self.conv(&y, &format!("{p}.conv2"), 1, 1)?;
        (y + shortcut)? * f64::from(std::f32::consts::FRAC_1_SQRT_2)
    }

    pub fn f0_noise(
        &self,
        expanded: &Tensor,
        style: &Tensor,
        trace: &mut Trace,
    ) -> Result<(Tensor, Tensor)> {
        let shared = self
            .bilstm(&expanded.transpose(1, 2)?, "predictor.shared")?
            .transpose(1, 2)?;
        trace.insert("predictor.shared.0", &shared.transpose(1, 2)?);
        let mut branches = Vec::with_capacity(2);
        for p in ["predictor.F0", "predictor.N"] {
            let mut x = shared.clone();
            for block in 0..3 {
                x = self.resblock(&x, style, &format!("{p}.{block}"), block == 1)?;
                trace.insert(&format!("{p}.{block}"), &x);
            }
            branches.push(self.conv(&x, &format!("{p}_proj"), 1, 1)?);
        }
        Ok((branches.remove(0), branches.remove(0)))
    }

    fn snake_block(&self, x: &Tensor, style: &Tensor, p: &str) -> Result<Tensor> {
        let mut x = x.clone();
        for (index, dilation) in [1, 3, 5].into_iter().enumerate() {
            let mut y = self.adain(&x, style, &format!("{p}.adain1.{index}"))?;
            let a = self.get(&format!("{p}.alpha1.{index}"))?;
            y = snake(&y, a)?;
            y = self.conv(&y, &format!("{p}.convs1.{index}"), 1, dilation)?;
            y = self.adain(&y, style, &format!("{p}.adain2.{index}"))?;
            let a = self.get(&format!("{p}.alpha2.{index}"))?;
            y = snake(&y, a)?;
            y = self.conv(&y, &format!("{p}.convs2.{index}"), 1, 1)?;
            x = (x + y)?;
        }
        Ok(x)
    }

    pub fn decode(
        &self,
        asr: &Tensor,
        f0: &Tensor,
        n: &Tensor,
        style: &Tensor,
        noise: &ExcitationNoise,
        trace: &mut Trace,
    ) -> Result<Tensor> {
        let f0_down = self.conv(f0, "decoder.F0_conv", 2, 1)?;
        let n_down = self.conv(n, "decoder.N_conv", 2, 1)?;
        let mut x = self.resblock(
            &Tensor::cat(&[asr, &f0_down, &n_down], 1)?,
            style,
            "decoder.encode",
            false,
        )?;
        trace.insert("decoder.encode", &x);
        let asr_res = self.conv(asr, "decoder.asr_res.0", 1, 1)?;
        for block in 0..4 {
            x = Tensor::cat(&[&x, &asr_res, &f0_down, &n_down], 1)?;
            x = self.resblock(&x, style, &format!("decoder.decode.{block}"), block == 3)?;
        }
        let source = spectral::excitation(f0, noise)?;
        trace.insert("excitation", &source);
        let source = self
            .linear(&source, "decoder.generator.m_source.l_linear")?
            .tanh()?;
        trace.insert("decoder.generator.m_source.0", &source);
        let (magnitude, phase) = spectral::stft(&source.squeeze(2)?.squeeze(0)?)?;
        let harmonic = Tensor::cat(&[magnitude, phase], 1)?;
        for (i, stride) in [10, 6].into_iter().enumerate() {
            x = leaky(&x, 0.1)?;
            let p = "decoder.generator";
            let w = self.get(&format!("{p}.noise_convs.{i}.weight"))?;
            let source = harmonic.conv1d(
                w,
                if i == 0 { 3 } else { 0 },
                if i == 0 { 6 } else { 1 },
                1,
                1,
            )?;
            let source = source.broadcast_add(
                &self
                    .get(&format!("{p}.noise_convs.{i}.bias"))?
                    .reshape((1, (), 1))?,
            )?;
            let source = self.snake_block(&source, style, &format!("{p}.noise_res.{i}"))?;
            x = self.transpose(
                &x,
                &format!("{p}.ups.{i}"),
                stride,
                if i == 0 { 5 } else { 3 },
                0,
                1,
            )?;
            if i == 1 {
                x = Tensor::cat(&[x.narrow(2, 1, 1)?, x], 2)?;
            }
            x = (x + source)?;
            let mut branches = Vec::with_capacity(3);
            for j in 0..3 {
                branches.push(self.snake_block(
                    &x,
                    style,
                    &format!("{p}.resblocks.{}", i * 3 + j),
                )?);
            }
            x = (((&branches[0] + &branches[1])? + &branches[2])? / 3.0)?;
        }
        x = self.conv(&leaky(&x, 0.01)?, "decoder.generator.conv_post", 1, 1)?;
        trace.insert("decoder.generator.conv_post", &x);
        spectral::istft(&x.narrow(1, 0, 11)?.exp()?, &x.narrow(1, 11, 11)?.sin()?)
    }
}
