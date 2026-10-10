//! Experimental native MOSS-TTS-Nano (GPT2 global/local, 16 RVQ channels).
//! Architecture follows OpenMOSS/MOSS-TTS-Nano, with attention conventions
//! cross-checked against ramishi/moss-tts-nano-rust-candle (Apache-2.0).
mod sampling;
mod transformer;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, embedding};
use serde::Deserialize;
use std::path::Path;
use transformer::{Config as TransformerConfig, Transformer};

#[derive(Deserialize)]
struct Config {
    gpt2_config: TransformerConfig,
    n_vq: usize,
    audio_codebook_sizes: Vec<usize>,
    local_transformer_layers: usize,
}

pub struct Nano {
    global: Transformer,
    local: Transformer,
    text: Embedding,
    audio: Vec<Embedding>,
    text_head: Linear,
    audio_heads: Vec<Linear>,
    device: Device,
}
impl Nano {
    pub fn load(directory: &Path, device: &Device) -> anyhow::Result<Self> {
        let config: Config =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        anyhow::ensure!(
            config.n_vq == 16 && config.audio_codebook_sizes == vec![1024; 16],
            "unsupported Nano RVQ configuration"
        );
        // Candle reads the official BF16 PyTorch checkpoint and converts to F32.
        let vb = VarBuilder::from_pth(directory.join("pytorch_model.bin"), DType::F32, device)?;
        let c = &config.gpt2_config;
        let text = embedding(c.vocab_size, c.n_embd, vb.pp("transformer.wte"))?;
        let text_head = Linear::new(text.embeddings().clone(), None);
        let audio = (0..16)
            .map(|i| embedding(1024, c.n_embd, vb.pp(format!("audio_embeddings.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let audio_heads = (0..16)
            .map(|i| {
                Ok(Linear::new(
                    vb.get((1024, c.n_embd), &format!("audio_lm_heads.{i}.weight"))?,
                    None,
                ))
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let global = Transformer::load(c, vb.pp("transformer"))?;
        let mut local_config = c.clone();
        local_config.n_layer = config.local_transformer_layers;
        let local = Transformer::load(&local_config, vb.pp("local_transformer"))?;
        Ok(Self {
            global,
            local,
            text,
            audio,
            text_head,
            audio_heads,
            device: device.clone(),
        })
    }
    fn embeds(&self, rows: &[Vec<u32>]) -> anyhow::Result<Tensor> {
        anyhow::ensure!(
            !rows.is_empty()
                && rows
                    .iter()
                    .all(|r| r.len() == 17 && r[0] < 16384 && r[1..].iter().all(|id| *id <= 1024)),
            "invalid Nano prompt rows"
        );
        let ids = Tensor::from_vec(
            rows.iter().map(|r| r[0]).collect::<Vec<_>>(),
            (1, rows.len()),
            &self.device,
        )?;
        let mut x = self.text.forward(&ids)?;
        for (channel, emb) in self.audio.iter().enumerate() {
            let ids = Tensor::from_vec(
                rows.iter().map(|r| r[channel + 1]).collect::<Vec<_>>(),
                (1, rows.len()),
                &self.device,
            )?;
            let mask = ids.ne(1024u32)?;
            let safe = mask.where_cond(&ids, &Tensor::zeros_like(&ids)?)?;
            x = (&x
                + emb
                    .forward(&safe)?
                    .broadcast_mul(&mask.to_dtype(DType::F32)?.unsqueeze(2)?)?)?;
        }
        Ok(x)
    }
    /// Emits complete 16-channel frames. Success requires explicit model EOS.
    pub fn generate(
        &mut self,
        rows: &[Vec<u32>],
        max_frames: usize,
        seed: u64,
        cancelled: &impl Fn() -> bool,
        mut frame: impl FnMut(Vec<u32>) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        self.global.reset();
        let mut x = self.embeds(rows)?;
        let mut history = vec![Vec::new(); 16];
        let mut rng = sampling::Sampler::new(seed);
        for _ in 0..max_frames {
            crate::check_cancel(cancelled)?;
            let hidden = self.global.forward(&x, cancelled)?;
            self.local.reset();
            let hidden = hidden.i((.., hidden.dim(1)? - 1, ..))?.unsqueeze(1)?;
            let local = self.local.forward(&hidden, cancelled)?.squeeze(1)?;
            let logits = self.text_head.forward(&local)?.flatten_all()?;
            let candidates = Tensor::cat(&[logits.narrow(0, 9, 1)?, logits.narrow(0, 7, 1)?], 0)?;
            let token = rng.sample(&candidates, &[], 1., 2, 1., 1.)?;
            if token == 1 {
                return Ok(());
            }
            let mut x_local = self.text.forward(&Tensor::new(&[[9u32]], &self.device)?)?;
            let mut codes = Vec::with_capacity(16);
            for (i, previous) in history.iter_mut().enumerate() {
                crate::check_cancel(cancelled)?;
                let hidden = self.local.forward(&x_local, cancelled)?.squeeze(1)?;
                let logits = self.audio_heads[i].forward(&hidden)?;
                let code = rng.sample(&logits, previous, 0.8, 25, 0.95, 1.2)?;
                previous.push(code);
                codes.push(code);
                x_local = self.audio[i].forward(&Tensor::new(&[[code]], &self.device)?)?;
            }
            let mut row = vec![9];
            row.extend_from_slice(&codes);
            frame(codes)?;
            x = self.embeds(&[row])?;
        }
        anyhow::bail!("Nano frame limit reached before end-of-speech")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_nano_generation_logits_match_official_reference() -> anyhow::Result<()> {
        let (Some(directory), Some(reference)) = (
            std::env::var_os("TRNOVEL_MOSS_NANO_CANDLE_DIR"),
            std::env::var_os("TALECHIME_NANO_GENERATION_REFERENCE"),
        ) else {
            return Ok(());
        };
        let cases: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(reference)?)?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let device = Device::new_metal(0)?;
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let device = Device::Cpu;
        let mut model = Nano::load(&std::path::PathBuf::from(directory).join("tts"), &device)?;
        for (index, case) in cases.iter().enumerate() {
            let rows: Vec<Vec<u32>> = serde_json::from_value(case["rows"].clone())?;
            let frames: Vec<Vec<u32>> = serde_json::from_value(case["frames"].clone())?;
            let text_logits: Vec<Vec<f32>> = serde_json::from_value(case["text_logits"].clone())?;
            let audio_logits: Vec<Vec<Vec<f32>>> =
                serde_json::from_value(case["audio_logits"].clone())?;
            model.global.reset();
            let mut x = model.embeds(&rows)?;
            let mut maximum = 0f32;
            for (step, expected_text) in text_logits.iter().enumerate() {
                let hidden = model.global.forward(&x, &|| false)?;
                let hidden = hidden.i((.., hidden.dim(1)? - 1, ..))?.unsqueeze(1)?;
                model.local.reset();
                let local = model.local.forward(&hidden, &|| false)?.squeeze(1)?;
                let logits = model
                    .text_head
                    .forward(&local)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                for (id, expected) in [9, 7].into_iter().zip(expected_text) {
                    maximum = maximum.max((logits[id] - expected).abs());
                }
                if step == frames.len() {
                    break;
                }
                let mut local_input = model.text.forward(&Tensor::new(&[[9u32]], &device)?)?;
                for channel in 0..16 {
                    let local = model.local.forward(&local_input, &|| false)?.squeeze(1)?;
                    let logits = model.audio_heads[channel]
                        .forward(&local)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    for (actual, expected) in
                        logits.iter().step_by(64).zip(&audio_logits[channel][step])
                    {
                        maximum = maximum.max((actual - expected).abs());
                    }
                    local_input = model.audio[channel]
                        .forward(&Tensor::new(&[[frames[step][channel]]], &device)?)?;
                }
                let mut row = vec![9];
                row.extend_from_slice(&frames[step]);
                x = model.embeds(&[row])?;
            }
            eprintln!(
                "Nano teacher-forced case {index}, {} frames: maximum logit error {maximum}",
                frames.len()
            );
            assert!(
                maximum < 1e-3,
                "Nano case {index} differs from official logits"
            );
        }
        Ok(())
    }

    #[test]
    fn real_nano_transformers_match_official_reference() -> anyhow::Result<()> {
        let (Some(directory), Some(reference)) = (
            std::env::var_os("TRNOVEL_MOSS_NANO_CANDLE_DIR"),
            std::env::var_os("TALECHIME_NANO_TTS_REFERENCE"),
        ) else {
            return Ok(());
        };
        let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(reference)?)?;
        let rows: Vec<Vec<u32>> = serde_json::from_value(reference["rows"].clone())?;
        #[allow(unused_mut)]
        let mut devices = vec![Device::Cpu];
        #[cfg(all(feature = "metal", target_os = "macos"))]
        devices.push(Device::new_metal(0)?);
        for device in devices {
            let mut model = Nano::load(&std::path::PathBuf::from(&directory).join("tts"), &device)?;
            let x = model.embeds(&rows)?;
            let h = model.global.forward(&x, &|| false)?;
            let h = h.i((.., h.dim(1)? - 1, ..))?.unsqueeze(1)?;
            let slot = model.text.forward(&Tensor::new(&[[9u32]], &device)?)?;
            let local = model
                .local
                .forward(&Tensor::cat(&[&h, &slot], 1)?, &|| false)?;
            for (name, actual) in [("global", h), ("local", local)] {
                let expected: Vec<f32> = serde_json::from_value(reference[name].clone())?;
                let actual = actual.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(actual.len(), expected.len());
                let error = actual
                    .iter()
                    .zip(expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("Nano {device:?} {name} reference maximum error: {error}");
                assert!(error < 1e-3, "Nano {name} differs from official reference");
            }
        }
        Ok(())
    }
}
