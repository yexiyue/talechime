//! MOSS Realtime: initial text lookahead followed by one text token per frame.
mod prompt;
use crate::{
    Generation,
    config::TransformerConfig,
    sampling::{Parameters, sample},
    transformer::Transformer,
};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, embedding, linear_no_bias};
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct Config {
    language_config: TransformerConfig,
    local_config: TransformerConfig,
    rvq: usize,
    audio_vocab_size: usize,
    audio_pad_token: u32,
    text_pad: u32,
}
pub struct RealtimeModel {
    cfg: Config,
    tokenizer: Tokenizer,
    embeddings: Vec<Embedding>,
    local_embeddings: Vec<Embedding>,
    backbone: Transformer,
    depth: Transformer,
    heads: Vec<Linear>,
    device: Device,
}
impl RealtimeModel {
    pub fn load(directory: &Path, device: &Device, dtype: DType) -> anyhow::Result<Self> {
        Self::load_with_weights(
            &directory.join("config.json"),
            &directory.join("tokenizer.json"),
            device,
            || crate::weights(directory, dtype, device),
        )
    }
    /// Load the model from separately resolved resources.
    pub fn load_files(
        config: &Path,
        tokenizer: &Path,
        shards: &[std::path::PathBuf],
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        Self::load_with_weights(config, tokenizer, device, || {
            crate::weights_files(shards, dtype, device)
        })
    }
    fn load_with_weights(
        config: &Path,
        tokenizer_path: &Path,
        device: &Device,
        weights: impl FnOnce() -> anyhow::Result<VarBuilder<'static>>,
    ) -> anyhow::Result<Self> {
        let cfg: Config = serde_json::from_slice(&std::fs::read(config)?)?;
        anyhow::ensure!(
            cfg.rvq == 16 && cfg.audio_vocab_size == 1027 && cfg.audio_pad_token == 1024,
            "unsupported Realtime configuration"
        );
        let vb = weights()?;
        let d = cfg.language_config.hidden_size;
        let mut embeddings = vec![embedding(
            cfg.language_config.vocab_size,
            d,
            vb.pp("embed_tokens.0"),
        )?];
        for i in 0..cfg.rvq {
            embeddings.push(embedding(
                cfg.audio_vocab_size,
                d,
                vb.pp(format!("embed_tokens.{}", i + 1)),
            )?);
        }
        let local_embeddings = (0..cfg.rvq - 1)
            .map(|i| {
                embedding(
                    cfg.audio_vocab_size,
                    cfg.local_config.hidden_size,
                    vb.pp(format!("local_transformer.model.embed_tokens.{i}")),
                )
            })
            .collect::<candle_core::Result<_>>()?;
        let heads = (0..cfg.rvq)
            .map(|i| {
                linear_no_bias(
                    cfg.local_config.hidden_size,
                    cfg.audio_vocab_size,
                    vb.pp(format!("local_transformer.local_lm_heads.{i}")),
                )
            })
            .collect::<candle_core::Result<_>>()?;
        let backbone = Transformer::load(&cfg.language_config, vb.pp("language_model"), true)?;
        let depth = Transformer::load(&cfg.local_config, vb.pp("local_transformer.model"), true)?;
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(Self {
            cfg,
            tokenizer,
            embeddings,
            local_embeddings,
            backbone,
            depth,
            heads,
            device: device.clone(),
        })
    }
    fn text(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let ids = encoding.get_ids().to_vec();
        if std::env::var("TALECHIME_TEXT_DIAGNOSTICS").is_ok_and(|value| value == "1") {
            eprintln!(
                "TALECHIME_TEXT_DIAGNOSTIC {}",
                serde_json::json!({"backend":"moss-realtime","phase":"text_tokens","execution_text":text,"tokens":ids})
            );
        }
        Ok(ids)
    }

    fn embed(&self, rows: &[Vec<u32>]) -> anyhow::Result<Tensor> {
        let mut x = None;
        for (i, embedding) in self.embeddings.iter().enumerate() {
            let value = embedding.forward(&Tensor::from_vec(
                rows.iter().map(|row| row[i]).collect::<Vec<_>>(),
                (1, rows.len()),
                &self.device,
            )?)?;
            x = Some(match x {
                None => value,
                Some(previous) => (previous + value)?,
            });
        }
        Ok(x.expect("validated Realtime channels"))
    }
    pub fn generate(
        &mut self,
        request: &Generation<'_>,
        cancelled: &impl Fn() -> bool,
        mut frame: impl FnMut(&[u32]) -> anyhow::Result<bool>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(!request.text.trim().is_empty(), "empty Realtime input");
        anyhow::ensure!(
            request.instruction.is_none(),
            "Realtime does not accept voice design descriptions"
        );
        self.backbone.reset();
        self.depth.reset();
        let text = self.text(request.text)?;
        let (rows, index) =
            prompt::initial_rows(|value| self.text(value), &text, request.reference)?;
        let mut hidden = self.backbone.forward(&self.embed(&rows)?, cancelled)?;
        let mut history = vec![Vec::<u32>::new(); self.cfg.rvq];
        let mut rng = StdRng::seed_from_u64(request.seed);
        for index in (index..).take(request.max_frames) {
            crate::check_cancel(cancelled)?;
            self.depth.reset();
            let mut input = hidden.narrow(1, hidden.dim(1)? - 1, 1)?;
            let mut tokens = Vec::with_capacity(self.cfg.rvq);
            for (i, previous) in history.iter().enumerate() {
                let local = self.depth.forward(&input, cancelled)?;
                let window = &previous[previous.len().saturating_sub(50)..];
                let id = sample(
                    &self.heads[i].forward(&local)?,
                    window,
                    Parameters {
                        temperature: 0.8,
                        top_k: 30,
                        top_p: 0.6,
                        penalty: 1.1,
                    },
                    None,
                    &mut rng,
                )?;
                if i == 0 && id == 1026 {
                    validate_eos(index, text.len())?;
                    return Ok(());
                }
                tokens.push(id);
                if i + 1 < self.cfg.rvq {
                    input =
                        self.local_embeddings[i].forward(&Tensor::new(&[[id]], &self.device)?)?;
                }
            }
            anyhow::ensure!(
                tokens.iter().all(|id| *id < 1024),
                "Realtime returned a reserved audio token"
            );
            if !frame(&tokens)? {
                anyhow::bail!("MOSS inference cancelled");
            }
            for (previous, id) in history.iter_mut().zip(&tokens) {
                previous.push(*id);
            }
            let mut next = vec![text.get(index).copied().unwrap_or(self.cfg.text_pad)];
            next.extend(tokens);
            hidden = self.backbone.forward(&self.embed(&[next])?, cancelled)?;
        }
        anyhow::bail!("MOSS Realtime frame limit reached before EOS")
    }
}

fn validate_eos(consumed: usize, total: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        consumed >= total,
        "Realtime EOS before all text was consumed"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn early_eos_cannot_accept_unconsumed_text() {
        assert!(super::validate_eos(12, 16).is_err());
        assert!(super::validate_eos(15, 16).is_err());
        assert!(super::validate_eos(16, 16).is_ok());
        assert!(super::validate_eos(3, 3).is_ok());
    }
}
