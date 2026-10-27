//! Optional official g2pW v2 contextual polyphone prediction on CPU ORT.
//!
//! This corrects readings before Misaki sandhi/token mapping. It does not change
//! the faithful official path or infer host/business entities such as characters.
use crate::Trace;
use candle_core::{Device, Result, Tensor};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor as OrtTensor,
};
use serde::Deserialize;
use std::{collections::HashMap, path::Path};
use tokenizers::{
    Tokenizer, models::wordpiece::WordPiece, normalizers::bert::BertNormalizer,
    pre_tokenizers::bert::BertPreTokenizer,
};

#[derive(Deserialize)]
struct ContextData {
    format: String,
    source_revision: String,
    archive_sha256: String,
    tokenizer_revision: String,
    labels: Vec<String>,
    label_pinyin: Vec<Option<String>>,
    label_readings: Vec<Option<[String; 2]>>,
    chars: Vec<char>,
    char2phonemes: HashMap<char, Vec<usize>>,
    s2t: HashMap<char, char>,
    window_size: usize,
    batch_size: usize,
}

/// One polyphonic-character prediction; index counts characters in the supplied
/// normalized text and is never an original-document UTF-8 byte offset.
#[derive(Clone, Debug)]
pub struct Prediction {
    /// Normalized input character index.
    pub index: usize,
    /// Standard initial/final-with-tone components consumed by Misaki.
    pub reading: [String; 2],
    /// Official pinyin label for diagnostics.
    pub pinyin: String,
    /// Probability of the selected official label.
    pub confidence: f32,
}

struct PronunciationQuery {
    index: usize,
    char_id: usize,
    position: i64,
    ids: Vec<i64>,
    mask: Vec<f32>,
}

/// Thread-owned contextual model. The acoustic model still owns actual speech
/// generation; this optional predictor only selects ambiguous pronunciations.
pub struct ContextModel {
    data: ContextData,
    tokenizer: Tokenizer,
    session: Session,
    char_ids: HashMap<char, usize>,
}

fn error(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(e.to_string())
}

impl ContextModel {
    /// Load an explicit `scripts/export_context.py` artifact. No downloading,
    /// provider auto-selection or Python execution takes place here.
    pub fn load(root: &Path) -> Result<Self> {
        let data: ContextData =
            serde_json::from_slice(&std::fs::read(root.join("context.json")).map_err(error)?)
                .map_err(error)?;
        if data.format != "kokoro-g2pw-v2"
            || data.source_revision != "b9328ce8129c4717a70f16a2688542a0197541dd"
            || data.archive_sha256
                != "699f3c1fd7fb0e2c2d49ed2486826fd5bff233fee7759350a91c3b49aedc4ed2"
            || data.tokenizer_revision != "8f23c25b06e129b6c986331a13d8d025a92cf0ea"
            || data.window_size != 32
            || data.batch_size != 32
        {
            candle_core::bail!("unsupported contextual pronunciation model format or revision");
        }
        if data.labels.is_empty()
            || data.labels.len() != data.label_readings.len()
            || data.labels.len() != data.label_pinyin.len()
            || data
                .char2phonemes
                .values()
                .flatten()
                .any(|&i| i >= data.labels.len())
        {
            candle_core::bail!("invalid contextual pronunciation label table");
        }
        let path = root.join("vocab.txt");
        let path = path
            .to_str()
            .ok_or_else(|| error("invalid vocabulary path"))?;
        let wordpiece = WordPiece::from_file(path)
            .unk_token("[UNK]".into())
            .build()
            .map_err(error)?;
        let mut tokenizer = Tokenizer::new(wordpiece);
        tokenizer.with_normalizer(Some(BertNormalizer::new(true, true, Some(true), true)));
        tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
        let session = Session::builder()
            .map_err(error)?
            .with_execution_providers([ort::ep::CPU::default().build().error_on_failure()])
            .map_err(error)?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(error)?
            .with_parallel_execution(false)
            .map_err(error)?
            .with_intra_threads(2)
            .map_err(error)?
            .commit_from_file(root.join("g2pw.onnx"))
            .map_err(error)?;
        let char_ids = data
            .chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        Ok(Self {
            data,
            tokenizer,
            session,
            char_ids,
        })
    }

    fn tokens(&self, text: &[char]) -> Result<(Vec<i64>, Vec<Option<usize>>)> {
        let mut ids = Vec::new();
        let mut mapping = vec![None; text.len()];
        let mut cursor = 0;
        while cursor < text.len() {
            if text[cursor] == ' ' {
                cursor += 1;
                continue;
            }
            let start = cursor;
            if text[cursor].is_ascii_alphanumeric() {
                while cursor < text.len() && text[cursor].is_ascii_alphanumeric() {
                    cursor += 1;
                }
            } else {
                cursor += 1;
            }
            let word: String = text[start..cursor].iter().collect();
            let encoded = self.tokenizer.encode(word.as_str(), false).map_err(error)?;
            if encoded.get_ids().is_empty() {
                let unknown = self
                    .tokenizer
                    .token_to_id("[UNK]")
                    .ok_or_else(|| error("missing BERT [UNK] token"))?;
                mapping[start..cursor].fill(Some(ids.len()));
                ids.push(i64::from(unknown));
            } else {
                let mut position = start;
                for (token, &id) in encoded.get_tokens().iter().zip(encoded.get_ids()) {
                    let length = if token == "[UNK]" {
                        cursor - start
                    } else {
                        token.strip_prefix("##").unwrap_or(token).chars().count()
                    };
                    let end = (position + length).min(cursor);
                    mapping[position..end].fill(Some(ids.len()));
                    ids.push(i64::from(id));
                    position = end;
                }
                if mapping[start..cursor].iter().any(Option::is_none) {
                    candle_core::bail!("context tokenizer lost character mapping");
                }
            }
        }
        Ok((ids, mapping))
    }

    fn queries(&self, text: &str) -> Result<Vec<PronunciationQuery>> {
        let original: Vec<_> = text.chars().collect();
        if original.len() > 4096 {
            candle_core::bail!("context pronunciation input exceeds character budget");
        }
        let translated: Vec<_> = original
            .iter()
            .map(|c| self.data.s2t.get(c).copied().unwrap_or(*c))
            .collect();
        let cls = self
            .tokenizer
            .token_to_id("[CLS]")
            .ok_or_else(|| error("missing BERT [CLS] token"))?;
        let sep = self
            .tokenizer
            .token_to_id("[SEP]")
            .ok_or_else(|| error("missing BERT [SEP] token"))?;
        let mut queries = Vec::new();
        for (index, &character) in translated.iter().enumerate() {
            let Some(&char_id) = self.char_ids.get(&character) else {
                continue;
            };
            let start = index.saturating_sub(self.data.window_size / 2);
            let end = (index + self.data.window_size / 2).min(translated.len());
            let lowered: Vec<_> = translated[start..end]
                .iter()
                .flat_map(|c| c.to_lowercase())
                .collect();
            if lowered.len() != end - start {
                candle_core::bail!("context lowercase changed character alignment");
            }
            let (tokens, mapping) = self.tokens(&lowered)?;
            if tokens.len() > 510 {
                candle_core::bail!("context tokenizer exceeded model context");
            }
            let position = mapping[index - start]
                .ok_or_else(|| error("context query lacks a BERT token"))?
                + 1;
            let mut ids = vec![i64::from(cls)];
            ids.extend(tokens);
            ids.push(i64::from(sep));
            let mut mask = vec![0.0; self.data.labels.len()];
            let allowed = self
                .data
                .char2phonemes
                .get(&character)
                .ok_or_else(|| error("context character has no phoneme mask"))?;
            if allowed.is_empty() {
                candle_core::bail!("empty context phoneme mask");
            }
            for &label in allowed {
                mask[label] = 1.0;
            }
            queries.push(PronunciationQuery {
                index,
                char_id,
                position: position as i64,
                ids,
                mask,
            });
        }
        Ok(queries)
    }

    /// Predict polyphonic characters only. Monophonic dictionary readings remain
    /// owned by the official Misaki frontend and its tone-sandhi rules.
    pub fn predict(&mut self, text: &str, cancelled: &dyn Fn() -> bool) -> Result<Vec<Prediction>> {
        self.predict_traced(text, cancelled, &mut Trace::default())
    }

    /// Predict while exposing exact model inputs/probabilities for independent
    /// official fixtures. Ordinary inference retains no trace buffers.
    pub fn predict_traced(
        &mut self,
        text: &str,
        cancelled: &dyn Fn() -> bool,
        trace: &mut Trace,
    ) -> Result<Vec<Prediction>> {
        let queries = self.queries(text)?;
        let mut predictions = Vec::with_capacity(queries.len());
        for (batch_index, batch) in queries.chunks(self.data.batch_size).enumerate() {
            if cancelled() {
                candle_core::bail!("context pronunciation cancelled");
            }
            let length = batch.iter().map(|q| q.ids.len()).max().unwrap_or(0);
            let mut ids = vec![0i64; batch.len() * length];
            let types = ids.clone();
            let mut attention = ids.clone();
            let mut masks = Vec::new();
            let mut chars = Vec::new();
            let mut positions = Vec::new();
            for (row, query) in batch.iter().enumerate() {
                ids[row * length..row * length + query.ids.len()].copy_from_slice(&query.ids);
                attention[row * length..row * length + query.ids.len()].fill(1);
                masks.extend_from_slice(&query.mask);
                chars.push(query.char_id as i64);
                positions.push(query.position);
            }
            let prefix = format!("batch.{batch_index}");
            if trace.tensors().is_some() {
                for (name, values) in [
                    ("input_ids", &ids),
                    ("token_type_ids", &types),
                    ("attention_mask", &attention),
                ] {
                    trace.insert(
                        &format!("{prefix}.{name}"),
                        &Tensor::from_slice(values, (batch.len(), length), &Device::Cpu)?,
                    );
                }
                trace.insert(
                    &format!("{prefix}.phoneme_mask"),
                    &Tensor::from_slice(
                        &masks,
                        (batch.len(), self.data.labels.len()),
                        &Device::Cpu,
                    )?,
                );
                trace.insert(
                    &format!("{prefix}.char_ids"),
                    &Tensor::from_slice(&chars, batch.len(), &Device::Cpu)?,
                );
                trace.insert(
                    &format!("{prefix}.position_ids"),
                    &Tensor::from_slice(&positions, batch.len(), &Device::Cpu)?,
                );
            }
            let output=self.session.run(ort::inputs![
                "input_ids"=>OrtTensor::from_array(([batch.len(),length],ids)).map_err(error)?,
                "token_type_ids"=>OrtTensor::from_array(([batch.len(),length],types)).map_err(error)?,
                "attention_mask"=>OrtTensor::from_array(([batch.len(),length],attention)).map_err(error)?,
                "phoneme_mask"=>OrtTensor::from_array(([batch.len(),self.data.labels.len()],masks)).map_err(error)?,
                "char_ids"=>OrtTensor::from_array(([batch.len()],chars)).map_err(error)?,
                "position_ids"=>OrtTensor::from_array(([batch.len()],positions)).map_err(error)?,
            ]).map_err(error)?;
            let (shape, probabilities) = output[0].try_extract_tensor::<f32>().map_err(error)?;
            if shape.as_ref() != [batch.len() as i64, self.data.labels.len() as i64] {
                candle_core::bail!("invalid contextual pronunciation output shape");
            }
            if trace.tensors().is_some() {
                trace.insert(
                    &format!("{prefix}.probabilities"),
                    &Tensor::from_slice(
                        probabilities,
                        (batch.len(), self.data.labels.len()),
                        &Device::Cpu,
                    )?,
                );
            }
            for (row, query) in batch.iter().enumerate() {
                let probs = &probabilities
                    [row * self.data.labels.len()..(row + 1) * self.data.labels.len()];
                if probs.iter().any(|p| !p.is_finite() || *p < 0.0) {
                    candle_core::bail!("nonfinite contextual pronunciation probabilities");
                }
                let (label, &confidence) = probs
                    .iter()
                    .enumerate()
                    .max_by(|(i, a), (j, b)| a.total_cmp(b).then(j.cmp(i)))
                    .ok_or_else(|| error("empty context prediction"))?;
                if query.mask[label] != 1.0 {
                    candle_core::bail!("context model predicted a masked pronunciation");
                }
                let reading = self.data.label_readings[label].clone().ok_or_else(|| {
                    error("context label cannot be mapped to Mandarin initials/finals")
                })?;
                let pinyin = self.data.label_pinyin[label]
                    .clone()
                    .ok_or_else(|| error("context label has no official pinyin"))?;
                predictions.push(Prediction {
                    index: query.index,
                    reading,
                    pinyin,
                    confidence,
                });
            }
        }
        Ok(predictions)
    }
}
