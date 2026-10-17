//! End-to-end orchestration mirroring `IndexTTS2.infer_generator`
//! (`indextts/infer_v2_5.py`) for the single-reference, non-streaming path.

use std::path::Path;

use crate::bigvgan::BigVgan;
use crate::campplus::CampPlus;
use crate::codec::SemanticCodec;
use crate::config::Config;
use crate::dsp;
use crate::front::{self, TextFrontend};
use crate::generation::GenerateConfig;
use crate::gpt::UnifiedVoice;
use crate::mel::mel_spectrogram;
use crate::s2mel::S2Mel;
use crate::tokenizer::{self, Tokenizer};
use crate::w2v_bert::{self, Wav2Vec2Bert};
use crate::weights::Weights;
use candle_core::{DType, Device, Result, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

/// Generation options; defaults mirror `infer_generator`'s kwargs.
#[derive(Debug, Clone)]
pub struct Options {
    pub generate: GenerateConfig,
    /// Pins the sampling RNG for reproducibility (upstream seeds torch
    /// globally; here the seed is per call).
    pub seed: Option<u64>,
    /// Acoustic target-length scale (`duration_factor`, 0.5..=2.0).
    pub duration_factor: f32,
    /// Emotion blend between speaker and emotion reference (`emo_alpha`).
    pub emo_alpha: f32,
    /// Optional 8-dim emotion vector `[happy, angry, sad, afraid,
    /// disgusted, melancholic, surprised, calm]`.
    pub emo_vector: Option<[f32; 8]>,
    /// With `emo_vector`: pick random anchor rows instead of the
    /// cosine-similar ones (`use_random`).
    pub emo_use_random: bool,
    /// Max tiktoken tokens per text segment (`max_text_tokens_per_segment`).
    pub max_text_tokens_per_segment: usize,
    /// Inserted silence between segments, milliseconds.
    pub interval_silence_ms: u32,
    /// Run the (approximate) text normalization stage.
    pub text_normalization: bool,
    /// CFM solver steps and guidance rate.
    pub diffusion_steps: usize,
    pub inference_cfg_rate: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            generate: GenerateConfig::default(),
            seed: None,
            duration_factor: 1.0,
            emo_alpha: 1.0,
            emo_vector: None,
            emo_use_random: false,
            max_text_tokens_per_segment: 120,
            interval_silence_ms: 200,
            text_normalization: true,
            diffusion_steps: 25,
            inference_cfg_rate: 0.7,
        }
    }
}

/// A preloaded reference clip at the two rates the pipeline needs.
pub struct Reference {
    /// 22.05 kHz mono f32 in [-1, 1].
    pub pcm_22k: Vec<f32>,
    /// 16 kHz mono f32 in [-1, 1].
    pub pcm_16k: Vec<f32>,
}

impl Reference {
    /// Build from arbitrary-rate mono PCM, resampling to both rates
    /// (approximate sinc resampler — see [`dsp::resample`]).
    pub fn from_pcm(pcm: &[f32], sample_rate: u32) -> Result<Self> {
        let pcm_22k = dsp::resample(pcm, sample_rate, crate::SAMPLE_RATE)?;
        let pcm_16k = dsp::resample(pcm, sample_rate, 16_000)?;
        Ok(Self { pcm_22k, pcm_16k })
    }
}

/// Cached conditioning derived from a [`Reference`].
pub struct ReferenceConditioning {
    /// w2v-bert hidden-17 features `(1, T, 1024)`.
    pub spk_cond_emb: Tensor,
    /// Reference mel `(1, 80, frames)` at 22.05 kHz.
    pub ref_mel: Tensor,
    /// CAMPPlus speaker style `(1, 192)`.
    pub style: Tensor,
    /// Length-regulated prompt condition `(1, L, 512)`.
    pub prompt_condition: Tensor,
}

/// Result of one synthesis call.
pub struct Outcome {
    /// PCM-scale samples ([-32767, 32767] clamped), mono 22.05 kHz.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Number of text segments actually synthesized.
    pub segments: usize,
}

pub struct Model {
    config: Config,
    device: Device,
    tokenizer: Tokenizer,
    frontend: TextFrontend,
    w2v_bert: Wav2Vec2Bert,
    codec: SemanticCodec,
    s2mel: S2Mel,
    campplus: CampPlus,
    bigvgan: BigVgan,
    gpt: UnifiedVoice,
    w2v_mean: Tensor,
    w2v_std: Tensor,
    /// Per-emotion-group anchor rows (split of `emo_matrix` by `emo_num`).
    emo_matrix: Vec<Tensor>,
    /// Same split of `spk_matrix`.
    spk_matrix: Vec<Tensor>,
    reference: Option<ReferenceConditioning>,
    emo_reference: Option<Tensor>,
}

impl Model {
    /// Load every submodel from a converted checkpoint directory (see
    /// [`crate::weights`]). `device` is CPU or a CUDA/Metal device selected
    /// by the caller.
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let config = Config::load(&dir.join("config.json"))?;
        config.validate()?;
        let weights = Weights::load_dir(dir, device)?;

        let tokenizer_path = dir.join("multilingual_zh_ja_yue_char_del.tiktoken");
        let tokenizer = Tokenizer::from_tiktoken_file(&tokenizer_path)?;

        let w2v_bert = Wav2Vec2Bert::load(&dir.join("w2v-bert-2.0"), device)?;
        let codec = SemanticCodec::load(&weights.codec, &config.semantic_codec)?;
        let s2mel = S2Mel::load(&weights.s2mel, &config.s2mel)?;
        let campplus = CampPlus::load(&weights.campplus)?;
        let bigvgan = BigVgan::load(&weights.bigvgan, &weights.bigvgan_config_json)?;
        let gpt = UnifiedVoice::load(&weights.gpt, &config.gpt, device)?;

        let emo_groups = split_groups(&weights.feats.emo_matrix, &config.emo_num)?;
        let spk_groups = split_groups(&weights.feats.spk_matrix, &config.emo_num)?;

        Ok(Self {
            config,
            device: device.clone(),
            tokenizer,
            frontend: TextFrontend::new(),
            w2v_bert,
            codec,
            s2mel,
            campplus,
            bigvgan,
            gpt,
            w2v_mean: weights.feats.w2v_mean.to_device(device)?,
            w2v_std: weights.feats.w2v_std.to_device(device)?,
            emo_matrix: emo_groups,
            spk_matrix: spk_groups,
            reference: None,
            emo_reference: None,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// w2v-bert hidden-17 features, normalized with the checkpoint stats.
    fn w2v_features(&self, pcm_16k: &[f32]) -> Result<Tensor> {
        let (input_features, attention_mask) = w2v_bert::seamless_m4t_features(pcm_16k)?;
        let input_features = input_features.to_device(&self.device)?;
        let attention_mask = attention_mask.to_device(&self.device)?;
        let feat = self.w2v_bert.hidden_17(&input_features, &attention_mask)?;
        feat.broadcast_sub(&self.w2v_mean)?
            .broadcast_div(&self.w2v_std)?
            .to_dtype(DType::F32)
    }

    /// Compute (and cache) the conditioning of a reference clip; required
    /// before [`Self::generate`]. Mirrors the cache-refresh block of
    /// `infer_generator`.
    pub fn prepare_reference(&mut self, reference: &Reference) -> Result<()> {
        let spk_cond_emb = self.w2v_features(&reference.pcm_16k)?;

        let wav = Tensor::from_vec(
            reference.pcm_22k.clone(),
            (1, reference.pcm_22k.len()),
            &self.device,
        )?;
        let spect = self.config.s2mel.preprocess_params.spect_params.clone();
        let ref_mel = mel_spectrogram(
            &wav,
            spect.n_fft,
            spect.win_length,
            spect.hop_length,
            spect.n_mels,
            self.config.s2mel.preprocess_params.sr,
            spect.fmin,
            spect.fmax,
            false,
        )?;

        let fbank = dsp::campplus_fbank_tensor(&reference.pcm_16k, &self.device)?;
        let style = self.campplus.forward(&fbank)?;

        let ref_frames = ref_mel.dim(2)?;
        let prompt_condition = self
            .s2mel
            .length_regulator
            .forward(&spk_cond_emb, &[ref_frames])?;

        self.reference = Some(ReferenceConditioning {
            spk_cond_emb,
            ref_mel,
            style,
            prompt_condition,
        });
        Ok(())
    }

    /// Optional separate emotion-reference conditioning; when never called
    /// (or called with the speaker reference) the speaker conditioning
    /// doubles as the emotion reference with `emo_alpha` forced to 1.0,
    /// matching upstream.
    pub fn prepare_emo_reference(&mut self, reference: &Reference) -> Result<()> {
        self.emo_reference = Some(self.w2v_features(&reference.pcm_16k)?);
        Ok(())
    }

    /// Synthesize `text` (a whole paragraph; segmentation mirrors
    /// `split_text_by_tokens`) into PCM.
    pub fn generate(&mut self, text: &str, lang: &str, opts: &Options) -> Result<Outcome> {
        let Some(reference) = self.reference.as_ref() else {
            candle_core::bail!("prepare_reference must run before generate");
        };
        let device = self.device.clone();

        // Emotion vector path forces the speaker clip to be the emotion
        // reference with alpha 1.0 (upstream lines 583-617).
        let (emo_cond_emb, emo_alpha, emovec_override) = match opts.emo_vector {
            Some(vector) => {
                let emovec_mat =
                    self.emotion_vector_matrix(&vector, &reference.style, opts.emo_use_random)?;
                (reference.spk_cond_emb.clone(), 1.0f32, Some(emovec_mat))
            }
            None => (
                self.emo_reference
                    .as_ref()
                    .unwrap_or(&reference.spk_cond_emb)
                    .clone(),
                if self.emo_reference.is_some() {
                    opts.emo_alpha
                } else {
                    1.0
                },
                None,
            ),
        };

        // Text pipeline (upstream lines 695-727).
        let lang_lower = lang.to_ascii_lowercase();
        let lang_prefix = format!("<|{lang_lower}|> ");
        let text = self.frontend.clean_pattern(text);
        let text = if opts.text_normalization && matches!(lang_lower.as_str(), "zh" | "zhen" | "en")
        {
            self.frontend
                .normalize(&text, matches!(lang_lower.as_str(), "zh" | "zhen"))
        } else {
            text
        };
        let text = if matches!(lang_lower.as_str(), "ja" | "zh" | "zhen" | "en") {
            text.to_lowercase()
        } else {
            text
        };
        let text = front::apply_pronunciation_annotations(&text);
        let segments =
            self.split_text_by_tokens(&text, opts.max_text_tokens_per_segment, &lang_prefix)?;

        let lang_id = tokenizer::lang_to_token(lang);
        let stop_text = self.config.gpt.stop_text_token;

        let mut rng = StdRng::seed_from_u64(opts.seed.unwrap_or_else(random_seed));
        let mut wav_parts: Vec<Vec<f32>> = Vec::new();

        for seg_text in &segments {
            let mut ids = self.tokenizer.encode(&format!("{lang_prefix}{seg_text}"));
            ids.push(stop_text);
            let id_len = ids.len();
            let text_tokens = Tensor::from_vec(ids, (1, id_len), &device)?.to_dtype(DType::I64)?;

            let emovec =
                self.gpt
                    .merge_emovec(&reference.spk_cond_emb, &emo_cond_emb, emo_alpha as f64)?;
            let emovec = match &emovec_override {
                Some(mat) => {
                    let weight_sum: f64 = opts
                        .emo_vector
                        .map(|v| v.iter().map(|x| *x as f64).sum())
                        .unwrap_or(0.0);
                    let scale = Tensor::new(((1.0 - weight_sum) as f32) as f64, emovec.device())?
                        .to_dtype(emovec.dtype())?;
                    (mat + &(&scale * &emovec)?)?
                }
                None => emovec,
            };

            // Sampling can emit the stop token immediately (upstream hits
            // this too on unlucky draws); empty codes would drive kernels
            // with zero-length tensors, so redraw with a fresh RNG.
            let mut codes = None;
            for attempt in 0..3 {
                let raw = self.gpt.inference_speech(
                    &reference.spk_cond_emb,
                    &text_tokens,
                    lang_id,
                    &emo_cond_emb,
                    &emovec,
                    &reference.style,
                    &opts.generate,
                    &mut rng,
                )?;
                let raw = truncate_at_stop(&raw, self.config.gpt.stop_mel_token)?;
                if raw.dim(1)? >= 8 || attempt == 2 {
                    codes = Some(raw);
                    break;
                }
                eprintln!(
                    "indextts: segment sampled the stop token first; retrying (attempt {})",
                    attempt + 1
                );
                rng = StdRng::seed_from_u64(random_seed());
            }
            let codes = codes.expect("at least one attempt");

            let s_infer = self.codec.decode(&codes)?;
            let target_len = (s_infer.dim(1)? as f32 * 1.72 * opts.duration_factor)
                .round()
                .max(1.0) as usize;
            let cond = self
                .s2mel
                .length_regulator
                .forward(&s_infer, &[target_len])?;

            let cat_condition = Tensor::cat(&[&reference.prompt_condition, &cond], 1)?;
            let vc_target = self.s2mel.cfm.inference(
                &cat_condition,
                &[cat_condition.dim(1)?],
                &reference.ref_mel,
                &reference.style,
                opts.diffusion_steps,
                opts.inference_cfg_rate,
            )?;
            let ref_frames = reference.ref_mel.dim(2)?;
            let vc_target = vc_target.narrow(2, ref_frames, vc_target.dim(2)? - ref_frames)?;

            let wav = self.bigvgan.forward(&vc_target.to_dtype(DType::F32)?)?;
            let wav = wav.squeeze(1)?.squeeze(0)?.to_dtype(DType::F32)?;
            let samples = wav.to_vec1::<f32>()?;
            let scaled: Vec<f32> = samples
                .iter()
                .map(|v| (v * crate::PCM16_MAX).clamp(-crate::PCM16_MAX, crate::PCM16_MAX))
                .collect();
            wav_parts.push(scaled);
        }

        // Join with interval silence and fade the tail
        // (upstream lines 861-868).
        let silence_len =
            (crate::SAMPLE_RATE as usize * opts.interval_silence_ms as usize / 1000).max(1);
        let mut joined: Vec<f32> = Vec::new();
        for (i, part) in wav_parts.iter().enumerate() {
            joined.extend_from_slice(part);
            if i + 1 < wav_parts.len() && opts.interval_silence_ms > 0 {
                joined.extend(std::iter::repeat_n(0.0, silence_len));
            }
        }
        dsp::fade_out_tail(&mut joined, crate::SAMPLE_RATE);

        Ok(Outcome {
            samples: joined,
            sample_rate: crate::SAMPLE_RATE,
            segments: segments.len(),
        })
    }

    /// Emotion-anchor blending (upstream lines 665-676): per emotion group
    /// pick the row most cosine-similar to the CAMPPlus style (or a random
    /// row), then combine rows with the caller's weights.
    fn emotion_vector_matrix(
        &self,
        vector: &[f32; 8],
        style: &Tensor,
        use_random: bool,
    ) -> Result<Tensor> {
        if self.emo_matrix.len() != vector.len() {
            candle_core::bail!(
                "emotion vector length {} != emotion groups {}",
                vector.len(),
                self.emo_matrix.len()
            );
        }
        let device = self.device.clone();
        let style = style.to_dtype(DType::F32)?.squeeze(0)?;
        let mut rows = Vec::with_capacity(vector.len());
        let mut rng = rand::rng();
        for (group_idx, group) in self.emo_matrix.iter().enumerate() {
            let group = group.to_dtype(DType::F32)?;
            let row = if use_random {
                let n = group.dim(0)?;
                group.get(rng.random_range(0..n))?
            } else {
                // Cosine similarity of the CAMPPlus style against each
                // speaker-anchor row picks the anchor index; the emotion
                // row at the same index enters the blend.
                let mut best = (f64::NEG_INFINITY, 0usize);
                let group_rows = group.dim(0)?;
                for r in 0..group_rows {
                    let a = self.spk_matrix[group_idx].get(r)?.to_dtype(DType::F32)?;
                    let dot = (&a * &style)?.sum_all()?.to_scalar::<f32>()? as f64;
                    let na = a.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
                    let ns = style.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
                    let sim = dot / (na * ns).max(1e-12);
                    if sim > best.0 {
                        best.0 = sim;
                        best.1 = r;
                    }
                }
                group.get(best.1)?
            };
            rows.push(row);
        }
        let matrix = Tensor::stack(&rows, 0)?; // (8, dim)
        let weights =
            Tensor::from_vec(vector.to_vec(), (vector.len(), 1), &device)?.to_dtype(DType::F32)?;
        // sum_i w_i * row_i -> (1, dim)
        let blended = weights.broadcast_mul(&matrix)?.sum(0)?.unsqueeze(0)?;
        Ok(blended)
    }

    /// `split_text_by_tokens`: split at punctuation into chunks whose
    /// tiktoken length (including the language prefix) fits the budget;
    /// `<|SPECIAL_TOKEN_n|>…<|SPECIAL_TOKEN_n|>` spans stay atomic.
    fn split_text_by_tokens(
        &self,
        text: &str,
        max_tokens: usize,
        lang_prefix: &str,
    ) -> Result<Vec<String>> {
        let capacity = self.config.gpt.max_text_tokens + 2;
        let prefix_len = self.tokenizer.encode(lang_prefix).len();
        let budget = max_tokens
            .min(capacity.saturating_sub(2))
            .saturating_sub(prefix_len)
            .max(1);

        let token_len = |s: &str| -> usize { self.tokenizer.encode(s).len() };
        if token_len(text) <= budget {
            return Ok(vec![text.to_string()]);
        }

        // Atomic pieces: protected <|SPECIAL_TOKEN_n|> spans.
        let mut chunks: Vec<String> = Vec::new();
        let protected = protected_spans(text);
        let mut pos = 0usize;
        let mut plain = String::new();
        for (start, end) in protected {
            plain.push_str(&text[pos..start]);
            if !plain.is_empty() {
                for part in split_punctuation(&plain) {
                    push_chunk(&mut chunks, &part, budget, &token_len);
                }
                plain.clear();
            }
            chunks.push(text[start..end].to_string());
            pos = end;
        }
        plain.push_str(&text[pos..]);
        if !plain.is_empty() {
            for part in split_punctuation(&plain) {
                push_chunk(&mut chunks, &part, budget, &token_len);
            }
        }

        // Merge neighbours within budget (upstream `segments` loop).
        let mut merged: Vec<String> = Vec::new();
        for chunk in chunks {
            let can_merge = merged
                .last()
                .is_some_and(|last| token_len(&format!("{last}{chunk}")) <= budget);
            if can_merge {
                let last = merged.pop().expect("checked last");
                merged.push(format!("{last}{chunk}"));
            } else {
                merged.push(chunk);
            }
        }
        if merged.is_empty() {
            merged.push(text.to_string());
        }
        Ok(merged)
    }
}

fn split_groups(matrix: &Tensor, emo_num: &[usize]) -> Result<Vec<Tensor>> {
    let rows = matrix.dim(0)?;
    let total: usize = emo_num.iter().sum();
    if total != rows {
        candle_core::bail!(
            "emotion matrix rows {rows} != sum(emo_num) {total} — checkpoint mismatch"
        );
    }
    let mut groups = Vec::with_capacity(emo_num.len());
    let mut offset = 0usize;
    for &n in emo_num {
        groups.push(matrix.narrow(0, offset, n)?);
        offset += n;
    }
    Ok(groups)
}

fn truncate_at_stop(codes: &Tensor, stop_token: u32) -> Result<Tensor> {
    let row = codes.get(0)?;
    let ids = row.to_vec1::<u32>()?;
    let len = ids
        .iter()
        .position(|&id| id == stop_token)
        .unwrap_or(ids.len());
    row.narrow(0, 0, len)?.unsqueeze(0)
}

fn random_seed() -> u64 {
    rand::rng().next_u64()
}

/// Byte ranges of `<|SPECIAL_TOKEN_\d+|>…<|SPECIAL_TOKEN_\d+|>` spans.
fn protected_spans(text: &str) -> Vec<(usize, usize)> {
    let re = regex::Regex::new(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>")
        .expect("static regex");
    re.find_iter(text).map(|m| (m.start(), m.end())).collect()
}

/// `re.split(r'(?<=[，。！？、；：,\.!\?;:\n])')` — keep delimiters attached.
fn split_punctuation(text: &str) -> Vec<String> {
    let delims = [
        '，', '。', '！', '？', '、', '；', '：', ',', '.', '!', '?', ';', ':', '\n',
    ];
    let mut parts = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if delims.contains(&ch) {
            parts.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts.into_iter().filter(|p| !p.is_empty()).collect()
}

/// Character-wise fallback split for over-budget pieces.
fn push_chunk(
    chunks: &mut Vec<String>,
    part: &str,
    budget: usize,
    token_len: &dyn Fn(&str) -> usize,
) {
    if token_len(part) <= budget {
        chunks.push(part.to_string());
        return;
    }
    let mut current = String::new();
    for ch in part.chars() {
        if !current.is_empty() && token_len(&format!("{current}{ch}")) > budget {
            chunks.push(std::mem::take(&mut current));
        }
        current.push(ch);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punctuation_split_keeps_delimiters() {
        let parts = split_punctuation("你好。世界！继续");
        assert_eq!(parts, vec!["你好。", "世界！", "继续"]);
        let parts = split_punctuation("a,b.c");
        assert_eq!(parts, vec!["a,", "b.", "c"]);
    }

    #[test]
    fn protected_spans_found() {
        let spans = protected_spans("x <|SPECIAL_TOKEN_1|>AB<|SPECIAL_TOKEN_1|> y");
        assert_eq!(spans.len(), 1);
        assert_eq!(
            &"x <|SPECIAL_TOKEN_1|>AB<|SPECIAL_TOKEN_1|> y"[spans[0].0..spans[0].1],
            "<|SPECIAL_TOKEN_1|>AB<|SPECIAL_TOKEN_1|>"
        );
    }

    #[test]
    fn groups_split_by_lengths() {
        let t = Tensor::arange(0f32, 10f32, &Device::Cpu)
            .unwrap()
            .reshape((10, 1))
            .unwrap();
        let groups = split_groups(&t, &[3, 7]).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].dim(0).unwrap(), 3);
        assert_eq!(groups[1].dim(0).unwrap(), 7);
    }
}
