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
    /// EXPERIMENT (off by default): decode the semantic codes with a CPU
    /// copy of the codec even when the model runs on CUDA. Measured on an
    /// RTX 5070: the per-segment codec tensors are small enough that the
    /// cuDNN convolutions run ~5x slower than the CPU ones (0.47-0.51 s vs
    /// 0.094 s per segment), so routing this post-sampling stage through
    /// the CPU shortens the critical path at the cost of two small device
    /// copies and one extra CPU codec load. Frame counts — and therefore
    /// audio durations — are device-independent, but the decoded features
    /// and final waveform differ from the all-GPU path within float32
    /// backend rounding. Not bit-identical to the default path; quality
    /// verification is on the caller.
    pub decode_on_cpu: bool,
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
            decode_on_cpu: false,
        }
    }
}

/// Maximum reference audio duration in seconds (`_load_and_cut_audio(..., 15)`).
pub const MAX_REFERENCE_SECS: usize = 15;

/// Normalize an 8-dim emotion vector `[happy, angry, sad, afraid, disgusted,
/// melancholic, surprised, calm]` with upstream `IndexTTS2.normalize_emo_vec`
/// (`infer_v2_5.py`): optional per-emotion bias factors followed by scaling so
/// the sum never exceeds `0.8`.
pub fn normalize_emo_vec(mut emo_vector: [f32; 8], apply_bias: bool) -> [f32; 8] {
    if apply_bias {
        const EMO_BIAS: [f32; 8] = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625];
        for (v, b) in emo_vector.iter_mut().zip(EMO_BIAS) {
            *v *= b;
        }
    }
    let sum: f32 = emo_vector.iter().sum();
    if sum > 0.8 {
        let scale = 0.8 / sum;
        for v in &mut emo_vector {
            *v *= scale;
        }
    }
    emo_vector
}

/// A preloaded reference clip at the two rates the pipeline needs.
pub struct Reference {
    /// 22.05 kHz mono f32 in [-1, 1].
    pub pcm_22k: Vec<f32>,
    /// 16 kHz mono f32 in [-1, 1].
    pub pcm_16k: Vec<f32>,
}

impl Reference {
    /// Build from arbitrary-rate mono PCM, truncating to [`MAX_REFERENCE_SECS`]
    /// (matching upstream `_load_and_cut_audio(..., 15)`) and resampling to
    /// both pipeline rates (approximate sinc resampler — see [`dsp::resample`]).
    pub fn from_pcm(pcm: &[f32], sample_rate: u32) -> Result<Self> {
        let max_samples = MAX_REFERENCE_SECS.saturating_mul(sample_rate as usize);
        let pcm = &pcm[..pcm.len().min(max_samples)];
        let pcm_22k = dsp::resample(pcm, sample_rate, crate::SAMPLE_RATE)?;
        let pcm_16k = dsp::resample(pcm, sample_rate, 16_000)?;
        Ok(Self { pcm_22k, pcm_16k })
    }
}

/// Cached conditioning derived from a [`Reference`].
#[derive(Clone)]
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
    /// Derived only from `reference`; invalidated on successful replacement.
    reference_emovec: Option<Tensor>,
    emo_reference: Option<Tensor>,
    /// Derived only from `emo_reference`; invalidated on replacement/clear.
    emo_reference_emovec: Option<Tensor>,
    /// Converted checkpoint directory (for the lazy CPU codec load of
    /// [`Options::decode_on_cpu`]).
    source_dir: std::path::PathBuf,
    /// Lazily loaded CPU copy of the codec (see [`Options::decode_on_cpu`]).
    codec_cpu: Option<SemanticCodec>,
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
            reference_emovec: None,
            emo_reference: None,
            emo_reference_emovec: None,
            source_dir: dir.to_path_buf(),
            codec_cpu: None,
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

    /// [`Options::decode_on_cpu`] backend: decode the semantic codes with a
    /// lazily loaded CPU copy of the codec and ship the features back to
    /// the model device. The codes themselves are produced on the model
    /// device, so this only adds one small download and one feature upload
    /// per segment.
    fn decode_codes_on_cpu(&mut self, codes: &Tensor) -> Result<Tensor> {
        if self.codec_cpu.is_none() {
            let path = self.source_dir.join("codec.safetensors");
            if !path.is_file() {
                candle_core::bail!(
                    "missing converted codec weights {} (run tools/tts/indextts_export_safetensors.py)",
                    path.display()
                );
            }
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &Device::Cpu)
            }?;
            self.codec_cpu = Some(SemanticCodec::load(&vb, &self.config.semantic_codec)?);
        }
        let codec = self.codec_cpu.as_ref().expect("just initialized");
        let features = codec.decode(&codes.to_device(&Device::Cpu)?)?;
        features.to_device(&self.device)
    }

    /// Compute (and cache) the conditioning of a reference clip; required
    /// before [`Self::generate`]. Mirrors the cache-refresh block of
    /// `infer_generator` (truncating to [`MAX_REFERENCE_SECS`]).
    pub fn prepare_reference(&mut self, reference: &Reference) -> Result<()> {
        let pcm_16k =
            &reference.pcm_16k[..reference.pcm_16k.len().min(MAX_REFERENCE_SECS * 16_000)];
        let pcm_22k = &reference.pcm_22k[..reference
            .pcm_22k
            .len()
            .min(MAX_REFERENCE_SECS * crate::SAMPLE_RATE as usize)];

        let spk_cond_emb = self.w2v_features(pcm_16k)?;

        let wav = Tensor::from_slice(pcm_22k, (1, pcm_22k.len()), &self.device)?;
        let preprocess = &self.config.s2mel.preprocess_params;
        let ref_mel = mel_spectrogram(&wav, &preprocess.spect_params, preprocess.sr, false)?;

        let fbank = dsp::campplus_fbank_tensor(pcm_16k, &self.device)?;
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
        self.reference_emovec = None;
        Ok(())
    }

    fn reference_emovec(&mut self) -> Result<Tensor> {
        if let Some(value) = &self.reference_emovec {
            return Ok(value.clone());
        }
        let reference = self.reference.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("prepare_reference must run before generate".into())
        })?;
        let value = self.gpt.emovec_from_reference(&reference.spk_cond_emb)?;
        self.reference_emovec = Some(value.clone());
        Ok(value)
    }

    fn emo_reference_emovec(&mut self) -> Result<Option<Tensor>> {
        if let Some(value) = &self.emo_reference_emovec {
            return Ok(Some(value.clone()));
        }
        let Some(emo_cond_emb) = &self.emo_reference else {
            return Ok(None);
        };
        let value = self.gpt.emovec_from_reference(emo_cond_emb)?;
        self.emo_reference_emovec = Some(value.clone());
        Ok(Some(value))
    }

    /// Optional separate emotion-reference conditioning; when never called
    /// (or cleared with [`Self::clear_emo_reference`]) the speaker conditioning
    /// doubles as the emotion reference with `emo_alpha` forced to 1.0,
    /// matching upstream.
    pub fn prepare_emo_reference(&mut self, reference: &Reference) -> Result<()> {
        let pcm_16k =
            &reference.pcm_16k[..reference.pcm_16k.len().min(MAX_REFERENCE_SECS * 16_000)];
        self.emo_reference = Some(self.w2v_features(pcm_16k)?);
        self.emo_reference_emovec = None;
        Ok(())
    }

    /// Clear any separate emotion reference set by [`Self::prepare_emo_reference`]
    /// so subsequent calls revert to speaker-reference emotion conditioning.
    pub fn clear_emo_reference(&mut self) {
        self.emo_reference = None;
        self.emo_reference_emovec = None;
    }

    /// Synthesize `text` (a whole paragraph; segmentation mirrors
    /// `split_text_by_tokens`) into PCM.
    pub fn generate(&mut self, text: &str, lang: &str, opts: &Options) -> Result<Outcome> {
        self.generate_with_cancel(text, lang, opts, &|| false)
    }

    /// Generate one segment, observing cancellation between model stages.
    pub fn generate_with_cancel(
        &mut self,
        text: &str,
        lang: &str,
        opts: &Options,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Outcome> {
        let _report = crate::profile::report();
        // Cloned tensor handles: the owned copy releases the `&self`
        // borrow so experiments like `decode_on_cpu` can take `&mut self`
        // inside the segment loop.
        let Some(reference) = self.reference.clone() else {
            candle_core::bail!("prepare_reference must run before generate");
        };
        let device = self.device.clone();

        // Emotion vector path scales weights by `emo_alpha` (upstream lines
        // 602-610) and forces the speaker clip to be the emotion reference
        // with alpha 1.0 (upstream lines 583-617).
        let mut rng = StdRng::seed_from_u64(opts.seed.unwrap_or_else(random_seed));
        let emovec_override = match opts.emo_vector {
            Some(raw_vector) => {
                let scale = opts.emo_alpha.clamp(0.0, 1.0);
                let vector = if (scale - 1.0).abs() > f32::EPSILON {
                    raw_vector.map(|x| ((x * scale * 10_000.0).trunc()) / 10_000.0)
                } else {
                    raw_vector
                };
                let emovec_mat = self.emotion_vector_matrix(
                    &vector,
                    &reference.style,
                    opts.emo_use_random,
                    &mut rng,
                )?;
                let weight_sum: f64 = vector.iter().map(|&x| x as f64).sum();
                Some((emovec_mat, weight_sum))
            }
            None => None,
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

        // Emotion conditioning is constant across the call's segments. Both
        // speaker and optional emotion-reference vectors are cached across
        // calls until replaced.
        let emovec = {
            let _g = crate::profile::scope(Some(&self.device), "emovec");
            let spk_emovec = self.reference_emovec()?;
            let base = if emovec_override.is_none() {
                if let Some(emo_emovec) = self.emo_reference_emovec()? {
                    let diff = (&emo_emovec - &spk_emovec)? * (opts.emo_alpha as f64);
                    (&spk_emovec + diff)?.contiguous()?
                } else {
                    spk_emovec
                }
            } else {
                spk_emovec
            };
            match &emovec_override {
                Some((mat, weight_sum)) => {
                    let scale = Tensor::new(((1.0 - weight_sum) as f32) as f64, base.device())?
                        .to_dtype(base.dtype())?;
                    (mat + &(&scale * &base)?)?
                }
                None => base,
            }
        };

        let mut wav_parts: Vec<Vec<f32>> = Vec::new();

        for seg_text in &segments {
            if cancelled() {
                candle_core::bail!("IndexTTS synthesis cancelled");
            }
            let mut ids = self.tokenizer.encode(&format!("{lang_prefix}{seg_text}"));
            ids.push(stop_text);
            let id_len = ids.len();
            let text_tokens = Tensor::from_vec(ids, (1, id_len), &device)?.to_dtype(DType::I64)?;

            let codes = {
                let _g = crate::profile::scope(Some(&self.device), "gpt");
                self.gpt.inference_speech_with_cancel(
                    &text_tokens,
                    lang_id,
                    &emovec,
                    &reference.style,
                    &opts.generate,
                    &mut rng,
                    cancelled,
                )?
            };
            if cancelled() {
                candle_core::bail!("IndexTTS synthesis cancelled");
            }
            let codes = truncate_at_stop(&codes, self.config.gpt.stop_mel_token)?;
            if codes.dim(1)? == 0 {
                candle_core::bail!("generation stopped without producing semantic tokens");
            }

            let s_infer = {
                let _g = crate::profile::scope(Some(&self.device), "codec");
                if opts.decode_on_cpu && !self.device.is_cpu() {
                    self.decode_codes_on_cpu(&codes)?
                } else {
                    self.codec.decode(&codes)?
                }
            };
            // Upstream uses int(...), which truncates positive frame counts.
            let target_len =
                (s_infer.dim(1)? as f32 * 1.72 * opts.duration_factor).max(1.0) as usize;
            let cond = {
                let _g = crate::profile::scope(Some(&self.device), "regulator");
                self.s2mel
                    .length_regulator
                    .forward(&s_infer, &[target_len])?
            };

            let cat_condition = Tensor::cat(&[&reference.prompt_condition, &cond], 1)?;
            let vc_target = {
                let _g = crate::profile::scope(Some(&self.device), "cfm");
                self.s2mel.cfm.inference_with_cancel(
                    &cat_condition,
                    &[cat_condition.dim(1)?],
                    &reference.ref_mel,
                    &reference.style,
                    opts.diffusion_steps,
                    opts.inference_cfg_rate,
                    cancelled,
                )?
            };
            if cancelled() {
                candle_core::bail!("IndexTTS synthesis cancelled");
            }
            let ref_frames = reference.ref_mel.dim(2)?;
            let vc_target = vc_target.narrow(2, ref_frames, vc_target.dim(2)? - ref_frames)?;

            let wav = {
                let _g = crate::profile::scope(Some(&self.device), "bigvgan");
                let wav = self.bigvgan.forward(&vc_target.to_dtype(DType::F32)?)?;
                let wav = wav.squeeze(1)?.squeeze(0)?.to_dtype(DType::F32)?;
                wav.to_vec1::<f32>()?
            };
            let scaled: Vec<f32> = wav
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
        rng: &mut StdRng,
    ) -> Result<Tensor> {
        if self.emo_matrix.len() != vector.len() {
            candle_core::bail!(
                "emotion vector length {} != emotion groups {}",
                vector.len(),
                self.emo_matrix.len()
            );
        }
        let device = self.device.clone();
        let style_row = style.to_dtype(DType::F32)?;
        let style_norm = if use_random {
            1.0
        } else {
            style_row.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64
        };
        let mut rows = Vec::with_capacity(vector.len());
        let rng = &mut *rng;
        for (group_idx, group) in self.emo_matrix.iter().enumerate() {
            let group = group.to_dtype(DType::F32)?;
            let row = if use_random {
                let n = group.dim(0)?;
                group.get(rng.random_range(0..n))?
            } else {
                // Vectorized cosine similarity against speaker-anchor rows.
                let spk_group = self.spk_matrix[group_idx].to_dtype(DType::F32)?;
                let dot = spk_group.broadcast_mul(&style_row)?.sum(1)?;
                let denom =
                    (spk_group.sqr()?.sum(1)?.sqrt()? * style_norm)?.clamp(1e-12f32, f32::MAX)?;
                let best_idx = (dot / denom)?.argmax(0)?.to_scalar::<u32>()? as usize;
                group.get(best_idx)?
            };
            rows.push(row);
        }
        let matrix = Tensor::stack(&rows, 0)?; // (8, dim)
        let weights =
            Tensor::from_vec(vector.to_vec(), (vector.len(), 1), &device)?.to_dtype(DType::F32)?;
        // sum_i w_i * row_i -> (1, dim)
        weights.broadcast_mul(&matrix)?.sum(0)?.unsqueeze(0)
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
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>").expect("static")
    });
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
    #[ignore = "requires converted weights and INDEXTTS_ORACLE_JSON"]
    fn reference_emovec_cache_is_reused_and_invalidated() {
        let dir = std::env::var_os("INDEXTTS_CONVERTED_DIR").expect("INDEXTTS_CONVERTED_DIR");
        let oracle = std::fs::read_to_string(
            std::env::var_os("INDEXTTS_ORACLE_JSON").expect("INDEXTTS_ORACLE_JSON"),
        )
        .unwrap();
        let oracle: serde_json::Value = serde_json::from_str(&oracle).unwrap();
        let pcm = |name: &str| {
            oracle["reference"][name]["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let mut reference = Reference {
            pcm_16k: pcm("pcm_16k"),
            pcm_22k: pcm("pcm_22k"),
        };
        let mut model = Model::load(Path::new(&dir), &Device::Cpu).unwrap();
        model.prepare_reference(&reference).unwrap();
        let first = model.reference_emovec().unwrap();
        assert_eq!(first.id(), model.reference_emovec().unwrap().id());
        reference.pcm_16k.reverse();
        reference.pcm_22k.reverse();
        model.prepare_reference(&reference).unwrap();
        assert!(model.reference_emovec.is_none());
        let replaced = model.reference_emovec().unwrap();
        assert_ne!(first.id(), replaced.id());
        let expected = model
            .gpt
            .emovec_from_reference(&model.reference.as_ref().unwrap().spk_cond_emb)
            .unwrap();
        assert_eq!(
            replaced.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

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

    #[test]
    fn reference_from_pcm_truncates_to_15_seconds() {
        let pcm = vec![0.1f32; 20 * 16_000];
        let reference = Reference::from_pcm(&pcm, 16_000).unwrap();
        assert_eq!(reference.pcm_16k.len(), 15 * 16_000);
        assert_eq!(reference.pcm_22k.len(), 15 * crate::SAMPLE_RATE as usize);
    }

    #[test]
    fn normalize_emo_vec_applies_bias_and_caps_sum() {
        let raw = [0.5f32, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5];
        let norm = normalize_emo_vec(raw, true);
        let sum: f32 = norm.iter().sum();
        assert!((sum - 0.8).abs() < 1e-6, "sum={sum}");
        let unscaled = normalize_emo_vec([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], false);
        assert!((unscaled[0] - 0.2).abs() < 1e-6);
    }
}
