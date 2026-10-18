//! Model loading and synthesis orchestration; playback belongs to the caller.
use crate::models::AnyKVCache;
use crate::models::code_predictor::{CodePredictorSession, sum_codebook_embeddings};
use crate::models::codec::{Decoder12Hz, Encoder12Hz};
use crate::models::speaker::SpeakerEncoder;
use crate::models::talker::codec_tokens;
use crate::{
    AudioBuffer, CodePredictor, CodePredictorConfig, Device, Language, ModelType,
    ParsedModelConfig, Speaker, SpeakerEncoderConfig, TalkerConfig, TalkerModel,
};
use crate::{audio, generation, models, tokenizer};
use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor};
use serde::Serialize;
use std::{collections::HashMap, path::Path, time::Instant};

mod options;
mod prompt;
mod streaming;
pub use options::SynthesisOptions;
pub use streaming::StreamingSession;

/// A sequence of codec frames, where each frame contains 16 codebook values
/// (1 semantic + 15 acoustic, formatted as `[semantic, acoustic_0..14]`).
pub type FrameCodes = Vec<Vec<u32>>;

/// Reference audio prompt for voice cloning.
///
/// Holds the speaker embedding and optional ICL (in-context learning) data.
/// Created via [`Qwen3TTS::create_voice_clone_prompt`].
pub struct VoiceClonePrompt {
    /// Speaker embedding from the ECAPA-TDNN encoder, shape `[enc_dim]` (typically 1024).
    pub speaker_embedding: Tensor,
    /// Reference audio codec codes for ICL mode, shape `[T, 16]`. `None` = x_vector_only mode.
    pub ref_codes: Option<Tensor>,
    /// Tokenized reference text for ICL mode.
    pub ref_text_ids: Option<Vec<u32>>,
}

/// Per-stage timing breakdown from a synthesis run.
#[derive(Debug, Clone, Serialize)]
pub struct SynthesisTiming {
    /// Time spent in the prefill phase (ms).
    pub prefill_ms: f64,
    /// Time spent in the autoregressive generation loop (ms).
    pub generation_ms: f64,
    /// Number of codec frames generated.
    pub generation_frames: usize,
    /// Time spent decoding codec frames to audio (ms).
    pub decode_ms: f64,
}

/// Main TTS interface using proper autoregressive pipeline.
///
/// Supports all 5 Qwen3-TTS model variants. Use [`model_type()`](Self::model_type)
/// to check which variant was loaded and [`supports_voice_cloning()`](Self::supports_voice_cloning)
/// / [`supports_preset_speakers()`](Self::supports_preset_speakers) to check capabilities.
pub struct Qwen3TTS {
    /// Talker model for semantic token generation
    talker: TalkerModel,
    /// Code predictor for acoustic token generation
    code_predictor: CodePredictor,
    /// 12Hz decoder for audio synthesis
    decoder: Decoder12Hz,
    /// Text tokenizer
    text_tokenizer: tokenizer::TextTokenizer,
    /// Speaker encoder for voice cloning (loaded when weights are present)
    speaker_encoder: Option<SpeakerEncoder>,
    /// Speech tokenizer encoder for ICL voice cloning (encodes reference audio → codes)
    speech_encoder: Option<Encoder12Hz>,
    /// Detected model variant (None if loaded without config.json)
    model_type: Option<ModelType>,
    /// Device to run inference on
    device: Device,
    /// Compute dtype for talker + code predictor (BF16 on CUDA/Metal, F32 on CPU)
    compute_dtype: DType,
}

impl Qwen3TTS {
    /// Load a model from a HuggingFace model ID or local path.
    ///
    /// Auto-detects the model variant (0.6B/1.7B, Base/CustomVoice/VoiceDesign)
    /// from `config.json` if present, falling back to weight inspection.
    ///
    /// The text tokenizer is resolved from `model_id/tokenizer.json` if present,
    /// otherwise downloaded from HuggingFace Hub. Use `tokenizer_id` to override.
    pub fn from_pretrained(model_id: &str, device: Device) -> Result<Self> {
        Self::from_pretrained_with_tokenizer(model_id, None, device)
    }

    /// Load a model with an explicit tokenizer source.
    ///
    /// `tokenizer_id` can be a local directory, a file path, or a HuggingFace
    /// model ID (e.g. `"Qwen/Qwen2-0.5B"`). If `None`, resolves from the
    /// model directory or falls back to the default tokenizer repo.
    pub fn from_pretrained_with_tokenizer(
        model_id: &str,
        tokenizer_id: Option<&str>,
        device: Device,
    ) -> Result<Self> {
        tracing::info!("Loading Qwen3-TTS from: {}", model_id);
        tracing::info!("Compute dtype: {:?}", compute_dtype_for_device(&device));

        // Try to parse config.json for auto-detection
        let config_path = Path::new(model_id).join("config.json");
        let parsed_config = if config_path.exists() {
            match ParsedModelConfig::from_file(&config_path) {
                Ok(cfg) => {
                    tracing::info!("Detected model variant: {}", cfg.label());
                    Some(cfg)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse config.json, falling back to weight inspection: {}",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        // Load text tokenizer
        let tok_source = tokenizer_id.unwrap_or(model_id);
        let text_tokenizer = tokenizer::TextTokenizer::from_pretrained(tok_source)?;

        // Load model weights
        let model_path = Path::new(model_id).join("model.safetensors");
        if !model_path.exists() {
            anyhow::bail!(
                "Model weights not found at {}. Please download the model first.",
                model_path.display()
            );
        }
        let weights = Self::load_weights(&model_path, &device)?;

        // Load speech tokenizer for decoder
        let st_path = Path::new(model_id).join("speech_tokenizer/model.safetensors");
        let st_weights = if st_path.exists() {
            Self::load_weights(&st_path, &device)?
        } else {
            // Fall back to looking in parent dir
            let alt_path = Path::new(model_id)
                .parent()
                .map(|p| p.join("speech_tokenizer/model.safetensors"));
            if let Some(p) = alt_path {
                if p.exists() {
                    Self::load_weights(&p, &device)?
                } else {
                    anyhow::bail!("Speech tokenizer weights not found");
                }
            } else {
                anyhow::bail!("Speech tokenizer weights not found");
            }
        };

        Self::build_from_components(
            &weights,
            &st_weights,
            text_tokenizer,
            parsed_config.as_ref(),
            &device,
        )
    }

    /// Load from pre-loaded weight tensors.
    ///
    /// Uses weight inspection for auto-detection. For config.json-based
    /// detection, use [`from_pretrained`](Self::from_pretrained) instead.
    pub fn from_weights(
        model_weights: &HashMap<String, Tensor>,
        decoder_weights: &HashMap<String, Tensor>,
        text_tokenizer: tokenizer::TextTokenizer,
        device: &Device,
    ) -> Result<Self> {
        Self::build_from_components(model_weights, decoder_weights, text_tokenizer, None, device)
    }

    /// Shared builder: assembles all model components from pre-loaded weights.
    ///
    /// When `parsed_config` is `Some`, uses config.json dimensions and model type.
    /// When `None`, auto-detects the model variant from weight shapes.
    fn build_from_components(
        model_weights: &HashMap<String, Tensor>,
        decoder_weights: &HashMap<String, Tensor>,
        text_tokenizer: tokenizer::TextTokenizer,
        parsed_config: Option<&ParsedModelConfig>,
        device: &Device,
    ) -> Result<Self> {
        let compute_dtype = compute_dtype_for_device(device);

        // Build TalkerModel
        let talker_config = if let Some(cfg) = parsed_config {
            TalkerConfig::from_parsed(cfg)
        } else {
            Self::detect_talker_config(model_weights)?
        };
        let talker = TalkerModel::from_weights_with_config_dtype(
            model_weights,
            talker_config,
            device,
            compute_dtype,
        )?;

        // Build CodePredictor
        let cp_config = if let Some(cfg) = parsed_config {
            CodePredictorConfig::from_parsed(cfg)
        } else {
            let talker_hidden = talker.config().hidden_size;
            if talker_hidden != 1024 {
                CodePredictorConfig {
                    codec_embed_dim: Some(talker_hidden),
                    ..CodePredictorConfig::default()
                }
            } else {
                CodePredictorConfig::default()
            }
        };
        let cp_weights = Self::filter_weights(model_weights, "talker.code_predictor.");
        let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, compute_dtype, device);
        let code_predictor = CodePredictor::new(cp_config, cp_vb)?;

        // Decoder (always F32 — convolutional, no attention)
        let decoder = Decoder12Hz::from_weights(decoder_weights, Default::default())?;

        // Speaker encoder (always F32, only present in Base models)
        let se_config = parsed_config.and_then(|c| c.speaker_encoder_config.clone());
        let speaker_encoder =
            Self::try_load_speaker_encoder(model_weights, se_config.as_ref(), device)?;

        // Speech encoder for ICL voice cloning
        let speech_encoder = Self::try_load_speech_encoder(decoder_weights, device)?;

        let model_type = parsed_config.map(|c| c.model_type);

        Ok(Self {
            talker,
            code_predictor,
            decoder,
            text_tokenizer,
            speaker_encoder,
            speech_encoder,
            model_type,
            device: device.clone(),
            compute_dtype,
        })
    }

    /// Detect talker config from weight shapes (fallback when no config.json).
    fn detect_talker_config(weights: &HashMap<String, Tensor>) -> Result<TalkerConfig> {
        let norm_weight = weights
            .get("talker.model.norm.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing talker.model.norm.weight"))?;
        let hidden_size = norm_weight.dim(0)?;
        Ok(if hidden_size == 2048 {
            TalkerConfig::custom_voice()
        } else {
            TalkerConfig::default()
        })
    }

    /// Returns the detected model type, or `None` if loaded without config.json.
    pub fn model_type(&self) -> Option<&ModelType> {
        self.model_type.as_ref()
    }

    /// Whether this model supports voice cloning (Base models with speaker encoder).
    pub fn supports_voice_cloning(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Whether this model supports preset speaker selection (CustomVoice models).
    ///
    /// Returns `true` for CustomVoice, `false` for Base and VoiceDesign.
    /// When `model_type` is unknown (loaded without config.json), returns `true`
    /// as a permissive default.
    pub fn supports_preset_speakers(&self) -> bool {
        match &self.model_type {
            Some(ModelType::CustomVoice) => true,
            Some(ModelType::Base) | Some(ModelType::VoiceDesign) => false,
            None => true, // permissive when unknown
        }
    }

    /// Whether this model supports voice design (text-described voice conditioning).
    ///
    /// Returns `true` for VoiceDesign, `false` for all other variants.
    pub fn supports_voice_design(&self) -> bool {
        matches!(&self.model_type, Some(ModelType::VoiceDesign))
    }

    /// Synthesize speech from text with default voice (Ryan, English).
    ///
    /// Convenience wrapper around [`synthesize_with_voice`](Self::synthesize_with_voice).
    pub fn synthesize(&self, text: &str, options: Option<SynthesisOptions>) -> Result<AudioBuffer> {
        self.synthesize_with_voice(text, Speaker::Ryan, Language::English, options)
    }

    /// Synthesize speech with per-stage timing breakdown.
    ///
    /// Same as [`synthesize_with_voice`](Self::synthesize_with_voice) but also
    /// returns a [`SynthesisTiming`] with prefill, generation, and decode durations.
    /// Uses [`sync_device`] at timing boundaries for accurate GPU measurements.
    pub fn synthesize_with_timing(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<(AudioBuffer, SynthesisTiming)> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;
        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // -- Prefill --
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        sync_device(&self.device)?;
        let t_prefill = Instant::now();

        let mut kv_caches = self.talker.new_kv_caches(gen_config.max_new_tokens + 256);
        let (hidden, logits) =
            self.talker
                .prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        sync_device(&self.device)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        // -- Generation --
        let t_gen = Instant::now();

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        sync_device(&self.device)?;
        let generation_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        let generation_frames = all_codes.len();

        // -- Decode --
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        let t_decode = Instant::now();
        let audio = self.decode_codes(&all_codes)?;

        sync_device(&self.device)?;
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

        let timing = SynthesisTiming {
            prefill_ms,
            generation_ms,
            generation_frames,
            decode_ms,
        };

        Ok((audio, timing))
    }

    /// Build trailing text embeddings from input token IDs.
    ///
    /// Returns (trailing_text_hidden, trailing_text_len, tts_pad_embed).
    /// The trailing text is: remaining text tokens (all except first) projected + tts_eos.
    /// After trailing text is exhausted, tts_pad is used for each subsequent step.
    fn build_trailing_text(&self, input_ids: &[u32]) -> Result<(Tensor, usize, Tensor)> {
        let trailing_text_hidden = if input_ids.len() > 1 {
            let remaining_proj = self.talker.get_projected_text_embeddings(&input_ids[1..])?;
            let tts_eos_embed = self.talker.get_tts_eos_embed()?;
            Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?
        } else {
            self.talker.get_tts_eos_embed()?
        };
        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;
        Ok((trailing_text_hidden, trailing_text_len, tts_pad_embed))
    }

    /// Core generation loop shared by all synthesis methods.
    ///
    /// Runs the autoregressive generation loop: for each frame, check EOS,
    /// generate acoustic codes via CodePredictor, build the residual VQ sum
    /// with trailing text fusion, and sample the next semantic token.
    ///
    /// Callers handle prefill (which varies by model variant) and post-processing
    /// (decode, ICL ref_codes prepending, etc.).
    #[allow(clippy::too_many_arguments)]
    fn generate_codes(
        &self,
        gen_config: &generation::GenerationConfig,
        sampling_ctx: &mut generation::SamplingContext,
        kv_caches: &mut [AnyKVCache],
        mut offset: usize,
        mut last_hidden: Tensor,
        initial_logits: &Tensor,
        trailing_text_hidden: &Tensor,
        trailing_text_len: usize,
        tts_pad_embed: &Tensor,
    ) -> Result<FrameCodes> {
        // Pre-build the token suppression mask once (reused every frame)
        let suppression_mask = generation::build_suppression_mask(
            codec_tokens::CODEC_VOCAB_SIZE,
            CODEC_EOS_TOKEN_ID,
            &self.device,
        )?;

        // GPU-side repetition penalty mask: [1, vocab] — updated incrementally
        // instead of transferring all generated tokens to CPU each frame.
        let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;
        let mut penalty_mask = Tensor::zeros((1, vocab_size), DType::F32, &self.device)?;

        // Per-generation acoustic code state: KV caches reused every frame,
        // plus the CUDA frame graph when it is enabled.
        let mut cp_codes = CodePredictorSession::new(&self.code_predictor);

        // Sample first semantic token
        let logits_2d = initial_logits.squeeze(1)?;
        let logits_2d = self.apply_generation_penalties_gpu(
            &logits_2d,
            &penalty_mask,
            gen_config,
            0,
            Some(&suppression_mask),
        )?;
        let mut semantic_token_tensor = generation::sample(&logits_2d, gen_config, sampling_ctx)?;
        tracing::trace!(target: "gpu_sync", "to_vec1 in generate_codes first token");
        let mut semantic_token: u32 = semantic_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
        // Update penalty mask with this token (O(1) CPU work)
        Self::update_penalty_mask(&mut penalty_mask, semantic_token, vocab_size)?;

        // Accumulate frames as GPU tensors: Vec of [16] U32 tensors
        // Deferred to_vec1 at the end eliminates per-frame acoustic code sync.
        let mut gpu_frames: Vec<Tensor> = Vec::new();

        #[cfg(feature = "profiling")]
        let _gen_span = tracing::info_span!("generate_frames").entered();

        for (token_count, frame_idx) in (1..).zip(0..gen_config.max_new_tokens) {
            if let Some(eos_id) = gen_config.eos_token_id
                && semantic_token == eos_id
            {
                break;
            }

            // Embedding lookup using GPU-resident token tensor (no CPU→GPU roundtrip)
            let semantic_embed = self
                .talker
                .get_codec_embedding_from_tensor(&semantic_token_tensor)?;

            #[cfg(feature = "profiling")]
            let _cp_span = tracing::info_span!("code_predictor", frame = frame_idx).entered();

            let acoustic_codes_tensor = cp_codes.generate(&last_hidden, &semantic_embed)?;

            #[cfg(feature = "profiling")]
            drop(_cp_span);

            // Build [16] frame tensor on GPU: [semantic_token, acoustic_0..14]
            let frame_tensor = Tensor::cat(
                &[&semantic_token_tensor.reshape(1)?, &acoustic_codes_tensor],
                0,
            )?;
            gpu_frames.push(frame_tensor);

            // Use GPU tensor directly for embedding lookup (avoids 15 CPU→GPU transfers)
            let summed = self
                .code_predictor
                .get_codec_embeddings_sum_from_tensor(&semantic_embed, &acoustic_codes_tensor)?;

            let text_addition = if frame_idx < trailing_text_len {
                trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            #[cfg(feature = "profiling")]
            let _talker_span = tracing::info_span!("talker_step", frame = frame_idx).entered();

            let (h, new_logits) =
                self.talker
                    .generate_step_with_embed(&step_input, kv_caches, offset)?;
            offset += 1;
            last_hidden = h;

            #[cfg(feature = "profiling")]
            drop(_talker_span);

            #[cfg(feature = "profiling")]
            let _sample_span = tracing::info_span!("sampling", frame = frame_idx).entered();

            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.apply_generation_penalties_gpu(
                &logits_2d,
                &penalty_mask,
                gen_config,
                token_count,
                Some(&suppression_mask),
            )?;
            semantic_token_tensor = generation::sample(&logits_2d, gen_config, sampling_ctx)?;
            tracing::trace!(target: "gpu_sync", "to_vec1 in generate_codes sampling");
            semantic_token = semantic_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
            Self::update_penalty_mask(&mut penalty_mask, semantic_token, vocab_size)?;
        }

        // Single GPU→CPU transfer: convert all accumulated GPU frames to FrameCodes
        self.gpu_frames_to_frame_codes(&gpu_frames)
    }

    /// Update the GPU-side penalty mask for a single token ID.
    ///
    /// Sets `penalty_mask[0, token_id] = 1.0` using slice_assign with a
    /// pre-built scalar. This is O(1) CPU work (no GPU→CPU transfer).
    fn update_penalty_mask(
        penalty_mask: &mut Tensor,
        token_id: u32,
        vocab_size: usize,
    ) -> Result<()> {
        let idx = token_id as usize;
        if idx < vocab_size {
            let one = Tensor::ones((1, 1), DType::F32, penalty_mask.device())?;
            *penalty_mask = penalty_mask.slice_assign(&[0..1, idx..idx + 1], &one)?;
        }
        Ok(())
    }

    /// Convert accumulated GPU frame tensors to FrameCodes via a single bulk transfer.
    fn gpu_frames_to_frame_codes(&self, gpu_frames: &[Tensor]) -> Result<FrameCodes> {
        if gpu_frames.is_empty() {
            return Ok(Vec::new());
        }
        // Stack all frames into [n_frames, 16], then single to_vec1
        let stacked = Tensor::stack(gpu_frames, 0)?; // [n_frames, 16]
        let n_frames = stacked.dim(0)?;
        let flat: Vec<u32> = stacked.flatten_all()?.to_vec1()?;
        let mut result = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let start = f * 16;
            result.push(flat[start..start + 16].to_vec());
        }
        Ok(result)
    }

    /// Synthesize speech with a specific voice and language.
    ///
    /// Uses the correct generation loop: CustomVoice prefill, autoregressive
    /// semantic tokens, per-frame acoustic code prediction via CodePredictor,
    /// residual VQ summation, and trailing text fusion.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `speaker` - Predefined speaker voice
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, etc.)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qwen3_tts::{Qwen3TTS, Speaker, Language, SynthesisOptions};
    ///
    /// let audio = model.synthesize_with_voice(
    ///     "Hello, world!",
    ///     Speaker::Ryan,
    ///     Language::English,
    ///     None,
    /// )?;
    /// audio.save("output.wav")?;
    /// ```
    pub fn synthesize_with_voice(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        if let Some(ModelType::Base) = &self.model_type {
            tracing::warn!(
                "Using preset speaker {:?} on a Base model. Base models are trained for \
                 voice cloning, not preset speakers — output will have an unpredictable voice. \
                 Use synthesize_voice_clone() with reference audio instead.",
                speaker
            );
        } else if let Some(ModelType::VoiceDesign) = &self.model_type {
            tracing::warn!(
                "Using preset speaker {:?} on a VoiceDesign model. VoiceDesign models \
                 are trained for text-described voice creation, not preset speakers.",
                speaker
            );
        }

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;

        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // Prefill with CustomVoice format
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        let mut kv_caches = self.talker.new_kv_caches(gen_config.max_new_tokens + 256);
        let (hidden, logits) =
            self.talker
                .prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        // Decode to audio
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        self.decode_codes(&all_codes)
    }

    /// Synthesize speech using a text-described voice (VoiceDesign model).
    ///
    /// Uses the same generation loop as [`Self::synthesize_with_voice`] but runs the
    /// VoiceDesign prefill instead of the predefined-speaker prefill. The voice
    /// is conditioned on a natural language description (e.g., "A cheerful young
    /// female voice with high pitch and energetic tone").
    ///
    /// The instruct text is tokenized with ChatML framing:
    /// `<|im_start|>user\n{instruct}<|im_end|>\n`
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `instruct` - Natural language voice description
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, etc.)
    pub fn synthesize_voice_design(
        &self,
        text: &str,
        instruct: &str,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        if let Some(ref mt) = self.model_type
            && *mt != ModelType::VoiceDesign
        {
            tracing::warn!(
                "Using VoiceDesign synthesis on a {:?} model. This model was not trained \
                     for text-described voice conditioning — output may be unpredictable.",
                mt
            );
        }

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;

        // Tokenize instruct with ChatML user framing: <|im_start|>user\n{instruct}<|im_end|>\n
        let instruct_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let instruct_ids = self.text_tokenizer.encode(&instruct_text)?;

        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // Prefill with VoiceDesign format
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        let mut kv_caches = self.talker.new_kv_caches(gen_config.max_new_tokens + 256);
        let (hidden, logits) = self.talker.prefill_voice_design(
            &input_ids,
            &instruct_ids,
            language,
            &mut kv_caches,
        )?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        // Decode to audio
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        self.decode_codes(&all_codes)
    }

    /// Convert list of frame codes to tensor [batch, 16, num_frames]
    pub fn codes_to_tensor(&self, codes: &[Vec<u32>]) -> Result<Tensor> {
        codes_to_tensor(codes, &self.device)
    }

    /// Decode raw frame codes to audio.
    ///
    /// Takes a slice of frames (each frame is a `Vec<u32>` of 16 codebook values)
    /// and runs the 12Hz decoder to produce an audio waveform at 24kHz.
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> Result<AudioBuffer> {
        let tensor = self.codes_to_tensor(codes)?;
        self.decode_tensor(&tensor)
    }

    /// Decode a codes tensor `[1, 16, T]` to audio.
    fn decode_tensor(&self, codes: &Tensor) -> Result<AudioBuffer> {
        let waveform = self.decoder.decode(codes)?;
        AudioBuffer::from_tensor(waveform, 24000)
    }

    /// Synthesize speech using a cloned voice, returning raw codes alongside audio.
    ///
    /// Identical to [`synthesize_voice_clone`](Self::synthesize_voice_clone) but also
    /// returns the raw generated codes (`Vec<Vec<u32>>`) for debugging.
    /// Each inner `Vec<u32>` is one frame: `[semantic, acoustic_0..14]` (16 values).
    pub fn synthesize_voice_clone_debug(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<(AudioBuffer, FrameCodes)> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;

        let gen_config = options.to_gen_config();

        let (mut kv_caches, offset, last_hidden, logits, trailing_text_hidden) =
            self.voice_clone_prefill(&input_ids, prompt, language, gen_config.max_new_tokens)?;

        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        // Prepend ref_codes for ICL decoder context (same fix as synthesize_voice_clone)
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        let audio = if let Some(ref_codes) = &prompt.ref_codes {
            let ref_frames = self.tensor_to_frame_codes(ref_codes)?;
            let ref_len = ref_frames.len();
            let mut combined = ref_frames;
            combined.extend(all_codes.iter().cloned());

            let mut audio = self.decode_codes(&combined)?;
            let total_frames = combined.len();
            // Proportional cut: matches official Qwen3-TTS Python implementation
            // cut = ref_len / total_len * wav.shape[0]
            let cut_samples = ref_len * audio.len() / total_frames.max(1);
            tracing::debug!(
                "ICL decode: ref_frames={}, gen_frames={}, total_samples={}, cut_samples={}",
                ref_len,
                all_codes.len(),
                audio.len(),
                cut_samples,
            );
            audio.samples = audio.samples[cut_samples.min(audio.len())..].to_vec();
            audio
        } else {
            self.decode_codes(&all_codes)?
        };
        Ok((audio, all_codes))
    }

    /// Get the device this model is running on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Create a streaming synthesis session with a specific voice and language.
    ///
    /// Returns an iterator that yields audio chunks as they are generated.
    /// Each chunk contains approximately `chunk_frames` frames worth of audio
    /// (default: 10 frames = ~800ms at 12.5 Hz frame rate).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qwen3_tts::{Qwen3TTS, Speaker, Language, SynthesisOptions};
    ///
    /// let options = SynthesisOptions::default();
    /// for chunk in model.synthesize_streaming("Hello!", Speaker::Ryan, Language::English, options)? {
    ///     let audio = chunk?;
    ///     // Play or process audio chunk (each ~800ms)
    /// }
    /// ```
    pub fn synthesize_streaming(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        let input_ids = self.text_tokenizer.encode(text)?;
        StreamingSession::new(self, &input_ids, speaker, language, options)
    }

    /// Stream a preset speaker with a ChatML style instruction (CustomVoice).
    pub fn synthesize_styled_streaming(
        &self,
        text: &str,
        instruct: &str,
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        anyhow::ensure!(
            self.model_type == Some(ModelType::CustomVoice),
            "style requires CustomVoice weights"
        );
        let input_ids = self.text_tokenizer.encode(text)?;
        let instruct_ids = self
            .text_tokenizer
            .encode(&format!("<|im_start|>user\n{instruct}<|im_end|>\n"))?;
        StreamingSession::new_with_style(
            self,
            &input_ids,
            &instruct_ids,
            speaker,
            language,
            options,
        )
    }

    /// Synthesize speech using a text-described voice (VoiceDesign model), streaming.
    ///
    /// Same as [`Self::synthesize_voice_design`] but returns a streaming session
    /// that yields audio chunks as they are generated.
    ///
    /// The instruct text is tokenized with ChatML framing:
    /// `<|im_start|>user\n{instruct}<|im_end|>\n`
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `instruct` - Natural language voice description (e.g., "A cheerful young female voice")
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, chunk_frames, etc.)
    pub fn synthesize_voice_design_streaming(
        &self,
        text: &str,
        instruct: &str,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        if let Some(ref mt) = self.model_type
            && *mt != ModelType::VoiceDesign
        {
            tracing::warn!(
                "Using VoiceDesign synthesis on a {:?} model. This model was not trained \
                     for text-described voice conditioning — output may be unpredictable.",
                mt
            );
        }

        let input_ids = self.text_tokenizer.encode(text)?;

        // Tokenize instruct with ChatML user framing: <|im_start|>user\n{instruct}<|im_end|>\n
        let instruct_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let instruct_ids = self.text_tokenizer.encode(&instruct_text)?;

        StreamingSession::new_voice_design(self, &input_ids, &instruct_ids, language, options)
    }

    // ── Voice cloning API ─────────────────────────────────────────────────

    /// Create a voice clone prompt from reference audio.
    ///
    /// When `ref_text` is `None`, produces an **x_vector_only** prompt (speaker
    /// embedding only). When `Some`, produces an **ICL** prompt (speaker embedding
    /// + reference audio codes + reference text) — requires a speech encoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the speaker encoder is not loaded.
    pub fn create_voice_clone_prompt(
        &self,
        ref_audio: &AudioBuffer,
        ref_text: Option<&str>,
    ) -> Result<VoiceClonePrompt> {
        let encoder = self.speaker_encoder.as_ref().ok_or_else(|| {
            let hint = match &self.model_type {
                Some(ModelType::CustomVoice) => {
                    " CustomVoice models use preset speakers (synthesize_with_voice), \
                     not voice cloning. Use a Base model for voice cloning."
                }
                Some(ModelType::VoiceDesign) => {
                    " VoiceDesign models use text-described voices, not voice cloning. \
                     Use a Base model for voice cloning."
                }
                _ => {
                    " Ensure model weights contain `speaker_encoder.*` keys \
                     (only Base models include a speaker encoder)."
                }
            };
            anyhow::anyhow!("Speaker encoder not available.{}", hint)
        })?;

        // Resample to 24kHz if needed — both encoders assume 24kHz input
        let ref_audio_24k;
        let ref_audio = if ref_audio.sample_rate != 24000 {
            tracing::info!(
                "Resampling reference audio from {}Hz to 24000Hz",
                ref_audio.sample_rate
            );
            ref_audio_24k = audio::resample_to_24k(ref_audio)?;
            &ref_audio_24k
        } else {
            ref_audio
        };

        let speaker_embedding = encoder.encode(ref_audio)?; // [enc_dim]

        // ICL data: encode reference audio to codes and tokenize reference text
        let (ref_codes, ref_text_ids) = if let Some(text) = ref_text {
            let speech_enc = self.speech_encoder.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "ICL voice cloning requires a speech encoder, but it was not loaded. \
                     Ensure the speech tokenizer weights contain encoder keys, or use \
                     x_vector_only mode by passing ref_text=None."
                )
            })?;

            let codes = speech_enc.encode(ref_audio)?; // [T_frames, 16]
            let text_ids = self.text_tokenizer.encode(text)?;

            (Some(codes), Some(text_ids))
        } else {
            (None, None)
        };

        Ok(VoiceClonePrompt {
            speaker_embedding,
            ref_codes,
            ref_text_ids,
        })
    }

    /// Synthesize speech using a cloned voice.
    ///
    /// Uses the same generation loop as [`Self::synthesize_with_voice`] but runs the
    /// voice-clone prefill instead of the predefined-speaker prefill.
    ///
    /// When the prompt contains ICL data (ref_codes + ref_text_ids), the model
    /// is conditioned on reference audio/text to better reproduce the speaker's voice.
    pub fn synthesize_voice_clone(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        self.synthesize_voice_clone_debug(text, prompt, language, options)
            .map(|(audio, _codes)| audio)
    }

    /// Incremental cloning with the same ICL prefill as full synthesis.
    pub fn synthesize_voice_clone_streaming(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        anyhow::ensure!(
            self.model_type == Some(ModelType::Base),
            "voice cloning requires a Base model"
        );
        let input_ids = self.text_tokenizer.encode(text)?;
        StreamingSession::new_voice_clone(self, &input_ids, prompt, language, options)
    }

    #[allow(clippy::type_complexity)]
    fn voice_clone_prefill(
        &self,
        input_ids: &[u32],
        prompt: &VoiceClonePrompt,
        language: Language,
        max_frames: usize,
    ) -> Result<(Vec<AnyKVCache>, usize, Tensor, Tensor, Tensor)> {
        let is_icl = prompt.ref_codes.is_some() && prompt.ref_text_ids.is_some();
        // Cast speaker embedding to compute dtype (speaker encoder produces F32)
        let speaker_embed = prompt.speaker_embedding.to_dtype(self.compute_dtype)?;

        // Voice clone prefill (9 positions for ICL, 10 for x_vector_only)
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        let mut kv_caches = self.talker.new_kv_caches(
            max_frames
                + input_ids.len()
                + prompt.ref_text_ids.as_ref().map_or(0, Vec::len)
                + prompt
                    .ref_codes
                    .as_ref()
                    .map_or(0, |codes| codes.dim(0).unwrap_or(0))
                + 256,
        );
        let (hidden, logits) = self.talker.prefill_voice_clone(
            input_ids,
            &speaker_embed,
            language,
            is_icl,
            &mut kv_caches,
        )?;
        let prefill_len = hidden.dim(1)?;
        let mut offset = prefill_len;

        // Initialize last_hidden from prefill; updated by ICL block if active.
        let mut last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        // ICL extension (if reference codes + text are provided)
        let (trailing_text_hidden, logits) = if let (Some(ref_codes), Some(ref_text_ids)) =
            (&prompt.ref_codes, &prompt.ref_text_ids)
        {
            let ref_codec_embeds = self.sum_ref_codec_embeddings(ref_codes)?;

            // In ICL mode, all text tokens go into the ICL prompt (Python:
            // text_id=input_id[:, 3:-5] passes ALL target text tokens).
            // In the non-ICL path the first text token is consumed by the prefill,
            // so only the remaining tokens go to trailing_text.
            let (icl_embed, icl_trailing) =
                self.talker
                    .build_icl_prompt(input_ids, ref_text_ids, &ref_codec_embeds, false)?;

            let icl_len = icl_embed.dim(1)?;
            if icl_len > 0 {
                let mask = models::transformer::create_causal_mask(icl_len, offset, &self.device)?;

                let mut icl_hidden = icl_embed;
                for (i, layer) in self.talker.layers_iter().enumerate() {
                    icl_hidden = layer.forward(
                        &icl_hidden,
                        self.talker.rope(),
                        Some(&mask),
                        Some(&mut kv_caches[i]),
                        offset,
                    )?;
                }
                icl_hidden = self.talker.apply_norm(&icl_hidden)?;
                offset += icl_len;

                let last_icl_hidden = icl_hidden.i((.., icl_len - 1..icl_len, ..))?;
                let new_logits = self.talker.apply_codec_head(&last_icl_hidden)?;

                // Update last_hidden so the code predictor is conditioned on
                // the ICL context, not the stale prefill hidden state.
                last_hidden = last_icl_hidden;

                (icl_trailing, new_logits)
            } else {
                let trailing = self.build_default_trailing_text(input_ids)?;
                (trailing, logits)
            }
        } else {
            let trailing = self.build_default_trailing_text(input_ids)?;
            (trailing, logits)
        };

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        Ok((kv_caches, offset, last_hidden, logits, trailing_text_hidden))
    }

    /// Convert a ref_codes tensor `[T_frames, 16]` to `Vec<Vec<u32>>` frame format.
    fn tensor_to_frame_codes(&self, codes: &Tensor) -> Result<FrameCodes> {
        let (n_frames, n_codebooks) = codes.dims2()?;
        let codes_u32 = codes.to_dtype(DType::U32)?;
        let mut frames = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let frame_tensor = codes_u32.i(f)?; // [16]
            let frame_vec: Vec<u32> = frame_tensor.to_vec1()?;
            debug_assert_eq!(frame_vec.len(), n_codebooks);
            frames.push(frame_vec);
        }
        Ok(frames)
    }

    /// Sum reference codec embeddings across all 16 codebook groups.
    ///
    /// For each frame:
    /// - Group 0 (semantic): `talker.codec_embedding(ref_codes[:, 0])`
    /// - Groups 1–15 (acoustic): `code_predictor.codec_embeddings[i-1](ref_codes[:, i])`
    /// - Sum all 16 → single embedding per frame
    ///
    /// # Arguments
    /// * `ref_codes` — shape `[T_frames, 16]` of i64 codes
    ///
    /// # Returns
    /// Tensor of shape `[1, T_frames, hidden_size]`
    fn sum_ref_codec_embeddings(&self, ref_codes: &Tensor) -> Result<Tensor> {
        // Group 0: semantic codes → talker.codec_embedding
        let semantic_codes = ref_codes.i((.., 0))?; // [T_frames]
        let semantic_codes = semantic_codes.to_dtype(candle_core::DType::U32)?;
        let semantic = self.talker.get_codec_embedding_batch(&semantic_codes)?; // [1, T, hidden]

        // Groups 1-15: acoustic codes → code_predictor.embed_codes_for_group
        let mut embeddings = vec![semantic];
        for group in 1..16 {
            let group_codes = ref_codes.i((.., group))?; // [T_frames]
            let group_codes = group_codes.to_dtype(candle_core::DType::U32)?;
            let group_embed = self
                .code_predictor
                .embed_codes_for_group(group - 1, &group_codes)?; // [1, T, embed_dim]
            embeddings.push(group_embed);
        }

        sum_codebook_embeddings(&embeddings)
    }

    /// Build default trailing text embeddings (for non-ICL mode).
    fn build_default_trailing_text(&self, input_ids: &[u32]) -> Result<Tensor> {
        let (hidden, _len, _pad) = self.build_trailing_text(input_ids)?;
        Ok(hidden)
    }

    /// Apply repetition penalty, token suppression, and min_new_tokens EOS suppression
    /// using a pre-built `[1, vocab]` penalty mask on GPU.
    ///
    /// The mask is updated incrementally via [`Self::update_penalty_mask`] after each
    /// sampled token, eliminating the O(n) GPU→CPU transfer that grows with
    /// each frame.
    fn apply_generation_penalties_gpu(
        &self,
        logits: &Tensor,
        penalty_mask: &Tensor,
        config: &generation::GenerationConfig,
        token_count: usize,
        suppression_mask: Option<&generation::SuppressionMask>,
    ) -> Result<Tensor> {
        let logits = logits.to_dtype(DType::F32)?;

        // 1. Repetition penalty via pre-built GPU mask
        let logits = if config.repetition_penalty != 1.0 {
            generation::apply_repetition_penalty_with_mask(
                &logits,
                penalty_mask,
                config.repetition_penalty,
            )?
        } else {
            logits
        };

        // 2. Token suppression
        let logits = if let Some(mask) = suppression_mask {
            generation::apply_token_suppression_with_mask(&logits, mask)?
        } else {
            generation::apply_token_suppression(
                &logits,
                codec_tokens::CODEC_VOCAB_SIZE,
                CODEC_EOS_TOKEN_ID,
            )?
        };

        // 3. Min new tokens EOS suppression
        if token_count < config.min_new_tokens
            && let Some(eos_id) = config.eos_token_id
        {
            let vocab = logits.dim(1)?;
            let batch = logits.dim(0)?;
            let mut mask_data = vec![0.0f32; vocab];
            mask_data[eos_id as usize] = 1.0;
            let eos_mask = Tensor::new(mask_data.as_slice(), logits.device())?
                .unsqueeze(0)?
                .broadcast_as((batch, vocab))?;
            let neg_inf =
                Tensor::new(&[f32::NEG_INFINITY], logits.device())?.broadcast_as((batch, vocab))?;
            let zeros = Tensor::zeros((batch, vocab), DType::F32, logits.device())?;
            let is_eos = eos_mask.gt(&zeros)?;
            return Ok(is_eos.where_cond(&neg_inf, &logits)?);
        }

        Ok(logits)
    }

    /// Returns `true` if a speech encoder is loaded (ICL voice cloning is available).
    pub fn has_speech_encoder(&self) -> bool {
        self.speech_encoder.is_some()
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Attempt to load the speaker encoder from model weights.
    ///
    /// Returns `Ok(Some(encoder))` if `speaker_encoder.*` keys are found,
    /// `Ok(None)` if they are absent. When `config` is provided, uses the
    /// parsed enc_dim; otherwise falls back to defaults (enc_dim=1024).
    fn try_load_speaker_encoder(
        weights: &HashMap<String, Tensor>,
        config: Option<&SpeakerEncoderConfig>,
        device: &Device,
    ) -> Result<Option<SpeakerEncoder>> {
        let has_se_weights = weights.keys().any(|k| k.starts_with("speaker_encoder."));
        if !has_se_weights {
            return Ok(None);
        }

        let config = config.cloned().unwrap_or_default();
        tracing::info!(
            "Loading speaker encoder (ECAPA-TDNN, enc_dim={}) for voice cloning...",
            config.enc_dim
        );
        let se_weights = Self::filter_weights(weights, "speaker_encoder.");
        let se_vb = candle_nn::VarBuilder::from_tensors(se_weights, DType::F32, device);
        let encoder = SpeakerEncoder::new(config, se_vb)?;
        Ok(Some(encoder))
    }

    /// Attempt to load the speech encoder (Mimi) from speech tokenizer weights.
    ///
    /// The speech encoder encodes raw audio to 12Hz codec codes, needed for
    /// ICL voice cloning. Returns `Ok(None)` if encoder keys are absent or
    /// loading fails (non-fatal — ICL mode just won't be available).
    fn try_load_speech_encoder(
        weights: &HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Option<Encoder12Hz>> {
        // Check for encoder-related keys (either HF or candle format)
        let has_encoder_keys = weights
            .keys()
            .any(|k| k.starts_with("encoder.") || k.starts_with("encoder_transformer."));
        if !has_encoder_keys {
            return Ok(None);
        }

        tracing::debug!("Attempting to load speech encoder (Mimi) for ICL voice cloning...");
        match Encoder12Hz::from_weights(weights, device) {
            Ok(enc) => {
                tracing::info!("Loaded speech encoder — ICL voice cloning available");
                Ok(Some(enc))
            }
            Err(e) => {
                tracing::debug!(
                    "Speech encoder not available ({}). ICL voice cloning disabled.",
                    e
                );
                Ok(None)
            }
        }
    }

    /// Load weights from safetensors file.
    ///
    /// Tensors are loaded in their native dtype (typically BF16 for Qwen3-TTS).
    /// Each component's VarBuilder handles casting to its target dtype.
    fn load_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
        Ok(candle_core::safetensors::load(path, device)?)
    }

    /// Filter weights by prefix, removing the prefix from keys.
    pub(crate) fn filter_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> HashMap<String, Tensor> {
        weights
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix(prefix)
                    .map(|stripped| (stripped.to_string(), v.clone()))
            })
            .collect()
    }
}

/// Convert a slice of codec frames into a tensor of shape `[1, 16, T]`.
///
/// Each frame must contain exactly 16 codebook values. The output layout is
/// `[q0_f0, q0_f1, ...], [q1_f0, q1_f1, ...]` matching the decoder's expectation.
pub fn codes_to_tensor(codes: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let num_frames = codes.len();
    if num_frames == 0 {
        return Ok(Tensor::zeros((1, 16, 0), DType::I64, device)?);
    }

    let mut data = vec![0i64; 16 * num_frames];
    for (frame, frame_codes) in codes.iter().enumerate() {
        for (q, &code) in frame_codes.iter().enumerate() {
            data[q * num_frames + frame] = code as i64;
        }
    }

    Ok(Tensor::from_vec(data, (1, 16, num_frames), device)?)
}

/// Return the recommended compute dtype for the given device.
///
/// Returns `BF16` for CUDA/Metal (lower memory, faster attention) and `F32` for CPU.
pub fn compute_dtype_for_device(device: &Device) -> DType {
    if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Force the GPU to complete all pending work before returning.
///
/// On CUDA/Metal, GPU operations are asynchronous — `Instant::now()` would
/// measure submission time, not completion time. This helper forces a sync
/// by creating a tiny tensor and reading it back to the CPU.
///
/// On CPU this is a no-op.
pub fn sync_device(device: &Device) -> Result<()> {
    match device {
        Device::Cpu => Ok(()),
        _ => {
            // Force a GPU→CPU sync by reading a scalar back
            let _: Vec<f32> = Tensor::zeros(1, DType::F32, device)?.to_vec1()?;
            Ok(())
        }
    }
}

/// The codec end-of-sequence token ID (2150).
///
/// Generation stops when this token is sampled. This is in the codec vocabulary
/// `[0, 3072)`, not the text vocabulary.
pub const CODEC_EOS_TOKEN_ID: u32 = codec_tokens::CODEC_EOS;

/// Number of audio samples per codec frame at 24kHz (1920 = 80ms per frame at 12Hz).
pub const SAMPLES_PER_FRAME: usize = 1920;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Qwen3TTSConfig, auto_device, device_info, parse_device};

    /// Qualifies CUDA replay against eager inference with the real model.
    ///
    /// Covers the conditions a fixed-frame graph must satisfy: identical codes
    /// for different hidden states, repeated replay of an earlier frame,
    /// scratch reuse across frames, and a failed launch that must disable the
    /// graph without corrupting later frames.
    #[test]
    #[ignore = "requires a local CUDA model; run serially"]
    fn cuda_code_predictor_replay_matches_eager() -> Result<()> {
        use anyhow::Context;
        let model_path = std::env::var("QWEN_CP_TEST_MODEL").context("QWEN_CP_TEST_MODEL")?;
        anyhow::ensure!(Path::new(&model_path).is_dir(), "local model required");
        let device = Device::new_cuda(0)?;
        let model = Qwen3TTS::from_pretrained(&model_path, device.clone())?;
        let predictor = &model.code_predictor;
        let dtype = model.compute_dtype;
        let dim = predictor.codec_embed_dim();
        let frame = |seed: usize| -> Result<Tensor> {
            let values: Vec<f32> = (0..dim)
                .map(|i| ((i as f32 * 0.37 + seed as f32 * 11.0).sin()) * 0.5)
                .collect();
            Ok(Tensor::from_vec(values, (1, 1, dim), &device)?.to_dtype(dtype)?)
        };

        let mut eager = CodePredictorSession::new(predictor);
        let mut replayed = CodePredictorSession::new(predictor);
        let mut first: Option<Vec<u32>> = None;
        for seed in 0..8usize {
            let (hidden, semantic) = (frame(seed)?, frame(seed + 64)?);
            // Read the eager result before the replay overwrites its scratch.
            let expected = eager.generate(&hidden, &semantic)?.to_vec1::<u32>()?;
            let actual = replayed.generate(&hidden, &semantic)?.to_vec1::<u32>()?;
            assert_eq!(actual, expected, "codes differ for seed {seed}");
            if seed == 0 {
                first = Some(expected);
            }
        }
        assert!(replayed.replaying(), "unexpected graph fallback");

        // A late replay of the first frame must not read stale KV state.
        let (hidden, semantic) = (frame(0)?, frame(64)?);
        let repeated = replayed.generate(&hidden, &semantic)?.to_vec1::<u32>()?;
        assert_eq!(repeated, first.expect("first frame recorded"));
        assert!(replayed.replaying());

        // A failed launch must disable the graph and keep later frames correct.
        let wrong = Tensor::zeros((1, 1, dim + 1), dtype, &device)?;
        assert!(replayed.generate(&wrong, &semantic).is_err());
        assert!(!replayed.replaying());
        let expected = eager.generate(&hidden, &semantic)?.to_vec1::<u32>()?;
        assert_eq!(
            replayed.generate(&hidden, &semantic)?.to_vec1::<u32>()?,
            expected
        );
        Ok(())
    }

    /// Qualifies capture when `new_kv_caches` fell back to concatenation.
    ///
    /// CUDA normally takes the pre-allocated path, but a failed allocation
    /// falls back to `cat`-grown K/V whose final tensors are allocated inside
    /// the capture and still held by the cache when it ends. Retaining
    /// capture-time allocations leaves unpaired allocation nodes, and on this
    /// setup the driver rejects the graph at `end_capture` with
    /// `CUDA_ERROR_INVALID_VALUE` (the recorder's RAII still restores the
    /// stream). The session therefore relies on eager fallback, and this test
    /// pins both halves of that contract: a rejection must be INVALID_VALUE
    /// with the stream usable again, and a driver that accepts the graph must
    /// reproduce eager codes exactly.
    #[test]
    #[ignore = "requires a local CUDA model; run serially"]
    #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
    fn cuda_concat_cache_capture_matches_eager() -> Result<()> {
        use crate::models::KVCache;
        use anyhow::Context;
        let model_path = std::env::var("QWEN_CP_TEST_MODEL").context("QWEN_CP_TEST_MODEL")?;
        anyhow::ensure!(Path::new(&model_path).is_dir(), "local model required");
        let device = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &device else {
            unreachable!()
        };
        let model = Qwen3TTS::from_pretrained(&model_path, device.clone())?;
        let predictor = &model.code_predictor;
        let dtype = model.compute_dtype;
        let dim = predictor.codec_embed_dim();
        let frame = |seed: usize| -> Result<Tensor> {
            let values: Vec<f32> = (0..dim)
                .map(|i| (i as f32 * 0.37 + seed as f32 * 11.0).sin() * 0.5)
                .collect();
            Ok(Tensor::from_vec(values, (1, 1, dim), &device)?.to_dtype(dtype)?)
        };
        let concat_caches = || {
            predictor
                .new_kv_caches()
                .into_iter()
                .map(|_| AnyKVCache::Concat(KVCache::new()))
                .collect::<Vec<_>>()
        };
        let mut captured = concat_caches();
        let mut eager = concat_caches();

        let _guard = cuda.enable_cuda_graph_htod_cache();
        let (hidden, semantic) = (frame(0)?, frame(64)?);
        let warmup = predictor.generate_acoustic_codes(&hidden, &semantic, &mut captured)?;
        // `captured` must outlive `replay`: its retained K/V were allocated
        // inside the capture and are the addresses every launch rewrites.
        let record = candle_cuda_graph::Replay::record(
            cuda,
            &[&hidden, &semantic],
            &warmup,
            vec![],
            |inputs| {
                predictor
                    .generate_acoustic_codes(&inputs[0], &inputs[1], &mut captured)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))
            },
        );
        let replay = match record {
            Ok(replay) => replay,
            Err(error) => {
                assert!(
                    format!("{error:?}").contains("INVALID_VALUE"),
                    "unexpected capture failure for the concat variant: {error:?}"
                );
                // A rejected recording must leave the stream capturable again.
                anyhow::ensure!(
                    !candle_cuda_graph::is_capturing(&cuda.cuda_stream())?,
                    "failed capture left the stream capturing"
                );
                return Ok(());
            }
        };

        let mut check = |seed: usize| -> Result<bool> {
            let (hidden, semantic) = (frame(seed)?, frame(seed + 64)?);
            // Read the eager result before the replay overwrites its scratch.
            let expected = predictor.generate_acoustic_codes(&hidden, &semantic, &mut eager)?;
            let expected = expected.to_vec1::<u32>()?;
            match replay.launch(&[&hidden, &semantic]) {
                Ok(actual) => {
                    assert_eq!(
                        actual.to_vec1::<u32>()?,
                        expected,
                        "codes differ for seed {seed}"
                    );
                    Ok(true)
                }
                Err(error) => {
                    assert!(
                        format!("{error:?}").contains("INVALID_VALUE"),
                        "unexpected replay failure for seed {seed}: {error:?}"
                    );
                    Ok(false)
                }
            }
        };
        assert!(check(0)?, "first launch must reproduce the warmup frame");
        for seed in 1..8usize {
            if !check(seed)? {
                return Ok(());
            }
        }
        // A late repeat of the first frame is where this variant gave up.
        let _ = check(0)?;
        Ok(())
    }

    /// Measures the dispatch headroom of the two loops left after the code
    /// predictor frame graph: one talker step at a fixed KV length, and the
    /// 12 Hz decoder at the shapes streaming chunks and full synthesis use.
    ///
    /// Evaluation only. Both paths rewrite the same KV position, every replay
    /// must reproduce the eager result bit for bit, and the reported numbers
    /// are per-call wall time rather than a synthesis benchmark.
    #[test]
    #[ignore = "requires a local CUDA model; run serially"]
    #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
    fn cuda_graph_headroom_probe() -> Result<()> {
        use anyhow::Context;
        use std::time::Instant;

        /// Per-call wall time in milliseconds, one untimed call first.
        fn per_call_ms(
            device: &Device,
            calls: usize,
            mut run: impl FnMut(usize) -> Result<Tensor>,
        ) -> Result<f64> {
            run(0)?;
            device.synchronize()?;
            let started = Instant::now();
            for call in 1..=calls {
                run(call)?;
            }
            device.synchronize()?;
            Ok(started.elapsed().as_secs_f64() * 1000.0 / calls as f64)
        }

        let model_path = std::env::var("QWEN_CP_TEST_MODEL").context("QWEN_CP_TEST_MODEL")?;
        anyhow::ensure!(Path::new(&model_path).is_dir(), "local model required");
        let device = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &device else {
            unreachable!("new_cuda returns a cuda device")
        };
        let model = Qwen3TTS::from_pretrained(&model_path, device.clone())?;

        // A talker step is only shape-stable at a fixed KV length. Every stage
        // below reads its own cache set warmed to the same offset, because a
        // set that a warm-up already advanced would hand the capture a longer
        // KV length than the one recorded.
        let offset = 30usize;
        let step_input = model
            .talker
            .get_codec_embedding_from_tensor(&Tensor::new(&[7u32], &device)?)?;
        let warm_caches = || -> Result<Vec<AnyKVCache>> {
            let mut caches = model.talker.new_kv_caches(offset + 64);
            for filled in 0..offset {
                model
                    .talker
                    .generate_step_with_embed(&step_input, &mut caches, filled)?;
            }
            Ok(caches)
        };
        let talker_step = |caches: &mut [AnyKVCache], at: usize| {
            model
                .talker
                .generate_step_with_embed(&step_input, caches, at)
                .map(|(_, logits)| logits)
        };
        // Eager timing walks the offset forward the way a session fills the
        // cache; the recorded graph is fixed at `offset`.
        let mut eager_caches = warm_caches()?;
        let eager_a = per_call_ms(&device, 12, |call| {
            talker_step(&mut eager_caches, offset + call)
        })?;
        let mut verify_caches = warm_caches()?;
        let expected = talker_step(&mut verify_caches, offset)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut capture_caches = warm_caches()?;
        // The warm-up has to run under the guard, on a cache set the capture
        // has not advanced, so the host data the step uploads is cached.
        let guard = cuda.enable_cuda_graph_htod_cache();
        let mut warmup_caches = warm_caches()?;
        let warmup = talker_step(&mut warmup_caches, offset)?;
        let talker_record =
            candle_cuda_graph::Replay::record(cuda, &[&step_input], &warmup, vec![], |inputs| {
                model
                    .talker
                    .generate_step_with_embed(&inputs[0], &mut capture_caches, offset)
                    .map(|(_, logits)| logits)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))
            });
        drop(guard);
        let talker_replay = match talker_record {
            Ok(replay) => replay,
            Err(error) => {
                println!(
                    "{}",
                    serde_json::json!({"probe": "talker_step", "captured": false, "error": error.to_string()})
                );
                return Ok(());
            }
        };
        let replayed = talker_replay
            .launch(&[&step_input])?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(replayed, expected, "talker replay differs from eager");
        let replay_a = per_call_ms(&device, 12, |_| {
            talker_replay
                .launch(&[&step_input])
                .map_err(anyhow::Error::from)
        })?;
        let eager_b = per_call_ms(&device, 12, |call| {
            talker_step(&mut eager_caches, offset + 13 + call)
        })?;
        let replay_b = per_call_ms(&device, 12, |_| {
            talker_replay
                .launch(&[&step_input])
                .map_err(anyhow::Error::from)
        })?;
        println!(
            "{}",
            serde_json::json!({"probe": "talker_step", "captured": true,
                "recorded_kv_len": offset + 1, "bitwise_equal": true,
                "eager_ms": [eager_a, eager_b], "replay_ms": [replay_a, replay_b]})
        );
        drop(talker_replay);

        // Streaming decodes 25 frames of left context plus 10 new frames on
        // every chunk after the first; full synthesis decodes all frames once.
        for frames in [35usize, 185] {
            let codes = Tensor::from_vec(
                (0..16 * frames)
                    .map(|index| (index % 2048) as i64)
                    .collect(),
                (1, 16, frames),
                &device,
            )?;
            let decode_eager = || model.decoder.decode(&codes);
            let eager_a = per_call_ms(&device, 6, |_| decode_eager())?;
            let guard = cuda.enable_cuda_graph_htod_cache();
            let warmup = decode_eager()?;
            let decode_record =
                candle_cuda_graph::Replay::record(cuda, &[&codes], &warmup, vec![], |inputs| {
                    model
                        .decoder
                        .decode(&inputs[0])
                        .map_err(|error| candle_core::Error::Msg(error.to_string()))
                });
            drop(guard);
            let decode_replay = match decode_record {
                Ok(replay) => replay,
                Err(error) => {
                    println!(
                        "{}",
                        serde_json::json!({"probe": "codec_decode", "frames": frames,
                            "captured": false, "error": error.to_string()})
                    );
                    continue;
                }
            };
            let expected = warmup
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let replayed = decode_replay
                .launch(&[&codes])?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(replayed, expected, "decode replay differs from eager");
            let replay_a = per_call_ms(&device, 6, |_| {
                decode_replay.launch(&[&codes]).map_err(anyhow::Error::from)
            })?;
            let eager_b = per_call_ms(&device, 6, |_| decode_eager())?;
            let replay_b = per_call_ms(&device, 6, |_| {
                decode_replay.launch(&[&codes]).map_err(anyhow::Error::from)
            })?;
            println!(
                "{}",
                serde_json::json!({"probe": "codec_decode", "frames": frames,
                    "captured": true, "bitwise_equal": true,
                    "eager_ms": [eager_a, eager_b], "replay_ms": [replay_a, replay_b]})
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires explicit local model, prompt, text and output paths"]
    fn probe_voice_clone_prefill() -> Result<()> {
        use anyhow::Context;
        let model_path = std::env::var("QWEN_PREFILL_MODEL").context("QWEN_PREFILL_MODEL")?;
        anyhow::ensure!(Path::new(&model_path).is_dir(), "local model required");
        let prompt_path = std::env::var("QWEN_PREFILL_PROMPT").context("QWEN_PREFILL_PROMPT")?;
        let output = std::env::var("QWEN_PREFILL_OUTPUT").context("QWEN_PREFILL_OUTPUT")?;
        let text = std::env::var("QWEN_PREFILL_TEXT").context("QWEN_PREFILL_TEXT")?;
        let selected = std::env::var("QWEN_PREFILL_DEVICE").context("QWEN_PREFILL_DEVICE")?;
        let model = Qwen3TTS::from_pretrained(&model_path, parse_device(&selected)?)?;
        let prompt = VoiceClonePrompt::load(Path::new(&prompt_path), model.device())?;
        let input_ids = model.text_tokenizer.encode(&text)?;
        let (_, offset, last_hidden, logits, _) =
            model.voice_clone_prefill(&input_ids, &prompt, Language::Chinese, 128)?;
        let mut prefix_cache = model.talker.new_kv_caches(256);
        let (prefix, _) = model.talker.prefill_voice_clone(
            &input_ids,
            &prompt.speaker_embedding.to_dtype(model.compute_dtype)?,
            Language::Chinese,
            true,
            &mut prefix_cache,
        )?;
        let ref_embeddings = model
            .sum_ref_codec_embeddings(prompt.ref_codes.as_ref().context("ICL codes required")?)?;
        let values = |tensor: &Tensor| -> Result<Vec<f32>> {
            Ok(tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
        };
        std::fs::write(
            output,
            serde_json::to_vec(&serde_json::json!({
                "offset": offset, "prefix_shape": prefix.dims(),
                "prefix_hidden": values(&prefix)?, "last_hidden": values(&last_hidden)?,
                "logits": values(&logits)?, "reference_embeddings": values(&ref_embeddings)?,
                "text_ids": input_ids, "ref_text_ids": prompt.ref_text_ids,
            }))?,
        )?;
        Ok(())
    }

    #[test]
    fn test_synthesis_options_default() {
        let options = SynthesisOptions::default();
        assert_eq!(options.max_length, 2048);
        assert!((options.temperature - 0.9).abs() < 1e-6);
        assert_eq!(options.top_k, 50);
        assert!((options.top_p - 0.9).abs() < 1e-6);
        assert!((options.repetition_penalty - 1.05).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(CODEC_EOS_TOKEN_ID));
        assert_eq!(options.chunk_frames, 10);
        assert_eq!(options.min_new_tokens, 2);
    }

    #[test]
    fn test_synthesis_options_custom() {
        let options = SynthesisOptions {
            max_length: 512,
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
            repetition_penalty: 1.2,
            eos_token_id: Some(CODEC_EOS_TOKEN_ID),
            chunk_frames: 5,
            min_new_tokens: 0,
            seed: Some(42),
        };
        assert_eq!(options.max_length, 512);
        assert!((options.temperature - 0.5).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(CODEC_EOS_TOKEN_ID));
        assert_eq!(options.chunk_frames, 5);
    }

    #[test]
    fn test_synthesis_options_clone() {
        let options = SynthesisOptions::default();
        let cloned = options.clone();
        assert_eq!(cloned.max_length, options.max_length);
        assert_eq!(cloned.top_k, options.top_k);
    }

    #[test]
    fn test_synthesis_options_debug() {
        let options = SynthesisOptions::default();
        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("max_length"));
        assert!(debug_str.contains("2048"));
    }

    #[test]
    fn test_auto_device() {
        // Should always succeed on CPU
        let device = auto_device().unwrap();
        // Just verify it returns a valid device
        assert!(
            matches!(device, Device::Cpu)
                || matches!(device, Device::Cuda(_))
                || matches!(device, Device::Metal(_))
        );
    }

    #[test]
    fn test_audio_buffer_reexport() {
        // Verify re-exports work
        let buffer = AudioBuffer::new(vec![0.0f32; 100], 24000);
        assert_eq!(buffer.sample_rate, 24000);
    }

    #[test]
    fn test_config_reexport() {
        // Verify config re-export works
        let config = Qwen3TTSConfig::default();
        assert_eq!(config.model_type, "qwen3_tts");
    }

    #[test]
    fn test_codes_to_tensor_empty() {
        let device = Device::Cpu;
        let codes: Vec<Vec<u32>> = vec![];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 0]);
    }

    #[test]
    fn test_codes_to_tensor_single_frame() {
        let device = Device::Cpu;
        let codes = vec![vec![0u32; 16]];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 1]);
    }

    #[test]
    fn test_codes_to_tensor_layout() {
        let device = Device::Cpu;
        // 2 frames, each with 16 codebooks
        let codes = vec![
            (0..16).map(|i| i as u32).collect::<Vec<_>>(), // frame 0
            (100..116).map(|i| i as u32).collect::<Vec<_>>(), // frame 1
        ];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 2]);

        // Verify layout: tensor[0, q, frame] = codes[frame][q]
        let vals: Vec<i64> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        // q=0: [frame0_q0, frame1_q0] = [0, 100]
        assert_eq!(vals[0], 0);
        assert_eq!(vals[1], 100);
        // q=1: [frame0_q1, frame1_q1] = [1, 101]
        assert_eq!(vals[2], 1);
        assert_eq!(vals[3], 101);
    }

    #[test]
    fn test_parse_device_cpu() {
        let device = parse_device("cpu").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_parse_device_auto() {
        let device = parse_device("auto").unwrap();
        // Should succeed regardless of hardware
        assert!(
            matches!(device, Device::Cpu)
                || matches!(device, Device::Cuda(_))
                || matches!(device, Device::Metal(_))
        );
    }

    #[test]
    fn test_parse_device_unknown() {
        let result = parse_device("tpu");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_device_case_insensitive() {
        let device = parse_device("CPU").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_device_info() {
        assert_eq!(device_info(&Device::Cpu), "CPU");
    }

    #[test]
    fn test_compute_dtype_for_device() {
        let dtype = compute_dtype_for_device(&Device::Cpu);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_update_penalty_mask() {
        let device = Device::Cpu;
        let vocab_size = 3072;
        let mut mask = Tensor::zeros((1, vocab_size), DType::F32, &device).unwrap();

        Qwen3TTS::update_penalty_mask(&mut mask, 42, vocab_size).unwrap();

        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals[42], 1.0);
        // Neighboring positions should be untouched
        assert_eq!(vals[41], 0.0);
        assert_eq!(vals[43], 0.0);
    }

    #[test]
    fn test_update_penalty_mask_out_of_range() {
        let device = Device::Cpu;
        let vocab_size = 3072;
        let mut mask = Tensor::zeros((1, vocab_size), DType::F32, &device).unwrap();

        // Token beyond vocab_size should be a no-op (no panic)
        Qwen3TTS::update_penalty_mask(&mut mask, 9999, vocab_size).unwrap();

        let sum: f32 = mask.sum_all().unwrap().to_scalar().unwrap();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_suppression_mask_deterministic() {
        let device = Device::Cpu;
        let vocab = codec_tokens::CODEC_VOCAB_SIZE;
        let mask1 = generation::build_suppression_mask(vocab, CODEC_EOS_TOKEN_ID, &device).unwrap();
        let mask2 = generation::build_suppression_mask(vocab, CODEC_EOS_TOKEN_ID, &device).unwrap();

        // Apply both masks to uniform logits and verify identical output
        let logits = Tensor::ones((1, vocab), DType::F32, &device).unwrap();
        let out1 = generation::apply_token_suppression_with_mask(&logits, &mask1).unwrap();
        let out2 = generation::apply_token_suppression_with_mask(&logits, &mask2).unwrap();
        let v1: Vec<f32> = out1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = out2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1, v2);
    }
}
