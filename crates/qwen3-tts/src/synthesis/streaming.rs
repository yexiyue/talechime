//! Incremental token generation and PCM decoding.
use super::*;

// Keep bounded codec history so each chunk does not restart from silence.
const DECODER_LEFT_CONTEXT_FRAMES: usize = 25;

/// Streaming synthesis session.
///
/// Yields audio chunks as they are generated. Use with
/// [`Qwen3TTS::synthesize_streaming`].
pub struct StreamingSession<'a> {
    model: &'a Qwen3TTS,
    config: generation::GenerationConfig,
    sampling_ctx: generation::SamplingContext,
    kv_caches: Vec<AnyKVCache>,
    offset: usize,
    last_hidden: Tensor,
    current_token_tensor: Option<Tensor>,
    frames_generated: usize,
    frame_buffer: Vec<Tensor>,
    chunk_frames: usize,
    done: bool,
    // Trailing text state for residual VQ + text fusion
    trailing_text_hidden: Tensor,
    trailing_text_len: usize,
    tts_pad_embed: Tensor,
    // GPU-side repetition penalty mask [1, vocab] — updated incrementally
    penalty_mask: Tensor,
    token_count: usize,
    // Pre-built suppression mask (reused every frame)
    suppression_mask: generation::SuppressionMask,
    // Pre-allocated code predictor KV caches (reused + reset each frame)
    cp_kv_caches: Vec<AnyKVCache>,
    decoder_context: Option<Tensor>,
}

impl<'a> StreamingSession<'a> {
    pub(super) fn new(
        model: &'a Qwen3TTS,
        input_ids: &[u32],
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        Self::new_with_style(model, input_ids, &[], speaker, language, options)
    }

    pub(super) fn new_with_style(
        model: &'a Qwen3TTS,
        input_ids: &[u32],
        instruct_ids: &[u32],
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        let sampling_ctx = generation::SamplingContext::new(options.seed);
        let config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            model.build_trailing_text(input_ids)?;

        let mut kv_caches = model
            .talker
            .new_kv_caches(config.max_new_tokens + 256 + instruct_ids.len());
        let prefill_result = model.talker.prefill_custom_voice_with_instruct(
            input_ids,
            instruct_ids,
            speaker,
            language,
            &mut kv_caches,
        )?;

        Self::from_prefill(
            model,
            config,
            sampling_ctx,
            kv_caches,
            prefill_result,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            options.chunk_frames,
        )
    }

    /// Create a streaming session using voice design (text-described voice).
    ///
    /// Uses `prefill_voice_design` instead of `prefill_custom_voice` to condition
    /// on a natural language voice description rather than a predefined speaker.
    pub(super) fn new_voice_design(
        model: &'a Qwen3TTS,
        input_ids: &[u32],
        instruct_ids: &[u32],
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        let sampling_ctx = generation::SamplingContext::new(options.seed);
        let config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            model.build_trailing_text(input_ids)?;

        let mut kv_caches = model.talker.new_kv_caches(config.max_new_tokens + 256);
        let prefill_result =
            model
                .talker
                .prefill_voice_design(input_ids, instruct_ids, language, &mut kv_caches)?;

        Self::from_prefill(
            model,
            config,
            sampling_ctx,
            kv_caches,
            prefill_result,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            options.chunk_frames,
        )
    }

    pub(super) fn new_voice_clone(
        model: &'a Qwen3TTS,
        input_ids: &[u32],
        prompt: &VoiceClonePrompt,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        let config = options.to_gen_config();
        let (kv_caches, offset, hidden, logits, trailing) =
            model.voice_clone_prefill(input_ids, prompt, language, config.max_new_tokens)?;
        let trailing_len = trailing.dim(1)?;
        let last_hidden = hidden.clone();
        let mut session = Self::from_prefill(
            model,
            config,
            generation::SamplingContext::new(options.seed),
            kv_caches,
            (hidden, logits),
            trailing,
            trailing_len,
            model.talker.get_tts_pad_embed()?,
            options.chunk_frames,
        )?;
        session.offset = offset;
        session.last_hidden = last_hidden;
        session.decoder_context = prompt.ref_codes.clone();
        Ok(session)
    }

    /// Shared post-prefill constructor.
    ///
    /// Extracts `last_hidden` from the prefill result, builds the suppression and
    /// penalty masks, samples the first semantic token, and assembles the session.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_prefill(
        model: &'a Qwen3TTS,
        config: generation::GenerationConfig,
        mut sampling_ctx: generation::SamplingContext,
        kv_caches: Vec<AnyKVCache>,
        prefill_result: (Tensor, Tensor),
        trailing_text_hidden: Tensor,
        trailing_text_len: usize,
        tts_pad_embed: Tensor,
        chunk_frames: usize,
    ) -> Result<Self> {
        anyhow::ensure!(chunk_frames > 0, "chunk_frames must be positive");
        let (hidden, logits) = prefill_result;
        let prefill_len = hidden.dim(1)?;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        // Build suppression mask once for reuse across all frames
        let suppression_mask = generation::build_suppression_mask(
            codec_tokens::CODEC_VOCAB_SIZE,
            CODEC_EOS_TOKEN_ID,
            &model.device,
        )?;

        // Sample first token with full penalty pipeline
        let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;
        let mut penalty_mask = Tensor::zeros((1, vocab_size), DType::F32, &model.device)?;
        let logits_2d = logits.squeeze(1)?;
        let logits_2d = model.apply_generation_penalties_gpu(
            &logits_2d,
            &penalty_mask,
            &config,
            0,
            Some(&suppression_mask),
        )?;
        let first_token = generation::sample(&logits_2d, &config, &mut sampling_ctx)?;
        let first_token_id: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
        Qwen3TTS::update_penalty_mask(&mut penalty_mask, first_token_id, vocab_size)?;

        let done = config.eos_token_id == Some(first_token_id);
        let cp_kv_caches = model.code_predictor.new_kv_caches();

        Ok(Self {
            model,
            config,
            sampling_ctx,
            kv_caches,
            offset: prefill_len,
            last_hidden,
            current_token_tensor: if done { None } else { Some(first_token) },
            frames_generated: 0,
            frame_buffer: Vec::new(),
            chunk_frames,
            done,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            penalty_mask,
            token_count: 1,
            suppression_mask,
            cp_kv_caches,
            decoder_context: None,
        })
    }

    /// Generate the next chunk of audio.
    ///
    /// Returns `Some(AudioBuffer)` for each chunk, or `None` when generation is complete.
    pub fn next_chunk(&mut self) -> Result<Option<AudioBuffer>> {
        self.next_chunk_with_cancel(|| false)
    }

    /// Check cancellation between generated frames and before decoding PCM.
    pub fn next_chunk_with_cancel(
        &mut self,
        cancelled: impl Fn() -> bool,
    ) -> Result<Option<AudioBuffer>> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("stream_chunk").entered();
        if cancelled() {
            return Ok(None);
        }
        if self.done {
            // Flush remaining buffer
            if !self.frame_buffer.is_empty() {
                return self.decode_buffer().map(Some);
            }
            return Ok(None);
        }

        // Generate frames until we have enough for a chunk
        while self.frame_buffer.len() < self.chunk_frames
            && self.frames_generated < self.config.max_new_tokens
        {
            if cancelled() {
                return Ok(None);
            }
            let token_tensor = match self.current_token_tensor.take() {
                Some(token) => token,
                None => {
                    self.done = true;
                    break;
                }
            };

            // Embedding lookup using GPU-resident token tensor (no CPU→GPU roundtrip)
            let semantic_embed = self
                .model
                .talker
                .get_codec_embedding_from_tensor(&token_tensor)?;

            // Generate 15 acoustic codes (stays on GPU)
            #[cfg(feature = "profiling")]
            let _cp_span =
                tracing::info_span!("code_predictor", frame = self.frames_generated).entered();
            let acoustic_codes_tensor = self.model.code_predictor.generate_acoustic_codes(
                &self.last_hidden,
                &semantic_embed,
                &mut self.cp_kv_caches,
            )?;
            #[cfg(feature = "profiling")]
            drop(_cp_span);

            let frame_idx = self.frames_generated;
            self.frames_generated += 1;
            self.frame_buffer.push(Tensor::cat(
                &[&token_tensor.reshape(1)?, &acoustic_codes_tensor],
                0,
            )?);

            // Build residual VQ sum + trailing text for next step
            let summed = self
                .model
                .code_predictor
                .get_codec_embeddings_sum_from_tensor(&semantic_embed, &acoustic_codes_tensor)?;

            let text_addition = if frame_idx < self.trailing_text_len {
                self.trailing_text_hidden
                    .i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                self.tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            // Run talker step with fused embedding
            #[cfg(feature = "profiling")]
            let _talker_span =
                tracing::info_span!("talker_step", frame = self.frames_generated).entered();
            let (h, new_logits) = self.model.talker.generate_step_with_embed(
                &step_input,
                &mut self.kv_caches,
                self.offset,
            )?;
            #[cfg(feature = "profiling")]
            drop(_talker_span);
            self.offset += 1;
            self.last_hidden = h;

            // Sample next semantic token with repetition penalty + token suppression + min_new_tokens
            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.model.apply_generation_penalties_gpu(
                &logits_2d,
                &self.penalty_mask,
                &self.config,
                self.token_count,
                Some(&self.suppression_mask),
            )?;
            let next_token_tensor =
                generation::sample(&logits_2d, &self.config, &mut self.sampling_ctx)?;
            let next_token_id: u32 = next_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
            Qwen3TTS::update_penalty_mask(
                &mut self.penalty_mask,
                next_token_id,
                codec_tokens::CODEC_VOCAB_SIZE,
            )?;
            self.token_count += 1;

            if self.config.eos_token_id == Some(next_token_id) {
                self.current_token_tensor = None;
                self.done = true;
            } else {
                self.current_token_tensor = Some(next_token_tensor);
            }
        }

        // Decode the buffered frames
        if self.frame_buffer.is_empty() {
            return Ok(None);
        }

        if cancelled() {
            return Ok(None);
        }
        self.decode_buffer().map(Some)
    }

    fn decode_buffer(&mut self) -> Result<AudioBuffer> {
        let context_len = self
            .decoder_context
            .as_ref()
            .map_or(0, |codes| codes.dims()[0]);
        let codes = decoder_codes(&self.frame_buffer, self.decoder_context.as_ref())?;
        let total_frames = codes.dim(2)?;
        let mut audio = AudioBuffer::from_tensor(self.model.decoder.decode(&codes)?, 24000)?;
        if context_len > 0 {
            let cut = context_len * audio.samples.len() / total_frames;
            audio.samples.drain(..cut);
        }
        self.decoder_context = Some(decoder_context_tail(&codes)?);
        self.frame_buffer.clear();
        Ok(audio)
    }

    /// Returns the total number of frames generated so far.
    pub fn frames_generated(&self) -> usize {
        self.frames_generated
    }

    /// Returns true if generation is complete.
    pub fn is_done(&self) -> bool {
        self.done && self.frame_buffer.is_empty()
    }
}

/// Retain recent reference/generated codes in frame-major layout for the next block.
fn decoder_context_tail(codes: &Tensor) -> Result<Tensor> {
    let frames = codes.dim(2)?;
    let keep = frames.min(DECODER_LEFT_CONTEXT_FRAMES);
    Ok(codes
        .narrow(2, frames - keep, keep)?
        .squeeze(0)?
        .transpose(0, 1)?
        .contiguous()?)
}

/// Pack device-resident frames into the decoder's `[1, 16, T]` layout.
fn decoder_codes(frames: &[Tensor], reference: Option<&Tensor>) -> Result<Tensor> {
    let generated = Tensor::stack(frames, 0)?.to_dtype(DType::I64)?;
    anyhow::ensure!(
        generated.dims2()?.1 == 16,
        "codec frames must contain 16 codes"
    );
    let combined = match reference {
        Some(reference) => Tensor::cat(&[&reference.to_dtype(DType::I64)?, &generated], 0)?,
        None => generated,
    };
    Ok(combined.transpose(0, 1)?.unsqueeze(0)?.contiguous()?)
}

impl<'a> Iterator for StreamingSession<'a> {
    type Item = Result<AudioBuffer>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(audio)) => Some(Ok(audio)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_history_advances_across_chunk_boundaries() -> Result<()> {
        let device = Device::Cpu;
        let reference = Tensor::new((0..44u32).map(|v| vec![v; 16]).collect::<Vec<_>>(), &device)?;
        let mut context = reference;
        for start in [44, 54, 64, 74] {
            let frames = (start..start + 10)
                .map(|v| Tensor::new(&[v; 16], &device))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let packed = decoder_codes(&frames, Some(&context))?;
            assert_eq!(packed.dim(2)?, context.dim(0)? + 10);
            context = decoder_context_tail(&packed)?;
            let expected: Vec<Vec<i64>> = (start + 10 - 25..start + 10)
                .map(|v| vec![i64::from(v); 16])
                .collect();
            assert_eq!(context.to_vec2::<i64>()?, expected);
        }
        let first_frames = [Tensor::new(&[0u32; 16], &device)?];
        let packed = decoder_codes(&first_frames, None)?;
        assert_eq!(decoder_context_tail(&packed)?.dims(), &[1, 16]);
        Ok(())
    }

    #[test]
    fn device_frame_packing_preserves_reference_and_codebook_order() -> Result<()> {
        let device = Device::Cpu;
        let reference = Tensor::new(&[[0u32; 16], [1u32; 16]], &device)?;
        let frames = [Tensor::from_vec(
            (2u32..18).collect::<Vec<_>>(),
            16,
            &device,
        )?];
        let actual = decoder_codes(&frames, Some(&reference))?;
        let expected = codes_to_tensor(&[vec![0; 16], vec![1; 16], (2..18).collect()], &device)?;
        assert_eq!(actual.dims(), &[1, 16, 3]);
        assert_eq!(
            actual.flatten_all()?.to_vec1::<i64>()?,
            expected.flatten_all()?.to_vec1::<i64>()?
        );
        assert_eq!(decoder_codes(&frames, None)?.dims(), &[1, 16, 1]);
        let malformed = [Tensor::new(&[0u32; 15], &device)?];
        assert!(decoder_codes(&malformed, None).is_err());
        Ok(())
    }
}
