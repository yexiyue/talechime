//! Incremental audio streaming, mirroring the official
//! `CosyVoice3Model.tts`/`token2wav` orchestration (commit 074ca6dc).
//!
//! Two layers:
//!
//! * [`TokenStreamDecoder`] owns flow + HiFT and the official chunk
//!   schedule. It is token-source agnostic: fixed tokens (parity probes)
//!   or an incrementally sampling LLM both drive it.
//! * [`StreamingTts`] adds the LLM with its KV cache and the official
//!   silent-token cap, producing a pull-based stream: [`StreamingTts::next_pcm`]
//!   performs enough LLM steps to reach the next hop, then one flow + HiFT
//!   decode. Audio buffers stay bounded because nothing is produced ahead
//!   of the consumer; dropping the session cancels generation.
//!
//! Official behavior reproduced here:
//!
//! * token hop starts at 25 and doubles per emitted chunk, capped at 100
//!   (`stream_scale_factor = 2`). The official code mutates this state on
//!   the shared model object so it leaks across requests; we keep it
//!   per-session, matching a freshly constructed model.
//! * the first chunk's hop is padded so that
//!   `prompt_tokens + first_hop` lands on a 25-token boundary.
//! * a chunk is emitted once `tokens - offset >= hop + pre_lookahead(3)`;
//!   the chunk solve always includes the 3 pre-lookahead tokens as context
//!   (`finalize = false`), so each chunk's mel covers exactly its hop.
//! * per chunk the flow re-solves the full prompt+token prefix with the
//!   static-chunk attention mask (`streaming = true`), the mel cache
//!   accumulates the whole mel so far, and HiFT re-decodes the whole
//!   accumulated mel with `finalize = false`, emitting only the tail
//!   beyond `speech_offset`. This O(n²) recompute is intentional in the
//!   official implementation; do not "optimize" it into incremental
//!   caches without re-validating seam continuity.
//! * silent/breath FSQ tokens are capped: after five of them without an
//!   intervening speech token, further silent tokens are dropped.
//! * `speed != 1.0` is only available on the non-streaming path; this
//!   module does not expose it (official asserts the same).
//! * hitting the LLM length limit ends the stream with
//!   [`StreamChunk::hit_token_limit`] set — never as plain success.

use candle_core::{DType, Result, Tensor};
use std::time::Instant;

use crate::config::SamplingConfig;
use crate::flow::CausalMaskedDiffWithDiT;
use crate::llm::{CosyVoice3Lm, IncrementalState};
use crate::vocoder::CausalHiFtGenerator;

/// Official `token_hop_len` — "must match training static_chunk_size".
pub const TOKEN_HOP_LEN: usize = 25;
/// Official `token_max_hop_len = 4 * token_hop_len`.
pub const TOKEN_MAX_HOP_LEN: usize = 100;
/// Official `stream_scale_factor`.
pub const STREAM_SCALE_FACTOR: usize = 2;
/// Official `max_silent_token_num` in `llm_job`.
pub const MAX_SILENT_TOKENS: u32 = 5;
/// Official CV3 FSQ silent/breath token list.
pub const SILENT_TOKENS: [u32; 11] = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323];

/// Wall-clock stage attribution for one streaming session, enabled by
/// `CV3_STREAM_PROFILE=1`. The pipeline is strictly sequential (LLM step →
/// flow re-solve → HiFT re-decode), so per-stage wall time attributes
/// without overlap. Zero cost when disabled: no `Instant` is taken and
/// nothing is printed. Diagnostics only — never part of the timing contract.
#[derive(Debug, Default)]
pub(crate) struct StageProfile {
    enabled: bool,
    /// `StreamingTts` construction: LLM prefill of text+speech prompt.
    prefill_ms: f64,
    /// LLM decode steps (one token each).
    llm_ms: f64,
    /// Per-chunk flow solves (whole prefix each time).
    flow_ms: f64,
    /// Per-chunk HiFT decodes (whole accumulated mel each time) including
    /// the PCM copy to CPU.
    hift_ms: f64,
    /// Everything else inside `next_pcm` (schedule, cat, narrow, sampling).
    other_ms: f64,
    chunks: usize,
    tokens: usize,
}

impl StageProfile {
    fn from_env() -> Self {
        Self {
            enabled: std::env::var("CV3_STREAM_PROFILE").is_ok_and(|v| v != "0"),
            ..Default::default()
        }
    }

    #[inline]
    fn ms(start: Instant) -> f64 {
        start.elapsed().as_secs_f64() * 1e3
    }

    fn chunk_line(&self, tokens_generated: usize, samples: usize) {
        if !self.enabled {
            return;
        }
        eprintln!(
            "[cv3-stream] chunk {} tokens {tokens_generated} new_samples {samples} | prefill {:.0} llm {:.0} flow {:.0} hift {:.0} other {:.1} (cumulative ms)",
            self.chunks, self.prefill_ms, self.llm_ms, self.flow_ms, self.hift_ms, self.other_ms,
        );
    }

    fn summary_line(&self) {
        if !self.enabled {
            return;
        }
        let total = self.prefill_ms + self.llm_ms + self.flow_ms + self.hift_ms + self.other_ms;
        let pct = |v: f64| if total > 0.0 { v / total * 100.0 } else { 0.0 };
        eprintln!(
            "[cv3-stream] TOTAL {total:.0} ms | prefill {:.0} ({:.0}%) llm {:.0} ({:.0}%) flow {:.0} ({:.0}%) hift {:.0} ({:.0}%) other {:.0} ({:.0}%) | chunks {} llm_tokens {}",
            self.prefill_ms,
            pct(self.prefill_ms),
            self.llm_ms,
            pct(self.llm_ms),
            self.flow_ms,
            pct(self.flow_ms),
            self.hift_ms,
            pct(self.hift_ms),
            self.other_ms,
            pct(self.other_ms),
            self.chunks,
            self.tokens,
        );
    }
}

/// One consumable PCM block.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// f32 samples at 24 kHz, mono.
    pub samples: Vec<f32>,
    /// True for the final chunk of the stream.
    pub final_chunk: bool,
    /// The LLM hit its length limit; audio is truncated, not complete.
    pub hit_token_limit: bool,
    /// Tokens consumed so far (progress reporting).
    pub tokens_generated: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    Open,
    Draining,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Copy)]
struct ChunkPlan {
    take: usize,
    offset: usize,
    finalize: bool,
}

/// Token ownership and scheduling are independent of model execution, so
/// boundary and terminal-state behavior can be tested without model files.
struct ChunkSchedule {
    tokens: Vec<u32>,
    offset: usize,
    hop: usize,
    prompt_pad: usize,
    lookahead: usize,
    max_tokens: usize,
    state: StreamState,
}

impl ChunkSchedule {
    fn new(prompt_len: usize, lookahead: usize, ratio: usize) -> Result<Self> {
        if ratio == 0 || prompt_len >= crate::flow::flow_matching::MAX_MEL_FRAMES / ratio {
            candle_core::bail!("prompt exceeds the flow noise table capacity");
        }
        Ok(Self {
            tokens: Vec::new(),
            offset: 0,
            hop: TOKEN_HOP_LEN,
            prompt_pad: (TOKEN_HOP_LEN - prompt_len % TOKEN_HOP_LEN) % TOKEN_HOP_LEN,
            lookahead,
            max_tokens: crate::flow::flow_matching::MAX_MEL_FRAMES / ratio - prompt_len,
            state: StreamState::Open,
        })
    }

    fn next_hop(&self) -> usize {
        self.hop + if self.offset == 0 { self.prompt_pad } else { 0 }
    }

    fn extend(&mut self, tokens: &[u32]) -> Result<()> {
        if self.state != StreamState::Open {
            candle_core::bail!("cannot append tokens after source end, completion or failure");
        }
        if tokens.len() > self.max_tokens - self.tokens.len()
            || tokens
                .iter()
                .any(|&t| t >= crate::config::SPEECH_TOKEN_SIZE as u32)
        {
            candle_core::bail!("speech tokens exceed the vocabulary or flow noise capacity");
        }
        self.tokens.extend_from_slice(tokens);
        Ok(())
    }

    fn next(&mut self, source_done: bool) -> Result<Option<ChunkPlan>> {
        match self.state {
            StreamState::Complete => return Ok(None),
            StreamState::Failed => candle_core::bail!("stream has failed; create a new session"),
            _ => {}
        }
        if source_done {
            self.state = StreamState::Draining;
        }
        if self.tokens.len() - self.offset >= self.next_hop() + self.lookahead {
            return Ok(Some(ChunkPlan {
                take: self.offset + self.next_hop() + self.lookahead,
                offset: self.offset,
                finalize: false,
            }));
        }
        if self.state == StreamState::Draining {
            if self.tokens.is_empty() {
                self.state = StreamState::Failed;
                candle_core::bail!("token source ended without any speech tokens");
            }
            return Ok(Some(ChunkPlan {
                take: self.tokens.len(),
                offset: self.offset,
                finalize: true,
            }));
        }
        Ok(None)
    }

    fn commit(&mut self, plan: ChunkPlan) {
        if plan.finalize {
            self.state = StreamState::Complete;
        } else {
            self.offset += self.next_hop();
            self.hop = (self.hop * STREAM_SCALE_FACTOR).min(TOKEN_MAX_HOP_LEN);
        }
    }
}

// Owned sessions remain available to probes; backend sessions borrow the
// already-loaded components so weights and KV state never need copying.
enum ModelOwner<'a, T> {
    Owned(Box<T>),
    Borrowed(&'a mut T),
}
impl<T> std::ops::Deref for ModelOwner<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Self::Owned(value) => value,
            Self::Borrowed(value) => value,
        }
    }
}
impl<T> std::ops::DerefMut for ModelOwner<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        match self {
            Self::Owned(value) => value,
            Self::Borrowed(value) => value,
        }
    }
}

/// Flow + HiFT side of the official streaming schedule. Output is pulled
/// one block at a time; cumulative tokens/mel remain bounded by the fixed
/// 300-second flow noise table, not by the PCM consumer's queue size.
pub struct TokenStreamDecoder<'a> {
    flow: ModelOwner<'a, CausalMaskedDiffWithDiT>,
    hift: ModelOwner<'a, CausalHiFtGenerator>,
    flow_prompt_token: Tensor,
    prompt_feat: Tensor,
    embedding: Tensor,
    schedule: ChunkSchedule,
    mel_cache: Option<Tensor>,
    last_new_mel: Option<Tensor>,
    speech_offset: usize,
    device: candle_core::Device,
    pub(crate) profile: StageProfile,
}

impl<'a> TokenStreamDecoder<'a> {
    pub fn new(
        flow: CausalMaskedDiffWithDiT,
        hift: CausalHiFtGenerator,
        flow_prompt_token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
    ) -> Result<Self> {
        Self::from_models(
            ModelOwner::Owned(Box::new(flow)),
            ModelOwner::Owned(Box::new(hift)),
            flow_prompt_token,
            prompt_feat,
            embedding,
        )
    }

    fn from_models(
        flow: ModelOwner<'a, CausalMaskedDiffWithDiT>,
        hift: ModelOwner<'a, CausalHiFtGenerator>,
        flow_prompt_token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
    ) -> Result<Self> {
        let device = flow.device().clone();
        let pre_lookahead = flow.pre_lookahead_len();
        let plen = flow_prompt_token.dim(1)?;
        if flow_prompt_token.dims() != [1, plen]
            || flow_prompt_token.dtype() != DType::U32
            || prompt_feat.dims() != [1, plen * flow.token_mel_ratio(), 80]
            || embedding.dims() != [1, 192]
        {
            candle_core::bail!("invalid streaming prompt token, mel or speaker shape/dtype");
        }
        let schedule = ChunkSchedule::new(plen, pre_lookahead, flow.token_mel_ratio())?;
        Ok(Self {
            flow,
            hift,
            flow_prompt_token: flow_prompt_token.to_device(&device)?,
            prompt_feat: prompt_feat.to_device(&device)?,
            embedding: embedding.to_device(&device)?,
            schedule,
            mel_cache: None,
            speech_offset: 0,
            last_new_mel: None,
            device,
            profile: StageProfile::from_env(),
        })
    }

    /// Append validated speech tokens before source end. Rejected batches
    /// do not partially modify the buffer.
    pub fn extend_tokens(&mut self, more: &[u32]) -> Result<()> {
        self.schedule.extend(more)
    }

    pub fn token_count(&self) -> usize {
        self.schedule.tokens.len()
    }

    /// Mel slice produced by the most recent successful emit (diagnostics).
    pub fn last_new_mel(&self) -> Option<&Tensor> {
        self.last_new_mel.as_ref()
    }

    /// `source_done` latches source exhaustion. `None` means either more
    /// tokens are needed or the final block has already been returned.
    /// A decoding error permanently fails this session.
    pub fn next_pcm(&mut self, source_done: bool) -> Result<Option<StreamChunk>> {
        let Some(plan) = self.schedule.next(source_done)? else {
            return Ok(None);
        };
        match self.emit(plan) {
            Ok(chunk) => Ok(Some(chunk)),
            Err(error) => {
                self.schedule.state = StreamState::Failed;
                self.last_new_mel = None;
                Err(error)
            }
        }
    }

    /// Official `token2wav`: re-solve flow on the prefix, re-decode HiFT on
    /// the whole accumulated mel, emit the tail past `speech_offset`.
    fn emit(&mut self, plan: ChunkPlan) -> Result<StreamChunk> {
        let token_t = Tensor::from_slice(
            &self.schedule.tokens[..plan.take],
            (1, plan.take),
            &self.device,
        )?;

        let t_flow = Instant::now();
        let mel = self.flow.inference(
            &token_t,
            &self.flow_prompt_token,
            &self.prompt_feat,
            &self.embedding,
            self.flow.n_timesteps(),
            !plan.finalize,
            plan.finalize,
        )?;
        let from = plan.offset * self.flow.token_mel_ratio();
        let mel = mel.narrow(2, from, mel.dim(2)? - from)?;
        let mel_all = match &self.mel_cache {
            Some(c) => Tensor::cat(&[c, &mel], 2)?,
            None => mel.clone(),
        };
        if self.profile.enabled {
            // Sync first: flow leaves its GPU work queued; without this the
            // drain lands in the next stage's CPU transfer and flow is
            // under-counted (observed ~0.6s/chunk absorbed into hift f0).
            self.device.synchronize().ok();
            self.profile.flow_ms += StageProfile::ms(t_flow);
        }
        let t_hift = Instant::now();
        let pcm = self.hift.inference(&mel_all, plan.finalize)?;
        let pcm = pcm.to_dtype(DType::F32)?.flatten_all()?;
        let total = pcm.dim(0)?;
        let new_len = total.checked_sub(self.speech_offset).ok_or_else(|| {
            candle_core::Error::Msg("HiFT output shrank behind the emitted sample offset".into())
        })?;
        let samples = pcm
            .narrow(0, self.speech_offset, new_len)?
            .to_vec1::<f32>()?;
        if self.profile.enabled {
            self.profile.hift_ms += StageProfile::ms(t_hift);
            self.profile.chunks += 1;
            self.profile
                .chunk_line(self.schedule.tokens.len(), samples.len());
        }
        // Commit only after every fallible synthesis/copy operation succeeded.
        self.last_new_mel = Some(mel);
        self.mel_cache = Some(mel_all);
        self.speech_offset = total;
        self.schedule.commit(plan);
        Ok(StreamChunk {
            samples,
            final_chunk: plan.finalize,
            hit_token_limit: false,
            tokens_generated: self.schedule.tokens.len(),
        })
    }
}

/// Full streaming session: LLM token generation + [`TokenStreamDecoder`].
pub struct StreamingTts<'a> {
    llm: ModelOwner<'a, CosyVoice3Lm>,
    llm_state: IncrementalState,
    decoder: TokenStreamDecoder<'a>,
    silent_run: u32,
    llm_done: bool,
    hit_token_limit: bool,
    failed: bool,
}

impl<'a> StreamingTts<'a> {
    /// Start a streaming request from pre-encoded inputs. The prompt-side
    /// tensors must come from the same preparation as non-streaming
    /// synthesis (`pipeline::extract_prompt_features` + text tokenizer).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        llm: CosyVoice3Lm,
        flow: CausalMaskedDiffWithDiT,
        hift: CausalHiFtGenerator,
        text_token: &Tensor,
        prompt_text_token: &Tensor,
        llm_prompt_speech_token: &Tensor,
        flow_prompt_token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        sampling: &SamplingConfig,
    ) -> Result<Self> {
        Self::from_models(
            ModelOwner::Owned(Box::new(llm)),
            ModelOwner::Owned(Box::new(flow)),
            ModelOwner::Owned(Box::new(hift)),
            text_token,
            prompt_text_token,
            llm_prompt_speech_token,
            flow_prompt_token,
            prompt_feat,
            embedding,
            sampling,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn borrowed(
        llm: &'a mut CosyVoice3Lm,
        flow: &'a mut CausalMaskedDiffWithDiT,
        hift: &'a mut CausalHiFtGenerator,
        text_token: &Tensor,
        prompt_text_token: &Tensor,
        llm_prompt_speech_token: &Tensor,
        flow_prompt_token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        sampling: &SamplingConfig,
    ) -> Result<Self> {
        Self::from_models(
            ModelOwner::Borrowed(llm),
            ModelOwner::Borrowed(flow),
            ModelOwner::Borrowed(hift),
            text_token,
            prompt_text_token,
            llm_prompt_speech_token,
            flow_prompt_token,
            prompt_feat,
            embedding,
            sampling,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_models(
        mut llm: ModelOwner<'a, CosyVoice3Lm>,
        flow: ModelOwner<'a, CausalMaskedDiffWithDiT>,
        hift: ModelOwner<'a, CausalHiFtGenerator>,
        text_token: &Tensor,
        prompt_text_token: &Tensor,
        llm_prompt_speech_token: &Tensor,
        flow_prompt_token: &Tensor,
        prompt_feat: &Tensor,
        embedding: &Tensor,
        sampling: &SamplingConfig,
    ) -> Result<Self> {
        let device = flow.device().clone();
        let mut decoder =
            TokenStreamDecoder::from_models(flow, hift, flow_prompt_token, prompt_feat, embedding)?;
        let t_prefill = Instant::now();
        let llm_state = llm.start_incremental(
            &text_token.to_device(&device)?,
            &prompt_text_token.to_device(&device)?,
            &llm_prompt_speech_token.to_device(&device)?,
            sampling,
        )?;
        if decoder.profile.enabled {
            decoder.profile.prefill_ms += StageProfile::ms(t_prefill);
        }
        Ok(Self {
            llm,
            llm_state,
            decoder,
            silent_run: 0,
            llm_done: false,
            hit_token_limit: false,
            failed: false,
        })
    }

    /// Advance the LLM by one token, applying the official silent-token
    /// cap. Completion and limits are recorded once in the session.
    fn llm_step(&mut self) -> Result<()> {
        if self.llm_done {
            return Ok(());
        }
        let Some(t) = self.llm.next_token(&mut self.llm_state)? else {
            self.llm_done = true;
            self.hit_token_limit = self.llm_state.hit_limit();
            return Ok(());
        };
        if self.llm_state.is_done() {
            // The returned token consumed the length budget.
            self.llm_done = true;
            self.hit_token_limit = self.llm_state.hit_limit();
        }
        if SILENT_TOKENS.contains(&t) {
            self.silent_run += 1;
            if self.silent_run > MAX_SILENT_TOKENS {
                return Ok(());
            }
        } else {
            self.silent_run = 0;
        }
        self.decoder.extend_tokens(&[t])?;
        Ok(())
    }

    /// Produce the next PCM chunk. One call = at most one flow+HiFT decode;
    /// LLM steps run only until the next hop is reachable. `None` follows
    /// the one final chunk; errors are terminal and cannot be retried.
    pub fn next_pcm(&mut self) -> Result<Option<StreamChunk>> {
        if self.failed {
            candle_core::bail!("stream has failed; create a new session");
        }
        let result = self.next_pcm_inner();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn next_pcm_inner(&mut self) -> Result<Option<StreamChunk>> {
        let t_call = Instant::now();
        let before = self
            .decoder
            .profile
            .enabled
            .then_some((self.decoder.profile.flow_ms, self.decoder.profile.hift_ms));
        let mut llm_ms = 0f64;
        let result = loop {
            if let Some(mut chunk) = self.decoder.next_pcm(self.llm_done)? {
                chunk.hit_token_limit = chunk.final_chunk && self.hit_token_limit;
                break Ok(Some(chunk));
            }
            if self.llm_done {
                self.decoder.profile.summary_line();
                break Ok(None);
            }
            let t_llm = Instant::now();
            let step = self.llm_step();
            if self.decoder.profile.enabled {
                let d = StageProfile::ms(t_llm);
                llm_ms += d;
                self.decoder.profile.llm_ms += d;
                self.decoder.profile.tokens += 1;
            }
            step?;
        };
        if let Some((f0, h0)) = before {
            let p = &mut self.decoder.profile;
            let wall = StageProfile::ms(t_call);
            p.other_ms += wall - llm_ms - ((p.flow_ms - f0) + (p.hift_ms - h0));
        }
        result
    }

    /// Whether the LLM side has finished producing tokens.
    pub fn llm_done(&self) -> bool {
        self.llm_done
    }

    /// Tokens accepted so far (after the silent-token cap).
    pub fn token_count(&self) -> usize {
        self.decoder.token_count()
    }
}

impl Drop for StreamingTts<'_> {
    fn drop(&mut self) {
        self.llm.clear_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_arrival_batching_does_not_change_prefixes_or_lookahead_ownership() -> Result<()> {
        for prompt in 0..=25 {
            for count in [1, 27, 28, 29, 77, 78, 79, 177, 178, 179, 283] {
                let tokens: Vec<u32> = (0..count).map(|i| i as u32 + 100).collect();
                let plans = |batch: usize| -> Result<Vec<(Vec<u32>, usize, bool)>> {
                    let mut schedule = ChunkSchedule::new(prompt, 3, 2)?;
                    let mut plans = Vec::new();
                    for more in tokens.chunks(batch) {
                        schedule.extend(more)?;
                        while let Some(plan) = schedule.next(false)? {
                            plans.push((
                                schedule.tokens[..plan.take].to_vec(),
                                plan.offset,
                                plan.finalize,
                            ));
                            schedule.commit(plan);
                        }
                    }
                    while let Some(plan) = schedule.next(true)? {
                        plans.push((
                            schedule.tokens[..plan.take].to_vec(),
                            plan.offset,
                            plan.finalize,
                        ));
                        schedule.commit(plan);
                    }
                    Ok(plans)
                };
                let single = plans(1)?;
                assert_eq!(single, plans(17)?, "prompt={prompt}, count={count}");
                assert_eq!(single, plans(count)?, "prompt={prompt}, count={count}");
                let mut committed = 0;
                for (prefix, offset, finalize) in single {
                    assert_eq!(offset, committed);
                    assert_eq!(prefix, tokens[..prefix.len()]);
                    committed = prefix.len() - if finalize { 0 } else { 3 };
                }
                assert_eq!(committed, count);
            }
        }
        Ok(())
    }

    #[test]
    fn chunk_boundaries_cover_every_token_once_and_finish_once() -> Result<()> {
        for prompt in [0, 1, 25, 87, 100] {
            for n in [
                1, 2, 3, 24, 25, 26, 27, 28, 29, 38, 40, 41, 42, 75, 100, 250,
            ] {
                let mut schedule = ChunkSchedule::new(prompt, 3, 2)?;
                schedule.extend(&vec![7; n])?;
                let mut consumed = 0;
                while let Some(plan) = schedule.next(true)? {
                    assert_eq!(plan.offset, consumed);
                    let count = plan.take - plan.offset - if plan.finalize { 0 } else { 3 };
                    assert!(count > 0);
                    consumed += count;
                    schedule.commit(plan);
                }
                assert_eq!(consumed, n, "prompt={prompt}, tokens={n}");
                assert!(schedule.next(false)?.is_none());
                assert!(schedule.extend(&[7]).is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn source_end_stays_latched_while_pending_hops_drain() -> Result<()> {
        let mut schedule = ChunkSchedule::new(87, 3, 2)?;
        schedule.extend(&[7; 200])?;
        let first = schedule.next(true)?.unwrap();
        assert_eq!(first.take, 41);
        assert!(!first.finalize);
        schedule.commit(first);
        assert!(schedule.extend(&[8]).is_err());
        let second = schedule.next(false)?.unwrap();
        assert_eq!(second.take - second.offset - 3, 50);
        schedule.commit(second);
        let third = schedule.next(false)?.unwrap();
        assert_eq!(third.take - third.offset - 3, 100);
        schedule.commit(third);
        let final_plan = schedule.next(false)?.unwrap();
        assert!(final_plan.finalize);
        assert_eq!(final_plan.take, 200);
        Ok(())
    }

    #[test]
    fn incomplete_input_waits_then_finalizes_short_tail() -> Result<()> {
        let mut schedule = ChunkSchedule::new(0, 3, 2)?;
        schedule.extend(&[7; 27])?;
        assert!(schedule.next(false)?.is_none());
        let final_plan = schedule.next(true)?.unwrap();
        assert!(final_plan.finalize);
        assert_eq!(final_plan.take, 27);
        Ok(())
    }

    #[test]
    fn token_validation_is_atomic_and_capacity_is_bounded() -> Result<()> {
        let mut schedule = ChunkSchedule::new(87, 3, 2)?;
        schedule.extend(&[7])?;
        assert!(
            schedule
                .extend(&[8, crate::config::SPEECH_TOKEN_SIZE as u32])
                .is_err()
        );
        assert_eq!(schedule.tokens, [7]);
        assert!(schedule.extend(&vec![7; schedule.max_tokens]).is_err());
        assert_eq!(schedule.tokens, [7]);
        assert!(ChunkSchedule::new(7_500, 3, 2).is_err());
        Ok(())
    }

    #[test]
    fn empty_source_and_failed_stream_never_report_completion() -> Result<()> {
        let mut schedule = ChunkSchedule::new(0, 3, 2)?;
        assert!(schedule.next(true).is_err());
        assert_eq!(schedule.state, StreamState::Failed);
        assert!(schedule.next(false).is_err());
        assert!(schedule.extend(&[7]).is_err());
        Ok(())
    }
}
