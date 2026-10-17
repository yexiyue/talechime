//! HF-compatible generation for the mel-token GPT
//! (`indextts/gpt/transformers_generation_utils.py` +
//! `transformers_beam_search.py`, the vendored HF copy).
//!
//! Upstream calls `GPT2InferenceModel.generate` with `num_beams=3,
//! do_sample=True` (beam-sample), `repetition_penalty=10`, `top_k=30`,
//! `top_p=0.8`, `temperature=0.8`, `eos = stop_mel_token`, early stopping.
//! This module ports exactly that path: the embedding/position logic of
//! `GPT2InferenceModel.forward` (mel token embedding + fixed mel position
//! embedding for the newest token), KV cache reuse across steps, beam
//! reorder, repetition-penalty processor, temperature/top-k/top-p warpers
//! and typical sampling (`indextts/utils/typical_sampling.py`).
//!
//! Beam-sample semantics (transformers 4.52.1, the version vendored by
//! index-tts): per step the last-position logits are `log_softmax`ed, the
//! processor list [repetition penalty, (user) typical warper, temperature,
//! top-k, top-p] is applied to those log-probs, the running beam scores
//! are added, and each batch row samples `2 * num_beams` candidates
//! *without replacement* from the softmax over the flattened
//! `(num_beams * vocab)` joint distribution. EOS candidates ranked below
//! `num_beams` move the beam row into the beam hypotheses; the rest refill
//! the beams. The loop stops when every batch's hypothesis set is done
//! (heuristic: `worst >= best / gen_len^length_penalty`,
//! `length_penalty = 0` upstream), every row's last token is EOS, or
//! `max_new_tokens` is reached.
//!
//! Mel position arithmetic: the start mel token gets position 0 (its
//! embedding is built by the "first step" branch of
//! `GPT2InferenceModel.forward`, which indexes `LearnedPositionEmbeddings`
//! by the *embedding row count*, i.e. 0). Incremental steps use the literal
//! HF `kv_cache=True` formula `attention_mask.shape[1] - mel_len`, which
//! yields position i+1 for the i-th generated token — position 1 is never
//! used. This matches the vendored `transformers` path that the Python
//! oracle runs; the accel engine's `pos = len(req) - 1 - (prompt_len - 1)`
//! (i.e. i) is NOT used here because it is not the reference path.

use crate::gpt2::{Gpt2Cache, Gpt2Model};
use candle_core::{DType, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module};
use rand::Rng;
use rand::rngs::StdRng;

#[derive(Debug, Clone)]
pub struct GenerateConfig {
    pub do_sample: bool,
    pub top_p: f64,
    pub top_k: usize,
    pub temperature: f64,
    pub length_penalty: f64,
    pub num_beams: usize,
    pub repetition_penalty: f64,
    pub max_new_tokens: usize,
    pub typical_sampling: bool,
    pub typical_mass: f64,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            do_sample: true,
            top_p: 0.8,
            top_k: 30,
            temperature: 0.8,
            length_penalty: 0.0,
            num_beams: 3,
            repetition_penalty: 10.0,
            max_new_tokens: 1500,
            typical_sampling: false,
            typical_mass: 0.9,
        }
    }
}

/// Borrowed view of the UnifiedVoice parts the generation loop needs,
/// mirroring `GPT2InferenceModel`'s constructor arguments.
pub struct InferenceParts<'a> {
    pub gpt: &'a Gpt2Model,
    /// `mel_embedding`
    pub embeddings: &'a Embedding,
    /// `mel_pos_embedding` (the argument upstream confusingly names
    /// `text_pos_emb`)
    pub mel_pos_embedding: &'a Embedding,
    pub final_norm: &'a LayerNorm,
    pub mel_head: &'a Linear,
}

// ---------------------------------------------------------------------------
// Row-wise logits processors / warpers.
//
// These mirror the transformers 4.52.1 implementations that the vendored
// generation utils import; all operate on one row of scores at a time.
// ---------------------------------------------------------------------------

/// `log_softmax` in place, numerically stable.
fn log_softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let sum: f32 = row.iter().map(|v| (v - max).exp()).sum::<f32>();
    let log_sum = sum.ln();
    for v in row.iter_mut() {
        *v = (*v - max) - log_sum;
    }
}

/// Softmax as f64 weights (for sampling).
fn softmax_f64(row: &[f64]) -> Vec<f64> {
    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return vec![0.0; row.len()];
    }
    let exps: Vec<f64> = row.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|v| v / sum).collect()
}

/// `RepetitionPenaltyLogitsProcessor` (transformers 4.52.1): gather the
/// scores at `input_ids`, apply `where(score < 0, score * penalty,
/// score / penalty)` and scatter back. Duplicated ids are applied once
/// (the gather reads the original scores, matching `scatter`).
fn apply_repetition_penalty(scores: &mut [f32], input_ids: &[u32], penalty: f64) {
    let orig = scores.to_vec();
    for &id in input_ids {
        if let Some(new) = scores.get_mut(id as usize) {
            let s = orig[id as usize];
            *new = if s < 0.0 {
                s * penalty as f32
            } else {
                s / penalty as f32
            };
        }
    }
}

/// `TemperatureLogitsWarper`: `scores / temperature`.
fn apply_temperature(scores: &mut [f32], temperature: f64) {
    let t = temperature as f32;
    for s in scores.iter_mut() {
        *s /= t;
    }
}

/// `TopKLogitsWarper` (4.52.1): `self.top_k = max(top_k, min_tokens_to_keep)`,
/// then drop (to `-inf`) every score strictly below the k-th largest;
/// ties at the k-th value survive.
fn apply_top_k(scores: &mut [f32], top_k: usize, min_tokens_to_keep: usize) {
    let k = top_k.max(min_tokens_to_keep).min(scores.len());
    if k == 0 || k >= scores.len() {
        return;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let kth = sorted[k - 1];
    for s in scores.iter_mut() {
        if *s < kth {
            *s = f32::NEG_INFINITY;
        }
    }
}

/// `TopPLogitsWarper` (4.52.1): sort ascending, drop tokens whose
/// cumulative probability (of the current softmax) is `<= 1 - top_p`,
/// always keeping the `min_tokens_to_keep` largest scores.
fn apply_top_p(scores: &mut [f32], top_p: f64, min_tokens_to_keep: usize) {
    let probs = softmax_f64(&scores.iter().map(|v| *v as f64).collect::<Vec<_>>());
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut remove = vec![false; scores.len()];
    let mut cum = 0.0;
    for &i in &order {
        cum += probs[i];
        if cum <= 1.0 - top_p {
            remove[i] = true;
        }
    }
    for &i in order.iter().rev().take(min_tokens_to_keep) {
        remove[i] = false;
    }
    for (i, r) in remove.iter().enumerate() {
        if *r {
            scores[i] = f32::NEG_INFINITY;
        }
    }
}

/// `TypicalLogitsWarper` override from `indextts/utils/typical_sampling.py`.
/// Entropy is computed from the log-softmax of the incoming scores;
/// candidates are ranked by `|(-normalized) - ent|` ascending and tokens
/// whose shifted score exceeds the boundary at the first cumulative mass
/// `>= mass` are dropped.
fn apply_typical(scores: &mut [f32], mass: f64, min_tokens_to_keep: usize) {
    let n = scores.len();
    let mut normalized = scores.to_vec();
    log_softmax(&mut normalized);
    // ent = -(normalized * p).nansum() with p = exp(normalized)
    let mut ent = 0.0f64;
    for nz in &normalized {
        let term = -(*nz as f64) * nz.exp() as f64;
        if !term.is_nan() {
            ent += term;
        }
    }
    let shifted: Vec<f64> = normalized
        .iter()
        .map(|nz| (-*nz as f64 - ent).abs())
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| shifted[a].total_cmp(&shifted[b]));
    // cumulative_probs over softmax of the raw scores in shifted order.
    let sorted_logits: Vec<f64> = order.iter().map(|&i| scores[i] as f64).collect();
    let probs = softmax_f64(&sorted_logits);
    let mut cum = 0.0;
    let mut last_ind = order.len() - 1;
    for (k, p) in probs.iter().enumerate() {
        cum += p;
        if cum >= mass {
            last_ind = k;
            break;
        }
    }
    // python: last_ind = (cumulative_probs < mass).sum() — the first index
    // with cumulative mass >= mass; if all are below (float rounding) the
    // loop above leaves the final index.
    let mut remove = vec![false; n];
    for k in 0..n {
        if shifted[order[k]] > shifted[order[last_ind]] {
            remove[order[k]] = true;
        }
    }
    for &i in order.iter().take(min_tokens_to_keep.saturating_sub(1)) {
        remove[i] = false;
    }
    for (i, r) in remove.iter().enumerate() {
        if *r {
            scores[i] = f32::NEG_INFINITY;
        }
    }
}

/// First index whose cumulative (positive) weight exceeds `u`.
fn inverse_cdf(weights: &[f64], u: f64) -> usize {
    let mut acc = 0.0;
    let mut last_positive = 0;
    for (i, w) in weights.iter().enumerate() {
        if *w <= 0.0 {
            continue;
        }
        last_positive = i;
        acc += w;
        if acc > u {
            return i;
        }
    }
    last_positive
}

/// `torch.multinomial(probs, num_samples=n)` without replacement: draw a
/// uniform, pick via the inverse CDF of the remaining weights, then zero
/// the drawn entry and renormalize. (Exact torch RNG parity is impossible;
/// the caller owns the `StdRng` seed.)
fn sample_without_replacement(weights: &[f64], n: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut remaining = weights.to_vec();
    let mut total: f64 = remaining.iter().sum();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if total <= 0.0 {
            break;
        }
        let u = rng.random::<f64>() * total;
        let idx = inverse_cdf(&remaining, u);
        out.push(idx);
        total -= remaining[idx];
        remaining[idx] = 0.0;
    }
    out
}

// ---------------------------------------------------------------------------
// Beam search bookkeeping (`transformers_beam_search.py`).
// ---------------------------------------------------------------------------

/// `BeamHypotheses` with `early_stopping=False`.
struct BeamHypotheses {
    num_beams: usize,
    length_penalty: f64,
    /// (score, generated tokens)
    beams: Vec<(f64, Vec<u32>)>,
    worst_score: f64,
}

impl BeamHypotheses {
    fn new(num_beams: usize, length_penalty: f64) -> Self {
        Self {
            num_beams,
            length_penalty,
            beams: Vec::new(),
            worst_score: 1e9,
        }
    }

    fn penalized(&self, sum_logprobs: f64, generated_len: usize) -> f64 {
        if self.length_penalty == 0.0 {
            // generated_len ** 0.0 == 1.0
            sum_logprobs
        } else {
            sum_logprobs / (generated_len as f64).powf(self.length_penalty)
        }
    }

    fn add(&mut self, hyp: Vec<u32>, sum_logprobs: f64, generated_len: usize) {
        let score = self.penalized(sum_logprobs, generated_len);
        if self.beams.len() < self.num_beams || score > self.worst_score {
            self.beams.push((score, hyp));
            if self.beams.len() > self.num_beams {
                let mut lowest = 0;
                for i in 1..self.beams.len() {
                    if self.beams[i].0 < self.beams[lowest].0 {
                        lowest = i;
                    }
                }
                self.beams.remove(lowest);
                self.worst_score = self.beams.iter().map(|b| b.0).fold(f64::MAX, f64::min);
            } else {
                self.worst_score = self.worst_score.min(score);
            }
        }
    }

    /// Heuristic from `BeamHypotheses.is_done` with `early_stopping=False`.
    fn is_done(&self, best_sum_logprobs: f64, cur_generated_len: usize) -> bool {
        if self.beams.len() < self.num_beams {
            return false;
        }
        self.worst_score >= self.penalized(best_sum_logprobs, cur_generated_len)
    }
}

/// `BeamSearchScorer` for a single beam group (`num_return_sequences=1`).
struct BeamSearchScorer {
    num_beams: usize,
    hyps: Vec<BeamHypotheses>,
    done: Vec<bool>,
}

impl BeamSearchScorer {
    fn new(batch: usize, num_beams: usize, length_penalty: f64) -> Self {
        Self {
            num_beams,
            hyps: (0..batch)
                .map(|_| BeamHypotheses::new(num_beams, length_penalty))
                .collect(),
            done: vec![false; batch],
        }
    }

    fn is_done(&self) -> bool {
        self.done.iter().all(|d| *d)
    }

    /// `BeamSearchScorer.process`: `candidates` is `(score, flat index in
    /// beam-major (beams * vocab))` sorted descending. Returns per-beam-slot
    /// `(scores, tokens, source rows)`.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn process(
        &mut self,
        batch_idx: usize,
        candidates: &[(f64, usize)],
        input_ids: &[Vec<u32>],
        prompt_len: usize,
        vocab: usize,
        eos_token: u32,
        pad_token: u32,
        generated_len: usize,
    ) -> Result<(Vec<f64>, Vec<u32>, Vec<usize>)> {
        let nb = self.num_beams;
        if self.done[batch_idx] {
            return Ok((vec![0.0; nb], vec![pad_token; nb], vec![0; nb]));
        }
        let mut next_scores = vec![0.0; nb];
        let mut next_tokens = vec![0u32; nb];
        let mut next_rows = vec![0usize; nb];
        let mut beam_idx = 0;
        let mut best = f64::NEG_INFINITY;
        for (rank, &(score, flat)) in candidates.iter().enumerate() {
            best = best.max(score);
            let next_index = flat / vocab;
            let token = (flat % vocab) as u32;
            if token == eos_token {
                // Only candidates in the top num_beams ranks may finish.
                if rank >= nb {
                    continue;
                }
                let row = batch_idx * nb + next_index;
                let hyp = input_ids[row][prompt_len..].to_vec();
                self.hyps[batch_idx].add(hyp, score, generated_len);
            } else {
                next_scores[beam_idx] = score;
                next_tokens[beam_idx] = token;
                next_rows[beam_idx] = batch_idx * nb + next_index;
                beam_idx += 1;
            }
            if beam_idx == nb {
                break;
            }
        }
        if beam_idx < nb {
            // More than `num_beams` sampled candidates hit EOS: like HF's
            // BeamSearchScorer (which returns padded beams once
            // `beam_hyp.is_done()`), mark the batch done and pad the
            // remaining beams — the caller stops before consuming them.
            for slot in beam_idx..nb {
                next_scores[slot] = 0.0;
                next_tokens[slot] = pad_token;
                next_rows[slot] = 0;
            }
            self.done[batch_idx] = true;
            return Ok((next_scores, next_tokens, next_rows));
        }
        self.done[batch_idx] =
            self.done[batch_idx] || self.hyps[batch_idx].is_done(best, generated_len);
        Ok((next_scores, next_tokens, next_rows))
    }
}

/// Builds the `(rows, len)` additive-mask input tensor from 1/0 rows.
fn mask_tensor(rows: &[Vec<f32>], device: &candle_core::Device) -> Result<Tensor> {
    let width = rows.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Tensor::from_vec(flat, (rows.len(), width), device)
}

/// Embedding for the newest token: `embeddings(id) + mel_pos[pos]`.
fn token_embed(
    parts: &InferenceParts,
    ids: &[u32],
    pos: usize,
    rows: usize,
    dim: usize,
) -> Result<Tensor> {
    let device = parts.mel_head.weight().device();
    let ids_t = Tensor::from_vec(ids.to_vec(), (rows, 1), device)?;
    let emb = parts.embeddings.forward(&ids_t)?;
    let pos_row = parts
        .mel_pos_embedding
        .embeddings()
        .narrow(0, pos, 1)?
        .unsqueeze(0)?
        .to_dtype(emb.dtype())?
        .broadcast_as((rows, 1, dim))?;
    emb + pos_row
}

/// Beam-sample generation. `fake_inputs`: `(B, S+1)` i64 ids ending with
/// `start_mel_token`; `prompt_embeds`: `(B, S, dim)` (the `[pad][cond][text]`
/// embeddings without the start token); `attention_mask`: `(B, S+1)`.
/// Returns `(B, max_new_tokens)` generated ids (pad-trimmed rows keep
/// `eos_token`), ready for the caller to slice at the stop token.
#[allow(clippy::too_many_arguments)]
pub fn generate_mel_tokens(
    parts: &InferenceParts,
    fake_inputs: &Tensor,
    prompt_embeds: &Tensor,
    attention_mask: &Tensor,
    eos_token: u32,
    start_mel_token: u32,
    cfg: &GenerateConfig,
    rng: &mut StdRng,
) -> Result<Tensor> {
    let device = prompt_embeds.device();
    let dims = fake_inputs.dims();
    if dims.len() != 2 {
        candle_core::bail!("fake_inputs must be 2D, got {dims:?}");
    }
    let (b, s1) = (dims[0], dims[1]);
    let pe_dims = prompt_embeds.dims();
    if pe_dims.len() != 3 || pe_dims[0] != b || pe_dims[1] + 1 != s1 {
        candle_core::bail!(
            "prompt_embeds {:?} must be (batch {b}, {}, dim)",
            pe_dims,
            s1 - 1
        );
    }
    let (s, dim) = (pe_dims[1], pe_dims[2]);
    let prompt_len = s1;
    if attention_mask.dims() != [b, s1] {
        candle_core::bail!("attention_mask must be ({b}, {s1})");
    }
    let num_beams = cfg.num_beams;
    if num_beams <= 1 {
        candle_core::bail!("beam-sample requires num_beams > 1 (got {num_beams})");
    }
    if cfg.repetition_penalty <= 0.0 {
        candle_core::bail!("repetition_penalty must be > 0");
    }
    if cfg.do_sample && cfg.temperature <= 0.0 {
        candle_core::bail!("temperature must be > 0 when sampling");
    }
    if cfg.typical_sampling && !(0.0 < cfg.typical_mass && cfg.typical_mass < 1.0) {
        candle_core::bail!("typical_mass must be in (0, 1)");
    }
    let rows = b * num_beams;
    let vocab = parts.mel_head.weight().dims()[0];

    // `_expand_inputs_for_generation`: repeat_interleave over the batch dim.
    let fake_rows: Vec<Vec<u32>> = fake_inputs
        .to_dtype(DType::U32)?
        .to_vec2::<u32>()?
        .iter()
        .flat_map(|row| (0..num_beams).map(move |_| row.clone()))
        .collect();
    // `beam_scores[:, 1:] = -1e9` interleaved per batch row.
    let mut beam_scores: Vec<f64> = (0..rows)
        .map(|r| if r % num_beams == 0 { 0.0 } else { -1e9 })
        .collect();
    let mut mask_rows: Vec<Vec<f32>> = attention_mask
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?
        .iter()
        .flat_map(|row| (0..num_beams).map(move |_| row.clone()))
        .collect();
    let mut prompt_parts = Vec::with_capacity(rows);
    for bi in 0..b {
        let slice = prompt_embeds.narrow(0, bi, 1)?;
        for _ in 0..num_beams {
            prompt_parts.push(slice.clone());
        }
    }
    let prompt_exp = Tensor::cat(&prompt_parts, 0)?;

    let mut cache = Gpt2Cache::new(parts.gpt.layer_count());

    // ---- First step of `GPT2InferenceModel.forward`: the whole
    // `[cached_mel_emb | embeddings(start) + mel_pos[0]]` sequence.
    let start_ids: Vec<u32> = fake_rows.iter().map(|r| r[s]).collect();
    if start_ids.iter().any(|&t| t != start_mel_token) {
        candle_core::bail!("fake_inputs must end with start_mel_token {start_mel_token}");
    }
    let start_emb = token_embed(parts, &start_ids, 0, rows, dim)?;
    let emb = Tensor::cat(&[&prompt_exp, &start_emb], 1)?;
    let mut hidden =
        parts
            .gpt
            .forward(&emb, Some(&mask_tensor(&mask_rows, device)?), &mut cache)?;

    // ---- Beam-sample loop (`_beam_search` with do_sample=True).
    let mut scorer = BeamSearchScorer::new(b, num_beams, cfg.length_penalty);
    let n_keep = 2 * num_beams; // max(2, 1 + n_eos_tokens) * num_beams
    let mut input_ids = fake_rows;
    let mut steps = 0usize;
    while steps < cfg.max_new_tokens {
        let seq = hidden.dims()[1];
        let h_last = hidden.narrow(1, seq - 1, 1)?;
        let logits = parts
            .mel_head
            .forward(&parts.final_norm.forward(&h_last)?)?
            .squeeze(1)?
            .to_vec2::<f32>()?;

        // Processors then warpers, in `_get_logits_processor` order, applied
        // to log-softmaxed scores; running beam scores are added afterwards.
        let mut joint: Vec<Vec<f64>> = vec![Vec::with_capacity(num_beams * vocab); b];
        for (r, row_logits) in logits.iter().enumerate() {
            let mut scores = row_logits.clone();
            log_softmax(&mut scores);
            if cfg.repetition_penalty != 1.0 {
                apply_repetition_penalty(&mut scores, &input_ids[r], cfg.repetition_penalty);
            }
            if cfg.typical_sampling {
                apply_typical(&mut scores, cfg.typical_mass, 2);
            }
            if cfg.do_sample {
                if cfg.temperature != 1.0 {
                    apply_temperature(&mut scores, cfg.temperature);
                }
                if cfg.top_k != 0 {
                    apply_top_k(&mut scores, cfg.top_k, 2);
                }
                if cfg.top_p < 1.0 {
                    apply_top_p(&mut scores, cfg.top_p, 2);
                }
            }
            let bonus = beam_scores[r];
            joint[r / num_beams].extend(scores.iter().map(|v| *v as f64 + bonus));
        }

        // Candidate selection per batch row, then `BeamSearchScorer.process`.
        let mut next_scores = vec![0.0f64; rows];
        let mut next_tokens = vec![0u32; rows];
        let mut next_rows = vec![0usize; rows];
        for bi in 0..b {
            let weights = softmax_f64(&joint[bi]);
            let mut candidates: Vec<(f64, usize)> = if cfg.do_sample {
                sample_without_replacement(&weights, n_keep, rng)
            } else {
                let mut order: Vec<usize> = (0..joint[bi].len()).collect();
                order.sort_by(|&a, &x| joint[bi][x].total_cmp(&joint[bi][a]));
                order.truncate(n_keep);
                order
            }
            .into_iter()
            .map(|flat| (joint[bi][flat], flat))
            .collect();
            candidates.sort_by(|a, x| x.0.total_cmp(&a.0));
            let (ns, nt, nr) = scorer.process(
                bi,
                &candidates,
                &input_ids,
                prompt_len,
                vocab,
                eos_token,
                eos_token,
                steps + 1,
            )?;
            next_scores[bi * num_beams..(bi + 1) * num_beams].copy_from_slice(&ns);
            next_tokens[bi * num_beams..(bi + 1) * num_beams].copy_from_slice(&nt);
            next_rows[bi * num_beams..(bi + 1) * num_beams].copy_from_slice(&nr);
        }

        // Append tokens, reorder the KV cache and the mask rows.
        let mut new_ids = input_ids.clone();
        for r in 0..rows {
            new_ids[r] = input_ids[next_rows[r]].clone();
            new_ids[r].push(next_tokens[r]);
        }
        let beam_idx = Tensor::from_vec(
            next_rows.iter().map(|r| *r as u32).collect::<Vec<_>>(),
            rows,
            device,
        )?;
        cache.reorder_beams(&beam_idx)?;
        input_ids = new_ids;
        beam_scores = next_scores;
        for row in &mut mask_rows {
            row.push(1.0);
        }
        steps += 1;

        if scorer.is_done() {
            break;
        }
        // `EosTokenCriteria`: every row's last token is EOS (pad == eos
        // upstream, so finished batches already pad with EOS).
        if input_ids.iter().all(|r| r[r.len() - 1] == eos_token) {
            break;
        }
        if steps == cfg.max_new_tokens {
            break;
        }

        // Incremental step of `GPT2InferenceModel.forward`: the token
        // appended at iteration k-1 (= the k-th generated token) is fed
        // with mel position k+1 — the literal HF `kv_cache=True` formula
        // `attention_mask.shape[1] - mel_len` (position 1 is never used;
        // the start token took position 0 in the prompt step).
        let last_ids: Vec<u32> = input_ids.iter().map(|r| r[r.len() - 1]).collect();
        let emb = token_embed(parts, &last_ids, steps + 1, rows, dim)?;
        hidden = parts
            .gpt
            .forward(&emb, Some(&mask_tensor(&mask_rows, device)?), &mut cache)?;
    }

    // ---- `BeamSearchScorer.finalize` (num_beam_hyps_to_keep = 1).
    for bi in 0..b {
        if scorer.done[bi] {
            continue;
        }
        for j in 0..num_beams {
            let row = bi * num_beams + j;
            scorer.hyps[bi].add(
                input_ids[row][prompt_len..].to_vec(),
                beam_scores[row],
                steps,
            );
        }
    }
    let mut out = vec![eos_token; b * cfg.max_new_tokens];
    for bi in 0..b {
        // python: stable ascending sort then pop() -> the last maximum wins.
        let best = scorer.hyps[bi]
            .beams
            .iter()
            .max_by(|a, x| a.0.total_cmp(&x.0))
            .map(|(_, hyp)| hyp.clone())
            .unwrap_or_default();
        for (i, token) in best.iter().enumerate() {
            if i >= cfg.max_new_tokens {
                break;
            }
            out[bi * cfg.max_new_tokens + i] = *token;
        }
        // Rows keep the EOS token: positions past the hypothesis (including
        // the one right after it) stay `eos_token` (== pad upstream).
    }
    Tensor::from_vec(out, (b, cfg.max_new_tokens), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarBuilder;
    use rand::SeedableRng;

    #[test]
    fn repetition_penalty_both_signs() {
        // transformers 4.52.1: where(score < 0, score * penalty, score / penalty)
        let mut scores = vec![-2.0f32, 3.0, -0.5, 1.0];
        apply_repetition_penalty(&mut scores, &[0, 1, 3], 2.0);
        assert_eq!(scores, vec![-4.0, 1.5, -0.5, 0.5]);
        // On log-probs (all negative) a penalty > 1 pushes scores down.
        let mut scores = vec![-2.0f32, -3.0];
        apply_repetition_penalty(&mut scores, &[0], 10.0);
        assert_eq!(scores, vec![-20.0, -3.0]);
        // Duplicated ids are applied once (gather reads the originals).
        let mut scores = vec![4.0f32, 1.0];
        apply_repetition_penalty(&mut scores, &[0, 0], 2.0);
        assert_eq!(scores, vec![2.0, 1.0]);
    }

    #[test]
    fn top_k_keeps_ties_at_kth() {
        let mut scores = vec![1.0f32, 4.0, 4.0, 2.0, 3.0];
        apply_top_k(&mut scores, 2, 2);
        assert_eq!(
            scores,
            vec![
                f32::NEG_INFINITY,
                4.0,
                4.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY
            ]
        );
        // min_tokens_to_keep raises an undersized k.
        let mut scores = vec![5.0f32, 4.0, 1.0, 1.0];
        apply_top_k(&mut scores, 1, 2);
        assert_eq!(scores, vec![5.0, 4.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        let mut scores = vec![5.0f32, 1.0, 2.0];
        apply_top_k(&mut scores, 2, 2);
        assert_eq!(scores[2], 2.0);
        assert_eq!(scores[1], f32::NEG_INFINITY);
    }

    #[test]
    fn top_p_mass_boundary() {
        // probs [0.5, 0.3, 0.2], top_p = 0.5: ascending cumulative probs are
        // 0.2, 0.5, 1.0; the removal rule is `cum <= 1 - top_p`, so the two
        // smallest qualify for removal, but `min_tokens_to_keep = 2` always
        // keeps the two largest scores.
        let mut scores = vec![0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        apply_top_p(&mut scores, 0.5, 2);
        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());
        assert_eq!(scores[2], f32::NEG_INFINITY);
        // The nucleus keeps as many tokens as fit under the mass even when
        // that is more than min_tokens_to_keep: probs [0.55, 0.25, 0.15,
        // 0.05], top_p = 0.85 -> cumulative 0.05, 0.2, 0.45, 1.0 against
        // 1 - top_p = 0.15, so only the 0.05 token is dropped.
        let mut scores = vec![0.55f32.ln(), 0.25f32.ln(), 0.15f32.ln(), 0.05f32.ln()];
        apply_top_p(&mut scores, 0.85, 2);
        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());
        assert!(scores[2].is_finite());
        assert_eq!(scores[3], f32::NEG_INFINITY);
        // Cumulative mass above the boundary everywhere: nothing dropped.
        let mut scores = vec![0.6f32.ln(), 0.25f32.ln(), 0.15f32.ln()];
        apply_top_p(&mut scores, 0.9, 2);
        assert!(scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn typical_warper_matches_python_semantics() {
        // Softmax(scores) = [0.5, 0.3, 0.2] -> ent ~ 1.0297;
        // shifted = |ln p - ent| = [0.3366, 0.1743, 0.5797]; ascending order
        // [1, 0, 2]; cumulative probs over sorted raw scores = [0.3, 0.8, 1];
        // mass 0.5 -> last_ind = 1 -> remove shifted > shifted[1]: only
        // token 2.
        let mut scores = vec![0.5f32.ln(), 0.3f32.ln(), 0.2f32.ln()];
        apply_typical(&mut scores, 0.5, 2);
        assert!(scores[0].is_finite());
        assert!(scores[1].is_finite());
        assert_eq!(scores[2], f32::NEG_INFINITY);
    }

    #[test]
    fn inverse_cdf_buckets() {
        // weights [0.2, 0.5, 0.3]: cumulative (0.2, 0.7, 1.0).
        let w = [0.2, 0.5, 0.3];
        assert_eq!(inverse_cdf(&w, 0.1), 0);
        assert_eq!(inverse_cdf(&w, 0.2), 1);
        assert_eq!(inverse_cdf(&w, 0.69), 1);
        assert_eq!(inverse_cdf(&w, 0.7), 2);
        assert_eq!(inverse_cdf(&w, 0.999), 2);
        // Zero weights are skipped.
        assert_eq!(inverse_cdf(&[0.0, 0.5, 0.5], 0.1), 1);
    }

    #[test]
    fn sampling_is_deterministic_and_mass_aware() {
        // Same seed -> same draws; without-replacement never repeats.
        let weights: Vec<f64> = [0.05, 0.05, 0.8, 0.05, 0.05].to_vec();
        let a = sample_without_replacement(&weights, 3, &mut StdRng::seed_from_u64(294_738));
        let b = sample_without_replacement(&weights, 3, &mut StdRng::seed_from_u64(294_738));
        assert_eq!(a, b);
        let unique: std::collections::HashSet<_> = a.iter().collect();
        assert_eq!(unique.len(), a.len());
        // The heavy bucket dominates first draws.
        let mut firsts = [0usize; 5];
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..400 {
            firsts[sample_without_replacement(&weights, 1, &mut rng)[0]] += 1;
        }
        assert!(firsts[2] > 300, "first draws {firsts:?}");
    }

    #[test]
    fn log_softmax_matches_reference() {
        let mut row = vec![1.0f32, 2.0, 3.0];
        log_softmax(&mut row);
        let expected = [
            (1.0f32 - 3.0) - (0f32.exp() + (-1.0f32).exp() + (-2.0f32).exp()).ln(),
            (2.0 - 3.0) - (0f32.exp() + (-1.0f32).exp() + (-2.0f32).exp()).ln(),
            0.0f32 - (0f32.exp() + (-1.0f32).exp() + (-2.0f32).exp()).ln(),
        ];
        for (a, e) in row.iter().zip(expected) {
            assert!((a - e).abs() < 1e-6);
        }
    }

    // ---- Tiny end-to-end beam-sample runs on a VarMap-backed model ----

    struct TinyParts {
        model: crate::gpt2::Gpt2Model,
        embeddings: Embedding,
        mel_pos: Embedding,
        final_norm: LayerNorm,
        mel_head: Linear,
    }

    fn tiny_parts(vocab: usize, dim: usize) -> TinyParts {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let cfg = crate::gpt2::Gpt2Config::for_indextts(dim, 2, 2, 64);
        let model = crate::gpt2::Gpt2Model::load(&vb.pp("gpt"), &cfg, &device).unwrap();
        let embeddings = Embedding::new(vb.get((vocab, dim), "mel_emb.weight").unwrap(), dim);
        let mel_pos = Embedding::new(vb.get((64, dim), "mel_pos.emb.weight").unwrap(), dim);
        let final_norm = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm")).unwrap();
        let mel_head = candle_nn::linear(dim, vocab, vb.pp("head")).unwrap();
        // Deterministic non-zero weights (VarMap starts at zero).
        let mut state: u32 = 0x1234_5678;
        let mut next = move || {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((state >> 16) as f32 / 32768.0 - 0.5) * 0.2
        };
        let data = varmap.data().lock().unwrap();
        for var in data.values() {
            let n = var.shape().elem_count();
            let values: Vec<f32> = (0..n).map(|_| next()).collect();
            let shape = var.shape().clone();
            var.set(&Tensor::from_vec(values, shape, &device).unwrap())
                .unwrap();
        }
        drop(data);
        TinyParts {
            model,
            embeddings,
            mel_pos,
            final_norm,
            mel_head,
        }
    }

    fn run_tiny(parts: &TinyParts, cfg: &GenerateConfig, seed: u64) -> Tensor {
        let device = Device::Cpu;
        let dim = 8;
        let fake = Tensor::from_vec(vec![1i64, 1, 1, 7], (1, 4), &device).unwrap();
        // Deterministic prompt (device randn is not seedable here).
        let mut state: u32 = 0xC0FF_EE01;
        let prompt_values: Vec<f32> = (0..3 * dim)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((state >> 16) as f32 / 32768.0 - 0.5) * 0.6
            })
            .collect();
        let prompt = Tensor::from_vec(prompt_values, (1, 3, dim), &device).unwrap();
        let mask = Tensor::ones((1, 4), DType::F32, &device).unwrap();
        generate_mel_tokens(
            &InferenceParts {
                gpt: &parts.model,
                embeddings: &parts.embeddings,
                mel_pos_embedding: &parts.mel_pos,
                final_norm: &parts.final_norm,
                mel_head: &parts.mel_head,
            },
            &fake,
            &prompt,
            &mask,
            5,
            7,
            cfg,
            &mut StdRng::seed_from_u64(seed),
        )
        .unwrap()
    }

    #[test]
    fn tiny_beam_sample_shape_and_values() {
        let parts = tiny_parts(16, 8);
        let cfg = GenerateConfig {
            num_beams: 2,
            top_k: 8,
            max_new_tokens: 5,
            repetition_penalty: 1.5,
            ..GenerateConfig::default()
        };
        let out = run_tiny(&parts, &cfg, 42);
        assert_eq!(out.dims(), vec![1, 5]);
        let ids = out.to_vec2::<u32>().unwrap();
        assert!(ids[0].iter().all(|t| (*t as usize) < 16));
    }

    #[test]
    fn tiny_beam_sample_is_seed_deterministic() {
        let parts = tiny_parts(16, 8);
        let cfg = GenerateConfig {
            num_beams: 2,
            top_k: 8,
            max_new_tokens: 5,
            repetition_penalty: 1.5,
            ..GenerateConfig::default()
        };
        let a = run_tiny(&parts, &cfg, 4242);
        let b = run_tiny(&parts, &cfg, 4242);
        assert_eq!(a.to_vec2::<u32>().unwrap(), b.to_vec2::<u32>().unwrap());
    }

    #[test]
    fn tiny_greedy_beam_search_is_stable() {
        let parts = tiny_parts(16, 8);
        let cfg = GenerateConfig {
            do_sample: false,
            num_beams: 2,
            max_new_tokens: 4,
            repetition_penalty: 1.2,
            ..GenerateConfig::default()
        };
        let a = run_tiny(&parts, &cfg, 0);
        let b = run_tiny(&parts, &cfg, 99);
        // Greedy ignores the RNG entirely.
        assert_eq!(a.to_vec2::<u32>().unwrap(), b.to_vec2::<u32>().unwrap());
        assert_eq!(a.dims(), vec![1, 4]);
    }

    /// Real-weights oracle. Run with
    /// `INDEXTTS_CONVERTED_DIR=<dir> cargo test -p indextts --lib -- --ignored generation_real`
    /// where `<dir>` contains `gpt.safetensors` (keys `gpt.h.*`,
    /// `mel_embedding.weight`, `mel_pos_embedding.emb.weight`,
    /// `final_norm.*`, `mel_head.*`).
    #[test]
    #[ignore = "requires converted real weights (INDEXTTS_CONVERTED_DIR)"]
    fn generation_real_weights_smoke() {
        let Some(dir) = std::env::var_os("INDEXTTS_CONVERTED_DIR") else {
            eprintln!("INDEXTTS_CONVERTED_DIR not set; skipping");
            return;
        };
        let device = Device::Cpu;
        let path = std::path::Path::new(&dir).join("gpt.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device) }
            .expect("open gpt.safetensors");
        let cfg = crate::gpt2::Gpt2Config::for_indextts(1280, 24, 20, 2417);
        let model = crate::gpt2::Gpt2Model::load(&vb.pp("gpt"), &cfg, &device).expect("load gpt2");
        let embeddings = Embedding::new(vb.get_unchecked("mel_embedding.weight").unwrap(), 1280);
        let mel_pos = Embedding::new(
            vb.get_unchecked("mel_pos_embedding.emb.weight").unwrap(),
            1280,
        );
        let final_norm = candle_nn::layer_norm(1280, 1e-5, vb.pp("final_norm")).unwrap();
        let mel_head = candle_nn::linear(1280, 8194, vb.pp("mel_head")).unwrap();

        let s = 10;
        let fake = Tensor::from_vec(
            {
                let mut v = vec![1i64; s];
                v.push(8192);
                v
            },
            (1, s + 1),
            &device,
        )
        .unwrap();
        let prompt = Tensor::randn(0f32, 1f32, (1, s, 1280), &device).unwrap();
        let mask = Tensor::ones((1, s + 1), DType::F32, &device).unwrap();
        let gen_cfg = GenerateConfig {
            max_new_tokens: 12,
            ..GenerateConfig::default()
        };
        let out = generate_mel_tokens(
            &InferenceParts {
                gpt: &model,
                embeddings: &embeddings,
                mel_pos_embedding: &mel_pos,
                final_norm: &final_norm,
                mel_head: &mel_head,
            },
            &fake,
            &prompt,
            &mask,
            8193,
            8192,
            &gen_cfg,
            &mut StdRng::seed_from_u64(0),
        )
        .expect("generate");
        assert_eq!(out.dims(), vec![1, 12]);
        assert!(
            out.to_vec2::<u32>().unwrap()[0]
                .iter()
                .all(|t| (*t as usize) < 8194)
        );
    }
}
