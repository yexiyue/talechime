//! CosyVoice3 speech-token language model (Qwen2-0.5B backbone).
//!
//! Mirrors `cosyvoice/llm/llm.py::CosyVoice3LM.inference` (unicast path):
//! `sos + [prompt_text, text] + task_id + prompt_speech_tokens` fed through
//! the Qwen2 stack, per-step logits from `llm_decoder` (no bias), sampling
//! via `ras_sampling`, stopping on any token >= `speech_token_size`.

use std::sync::Arc;

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{Embedding, RmsNorm, VarBuilder};

use crate::config::{LlmConfig, SOS_TOKEN, SamplingConfig, TASK_ID_TOKEN};
use crate::flow::embeddings::LinT2d;
use crate::rng::StepRng;

/// Qwen2 rotary table over `head_dim` (interleaved-half convention).
#[derive(Debug)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dim: usize,
        max_seq_len: usize,
        theta: f64,
        dtype: DType,
        dev: &candle_core::Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, dim / 2), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq = q.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq)?;
        let sin = self.sin.narrow(0, offset, seq)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

#[derive(Debug)]
struct Attention {
    q_proj: LinT2d,
    k_proj: LinT2d,
    v_proj: LinT2d,
    o_proj: LinT2d,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary: Arc<RotaryEmbedding>,
    /// Cached `(k_t [B, H, D, T], v [B, H, T, D])` already expanded across KV groups.
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = hidden / heads;
        Ok(Self {
            // Qwen2: bias on q/k/v, none on o.
            q_proj: LinT2d::from_linear(candle_nn::linear(
                hidden,
                heads * head_dim,
                vb.pp("q_proj"),
            )?)?,
            k_proj: LinT2d::from_linear(candle_nn::linear(
                hidden,
                kv_heads * head_dim,
                vb.pp("k_proj"),
            )?)?,
            v_proj: LinT2d::from_linear(candle_nn::linear(
                hidden,
                kv_heads * head_dim,
                vb.pp("v_proj"),
            )?)?,
            o_proj: LinT2d::from_linear(candle_nn::linear_no_bias(
                heads * head_dim,
                hidden,
                vb.pp("o_proj"),
            )?)?,
            num_heads: heads,
            num_kv_heads: kv_heads,
            num_kv_groups: heads / kv_heads,
            head_dim,
            hidden_size: hidden,
            rotary,
            kv_cache: None,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (q, k) = self.rotary.apply(&q, &k, offset)?;
        let k_t = if t == 1 {
            k.reshape((b, self.num_kv_heads, self.head_dim, 1))?
        } else {
            k.transpose(2, 3)?.contiguous()?
        };
        let k = candle_transformers::utils::repeat_kv(k_t, self.num_kv_groups)?;
        let v = candle_transformers::utils::repeat_kv(v, self.num_kv_groups)?;
        let (k, v) = match &self.kv_cache {
            Some((pk, pv)) => (Tensor::cat(&[pk, &k], 3)?, Tensor::cat(&[pv, &v], 2)?),
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));
        let attn = q
            .matmul(&k)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = match mask {
            Some(m) => attn.broadcast_add(m)?,
            None => attn,
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, self.hidden_size))?;
        self.o_proj.forward(&out)
    }

    fn clear_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Debug)]
struct Layer {
    attn: Attention,
    gate_proj: LinT2d,
    up_proj: LinT2d,
    down_proj: LinT2d,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Layer {
    fn new(cfg: &LlmConfig, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.llm_input_size;
        Ok(Self {
            attn: Attention::new(
                hidden,
                cfg.qwen2.num_attention_heads,
                cfg.qwen2.num_key_value_heads,
                rotary,
                vb.pp("self_attn"),
            )?,
            gate_proj: LinT2d::from_linear(candle_nn::linear_no_bias(
                hidden,
                cfg.qwen2.intermediate_size,
                vb.pp("mlp.gate_proj"),
            )?)?,
            up_proj: LinT2d::from_linear(candle_nn::linear_no_bias(
                hidden,
                cfg.qwen2.intermediate_size,
                vb.pp("mlp.up_proj"),
            )?)?,
            down_proj: LinT2d::from_linear(candle_nn::linear_no_bias(
                cfg.qwen2.intermediate_size,
                hidden,
                vb.pp("mlp.down_proj"),
            )?)?,
            input_layernorm: candle_nn::rms_norm(
                hidden,
                cfg.qwen2.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: candle_nn::rms_norm(
                hidden,
                cfg.qwen2.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = (self.attn.forward(&x, mask, offset)? + residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let mlp = self
            .down_proj
            .forward(&((self.gate_proj.forward(&x)?.silu()? * self.up_proj.forward(&x)?)?))?;
        residual + mlp
    }
}

/// The speech-token LLM. Weight layout matches the converted `llm.safetensors`
/// (`llm.embed_tokens`, `llm.layers.N...`, `llm.norm`, `speech_embedding`,
/// `llm_decoder`).
#[derive(Debug)]
pub struct CosyVoice3Lm {
    embed_tokens: Embedding,
    speech_embedding: Embedding,
    llm_decoder: LinT2d,
    layers: Vec<Layer>,
    norm: RmsNorm,
    speech_token_size: u32,
    sos_emb: Tensor,
    task_emb: Tensor,
    eos_mask: Tensor,
    device: candle_core::Device,
    dtype: DType,
    /// Identity of the single KV-cache run currently owned by this model.
    active_run: Option<Arc<()>>,
}

/// Carry-along state for [`CosyVoice3Lm::next_token`].
pub struct IncrementalState {
    rng: StepRng,
    min_len: usize,
    max_len: usize,
    offset: usize,
    step: usize,
    logits: Tensor,
    /// Tokens sampled so far (post silent-filter is the caller's concern).
    decoded: Vec<u32>,
    /// Generation finished (stop token or length limit).
    done: bool,
    /// Generation ended by hitting `max_len`, not by a stop token.
    hit_limit: bool,
    failed: bool,
    run_id: Arc<()>,
    sampling: SamplingConfig,
}

impl IncrementalState {
    /// Whether this run is terminal (stop, length budget or inference failure).
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Whether generation exhausted its length budget without a stop token.
    pub fn hit_limit(&self) -> bool {
        self.hit_limit
    }
}

impl CosyVoice3Lm {
    pub fn new(cfg: &LlmConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.llm_input_size;
        let head_dim = hidden / cfg.qwen2.num_attention_heads;
        let rotary = Arc::new(RotaryEmbedding::new(
            head_dim,
            cfg.qwen2.max_position_embeddings,
            cfg.qwen2.rope_theta,
            vb.dtype(),
            vb.device(),
        )?);
        let mut layers = Vec::with_capacity(cfg.qwen2.num_hidden_layers);
        let vb_layers = vb.pp("llm.layers");
        for i in 0..cfg.qwen2.num_hidden_layers {
            layers.push(Layer::new(cfg, rotary.clone(), vb_layers.pp(i))?);
        }
        let speech_vocab = cfg.speech_token_size + 200;
        let speech_embedding =
            candle_nn::embedding(speech_vocab, hidden, vb.pp("speech_embedding"))?;
        let sos = Tensor::from_slice(&[SOS_TOKEN], (1, 1), vb.device())?;
        let task = Tensor::from_slice(&[TASK_ID_TOKEN], (1, 1), vb.device())?;
        let sos_emb = speech_embedding.forward(&sos)?;
        let task_emb = speech_embedding.forward(&task)?;
        let mut eos_bias = vec![0.0f32; speech_vocab];
        eos_bias[cfg.speech_token_size] = f32::NEG_INFINITY;
        let eos_mask = Tensor::from_vec(eos_bias, speech_vocab, &candle_core::Device::Cpu)?
            .to_dtype(vb.dtype())?
            .to_device(vb.device())?;
        Ok(Self {
            embed_tokens: candle_nn::embedding(
                cfg.qwen2.vocab_size,
                hidden,
                vb.pp("llm.embed_tokens"),
            )?,
            speech_embedding,
            llm_decoder: LinT2d::from_linear(candle_nn::linear_no_bias(
                cfg.llm_output_size,
                speech_vocab,
                vb.pp("llm_decoder"),
            )?)?,
            layers,
            norm: candle_nn::rms_norm(hidden, cfg.qwen2.rms_norm_eps, vb.pp("llm.norm"))?,
            speech_token_size: cfg.speech_token_size as u32,
            sos_emb,
            task_emb,
            eos_mask,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            active_run: None,
        })
    }

    pub fn speech_token_size(&self) -> u32 {
        self.speech_token_size
    }

    pub fn clear_cache(&mut self) {
        self.active_run = None;
        for l in self.layers.iter_mut() {
            l.attn.clear_cache();
        }
    }

    fn forward_embeds(&mut self, embeds: &Tensor, offset: usize) -> Result<Tensor> {
        let seq = embeds.dim(1)?;
        let mask = if seq > 1 {
            let m: Vec<f32> = (0..seq)
                .flat_map(|i| (0..seq).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
                .collect();
            Some(Tensor::from_vec(m, (1, 1, seq, seq), &self.device)?.to_dtype(self.dtype)?)
        } else {
            None
        };
        let mut x = embeds.clone();
        for l in self.layers.iter_mut() {
            x = l.forward(&x, mask.as_ref(), offset)?;
        }
        self.norm.forward(&x)
    }

    /// Reset the KV cache and return the first-step logits `[vocab]`.
    /// Shared by synthesis and teacher-forced numerical comparison.
    pub fn prefill_logits(
        &mut self,
        text_tokens: &Tensor,
        prompt_text_tokens: &Tensor,
        prompt_speech_tokens: &Tensor,
    ) -> Result<Tensor> {
        let prompt_text_len = prompt_text_tokens.dim(1)?;
        let text = if prompt_text_len == 0 {
            text_tokens.clone()
        } else {
            Tensor::cat(&[prompt_text_tokens, text_tokens], 1)?
        };
        let text_emb = self.embed_tokens.forward(&text)?;

        let lm_input = if prompt_speech_tokens.dim(1)? == 0 {
            Tensor::cat(&[&self.sos_emb, &text_emb, &self.task_emb], 1)?
        } else {
            let prompt_emb = self.speech_embedding.forward(prompt_speech_tokens)?;
            Tensor::cat(&[&self.sos_emb, &text_emb, &self.task_emb, &prompt_emb], 1)?
        };

        self.clear_cache();
        let hidden = self.forward_embeds(&lm_input, 0)?;
        let seq = hidden.dim(1)?;
        self.llm_decoder
            .forward(&hidden.narrow(1, seq - 1, 1)?)?
            .flatten_all()
    }

    /// Consume one speech token at an explicit KV-cache offset.
    pub fn decode_logits(&mut self, token: u32, offset: usize) -> Result<Tensor> {
        self.active_run = None;
        self.decode_logits_inner(token, offset)
    }

    fn decode_logits_inner(&mut self, token: u32, offset: usize) -> Result<Tensor> {
        let token = Tensor::from_slice(&[token], (1, 1), &self.device)?;
        let hidden = self.forward_embeds(&self.speech_embedding.forward(&token)?, offset)?;
        self.llm_decoder.forward(&hidden)?.flatten_all()
    }

    /// Autoregressive generation for one request. Hitting the length limit
    /// is an error, rather than a falsely completed utterance.
    pub fn inference(
        &mut self,
        text_tokens: &Tensor,
        prompt_text_tokens: &Tensor,
        prompt_speech_tokens: &Tensor,
        sampling: &SamplingConfig,
    ) -> Result<Vec<u32>> {
        let mut st = self.start_incremental(
            text_tokens,
            prompt_text_tokens,
            prompt_speech_tokens,
            sampling,
        )?;
        while self.next_token(&mut st)?.is_some() {}
        if st.hit_limit {
            candle_core::bail!(
                "LLM reached the {}-token limit without a stop token",
                st.max_len
            );
        }
        Ok(st.decoded)
    }

    /// Begin incremental decoding; [`Self::next_token`] advances it. This is
    /// the single generation path shared by batch inference and streaming.
    pub fn start_incremental(
        &mut self,
        text_tokens: &Tensor,
        prompt_text_tokens: &Tensor,
        prompt_speech_tokens: &Tensor,
        sampling: &SamplingConfig,
    ) -> Result<IncrementalState> {
        let text_len = text_tokens.dim(1)?;
        if text_len == 0 {
            candle_core::bail!("text must contain at least one token");
        }
        if !sampling.top_p.is_finite()
            || sampling.top_p <= 0.0
            || sampling.top_p > 1.0
            || sampling.top_k == 0
            || sampling.win_size == 0
            || !sampling.tau_r.is_finite()
            || !(0.0..=1.0).contains(&sampling.tau_r)
        {
            candle_core::bail!("invalid RAS sampling parameters");
        }
        let logits = self.prefill_logits(text_tokens, prompt_text_tokens, prompt_speech_tokens)?;
        let run_id = Arc::new(());
        self.active_run = Some(run_id.clone());
        Ok(IncrementalState {
            rng: StepRng::new(sampling.seed.unwrap_or_else(rand_seed)),
            min_len: text_len * 2,
            max_len: text_len * 20,
            offset: 2 + text_len + prompt_text_tokens.dim(1)? + prompt_speech_tokens.dim(1)?,
            step: 0,
            logits,
            decoded: Vec::new(),
            done: false,
            hit_limit: false,
            failed: false,
            run_id,
            sampling: sampling.clone(),
        })
    }

    /// Sample one more speech token. `Ok(None)` marks the end of generation
    /// (stop token, or the length limit — see `IncrementalState::hit_limit`).
    pub fn next_token(&mut self, st: &mut IncrementalState) -> Result<Option<u32>> {
        if st.failed {
            candle_core::bail!("LLM run has failed; start a new run");
        }
        if !self
            .active_run
            .as_ref()
            .is_some_and(|id| Arc::ptr_eq(id, &st.run_id))
        {
            candle_core::bail!("incremental state belongs to a different or replaced KV-cache run");
        }
        let result = self.next_token_inner(st);
        if result.is_err() {
            st.failed = true;
            st.done = true;
            self.clear_cache();
        }
        result
    }

    fn next_token_inner(&mut self, st: &mut IncrementalState) -> Result<Option<u32>> {
        if st.done {
            return Ok(None);
        }
        let next = self.sample(
            &st.logits,
            &st.decoded,
            &st.sampling,
            &mut st.rng,
            st.step < st.min_len,
        )?;
        if next >= self.speech_token_size {
            st.done = true;
            return Ok(None);
        }
        st.decoded.push(next);
        st.step += 1;
        if st.step < st.max_len {
            st.logits = self.decode_logits_inner(next, st.offset)?;
            st.offset += 1;
        } else {
            st.done = true;
            st.hit_limit = true;
        }
        Ok(Some(next))
    }

    /// `sampling_ids` + `ras_sampling` from `utils.common`: mask eos when
    /// `ignore_eos`, nucleus sample, fall back to full softmax sampling when
    /// a token repeats too often in the trailing window.
    fn sample(
        &self,
        logits: &Tensor,
        decoded: &[u32],
        cfg: &SamplingConfig,
        rng: &mut StepRng,
        ignore_eos: bool,
    ) -> Result<u32> {
        let logits = if ignore_eos {
            (logits + &self.eos_mask)?
        } else {
            logits.clone()
        };
        let probs: Vec<f32> = candle_nn::ops::softmax_last_dim(&logits)?.to_vec1()?;
        let top = nucleus_sample_probs(&probs, cfg.top_p, cfg.top_k, rng);
        let recent = decoded.len().saturating_sub(cfg.win_size)..decoded.len();
        let rep_num = decoded[recent].iter().filter(|&&t| t == top).count();
        if rep_num as f32 >= cfg.win_size as f32 * cfg.tau_r {
            return Ok(random_sample_probs(&probs, rng));
        }
        Ok(top)
    }
}

fn rand_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5EED)
}

/// `nucleus_sampling`: softmax, stable sort descending, accumulate until
/// `top_p` or `top_k`, renormalize, multinomial draw.
#[cfg(test)]
fn nucleus_sample(logits: &Tensor, top_p: f32, top_k: usize, rng: &mut StepRng) -> Result<u32> {
    let probs: Vec<f32> = candle_nn::ops::softmax_last_dim(logits)?.to_vec1()?;
    Ok(nucleus_sample_probs(&probs, top_p, top_k, rng))
}

fn nucleus_sample_probs(probs: &[f32], top_p: f32, top_k: usize, rng: &mut StepRng) -> u32 {
    // Python sorts values descending with stable=True; ties keep original
    // (ascending-index) order.
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    let cmp = |&a: &usize, &b: &usize| probs[b].total_cmp(&probs[a]).then(a.cmp(&b));
    if top_k > 0 && top_k < idx.len() {
        let (head, _, _) = idx.select_nth_unstable_by(top_k, cmp);
        head.sort_unstable_by(cmp);
        idx.truncate(top_k + 1);
    } else {
        idx.sort_unstable_by(cmp);
    }

    let mut cum = 0.0f32;
    let mut cands: Vec<(usize, f32)> = Vec::new();
    for &i in &idx {
        if cum < top_p && cands.len() < top_k {
            cum += probs[i];
            cands.push((i, probs[i]));
        } else {
            break;
        }
    }
    if cands.is_empty() {
        return idx[0] as u32;
    }
    let total: f32 = cands.iter().map(|(_, p)| p).sum();
    let pick = rng.next_f32() * total;
    let mut acc = 0.0f32;
    for &(i, p) in &cands {
        acc += p;
        if pick < acc {
            return i as u32;
        }
    }
    cands.last().unwrap().0 as u32
}

fn random_sample_probs(probs: &[f32], rng: &mut StepRng) -> u32 {
    let pick = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if pick < acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model(max_positions: usize) -> Result<CosyVoice3Lm> {
        let mut cfg = LlmConfig {
            llm_input_size: 56,
            llm_output_size: 56,
            ..LlmConfig::default()
        };
        cfg.qwen2.num_hidden_layers = 1;
        cfg.qwen2.vocab_size = 32;
        cfg.qwen2.intermediate_size = 56;
        cfg.qwen2.max_position_embeddings = max_positions;
        CosyVoice3Lm::new(
            &cfg,
            VarBuilder::zeros(DType::F32, &candle_core::Device::Cpu),
        )
    }

    #[test]
    fn incremental_state_rejects_replaced_or_foreign_cache() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let text = Tensor::from_slice(&[3u32], (1, 1), &device)?;
        let empty = Tensor::zeros((1, 0), DType::U32, &device)?;
        let sampling = SamplingConfig::default();
        let mut model = tiny_model(128)?;
        let mut old = model.start_incremental(&text, &empty, &empty, &sampling)?;
        let mut current = model.start_incremental(&text, &empty, &empty, &sampling)?;
        assert!(model.next_token(&mut old).is_err());
        let mut other = tiny_model(128)?;
        assert!(other.next_token(&mut current).is_err());
        assert!(model.next_token(&mut current)?.is_some());
        model.clear_cache();
        assert!(model.next_token(&mut current).is_err());
        Ok(())
    }

    #[test]
    fn decode_failure_is_terminal_instead_of_reusing_partial_kv_cache() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mut model = tiny_model(3)?;
        let text = Tensor::from_slice(&[3u32], (1, 1), &device)?;
        let empty = Tensor::zeros((1, 0), DType::U32, &device)?;
        let sampling = SamplingConfig {
            top_k: 1,
            win_size: 100,
            tau_r: 1.0,
            seed: Some(7),
            ..SamplingConfig::default()
        };
        let mut state = model.start_incremental(&text, &empty, &empty, &sampling)?;
        assert!(model.next_token(&mut state).is_err());
        assert!(state.failed);
        assert!(
            model
                .next_token(&mut state)
                .unwrap_err()
                .to_string()
                .contains("has failed")
        );
        Ok(())
    }

    #[test]
    fn gqa_prefill_decode_and_sampling_keep_single_request_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mut cfg = LlmConfig {
            llm_input_size: 56,
            llm_output_size: 56,
            ..LlmConfig::default()
        };
        cfg.qwen2.num_hidden_layers = 1;
        cfg.qwen2.vocab_size = 32;
        cfg.qwen2.intermediate_size = 56;
        cfg.qwen2.max_position_embeddings = 128;
        let mut llm = CosyVoice3Lm::new(&cfg, VarBuilder::zeros(DType::F32, &device))?;
        let text = Tensor::from_slice(&[3u32], (1, 1), &device)?;
        let empty = Tensor::zeros((1, 0), DType::U32, &device)?;
        assert_eq!(llm.prefill_logits(&text, &empty, &empty)?.dims(), &[6761]);
        assert_eq!(llm.decode_logits(7, 3)?.dims(), &[6761]);
        let sampling = SamplingConfig {
            top_k: 1,
            win_size: 100,
            tau_r: 1.0,
            seed: Some(7),
            ..SamplingConfig::default()
        };
        let err = llm.inference(&text, &empty, &empty, &sampling).unwrap_err();
        assert!(err.to_string().contains("without a stop token"));
        Ok(())
    }

    #[test]
    fn nucleus_respects_top_k() -> Result<()> {
        let device = candle_core::Device::Cpu;
        // Two dominant tokens out of five.
        let logits = Tensor::from_slice(&[0.1f32, 10.0, 9.0, 0.1, 0.1], 5, &device)?;
        let mut rng = StepRng::new(7);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..64 {
            seen.insert(nucleus_sample(&logits, 0.9, 2, &mut rng)?);
        }
        assert!(seen.iter().all(|&t| t == 1 || t == 2));
        Ok(())
    }

    #[test]
    fn sample_ignore_eos_blocks_eos_token() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let model = tiny_model(32)?;
        let vocab = model.speech_token_size as usize + 200;
        let mut raw = vec![-20.0f32; vocab];
        raw[model.speech_token_size as usize] = 50.0;
        raw[7] = 10.0;
        let logits = Tensor::from_vec(raw, vocab, &device)?;
        let cfg = SamplingConfig {
            top_p: 0.9,
            top_k: 5,
            win_size: 10,
            tau_r: 0.1,
            seed: Some(42),
        };
        let mut rng = StepRng::new(42);
        let picked = model.sample(&logits, &[], &cfg, &mut rng, true)?;
        assert_eq!(picked, 7);
        let picked_eos = model.sample(&logits, &[], &cfg, &mut rng, false)?;
        assert_eq!(picked_eos, model.speech_token_size);
        Ok(())
    }
}
