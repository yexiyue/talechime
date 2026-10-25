//! End-to-end non-streaming pipeline over the three Candle components.
//!
//! Mirrors `CosyVoice3Model.tts` in its single-shot (`finalize=True`)
//! branch: LLM speech tokens -> flow mel -> HiFT waveform. Speed uses the
//! official mel-domain linear interpolation of the full utterance.

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::{CosyVoice3Config, SamplingConfig};
use crate::flow::CausalMaskedDiffWithDiT;
use crate::frontend::audio::{MatchaMel, resample};
use crate::llm::CosyVoice3Lm;
use crate::vocoder::CausalHiFtGenerator;

/// Default instruction prefix the official examples pair with zero-shot
/// prompts; `<|endofprompt|>` is a hard requirement of the LLM input.
pub const DEFAULT_PROMPT_PREFIX: &str = "You are a helpful assistant.<|endofprompt|>";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Mode {
    /// prompt text + prompt speech tokens condition the LLM.
    #[default]
    ZeroShot,
    /// No prompt text / speech tokens in the LLM; flow still uses them.
    CrossLingual,
    /// Instruction text replaces the prompt text in the LLM.
    Instruct,
}

/// Prompt-side features, produced by the ONNX frontend or loaded from
/// pre-extracted safetensors.
pub struct PromptFeatures {
    /// Exact reference transcript, including the instruction prefix and end-of-prompt marker.
    pub text: String,
    pub speech_tokens: Vec<u32>,
    /// `[T * 2, 80]` prompt mel (already 50 Hz frames).
    pub mel: Vec<f32>,
    pub mel_frames: usize,
    /// 192-d speaker embedding.
    pub embedding: Vec<f32>,
}

pub struct CosyVoice3 {
    llm: CosyVoice3Lm,
    flow: CausalMaskedDiffWithDiT,
    hift: CausalHiFtGenerator,
    tokenizer: tokenizers::Tokenizer,
    config: CosyVoice3Config,
    device: candle_core::Device,
    dtype: DType,
    n_timesteps: usize,
}

impl CosyVoice3 {
    /// Load a converted model directory (`llm.safetensors`,
    /// `flow.safetensors`, `hift.safetensors`, `config.json`,
    /// `tokenizer/`). `seed` drives the vocoder's fixed noise buffers.
    pub fn load(
        dir: &std::path::Path,
        device: &candle_core::Device,
        dtype: DType,
        seed: u64,
    ) -> std::io::Result<Self> {
        let config = CosyVoice3Config::from_model_dir(dir)?;
        let tokenizer = load_tokenizer(&dir.join("tokenizer"))?;
        let llm_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("llm.safetensors")], dtype, device)
        }
        .map_err(io_msg)?;
        let flow_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("flow.safetensors")], dtype, device)
        }
        .map_err(io_msg)?;
        let hift_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("hift.safetensors")], dtype, device)
        }
        .map_err(io_msg)?;

        let llm = CosyVoice3Lm::new(&config.llm, llm_vb).map_err(io_msg)?;
        let dit = crate::flow::dit::DiT::new(config.flow.dit.clone(), flow_vb.pp("dit"))
            .map_err(io_msg)?;
        let flow =
            CausalMaskedDiffWithDiT::new(&config.flow, crate::config::SPK_EMBED_DIM, dit, flow_vb)
                .map_err(io_msg)?;
        let hift = CausalHiFtGenerator::new(config.hift.clone(), seed, hift_vb).map_err(io_msg)?;

        Ok(Self {
            llm,
            flow,
            hift,
            tokenizer,
            n_timesteps: config.flow.cfm.n_timesteps,
            config,
            device: device.clone(),
            dtype,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.config.sample_rate
    }

    pub fn config(&self) -> &CosyVoice3Config {
        &self.config
    }

    /// Borrow the loaded components for one zero-shot streaming request.
    /// Dropping the stream releases all per-request buffers; a new request
    /// resets the LLM cache and can reuse the same model weights.
    pub fn start_stream(
        &mut self,
        text: &str,
        prompt: &PromptFeatures,
        sampling: &SamplingConfig,
    ) -> Result<crate::StreamingTts<'_>> {
        self.start_stream_with_mode(text, prompt, Mode::ZeroShot, None, sampling)
    }

    /// Stream zero-shot, cross-lingual or instructed speech. In instructed
    /// mode, only the LLM speech prefix is removed; Flow/HiFT retain the
    /// original reference features and the bounded streaming schedule.
    pub fn start_stream_with_mode(
        &mut self,
        text: &str,
        prompt: &PromptFeatures,
        mode: Mode,
        instruct_text: Option<&str>,
        sampling: &SamplingConfig,
    ) -> Result<crate::StreamingTts<'_>> {
        let conditioning = llm_conditioning(prompt, mode, instruct_text)?;
        let has_marker = match mode {
            Mode::ZeroShot => prompt.text.contains("<|endofprompt|>"),
            _ => conditioning.text.contains("<|endofprompt|>") || text.contains("<|endofprompt|>"),
        };
        if text.trim().is_empty() || !has_marker {
            candle_core::bail!("nonempty text and an end-of-prompt marker are required");
        }
        if prompt.speech_tokens.is_empty()
            || prompt.mel_frames != prompt.speech_tokens.len() * 2
            || prompt.mel.len() != prompt.mel_frames * 80
            || prompt.embedding.len() != 192
            || prompt
                .mel
                .iter()
                .chain(&prompt.embedding)
                .any(|v| !v.is_finite())
        {
            candle_core::bail!("invalid zero-shot prompt features");
        }
        let text = u32_tensor(&self.encode(text)?, &self.device)?;
        let prompt_text = u32_tensor(&self.encode(conditioning.text)?, &self.device)?;
        let tokens = u32_tensor(&prompt.speech_tokens, &self.device)?;
        let llm_tokens = if mode == Mode::ZeroShot {
            tokens.clone()
        } else {
            u32_tensor(conditioning.speech_tokens, &self.device)?
        };
        let mel = Tensor::from_slice(&prompt.mel, (1, prompt.mel_frames, 80), &self.device)?
            .to_dtype(self.dtype)?;
        let embedding =
            Tensor::from_slice(&prompt.embedding, (1, 192), &self.device)?.to_dtype(self.dtype)?;
        crate::StreamingTts::borrowed(
            &mut self.llm,
            &mut self.flow,
            &mut self.hift,
            &text,
            &prompt_text,
            &llm_tokens,
            &tokens,
            &mel,
            &embedding,
            sampling,
        )
    }

    /// Full synthesis. Returns 24 kHz PCM f32.
    pub fn synthesize(
        &mut self,
        text: &str,
        prompt: &PromptFeatures,
        mode: Mode,
        instruct_text: Option<&str>,
        sampling: &SamplingConfig,
        speed: f32,
    ) -> Result<Vec<f32>> {
        if text.trim().is_empty() || !speed.is_finite() || speed <= 0.0 {
            candle_core::bail!("text must be nonempty and speed must be finite and positive");
        }
        if prompt.speech_tokens.is_empty()
            || prompt.mel_frames != prompt.speech_tokens.len() * 2
            || prompt.mel.len() != prompt.mel_frames * 80
            || prompt.embedding.len() != 192
        {
            candle_core::bail!(
                "prompt requires nonempty tokens, exactly two mel frames per token and a 192-d embedding"
            );
        }
        let text_tokens = self.encode(text)?;
        let conditioning = llm_conditioning(prompt, mode, instruct_text)?;
        let prompt_text = conditioning.text;
        // The official LLM requires this marker in either text input.
        if !text.contains("<|endofprompt|>") && !prompt_text.contains("<|endofprompt|>") {
            candle_core::bail!("<|endofprompt|> is required in text or prompt text");
        }
        let prompt_text_tokens = self.encode(prompt_text)?;

        let text_t = u32_tensor(&text_tokens, &self.device)?;
        let prompt_text_t = if prompt_text_tokens.is_empty() {
            Tensor::zeros((1, 0), DType::U32, &self.device)?
        } else {
            u32_tensor(&prompt_text_tokens, &self.device)?
        };
        let llm_prompt_tokens = u32_tensor(conditioning.speech_tokens, &self.device)?;

        let speech_tokens =
            self.llm
                .inference(&text_t, &prompt_text_t, &llm_prompt_tokens, sampling)?;
        if speech_tokens.is_empty() {
            candle_core::bail!("LLM produced no speech tokens");
        }
        let tokens_t = u32_tensor(&speech_tokens, &self.device)?;

        let prompt_tokens_t = u32_tensor(&prompt.speech_tokens, &self.device)?;
        let prompt_mel = Tensor::from_slice(&prompt.mel, (1, prompt.mel_frames, 80), &self.device)?
            .to_dtype(self.dtype)?;
        let embedding = Tensor::from_vec(prompt.embedding.clone(), (1, 192), &self.device)?
            .to_dtype(self.dtype)?;

        let mel = self.flow.inference(
            &tokens_t,
            &prompt_tokens_t,
            &prompt_mel,
            &embedding,
            self.n_timesteps,
            false,
            true,
        )?;
        let mel = mel_at_speed(&mel, speed)?;
        self.hift
            .inference(&mel, true)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .to_vec1()
    }

    /// Prompt features from a reference waveform (any sample rate >= 16k),
    /// for builds without the ORT frontend: the mel is always computable
    /// locally; tokens and embedding must come from elsewhere.
    pub fn prompt_mel(samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let device = candle_core::Device::Cpu;
        let audio = Tensor::from_vec(samples.to_vec(), samples.len(), &device)?;
        let audio_24k = if sample_rate != 24_000 {
            resample(&audio, sample_rate as usize, 24_000)?
        } else {
            audio
        };
        let mel = MatchaMel::new().forward(&audio_24k)?; // [1, 80, T]
        let mel_t = mel.transpose(1, 2)?.squeeze(0)?.to_vec2()?;
        let flat: Vec<f32> = mel_t.into_iter().flatten().collect();
        Ok(flat)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| candle_core::Error::Msg(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }
}

/// Extract prompt features through the official ONNX frontend.
#[cfg(feature = "ort")]
pub fn extract_prompt_features(
    frontend: &mut crate::frontend::ort_models::FrontendOnnx,
    samples: &[f32],
    sample_rate: u32,
    prompt_text: &str,
) -> Result<PromptFeatures> {
    let device = candle_core::Device::Cpu;
    let audio = Tensor::from_vec(samples.to_vec(), samples.len(), &device)?;
    let audio_16k = if sample_rate != 16_000 {
        resample(&audio, sample_rate as usize, 16_000)?
    } else {
        audio.clone()
    };
    let tokens_t = frontend.speech_tokens(&audio_16k)?;
    let speech_tokens: Vec<u32> = tokens_t.flatten_all()?.to_vec1()?;
    let embedding: Vec<f32> = frontend
        .speaker_embedding(&audio_16k)?
        .flatten_all()?
        .to_vec1()?;
    let mel = CosyVoice3::prompt_mel(samples, sample_rate)?;
    // Match frontend_zero_shot: floor the mel count before pairing.
    let token_len = speech_tokens.len().min(mel.len() / 80 / 2);
    let frames = 2 * token_len;
    Ok(PromptFeatures {
        text: prompt_text.to_owned(),
        speech_tokens: speech_tokens[..token_len].to_vec(),
        mel: mel[..frames * 80].to_vec(),
        mel_frames: frames,
        embedding,
    })
}

fn u32_tensor(v: &[u32], device: &candle_core::Device) -> Result<Tensor> {
    Tensor::from_slice(v, (1, v.len()), device)
}

fn io_msg(e: candle_core::Error) -> std::io::Error {
    std::io::Error::other(e.to_string())
}

/// Keep LLM conditioning selection identical for streaming and single-shot
/// generation. Flow always receives the complete original prompt separately.
struct LlmConditioning<'a> {
    text: &'a str,
    speech_tokens: &'a [u32],
}

fn llm_conditioning<'a>(
    prompt: &'a PromptFeatures,
    mode: Mode,
    instruction: Option<&'a str>,
) -> Result<LlmConditioning<'a>> {
    let (text, speech_tokens) = match mode {
        Mode::ZeroShot => (prompt.text.as_str(), prompt.speech_tokens.as_slice()),
        Mode::CrossLingual => ("", &[][..]),
        Mode::Instruct => (
            instruction
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| {
                    candle_core::Error::msg("instruct mode requires instruction text")
                })?,
            &[][..],
        ),
    };
    Ok(LlmConditioning {
        text,
        speech_tokens,
    })
}

fn load_tokenizer(dir: &std::path::Path) -> std::io::Result<tokenizers::Tokenizer> {
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| {
        std::io::Error::other(format!("load tokenizer.json (run conversion first): {e}"))
    })?;
    if tokenizer.token_to_id("<|endofprompt|>") != Some(151646) {
        return Err(std::io::Error::other(
            "tokenizer must encode <|endofprompt|> as 151646",
        ));
    }
    Ok(tokenizer)
}

/// Official non-streaming speed control interpolates mel frames before HiFT.
pub fn mel_at_speed(mel: &Tensor, speed: f32) -> Result<Tensor> {
    if !speed.is_finite() || speed <= 0.0 {
        candle_core::bail!("speed must be finite and positive");
    }
    if speed == 1.0 {
        return Ok(mel.clone());
    }
    let frames = (mel.dim(2)? as f64 / f64::from(speed)) as usize;
    if frames == 0 {
        candle_core::bail!("speed produces an empty mel sequence");
    }
    crate::vocoder::source_module::interp_linear_time(&mel.transpose(1, 2)?, frames)?
        .transpose(1, 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_separate_llm_and_acoustic_prompt_conditioning() -> Result<()> {
        let prompt = PromptFeatures {
            text: format!("{DEFAULT_PROMPT_PREFIX}原始参考文字"),
            speech_tokens: vec![4, 5],
            mel: vec![0.; 320],
            mel_frames: 4,
            embedding: vec![0.; 192],
        };
        let zero_shot = llm_conditioning(&prompt, Mode::ZeroShot, None)?;
        assert_eq!(zero_shot.text, prompt.text);
        assert_eq!(zero_shot.speech_tokens, &[4, 5]);
        let instruction = "You are a helpful assistant. 请用悲伤的语气说。<|endofprompt|>";
        let instructed = llm_conditioning(&prompt, Mode::Instruct, Some(instruction))?;
        assert_eq!(instructed.text, instruction);
        assert!(instructed.speech_tokens.is_empty());
        let cross_lingual = llm_conditioning(&prompt, Mode::CrossLingual, None)?;
        assert!(cross_lingual.text.is_empty() && cross_lingual.speech_tokens.is_empty());
        assert!(llm_conditioning(&prompt, Mode::Instruct, None).is_err());
        assert!(llm_conditioning(&prompt, Mode::Instruct, Some("  ")).is_err());
        // The original Flow prompt remains available unchanged in all modes.
        assert_eq!(prompt.speech_tokens, vec![4, 5]);
        assert_eq!(prompt.mel_frames, 4);
        Ok(())
    }

    #[test]
    fn speed_two_halves_mel_length() -> Result<()> {
        let mel = Tensor::arange(0f32, 100.0, &candle_core::Device::Cpu)?.reshape((1, 1, 100))?;
        let out = mel_at_speed(&mel, 2.0)?;
        assert_eq!(out.dims(), &[1, 1, 50]);
        assert!((out.flatten_all()?.to_vec1::<f32>()?[10] - 20.5).abs() < 1e-6);
        for speed in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(mel_at_speed(&mel, speed).is_err());
        }
        Ok(())
    }
}
