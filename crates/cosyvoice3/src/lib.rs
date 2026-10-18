//! Fun-CosyVoice3-0.5B inference in Candle, with optional ORT reference
//! paths for the official ONNX artifacts.
//!
//! This crate is a straight port of the official PyTorch implementation
//! (FunAudioLLM/CosyVoice commit 074ca6dc) with the community Rust layout
//! as a starting skeleton; numeric deviations found during the audit were
//! fixed against the Python sources. See
//! `docs/records/cosyvoice3/cosyvoice3-candle-audit-2026-10-17.md` for the component
//! map. Local component/real-weight results, the remaining end-to-end
//! waveform gates (kept failing with f64-anchor root-cause evidence), and
//! the streaming implementation are recorded in
//! `docs/records/cosyvoice3/cosyvoice3-local-parity-2026-10-17.md`. Talechime backend
//! registration is available through the opt-in `cosyvoice3` adapter feature.
//!
//! Weights come from the conversion script
//! `tools/tts/cosyvoice3_convert_weights.py`: fp32 safetensors with
//! weight-norm fusion and key renames.

pub mod activations;
pub mod config;
pub mod flow;
pub mod frontend;
pub mod llm;
pub mod pipeline;
pub mod rng;
pub mod streaming;
pub mod vocoder;

pub use config::{
    CfmConfig, CosyVoice3Config, DiTConfig, FlowConfig, HiftConfig, LlmConfig, Qwen2Config,
    SPEECH_TOKEN_SIZE, SPEECH_VOCAB_SIZE, SamplingConfig,
};
pub use flow::{CausalMaskedDiffWithDiT, DiT};
pub use llm::CosyVoice3Lm;
pub use pipeline::{DEFAULT_PROMPT_PREFIX, Mode, PromptFeatures};
pub use streaming::{
    StreamChunk, StreamingTts, TOKEN_HOP_LEN, TOKEN_MAX_HOP_LEN, TokenStreamDecoder,
};
pub use vocoder::CausalHiFtGenerator;
