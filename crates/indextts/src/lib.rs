//! Native Candle inference for IndexTTS 2.5.
//!
//! This crate is a faithful port of the official Python implementation
//! (index-tts at the revision pinned in `SOURCE.md`). It contains only model
//! computation: loading converted safetensors checkpoints, conditioning on
//! reference audio, autoregressive semantic-token generation, the semantic to
//! mel flow-matching stage and the BigVGAN vocoder. Resource download, voice
//! stores and playback live in the adapter layer.
//!
//! Ordinary tests never download models; real-model numeric oracles are
//! `#[ignore]` tests driven by fixtures produced by `tools/tts`.

pub mod bigvgan;
pub mod campplus;
pub mod cfm;
pub mod codec;
pub mod config;
pub mod conformer;
pub mod dit;
pub mod dsp;
pub mod front;
pub mod generation;
pub mod gpt;
pub mod gpt2;
pub mod length_regulator;
pub mod mel;
pub mod model;
pub mod perceiver;
pub mod s2mel;
pub mod tokenizer;
pub mod validation;
pub mod w2v_bert;
pub mod wavenet;
pub mod weights;

/// Output sampling rate of the full IndexTTS 2.5 pipeline (BigVGAN vocoder).
pub const SAMPLE_RATE: u32 = 22_050;

/// Full-scale value used when converting between normalized waveforms and
/// 16-bit PCM; matches upstream `PCM16_MAX`.
pub const PCM16_MAX: f32 = 32_767.0;

pub use config::Config;
pub use model::{Model, Options, Outcome, Reference};
