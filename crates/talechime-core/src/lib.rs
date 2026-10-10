//! Text-to-speech session core. Models and output devices stay with their owners;
//! consumers use model-independent PCM, controls and original-text byte events.
//!
//! Run sessions inside a Tokio `LocalSet`; concrete models live in talechime-backends.

pub mod audio;
pub mod backend;
pub mod checkpoint;
pub mod config;
mod continuation;
pub use continuation::SpeechContext;
pub mod download;
mod error;
pub mod params;
pub mod paths;
mod plan;
pub mod player;
pub mod session;
mod storage;
pub mod text;
pub mod verification;
pub mod voices;

pub use error::{ResourceError, Result};
pub use params::{GenerationParams, SeedPolicy};
pub use plan::{
    PlanError, PlanState, PlaybackPolicy, SourceSnapshot, SpeechPlan, SpeechSpan, VoiceSnapshot,
};
pub use player::AudioPlayer;
pub use player::Playback;
pub use session::{
    CancellationHandle, PlanProgress, PlanSessionOptions, SpeechAudio, StagingError,
    StagingOptions, SynthesisItem, SynthesisOptions, SynthesisState, SynthesisStream,
};
