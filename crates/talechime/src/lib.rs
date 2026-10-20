//! Embed local synthesis and listening without CLI configuration or protocol plumbing.
//!
//! The host owns the runtime and thread. Call [`run_local`] inside that runtime,
//! prepare an [`Engine`] with explicit resources, and close owned executions.
//! Engines/streams remain local; [`ListeningHandle`] and [`CancellationHandle`]
//! carry Send controls without moving models or audio devices.
//!
//! ```no_run
//! use talechime::{Engine, ModelOptions, SynthesisState, run_local};
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! run_local(async {
//!     // Explicit preparation may download model resources to this directory.
//!     let mut engine = Engine::prepare(ModelOptions::new("moss", "./models"), |_| {}).await?;
//!     let voice = engine.capabilities()?.default_voice;
//!     let mut stream = engine.synthesize("你好。", &voice, None)?;
//!     while let Some(block) = stream.recv().await {
//!         let block = block?;
//!         println!("{} samples", block.pcm().samples.len());
//!     }
//!     assert_eq!(stream.state(), SynthesisState::Completed);
//!     engine.close().await?;
//!     Ok::<_, Box<dyn std::error::Error>>(())
//! }).await?;
//! # Ok(())
//! # }
//! ```
mod engine;
mod listening;
mod model_preparation;
mod voice_import;

pub use engine::{Engine, EngineError, ModelOptions, PcmStream, run_local};
pub use listening::{Listening, ListeningHandle, ListeningOptions, ListeningSession};
pub use talechime_core::{
    CancellationHandle, ChapterCacheOptions, DEFAULT_MAX_BYTES, GenerationParams, PlanError,
    PlanProgress, PlanSessionOptions, PlanState, Playback, PlaybackPolicy, SeedPolicy,
    SourceSnapshot, SpeechAudio, SpeechContext, SpeechPlan, SpeechSpan, StagingError,
    StagingOptions, SynthesisItem, SynthesisOptions, SynthesisState, VoiceSnapshot,
    backend::{AudioChunk, Backend, BackendError, Pcm, SegmentRequest, Segmentation, Streaming},
    session::{SessionError, SessionEvent},
};
pub use talechime_protocol::{
    Capabilities, Device, EndReason, Event, ParamKind, ParamValue, ParameterSpec, SessionState,
    SourceId, TextRange, text_hash,
};
pub use voice_import::{VoiceImportError, import_reference_voice, list_voices};

pub use talechime_core::verification::{
    DifferenceKind, ReadbackAudio, ReadbackDifference, ReadbackEvidence, ReadbackRequest,
    Recognition, RecognitionRequest, Recognizer, RecognizerIdentity, VerificationError,
    VerificationOptions, VerificationPolicy, VerificationReport, VerificationVerdict, Verifier,
};

/// Typed per-model generation parameter builders; available with each backend feature.
#[cfg(feature = "moss-candle")]
pub use talechime_backends::moss::candle::params::MossParams;
#[cfg(feature = "omnivoice")]
pub use talechime_backends::omnivoice::params::OmniVoiceParams;
#[cfg(feature = "qwen")]
pub use talechime_backends::qwen::params::QwenParams;
#[cfg(feature = "voxcpm")]
pub use talechime_backends::voxcpm::params::VoxCpmParams;

#[cfg(feature = "asr")]
pub use talechime_backends::asr::ReadbackModelOptions;
/// Explicitly prepare the selected native CPU ASR group and drain progress concurrently.
#[cfg(feature = "asr")]
pub async fn prepare_readback(
    options: ReadbackModelOptions,
    progress: impl FnMut(Event),
) -> Result<std::rc::Rc<Verifier>, EngineError> {
    model_preparation::with_progress(
        |tx| talechime_backends::asr::prepare(&options, tx),
        progress,
    )
    .await
}
