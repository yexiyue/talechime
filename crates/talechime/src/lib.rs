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

pub use engine::{Engine, EngineError, ModelOptions, PcmStream, run_local};
pub use listening::{Listening, ListeningHandle, ListeningOptions, ListeningSession};
pub use tts_core::{
    CancellationHandle, PlanError, PlanProgress, PlanSessionOptions, PlanState, Playback,
    PlaybackPolicy, SourceSnapshot, SpeechAudio, SpeechContext, SpeechPlan, SpeechSpan,
    StagingError, StagingOptions, SynthesisItem, SynthesisOptions, SynthesisState, VoiceSnapshot,
    backend::{AudioChunk, Backend, BackendError, Pcm, Segmentation, Streaming},
    session::{SessionError, SessionEvent},
};
pub use tts_protocol::{
    Capabilities, Device, EndReason, Event, SessionState, SourceId, TextRange, text_hash,
};

pub use tts_core::verification::{
    DifferenceKind, ReadbackAudio, ReadbackDifference, ReadbackEvidence, ReadbackRequest,
    Recognition, RecognitionRequest, Recognizer, RecognizerIdentity, VerificationError,
    VerificationOptions, VerificationPolicy, VerificationReport, VerificationVerdict, Verifier,
};

#[cfg(feature = "asr")]
pub use tts_backends::asr::ReadbackModelOptions;
/// Explicitly prepare the selected native CPU ASR group and drain progress concurrently.
#[cfg(feature = "asr")]
pub async fn prepare_readback(
    options: ReadbackModelOptions,
    progress: impl FnMut(Event),
) -> Result<std::rc::Rc<Verifier>, EngineError> {
    model_preparation::with_progress(|tx| tts_backends::asr::prepare(&options, tx), progress).await
}
