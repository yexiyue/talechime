//! Synthesis boundary with model-independent PCM and capabilities.

use std::{future::Future, pin::Pin};
use talechime_protocol::Capabilities;

/// Backend failures do not expose inference-library types to consumers.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("model initialization failed: {0}")]
    Initialize(String),
    #[error("synthesis failed: {0}")]
    Synthesis(String),
    #[error("invalid audio or unsupported capability: {0}")]
    Unsupported(String),
}

/// Interleaved float PCM; channels and sample rate belong to the backend.
#[derive(Debug)]
pub struct Pcm {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl Pcm {
    /// Validate framing and return the rounded-up duration in milliseconds.
    pub fn duration_ms(&self) -> Result<u32, BackendError> {
        if self.sample_rate == 0
            || self.channels == 0
            || self.samples.is_empty()
            || !self.samples.len().is_multiple_of(self.channels as usize)
            || self.samples.iter().any(|sample| !sample.is_finite())
        {
            return Err(BackendError::Unsupported(
                "invalid PCM format or samples".into(),
            ));
        }
        let frames = self.samples.len() as u64 / self.channels as u64;
        u32::try_from((frames * 1000).div_ceil(self.sample_rate as u64))
            .map_err(|_| BackendError::Unsupported("audio duration overflow".into()))
    }
}

/// Local future: model objects and audio devices stay on their owning thread.
pub type Synthesis<'a> = Pin<Box<dyn Future<Output = Result<Pcm, BackendError>> + 'a>>;

/// One synthesis request crossing the backend boundary.
///
/// Style, continuation context, the concrete sampling seed and validated
/// generation parameters travel together; backends reject what their model
/// does not support instead of silently ignoring it.
pub struct SegmentRequest<'a> {
    pub text: &'a str,
    pub voice: &'a str,
    /// Optional per-utterance style instruction.
    pub style: Option<&'a str>,
    /// Optional previous complete utterance, owned by this execution rather than a voice cache.
    pub context: Option<&'a crate::SpeechContext>,
    /// Concrete sampling seed resolved by the session producer for this attempt.
    pub seed: u64,
    /// Host-supplied generation parameters, validated at session start.
    pub params: &'a crate::params::GenerationParams,
}

impl SegmentRequest<'_> {
    /// Reject a style or continuation conditioning the model does not support.
    pub fn reject_unsupported(&self, style: bool, continuation: bool) -> Result<(), BackendError> {
        if !style && self.style.is_some_and(|value| !value.trim().is_empty()) {
            return Err(BackendError::Unsupported(
                "this model does not support speaking style".into(),
            ));
        }
        if !continuation && self.context.is_some() {
            return Err(BackendError::Unsupported(
                "this model does not support continuation".into(),
            ));
        }
        Ok(())
    }
}

/// Domain interface shared by real and deterministic testing backends.
pub trait Backend {
    fn capabilities(&self) -> Capabilities;
    /// Stable identity of the prepared implementation, device, model and voice
    /// resources. Return `None` when reproducible cache reuse cannot be proven.
    /// Changing any synthesis resource must change this identity.
    fn cache_identity(&self) -> Option<String> {
        None
    }
    /// Primary synthesis boundary. Every successful segment terminates with End.
    fn stream<'a>(&'a self, request: SegmentRequest<'a>) -> Streaming<'a>;
    /// Collect a stream for offline export and model comparison.
    fn synthesize<'a>(&'a self, request: SegmentRequest<'a>) -> Synthesis<'a> {
        Box::pin(async move {
            let mut stream = self.stream(request).await?;
            let mut audio: Option<Pcm> = None;
            while let Some(chunk) = stream.recv().await {
                match chunk? {
                    AudioChunk::Pcm(pcm) => {
                        pcm.duration_ms()?;
                        if let Some(audio) = &mut audio {
                            if audio.sample_rate != pcm.sample_rate
                                || audio.channels != pcm.channels
                            {
                                return Err(BackendError::Synthesis(
                                    "PCM format changed inside segment".into(),
                                ));
                            }
                            audio.samples.extend(pcm.samples);
                        } else {
                            audio = Some(pcm);
                        }
                    }
                    AudioChunk::End => {
                        return audio
                            .ok_or_else(|| BackendError::Synthesis("empty audio stream".into()));
                    }
                }
            }
            Err(BackendError::Synthesis("incomplete audio stream".into()))
        })
    }
    /// Whether this block ends a semantic paragraph (not merely a layout line).
    fn paragraph_end(&self, segment: &str, remaining: &str) -> bool {
        segment.ends_with('\n') || remaining.trim().is_empty()
    }
    /// Segment only the supplied voice assignment using execution-owned estimates.
    fn next_segment<'a>(&'a self, request: SegmentationRequest<'a>) -> Segmentation<'a> {
        Box::pin(async move {
            Ok(crate::text::preprocess_text(request.text, 200)
                .into_iter()
                .next())
        })
    }
}

/// A segment ends explicitly; a disconnected producer is not successful completion.
#[derive(Debug)]
pub enum AudioChunk {
    Pcm(Pcm),
    End,
}
pub type AudioStream = tokio::sync::mpsc::Receiver<Result<AudioChunk, BackendError>>;
pub type Streaming<'a> = Pin<Box<dyn Future<Output = Result<AudioStream, BackendError>> + 'a>>;
pub type Segmentation<'a> =
    Pin<Box<dyn Future<Output = Result<Option<crate::text::TextSegment>, BackendError>> + 'a>>;

/// Explicit segmentation context; duration learning belongs to one execution.
#[derive(Clone, Copy)]
pub struct SegmentationRequest<'a> {
    pub text: &'a str,
    pub voice: &'a str,
    pub duration: &'a crate::text::duration::DurationEstimator,
}
