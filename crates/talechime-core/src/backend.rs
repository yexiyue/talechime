//! Synthesis boundary with model-independent PCM and capabilities.

use std::{future::Future, pin::Pin};
use tts_protocol::Capabilities;

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
    fn select_voice(&self, _voice: &str) {}
    fn next_segment<'a>(
        &'a self,
        text: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<crate::text::TextSegment>, BackendError>> + 'a>>
    {
        Box::pin(async move { Ok(self.segments(text).await?.into_iter().next()) })
    }
    /// Learn the current voice from a successful block, independently of playback speed.
    fn observe_duration(&self, _text: &str, _voice: &str, _seconds: f64) {}
    fn segments<'a>(&'a self, text: &'a str) -> Segmentation<'a> {
        Box::pin(async move { Ok(crate::text::preprocess_text(text, 200)) })
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
    Pin<Box<dyn Future<Output = Result<Vec<crate::text::TextSegment>, BackendError>> + 'a>>;
