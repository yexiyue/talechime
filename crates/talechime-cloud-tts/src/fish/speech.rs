use super::{Client, Error, Model, Voice};
use bytes::Bytes;
use serde::Serialize;

/// Encoding requested from Fish Audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// MPEG audio.
    Mp3,
    /// WAV container.
    Wav,
    /// Raw PCM; callers must establish its sample representation separately.
    Pcm,
    /// Opus encoded audio.
    Opus,
}

/// Explicit inputs for a single synthesis request.
pub struct SpeechRequest {
    /// Model selection is required; no fallback is applied.
    pub model: Model,
    /// Text to synthesize.
    pub text: String,
    /// Remote voice or inline references.
    pub voice: Voice,
    /// Requested audio format.
    pub format: AudioFormat,
    /// Optional requested sample rate; `None` uses the provider default.
    pub sample_rate: Option<u32>,
    /// Maximum response bytes to accept.
    pub max_audio_bytes: usize,
}

#[derive(Serialize)]
struct InlineReference<'a> {
    #[serde(with = "serde_bytes")]
    audio: &'a [u8],
    text: &'a str,
}

#[derive(Serialize)]
struct Body<'a> {
    text: &'a str,
    format: AudioFormat,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    references: Option<Vec<InlineReference<'a>>>,
}

/// Local state of a response stream, independent of playback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamState {
    /// Response is still being consumed.
    Active,
    /// The HTTP body ended normally after nonempty audio.
    Completed,
    /// The response was empty, too large, or interrupted.
    Failed,
    /// The caller explicitly cancelled the response.
    Cancelled,
}

/// Pull-based response stream. Dropping it closes the local HTTP response.
pub struct AudioStream {
    response: Option<reqwest::Response>,
    state: StreamState,
    received: usize,
    limit: usize,
    /// Requested format; network chunks do not correspond to audio frames.
    pub format: AudioFormat,
    /// Requested sample rate, or `None` when the server chooses it.
    pub sample_rate: Option<u32>,
}

impl AudioStream {
    /// Current transport state. Completion does not prove playback or content fidelity.
    pub fn state(&self) -> StreamState {
        self.state
    }

    /// Close the local response; remote work or billing may continue.
    pub fn cancel(&mut self) {
        if self.state == StreamState::Active {
            self.finish(StreamState::Cancelled);
        }
    }

    fn finish(&mut self, state: StreamState) {
        self.response = None;
        self.state = state;
    }

    /// Read one network chunk. An error is terminal; `None` means normal completion.
    pub async fn next_chunk(&mut self) -> Result<Option<Bytes>, Error> {
        let Some(response) = self.response.as_mut() else {
            return match self.state {
                StreamState::Completed => Ok(None),
                StreamState::Cancelled => Err(Error::Cancelled),
                _ => Err(Error::InvalidResponse),
            };
        };
        match response.chunk().await {
            Ok(Some(chunk)) if chunk.len() <= self.limit.saturating_sub(self.received) => {
                self.received += chunk.len();
                Ok(Some(chunk))
            }
            Ok(None) if self.received > 0 => {
                self.finish(StreamState::Completed);
                Ok(None)
            }
            Err(_) => {
                self.finish(StreamState::Failed);
                Err(Error::Transport)
            }
            _ => {
                self.finish(StreamState::Failed);
                Err(Error::InvalidResponse)
            }
        }
    }
}

impl Client {
    /// Start synthesis without buffering the resulting audio in memory.
    pub async fn synthesize(&self, request: &SpeechRequest) -> Result<AudioStream, Error> {
        if request.text.trim().is_empty()
            || request.text.len() > 4 * 1024 * 1024
            || request.max_audio_bytes == 0
        {
            return Err(Error::InvalidInput("text or audio limit"));
        }
        let (reference_id, references) = match &request.voice {
            Voice::RemoteId(id) if !id.trim().is_empty() => (Some(id.as_str()), None),
            Voice::References(samples) if super::valid_samples(samples) => (
                None,
                Some(
                    samples
                        .iter()
                        .map(|s| InlineReference {
                            audio: &s.audio,
                            text: &s.transcript,
                        })
                        .collect(),
                ),
            ),
            _ => return Err(Error::InvalidInput("voice")),
        };
        let body = Body {
            text: &request.text,
            format: request.format,
            sample_rate: request.sample_rate,
            reference_id,
            references,
        };
        let builder = self
            .request(reqwest::Method::POST, "/v1/tts")
            .header("model", request.model.as_str());
        let builder = if body.references.is_some() {
            let encoded =
                rmp_serde::to_vec_named(&body).map_err(|_| Error::InvalidInput("references"))?;
            builder
                .header(reqwest::header::CONTENT_TYPE, "application/msgpack")
                .body(encoded)
        } else {
            builder.json(&body)
        };
        let response = self.send(builder, true).await?;
        Ok(AudioStream {
            response: Some(response),
            state: StreamState::Active,
            received: 0,
            limit: request.max_audio_bytes,
            format: request.format,
            sample_rate: request.sample_rate,
        })
    }
}
