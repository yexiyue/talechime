use super::{Client, Error, Model};
use bytes::Bytes;
use serde::Serialize;

/// Supported raw-byte output profiles. Neither profile uses WAV or SSE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioFormat {
    /// Mono signed little-endian PCM16 at 24 kHz, without a header.
    Pcm16,
    /// Mono MP3 at 44.1 kHz, 128 kbps.
    Mp3,
}
impl AudioFormat {
    /// Requested sampling rate in Hz.
    pub fn sample_rate(self) -> u32 {
        match self {
            Self::Pcm16 => 24000,
            Self::Mp3 => 44100,
        }
    }
    fn content_type(self) -> &'static str {
        match self {
            Self::Pcm16 => "audio/pcm",
            Self::Mp3 => "audio/mpeg",
        }
    }
}
/// Explicit inputs; the client never selects another model or voice.
pub struct SpeechRequest {
    /// Required model, including an explicit free or paid tier.
    pub model: Model,
    /// Nonblank text, at most 5000 Unicode scalar values.
    pub text: String,
    /// Provider-owned voice identifier.
    pub voice_id: String,
    /// Explicit output profile.
    pub format: AudioFormat,
    /// Optional supported base language code, such as `zh`.
    pub language: Option<String>,
    /// Maximum cumulative audio bytes to deliver.
    pub max_audio_bytes: usize,
}
/// Audio properties verified against the request before any bytes are delivered.
#[derive(Clone, Debug)]
pub struct AudioMetadata {
    /// Server-confirmed MIME type.
    pub content_type: String,
    /// Server-confirmed sampling rate.
    pub sample_rate: u32,
    /// Server-confirmed channel count (one).
    pub channels: u16,
}
impl AudioMetadata {
    fn from_headers(
        headers: &reqwest::header::HeaderMap,
        requested: AudioFormat,
    ) -> Result<Self, Error> {
        for name in ["content-type", "x-audio-sample-rate", "x-audio-channels"] {
            if headers.get_all(name).iter().count() != 1 {
                return Err(Error::InvalidResponse);
            }
        }
        let content_type = headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(';').next())
            .map(str::trim)
            .ok_or(Error::InvalidResponse)?;
        let sample_rate = headers
            .get("X-Audio-Sample-Rate")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u32>().ok())
            .ok_or(Error::InvalidResponse)?;
        let channels = headers
            .get("X-Audio-Channels")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u16>().ok())
            .ok_or(Error::InvalidResponse)?;
        if !content_type.eq_ignore_ascii_case(requested.content_type())
            || sample_rate != requested.sample_rate()
            || channels != 1
        {
            return Err(Error::InvalidResponse);
        }
        Ok(Self {
            content_type: content_type.to_ascii_lowercase(),
            sample_rate,
            channels,
        })
    }
}

#[derive(Serialize)]
struct OutputFormat {
    encoding: &'static str,
    sample_rate_hertz: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    bitrate_bps: Option<u32>,
}
#[derive(Serialize)]
struct Body<'a> {
    model: &'static str,
    text: &'a str,
    voice_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
    output_format: OutputFormat,
    with_timestamps: bool,
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
    /// Audio properties confirmed by response headers.
    pub metadata: AudioMetadata,
    /// Requested sample rate in Hz.
    pub sample_rate: u32,
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
    /// Start a bounded raw audio response. No automatic retries or paid fallback.
    /// Transport completion does not prove content fidelity, playback, or absence of charges.
    pub async fn synthesize(&self, request: &SpeechRequest) -> Result<AudioStream, Error> {
        if request.text.trim().is_empty()
            || request.text.chars().count() > 5000
            || !super::valid_id(&request.voice_id)
            || request.max_audio_bytes == 0
            || request
                .language
                .as_deref()
                .is_some_and(|s| !super::valid_language(s))
        {
            return Err(Error::InvalidInput("speech request"));
        }
        let format = request.format;
        let body = Body {
            model: request.model.as_str(),
            text: &request.text,
            voice_id: &request.voice_id,
            language: request.language.as_deref(),
            with_timestamps: false,
            output_format: OutputFormat {
                encoding: match format {
                    AudioFormat::Pcm16 => "pcm",
                    AudioFormat::Mp3 => "mp3",
                },
                sample_rate_hertz: format.sample_rate(),
                bitrate_bps: (format == AudioFormat::Mp3).then_some(128000),
            },
        };
        let response = self
            .send(
                self.request(reqwest::Method::POST, "/v3/models/audio/tts/stream")
                    .header(reqwest::header::ACCEPT, format.content_type())
                    .json(&body),
                true,
            )
            .await?;
        let metadata = AudioMetadata::from_headers(response.headers(), format)?;
        Ok(AudioStream {
            response: Some(response),
            state: StreamState::Active,
            received: 0,
            limit: request.max_audio_bytes,
            format,
            sample_rate: format.sample_rate(),
            metadata,
        })
    }
}
