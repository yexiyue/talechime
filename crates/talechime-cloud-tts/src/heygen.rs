//! HeyGen Audio service types and client.

use std::{fmt, str::FromStr};

mod client;
mod speech;
mod voices;

pub use client::Client;
pub use speech::{AudioFormat, AudioMetadata, AudioStream, SpeechRequest, StreamState};
pub use voices::{VoicePage, VoiceQuery, VoiceRecord, VoiceScope};

/// Supported models. The model must be selected explicitly for every request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// Standard paid voice model.
    Voice1,
    /// Paid Turbo model.
    Turbo,
    /// Temporary free Turbo tier; availability follows provider policy.
    TurboFree,
}

impl Model {
    /// Exact value sent to the provider.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Voice1 => "heygen-voice-1",
            Self::Turbo => "heygen-voice-1-turbo",
            Self::TurboFree => "heygen-voice-1-turbo-free",
        }
    }
}

impl FromStr for Model {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "heygen-voice-1" => Ok(Self::Voice1),
            "heygen-voice-1-turbo" => Ok(Self::Turbo),
            "heygen-voice-1-turbo-free" => Ok(Self::TurboFree),
            _ => Err(Error::InvalidModel),
        }
    }
}

/// Secret API credential. Debug output never contains its value.
#[derive(Clone)]
pub struct ApiKey(String);

impl ApiKey {
    /// Validate and wrap a nonempty API credential.
    pub fn new(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        if value.trim().is_empty()
            || !value.is_ascii()
            || value.bytes().any(|b| b.is_ascii_control())
        {
            return Err(Error::InvalidCredential);
        }
        Ok(Self(value))
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey([REDACTED])")
    }
}

/// Errors never contain provider response bodies, credentials, text, or audio.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The model is missing or not among the supported explicit values.
    #[error("unsupported HeyGen model")]
    InvalidModel,
    /// The credential is invalid.
    #[error("invalid HeyGen API credential")]
    InvalidCredential,
    /// Input validation failed.
    #[error("invalid request: {0}")]
    InvalidInput(&'static str),
    /// The provider rejected the credential.
    #[error("HeyGen authentication failed")]
    Authentication,
    /// The provider denied access (HTTP 403).
    #[error("HeyGen denied access")]
    Forbidden,
    /// The provider limited the request.
    #[error("HeyGen rate or concurrency limit reached")]
    RateLimited,
    /// An HTTP response other than authentication or rate limiting.
    #[error("HeyGen returned HTTP {0}")]
    Http(u16),
    /// A request with potential side effects may have reached the service.
    #[error("request outcome is unknown")]
    OutcomeUnknown,
    /// A request or response body failed in transport.
    #[error("network transport failed")]
    Transport,
    /// The caller cancelled the local response.
    #[error("response cancelled")]
    Cancelled,
    /// The response was invalid or exceeded the configured bound.
    #[error("invalid or oversized response")]
    InvalidResponse,
}

fn valid_id(id: &str) -> bool {
    !id.trim().is_empty() && id.len() <= 256 && id != "." && id != ".."
}

fn valid_language(language: &str) -> bool {
    matches!(
        language,
        "ar" | "be"
            | "bg"
            | "ca"
            | "cs"
            | "da"
            | "de"
            | "el"
            | "en"
            | "es"
            | "fa"
            | "fi"
            | "fr"
            | "he"
            | "hi"
            | "hr"
            | "hu"
            | "id"
            | "it"
            | "ja"
            | "ko"
            | "mk"
            | "ms"
            | "nl"
            | "pl"
            | "pt"
            | "ro"
            | "ru"
            | "sk"
            | "sl"
            | "sr"
            | "sv"
            | "ta"
            | "th"
            | "tl"
            | "fil"
            | "tr"
            | "uk"
            | "vi"
            | "zh"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn models_and_credentials_are_explicit_and_redacted() {
        for model in [Model::Voice1, Model::Turbo, Model::TurboFree] {
            assert_eq!(model.as_str().parse::<Model>().unwrap(), model);
        }
        for name in ["", "unknown"] {
            assert!(name.parse::<Model>().is_err());
        }
        for key in ["", "  ", "x\ny", "非ascii"] {
            assert!(ApiKey::new(key).is_err());
        }
        assert!(!format!("{:?}", ApiKey::new("test-secret").unwrap()).contains("test-secret"));
    }
}
