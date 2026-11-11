//! Fish Audio service types and client.

use std::{fmt, str::FromStr};

mod client;
mod speech;
mod voices;

pub use client::Client;
pub use speech::{AudioFormat, AudioStream, SpeechRequest, StreamState};
pub use voices::{CreateVoice, VoicePage, VoiceQuery, VoiceRecord, VoiceScope};

/// Supported models. The model must be selected explicitly for every request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    /// Free developer model, subject to the provider's current policies.
    S21ProFree,
    /// Paid S2.1 Pro model.
    S21Pro,
    /// Paid S2 Pro model.
    S2Pro,
    /// Paid Drama 3 model.
    Drama3,
}

impl Model {
    /// Exact value sent to the provider.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::S21ProFree => "s2.1-pro-free",
            Self::S21Pro => "s2.1-pro",
            Self::S2Pro => "s2-pro",
            Self::Drama3 => "drama-3",
        }
    }
}

impl FromStr for Model {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "s2.1-pro-free" => Ok(Self::S21ProFree),
            "s2.1-pro" => Ok(Self::S21Pro),
            "s2-pro" => Ok(Self::S2Pro),
            "drama-3" => Ok(Self::Drama3),
            _ => Err(Error::InvalidModel),
        }
    }
}

/// Secret bearer credential. Debug output never contains its value.
#[derive(Clone)]
pub struct ApiKey(String);

impl ApiKey {
    /// Validate and wrap a nonempty bearer credential.
    pub fn new(value: impl Into<String>) -> Result<Self, Error> {
        let value = value.into();
        if value.is_empty() || value.bytes().any(|b| b.is_ascii_control()) {
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

/// A reference sample and its exact transcript, supplied by the caller.
#[derive(Clone)]
pub struct ReferenceAudio {
    /// Encoded audio file bytes.
    pub audio: Vec<u8>,
    /// Words spoken in the audio.
    pub transcript: String,
}

fn valid_samples(samples: &[ReferenceAudio]) -> bool {
    !samples.is_empty()
        && samples.len() <= 20
        && samples
            .iter()
            .all(|sample| !sample.audio.is_empty() && !sample.transcript.trim().is_empty())
        && samples
            .iter()
            .try_fold(0usize, |total, sample| {
                total.checked_add(sample.audio.len())
            })
            .is_some_and(|total| total <= 64 * 1024 * 1024)
}

/// A single request's explicit source of voice identity.
#[derive(Clone)]
pub enum Voice {
    /// An ID belonging to Fish Audio, whose availability depends on the account.
    RemoteId(String),
    /// Reference audio sent for this request only.
    References(Vec<ReferenceAudio>),
}

/// Errors never contain provider response bodies, credentials, text, or audio.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The model is missing or not among the supported explicit values.
    #[error("unsupported Fish model")]
    InvalidModel,
    /// The credential is invalid.
    #[error("invalid Fish API credential")]
    InvalidCredential,
    /// Input validation failed.
    #[error("invalid request: {0}")]
    InvalidInput(&'static str),
    /// The provider rejected the credential.
    #[error("Fish authentication failed")]
    Authentication,
    /// The provider denied access (HTTP 403).
    #[error("Fish denied access")]
    Forbidden,
    /// The provider limited the request.
    #[error("Fish rate or concurrency limit reached")]
    RateLimited,
    /// An HTTP response other than authentication or rate limiting.
    #[error("Fish returned HTTP {0}")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_models_and_redacts_secrets() {
        assert!("s2.1-pro-fre".parse::<Model>().is_err());
        assert!("".parse::<Model>().is_err());
        let key = ApiKey::new("private-key").unwrap();
        assert!(!format!("{key:?}").contains("private-key"));
        assert!(!format!("{}", Error::InvalidModel).contains("s2.1-pro-fre"));
    }
}
