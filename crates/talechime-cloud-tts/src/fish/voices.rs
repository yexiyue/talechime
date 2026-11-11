use super::client::bounded_json;
use super::{Client, Error, ReferenceAudio};
use serde::Deserialize;

/// Scope of a Fish voice search.
#[derive(Clone, Copy, Debug)]
pub enum VoiceScope {
    /// Public voice library.
    Public,
    /// Voices owned by the authenticated account.
    Own,
}

/// Voice library query, with one-based pages.
pub struct VoiceQuery {
    /// Public or account-owned voices.
    pub scope: VoiceScope,
    /// One-based page number.
    pub page_number: u32,
    /// Page size from 1 through 100.
    pub page_size: u32,
    /// Optional title filter.
    pub title: Option<String>,
    /// Optional tag filter.
    pub tag: Option<String>,
    /// Optional language filter.
    pub language: Option<String>,
}

/// A Fish remote voice; state is preserved even when unknown to this client.
#[derive(Debug, Deserialize)]
pub struct VoiceRecord {
    /// Provider-owned voice identifier.
    #[serde(rename = "_id", alias = "id")]
    pub id: String,
    /// Display title.
    pub title: String,
    /// Provider state, if supplied.
    pub state: Option<String>,
    /// Provider visibility, if supplied.
    pub visibility: Option<String>,
    /// Additional provider metadata.
    #[serde(flatten)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// A page of remote voices.
#[derive(Debug, Deserialize)]
pub struct VoicePage {
    /// Total reported by the service; provider pagination may limit access.
    pub total: u64,
    /// Returned voices.
    pub items: Vec<VoiceRecord>,
    /// Provider continuation indication, when available.
    pub has_more: Option<bool>,
    /// Additional pagination metadata.
    #[serde(flatten)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// Explicit request to create a private, persistent clone.
pub struct CreateVoice {
    /// Display title.
    pub title: String,
    /// One to twenty audio samples and their transcripts.
    pub samples: Vec<ReferenceAudio>,
}

impl Client {
    /// Search public or account-owned Fish voices.
    pub async fn search_voices(&self, query: &VoiceQuery) -> Result<VoicePage, Error> {
        if query.page_number == 0 || !(1..=100).contains(&query.page_size) {
            return Err(Error::InvalidInput("pagination"));
        }
        let mut params = vec![
            ("page_number", query.page_number.to_string()),
            ("page_size", query.page_size.to_string()),
        ];
        if matches!(query.scope, VoiceScope::Own) {
            params.push(("self", "true".into()));
        }
        for (name, value) in [
            ("title", &query.title),
            ("tag", &query.tag),
            ("language", &query.language),
        ] {
            if let Some(value) = value {
                params.push((name, value.clone()));
            }
        }
        let response = self
            .send(
                self.request(reqwest::Method::GET, "/model").query(&params),
                false,
            )
            .await?;
        bounded_json(response).await
    }

    /// Query a voice's current provider state and metadata.
    pub async fn get_voice(&self, id: &str) -> Result<VoiceRecord, Error> {
        let response = self.send(self.voice_request(id)?, false).await?;
        bounded_json(response).await
    }

    /// Upload samples and create a private remote voice. Never retries automatically.
    pub async fn create_voice(&self, voice: CreateVoice) -> Result<VoiceRecord, Error> {
        if voice.title.trim().is_empty() || !super::valid_samples(&voice.samples) {
            return Err(Error::InvalidInput("voice samples"));
        }
        let mut form = reqwest::multipart::Form::new()
            .text("type", "tts")
            .text("train_mode", "fast")
            .text("visibility", "private")
            .text("title", voice.title);
        for sample in voice.samples {
            form = form
                .part(
                    "voices",
                    reqwest::multipart::Part::bytes(sample.audio).file_name("reference.wav"),
                )
                .text("texts", sample.transcript);
        }
        let response = self
            .send(
                self.request(reqwest::Method::POST, "/model")
                    .multipart(form),
                true,
            )
            .await?;
        bounded_json(response)
            .await
            .map_err(|_| Error::OutcomeUnknown)
    }
}
