use super::client::bounded_json;
use super::{Client, Error};
use serde::Deserialize;

/// Directory ownership scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceScope {
    /// Provider public catalog.
    Public,
    /// Voices belonging to the caller's workspace.
    Private,
}
/// One explicit page. Reuse filters when passing the next token.
pub struct VoiceQuery {
    /// Catalog scope.
    pub scope: VoiceScope,
    /// Page size, from 1 through 50.
    pub limit: u32,
    /// Opaque continuation token.
    pub token: Option<String>,
    /// Public-only base language filter.
    pub language: Option<String>,
    /// Public-only `male` or `female` filter.
    pub gender: Option<String>,
    /// Public-only `young`, `middle_aged` or `old` filter.
    pub age: Option<String>,
    /// Public-only free-text filter (1–256 Unicode scalar values).
    pub query: Option<String>,
}
/// Provider voice identity. Unknown states and fields are retained without interpretation.
#[derive(Debug, Deserialize)]
pub struct VoiceRecord {
    /// Remote identifier; not a local reference asset.
    pub voice_id: String,
    /// Provider display name.
    pub name: String,
    /// Provider voice mode.
    pub mode: String,
    /// Unmodified provider status; only the service can determine availability.
    pub status: String,
    /// Language if supplied.
    pub language: Option<String>,
    /// Preview URL if supplied. This client does not fetch it.
    pub preview_audio_url: Option<String>,
    /// Other provider fields, including failure information.
    #[serde(flatten)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}
/// One provider page; no automatic traversal occurs.
#[derive(Debug, Deserialize)]
pub struct VoicePage {
    /// Returned voices.
    pub data: Vec<VoiceRecord>,
    /// Whether more pages exist.
    pub has_more: bool,
    /// Opaque continuation cursor.
    pub next_token: Option<String>,
    /// Other pagination metadata.
    #[serde(flatten)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}
impl Client {
    /// Fetch a bounded page of voices. Public filters are rejected for private queries.
    pub async fn search_voices(&self, query: &VoiceQuery) -> Result<VoicePage, Error> {
        let filters = [
            ("language", &query.language),
            ("gender", &query.gender),
            ("age", &query.age),
            ("query", &query.query),
        ];
        if !(1..=50).contains(&query.limit)
            || query.token.as_ref().is_some_and(|s| s.trim().is_empty())
            || (query.scope == VoiceScope::Private && filters.iter().any(|(_, v)| v.is_some()))
            || query
                .language
                .as_deref()
                .is_some_and(|s| !super::valid_language(s))
            || query
                .gender
                .as_deref()
                .is_some_and(|s| !matches!(s, "male" | "female"))
            || query
                .age
                .as_deref()
                .is_some_and(|s| !matches!(s, "young" | "middle_aged" | "old"))
            || query
                .query
                .as_ref()
                .is_some_and(|s| s.trim().is_empty() || s.chars().count() > 256)
        {
            return Err(Error::InvalidInput("voice query"));
        }
        let mut params = vec![
            (
                "type",
                match query.scope {
                    VoiceScope::Public => "public".into(),
                    VoiceScope::Private => "private".into(),
                },
            ),
            ("limit", query.limit.to_string()),
        ];
        for (name, value) in filters.into_iter().chain([("token", &query.token)]) {
            if let Some(value) = value {
                params.push((name, value.clone()));
            }
        }
        let response = self
            .send(
                self.request(reqwest::Method::GET, "/v3/models/audio/voices")
                    .query(&params),
                false,
            )
            .await?;
        bounded_json(response).await
    }
    /// Fetch current remote voice state; never substitutes another voice.
    pub async fn get_voice(&self, id: &str) -> Result<VoiceRecord, Error> {
        #[derive(Deserialize)]
        struct Envelope {
            data: VoiceRecord,
        }
        let response = self.send(self.voice_request(id)?, false).await?;
        let envelope: Envelope = bounded_json(response).await?;
        Ok(envelope.data)
    }
}
