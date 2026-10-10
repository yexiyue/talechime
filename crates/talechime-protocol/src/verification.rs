//! Optional readback policy and transport-neutral evidence. No alignment timestamps.
use crate::TextRange;
use serde::{Deserialize, Serialize};

/// Disabled by default; a gate replaces only unpublished synthesis attempts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum VerificationPolicy {
    #[default]
    Off,
    ReportOnly,
    Gate {
        max_retries: u8,
        strict_suspect: bool,
    },
}

/// Independent bounds for segment collection and each recognizer call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VerificationOptions {
    pub policy: VerificationPolicy,
    pub timeout_ms: u64,
    pub max_segment_ms: u32,
    pub max_segment_bytes: usize,
}
impl Default for VerificationOptions {
    fn default() -> Self {
        Self {
            policy: VerificationPolicy::Off,
            timeout_ms: 30_000,
            max_segment_ms: 30_000,
            max_segment_bytes: 16 * 1024 * 1024,
        }
    }
}

/// Stable model and implementation identity, including the audio frontend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecognizerIdentity {
    pub family: String,
    pub model: String,
    pub revision: String,
    pub implementation: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationVerdict {
    Passed,
    Suspect,
    ConfirmedError,
    Unverified,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferenceKind {
    Missing,
    Extra,
    Substitution,
}
/// Original source UTF-8 range; insertions use a zero-width source boundary.
/// When speech preprocessing changes the source, range covers the actual segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadbackDifference {
    pub kind: DifferenceKind,
    pub range: TextRange,
    /// UTF-8 range in normalized spoken text, retained for exact consensus when
    /// the original-source range must fall back to the complete segment.
    pub normalized_range: TextRange,
    pub expected: String,
    pub observed: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadbackEvidence {
    pub recognizer: RecognizerIdentity,
    pub transcript: Option<String>,
    pub error: Option<String>,
    pub differences: Vec<ReadbackDifference>,
    /// Recognizer-side audio labels (emotion/event/language tags); empty when
    /// the recognizer does not produce them. Text comparison never uses them.
    #[serde(default)]
    pub labels: Vec<String>,
}
/// One actual synthesis attempt. A report never proves playback completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub text_hash: String,
    pub range: TextRange,
    pub spoken_text: String,
    pub backend: String,
    pub model: Option<String>,
    pub voice: String,
    pub style: Option<String>,
    pub audio_hash: String,
    pub attempt: u8,
    pub cache_hit: bool,
    pub rules: String,
    pub verdict: VerificationVerdict,
    pub evidence: Vec<ReadbackEvidence>,
}
