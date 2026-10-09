//! Validated, append-only speech plans, independent of execution and transport.
#![deny(missing_docs)]

use std::{collections::BTreeSet, sync::Arc};
use tts_protocol::{Capabilities, SourceId, TextRange, text_hash};

use crate::config::{ConfigError, validate_style, validate_voice};

/// Validation failures never partially accept an append batch or close a plan.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PlanError {
    /// The supplied digest does not identify the exact supplied text.
    #[error("source digest mismatch")]
    DigestMismatch,
    /// A voice snapshot must identify the prepared model exactly.
    #[error("voice snapshot does not match the prepared backend/model")]
    ModelMismatch,
    /// At least one permitted voice must be selected.
    #[error("voice snapshot is empty")]
    EmptyVoices,
    /// Voice IDs must be unique within a snapshot.
    #[error("duplicate voice: {0}")]
    DuplicateVoice(String),
    /// A voice or style violates the same rules as listening configuration.
    #[error(transparent)]
    Settings(#[from] ConfigError),
    /// Appended batches must contain at least one span.
    #[error("append batch is empty")]
    EmptyBatch,
    /// A span must be nonempty and bounded by UTF-8 character boundaries.
    #[error("invalid source range: {0:?}")]
    InvalidRange(TextRange),
    /// A recovery position must be within the accepted UTF-8 prefix.
    #[error("resume position {byte} is outside the accepted UTF-8 prefix ending at {accepted_end}")]
    InvalidResume {
        /// Requested original-text byte.
        byte: usize,
        /// End of accepted input.
        accepted_end: usize,
    },
    /// Ranges must extend the accepted prefix without gaps or overlaps.
    #[error("expected range start {expected}, received {actual}")]
    NonContiguous {
        /// End of the previously accepted prefix.
        expected: usize,
        /// Start of the rejected span.
        actual: usize,
    },
    /// A span cannot use a voice outside the fixed snapshot.
    #[error("voice is not in this plan's snapshot: {0}")]
    VoiceNotSelected(String),
    /// Full source coverage is required before sealing.
    #[error("source coverage is incomplete: {accepted_end} of {source_len} bytes")]
    Incomplete {
        /// Accepted source prefix in bytes.
        accepted_end: usize,
        /// Full source length in bytes.
        source_len: usize,
    },
    /// Sealed and failed plans cannot be mutated, including repeated sealing.
    #[error("plan is not open: {0:?}")]
    NotOpen(PlanState),
}

/// Immutable exact text, bound to a source identity and verified SHA-256 digest.
///
/// No newline or Unicode normalization is performed. A host must provide the
/// same snapshot to analysis, planning and highlighting.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceSnapshot {
    source: SourceId,
    text: Arc<str>,
    hash: String,
}

impl SourceSnapshot {
    /// Verify the supplied digest and retain the exact UTF-8 text.
    ///
    /// Returns [`PlanError::DigestMismatch`] if the digest differs from
    /// [`text_hash`] (including its lowercase hexadecimal representation).
    pub fn new(source: SourceId, text: impl Into<Arc<str>>, hash: &str) -> Result<Self, PlanError> {
        let text = text.into();
        if text_hash(&text) != hash {
            return Err(PlanError::DigestMismatch);
        }
        Ok(Self {
            source,
            text,
            hash: hash.into(),
        })
    }

    /// Source identity supplied by the host.
    pub fn source(&self) -> &SourceId {
        &self.source
    }

    /// Exact full source, including whitespace and original line endings.
    pub fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn shared_text(&self) -> Arc<str> {
        self.text.clone()
    }

    /// Verified digest of [`Self::text`].
    pub fn hash(&self) -> &str {
        &self.hash
    }
}

/// Fixed allowed voice IDs and capability metadata for one prepared model.
///
/// Construct from the prepared backend's capabilities, not a different model's
/// catalog entry. This is a metadata snapshot, not a lease on reference files:
/// resource revision/lifetime protection belongs to the future executor.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceSnapshot {
    capabilities: Capabilities,
    voices: BTreeSet<String>,
}

impl VoiceSnapshot {
    /// Validate a nonempty, unique set of voice IDs against the prepared model.
    ///
    /// Model identity is exact: unlike default-model configuration lookup,
    /// `None` does not match a capability record with an explicit model ID.
    pub fn new(
        backend: &str,
        model: Option<&str>,
        capabilities: &Capabilities,
        voices: Vec<String>,
    ) -> Result<Self, PlanError> {
        if capabilities.backend != backend || capabilities.model.as_deref() != model {
            return Err(PlanError::ModelMismatch);
        }
        if voices.is_empty() {
            return Err(PlanError::EmptyVoices);
        }
        let mut selected = BTreeSet::new();
        for voice in voices {
            validate_voice(&voice, capabilities)?;
            if !selected.insert(voice.clone()) {
                return Err(PlanError::DuplicateVoice(voice));
            }
        }
        Ok(Self {
            capabilities: capabilities.clone(),
            voices: selected,
        })
    }

    /// Prepared backend identity.
    pub fn backend(&self) -> &str {
        &self.capabilities.backend
    }

    /// Prepared model identity, retaining `None` for a legacy default model.
    pub fn model(&self) -> Option<&str> {
        self.capabilities.model.as_deref()
    }

    /// Allowed voice IDs, in deterministic lexical order.
    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.voices.iter().map(String::as_str)
    }

    fn validate(&self, span: &SpeechSpan) -> Result<(), PlanError> {
        if !self.voices.contains(&span.voice) {
            return Err(PlanError::VoiceNotSelected(span.voice.clone()));
        }
        validate_style(span.style.as_deref(), &self.capabilities)?;
        Ok(())
    }

    fn validate_capabilities(&self, capabilities: &Capabilities) -> Result<(), PlanError> {
        if self.backend() != capabilities.backend || self.model() != capabilities.model.as_deref() {
            return Err(PlanError::ModelMismatch);
        }
        if self.capabilities.style != capabilities.style {
            return Err(ConfigError::Invalid(
                "style capability changed; rebuild the voice snapshot".into(),
            )
            .into());
        }
        for voice in &self.voices {
            validate_voice(voice, capabilities)?;
        }
        Ok(())
    }
}

/// A proposed voice assignment; validated only when accepted by [`SpeechPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSpan {
    range: TextRange,
    voice: String,
    style: Option<String>,
}

impl SpeechSpan {
    /// Build a proposed assignment without assuming a particular source/model.
    pub fn new(range: TextRange, voice: impl Into<String>, style: Option<String>) -> Self {
        Self {
            range,
            voice: voice.into(),
            style,
        }
    }

    /// Source byte range, valid and nonempty once accepted by a plan.
    pub fn range(&self) -> TextRange {
        self.range
    }

    /// Voice ID in the plan's fixed voice snapshot.
    pub fn voice(&self) -> &str {
        &self.voice
    }

    /// Optional style; `None` means no instruction, not inherited global config.
    pub fn style(&self) -> Option<&str> {
        self.style.as_deref()
    }
}

/// When the future executor is allowed to start playback.
/// This metadata does not itself generate or play audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackPolicy {
    /// Play after prebuffering, while accepting more source ranges.
    Streaming,
    /// Play only after sealing and successful generation of the whole chapter.
    AfterChapterReady,
}

/// Input state, independent of generation and actual playback completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanState {
    /// More contiguous ranges may be appended.
    Open,
    /// All source ranges are accepted; this does not mean audio is complete.
    Sealed,
    /// The host explicitly terminated input; accepted ranges remain inspectable.
    Failed,
}

/// Append-only, validated source prefix with a fixed model and voice snapshot.
/// No model, runtime, file, player or transport is created by this type.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechPlan {
    source: SourceSnapshot,
    voices: VoiceSnapshot,
    policy: PlaybackPolicy,
    spans: Vec<SpeechSpan>,
    accepted_end: usize,
    state: PlanState,
}

impl SpeechPlan {
    /// Open a plan over a verified source and validated voice snapshot.
    pub fn new(source: SourceSnapshot, voices: VoiceSnapshot, policy: PlaybackPolicy) -> Self {
        Self {
            source,
            voices,
            policy,
            spans: Vec::new(),
            accepted_end: 0,
            state: PlanState::Open,
        }
    }

    /// Accept a complete batch atomically; any validation error leaves the plan unchanged.
    pub fn append(&mut self, batch: Vec<SpeechSpan>) -> Result<(), PlanError> {
        self.require_open()?;
        if batch.is_empty() {
            return Err(PlanError::EmptyBatch);
        }
        let mut end = self.accepted_end;
        for span in &batch {
            if !span.range.is_valid(self.source.text()) || span.range.start == span.range.end {
                return Err(PlanError::InvalidRange(span.range));
            }
            if span.range.start != end {
                return Err(PlanError::NonContiguous {
                    expected: end,
                    actual: span.range.start,
                });
            }
            self.voices.validate(span)?;
            end = span.range.end;
        }
        self.spans.extend(batch);
        self.accepted_end = end;
        Ok(())
    }

    /// Close complete input; early or repeated sealing is rejected without mutation.
    pub fn seal(&mut self) -> Result<(), PlanError> {
        self.require_open()?;
        if self.accepted_end != self.source.text().len() {
            return Err(PlanError::Incomplete {
                accepted_end: self.accepted_end,
                source_len: self.source.text().len(),
            });
        }
        self.state = PlanState::Sealed;
        Ok(())
    }

    /// Explicitly terminate open input. Failure details belong to the host/executor.
    /// Already sealed or failed input cannot be failed again.
    pub fn fail(&mut self) -> Result<(), PlanError> {
        self.require_open()?;
        self.state = PlanState::Failed;
        Ok(())
    }

    /// Accept and seal a full chapter using the same incremental validation path.
    pub fn complete(
        source: SourceSnapshot,
        voices: VoiceSnapshot,
        policy: PlaybackPolicy,
        spans: Vec<SpeechSpan>,
    ) -> Result<Self, PlanError> {
        let mut plan = Self::new(source, voices, policy);
        if !spans.is_empty() {
            plan.append(spans)?;
        }
        plan.seal()?;
        Ok(plan)
    }

    /// Assign a single voice/style to the full chapter and seal it.
    /// Even for an empty chapter, the requested voice/style must be valid.
    pub fn single_voice(
        source: SourceSnapshot,
        voices: VoiceSnapshot,
        policy: PlaybackPolicy,
        voice: &str,
        style: Option<String>,
    ) -> Result<Self, PlanError> {
        let span = SpeechSpan::new(
            TextRange {
                start: 0,
                end: source.text().len(),
            },
            voice,
            style,
        );
        voices.validate(&span)?;
        let spans = if source.text().is_empty() {
            Vec::new()
        } else {
            vec![span]
        };
        Self::complete(source, voices, policy, spans)
    }

    /// Validate an explicit recovery byte without preparing a model or opening audio.
    pub fn validate_resume_byte(&self, byte: usize) -> Result<(), PlanError> {
        if byte > self.accepted_end || !self.source.text().is_char_boundary(byte) {
            return Err(PlanError::InvalidResume {
                byte,
                accepted_end: self.accepted_end,
            });
        }
        Ok(())
    }

    /// Immutable full source and its identity.
    pub fn source(&self) -> &SourceSnapshot {
        &self.source
    }

    /// Fixed permitted voices and model identity.
    pub fn voices(&self) -> &VoiceSnapshot {
        &self.voices
    }

    /// Requested playback policy, without execution side effects.
    pub fn playback_policy(&self) -> PlaybackPolicy {
        self.policy
    }

    /// Read-only accepted assignments, in original source order.
    pub fn spans(&self) -> &[SpeechSpan] {
        &self.spans
    }

    /// Byte end of the accepted prefix, not generated or played progress.
    pub fn accepted_end(&self) -> usize {
        self.accepted_end
    }

    /// Current input state; sealing is unrelated to playback completion.
    pub fn state(&self) -> PlanState {
        self.state
    }

    pub(crate) fn validate_capabilities(
        &self,
        capabilities: &Capabilities,
    ) -> Result<(), PlanError> {
        self.voices.validate_capabilities(capabilities)
    }

    fn require_open(&self) -> Result<(), PlanError> {
        if self.state != PlanState::Open {
            return Err(PlanError::NotOpen(self.state));
        }
        Ok(())
    }
}
