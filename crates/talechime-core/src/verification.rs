//! Readback without source prompts, with conservative cross-family consensus.
mod comparison;
mod normalization;
mod work;
use crate::{SourceSnapshot, backend::Pcm};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell, collections::VecDeque, future::Future, pin::Pin, rc::Rc, sync::Arc,
    time::Duration,
};
use tts_protocol::TextRange;
pub use tts_protocol::{
    DifferenceKind, ReadbackDifference, ReadbackEvidence, RecognizerIdentity, VerificationOptions,
    VerificationPolicy, VerificationReport, VerificationVerdict,
};
pub(crate) use work::RequestCompletions;

const RULES: &str = "readback-v1-conservative";

/// Prepared ASR receives only audio; never reference text, hotwords or voice hints.
pub trait Recognizer {
    fn identity(&self) -> RecognizerIdentity;
    /// Cancellation must release request ownership and stop native work cooperatively.
    fn transcribe(&self, audio: Arc<ReadbackAudio>) -> Recognition<'_>;
    /// Native owners attach this request's cleanup receipt. Async-only recognizers
    /// need no receipt: dropping their future releases all request work.
    fn request(&self, audio: Arc<ReadbackAudio>) -> RecognitionRequest<'_> {
        RecognitionRequest {
            future: self.transcribe(audio),
            completion: None,
        }
    }
}
pub type Recognition<'a> = Pin<Box<dyn Future<Output = Result<String, VerificationError>> + 'a>>;
/// A transcription and optional per-request native completion receipt.
/// Send `true` or close the sender only after all native work has stopped.
pub struct RecognitionRequest<'a> {
    pub future: Recognition<'a>,
    pub completion: Option<tokio::sync::watch::Receiver<bool>>,
}
/// Immutable framed PCM handed to prepared inference owners.
#[derive(Debug)]
pub struct ReadbackAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}
/// All text and synthesis identity is kept outside the recognizer boundary.
pub struct ReadbackRequest<'a> {
    pub source: &'a SourceSnapshot,
    pub range: TextRange,
    pub spoken_text: &'a str,
    pub backend: &'a str,
    pub model: Option<&'a str>,
    pub voice: &'a str,
    pub style: Option<&'a str>,
    pub attempt: u8,
}
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum VerificationError {
    #[error("invalid readback input: {0}")]
    Invalid(String),
    #[error("ASR failed: {0}")]
    Recognition(String),
    #[error("ASR timed out")]
    Timeout,
    #[error("readback segment exceeds its PCM bound")]
    Capacity,
    #[error("verification requested without a prepared verifier")]
    NotPrepared,
    #[error("readback blocked delivery: {verdict:?} after attempt {attempt}")]
    Rejected {
        verdict: VerificationVerdict,
        attempt: u8,
    },
}

/// Validate before replacing any active execution. Ordinary calls do not load models.
pub fn validate_options(options: &VerificationOptions) -> Result<(), VerificationError> {
    if options.timeout_ms == 0
        || options.timeout_ms > 300_000
        || options.max_segment_ms == 0
        || options.max_segment_ms > 30_000
        || options.max_segment_bytes == 0
        || options.max_segment_bytes > 16 * 1024 * 1024
        || matches!(
            options.policy,
            VerificationPolicy::Gate {
                max_retries: 4..,
                ..
            }
        )
    {
        return Err(VerificationError::Invalid(
            "invalid timeout, segment bound or retry count (0..=3)".into(),
        ));
    }
    Ok(())
}

/// Reusable prepared model group. Own on the same LocalSet as the synthesis engine.
pub struct Verifier {
    primary: Rc<dyn Recognizer>,
    reviewer: Rc<dyn Recognizer>,
    cache: Rc<RefCell<VecDeque<(String, VerificationReport)>>>,
    cache_capacity: usize,
    normalization: Rc<normalization::Normalizer>,
    work: Arc<RequestCompletions>,
}
impl Verifier {
    /// Reviewer must belong to a different ASR family; size variants are not independent.
    pub fn new(
        primary: Rc<dyn Recognizer>,
        reviewer: Rc<dyn Recognizer>,
    ) -> Result<Self, VerificationError> {
        let a = primary.identity();
        let b = reviewer.identity();
        if a.family.is_empty()
            || a.family == b.family
            || b.family.is_empty()
            || [
                &a.model,
                &a.revision,
                &a.implementation,
                &b.model,
                &b.revision,
                &b.implementation,
            ]
            .iter()
            .any(|s| s.is_empty())
        {
            return Err(VerificationError::Invalid(
                "two pinned, different ASR families are required".into(),
            ));
        }
        Ok(Self {
            primary,
            reviewer,
            cache: Default::default(),
            cache_capacity: 32,
            normalization: Rc::new(normalization::Normalizer::new()?),
            work: Default::default(),
        })
    }
    /// Bound volatile evidence reuse to 0..=256 entries. Audio is never cached here.
    pub fn with_cache_capacity(mut self, capacity: usize) -> Result<Self, VerificationError> {
        if capacity > 256 {
            return Err(VerificationError::Invalid(
                "readback cache exceeds 256 entries".into(),
            ));
        }
        self.cache_capacity = capacity;
        self.cache.borrow_mut().clear();
        Ok(self)
    }
    /// Await native requests after timeout or cancellation; no new transcription starts here.
    pub async fn settled(&self) {
        self.work.settled().await;
    }
    pub(crate) fn work(&self) -> Arc<RequestCompletions> {
        self.work.clone()
    }
    /// Share prepared models/evidence, but isolate the execution's cleanup receipts.
    pub(crate) fn scoped(&self) -> Rc<Self> {
        Rc::new(Self {
            primary: self.primary.clone(),
            reviewer: self.reviewer.clone(),
            cache: self.cache.clone(),
            cache_capacity: self.cache_capacity,
            normalization: self.normalization.clone(),
            work: Default::default(),
        })
    }
    /// Standalone report API. ASR failure/timeout yields Unverified evidence; invalid
    /// input is an error. It never regenerates audio or creates a player/runtime.
    pub async fn report(
        &self,
        request: ReadbackRequest<'_>,
        pcm: &Pcm,
        options: &VerificationOptions,
    ) -> Result<VerificationReport, VerificationError> {
        validate_options(options)?;
        let duration = pcm
            .duration_ms()
            .map_err(|e| VerificationError::Invalid(e.to_string()))?;
        if duration > options.max_segment_ms || pcm.samples.len() > options.max_segment_bytes / 4 {
            return Err(VerificationError::Capacity);
        }
        if !request.range.is_valid(request.source.text())
            || request.range.start >= request.range.end
            || request.spoken_text.trim().is_empty()
            || request.spoken_text.chars().count() > 512
            || request.source.text()[request.range.start..request.range.end]
                .chars()
                .count()
                > 512
        {
            return Err(VerificationError::Invalid(
                "invalid source range or spoken text (max 512 chars)".into(),
            ));
        }
        let audio_hash = audio_hash(pcm);
        let key = tts_protocol::text_hash(
            &serde_json::to_string(&(
                request.source.hash(),
                request.range,
                request.spoken_text,
                request.backend,
                request.model,
                request.voice,
                request.style,
                &audio_hash,
                RULES,
                self.primary.identity(),
                self.reviewer.identity(),
            ))
            .map_err(|error| VerificationError::Invalid(error.to_string()))?,
        );
        if let Some((_, report)) = self
            .cache
            .borrow()
            .iter()
            .find(|(cached, _)| *cached == key)
        {
            let mut report = report.clone();
            report.attempt = request.attempt;
            report.cache_hit = true;
            return Ok(report);
        }
        let audio = Arc::new(ReadbackAudio {
            samples: pcm.samples.clone(),
            sample_rate: pcm.sample_rate,
            channels: pcm.channels,
        });
        let first = self
            .evidence(&*self.primary, audio.clone(), &request, options)
            .await;
        let mut evidence = vec![first];
        let verdict = if evidence[0].error.is_some() {
            VerificationVerdict::Unverified
        } else if evidence[0].differences.is_empty() {
            VerificationVerdict::Passed
        } else {
            evidence.push(
                self.evidence(&*self.reviewer, audio, &request, options)
                    .await,
            );
            if evidence[1].error.is_some() {
                VerificationVerdict::Unverified
            } else if comparison::confirmed(
                &self.normalization,
                &evidence[0].differences,
                &evidence[1].differences,
                request.spoken_text,
            ) {
                VerificationVerdict::ConfirmedError
            } else {
                VerificationVerdict::Suspect
            }
        };
        let report = VerificationReport {
            text_hash: request.source.hash().into(),
            range: request.range,
            spoken_text: request.spoken_text.into(),
            backend: request.backend.into(),
            model: request.model.map(Into::into),
            voice: request.voice.into(),
            style: request.style.map(Into::into),
            audio_hash,
            attempt: request.attempt,
            cache_hit: false,
            rules: RULES.into(),
            verdict,
            evidence,
        };
        if verdict != VerificationVerdict::Unverified && self.cache_capacity > 0 {
            let mut cache = self.cache.borrow_mut();
            if cache.len() == self.cache_capacity {
                cache.pop_front();
            }
            cache.push_back((key, report.clone()));
        }
        Ok(report)
    }
    async fn evidence(
        &self,
        recognizer: &dyn Recognizer,
        audio: Arc<ReadbackAudio>,
        request: &ReadbackRequest<'_>,
        options: &VerificationOptions,
    ) -> ReadbackEvidence {
        let task = recognizer.request(audio);
        if let Some(completion) = task.completion {
            self.work.register(completion);
        }
        let result = tokio::time::timeout(Duration::from_millis(options.timeout_ms), task.future)
            .await
            .unwrap_or(Err(VerificationError::Timeout));
        match result {
            Ok(text) if text.chars().count() <= 1024 => ReadbackEvidence {
                recognizer: recognizer.identity(),
                differences: comparison::compare(&self.normalization, request, &text),
                transcript: Some(text),
                error: None,
            },
            Ok(_) => ReadbackEvidence {
                recognizer: recognizer.identity(),
                transcript: None,
                error: Some("ASR transcript exceeds 1024 characters".into()),
                differences: vec![],
            },
            Err(error) => ReadbackEvidence {
                recognizer: recognizer.identity(),
                transcript: None,
                error: Some(error.to_string()),
                differences: vec![],
            },
        }
    }
}

fn audio_hash(pcm: &Pcm) -> String {
    let mut hash = Sha256::new();
    hash.update(pcm.sample_rate.to_le_bytes());
    hash.update(pcm.channels.to_le_bytes());
    for sample in &pcm.samples {
        hash.update(sample.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
