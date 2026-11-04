//! Validated, model-independent voice identities and asset metadata.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek};

use super::VoiceLibraryError;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct VoiceId(String);
impl VoiceId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl TryFrom<String> for VoiceId {
    type Error = VoiceLibraryError;
    fn try_from(id: String) -> Result<Self, Self::Error> {
        validate_id(&id)?;
        Ok(Self(id))
    }
}
impl std::str::FromStr for VoiceId {
    type Err = VoiceLibraryError;
    fn from_str(id: &str) -> Result<Self, Self::Err> {
        Self::try_from(id.to_owned())
    }
}
impl From<VoiceId> for String {
    fn from(id: VoiceId) -> Self {
        id.0
    }
}
impl std::ops::Deref for VoiceId {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}
impl std::fmt::Display for VoiceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "engine",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum VoiceSource {
    Designed(String),
    Imported,
}

/// An immutable asset. Deserialize only the private storage DTO, then validate it.
#[derive(Clone, Debug, Serialize)]
pub struct VoiceAsset {
    pub(super) id: VoiceId,
    pub(super) name: String,
    pub(super) transcript: String,
    pub(super) description: Option<String>,
    pub(super) source: VoiceSource,
    pub(super) created_at_unix: u64,
    pub(super) imported_at_unix: u64,
    pub(super) fingerprint: String,
    pub(super) asset_digest: String,
    pub(super) wav_bytes: u64,
    pub(super) duration_seconds: f64,
}
impl VoiceAsset {
    pub fn id(&self) -> &str {
        self.id.as_str()
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn transcript(&self) -> &str {
        &self.transcript
    }
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
    pub fn source(&self) -> &VoiceSource {
        &self.source
    }
    pub fn created_at_unix(&self) -> u64 {
        self.created_at_unix
    }
    pub fn imported_at_unix(&self) -> u64 {
        self.imported_at_unix
    }
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
    pub fn asset_digest(&self) -> &str {
        &self.asset_digest
    }
    pub fn wav_bytes(&self) -> u64 {
        self.wav_bytes
    }
    pub fn duration_seconds(&self) -> f64 {
        self.duration_seconds
    }
    pub(super) fn validate(&self) -> Result<(), VoiceLibraryError> {
        validate_id(self.id())?;
        if self.created_at_unix > 253402300799
            || self.imported_at_unix > 253402300799
            || self.name.trim().is_empty()
            || self.name.len() > 256
            || self.transcript.trim().is_empty()
            || self.transcript.len() > 16384
            || self.description.as_ref().is_some_and(|d| d.len() > 16384)
            || matches!(&self.source, VoiceSource::Designed(s) if s.is_empty() || s.len() > 128)
            || !self.duration_seconds.is_finite()
            || self.duration_seconds <= 0.0
            || self.wav_bytes == 0
            || self.wav_bytes > 64 * 1024 * 1024
        {
            return Err(VoiceLibraryError::InvalidPackage(
                "invalid asset metadata".into(),
            ));
        }
        if self.asset_digest != self.compute_digest()? {
            return Err(VoiceLibraryError::ContentMismatch);
        }
        Ok(())
    }
    pub(super) fn compute_digest(&self) -> Result<String, VoiceLibraryError> {
        let bytes = serde_json::to_vec(&(
            "talechime/asset/2",
            self.id(),
            &self.name,
            &self.transcript,
            &self.description,
            &self.source,
            self.created_at_unix,
            &self.fingerprint,
            self.wav_bytes,
            self.duration_seconds,
        ))
        .map_err(|e| VoiceLibraryError::InvalidPackage(e.to_string()))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

pub fn validate_id(id: &str) -> Result<(), VoiceLibraryError> {
    if id.is_empty()
        || id.len() > 64
        || !id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_'))
    {
        return Err(VoiceLibraryError::InvalidPackage("invalid voice ID".into()));
    }
    Ok(())
}

pub(super) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|t| t.as_secs())
        .unwrap_or_default()
}

/// Hash the exact reference and its synthesis-relevant metadata.
pub fn fingerprint(transcript: &str, description: Option<&str>, wav: &[u8]) -> String {
    let mut hash = reference_hasher(transcript, description);
    hash.update(wav);
    format!("{:x}", hash.finalize())
}
pub(super) fn reference_hasher(transcript: &str, description: Option<&str>) -> Sha256 {
    let mut hash = Sha256::new();
    hash.update(b"talechime/reference/2");
    for text in [transcript, description.unwrap_or_default()] {
        hash.update((text.len() as u64).to_le_bytes());
        hash.update(text.as_bytes());
    }
    hash
}

pub(super) fn validate_wav(reader: impl Read + Seek) -> Result<f64, VoiceLibraryError> {
    validate_wav_with(reader, || Ok(()))
}
pub(super) fn validate_wav_with(
    reader: impl Read + Seek,
    mut check: impl FnMut() -> Result<(), VoiceLibraryError>,
) -> Result<f64, VoiceLibraryError> {
    let mut wav = hound::WavReader::new(reader)
        .map_err(|e| VoiceLibraryError::InvalidAudio(e.to_string()))?;
    let spec = wav.spec();
    if spec.sample_rate == 0 || spec.channels == 0 || wav.len() == 0 {
        return Err(VoiceLibraryError::InvalidAudio("empty WAV".into()));
    }
    let duration = f64::from(wav.duration()) / f64::from(spec.sample_rate);
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for (index, sample) in wav.samples::<f32>().enumerate() {
                if index % 4096 == 0 {
                    check()?;
                }
                if !sample
                    .map_err(|e| VoiceLibraryError::InvalidAudio(e.to_string()))?
                    .is_finite()
                {
                    return Err(VoiceLibraryError::InvalidAudio("nonfinite sample".into()));
                }
            }
        }
        hound::SampleFormat::Int => {
            for (index, sample) in wav.samples::<i32>().enumerate() {
                if index % 4096 == 0 {
                    check()?;
                }
                sample.map_err(|e| VoiceLibraryError::InvalidAudio(e.to_string()))?;
            }
        }
    }
    Ok(duration)
}
