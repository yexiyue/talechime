//! Validated reference audio before an explicit library commit.
use super::{VoiceAsset, VoiceLibraryError, VoiceSource, asset};

/// A model-independent, validated recording. Construction performs no file writes.
/// Candidate identity is allocated by the caller and stays stable when committing.
#[derive(Debug, Clone)]
pub struct VoiceCandidate {
    asset: VoiceAsset,
    wav: Vec<u8>,
}
impl VoiceCandidate {
    /// Validate bounded WAV bytes and metadata without creating a library record.
    pub fn new(
        id: &str,
        name: &str,
        wav: Vec<u8>,
        transcript: &str,
        description: Option<&str>,
        source: VoiceSource,
    ) -> Result<Self, VoiceLibraryError> {
        if wav.len() > 64 * 1024 * 1024 {
            return Err(VoiceLibraryError::LimitExceeded("reference WAV".into()));
        }
        let duration = asset::validate_wav(std::io::Cursor::new(&wav))?;
        let mut voice = VoiceAsset {
            id: id.parse()?,
            name: name.into(),
            transcript: transcript.into(),
            description: description.map(str::to_owned),
            source,
            created_at_unix: asset::now(),
            imported_at_unix: asset::now(),
            fingerprint: asset::fingerprint(transcript, description, &wav),
            asset_digest: String::new(),
            wav_bytes: wav.len() as u64,
            duration_seconds: duration,
        };
        voice.asset_digest = voice.compute_digest()?;
        voice.validate()?;
        Ok(Self { asset: voice, wav })
    }
    /// Read-only validated metadata; not evidence of a committed library record.
    pub fn asset(&self) -> &VoiceAsset {
        &self.asset
    }
    /// Exact validated reference WAV.
    pub fn wav(&self) -> &[u8] {
        &self.wav
    }
}
