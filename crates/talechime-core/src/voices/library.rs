//! Model-free public library operations. Storage and archive layouts stay private.
use super::{VoiceAsset, VoiceSource, asset, store};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum VoiceLibraryError {
    #[error("input package {index}: {error}")]
    Input {
        index: usize,
        #[source]
        error: Box<VoiceLibraryError>,
    },
    #[error("unsupported voice format: {0}")]
    UnsupportedFormat(String),
    #[error("invalid voice package: {0}")]
    InvalidPackage(String),
    #[error("voice transfer limit exceeded: {0}")]
    LimitExceeded(String),
    #[error("invalid reference audio: {0}")]
    InvalidAudio(String),
    #[error("voice contents do not match their fingerprint")]
    ContentMismatch,
    #[error("voice conflict: {0}")]
    Conflict(String),
    #[error("voice preview is no longer valid")]
    StalePreview,
    #[error("voice library is busy")]
    Busy,
    #[error("voice transfer cancelled")]
    Cancelled,
    #[error("voice disk operation failed: {0}")]
    Disk(#[source] std::io::Error),
    #[error("voice I/O failed: {0}")]
    Io(#[source] std::io::Error),
}
impl From<std::io::Error> for VoiceLibraryError {
    fn from(e: std::io::Error) -> Self {
        match e.kind() {
            std::io::ErrorKind::StorageFull => Self::Disk(e),
            std::io::ErrorKind::InvalidData | std::io::ErrorKind::UnexpectedEof => {
                Self::InvalidPackage(e.to_string())
            }
            _ => Self::Io(e),
        }
    }
}
#[derive(Clone, Debug)]
pub struct VoiceLibrary {
    pub(super) root: PathBuf,
}
#[derive(Debug)]
pub struct LibraryDiagnostic {
    pub id: String,
    pub error: VoiceLibraryError,
}
#[derive(Debug, Default)]
pub struct LibraryInventory {
    pub assets: Vec<VoiceAsset>,
    pub diagnostics: Vec<LibraryDiagnostic>,
}

impl VoiceLibrary {
    /// Open an explicitly chosen asset root without creating or migrating data.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, VoiceLibraryError> {
        let root = root.into();
        let root = if root.is_absolute() {
            root
        } else {
            std::env::current_dir()?.join(root)
        };
        let library = Self { root };
        store::recover_staging(&library)?;
        store::recover_scratch();
        Ok(library)
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// Create from a reference. Imported packages use the transfer API instead.
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        &self,
        id: &str,
        name: &str,
        wav: &[u8],
        transcript: &str,
        description: Option<&str>,
        source: VoiceSource,
    ) -> Result<VoiceAsset, VoiceLibraryError> {
        if wav.len() > 64 * 1024 * 1024 {
            return Err(VoiceLibraryError::LimitExceeded("reference WAV".into()));
        }
        let duration = asset::validate_wav(std::io::Cursor::new(wav))?;
        let mut voice = VoiceAsset {
            id: id.parse()?,
            name: name.into(),
            transcript: transcript.into(),
            description: description.map(str::to_owned),
            source,
            created_at_unix: asset::now(),
            imported_at_unix: asset::now(),
            fingerprint: asset::fingerprint(transcript, description, wav),
            asset_digest: String::new(),
            wav_bytes: wav.len() as u64,
            duration_seconds: duration,
        };
        voice.asset_digest = voice.compute_digest()?;
        voice.validate()?;
        let mut reference = store::temporary()?;
        std::io::Write::write_all(&mut reference, wav)?;
        store::publish(
            self,
            voice.clone(),
            reference.path(),
            None,
            &super::TransferControl::default(),
        )?;
        Ok(voice)
    }
    pub fn load(&self, id: &str) -> Result<(VoiceAsset, Vec<u8>), VoiceLibraryError> {
        let _lock = store::lock(self)?;
        let snapshot = store::snapshot(self, id)?;
        Ok((snapshot.asset, std::fs::read(snapshot.wav.path())?))
    }
    pub fn get(&self, id: &str) -> Result<VoiceAsset, VoiceLibraryError> {
        self.load(id).map(|(a, _)| a)
    }
    pub fn read_wav(&self, id: &str) -> Result<Vec<u8>, VoiceLibraryError> {
        self.load(id).map(|(_, wav)| wav)
    }
    /// Read metadata without loading every recording. Diagnostics do not hide valid assets.
    pub fn inventory(&self) -> Result<LibraryInventory, VoiceLibraryError> {
        store::inventory(self)
    }
    pub fn list(&self) -> Result<Vec<VoiceAsset>, VoiceLibraryError> {
        let inventory = self.inventory()?;
        if let Some(diagnostic) = inventory.diagnostics.into_iter().next() {
            return Err(diagnostic.error);
        }
        Ok(inventory.assets)
    }
    pub fn exists(&self, id: &str) -> bool {
        asset::validate_id(id).is_ok() && self.root.join(id).join("voice.json").is_file()
    }
    pub fn rename(&self, id: &str, name: &str) -> Result<VoiceAsset, VoiceLibraryError> {
        store::rename(self, id, name)
    }
    pub fn remove(&self, id: &str) -> Result<(), VoiceLibraryError> {
        store::remove(self, id)
    }
}
