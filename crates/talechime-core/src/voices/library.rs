//! Backend-agnostic voice assets: the single source of truth for designed and
//! imported reference recordings. Lives under the app root (`~/.talechime/voices`),
//! never inside model resources; per-model stores are regenerable caches of it.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

/// How a library voice came to exist.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "engine", rename_all = "snake_case")]
pub enum VoiceSource {
    /// Produced by the named design engine.
    Designed(String),
    /// Installed by the user from a reference recording.
    Imported,
}

/// One library voice record; `reference.wav` sits beside `voice.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceAsset {
    pub id: String,
    pub name: String,
    pub transcript: String,
    pub description: Option<String>,
    pub source: VoiceSource,
    pub created_at_unix: u64,
    /// sha256 over transcript, description and reference WAV bytes; materializations compare against it
    /// to detect stale caches.
    pub fingerprint: String,
}

const RECORD_FORMAT: &str = "talechime/voice/1";

#[derive(Serialize, Deserialize)]
struct StoredVoice {
    format: String,
    #[serde(flatten)]
    asset: VoiceAsset,
}

fn validate_metadata(name: &str, transcript: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !name.trim().is_empty() && name.len() <= 256,
        "voice name must contain 1..256 bytes"
    );
    anyhow::ensure!(
        !transcript.trim().is_empty() && transcript.len() <= 16384,
        "provide the reference recording's transcript (up to 16384 bytes)"
    );
    Ok(())
}

/// Validate the full recording without changing its bytes. Model-specific
/// channel, rate and duration limits belong to materialization.
fn validate_wav(wav: &[u8]) -> anyhow::Result<()> {
    let mut reader =
        hound::WavReader::new(std::io::Cursor::new(wav)).context("invalid reference WAV")?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_rate > 0 && reader.len() > 0,
        "WAV recording must have a sample rate and nonempty audio"
    );
    // WavReader rejects partial samples/frames in the declared data length;
    // iterating every sample also detects physically truncated recordings.
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                anyhow::ensure!(sample?.is_finite(), "WAV samples must be finite");
            }
        }
        hound::SampleFormat::Int => {
            for sample in reader.samples::<i32>() {
                sample?;
            }
        }
    }
    Ok(())
}

/// Library IDs share the `custom:` character set so materialization can reuse
/// them verbatim after the prefix.
pub fn validate_id(id: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !id.is_empty()
            && id.len() <= 64
            && id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_')),
        "invalid voice ID: use 1..64 ASCII letters, digits, '-' or '_'"
    );
    Ok(())
}

pub fn fingerprint(transcript: &str, description: Option<&str>, wav: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(transcript.as_bytes());
    hasher.update([0]);
    hasher.update(description.unwrap_or_default().as_bytes());
    hasher.update([0]);
    hasher.update(wav);
    format!("{:x}", hasher.finalize())
}

#[derive(Clone)]
pub struct VoiceLibrary {
    root: PathBuf,
}
impl VoiceLibrary {
    pub fn open(root: impl Into<PathBuf>) -> anyhow::Result<Self> {
        Ok(Self { root: root.into() })
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    fn directory(&self, id: &str) -> anyhow::Result<PathBuf> {
        validate_id(id)?;
        Ok(self.root.join(id))
    }
    fn load_record(&self, directory: &Path) -> anyhow::Result<VoiceAsset> {
        let stored: StoredVoice =
            serde_json::from_reader(std::fs::File::open(directory.join("voice.json"))?)?;
        anyhow::ensure!(
            stored.format == RECORD_FORMAT,
            "unsupported voice record format {}; refusing to read",
            stored.format
        );
        let asset = stored.asset;
        anyhow::ensure!(
            self.directory(&asset.id)? == directory,
            "voice filename identity mismatch"
        );
        validate_metadata(&asset.name, &asset.transcript)?;
        Ok(asset)
    }
    fn load_unlocked(&self, directory: &Path) -> anyhow::Result<(VoiceAsset, Vec<u8>)> {
        anyhow::ensure!(
            std::fs::symlink_metadata(directory)?.file_type().is_dir(),
            "voice record must be a directory"
        );
        let asset = self.load_record(directory)?;
        let wav = std::fs::read(directory.join("reference.wav"))?;
        validate_wav(&wav)?;
        anyhow::ensure!(
            asset.fingerprint == fingerprint(&asset.transcript, asset.description.as_deref(), &wav),
            "voice contents do not match their fingerprint"
        );
        Ok((asset, wav))
    }
    /// Atomically create a voice; a collision fails without touching the existing record.
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        &self,
        id: &str,
        name: &str,
        wav: &[u8],
        transcript: &str,
        description: Option<&str>,
        source: VoiceSource,
    ) -> anyhow::Result<VoiceAsset> {
        validate_metadata(name, transcript)?;
        validate_wav(wav)?;
        let destination = self.directory(id)?;
        let _lock = crate::storage::lock(&self.root.join(".lock"))?;
        anyhow::ensure!(
            !destination.exists(),
            "voice {id} already exists; choose another ID or remove it first"
        );
        let staging = tempfile::Builder::new()
            .prefix(".staging-")
            .tempdir_in(&self.root)?;
        let mut file = tempfile::NamedTempFile::new_in(staging.path())?;
        file.write_all(wav)?;
        file.as_file().sync_all()?;
        file.persist(staging.path().join("reference.wav"))
            .map_err(|error| error.error)?;
        let asset = VoiceAsset {
            id: id.into(),
            name: name.into(),
            transcript: transcript.into(),
            description: description.map(str::to_owned),
            source,
            created_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|since| since.as_secs())
                .unwrap_or_default(),
            fingerprint: fingerprint(transcript, description, wav),
        };
        self.save_record(staging.path(), &asset)?;
        std::fs::rename(staging.path(), destination)?;
        Ok(asset)
    }
    fn save_record(&self, directory: &Path, asset: &VoiceAsset) -> anyhow::Result<()> {
        crate::storage::save(
            &directory.join("voice.json"),
            &StoredVoice {
                format: RECORD_FORMAT.into(),
                asset: asset.clone(),
            },
        )?;
        Ok(())
    }
    /// Read metadata and WAV as one locked snapshot, validating the current
    /// record format, metadata, WAV and content fingerprint. Malformed or
    /// missing data and lock contention return errors rather than partial data.
    pub fn load(&self, id: &str) -> anyhow::Result<(VoiceAsset, Vec<u8>)> {
        let directory = self.directory(id)?;
        let _lock = crate::storage::lock(&self.root.join(".lock"))?;
        self.load_unlocked(&directory)
            .map_err(|error| anyhow::anyhow!("cannot load voice {id}: {error:#}"))
    }
    pub fn get(&self, id: &str) -> anyhow::Result<VoiceAsset> {
        self.load(id).map(|(asset, _)| asset)
    }
    pub fn read_wav(&self, id: &str) -> anyhow::Result<Vec<u8>> {
        self.load(id).map(|(_, wav)| wav)
    }
    pub fn list(&self) -> anyhow::Result<Vec<VoiceAsset>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let _lock = crate::storage::lock(&self.root.join(".lock"))?;
        let mut assets = Vec::new();
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_str().context("invalid voice directory name")?;
            if name == ".lock" || name.starts_with(".staging-") || name.starts_with(".removed-") {
                continue;
            }
            if entry.file_type()?.is_file() {
                continue;
            }
            assets.push(
                self.load_unlocked(&entry.path())
                    .map_err(|error| anyhow::anyhow!("cannot load voice {name}: {error:#}"))?
                    .0,
            );
        }
        assets.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(assets)
    }
    pub fn exists(&self, id: &str) -> bool {
        self.directory(id)
            .is_ok_and(|directory| directory.join("voice.json").is_file())
    }
    /// Rename the display name in place; the ID and recording are immutable.
    pub fn rename(&self, id: &str, name: &str) -> anyhow::Result<VoiceAsset> {
        anyhow::ensure!(
            !name.trim().is_empty() && name.len() <= 256,
            "voice name must contain 1..256 bytes"
        );
        let directory = self.directory(id)?;
        let _lock = crate::storage::lock(&self.root.join(".lock"))?;
        let (mut asset, _) = self.load_unlocked(&directory)?;
        asset.name = name.into();
        self.save_record(&directory, &asset)?;
        Ok(asset)
    }
    pub fn remove(&self, id: &str) -> anyhow::Result<()> {
        let directory = self.directory(id)?;
        let _lock = crate::storage::lock(&self.root.join(".lock"))?;
        self.load_unlocked(&directory)?;
        let tombstone = tempfile::Builder::new()
            .prefix(".removed-")
            .tempdir_in(&self.root)?;
        std::fs::rename(directory, tombstone.path().join("voice"))?;
        drop(_lock);
        tombstone.close()?;
        Ok(())
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(crate) fn wav_fixture() -> Vec<u8> {
        let mut output = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(
            &mut output,
            hound::WavSpec {
                channels: 1,
                sample_rate: 24_000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for sample in [100i16, -100, 200, -200] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();
        output.into_inner()
    }

    fn create_fixture(library: &VoiceLibrary) {
        library
            .create(
                "hero",
                "主角",
                &wav_fixture(),
                "转写",
                None,
                VoiceSource::Imported,
            )
            .unwrap();
    }

    #[test]
    fn reads_reject_missing_tampered_and_malformed_assets() {
        for corruption in [
            "missing-wav",
            "changed-wav",
            "bad-json",
            "missing-json",
            "bad-transcript",
            "future-format",
            "missing-format",
        ] {
            let root = tempfile::tempdir().unwrap();
            let library = VoiceLibrary::open(root.path()).unwrap();
            create_fixture(&library);
            let directory = root.path().join("hero");
            match corruption {
                "missing-wav" => std::fs::remove_file(directory.join("reference.wav")).unwrap(),
                "changed-wav" => {
                    let mut wav = wav_fixture();
                    wav[44] ^= 1;
                    std::fs::write(directory.join("reference.wav"), wav).unwrap();
                }
                "bad-json" => std::fs::write(directory.join("voice.json"), b"{").unwrap(),
                "missing-json" => std::fs::remove_file(directory.join("voice.json")).unwrap(),
                other => {
                    let path = directory.join("voice.json");
                    let mut record: serde_json::Value =
                        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                    match other {
                        "bad-transcript" => record["transcript"] = " ".into(),
                        "future-format" => record["format"] = "talechime/voice/999".into(),
                        "missing-format" => {
                            record.as_object_mut().unwrap().remove("format");
                        }
                        _ => unreachable!(),
                    }
                    std::fs::write(path, serde_json::to_vec(&record).unwrap()).unwrap();
                }
            }
            assert!(library.load("hero").is_err(), "{corruption}");
            assert!(library.get("hero").is_err(), "{corruption}");
            assert!(library.read_wav("hero").is_err(), "{corruption}");
            assert!(library.list().is_err(), "{corruption}");
            assert!(directory.exists(), "reading must not remove malformed data");
        }
    }

    #[test]
    fn snapshot_verifies_fingerprint_and_obeys_the_library_lock() {
        let root = tempfile::tempdir().unwrap();
        let library = VoiceLibrary::open(root.path()).unwrap();
        create_fixture(&library);
        let (asset, wav) = library.load("hero").unwrap();
        assert_eq!(wav, wav_fixture());
        assert_eq!(
            asset.fingerprint,
            fingerprint(&asset.transcript, asset.description.as_deref(), &wav)
        );
        let lock = crate::storage::lock(&root.path().join(".lock")).unwrap();
        assert!(library.load("hero").is_err());
        assert!(library.list().is_err());
        drop(lock);
        assert!(library.load("hero").is_ok());
    }

    #[test]
    fn removal_hides_the_whole_record_before_recursive_cleanup() {
        let root = tempfile::tempdir().unwrap();
        let library = VoiceLibrary::open(root.path()).unwrap();
        create_fixture(&library);
        // Reproduce the on-disk state after atomic rename but before cleanup,
        // including the state left by a process interrupted at that point.
        let tombstone = tempfile::Builder::new()
            .prefix(".removed-")
            .tempdir_in(root.path())
            .unwrap();
        std::fs::rename(root.path().join("hero"), tombstone.path().join("voice")).unwrap();
        assert!(library.list().unwrap().is_empty());
        assert!(library.load("hero").is_err());
        // The original id can immediately be reused while cleanup is pending.
        create_fixture(&library);
        tombstone.close().unwrap();
        assert!(library.load("hero").is_ok());
        library.remove("hero").unwrap();
        assert!(library.list().unwrap().is_empty());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1); // .lock only
    }

    #[test]
    fn create_rejects_invalid_wav_before_writing() {
        let root = tempfile::tempdir().unwrap();
        let library = VoiceLibrary::open(root.path()).unwrap();
        let mut truncated = wav_fixture();
        truncated.pop();
        let mut invalid_align = wav_fixture();
        invalid_align[32..34].copy_from_slice(&0u16.to_le_bytes());
        let mut empty = wav_fixture();
        empty.truncate(44);
        empty[4..8].copy_from_slice(&36u32.to_le_bytes());
        empty[40..44].copy_from_slice(&0u32.to_le_bytes());
        for wav in [b"not WAV".to_vec(), truncated, invalid_align, empty] {
            assert!(
                library
                    .create("bad", "n", &wav, "t", None, VoiceSource::Imported)
                    .is_err()
            );
        }
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }

    #[test]
    fn float_wav_accepts_finite_samples_and_rejects_nan() {
        let mut wav = wav_fixture();
        wav[20..22].copy_from_slice(&3u16.to_le_bytes());
        wav[28..32].copy_from_slice(&96_000u32.to_le_bytes());
        wav[32..34].copy_from_slice(&4u16.to_le_bytes());
        wav[34..36].copy_from_slice(&32u16.to_le_bytes());
        wav[44..48].copy_from_slice(&0.1f32.to_le_bytes());
        wav[48..52].copy_from_slice(&(-0.1f32).to_le_bytes());
        validate_wav(&wav).unwrap();
        wav[44..48].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(validate_wav(&wav).is_err());
    }

    #[test]
    fn create_list_rename_remove_roundtrip_is_atomic() {
        let root = tempfile::tempdir().unwrap();
        let library = VoiceLibrary::open(root.path()).unwrap();
        let wav = wav_fixture();
        let asset = library
            .create(
                "hero",
                "主角",
                &wav,
                "试音转写",
                Some("低沉"),
                VoiceSource::Imported,
            )
            .unwrap();
        assert_eq!(asset.id, "hero");
        assert_eq!(library.list().unwrap().len(), 1);
        assert_eq!(library.read_wav("hero").unwrap(), wav);
        // Collision refuses without overwriting.
        let error = library
            .create("hero", "again", &wav, "x", None, VoiceSource::Imported)
            .unwrap_err();
        assert!(error.to_string().contains("already exists"));
        assert_eq!(library.read_wav("hero").unwrap(), wav);
        let renamed = library.rename("hero", "新名字").unwrap();
        assert_eq!(renamed.name, "新名字");
        assert_eq!(library.get("hero").unwrap().name, "新名字");
        library.remove("hero").unwrap();
        assert!(library.list().unwrap().is_empty());
        assert!(!root.path().join("hero").exists());
    }

    #[test]
    fn ids_reject_traversal_and_bad_charset() {
        let root = tempfile::tempdir().unwrap();
        let library = VoiceLibrary::open(root.path()).unwrap();
        for bad in ["", "a b", "../escape", "a/b", ".", "..", "café"] {
            assert!(validate_id(bad).is_err(), "{bad} should be rejected");
            assert!(
                library
                    .create(bad, "n", &wav_fixture(), "t", None, VoiceSource::Imported)
                    .is_err(),
                "{bad} should be rejected"
            );
        }
        assert!(validate_id("A-b_9").is_ok());
        let entries = std::fs::read_dir(root.path())
            .map(|dir| dir.filter_map(|entry| entry.ok()).count())
            .unwrap_or(0);
        assert_eq!(entries, 0, "rejected IDs must not create directories");
    }

    #[test]
    fn fingerprint_changes_with_content() {
        let a = fingerprint("t", None, b"a");
        assert_eq!(a, fingerprint("t", None, b"a"));
        assert_ne!(a, fingerprint("t2", None, b"a"));
        assert_ne!(a, fingerprint("t", Some("d"), b"a"));
        assert_ne!(a, fingerprint("t", None, b"b"));
    }

    #[test]
    fn transcript_and_name_bounds_hold() {
        let root = tempfile::tempdir().unwrap();
        let library = VoiceLibrary::open(root.path()).unwrap();
        assert!(
            library
                .create("x", "n", &wav_fixture(), "  ", None, VoiceSource::Imported)
                .is_err()
        );
        assert!(
            library
                .create("x", " ", &wav_fixture(), "t", None, VoiceSource::Imported)
                .is_err()
        );
        assert!(
            library
                .create("x", "n", b"", "t", None, VoiceSource::Imported)
                .is_err()
        );
    }
}
