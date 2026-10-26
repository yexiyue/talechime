//! Reference materialization ownership, integrity checks and replacement transaction.
use super::ModelAdapter;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};
use talechime_core::voices::library::{VoiceAsset, validate_id};

/// A library asset and its verbatim reference recording, borrowed for installation.
#[derive(Clone, Copy)]
pub struct ReferenceVoice<'a> {
    /// Metadata read from the voice library.
    pub asset: &'a VoiceAsset,
    /// Original WAV bytes matching the asset fingerprint.
    pub wav: &'a [u8],
}

pub(super) enum ReferenceEntry {
    #[cfg_attr(
        not(any(
            feature = "moss",
            feature = "qwen",
            feature = "voxcpm",
            feature = "omnivoice",
            feature = "cosyvoice3",
            feature = "indextts",
            test
        )),
        expect(
            dead_code,
            reason = "directory caches are constructed only by compiled model adapters"
        )
    )]
    Directory(PathBuf),
    #[cfg(any(feature = "moss-nano-candle", test))]
    File(PathBuf),
}
impl ReferenceEntry {
    fn path(&self) -> &Path {
        match self {
            Self::Directory(path) => path,
            #[cfg(any(feature = "moss-nano-candle", test))]
            Self::File(path) => path,
        }
    }
    fn marker(&self) -> PathBuf {
        // Never use .json: Nano enumerates that extension as user voices.
        let mut path = self.path().as_os_str().to_os_string();
        path.push(".library");
        PathBuf::from(path)
    }
    fn digest(&self) -> anyhow::Result<String> {
        let paths = match self {
            Self::Directory(path) => vec![path.join("voice.json"), path.join("reference.wav")],
            #[cfg(any(feature = "moss-nano-candle", test))]
            Self::File(path) => vec![path.clone()],
        };
        let mut hash = Sha256::new();
        for path in paths {
            anyhow::ensure!(
                std::fs::symlink_metadata(&path)?.is_file(),
                "reference must be a regular file"
            );
            let mut file = File::open(path)?;
            hash.update(file.metadata()?.len().to_le_bytes());
            std::io::copy(&mut file, &mut hash)?;
        }
        Ok(format!("{:x}", hash.finalize()))
    }
    fn ownership(&self) -> anyhow::Result<Materialization> {
        let marker = self.marker();
        anyhow::ensure!(
            std::fs::symlink_metadata(&marker)?.is_file(),
            "reference marker must be a regular file"
        );
        Ok(serde_json::from_reader(File::open(marker)?)?)
    }
    pub(super) fn fresh(&self, id: &str, fingerprint: &str) -> bool {
        self.ownership().is_ok_and(|marker| {
            marker.library_id == id
                && marker.fingerprint == fingerprint
                && self
                    .digest()
                    .is_ok_and(|digest| digest == marker.payload_digest)
        })
    }
    fn remove_payload(&self) -> std::io::Result<()> {
        match std::fs::symlink_metadata(self.path()) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
            Ok(metadata) if metadata.is_dir() => std::fs::remove_dir_all(self.path()),
            Ok(_) => std::fs::remove_file(self.path()),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Materialization {
    library_id: String,
    fingerprint: String,
    payload_digest: String,
}

// Explicit unlock also releases a lock when another test or host thread forks
// while this file is open; relying on the last descriptor closing can lag.
struct ReferenceLock(File);
impl Drop for ReferenceLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

/// Holds the lock and old payload until commit; cancellation restores the old entry.
struct Replacement<'a> {
    entry: &'a ReferenceEntry,
    backup: Option<tempfile::TempDir>,
    _lock: ReferenceLock,
    committed: bool,
}
impl<'a> Replacement<'a> {
    fn begin(
        entry: &'a ReferenceEntry,
        id: &str,
        fingerprint: &str,
    ) -> anyhow::Result<Option<Self>> {
        let parent = entry
            .path()
            .parent()
            .ok_or_else(|| anyhow::anyhow!("reference needs a parent"))?;
        std::fs::create_dir_all(parent)?;
        let lock = File::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(parent.join(".materialization.lock"))?;
        lock.try_lock()?;
        let lock = ReferenceLock(lock);
        if entry.fresh(id, fingerprint) {
            return Ok(None);
        }
        let exists = match std::fs::symlink_metadata(entry.path()) {
            Ok(metadata) => {
                anyhow::ensure!(
                    !metadata.file_type().is_symlink(),
                    "refusing a symbolic reference entry"
                );
                true
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => return Err(error.into()),
        };
        if exists || entry.marker().try_exists()? {
            let marker = entry.ownership().map_err(|error| anyhow::anyhow!(
                "reference collision for {id}: unmarked or unknown user data will not be replaced: {error}"
            ))?;
            anyhow::ensure!(
                marker.library_id == id,
                "reference collision for {id}: owned by another library asset"
            );
        }
        let backup = tempfile::tempdir_in(parent)?;
        if exists {
            std::fs::rename(entry.path(), backup.path().join("payload"))?;
        }
        Ok(Some(Self {
            entry,
            backup: Some(backup),
            _lock: lock,
            committed: false,
        }))
    }
    fn commit(mut self, voice: ReferenceVoice<'_>) -> anyhow::Result<()> {
        let marker = Materialization {
            library_id: voice.asset.id.clone(),
            fingerprint: voice.asset.fingerprint.clone(),
            payload_digest: self.entry.digest()?,
        };
        let mut file = tempfile::NamedTempFile::new_in(
            self.backup
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("reference transaction has no backup"))?
                .path(),
        )?;
        serde_json::to_writer_pretty(&mut file, &marker)?;
        file.flush()?;
        file.as_file().sync_all()?;
        file.persist(self.entry.marker())
            .map_err(|error| error.error)?;
        self.committed = true;
        Ok(())
    }
}
impl Drop for Replacement<'_> {
    fn drop(&mut self) {
        if !self.committed
            && let Some(backup) = self.backup.take()
        {
            let payload = backup.path().join("payload");
            if self.entry.remove_payload().is_err()
                || (payload.exists() && std::fs::rename(&payload, self.entry.path()).is_err())
            {
                // Never delete the last good copy if filesystem cleanup fails.
                let _ = backup.keep();
            }
        }
    }
}

pub(super) async fn install(
    adapter: &dyn ModelAdapter,
    root: &Path,
    voice: ReferenceVoice<'_>,
) -> anyhow::Result<()> {
    validate_id(&voice.asset.id)?;
    anyhow::ensure!(
        talechime_core::voices::library::fingerprint(
            &voice.asset.transcript,
            voice.asset.description.as_deref(),
            voice.wav
        ) == voice.asset.fingerprint,
        "reference recording does not match library fingerprint"
    );
    let custom = format!("custom:{}", voice.asset.id);
    let entry = adapter.reference_entry(root, &custom)?;
    let Some(replacement) = Replacement::begin(&entry, &voice.asset.id, &voice.asset.fingerprint)?
    else {
        return Ok(());
    };
    let mut wav = tempfile::NamedTempFile::new()?;
    wav.write_all(voice.wav)?;
    wav.flush()?;
    adapter
        .install_reference_async(
            root,
            wav.path(),
            &custom,
            &voice.asset.name,
            &voice.asset.transcript,
        )
        .await?;
    replacement.commit(voice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Installing, Loading, Preparing};
    use std::cell::Cell;
    use talechime_core::voices::library::{VoiceSource, fingerprint};
    use talechime_protocol::{Capabilities, Device, Event};
    use tokio::sync::mpsc;

    struct Adapter {
        file: bool,
        installs: Cell<usize>,
        fail: Cell<bool>,
        pending: Cell<bool>,
    }
    impl Adapter {
        fn new(file: bool) -> Self {
            Self {
                file,
                installs: Cell::new(0),
                fail: Cell::new(false),
                pending: Cell::new(false),
            }
        }
    }
    impl ModelAdapter for Adapter {
        fn id(&self) -> &'static str {
            "fixture"
        }
        fn capabilities(&self, _: &Path) -> anyhow::Result<Capabilities> {
            anyhow::bail!("must not enumerate corrupt references before repairing")
        }
        fn compiled_devices(&self) -> Vec<Device> {
            vec![]
        }
        fn available_devices(&self) -> Vec<Device> {
            vec![]
        }
        fn revision(&self) -> &'static str {
            "fixture"
        }
        fn prepare<'a>(&'a self, _: &'a Path, _: mpsc::Sender<Event>) -> Preparing<'a> {
            Box::pin(async { anyhow::bail!("no model preparation in reference tests") })
        }
        fn load<'a>(
            &'a self,
            _: &'a Path,
            _: mpsc::Sender<Event>,
            _: Device,
            _: bool,
        ) -> Loading<'a> {
            Box::pin(async { anyhow::bail!("no model loading in reference tests") })
        }
        fn reference_entry(&self, root: &Path, _: &str) -> anyhow::Result<ReferenceEntry> {
            Ok(if self.file {
                ReferenceEntry::File(root.join("hero.json"))
            } else {
                ReferenceEntry::Directory(root.join("hero"))
            })
        }
        fn install_reference_async<'a>(
            &'a self,
            root: &'a Path,
            wav: &'a Path,
            id: &'a str,
            _: &'a str,
            _: &'a str,
        ) -> Installing<'a> {
            Box::pin(async move {
                self.installs.set(self.installs.get() + 1);
                let entry = self.reference_entry(root, id)?;
                match &entry {
                    ReferenceEntry::File(path) => std::fs::copy(wav, path).map(|_| ())?,
                    ReferenceEntry::Directory(path) => {
                        std::fs::create_dir(path)?;
                        std::fs::copy(wav, path.join("reference.wav"))?;
                        std::fs::write(path.join("voice.json"), b"fixture metadata")?;
                    }
                }
                if self.fail.get() {
                    anyhow::bail!("fixture failure after writing");
                }
                if self.pending.get() {
                    std::future::pending::<()>().await;
                }
                Ok(())
            })
        }
    }
    fn asset(wav: &[u8]) -> VoiceAsset {
        VoiceAsset {
            id: "hero".into(),
            name: "Hero".into(),
            transcript: "Hello".into(),
            description: None,
            source: VoiceSource::Imported,
            created_at_unix: 0,
            fingerprint: fingerprint("Hello", None, wav),
        }
    }

    #[tokio::test]
    async fn directory_and_nano_shape_reuse_refresh_and_repair() -> anyhow::Result<()> {
        for file in [false, true] {
            let root = tempfile::tempdir()?;
            let adapter = Adapter::new(file);
            let old = asset(b"old");
            let voice = ReferenceVoice {
                asset: &old,
                wav: b"old",
            };
            install(&adapter, root.path(), voice).await?;
            install(&adapter, root.path(), voice).await?;
            assert_eq!(adapter.installs.get(), 1);
            let new = asset(b"new");
            let voice = ReferenceVoice {
                asset: &new,
                wav: b"new",
            };
            install(&adapter, root.path(), voice).await?;
            let entry = adapter.reference_entry(root.path(), "custom:hero")?;
            assert!(entry.fresh("hero", &new.fingerprint));
            assert!(!entry.fresh("hero", &old.fingerprint));
            let payload = if file {
                entry.path().to_path_buf()
            } else {
                entry.path().join("reference.wav")
            };
            std::fs::write(&payload, b"damaged")?;
            assert!(!entry.fresh("hero", &new.fingerprint));
            install(&adapter, root.path(), voice).await?;
            std::fs::remove_file(&payload)?;
            assert!(!entry.fresh("hero", &new.fingerprint));
            install(&adapter, root.path(), voice).await?;
            assert!(entry.fresh("hero", &new.fingerprint));
            assert_eq!(adapter.installs.get(), 4);
            assert_ne!(entry.marker().extension().unwrap(), "json");
        }
        Ok(())
    }

    #[tokio::test]
    async fn unmarked_unknown_and_foreign_entries_are_never_removed() -> anyhow::Result<()> {
        for file in [false, true] {
            let root = tempfile::tempdir()?;
            let adapter = Adapter::new(file);
            let entry = adapter.reference_entry(root.path(), "custom:hero")?;
            if file {
                std::fs::write(entry.path(), b"user audio")?;
            } else {
                std::fs::create_dir(entry.path())?;
                std::fs::write(entry.path().join("user.wav"), b"user audio")?;
            }
            let asset = asset(b"new");
            for marker in [
                None,
                Some(b"unknown".as_slice()),
                Some(
                    br#"{"library_id":"other","fingerprint":"x","payload_digest":"x"}"#.as_slice(),
                ),
            ] {
                if let Some(marker) = marker {
                    std::fs::write(entry.marker(), marker)?;
                }
                assert!(
                    install(
                        &adapter,
                        root.path(),
                        ReferenceVoice {
                            asset: &asset,
                            wav: b"new"
                        }
                    )
                    .await
                    .is_err()
                );
                let preserved = if file {
                    entry.path().to_path_buf()
                } else {
                    entry.path().join("user.wav")
                };
                assert_eq!(std::fs::read(preserved)?, b"user audio");
                assert_eq!(adapter.installs.get(), 0);
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn failed_or_cancelled_replacement_restores_owned_reference() -> anyhow::Result<()> {
        for file in [false, true] {
            let root = tempfile::tempdir()?;
            let adapter = Adapter::new(file);
            let old = asset(b"old");
            install(
                &adapter,
                root.path(),
                ReferenceVoice {
                    asset: &old,
                    wav: b"old",
                },
            )
            .await?;
            let new = asset(b"new");
            adapter.fail.set(true);
            assert!(
                install(
                    &adapter,
                    root.path(),
                    ReferenceVoice {
                        asset: &new,
                        wav: b"new"
                    }
                )
                .await
                .is_err()
            );
            let entry = adapter.reference_entry(root.path(), "custom:hero")?;
            assert!(entry.fresh("hero", &old.fingerprint));
            adapter.fail.set(false);
            adapter.pending.set(true);
            {
                let operation = install(
                    &adapter,
                    root.path(),
                    ReferenceVoice {
                        asset: &new,
                        wav: b"new",
                    },
                );
                tokio::pin!(operation);
                tokio::select! {
                    biased;
                    result = &mut operation => panic!("should remain pending: {result:?}"),
                    _ = std::future::ready(()) => {}
                }
                assert!(!entry.fresh("hero", &old.fingerprint));
            }
            assert!(entry.fresh("hero", &old.fingerprint));
        }
        Ok(())
    }
}
