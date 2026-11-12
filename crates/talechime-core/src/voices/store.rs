//! Private versioned storage, snapshots and short publication transactions.
use super::{
    LibraryDiagnostic, LibraryInventory, TransferControl, VoiceAsset, VoiceId, VoiceLibrary,
    VoiceLibraryError as Error, VoiceSource, asset,
};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

pub(super) const FORMAT: &str = "talechime/voice/2";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AssetData {
    id: VoiceId,
    name: String,
    transcript: String,
    description: Option<String>,
    source: VoiceSource,
    created_at_unix: u64,
    fingerprint: String,
    asset_digest: String,
    wav_bytes: u64,
    duration_seconds: f64,
}
impl From<&VoiceAsset> for AssetData {
    fn from(a: &VoiceAsset) -> Self {
        Self {
            id: a.id.clone(),
            name: a.name.clone(),
            transcript: a.transcript.clone(),
            description: a.description.clone(),
            source: a.source.clone(),
            created_at_unix: a.created_at_unix,
            fingerprint: a.fingerprint.clone(),
            asset_digest: a.asset_digest.clone(),
            wav_bytes: a.wav_bytes,
            duration_seconds: a.duration_seconds,
        }
    }
}
impl AssetData {
    pub(super) fn validated(self, imported_at_unix: u64) -> Result<VoiceAsset, Error> {
        let a = VoiceAsset {
            id: self.id,
            name: self.name,
            transcript: self.transcript,
            description: self.description,
            source: self.source,
            created_at_unix: self.created_at_unix,
            imported_at_unix,
            fingerprint: self.fingerprint,
            asset_digest: self.asset_digest,
            wav_bytes: self.wav_bytes,
            duration_seconds: self.duration_seconds,
        };
        a.validate()?;
        Ok(a)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    format: String,
    asset: AssetData,
    imported_at_unix: u64,
    origin: Option<String>,
}

pub(super) struct Snapshot {
    pub asset: VoiceAsset,
    pub wav: FrozenWav,
}

/// One lease protects an entire batch without retaining one descriptor per WAV.
pub(super) struct Scratch {
    _lease: File,
    directory: tempfile::TempDir,
}
impl Scratch {
    pub fn new() -> Result<std::sync::Arc<Self>, Error> {
        let directory = tempfile::Builder::new()
            .prefix(".talechime-voice-v2-batch-")
            .tempdir()?;
        let lease = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(directory.path().join(".lease"))?;
        lease.try_lock().map_err(|_| Error::Busy)?;
        Ok(std::sync::Arc::new(Self {
            _lease: lease,
            directory,
        }))
    }
    pub fn file(&self) -> Result<tempfile::NamedTempFile, Error> {
        Ok(tempfile::NamedTempFile::new_in(self.directory.path())?)
    }
    pub fn freeze(self: &std::sync::Arc<Self>, file: tempfile::NamedTempFile) -> FrozenWav {
        FrozenWav {
            path: file.into_temp_path(),
            _scratch: self.clone(),
        }
    }
}
pub(super) struct FrozenWav {
    path: tempfile::TempPath,
    _scratch: std::sync::Arc<Scratch>,
}
impl FrozenWav {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

pub(super) fn lock(library: &VoiceLibrary) -> Result<File, Error> {
    std::fs::create_dir_all(&library.root)?;
    let file = File::options()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(library.root.join(".lock"))?;
    file.try_lock().map_err(|e| match e {
        std::fs::TryLockError::WouldBlock => Error::Busy,
        std::fs::TryLockError::Error(e) => e.into(),
    })?;
    Ok(file)
}
fn read_record(library: &VoiceLibrary, id: &str) -> Result<(VoiceAsset, Option<String>), Error> {
    asset::validate_id(id)?;
    let directory = library.root.join(id);
    if !std::fs::symlink_metadata(&directory)?.is_dir() {
        return Err(Error::InvalidPackage("asset must be a directory".into()));
    }
    let path = directory.join("voice.json");
    if !std::fs::symlink_metadata(&path)?.is_file() {
        return Err(Error::InvalidPackage(
            "metadata must be a regular file".into(),
        ));
    }
    let bytes = bounded_bytes(File::open(path)?, 128 * 1024)?;
    let header: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| Error::InvalidPackage(e.to_string()))?;
    if header.get("format").and_then(|v| v.as_str()) != Some(FORMAT) {
        return Err(Error::UnsupportedFormat(
            header
                .get("format")
                .map(ToString::to_string)
                .unwrap_or_default(),
        ));
    }
    let record: Record =
        serde_json::from_slice(&bytes).map_err(|e| Error::InvalidPackage(e.to_string()))?;
    let a = record.asset.validated(record.imported_at_unix)?;
    if a.id() != id {
        return Err(Error::InvalidPackage("asset identity mismatch".into()));
    }
    let wav = std::fs::symlink_metadata(directory.join("reference.wav"))?;
    if !wav.is_file() || wav.len() != a.wav_bytes() {
        return Err(Error::InvalidPackage(
            "missing or invalid reference recording".into(),
        ));
    }
    Ok((a, record.origin))
}
pub(super) fn bounded_bytes(reader: impl Read, limit: u64) -> Result<Vec<u8>, Error> {
    let mut bytes = Vec::new();
    reader.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(Error::LimitExceeded("metadata".into()));
    }
    Ok(bytes)
}
pub(super) fn snapshot(library: &VoiceLibrary, id: &str) -> Result<Snapshot, Error> {
    snapshot_in(library, id, &Scratch::new()?)
}
pub(super) fn snapshot_in(
    library: &VoiceLibrary,
    id: &str,
    scratch: &std::sync::Arc<Scratch>,
) -> Result<Snapshot, Error> {
    let (a, _) = read_record(library, id)?;
    let path = library.root.join(id).join("reference.wav");
    if !std::fs::symlink_metadata(&path)?.is_file() {
        return Err(Error::InvalidPackage(
            "reference must be a regular file".into(),
        ));
    }
    let mut source = File::open(path)?;
    let mut wav = scratch.file()?;
    copy_checked(&mut source, &mut wav, &a, &TransferControl::default())?;
    Ok(Snapshot {
        asset: a,
        wav: scratch.freeze(wav),
    })
}
pub(super) fn copy_checked(
    reader: &mut impl Read,
    writer: &mut tempfile::NamedTempFile,
    a: &VoiceAsset,
    control: &TransferControl,
) -> Result<(), Error> {
    let mut hash = asset::reference_hasher(a.transcript(), a.description());
    let mut count = 0u64;
    let mut buffer = [0u8; 65536];
    loop {
        control.check()?;
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        count = count
            .checked_add(n as u64)
            .ok_or_else(|| Error::LimitExceeded("audio bytes".into()))?;
        if count > a.wav_bytes() {
            return Err(Error::LimitExceeded("audio bytes".into()));
        }
        hash.update(&buffer[..n]);
        writer.write_all(&buffer[..n])?;
    }
    writer.flush()?;
    if count != a.wav_bytes() || format!("{:x}", hash.finalize()) != a.fingerprint() {
        return Err(Error::ContentMismatch);
    }
    let duration = asset::validate_wav_with(File::open(writer.path())?, || control.check())?;
    if duration != a.duration_seconds() {
        return Err(Error::ContentMismatch);
    }
    Ok(())
}
pub(super) fn inventory(library: &VoiceLibrary) -> Result<LibraryInventory, Error> {
    if !library.root.exists() {
        return Ok(LibraryInventory::default());
    }
    let _lock = lock(library)?;
    inventory_unlocked(library)
}
pub(super) fn inventory_unlocked(library: &VoiceLibrary) -> Result<LibraryInventory, Error> {
    let mut result = LibraryInventory::default();
    if !library.root.exists() {
        return Ok(result);
    }
    for entry in std::fs::read_dir(&library.root)? {
        let entry = entry?;
        let id = entry.file_name().to_string_lossy().into_owned();
        if id.starts_with('.') {
            continue;
        }
        match read_record(library, &id) {
            Ok((a, _)) => result.assets.push(a),
            Err(error) => result.diagnostics.push(LibraryDiagnostic { id, error }),
        }
    }
    result.assets.sort_by(|a, b| a.id().cmp(b.id()));
    Ok(result)
}
pub(super) fn find_origin(
    library: &VoiceLibrary,
    origin: &str,
) -> Result<Option<VoiceAsset>, Error> {
    for a in inventory_unlocked(library)?.assets {
        if read_record(library, a.id())?.1.as_deref() == Some(origin) {
            return Ok(Some(a));
        }
    }
    Ok(None)
}
fn save(directory: &Path, a: &VoiceAsset, origin: Option<String>) -> Result<(), Error> {
    let mut file = File::create(directory.join("voice.json"))?;
    serde_json::to_writer_pretty(
        &mut file,
        &Record {
            format: FORMAT.into(),
            asset: AssetData::from(a),
            imported_at_unix: a.imported_at_unix,
            origin,
        },
    )
    .map_err(|e| Error::InvalidPackage(e.to_string()))?;
    file.flush()?;
    file.sync_all()?;
    Ok(())
}
/// Returns the existing asset for an idempotent replay, otherwise publishes the new asset.
pub(super) fn publish(
    library: &VoiceLibrary,
    mut a: VoiceAsset,
    wav: &Path,
    origin: Option<String>,
    control: &TransferControl,
) -> Result<(VoiceAsset, bool), Error> {
    std::fs::create_dir_all(&library.root)?;
    let staging = tempfile::Builder::new()
        .prefix(".staging-")
        .tempdir_in(&library.root)?;
    let lease = File::options()
        .create_new(true)
        .read(true)
        .write(true)
        .open(staging.path().join(".lease"))?;
    lease.try_lock().map_err(|_| Error::Busy)?;
    let mut frozen = tempfile::NamedTempFile::new_in(staging.path())?;
    copy_checked(&mut File::open(wav)?, &mut frozen, &a, control)?;
    frozen.as_file().sync_all()?;
    frozen
        .persist(staging.path().join("reference.wav"))
        .map_err(|e| Error::from(e.error))?;
    a.imported_at_unix = asset::now();
    save(staging.path(), &a, origin.clone())?;
    #[cfg(test)]
    FAIL_AFTER.with(|remaining| match remaining.get() {
        Some(0) => Err(Error::Disk(std::io::Error::from(
            std::io::ErrorKind::StorageFull,
        ))),
        Some(n) => {
            remaining.set(Some(n - 1));
            Ok(())
        }
        None => Ok(()),
    })?;
    let _lock = lock(library)?;
    if let Some(origin) = &origin
        && let Some(existing) = find_origin(library, origin)?
    {
        // Verify the asset actually present, not just its cached metadata.
        let verified = snapshot(library, existing.id())?;
        if verified.asset.fingerprint() != a.fingerprint() {
            return Err(Error::ContentMismatch);
        }
        return Ok((verified.asset, false));
    }
    let destination = library.root.join(a.id());
    if destination.exists() {
        let existing = snapshot(library, a.id())?.asset;
        if origin.is_some() && existing.fingerprint() == a.fingerprint() {
            return Ok((existing, false));
        }
        return Err(Error::Conflict(format!("voice {} already exists", a.id())));
    }
    control.check()?;
    // Windows cannot rename a directory containing a locked file. Recovery holds
    // the same library lock, so releasing the staging lease here cannot race it.
    drop(lease);
    std::fs::rename(staging.path(), destination)?;
    Ok((a, true))
}
pub(super) fn rename(library: &VoiceLibrary, id: &str, name: &str) -> Result<VoiceAsset, Error> {
    update_metadata(library, id, |asset| asset.name = name.into())
}
pub(super) fn describe(
    library: &VoiceLibrary,
    id: &str,
    description: Option<&str>,
) -> Result<VoiceAsset, Error> {
    update_metadata(library, id, |asset| {
        asset.description = description.map(str::to_owned);
    })
}

fn update_metadata(
    library: &VoiceLibrary,
    id: &str,
    update: impl FnOnce(&mut VoiceAsset),
) -> Result<VoiceAsset, Error> {
    let _lock = lock(library)?;
    let (mut a, origin) = read_record(library, id)?;
    update(&mut a);
    a.asset_digest = a.compute_digest()?;
    a.validate()?;
    let record = Record {
        format: FORMAT.into(),
        asset: AssetData::from(&a),
        imported_at_unix: a.imported_at_unix,
        origin,
    };
    crate::storage::save(&library.root.join(id).join("voice.json"), &record)?;
    Ok(a)
}
pub(super) fn remove(library: &VoiceLibrary, id: &str) -> Result<(), Error> {
    let _lock = lock(library)?;
    snapshot(library, id)?;
    let tombstone = tempfile::Builder::new()
        .prefix(".removed-")
        .tempdir_in(&library.root)?;
    std::fs::rename(library.root.join(id), tombstone.path().join("asset"))?;
    Ok(())
}

#[cfg(test)]
thread_local! { static FAIL_AFTER: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) }; }

/// Reclaim only our abandoned, current-format staging directories. Active owners
/// hold a lease; an absent lease is left alone to avoid racing initial creation.
pub(super) fn recover_staging(library: &VoiceLibrary) -> Result<(), Error> {
    if !library.root.exists() {
        return Ok(());
    }
    let _library_lock = match lock(library) {
        Ok(lock) => lock,
        Err(Error::Busy) => return Ok(()),
        Err(error) => return Err(error),
    };
    for entry in std::fs::read_dir(&library.root)? {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with(".staging-")
            || !entry.file_type()?.is_dir()
        {
            continue;
        }
        let path = entry.path();
        let lease_path = path.join(".lease");
        if !std::fs::symlink_metadata(&lease_path).is_ok_and(|m| m.is_file()) {
            continue;
        }
        if let Ok(lease) = File::options().read(true).write(true).open(lease_path)
            && lease.try_lock().is_ok()
        {
            drop(lease);
            let _ = std::fs::remove_dir_all(path);
        }
    }
    Ok(())
}

/// Protect the containing directory, rather than locking the content file.
/// Windows byte-range locks also reject reads through a separately opened handle.
pub(super) struct Temporary {
    file: tempfile::NamedTempFile,
    _scratch: std::sync::Arc<Scratch>,
}
impl std::ops::Deref for Temporary {
    type Target = tempfile::NamedTempFile;
    fn deref(&self) -> &Self::Target {
        &self.file
    }
}
impl std::ops::DerefMut for Temporary {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.file
    }
}
impl Write for Temporary {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.file.write(bytes)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}
pub(super) fn temporary() -> Result<Temporary, Error> {
    let scratch = Scratch::new()?;
    Ok(Temporary {
        file: scratch.file()?,
        _scratch: scratch,
    })
}
pub(super) fn recover_scratch() {
    let Ok(entries) = std::fs::read_dir(std::env::temp_dir()) else {
        return;
    };
    for entry in entries.flatten() {
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with(".talechime-voice-v2-")
        {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        if metadata
            .modified()
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_none_or(|age| age < std::time::Duration::from_secs(300))
        {
            continue;
        }
        if entry
            .file_name()
            .to_string_lossy()
            .starts_with(".talechime-voice-v2-batch-")
            && entry.file_type().is_ok_and(|t| t.is_dir())
        {
            let lease_path = entry.path().join(".lease");
            if std::fs::symlink_metadata(&lease_path).is_ok_and(|m| m.is_file())
                && let Ok(lease) = File::options().read(true).write(true).open(lease_path)
                && lease.try_lock().is_ok()
            {
                // Unique scratch directories are never reacquired by a new owner.
                // Release our own Windows lock before removing the abandoned directory.
                drop(lease);
                let _ = std::fs::remove_dir_all(entry.path());
            }
            continue;
        }
        if !entry.file_type().is_ok_and(|t| t.is_file()) {
            continue;
        }
        if let Ok(file) = File::options().read(true).write(true).open(entry.path())
            && file.try_lock().is_ok()
        {
            drop(file);
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voices::*;
    fn wav() -> Vec<u8> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        let mut writer = hound::WavWriter::new(
            &mut bytes,
            hound::WavSpec {
                channels: 1,
                sample_rate: 24000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        writer.write_sample(1i16).unwrap();
        writer.finalize().unwrap();
        bytes.into_inner()
    }
    fn fixture(root: &Path) -> (VoiceLibrary, Vec<std::path::PathBuf>) {
        let source = VoiceLibrary::open(root.join("source")).unwrap();
        for id in ["one", "two", "three"] {
            source
                .create(id, id, &wav(), "你好", None, VoiceSource::Imported)
                .unwrap();
        }
        let package = root.join("all.voice");
        source
            .export_bundle(
                &[
                    "one".parse().unwrap(),
                    "two".parse().unwrap(),
                    "three".parse().unwrap(),
                ],
                &package,
                ExportTarget::CreateNew,
                TransferLimits::default(),
                &TransferControl::default(),
            )
            .unwrap();
        (
            VoiceLibrary::open(root.join("dest")).unwrap(),
            vec![package],
        )
    }
    fn import(
        library: &VoiceLibrary,
        paths: &[std::path::PathBuf],
        control: &TransferControl,
    ) -> ImportReport {
        let preview = library
            .inspect_import(
                paths,
                TransferLimits::default(),
                &TransferControl::default(),
            )
            .unwrap();
        let decisions = preview
            .entries()
            .iter()
            .map(|e| ImportDecision {
                entry: e.index,
                action: ImportAction::Import,
            })
            .collect();
        library.import_selected(library.plan_import(preview, decisions).unwrap(), control)
    }
    #[test]
    fn disk_failure_keeps_completed_assets_and_cleans_unpublished_records() {
        let root = tempfile::tempdir().unwrap();
        let (library, paths) = fixture(root.path());
        FAIL_AFTER.with(|f| f.set(Some(1)));
        let report = import(&library, &paths, &TransferControl::default());
        FAIL_AFTER.with(|f| f.set(None));
        assert_eq!(report.items[0].status, ImportStatus::Imported);
        assert!(matches!(report.items[1].error, Some(Error::Disk(_))));
        assert_eq!(report.items[2].status, ImportStatus::NotAttempted);
        assert_eq!(library.list().unwrap().len(), 1);
        assert!(std::fs::read_dir(library.root()).unwrap().all(|e| {
            !e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".staging-")
        }));
        let report = import(&library, &paths, &TransferControl::default());
        assert_eq!(report.items[0].status, ImportStatus::AlreadyPresent);
        assert_eq!(library.list().unwrap().len(), 3);
    }
    #[test]
    fn interrupted_batch_can_be_reimported_without_duplicate_publication() {
        const KEY: &str = "TALECHIME_TRANSFER_CRASH_FIXTURE";
        if let Some(root) = std::env::var_os(KEY) {
            let root = std::path::PathBuf::from(root);
            let library = VoiceLibrary::open(root.join("dest")).unwrap();
            let control = TransferControl::with_progress(|p| {
                if matches!(p.stage, TransferStage::Committing) && p.completed == 1 {
                    std::process::exit(42)
                }
            });
            import(&library, &[root.join("all.voice")], &control);
            panic!("child did not stop");
        }
        let root = tempfile::tempdir().unwrap();
        let (library, paths) = fixture(root.path());
        let status=std::process::Command::new(std::env::current_exe().unwrap()).args(["--exact","voices::store::tests::interrupted_batch_can_be_reimported_without_duplicate_publication","--nocapture"]).env(KEY,root.path()).status().unwrap();
        assert_eq!(status.code(), Some(42));
        assert_eq!(library.list().unwrap().len(), 1);
        let report = import(&library, &paths, &TransferControl::default());
        assert_eq!(report.items[0].status, ImportStatus::AlreadyPresent);
        assert_eq!(library.list().unwrap().len(), 3);
    }
    #[test]
    fn startup_reclaims_abandoned_staging_but_preserves_active_leases() {
        let root = tempfile::tempdir().unwrap();
        let staging = root.path().join(".staging-test");
        std::fs::create_dir(&staging).unwrap();
        let lease = File::options()
            .create_new(true)
            .read(true)
            .write(true)
            .open(staging.join(".lease"))
            .unwrap();
        lease.try_lock().unwrap();
        VoiceLibrary::open(root.path()).unwrap();
        assert!(staging.exists());
        drop(lease);
        VoiceLibrary::open(root.path()).unwrap();
        assert!(!staging.exists());
    }
    fn age_directory(path: &Path) {
        let mut options = File::options();
        options.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // FILE_WRITE_ATTRIBUTES and FILE_FLAG_BACKUP_SEMANTICS for a directory handle.
            options.access_mode(0x100).custom_flags(0x02000000);
        }
        options
            .open(path)
            .unwrap()
            .set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(600))
            .unwrap();
    }
    #[test]
    fn abandoned_batch_scratch_is_reclaimed_without_removing_active_batches() {
        let scratch = Scratch::new().unwrap();
        let Scratch {
            _lease: lease,
            directory,
        } = std::sync::Arc::try_unwrap(scratch).ok().unwrap();
        let path = directory.keep();
        age_directory(&path);
        recover_scratch();
        assert!(path.exists());
        drop(lease);
        recover_scratch();
        assert!(!path.exists());
    }
    #[test]
    fn old_scratch_is_reclaimed_only_after_its_owner_releases_the_lease() {
        let scratch = temporary().unwrap();
        let Temporary {
            file,
            _scratch: owner,
        } = scratch;
        let owner = match std::sync::Arc::try_unwrap(owner) {
            Ok(owner) => owner,
            Err(_) => panic!("unexpected scratch owner"),
        };
        // Simulate a crashed owner: closing the content handle must leave its file
        // behind. NamedTempFile::drop would unlink it and refresh the directory age.
        let (file, content_path) = file.keep().unwrap();
        let path = owner.directory.keep();
        owner
            ._lease
            .set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(600))
            .unwrap();
        age_directory(&path);
        recover_scratch();
        assert!(path.exists());
        assert!(content_path.exists());
        drop(file);
        drop(owner._lease);
        recover_scratch();
        assert!(!path.exists());
    }

    #[test]
    fn leased_temporary_content_can_be_reopened_while_owner_is_alive() {
        let mut scratch = temporary().unwrap();
        scratch.write_all(b"frozen content").unwrap();
        scratch.flush().unwrap();
        assert_eq!(std::fs::read(scratch.path()).unwrap(), b"frozen content");
        let mut copy = Vec::new();
        scratch.reopen().unwrap().read_to_end(&mut copy).unwrap();
        assert_eq!(copy, b"frozen content");
        let path = scratch.path().to_owned();
        drop(scratch);
        assert!(!path.exists());
    }
}
