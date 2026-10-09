//! Execution-owned temporary PCM spool. No model, audio device or business IDs on I/O threads.
use super::{producer::Item, *};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
    sync::Mutex,
};

const MAGIC: &[u8; 8] = b"TCHSTG01";
const MAX_RECORD: usize = 16 * 1024 * 1024 + 6;

/// Bounds for private, disposable chapter storage; never a persistent audio cache.
#[derive(Debug, Clone)]
pub struct StagingOptions {
    /// Existing parent directory. None uses the OS temporary directory.
    pub directory: Option<PathBuf>,
    /// Maximum file size, including framing. Default: 1 GiB.
    pub max_bytes: u64,
    /// Maximum number of metadata and PCM records. Default: one million.
    pub max_records: u64,
}
/// Distinguishable disk, capacity and validation failures from private storage.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StagingError {
    /// Filesystem operation failed, including truncated input.
    #[error("chapter staging I/O: {0}")]
    Io(#[from] std::io::Error),
    /// Execution exceeded its configured disk or record bound.
    #[error("chapter staging capacity exceeded")]
    Capacity,
    /// Framing, source order, ownership or integrity validation failed.
    #[error("invalid chapter storage: {0}")]
    InvalidStorage(String),
}

impl Default for StagingOptions {
    fn default() -> Self {
        Self {
            directory: None,
            max_bytes: 1024 * 1024 * 1024,
            max_records: 1_000_000,
        }
    }
}
impl StagingOptions {
    pub(super) fn validate(&self) -> Result<(), SessionError> {
        if self.max_bytes < 21 || self.max_records == 0 {
            return Err(SessionError::Invalid(
                "invalid chapter storage limits".into(),
            ));
        }
        if let Some(parent) = &self.directory
            && !parent.is_dir()
        {
            return Err(SessionError::Invalid(
                "chapter storage parent is not a directory".into(),
            ));
        }
        Ok(())
    }
}

pub(super) struct Storage {
    // Close handles before TempDir removal (required on Windows).
    file: Mutex<File>,
    _owner: File,
    // I/O closures retain this owner until their operation finishes.
    _directory: tempfile::TempDir,
}

async fn io<T: Send + 'static>(
    writes: Arc<PendingWrites>,
    work: impl FnOnce() -> Result<T, SessionError> + Send + 'static,
) -> Result<T, SessionError> {
    writes.count.fetch_add(1, Ordering::SeqCst);
    let guard = WriteGuard(writes);
    let (result, _guard) = tokio::task::spawn_blocking(move || (work(), guard))
        .await
        .map_err(|error| SessionError::Invalid(error.to_string()))?;
    result
}
fn invalid(message: &str) -> SessionError {
    StagingError::InvalidStorage(message.into()).into()
}
fn checksum(kind: u8, bytes: &[u8]) -> u64 {
    std::iter::once(kind)
        .chain(bytes.iter().copied())
        .fold(0xcbf29ce484222325, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
}
// Records are self-framed, versioned and checksummed; source markers form the sequential index.
enum Record {
    Start(TextRange),
    Audio(Pcm),
    End(TextRange),
    Skipped(TextRange),
    Finished,
}
impl Record {
    fn from_item(item: Item) -> Self {
        match item {
            Item::Start(r) => Self::Start(r),
            Item::End(r) => Self::End(r),
            Item::Skipped(r) => Self::Skipped(r),
            Item::Finished => Self::Finished,
            Item::Audio(packet) => Self::Audio(packet.audio), // release playback budget immediately
        }
    }
    fn encode(self) -> Vec<u8> {
        match self {
            Self::Start(r) | Self::End(r) | Self::Skipped(r) => {
                let mut bytes = Vec::with_capacity(16);
                bytes.extend_from_slice(&(r.start as u64).to_le_bytes());
                bytes.extend_from_slice(&(r.end as u64).to_le_bytes());
                bytes
            }
            Self::Audio(pcm) => {
                let mut bytes = Vec::with_capacity(6 + pcm.samples.len() * 4);
                bytes.extend_from_slice(&pcm.sample_rate.to_le_bytes());
                bytes.extend_from_slice(&pcm.channels.to_le_bytes());
                for sample in pcm.samples {
                    bytes.extend_from_slice(&sample.to_le_bytes());
                }
                bytes
            }
            Self::Finished => vec![],
        }
    }
    fn kind(&self) -> u8 {
        match self {
            Self::Start(_) => 1,
            Self::Audio(_) => 2,
            Self::End(_) => 3,
            Self::Skipped(_) => 4,
            Self::Finished => 5,
        }
    }
}
fn write_record(
    file: &mut File,
    record: Record,
    bytes: &mut u64,
    count: &mut u64,
    limits: &StagingOptions,
) -> Result<(), SessionError> {
    let kind = record.kind();
    let payload = record.encode();
    let size = 13 + payload.len() as u64;
    if payload.len() > MAX_RECORD
        || size > limits.max_bytes.saturating_sub(*bytes)
        || *count >= limits.max_records
    {
        return Err(StagingError::Capacity.into());
    }
    file.write_all(&[kind])?;
    file.write_all(&(payload.len() as u32).to_le_bytes())?;
    file.write_all(&checksum(kind, &payload).to_le_bytes())?;
    file.write_all(&payload)?;
    *bytes += size;
    *count += 1;
    Ok(())
}
fn read_record(file: &mut File) -> Result<Record, SessionError> {
    let mut header = [0u8; 13];
    file.read_exact(&mut header)?;
    let size = u32::from_le_bytes(header[1..5].try_into().unwrap()) as usize;
    if size > MAX_RECORD {
        return Err(invalid("record too large"));
    }
    let mut payload = vec![0u8; size];
    file.read_exact(&mut payload)?;
    if checksum(header[0], &payload) != u64::from_le_bytes(header[5..13].try_into().unwrap()) {
        return Err(invalid("checksum mismatch"));
    }
    match header[0] {
        kind @ (1 | 3 | 4) if size == 16 => {
            let range = TextRange {
                start: usize::try_from(u64::from_le_bytes(payload[..8].try_into().unwrap()))
                    .map_err(|_| invalid("range overflow"))?,
                end: usize::try_from(u64::from_le_bytes(payload[8..].try_into().unwrap()))
                    .map_err(|_| invalid("range overflow"))?,
            };
            Ok(match kind {
                1 => Record::Start(range),
                3 => Record::End(range),
                _ => Record::Skipped(range),
            })
        }
        2 if size >= 10 && (size - 6).is_multiple_of(4) => {
            let pcm = Pcm {
                sample_rate: u32::from_le_bytes(payload[..4].try_into().unwrap()),
                channels: u16::from_le_bytes(payload[4..6].try_into().unwrap()),
                samples: payload[6..]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
            };
            if pcm
                .duration_ms()
                .map_err(|error| StagingError::InvalidStorage(error.to_string()))?
                > 30_000
            {
                return Err(invalid("PCM exceeds replay budget"));
            }
            Ok(Record::Audio(pcm))
        }
        5 if size == 0 => Ok(Record::Finished),
        _ => Err(invalid("invalid record")),
    }
}
fn rewind(file: &mut File) -> Result<(), SessionError> {
    file.seek(SeekFrom::Start(0))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(invalid("unsupported version"));
    }
    Ok(())
}
fn verify(file: &mut File, text: &str, mut at: usize) -> Result<(), SessionError> {
    rewind(file)?;
    let mut active = None;
    let mut format = None;
    loop {
        match read_record(file)? {
            Record::Start(range) | Record::Skipped(range)
                if !range.is_valid(text) || range.start != at || range.start >= range.end =>
            {
                return Err(invalid("invalid source index"));
            }
            Record::Start(range) if active.is_none() => {
                active = Some(range);
                format = None;
            }
            Record::Audio(pcm) if active.is_some() => {
                let current = (pcm.sample_rate, pcm.channels);
                if format.is_some_and(|old| old != current) {
                    return Err(invalid("PCM format changed"));
                }
                format = Some(current);
            }
            Record::End(range) if active == Some(range) && format.is_some() => {
                at = range.end;
                active = None;
            }
            Record::Skipped(range) if active.is_none() => {
                at = range.end;
            }
            Record::Finished if active.is_none() && at == text.len() => {
                let mut trailing = [0];
                if file.read(&mut trailing)? != 0 {
                    return Err(invalid("trailing data"));
                }
                return rewind(file);
            }
            _ => return Err(invalid("incomplete or unordered chapter")),
        }
    }
}

const OWNER: &[u8] = b"talechime-private-staging-v1\n";
const ROOT: &str = "talechime-staging-v1";

fn create_storage(options: &StagingOptions) -> Result<Arc<Storage>, SessionError> {
    let parent = options.directory.clone().unwrap_or_else(std::env::temp_dir);
    let root = parent.join(ROOT);
    match std::fs::create_dir(&root) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error.into()),
    }
    if !std::fs::symlink_metadata(&root)?.file_type().is_dir() {
        return Err(invalid("managed root is not a real directory"));
    }
    let lock_path = root.join(".owner");
    match std::fs::symlink_metadata(&lock_path) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(invalid("managed root lock is not a regular file"));
        }
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error.into()),
        _ => {}
    }
    let mut root_lock = match File::options().read(true).write(true).open(&lock_path) {
        Ok(file) => {
            file.lock()?;
            file
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // Publish a complete locked marker atomically, including concurrent first use.
            let mut marker = tempfile::NamedTempFile::new_in(&root)?;
            marker.write_all(OWNER)?;
            marker.as_file().sync_all()?;
            marker.as_file().lock()?;
            match marker.persist_noclobber(&lock_path) {
                Ok(mut file) => {
                    file.rewind()?;
                    file
                }
                Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let file = File::options().read(true).write(true).open(&lock_path)?;
                    file.lock()?;
                    file
                }
                Err(error) => return Err(error.error.into()),
            }
        }
        Err(error) => return Err(error.into()),
    };
    let mut marker = Vec::new();
    Read::by_ref(&mut root_lock)
        .take(OWNER.len() as u64 + 1)
        .read_to_end(&mut marker)?;
    if marker != OWNER {
        return Err(invalid("unrecognized managed root"));
    }
    cleanup_orphans(&root)?;
    let directory = tempfile::Builder::new()
        .prefix("execution-")
        .tempdir_in(&root)?;
    let mut owner = File::options()
        .read(true)
        .write(true)
        .create_new(true)
        .open(directory.path().join(".owner"))?;
    owner.try_lock().map_err(std::io::Error::other)?;
    owner.write_all(OWNER)?;
    owner.sync_all()?;
    let mut file = File::options()
        .read(true)
        .write(true)
        .create_new(true)
        .open(directory.path().join("chapter.pcm"))?;
    file.write_all(MAGIC)?;
    Ok(Arc::new(Storage {
        file: Mutex::new(file),
        _owner: owner,
        _directory: directory,
    }))
}

// Only our marked, unlocked directories containing known regular files qualify.
// Unknown files, symlinks and live executions are never removed.
fn cleanup_orphans(root: &std::path::Path) -> Result<(), SessionError> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with("execution-")
            || !entry.file_type()?.is_dir()
        {
            continue;
        }
        let path = entry.path();
        let contents = std::fs::read_dir(&path)?.collect::<Result<Vec<_>, _>>()?;
        if contents.iter().any(|item| {
            !matches!(item.file_name().to_str(), Some(".owner" | "chapter.pcm"))
                || !item.file_type().is_ok_and(|kind| kind.is_file())
        }) {
            continue;
        }
        let owner_path = path.join(".owner");
        if !owner_path.exists() {
            continue;
        }
        let mut owner = File::options().read(true).write(true).open(owner_path)?;
        if owner.try_lock().is_err() {
            continue;
        }
        let mut marker = Vec::new();
        Read::by_ref(&mut owner)
            .take(OWNER.len() as u64 + 1)
            .read_to_end(&mut marker)?;
        if marker != OWNER {
            continue;
        }
        drop(owner);
        std::fs::remove_dir_all(path)?;
    }
    Ok(())
}

pub(super) async fn prepare(
    rx: &mut mpsc::Receiver<Result<Item, SessionError>>,
    options: &StagingOptions,
    writes: Arc<PendingWrites>,
    text: &str,
    byte: usize,
) -> Result<Arc<Storage>, SessionError> {
    let limits = options.clone();
    let storage = io(writes.clone(), move || create_storage(&limits)).await?;
    let mut bytes = 8;
    let mut count = 0;
    loop {
        let record = Record::from_item(
            rx.recv()
                .await
                .ok_or_else(|| invalid("generation disconnected"))??,
        );
        let finished = matches!(record, Record::Finished);
        let owner = storage.clone();
        let limits = options.clone();
        (bytes, count) = io(writes.clone(), move || {
            write_record(
                &mut owner.file.lock().unwrap(),
                record,
                &mut bytes,
                &mut count,
                &limits,
            )?;
            Ok((bytes, count))
        })
        .await?;
        if finished {
            break;
        }
    }
    let owner = storage.clone();
    let text = text.to_owned();
    io(writes, move || {
        let mut file = owner.file.lock().unwrap();
        file.sync_all()?;
        verify(&mut file, &text, byte)
    })
    .await?;
    Ok(storage)
}

pub(super) fn replay(
    storage: Arc<Storage>,
    writes: Arc<PendingWrites>,
) -> (AbortOnDrop, mpsc::Receiver<Result<Item, SessionError>>) {
    // Track the reader lifetime too: aborting its parent does not synchronously
    // poll/drop a local child task blocked on playback budget or channel space.
    struct ReaderLifetime {
        storage: Arc<Storage>,
        _guard: WriteGuard,
    }
    writes.count.fetch_add(1, Ordering::SeqCst);
    let lifetime = ReaderLifetime {
        storage,
        _guard: WriteGuard(writes.clone()),
    };
    let (tx, rx) = mpsc::channel(1);
    let task = tokio::task::spawn_local(async move {
        let reader = lifetime;
        let budget = Budget::default();
        let result: Result<(), SessionError> = async {
            loop {
                let owner = reader.storage.clone();
                let record = io(writes.clone(), move || {
                    read_record(&mut owner.file.lock().unwrap())
                })
                .await?;
                let item = match record {
                    Record::Start(r) => Item::Start(r),
                    Record::End(r) => Item::End(r),
                    Record::Skipped(r) => Item::Skipped(r),
                    Record::Audio(pcm) => Item::Audio(budget.acquire(pcm).await?),
                    Record::Finished => Item::Finished,
                };
                let finished = matches!(item, Item::Finished);
                tx.send(Ok(item))
                    .await
                    .map_err(|_| SessionError::Disconnected)?;
                if finished {
                    return Ok(());
                }
            }
        }
        .await;
        if let Err(error) = result {
            let _ = tx.send(Err(error)).await;
        }
    });
    (AbortOnDrop(task), rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> File {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(MAGIC).unwrap();
        let limits = StagingOptions::default();
        let mut bytes = 8;
        let mut count = 0;
        for record in [
            Record::Start(TextRange { start: 0, end: 3 }),
            Record::Audio(Pcm {
                sample_rate: 1000,
                channels: 1,
                samples: vec![0.1; 10],
            }),
            Record::End(TextRange { start: 0, end: 3 }),
            Record::Finished,
        ] {
            write_record(&mut file, record, &mut bytes, &mut count, &limits).unwrap();
        }
        file
    }
    #[test]
    fn checks_version_checksum_truncation_and_trailing_data() {
        let mut file = fixture();
        verify(&mut file, "甲", 0).unwrap();
        let size = file.metadata().unwrap().len();
        file.seek(SeekFrom::End(0)).unwrap();
        file.write_all(&[1]).unwrap();
        assert!(verify(&mut file, "甲", 0).is_err());
        file.set_len(size - 1).unwrap();
        assert!(verify(&mut file, "甲", 0).is_err());
        let mut file = fixture();
        file.seek(SeekFrom::Start(25)).unwrap();
        file.write_all(&[255]).unwrap();
        assert!(verify(&mut file, "甲", 0).is_err());
        let mut file = fixture();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(b"UNKNOWN!").unwrap();
        assert!(verify(&mut file, "甲", 0).is_err());
    }
    #[test]
    fn rejects_incomplete_utf8_index_and_record_limit() {
        let mut file = fixture();
        assert!(verify(&mut file, "甲乙", 0).is_err());
        assert!(verify(&mut file, "ab中", 0).is_err());
        let mut file = tempfile::tempfile().unwrap();
        let limits = StagingOptions {
            max_records: 1,
            ..Default::default()
        };
        let mut bytes = 8;
        let mut count = 0;
        write_record(
            &mut file,
            Record::Start(TextRange { start: 0, end: 3 }),
            &mut bytes,
            &mut count,
            &limits,
        )
        .unwrap();
        assert!(
            write_record(&mut file, Record::Finished, &mut bytes, &mut count, &limits).is_err()
        );
    }
}

#[cfg(test)]
mod cleanup_tests {
    use super::*;
    #[test]
    fn only_owned_unlocked_orphans_are_removed() {
        let parent = tempfile::tempdir().unwrap();
        let options = StagingOptions {
            directory: Some(parent.path().into()),
            ..Default::default()
        };
        let active = create_storage(&options).unwrap();
        let root = parent.path().join(ROOT);
        let orphan = root.join("execution-orphan");
        std::fs::create_dir(&orphan).unwrap();
        std::fs::write(orphan.join(".owner"), OWNER).unwrap();
        std::fs::write(orphan.join("chapter.pcm"), b"partial").unwrap();
        let unknown = root.join("execution-user");
        std::fs::create_dir(&unknown).unwrap();
        std::fs::write(unknown.join(".owner"), OWNER).unwrap();
        std::fs::write(unknown.join("keep"), b"user").unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&orphan, root.join("execution-link")).unwrap();
        }
        let second = create_storage(&options).unwrap();
        assert!(!orphan.exists());
        assert!(unknown.join("keep").exists());
        assert!(active._directory.path().exists());
        #[cfg(unix)]
        assert!(
            root.join("execution-link")
                .symlink_metadata()
                .unwrap()
                .file_type()
                .is_symlink()
        );
        let active_path = active._directory.path().to_owned();
        drop(active);
        assert!(!active_path.exists());
        drop(second);
    }
    #[test]
    fn unrecognized_root_and_disk_errors_are_explicit() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join(ROOT);
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join(".owner"), b"user data").unwrap();
        let options = StagingOptions {
            directory: Some(parent.path().into()),
            ..Default::default()
        };
        assert!(matches!(
            create_storage(&options),
            Err(SessionError::Staging(StagingError::InvalidStorage(_)))
        ));
        assert_eq!(std::fs::read(root.join(".owner")).unwrap(), b"user data");
        let options = StagingOptions {
            directory: Some(parent.path().join("missing")),
            ..Default::default()
        };
        assert!(matches!(
            create_storage(&options),
            Err(SessionError::Staging(StagingError::Io(_)))
        ));
    }
}
