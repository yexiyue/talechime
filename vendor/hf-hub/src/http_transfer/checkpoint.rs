use super::CHECKPOINT_BYTES;
use crate::{
    error::HFResult,
    progress::{DownloadEvent, EmitEvent, FileProgress, FileStatus, Progress},
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

pub(super) fn checkpoint_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.ranges.json", path.display()))
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Plan {
    pub(super) version: u32,
    pub(super) total: u64,
    pub(super) chunk: u64,
    pub(super) etag: String,
    pub(super) completed: Vec<u64>,
}
impl Plan {
    pub(super) fn length(&self, index: usize) -> u64 {
        self.chunk.min(self.total - index as u64 * self.chunk)
    }
    pub(super) fn save(&self, path: &Path) -> HFResult<()> {
        let mut temp = tempfile::NamedTempFile::new_in(path.parent().unwrap_or(Path::new(".")))?;
        serde_json::to_writer(&mut temp, self)?;
        temp.as_file_mut().sync_all()?;
        temp.persist(path).map_err(|e| e.error)?;
        #[cfg(unix)]
        File::open(path.parent().unwrap_or(Path::new(".")))?.sync_all()?;
        Ok(())
    }
}
pub(super) struct Journal {
    pub(super) path: PathBuf,
    pub(super) plan: Mutex<Plan>,
    pub(super) observed: AtomicU64,
    pub(super) progress: Option<Progress>,
    pub(super) filename: String,
    pub(super) checkpoint_failed: Arc<AtomicBool>,
}
impl Journal {
    pub(super) fn emit(&self) {
        let total = self.plan.lock().unwrap().total;
        self.progress.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: self.filename.clone(),
                bytes_completed: self.observed.load(Ordering::Relaxed),
                total_bytes: total,
                status: FileStatus::InProgress,
            }],
        });
    }
}
pub(super) struct Writer {
    file: File,
    index: usize,
    position: u64,
    journal: Arc<Journal>,
    saved: u64,
    last_save: Instant,
}
impl Writer {
    pub(super) fn open(path: &Path, index: usize, journal: Arc<Journal>) -> HFResult<Self> {
        let plan = journal.plan.lock().unwrap();
        let position = plan.completed[index];
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)?;
        file.seek(SeekFrom::Start(index as u64 * plan.chunk + position))?;
        drop(plan);
        Ok(Self {
            file,
            index,
            position,
            journal,
            saved: position,
            last_save: Instant::now(),
        })
    }
    pub(super) fn save(&mut self) -> HFResult<()> {
        if self.position == self.saved {
            return Ok(());
        }
        self.file.sync_data()?;
        let mut plan = self.journal.plan.lock().unwrap();
        plan.completed[self.index] = self.position;
        plan.save(&self.journal.path)?;
        self.saved = self.position;
        self.last_save = Instant::now();
        Ok(())
    }
    pub(super) fn write(&mut self, bytes: &[u8]) -> HFResult<()> {
        self.file.write_all(bytes)?;
        self.position += bytes.len() as u64;
        self.journal
            .observed
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        self.journal.emit();
        if self.position - self.saved >= CHECKPOINT_BYTES
            || self.last_save.elapsed() >= Duration::from_secs(1)
        {
            self.save()?;
        }
        Ok(())
    }
}
impl Drop for Writer {
    fn drop(&mut self) {
        // Dropping a transfer (pause/cancel) checkpoints all bytes already written.
        if let Err(error) = self.save() {
            self.journal
                .checkpoint_failed
                .store(true, Ordering::Relaxed);
            tracing::error!(%error, "range checkpoint failed");
        }
    }
}
