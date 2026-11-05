//! Bounded HTTP ranges with a durable per-range checkpoint. The Hub cache lock
//! remains owned by the caller. No paths, credentials or signed URLs are persisted.
use crate::{
    error::{HFError, HFResult},
    progress::Progress,
};
use futures::{StreamExt, TryStreamExt};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::Read,
    path::Path,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

mod source;
pub(crate) use source::{Source, body_url};
mod checkpoint;
use checkpoint::{Journal, Plan, Writer, checkpoint_path};

const CHUNK: u64 = 4 * 1024 * 1024;
const CHECKPOINT_BYTES: u64 = 1024 * 1024;
const MAX_CHUNKS: u64 = 262_144;
static CONNECTIONS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
pub(crate) async fn permit() -> HFResult<tokio::sync::OwnedSemaphorePermit> {
    CONNECTIONS
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(4)))
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| HFError::malformed_response("download scheduler unavailable"))
}

enum RangeError {
    Unsupported,
    Hub(HFError),
}
impl From<HFError> for RangeError {
    fn from(e: HFError) -> Self {
        Self::Hub(e)
    }
}
impl From<reqwest::Error> for RangeError {
    fn from(e: reqwest::Error) -> Self {
        Self::Hub(e.into())
    }
}

/// Returns false if the server ignores ranges; the caller then uses its normal
/// sequential stream. Every in-flight range is dropped before returning false.
pub(crate) async fn download(
    source: &Source,
    path: &Path,
    filename: &str,
    progress: &Option<Progress>,
    force: bool,
) -> HFResult<bool> {
    let total = source.total;
    let etag = source.etag.as_str();
    let saved_path = checkpoint_path(path);
    if force {
        if saved_path.exists() {
            std::fs::remove_file(&saved_path)?;
        }
        File::create(path)?.sync_all()?;
    }
    let count = total.div_ceil(CHUNK);
    if count > MAX_CHUNKS {
        return Err(HFError::InvalidParameter(
            "model has too many download ranges".into(),
        ));
    }
    let plan = if saved_path.exists() {
        if std::fs::metadata(&saved_path)?.len() > 4 * 1024 * 1024 {
            return Err(HFError::malformed_response("download checkpoint too large"));
        }
        let plan: Plan = serde_json::from_reader(File::open(&saved_path)?)?;
        let file_len = std::fs::metadata(path)?.len();
        if plan.version != 1
            || plan.total != total
            || plan.etag != etag
            || plan.chunk != CHUNK
            || plan.completed.len() != count as usize
            || plan
                .completed
                .iter()
                .enumerate()
                .any(|(i, &n)| n > plan.length(i) || (n > 0 && i as u64 * CHUNK + n > file_len))
        {
            return Err(HFError::malformed_response(
                "invalid download range checkpoint",
            ));
        }
        plan
    } else {
        OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)?;
        // Import the contiguous prefix left by an ordinary HTTP stream.
        let prefix = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0)
            .min(total);
        let completed = (0..count)
            .map(|i| {
                prefix
                    .saturating_sub(i * CHUNK)
                    .min(CHUNK.min(total - i * CHUNK))
            })
            .collect();
        let plan = Plan {
            version: 1,
            total,
            chunk: CHUNK,
            etag: etag.into(),
            completed,
        };
        plan.save(&saved_path)?;
        plan
    };
    let initial: u64 = plan.completed.iter().sum();
    let pending: Vec<_> = (0..plan.completed.len())
        .filter(|&i| plan.completed[i] != plan.length(i))
        .collect();
    let journal = Arc::new(Journal {
        path: saved_path,
        plan: Mutex::new(plan),
        observed: AtomicU64::new(initial),
        progress: progress.clone(),
        filename: filename.into(),
        checkpoint_failed: source.client.inner.checkpoint_failed.clone(),
    });
    journal.emit();
    let mut transfers = futures::stream::iter(pending.into_iter().map(|index| {
        let journal = journal.clone();
        async move {
            for attempt in 0..3 {
                match range(source, path, index, journal.clone()).await {
                    Err(RangeError::Hub(e))
                        if e.is_transient()
                            && !matches!(&e, HFError::Request { source, .. } if source.is_timeout())
                            && attempt < 2 =>
                    {
                        tokio::time::sleep(Duration::from_millis(250 * (1 << attempt))).await
                    }
                    result => return result,
                }
            }
            unreachable!()
        }
    }))
    .buffer_unordered(source.client.inner.http_download_connections);
    while let Some(result) = transfers.next().await {
        match result {
            Ok(()) => {}
            Err(RangeError::Unsupported) => {
                drop(transfers);
                // Retain a valid state on disk even if cancellation happens before fallback.
                File::create(path)?.sync_all()?;
                std::fs::remove_file(&journal.path)?;
                return Ok(false);
            }
            Err(RangeError::Hub(e)) => return Err(e),
        }
    }
    Ok(true)
}
async fn range(
    source: &Source,
    path: &Path,
    index: usize,
    journal: Arc<Journal>,
) -> Result<(), RangeError> {
    let _permit = permit().await?;
    let (offset, end, total) = {
        let p = journal.plan.lock().unwrap();
        if p.completed[index] == p.length(index) {
            return Ok(());
        }
        (
            index as u64 * p.chunk + p.completed[index],
            index as u64 * p.chunk + p.length(index) - 1,
            p.total,
        )
    };
    let response = source.fetch(Some(&format!("bytes={offset}-{end}"))).await?;
    if response.status() == reqwest::StatusCode::OK {
        return Err(RangeError::Unsupported);
    }
    let response = source
        .client
        .check_response(response, None, crate::error::NotFoundContext::Generic)
        .await?;
    let expected = format!("bytes {offset}-{end}/{total}");
    if response.status() != reqwest::StatusCode::PARTIAL_CONTENT
        || response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|v| v.to_str().ok())
            != Some(expected.as_str())
    {
        return Err(
            HFError::malformed_response("unexpected Content-Range for parallel download").into(),
        );
    }
    let mut writer = Writer::open(path, index, journal)?;
    let mut received = 0;
    let mut stream = response.bytes_stream();
    while let Some(bytes) = stream.try_next().await? {
        received += bytes.len() as u64;
        if received > end - offset + 1 {
            return Err(
                HFError::malformed_response("download range exceeded expected length").into(),
            );
        }
        writer.write(&bytes)?;
    }
    if received != end - offset + 1 {
        return Err(HFError::malformed_response("truncated download range").into());
    }
    writer.save()?;
    Ok(())
}
/// Validate content-addressed large weights before publication. Dropping the
/// future cancels the background hash at the next bounded read.
pub(crate) async fn verify(path: &Path, etag: &str) -> HFResult<()> {
    if etag.len() != 64 || !etag.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Ok(());
    }
    struct Cancel(Arc<AtomicBool>);
    impl Drop for Cancel {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }
    let cancel = Cancel(Arc::new(AtomicBool::new(false)));
    let signal = cancel.0.clone();
    let path = path.to_owned();
    let expected = etag.to_ascii_lowercase();
    tokio::task::spawn_blocking(move || {
        let mut f = File::open(path)?;
        let mut hash = Sha256::new();
        let mut bytes = vec![0; 1024 * 1024];
        loop {
            if signal.load(Ordering::Relaxed) {
                return Err(HFError::malformed_response(
                    "download verification cancelled",
                ));
            }
            let count = f.read(&mut bytes)?;
            if count == 0 {
                break;
            }
            hash.update(&bytes[..count]);
        }
        if hash
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
            != expected
        {
            return Err(HFError::malformed_response("download SHA-256 mismatch"));
        }
        Ok(())
    })
    .await
    .map_err(|_| HFError::malformed_response("download verification task failed"))?
}
pub(crate) fn clear_checkpoint(path: &Path) -> HFResult<()> {
    let path = checkpoint_path(path);
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    Ok(())
}
