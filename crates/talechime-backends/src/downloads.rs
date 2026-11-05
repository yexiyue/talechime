//! Shared HF Hub downloads with host-configured cache, endpoint and token.
//!
//! Model weights live in the content-addressed HF cache (blobs named by etag,
//! snapshotted per revision) so other Hub tools on the same machine share them.
//! Engine-layout entries are hardlinks into that cache: no second copy, and the
//! link keeps the inode alive even if the cache is later cleaned. The engine
//! layout stays the consumption view adapters load from.
//!
//! `DownloadConfig` must carry an absolute cache directory. Cache, endpoint and
//! optional token are passed explicitly instead of inheriting hf-hub's implicit
//! environment or token-file configuration. `default_cache_dir` below is NOT in
//! `talechime-core` paths.rs: the shared cache belongs to the user's Hugging
//! Face ecosystem, not to TaleChime's own `~/.talechime` layout.
use sha2::Digest;
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use hf_hub::progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler};
use talechime_protocol::Event;
use tokio::sync::{mpsc, watch};

/// Host-provided download configuration.
#[derive(Clone)]
pub struct DownloadConfig {
    /// Absolute HF cache directory (e.g. `~/.cache/huggingface/hub`).
    pub cache_dir: PathBuf,
    /// Hub endpoint override such as a mirror; `None` uses the official host.
    pub endpoint: Option<String>,
    /// Explicit Hub access token; `None` requests anonymous access.
    pub token: Option<String>,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            cache_dir: default_cache_dir(),
            endpoint: None,
            token: None,
        }
    }
}

impl std::fmt::Debug for DownloadConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DownloadConfig")
            .field("cache_dir", &self.cache_dir)
            .field("endpoint_configured", &self.endpoint.is_some())
            .field("token_configured", &self.token.is_some())
            .finish()
    }
}

/// Shared-cache default matching the Python `huggingface_hub` ecosystem layout.
pub fn default_cache_dir() -> PathBuf {
    std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/huggingface/hub")
}

/// Transfer policy, independent of cache and credentials. Clients share four
/// active range slots; Xet handles chunk concurrency within each range.
#[derive(Debug, Clone, Copy)]
pub struct DownloadOptions {
    /// Concurrent missing ranges per file (1..=4); defaults to two.
    pub connections: usize,
    /// Prefer Xet when advertised by the endpoint, falling back to HTTP by default.
    pub transport: DownloadTransport,
}
/// Transport selection independent of credentials, endpoints and cache ownership.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum DownloadTransport {
    /// Prefer Xet; use HTTP when unavailable. Both paths retain range checkpoints.
    #[default]
    Auto,
    /// Use HTTP only.
    Http,
    /// Prefer Xet and surface Xet failures; non-Xet files still use HTTP.
    Xet,
}
impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            connections: 2,
            transport: DownloadTransport::Auto,
        }
    }
}

/// Cheap-to-clone download client bound to one configuration.
#[derive(Clone)]
pub struct Downloads {
    client: hf_hub::HFClient,
    cache_dir: PathBuf,
    source_identity: String,
    transport: DownloadTransport,
}

impl std::fmt::Debug for Downloads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Downloads").finish_non_exhaustive()
    }
}

impl Downloads {
    pub fn new(config: &DownloadConfig) -> anyhow::Result<Self> {
        Self::with_options(config, DownloadOptions::default())
    }
    /// Construct a client with explicit per-file concurrency (1..=4).
    pub fn with_options(config: &DownloadConfig, options: DownloadOptions) -> anyhow::Result<Self> {
        anyhow::ensure!(
            (1..=4).contains(&options.connections),
            "download connections must be 1..=4"
        );
        anyhow::ensure!(
            config.cache_dir.is_absolute(),
            "download cache directory must be absolute, got {}",
            config.cache_dir.display()
        );
        let builder = hf_hub::HFClient::builder()
            .cache_dir(config.cache_dir.clone())
            .endpoint(
                config
                    .endpoint
                    .as_deref()
                    .unwrap_or("https://huggingface.co"),
            )
            .maybe_token(config.token.clone())
            .download_connections(options.connections)
            .retry_max_attempts(1);
        Ok(Self {
            client: builder.build()?,
            cache_dir: config.cache_dir.clone(),
            source_identity: format!(
                "{:x}",
                sha2::Sha256::digest(serde_json::to_vec(&(
                    &config.cache_dir,
                    config
                        .endpoint
                        .as_deref()
                        .unwrap_or("https://huggingface.co"),
                    config.token.as_deref(),
                    options.connections,
                    format!("{:?}", options.transport),
                ))?)
            ),
            transport: options.transport,
        })
    }

    /// Shared cache used by this explicitly configured client.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    pub(crate) fn source_identity(&self) -> &str {
        &self.source_identity
    }

    pub(crate) fn cached_path(
        &self,
        repo: &str,
        revision: &str,
        filename: &str,
    ) -> Option<PathBuf> {
        let (owner, name) = split_repo(repo).ok()?;
        let path = self
            .cache_dir
            .join(format!("models--{owner}--{name}"))
            .join("snapshots")
            .join(revision)
            .join(filename);
        let canonical = path.canonicalize().ok()?;
        canonical
            .starts_with(self.cache_dir.canonicalize().ok()?)
            .then_some(canonical)
    }

    /// Test-only client with an isolated cache at `root/hf-cache` and the
    /// official endpoint; nothing is written until a download happens.
    #[cfg(test)]
    pub(crate) fn test(root: &Path) -> Self {
        Self::new(&DownloadConfig {
            cache_dir: root.join("hf-cache"),
            endpoint: None,
            token: None,
        })
        .expect("test downloads")
    }

    /// Check after dropping a transfer before acknowledging a durable pause.
    pub fn checkpoint_failed(&self) -> bool {
        self.client.download_checkpoint_failed()
    }

    /// Ensure one pinned manifest entry is usable at `dest` (engine layout).
    ///
    /// An existing `dest` is kept only when its size matches the manifest.
    /// A truncated hardlink also requires refreshing its shared cache blob.
    /// Dropping the returned future cancels the transfer.
    pub async fn ensure(
        &self,
        dest: &Path,
        repo: &str,
        revision: &str,
        filename: &str,
        progress: FileProgressBridge,
    ) -> anyhow::Result<()> {
        self.ensure_inner(dest, repo, revision, filename, progress, false)
            .await
    }

    pub(crate) async fn refresh(
        &self,
        dest: &Path,
        repo: &str,
        revision: &str,
        filename: &str,
        progress: FileProgressBridge,
    ) -> anyhow::Result<()> {
        self.ensure_inner(dest, repo, revision, filename, progress, true)
            .await
    }

    async fn ensure_inner(
        &self,
        dest: &Path,
        repo: &str,
        revision: &str,
        filename: &str,
        progress: FileProgressBridge,
        refresh: bool,
    ) -> anyhow::Result<()> {
        let expected_size = progress.size;
        if !refresh && crate::resource_state::entry_ready(dest, expected_size) {
            return Ok(());
        }
        let relay = ProgressRelay {
            tx: progress.tx.clone(),
            resource: progress.resource.clone(),
            size: progress.size,
            latest: progress.latest.subscribe(),
        };
        relay
            .run(async {
                let (owner, name) = split_repo(repo)?;
                let model = self.client.model(owner, name);
                let progress = hf_hub::progress::Progress::new(progress);
                let download = model
                    .download_file()
                    .filename(filename)
                    .revision(revision)
                    .use_xet(self.transport != DownloadTransport::Http)
                    .http_fallback(self.transport == DownloadTransport::Auto)
                    .progress(progress);
                let initial = if refresh {
                    download.clone().force_download(true).send().await
                } else {
                    download.clone().send().await
                };
                let mut path = initial
                    .map_err(|error| download_error(repo, revision, filename, classify(error)))?;
                // A cache hit may itself be truncated (the layout can share its inode).
                // Let hf-hub replace the blob using its own locks and atomic writes.
                if !crate::resource_state::entry_ready(&path, expected_size) {
                    path = download
                        .force_download(true)
                        .send()
                        .await
                        .map_err(|error| {
                            download_error(repo, revision, filename, classify(error))
                        })?;
                }
                anyhow::ensure!(
                    crate::resource_state::entry_ready(&path, expected_size),
                    "downloaded file size does not match manifest: {repo}@{revision}/{filename}"
                );
                link_into_layout(&path, dest).map_err(|error| {
                    download_error(
                        repo,
                        revision,
                        filename,
                        DownloadFailure::Disk(error.to_string()),
                    )
                })
            })
            .await
    }
}

/// Typed download failure so hosts can react without string matching.
/// Carried inside the anyhow chain; downcast to read.
#[derive(Debug)]
pub enum DownloadFailure {
    /// Transport or server communication failure.
    Network(String),
    /// Missing or rejected credentials.
    Auth(String),
    /// Hub rate limiting.
    RateLimited,
    /// Pinned repo/revision/file missing upstream.
    NotFound(String),
    /// Local cache problems (locks, disabled cache).
    Cache(String),
    /// Local filesystem write or linking failure.
    Disk(String),
    /// Anything else.
    Other(String),
}

impl std::fmt::Display for DownloadFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DownloadFailure::Network(detail) => write!(f, "network failure: {detail}"),
            DownloadFailure::Auth(detail) => write!(f, "authentication failed: {detail}"),
            DownloadFailure::RateLimited => write!(f, "rate limited by the hub"),
            DownloadFailure::NotFound(detail) => write!(f, "not found upstream: {detail}"),
            DownloadFailure::Cache(detail) => write!(f, "cache failure: {detail}"),
            DownloadFailure::Disk(detail) => write!(f, "disk failure: {detail}"),
            DownloadFailure::Other(detail) => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for DownloadFailure {}

fn classify(error: hf_hub::HFError) -> DownloadFailure {
    use hf_hub::HFError;
    match error {
        HFError::AuthRequired { .. } | HFError::Forbidden { .. } => {
            DownloadFailure::Auth(error.to_string())
        }
        HFError::RateLimited { .. } => DownloadFailure::RateLimited,
        HFError::RepoNotFound { .. }
        | HFError::RevisionNotFound { .. }
        | HFError::EntryNotFound { .. }
        | HFError::BucketNotFound { .. } => DownloadFailure::NotFound(error.to_string()),
        HFError::LocalEntryNotFound { .. }
        | HFError::CacheLockTimeout { .. }
        | HFError::CacheNotEnabled => DownloadFailure::Cache(error.to_string()),
        HFError::Request { source, .. } => DownloadFailure::Network(source.to_string()),
        HFError::Xet { source, .. } => DownloadFailure::Network(source.to_string()),
        HFError::Io(source) => DownloadFailure::Disk(source.to_string()),
        other => DownloadFailure::Other(other.to_string()),
    }
}

fn download_error(
    repo: &str,
    revision: &str,
    filename: &str,
    failure: DownloadFailure,
) -> anyhow::Error {
    anyhow::Error::new(failure).context(format!("download failed for {repo}@{revision}/{filename}"))
}

fn split_repo(repo: &str) -> anyhow::Result<(&str, &str)> {
    let (owner, name) = repo
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("repository id must be owner/name, got {repo}"))?;
    anyhow::ensure!(
        !owner.is_empty() && !name.is_empty(),
        "empty repository id part in {repo}"
    );
    Ok((owner, name))
}

/// Hardlink the cached blob into the engine layout, falling back to a symlink
/// (Unix) or a copy when the cache sits on another volume or filesystem.
///
/// The cached path hf-hub returns is a snapshot *symlink*; `link(2)` does not
/// follow symlinks, so hardlinking it verbatim would produce another name for
/// the symlink whose relative target dangles in the engine layout. Resolve to
/// the real blob first.
fn link_into_layout(cached: &Path, dest: &Path) -> anyhow::Result<()> {
    let parent = dest.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let resolved = cached.canonicalize()?;
    // A unique staging directory cleans up on both failure and same-inode
    // rename (a no-op on Unix), without removing the previous destination.
    let staging = tempfile::tempdir_in(parent)?;
    let linking = staging.path().join("resource");
    if std::fs::hard_link(&resolved, &linking).is_err() {
        #[cfg(unix)]
        std::os::unix::fs::symlink(&resolved, &linking)?;
        #[cfg(not(unix))]
        std::fs::copy(&resolved, &linking)?;
    }
    std::fs::rename(&linking, dest)?;
    Ok(())
}

/// Bridge hf-hub per-file delta events to the engine's `Event::ModelProgress`.
///
/// hf-hub progress is a delta stream keyed by filename; this handler tracks the
/// high-water mark for the single file being fetched. Handlers run on the
/// stream-read path and never block: a watch slot retains only the latest value.
pub struct FileProgressBridge {
    tx: mpsc::Sender<Event>,
    resource: String,
    size: u64,
    downloaded: AtomicU64,
    latest: watch::Sender<Option<u64>>,
}

impl FileProgressBridge {
    pub fn new(tx: mpsc::Sender<Event>, resource: String, size: u64) -> Self {
        Self {
            tx,
            resource,
            size,
            downloaded: AtomicU64::new(0),
            latest: watch::channel(None).0,
        }
    }
}

impl ProgressHandler for FileProgressBridge {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(DownloadEvent::Progress { files }) = event else {
            return;
        };
        for file in files {
            if file.status == FileStatus::Complete {
                // Cache-hit completions report zero totals; clamp to the
                // manifest size so consumers see the full file.
                self.downloaded.store(self.size, Ordering::Relaxed);
            } else {
                self.downloaded
                    .fetch_max(file.bytes_completed, Ordering::Relaxed);
            }
        }
        self.latest
            .send_replace(Some(self.downloaded.load(Ordering::Relaxed)));
    }
}

/// Forwarding can wait for capacity without blocking hf-hub callbacks or spawning
/// detached tasks. When the download ends, flush the latest value before returning.
struct ProgressRelay {
    tx: mpsc::Sender<Event>,
    resource: String,
    size: u64,
    latest: watch::Receiver<Option<u64>>,
}
impl ProgressRelay {
    async fn run(
        mut self,
        operation: impl std::future::Future<Output = anyhow::Result<()>>,
    ) -> anyhow::Result<()> {
        let mut latest = self.latest.clone();
        let forward = async {
            while latest.changed().await.is_ok() {
                let value = *latest.borrow_and_update();
                if let Some(downloaded) = value
                    && self
                        .tx
                        .send(Event::ModelProgress {
                            resource: self.resource.clone(),
                            downloaded,
                            total: self.size,
                        })
                        .await
                        .is_err()
                {
                    break;
                }
            }
        };
        tokio::pin!(operation);
        let result = tokio::select! {
            result = &mut operation => result,
            () = forward => operation.await,
        };
        let final_bytes = if result.is_ok() {
            Some(self.size)
        } else {
            *self.latest.borrow_and_update()
        };
        if let Some(downloaded) = final_bytes {
            let _ = self
                .tx
                .send(Event::ModelProgress {
                    resource: self.resource,
                    downloaded,
                    total: self.size,
                })
                .await;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn progress_backpressure_keeps_latest_without_blocking_download_completion() {
        let (tx, mut rx) = mpsc::channel(1);
        tx.send(Event::ModelProgress {
            resource: "queued".into(),
            downloaded: 0,
            total: 1000,
        })
        .await
        .unwrap();
        let (latest, updates) = watch::channel(None);
        let relay = ProgressRelay {
            tx,
            resource: "weights".into(),
            size: 1000,
            latest: updates,
        };
        let (complete, completed) = tokio::sync::oneshot::channel();
        let operation = async move {
            for value in 1..=1000 {
                latest.send_replace(Some(value));
            }
            complete.send(()).unwrap();
            Ok(())
        };
        let consume = async move {
            tokio::time::timeout(std::time::Duration::from_secs(2), completed)
                .await
                .unwrap()
                .unwrap();
            let mut last = 0;
            while let Some(Event::ModelProgress { downloaded, .. }) = rx.recv().await {
                last = downloaded;
                if last == 1000 {
                    break;
                }
            }
            last
        };
        let (result, last) = tokio::join!(relay.run(operation), consume);
        result.unwrap();
        assert_eq!(last, 1000);
    }
    #[tokio::test]
    async fn failed_download_flushes_real_bytes_without_reporting_completion() {
        let (tx, mut rx) = mpsc::channel(1);
        let (latest, updates) = watch::channel(None);
        let relay = ProgressRelay {
            tx,
            resource: "weights".into(),
            size: 100,
            latest: updates,
        };
        let operation = async move {
            latest.send_replace(Some(25));
            Err(anyhow::anyhow!("download failed"))
        };
        let consume = async move {
            let mut values = Vec::new();
            while let Some(Event::ModelProgress { downloaded, .. }) = rx.recv().await {
                values.push(downloaded);
            }
            values
        };
        let (result, values) = tokio::join!(relay.run(operation), consume);
        assert!(result.is_err());
        assert_eq!(values.last(), Some(&25));
        assert!(values.iter().all(|value| *value < 100));
    }
    #[tokio::test]
    async fn closed_progress_receiver_does_not_cancel_download() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let (latest, updates) = watch::channel(None);
        let relay = ProgressRelay {
            tx,
            resource: "weights".into(),
            size: 100,
            latest: updates,
        };
        relay
            .run(async move {
                latest.send_replace(Some(50));
                tokio::task::yield_now().await;
                Ok(())
            })
            .await
            .unwrap();
    }
    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn default_cache_dir_is_absolute_and_shared() {
        let dir = default_cache_dir();
        assert!(dir.is_absolute());
        assert!(dir.ends_with(".cache/huggingface/hub"));
    }

    #[test]
    fn rejects_relative_cache_dir() {
        let error = Downloads::new(&DownloadConfig {
            cache_dir: PathBuf::from("relative/hub"),
            endpoint: None,
            token: None,
        })
        .expect_err("relative cache dir must be rejected");
        assert!(error.to_string().contains("absolute"));
    }

    #[test]
    fn accepts_mirror_endpoint_and_token() {
        let dir = tempdir();
        Downloads::new(&DownloadConfig {
            cache_dir: dir.path().into(),
            endpoint: Some("https://hf-mirror.com".into()),
            token: Some("hf_token".into()),
        })
        .expect("mirror endpoint and token must be accepted");
    }

    #[test]
    fn split_repo_rejects_ids_without_owner() {
        assert!(split_repo("OpenMOSS-Team/MOSS-TTS-Nano-100M").is_ok());
        assert!(split_repo("MOSS-TTS-Nano-100M").is_err());
        assert!(split_repo("OpenMOSS-Team/").is_err());
    }

    #[test]
    fn link_into_layout_hardlinks_same_volume() {
        let dir = tempdir();
        let cached = dir.path().join("blob");
        std::fs::write(&cached, b"payload").unwrap();
        let dest = dir.path().join("engine/model.bin");
        link_into_layout(&cached, &dest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"payload");
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let cached_meta = std::fs::metadata(&cached).unwrap();
            let dest_meta = std::fs::metadata(&dest).unwrap();
            assert_eq!(
                cached_meta.ino(),
                dest_meta.ino(),
                "same-volume link must be a hardlink"
            );
        }
        // Relinking replaces the destination atomically.
        std::fs::write(&cached, b"payload-2").unwrap();
        link_into_layout(&cached, &dest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"payload-2");
        assert!(
            dir.path()
                .join("engine/model.bin.linking")
                .try_exists()
                .is_ok_and(|yes| !yes)
        );
    }

    #[test]
    fn link_into_layout_creates_nested_parents() {
        let dir = tempdir();
        let cached = dir.path().join("blob");
        let mut file = std::fs::File::create(&cached).unwrap();
        file.write_all(b"x").unwrap();
        let dest = dir.path().join("a/b/c/model.bin");
        link_into_layout(&cached, &dest).unwrap();
        assert!(dest.is_file());
    }
}
