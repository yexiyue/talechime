//! Shared HF Hub downloads with host-configured cache, endpoint and token.
//!
//! Model weights live in the content-addressed HF cache (blobs named by etag,
//! snapshotted per revision) so other Hub tools on the same machine share them.
//! Fixed model resources are verified and loaded directly from this cache.
//!
//! `DownloadConfig` must carry an absolute cache directory. Cache, endpoint and
//! optional token are passed explicitly instead of inheriting hf-hub's implicit
//! environment or token-file configuration. `default_cache_dir` below is NOT in
//! `talechime-core` paths.rs: the shared cache belongs to the user's Hugging
//! Face ecosystem, not to TaleChime's own `~/.talechime` layout.
use sha2::Digest;
use std::{
    io::Read,
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
    transport: DownloadTransport,
}

/// Immutable identity and integrity information for one Hub resource.
#[derive(Clone, Copy, Debug)]
pub struct HubResource<'a> {
    /// Repository in `owner/name` form.
    pub repo: &'a str,
    /// Immutable repository commit.
    pub revision: &'a str,
    /// Repository-relative file name.
    pub filename: &'a str,
    /// Expected file size in bytes.
    pub size: u64,
    /// Expected SHA-256, encoded as hexadecimal.
    pub sha256: &'a str,
}

fn resource_verified(
    path: &Path,
    resource: HubResource<'_>,
    check: &impl Fn() -> anyhow::Result<()>,
) -> anyhow::Result<bool> {
    check()?;
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() || metadata.len() != resource.size {
        return Ok(false);
    }
    let mut file = std::fs::File::open(path)?;
    let mut hash = sha2::Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        check()?;
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{hash:x}", hash = hash.finalize()) == resource.sha256)
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
            transport: options.transport,
        })
    }

    /// Shared cache used by this explicitly configured client.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    fn checked_cached_path(
        &self,
        repo: &str,
        revision: &str,
        filename: &str,
    ) -> anyhow::Result<Option<PathBuf>> {
        let (owner, name) = split_repo(repo)?;
        let path = self
            .cache_dir
            .join(format!("models--{owner}--{name}"))
            .join("snapshots")
            .join(revision)
            .join(filename);
        let canonical = match path.canonicalize() {
            Ok(path) => path,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        anyhow::ensure!(
            canonical.starts_with(self.cache_dir.canonicalize()?),
            "cached resource escapes Hub cache"
        );
        Ok(Some(canonical))
    }

    /// Resolve a pinned resource offline, rejecting missing or corrupted content.
    #[cfg(any(
        feature = "moss",
        feature = "qwen",
        feature = "voxcpm",
        feature = "omnivoice",
        feature = "asr",
        feature = "kokoro",
        feature = "cosyvoice3",
        feature = "indextts",
        test
    ))]
    pub(crate) fn resolve_resource(
        &self,
        resource: HubResource<'_>,
        check: impl Fn() -> anyhow::Result<()>,
    ) -> anyhow::Result<Option<PathBuf>> {
        let Some(path) =
            self.checked_cached_path(resource.repo, resource.revision, resource.filename)?
        else {
            return Ok(None);
        };
        Ok(resource_verified(&path, resource, &check)?.then_some(path))
    }

    /// Return the verified Hub blob, refreshing a corrupt cached copy if needed.
    pub async fn ensure_resource(
        &self,
        resource: HubResource<'_>,
        progress: FileProgressBridge,
    ) -> anyhow::Result<PathBuf> {
        let (owner, name) = split_repo(resource.repo)?;
        anyhow::ensure!(
            [owner, name].iter().all(|part| {
                *part != "."
                    && *part != ".."
                    && part
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
            }) && !resource.filename.contains('\\')
                && !resource.filename.is_empty()
                && Path::new(resource.filename)
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_)))
                && resource.revision.len() == 40
                && resource.revision.bytes().all(|b| b.is_ascii_hexdigit())
                && resource.sha256.len() == 64
                && resource
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                && resource.size > 0
                && progress.size == resource.size,
            "invalid pinned Hub resource"
        );
        let cached =
            self.checked_cached_path(resource.repo, resource.revision, resource.filename)?;
        if let Some(path) = cached.as_ref()
            && verify_resource_async(path.clone(), resource).await?
        {
            return Ok(path.clone());
        }
        let path = self
            .download_cached(
                resource.repo,
                resource.revision,
                resource.filename,
                progress,
                cached.is_some(),
            )
            .await?;
        anyhow::ensure!(
            verify_resource_async(path.clone(), resource).await?,
            "downloaded resource integrity mismatch: {}@{}/{}",
            resource.repo,
            resource.revision,
            resource.filename
        );
        self.checked_cached_path(resource.repo, resource.revision, resource.filename)?
            .ok_or_else(|| anyhow::anyhow!("downloaded resource is absent from Hub cache"))
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

    async fn download_cached(
        &self,
        repo: &str,
        revision: &str,
        filename: &str,
        progress: FileProgressBridge,
        refresh: bool,
    ) -> anyhow::Result<PathBuf> {
        let expected_size = progress.size;
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
                Ok(path)
            })
            .await
    }
}

async fn verify_resource_async(path: PathBuf, resource: HubResource<'_>) -> anyhow::Result<bool> {
    let size = resource.size;
    let sha256 = resource.sha256.to_owned();
    crate::resource_worker::run(move |control| {
        let resource = HubResource {
            repo: "",
            revision: "",
            filename: "",
            size,
            sha256: &sha256,
        };
        resource_verified(&path, resource, &|| control.check())
    })
    .await
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
        HFError::Http { ref context }
            if context.status.is_server_error() || context.status.as_u16() == 408 =>
        {
            DownloadFailure::Network(error.to_string())
        }
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
    async fn run<T>(
        mut self,
        operation: impl std::future::Future<Output = anyhow::Result<T>>,
    ) -> anyhow::Result<T> {
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
            Err::<(), _>(anyhow::anyhow!("download failed"))
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
    fn pinned_resource_checks_content_and_cancellation() {
        let dir = tempdir();
        let client = Downloads::test(dir.path());
        let resource = HubResource {
            repo: "owner/model",
            revision: "0123456789012345678901234567890123456789",
            filename: "nested/weights",
            size: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        assert!(
            client
                .resolve_resource(resource, || Ok(()))
                .unwrap()
                .is_none()
        );
        let path = client
            .cache_dir()
            .join("models--owner--model/snapshots")
            .join(resource.revision)
            .join(resource.filename);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            client.resolve_resource(resource, || Ok(())).unwrap(),
            Some(path.canonicalize().unwrap())
        );
        assert!(
            client
                .resolve_resource(resource, || anyhow::bail!("cancelled"))
                .is_err()
        );
        std::fs::write(&path, b"abd").unwrap();
        assert!(
            client
                .resolve_resource(resource, || Ok(()))
                .unwrap()
                .is_none()
        );
    }

    #[cfg(unix)]
    #[test]
    fn cached_resource_rejects_snapshot_link_outside_cache() {
        let root = tempdir();
        let client = Downloads::test(root.path());
        let resource = HubResource {
            repo: "owner/model",
            revision: "0123456789012345678901234567890123456789",
            filename: "weights",
            size: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        let outside = root.path().join("outside");
        std::fs::write(&outside, b"abc").unwrap();
        let snapshot = client
            .cache_dir()
            .join("models--owner--model/snapshots")
            .join(resource.revision)
            .join(resource.filename);
        std::fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(outside, snapshot).unwrap();
        assert!(client.resolve_resource(resource, || Ok(())).is_err());
    }

    #[tokio::test]
    async fn direct_resource_repairs_same_size_corruption_and_loads_offline() {
        let server = httpmock::MockServer::start();
        let repo = "owner/model";
        let revision = "0123456789012345678901234567890123456789";
        let filename = "nested/weights";
        let sha256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let mock = server.mock(|when, then| {
            when.path_includes(format!("/{revision}/{filename}"));
            then.status(200)
                .header("ETag", format!("\"{sha256}\""))
                .header("X-Repo-Commit", revision)
                .body("abc");
        });
        let dir = tempdir();
        let client = Downloads::with_options(
            &DownloadConfig {
                cache_dir: dir.path().join("hub"),
                endpoint: Some(server.base_url()),
                token: None,
            },
            DownloadOptions {
                transport: DownloadTransport::Http,
                ..Default::default()
            },
        )
        .unwrap();
        let resource = HubResource {
            repo,
            revision,
            filename,
            size: 3,
            sha256,
        };
        let bridge = || {
            let (tx, rx) = mpsc::channel(1);
            drop(rx);
            FileProgressBridge::new(tx, "weights".into(), 3)
        };
        let path = client.ensure_resource(resource, bridge()).await.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"abc");
        let first_requests = mock.calls();
        assert!(first_requests > 0);
        // A valid cached file must work with an unreachable endpoint.
        let offline = Downloads::new(&DownloadConfig {
            cache_dir: dir.path().join("hub"),
            endpoint: Some("http://127.0.0.1:9".into()),
            token: None,
        })
        .unwrap();
        assert_eq!(
            offline.ensure_resource(resource, bridge()).await.unwrap(),
            path.canonicalize().unwrap()
        );
        assert_eq!(mock.calls(), first_requests);
        std::fs::write(&path, b"bad").unwrap();
        assert!(offline.ensure_resource(resource, bridge()).await.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"bad");
        let repaired = client.ensure_resource(resource, bridge()).await.unwrap();
        assert_eq!(std::fs::read(repaired).unwrap(), b"abc");
        assert!(mock.calls() > first_requests);
        assert!(!dir.path().join("engine").exists());
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
}
