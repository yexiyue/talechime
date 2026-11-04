//! Shared HF Hub downloads with host-configured cache, endpoint and token.
//!
//! Model weights live in the content-addressed HF cache (blobs named by etag,
//! snapshotted per revision) so other Hub tools on the same machine share them.
//! Engine-layout entries are hardlinks into that cache: no second copy, and the
//! link keeps the inode alive even if the cache is later cleaned. The engine
//! layout stays the consumption view adapters load from.
//!
//! `DownloadConfig` must carry an absolute cache directory: hf-hub's builder
//! default is `.cache/huggingface/hub` relative to the current working
//! directory and reads no environment variables, which is never right for a
//! desktop host. `default_cache_dir` below is deliberately NOT in
//! `talechime-core` paths.rs: the shared cache belongs to the user's Hugging
//! Face ecosystem, not to TaleChime's own `~/.talechime` layout.
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use hf_hub::progress::{DownloadEvent, FileStatus, ProgressEvent, ProgressHandler};
use talechime_protocol::Event;
use tokio::sync::mpsc;

/// Host-provided download configuration.
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Absolute HF cache directory (e.g. `~/.cache/huggingface/hub`).
    pub cache_dir: PathBuf,
    /// Hub endpoint override such as a mirror; `None` uses the official host.
    pub endpoint: Option<String>,
    /// Optional Hub access token.
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

/// Shared-cache default matching the Python `huggingface_hub` ecosystem layout.
pub fn default_cache_dir() -> PathBuf {
    std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/huggingface/hub")
}

/// Cheap-to-clone download client bound to one configuration.
#[derive(Clone)]
pub struct Downloads {
    client: hf_hub::HFClient,
}

impl std::fmt::Debug for Downloads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Downloads").finish_non_exhaustive()
    }
}

impl Downloads {
    pub fn new(config: &DownloadConfig) -> anyhow::Result<Self> {
        anyhow::ensure!(
            config.cache_dir.is_absolute(),
            "download cache directory must be absolute, got {}",
            config.cache_dir.display()
        );
        let mut builder = hf_hub::HFClient::builder().cache_dir(config.cache_dir.clone());
        if let Some(endpoint) = &config.endpoint {
            builder = builder.endpoint(endpoint);
        }
        if let Some(token) = &config.token {
            builder = builder.token(token);
        }
        Ok(Self {
            client: builder.build()?,
        })
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
        let expected_size = progress.size;
        if crate::resource_state::entry_ready(dest, expected_size) {
            return Ok(());
        }
        let (owner, name) = split_repo(repo)?;
        let model = self.client.model(owner, name);
        let progress = hf_hub::progress::Progress::new(progress);
        let download = model
            .download_file()
            .filename(filename)
            .revision(revision)
            .progress(progress);
        let mut path = download
            .clone()
            .send()
            .await
            .map_err(|error| download_error(repo, revision, filename, classify(error)))?;
        // A cache hit may itself be truncated (the layout can share its inode).
        // Let hf-hub replace the blob using its own locks and atomic writes.
        if !crate::resource_state::entry_ready(&path, expected_size) {
            path = download
                .force_download(true)
                .send()
                .await
                .map_err(|error| download_error(repo, revision, filename, classify(error)))?;
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
/// stream-read path and must never block, hence `try_send`.
pub struct FileProgressBridge {
    tx: mpsc::Sender<Event>,
    resource: String,
    size: u64,
    downloaded: AtomicU64,
}

impl FileProgressBridge {
    pub fn new(tx: mpsc::Sender<Event>, resource: String, size: u64) -> Self {
        Self {
            tx,
            resource,
            size,
            downloaded: AtomicU64::new(0),
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
        let _ = self.tx.try_send(Event::ModelProgress {
            resource: self.resource.clone(),
            downloaded: self.downloaded.load(Ordering::Relaxed),
            total: self.size,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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
