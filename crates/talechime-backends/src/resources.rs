//! Manifest schema and preparation policy for pinned remote resources.
//!
//! Every entry pins an immutable upstream commit. Downloads go through the
//! shared HF cache (see [`crate::downloads`]); integrity for remote entries is
//! the Hub's etag-named blobs plus transport integrity, and local-conversion
//! manifests keep explicit size + SHA-256 verification via [`verify_local`].
use serde::Deserialize;
#[cfg(any(
    feature = "cosyvoice3",
    feature = "kokoro",
    feature = "performance-tools",
    test
))]
use sha2::{Digest, Sha256};
#[cfg(any(
    feature = "cosyvoice3",
    feature = "kokoro",
    feature = "performance-tools",
    test
))]
use std::io::Read;
use std::path::Path;
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "asr"
))]
use talechime_protocol::Event;
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "asr"
))]
use tokio::sync::mpsc;

#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "asr"
))]
#[derive(Deserialize)]
pub(crate) struct Resource {
    /// Destination path inside the engine resource layout.
    pub path: String,
    /// Hub repository id (`owner/name`).
    pub repo: String,
    /// Pinned upstream commit hash.
    pub revision: String,
    /// Path of the file inside the repository.
    pub filename: String,
    pub size: u64,
}

/// A local-conversion file with explicit integrity pins (CosyVoice3-style).
#[cfg(any(
    feature = "cosyvoice3",
    feature = "kokoro",
    feature = "performance-tools",
    test
))]
#[derive(Deserialize)]
pub(crate) struct LocalResource {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// Verify a local converted file against its size and SHA-256 pins.
#[cfg(any(feature = "cosyvoice3", feature = "performance-tools", test))]
pub(crate) fn verify_local(directory: &Path, resource: &LocalResource) -> anyhow::Result<()> {
    verify_local_cancellable(directory, resource, || false)
}

/// Cancellable variant for background verification tasks.
#[cfg(any(
    feature = "cosyvoice3",
    feature = "kokoro",
    feature = "performance-tools",
    test
))]
pub(crate) fn verify_local_cancellable(
    directory: &Path,
    resource: &LocalResource,
    cancelled: impl Fn() -> bool,
) -> anyhow::Result<()> {
    let path = directory.join(&resource.path);
    anyhow::ensure!(!cancelled(), "model verification cancelled");
    let mut file = std::fs::File::open(&path)?;
    anyhow::ensure!(
        file.metadata()?.len() == resource.size,
        "invalid model size: {}",
        path.display()
    );
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        anyhow::ensure!(!cancelled(), "model verification cancelled");
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    anyhow::ensure!(
        format!("{:x}", hash.finalize()) == resource.sha256,
        "model checksum mismatch: {}",
        path.display()
    );
    Ok(())
}

/// Recompute the SHA-256 of a local file; used for evidence receipts where no
/// expected hash is pinned.
#[cfg(any(feature = "performance-tools", test))]
pub(crate) fn sha256_of(path: &Path) -> anyhow::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "asr"
))]
pub(crate) async fn prepare(
    directory: &Path,
    namespace: &str,
    resources: Vec<Resource>,
    downloads: &crate::downloads::Downloads,
    progress: mpsc::Sender<Event>,
) -> anyhow::Result<()> {
    for resource in resources {
        let dest = directory.join(&resource.path);
        // 与盘点同一谓词：大小不符的截断文件要重新下载，否则盘点永远
        // 报缺失而这里永远跳过（见 `resource_state::entry_ready`）。
        if crate::resource_state::entry_ready(&dest, resource.size) {
            continue;
        }
        let name = format!("{namespace}/{}", resource.path);
        let handler =
            crate::downloads::FileProgressBridge::new(progress.clone(), name, resource.size);
        downloads
            .ensure(
                &dest,
                &resource.repo,
                &resource.revision,
                &resource.filename,
                handler,
            )
            .await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_local_accepts_pinned_file_and_rejects_corruption() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model");
        let resource = LocalResource {
            path: "model".into(),
            size: 3,
            sha256: format!("{:x}", Sha256::digest(b"abc")),
        };
        std::fs::write(&path, b"abc").unwrap();
        verify_local(directory.path(), &resource).unwrap();
        for bad in [b"abd".as_slice(), b"a".as_slice()] {
            std::fs::write(&path, bad).unwrap();
            assert!(verify_local(directory.path(), &resource).is_err());
        }
    }

    #[test]
    fn sha256_of_matches_reference_digest() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_of(&path).unwrap(),
            format!("{:x}", Sha256::digest(b"abc"))
        );
    }

    #[cfg(any(
        feature = "moss",
        feature = "qwen",
        feature = "voxcpm",
        feature = "omnivoice",
        feature = "asr"
    ))]
    #[test]
    fn manifest_entries_parse_with_pinned_revisions() {
        // Spot-check one real manifest per backend family compiled in.
        #[cfg(feature = "moss")]
        {
            let entries: Vec<Resource> =
                serde_json::from_str(include_str!("moss/nano/resources.json")).unwrap();
            assert!(!entries.is_empty());
            for entry in &entries {
                assert!(entry.repo.contains('/'), "repo must be owner/name");
                assert_eq!(entry.revision.len(), 40, "revision must be a commit");
                assert!(!entry.filename.is_empty());
                assert!(entry.size > 0);
            }
        }
        #[cfg(feature = "asr")]
        {
            let entries: Vec<Resource> =
                serde_json::from_str(include_str!("asr/qwen06.json")).unwrap();
            assert_eq!(entries.len(), 4);
            assert!(entries.iter().all(|entry| entry.revision.len() == 40));
        }
    }
}
