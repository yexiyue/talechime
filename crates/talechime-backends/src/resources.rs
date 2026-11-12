//! Manifest schema and preparation policy for pinned remote resources.
//!
//! Every entry pins an immutable upstream commit. Downloads go through the
//! shared HF cache (see [`crate::downloads`]) and verify the pinned size and
//! SHA-256. Local-conversion inputs also use explicit integrity checks.
use serde::Deserialize;
#[cfg(any(feature = "performance-tools", test))]
use sha2::{Digest, Sha256};
#[cfg(any(feature = "performance-tools", test))]
use std::io::Read;
#[cfg(any(feature = "performance-tools", test))]
use std::path::Path;
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "asr",
    feature = "kokoro",
    feature = "cosyvoice3",
    feature = "indextts"
))]
use talechime_protocol::Event;
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "asr",
    feature = "kokoro",
    feature = "cosyvoice3",
    feature = "indextts"
))]
use tokio::sync::mpsc;

#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "asr",
    feature = "kokoro",
    feature = "cosyvoice3",
    feature = "indextts"
))]
#[derive(Deserialize)]
pub(crate) struct Resource {
    /// Logical name used by the model loader.
    pub path: String,
    /// Hub repository id (`owner/name`).
    pub repo: String,
    /// Pinned upstream commit hash.
    pub revision: String,
    /// Path of the file inside the repository.
    pub filename: String,
    pub size: u64,
    pub sha256: String,
}

#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "asr",
    feature = "kokoro",
    feature = "cosyvoice3",
    feature = "indextts"
))]
impl Resource {
    fn validate(&self) -> anyhow::Result<()> {
        fn relative(value: &str) -> bool {
            !value.is_empty()
                && !value.contains('\\')
                && std::path::Path::new(value)
                    .components()
                    .all(|part| matches!(part, std::path::Component::Normal(_)))
        }
        let repo = self.repo.split_once('/');
        anyhow::ensure!(
            repo.is_some_and(|(owner, name)| {
                [owner, name].iter().all(|part| {
                    !part.is_empty()
                        && *part != "."
                        && *part != ".."
                        && part
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
                })
            }) && relative(&self.filename)
                && relative(&self.path)
                && self.revision.len() == 40
                && self.revision.bytes().all(|b| b.is_ascii_hexdigit())
                && self.sha256.len() == 64
                && self
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                && self.size > 0,
            "invalid pinned model resource identity"
        );
        Ok(())
    }
}

/// Resolve the logical names in a fixed manifest to verified cached blobs.
#[cfg(any(
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "qwen",
    feature = "moss",
    feature = "asr",
    feature = "kokoro",
    feature = "cosyvoice3",
    feature = "indextts"
))]
pub(crate) async fn resolve_manifest(
    manifest: &str,
    downloads: crate::downloads::Downloads,
) -> anyhow::Result<std::collections::BTreeMap<String, std::path::PathBuf>> {
    let inventory = scan_manifest(manifest, downloads).await?;
    debug_assert!(
        matches!(inventory.status, crate::ResourceStatus::Ready)
            == inventory.first_missing.is_none()
    );
    if let Some(missing) = inventory.first_missing {
        anyhow::bail!("model resource missing or corrupt: {missing}");
    }
    Ok(inventory.paths)
}

/// Inventory fixed Hub entries without a network request or a model directory.
#[cfg(any(
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "qwen",
    feature = "moss",
    feature = "kokoro",
    feature = "cosyvoice3",
    feature = "indextts"
))]
pub(crate) async fn cached_manifest_status(
    manifest: &str,
    downloads: crate::downloads::Downloads,
) -> anyhow::Result<crate::ResourceStatus> {
    Ok(scan_manifest(manifest, downloads).await?.status)
}

/// Combine separate manifests without merging logical names from different repositories.
#[cfg(feature = "moss")]
pub(crate) async fn cached_manifests_status(
    manifests: &[&str],
    downloads: crate::downloads::Downloads,
) -> anyhow::Result<crate::ResourceStatus> {
    anyhow::ensure!(!manifests.is_empty(), "no model resource manifests");
    let mut total = 0usize;
    let mut ready = 0usize;
    let mut missing_bytes = 0u64;
    for manifest in manifests {
        let count = serde_json::from_str::<Vec<Resource>>(manifest)?.len();
        anyhow::ensure!(count > 0, "empty model resource manifest");
        total = total
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("resource count overflow"))?;
        match cached_manifest_status(manifest, downloads.clone()).await? {
            crate::ResourceStatus::Ready => ready += count,
            crate::ResourceStatus::Partial {
                ready: count,
                missing_bytes: bytes,
                ..
            } => {
                ready += count;
                missing_bytes = missing_bytes.saturating_add(bytes);
            }
            crate::ResourceStatus::Missing {
                missing_bytes: bytes,
            } => {
                missing_bytes = missing_bytes.saturating_add(bytes);
            }
        }
    }
    Ok(if ready == total {
        crate::ResourceStatus::Ready
    } else if ready == 0 {
        crate::ResourceStatus::Missing { missing_bytes }
    } else {
        crate::ResourceStatus::Partial {
            ready,
            total,
            missing_bytes,
        }
    })
}

#[cfg(any(
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "qwen",
    feature = "moss",
    feature = "asr",
    feature = "kokoro",
    feature = "cosyvoice3",
    feature = "indextts"
))]
struct CacheInventory {
    status: crate::ResourceStatus,
    paths: std::collections::BTreeMap<String, std::path::PathBuf>,
    first_missing: Option<String>,
    #[cfg_attr(
        not(any(
            feature = "voxcpm",
            feature = "omnivoice",
            feature = "qwen",
            feature = "moss",
            feature = "kokoro",
            feature = "cosyvoice3",
            feature = "indextts"
        )),
        expect(
            dead_code,
            reason = "recognizer-only builds do not request confirmation identities"
        )
    )]
    fingerprint: String,
}

#[cfg(any(
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "qwen",
    feature = "moss",
    feature = "asr",
    feature = "kokoro",
    feature = "cosyvoice3",
    feature = "indextts"
))]
async fn scan_manifest(
    manifest: &str,
    downloads: crate::downloads::Downloads,
) -> anyhow::Result<CacheInventory> {
    let manifest = manifest.to_owned();
    crate::resource_worker::run(move |control| {
        use sha2::{Digest, Sha256};
        let entries: Vec<Resource> = serde_json::from_str(&manifest)?;
        anyhow::ensure!(!entries.is_empty(), "empty model resource manifest");
        let total = entries.len();
        let mut names = std::collections::BTreeSet::new();
        let mut ready = 0usize;
        let mut missing_bytes = 0u64;
        let mut paths = std::collections::BTreeMap::new();
        let mut first_missing = None;
        let mut fingerprint = Sha256::new();
        for entry in entries {
            entry.validate()?;
            anyhow::ensure!(
                names.insert(entry.path.clone()),
                "duplicate model resource path"
            );
            let resource = crate::downloads::HubResource {
                repo: &entry.repo,
                revision: &entry.revision,
                filename: &entry.filename,
                size: entry.size,
                sha256: &entry.sha256,
            };
            let found = downloads.resolve_resource(resource, || control.check())?;
            for value in [
                &entry.path,
                &entry.repo,
                &entry.revision,
                &entry.filename,
                &entry.sha256,
            ] {
                fingerprint.update((value.len() as u64).to_le_bytes());
                fingerprint.update(value.as_bytes());
            }
            {
                fingerprint.update(entry.size.to_le_bytes());
                fingerprint.update([u8::from(found.is_some())]);
            }
            if let Some(path) = found {
                ready += 1;
                paths.insert(entry.path, path);
            } else {
                missing_bytes = missing_bytes.saturating_add(entry.size);
                first_missing.get_or_insert_with(|| {
                    format!("{}@{}/{}", entry.repo, entry.revision, entry.filename)
                });
            }
        }
        let status = if ready == total {
            crate::ResourceStatus::Ready
        } else if ready == 0 {
            crate::ResourceStatus::Missing { missing_bytes }
        } else {
            crate::ResourceStatus::Partial {
                ready,
                total,
                missing_bytes,
            }
        };
        Ok(CacheInventory {
            status,
            paths,
            first_missing,
            fingerprint: format!("{:x}", fingerprint.finalize()),
        })
    })
    .await
}

/// A stable snapshot binds both resource identities and each file's readiness.
#[cfg(any(
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "qwen",
    feature = "moss",
    feature = "kokoro",
    feature = "cosyvoice3",
    feature = "indextts"
))]
pub(crate) async fn inventory_identity(
    manifests: &[&str],
    downloads: crate::downloads::Downloads,
) -> anyhow::Result<(u64, String)> {
    use sha2::{Digest, Sha256};
    anyhow::ensure!(!manifests.is_empty(), "no model resource manifests");
    let mut hash = Sha256::new();
    let mut missing = 0u64;
    for manifest in manifests {
        let inventory = scan_manifest(manifest, downloads.clone()).await?;
        missing = missing.saturating_add(inventory.status.missing_bytes());
        hash.update(inventory.fingerprint.as_bytes());
    }
    Ok((missing, format!("{:x}", hash.finalize())))
}

/// Recompute the SHA-256 of a local file; used for evidence receipts where no
/// expected hash is pinned.
#[cfg(any(feature = "performance-tools", test))]
pub(crate) fn sha256_of(path: &Path) -> anyhow::Result<String> {
    let mut file = std::fs::File::open(path)?;
    hash_file(&mut file, || false)
}

#[cfg(any(feature = "performance-tools", test))]
fn hash_file(file: &mut std::fs::File, cancelled: impl Fn() -> bool) -> anyhow::Result<String> {
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
    Ok(format!("{:x}", hash.finalize()))
}

/// Fetch pinned resources and return their logical names and verified Hub paths.
/// This operation does not create a model directory or write application data.
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "asr",
    feature = "kokoro",
    feature = "cosyvoice3",
    feature = "indextts"
))]
pub(crate) async fn prepare_cached(
    namespace: &str,
    resources: Vec<Resource>,
    downloads: &crate::downloads::Downloads,
    progress: mpsc::Sender<Event>,
) -> anyhow::Result<Vec<(String, std::path::PathBuf)>> {
    use futures::{TryStreamExt, stream};
    let mut names = std::collections::BTreeSet::new();
    for entry in &resources {
        entry.validate()?;
        anyhow::ensure!(names.insert(&entry.path), "duplicate model resource path");
    }
    stream::iter(resources.into_iter().map(Ok::<_, anyhow::Error>))
        .map_ok(|resource| {
            let progress = progress.clone();
            async move {
                let name = format!("{namespace}/{}", resource.path);
                let identity = crate::downloads::HubResource {
                    repo: &resource.repo,
                    revision: &resource.revision,
                    filename: &resource.filename,
                    size: resource.size,
                    sha256: &resource.sha256,
                };
                let cached = downloads
                    .ensure_resource(
                        identity,
                        crate::downloads::FileProgressBridge::new(progress, name, resource.size),
                    )
                    .await?;
                Ok::<_, anyhow::Error>((resource.path, cached))
            }
        })
        .try_buffer_unordered(2)
        .try_collect()
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "voxcpm")]
    #[tokio::test]
    async fn inventory_identity_changes_when_equal_size_missing_files_are_swapped() {
        let root = tempfile::tempdir().unwrap();
        let downloads = crate::downloads::Downloads::test(root.path());
        let revision = "0123456789012345678901234567890123456789";
        let dir = downloads
            .cache_dir()
            .join("models--owner--model/snapshots")
            .join(revision);
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = serde_json::json!([
            {"repo":"owner/model", "revision":revision, "filename":"a", "path":"a", "size":3, "sha256":format!("{:x}", Sha256::digest(b"abc"))},
            {"repo":"owner/model", "revision":revision, "filename":"b", "path":"b", "size":3, "sha256":format!("{:x}", Sha256::digest(b"xyz"))}
        ]).to_string();
        std::fs::write(dir.join("a"), b"abc").unwrap();
        let first = inventory_identity(&[&manifest], downloads.clone())
            .await
            .unwrap();
        std::fs::remove_file(dir.join("a")).unwrap();
        std::fs::write(dir.join("b"), b"xyz").unwrap();
        let second = inventory_identity(&[&manifest], downloads).await.unwrap();
        assert_eq!(first.0, 3);
        assert_eq!(second.0, first.0);
        assert_ne!(first.1, second.1);
    }

    #[cfg(all(unix, feature = "voxcpm"))]
    #[tokio::test]
    async fn ready_resource_is_reused_from_read_only_shared_cache() {
        use std::os::unix::fs::PermissionsExt;
        struct Restore(std::path::PathBuf, std::path::PathBuf);
        impl Drop for Restore {
            fn drop(&mut self) {
                let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
                let _ = std::fs::set_permissions(&self.1, std::fs::Permissions::from_mode(0o644));
            }
        }
        let root = tempfile::tempdir().unwrap();
        let downloads = crate::downloads::Downloads::test(root.path());
        let revision = "0123456789012345678901234567890123456789";
        let snapshot = downloads
            .cache_dir()
            .join("models--owner--shared/snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        let path = snapshot.join("weights.bin");
        std::fs::write(&path, b"abc").unwrap();
        let manifest = serde_json::json!([{"repo":"owner/shared", "revision":revision,
            "filename":"weights.bin", "path":"model.bin", "size":3,
            "sha256":format!("{:x}", Sha256::digest(b"abc"))}])
        .to_string();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).unwrap();
        std::fs::set_permissions(&snapshot, std::fs::Permissions::from_mode(0o555)).unwrap();
        let _restore = Restore(snapshot.clone(), path.clone());
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        let files = prepare_cached(
            "fixture",
            serde_json::from_str(&manifest).unwrap(),
            &downloads,
            tx,
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read(&files[0].1).unwrap(), b"abc");
        assert_eq!(
            cached_manifest_status(&manifest, downloads).await.unwrap(),
            crate::ResourceStatus::Ready
        );
        assert_eq!(std::fs::read_dir(&snapshot).unwrap().count(), 1);
    }

    #[cfg(feature = "moss")]
    #[tokio::test]
    async fn combined_manifests_track_independent_same_named_resources() {
        let root = tempfile::tempdir().unwrap();
        let downloads = crate::downloads::Downloads::test(root.path());
        let revision = "0123456789012345678901234567890123456789";
        let mut manifests = Vec::new();
        let mut paths = Vec::new();
        for (repo, content) in [
            ("owner/model", b"abc".as_slice()),
            ("owner/codec", b"xyz".as_slice()),
        ] {
            let path = downloads
                .cache_dir()
                .join(format!("models--{}", repo.replace('/', "--")))
                .join("snapshots")
                .join(revision)
                .join("config.json");
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, content).unwrap();
            manifests.push(
                serde_json::json!([{"repo": repo, "revision": revision,
                "filename": "config.json", "path": "config.json", "size": content.len(),
                "sha256": format!("{:x}", Sha256::digest(content))}])
                .to_string(),
            );
            paths.push(path);
        }
        let refs: Vec<&str> = manifests.iter().map(String::as_str).collect();
        assert_eq!(
            cached_manifests_status(&refs, downloads.clone())
                .await
                .unwrap(),
            crate::ResourceStatus::Ready
        );
        std::fs::write(&paths[1], b"bad").unwrap();
        assert_eq!(
            cached_manifests_status(&refs, downloads.clone())
                .await
                .unwrap(),
            crate::ResourceStatus::Partial {
                ready: 1,
                total: 2,
                missing_bytes: 3
            }
        );
        std::fs::remove_file(&paths[0]).unwrap();
        assert_eq!(
            cached_manifests_status(&refs, downloads).await.unwrap(),
            crate::ResourceStatus::Missing { missing_bytes: 6 }
        );
    }

    #[cfg(feature = "voxcpm")]
    #[test]
    fn remote_resource_rejects_path_escape_and_bad_digest() {
        let mut resource = Resource {
            path: "model.bin".into(),
            repo: "owner/model".into(),
            revision: "0".repeat(40),
            filename: "weights.bin".into(),
            size: 1,
            sha256: "0".repeat(64),
        };
        assert!(resource.validate().is_ok());
        resource.path = "../model.bin".into();
        assert!(resource.validate().is_err());
        resource.path = "model.bin".into();
        resource.filename = "nested\\weights.bin".into();
        assert!(resource.validate().is_err());
        resource.filename = "weights.bin".into();
        resource.sha256 = "z".repeat(64);
        assert!(resource.validate().is_err());
    }

    #[cfg(feature = "voxcpm")]
    #[tokio::test]
    async fn resolved_manifest_maps_renamed_files_from_independent_repositories() {
        let root = tempfile::tempdir().unwrap();
        let downloads = crate::downloads::Downloads::test(root.path());
        let revision = "0123456789012345678901234567890123456789";
        let mut entries = Vec::new();
        for (repo, filename, logical, content) in [
            ("owner/one", "a.bin", "logical/model.bin", b"abc".as_slice()),
            ("owner/two", "nested/b.bin", "codec.bin", b"xyz".as_slice()),
        ] {
            let path = downloads
                .cache_dir()
                .join(format!("models--{}", repo.replace('/', "--")))
                .join("snapshots")
                .join(revision)
                .join(filename);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, content).unwrap();
            entries.push(serde_json::json!({"repo": repo, "revision": revision, "filename": filename,
                "path": logical, "size": content.len(), "sha256": format!("{:x}", Sha256::digest(content))}));
        }
        let manifest = serde_json::to_string(&entries).unwrap();
        let (progress, receiver) = mpsc::channel(1);
        drop(receiver);
        let fetched = prepare_cached(
            "fixture",
            serde_json::from_str(&manifest).unwrap(),
            &downloads,
            progress,
        )
        .await
        .unwrap();
        assert_eq!(fetched.len(), 2);
        assert!(!root.path().join("engine").exists());
        assert_eq!(
            cached_manifest_status(&manifest, downloads.clone())
                .await
                .unwrap(),
            crate::ResourceStatus::Ready
        );
        let resolved = resolve_manifest(&manifest, downloads.clone())
            .await
            .unwrap();
        assert_eq!(
            std::fs::read(&resolved["logical/model.bin"]).unwrap(),
            b"abc"
        );
        assert_eq!(std::fs::read(&resolved["codec.bin"]).unwrap(), b"xyz");
        std::fs::write(&resolved["codec.bin"], b"bad").unwrap();
        assert_eq!(
            cached_manifest_status(&manifest, downloads.clone())
                .await
                .unwrap(),
            crate::ResourceStatus::Partial {
                ready: 1,
                total: 2,
                missing_bytes: 3
            }
        );
        assert!(
            resolve_manifest(&manifest, downloads.clone())
                .await
                .is_err()
        );
        for path in resolved.values() {
            std::fs::remove_file(path).unwrap();
        }
        assert_eq!(
            cached_manifest_status(&manifest, downloads).await.unwrap(),
            crate::ResourceStatus::Missing { missing_bytes: 6 }
        );
    }

    #[test]
    fn remote_manifests_pin_file_integrity() {
        let manifests = [
            include_str!("moss/nano/resources.json"),
            include_str!("omnivoice/resources.json"),
            include_str!("qwen/resources.json"),
            include_str!("qwen/models/0.6b-base.json"),
            include_str!("qwen/models/1.7b-base.json"),
            include_str!("qwen/models/1.7b-customvoice.json"),
            include_str!("qwen/models/1.7b-voicedesign.json"),
            include_str!("asr/qwen06.json"),
            include_str!("asr/sensevoice.json"),
            include_str!("voxcpm/resources.json"),
            include_str!("voxcpm/original-resources.json"),
            include_str!("kokoro/distribution-context-inputs.json"),
            include_str!("cosyvoice3/resources.json"),
            include_str!("indextts/resources.json"),
        ];
        #[cfg(feature = "moss")]
        let manifests = manifests.into_iter().chain([
            talechime_moss_tts::manifests::CODEC,
            talechime_moss_tts::manifests::REALTIME,
            talechime_moss_tts::manifests::VOICE_DESIGN,
        ]);
        for manifest in manifests {
            let entries: Vec<serde_json::Value> = serde_json::from_str(manifest).unwrap();
            assert!(!entries.is_empty());
            for entry in entries {
                let sha256 = entry["sha256"].as_str().unwrap();
                assert!(sha256.len() == 64 && sha256.bytes().all(|b| b.is_ascii_hexdigit()));
                let revision = entry["revision"].as_str().unwrap();
                assert!(revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()));
            }
            #[cfg(any(
                feature = "moss",
                feature = "qwen",
                feature = "voxcpm",
                feature = "omnivoice",
                feature = "asr",
                feature = "kokoro",
                feature = "cosyvoice3",
                feature = "indextts"
            ))]
            for resource in serde_json::from_str::<Vec<Resource>>(manifest).unwrap() {
                resource.validate().unwrap();
            }
        }
    }

    #[test]
    fn sha256_of_matches_reference_digest() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model");
        let contents = vec![42u8; 65537];
        std::fs::write(&path, &contents).unwrap();
        assert_eq!(
            sha256_of(&path).unwrap(),
            format!("{:x}", Sha256::digest(&contents))
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
