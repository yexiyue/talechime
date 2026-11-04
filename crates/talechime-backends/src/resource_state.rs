//! Offline resource readiness computed from the engine layout on disk.
//!
//! The engine layout is the consumption view (see [`crate::downloads`]):
//! a present file of the pinned size is usable — hardlinks keep their inode
//! alive even if the shared cache is cleaned. Inventory therefore never
//! touches the network and never trusts stale task state.
use std::path::Path;

/// Per-model readiness aggregated over its manifest entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceStatus {
    /// No manifest file is present locally; the payload is what preparation
    /// still needs to fetch.
    Missing { missing_bytes: u64 },
    /// Some entries are present; the rest still need preparation.
    Partial {
        ready: usize,
        total: usize,
        /// Sum of the sizes of the entries still missing.
        missing_bytes: u64,
    },
    /// Every entry is present with its pinned size.
    Ready,
}

/// Single source of readiness: an entry is usable iff present with its pinned
/// size. Inventory aggregates and preparation skips MUST agree on this
/// predicate, otherwise a truncated file reports missing but never heals.
pub(crate) fn entry_ready(path: &Path, size: u64) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.len() == size)
}

/// Aggregate `(relative path, pinned size)` entries against one directory.
#[cfg(any(feature = "kokoro", test))]
pub(crate) fn status_of(entries: &[(String, u64)], directory: &Path) -> ResourceStatus {
    let total = entries.len();
    let mut ready = 0usize;
    let mut missing_bytes = 0u64;
    for (path, size) in entries {
        if entry_ready(&directory.join(path), *size) {
            ready += 1;
        } else {
            missing_bytes += size;
        }
    }
    resource_status(ready, total, missing_bytes)
}

#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "cosyvoice3",
    feature = "kokoro",
    feature = "indextts",
    test
))]
fn resource_status(ready: usize, total: usize, missing_bytes: u64) -> ResourceStatus {
    if total == 0 || ready == total {
        ResourceStatus::Ready
    } else if ready == 0 {
        ResourceStatus::Missing { missing_bytes }
    } else {
        ResourceStatus::Partial {
            ready,
            total,
            missing_bytes,
        }
    }
}

/// Aggregate readiness across several `(embedded manifest, directory)` pairs,
/// covering models whose files span multiple source directories.
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "cosyvoice3",
    test
))]
pub(crate) fn status_of_manifests(manifests: &[(&str, &Path)]) -> ResourceStatus {
    let mut ready = 0usize;
    let mut total = 0usize;
    let mut missing_bytes = 0u64;
    for (json, directory) in manifests {
        // Embedded manifests are compile-time inputs; if one fails to parse
        // here, preparation will surface the error. Report missing meanwhile.
        let Ok(entries) = manifest_entries(json) else {
            return ResourceStatus::Missing { missing_bytes: 0 };
        };
        for (path, size) in entries {
            total += 1;
            if entry_ready(&directory.join(path), size) {
                ready += 1;
            } else {
                missing_bytes += size;
            }
        }
    }
    resource_status(ready, total, missing_bytes)
}

/// Readiness for presence-checked conversion layouts without pinned sizes.
/// `missing_bytes` stays zero because entry sizes are unknown.
#[cfg(feature = "indextts")]
pub(crate) fn status_of_names(names: &[&str], directory: &Path) -> ResourceStatus {
    let total = names.len();
    let ready = names
        .iter()
        .filter(|name| {
            directory
                .join(name)
                .metadata()
                .is_ok_and(|metadata| metadata.len() > 0)
        })
        .count();
    resource_status(ready, total, 0)
}

/// Parse `(path, size)` pairs out of an embedded manifest, ignoring other
/// fields. Works for both remote and local-conversion manifest shapes.
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "cosyvoice3",
    test
))]
pub(crate) fn manifest_entries(json: &str) -> anyhow::Result<Vec<(String, u64)>> {
    #[derive(serde::Deserialize)]
    struct Entry {
        path: String,
        size: u64,
    }

    let entries: Vec<Entry> = serde_json::from_str(json)?;
    Ok(entries
        .into_iter()
        .map(|entry| (entry.path, entry.size))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_of_reports_missing_partial_and_ready() {
        let dir = tempfile::tempdir().unwrap();
        let entries = vec![("a.bin".to_owned(), 10u64), ("b.bin".to_owned(), 20u64)];
        assert_eq!(
            status_of(&entries, dir.path()),
            ResourceStatus::Missing { missing_bytes: 30 }
        );
        std::fs::write(dir.path().join("a.bin"), vec![0u8; 10]).unwrap();
        assert_eq!(
            status_of(&entries, dir.path()),
            ResourceStatus::Partial {
                ready: 1,
                total: 2,
                missing_bytes: 20
            }
        );
        // Wrong size counts as missing.
        std::fs::write(dir.path().join("b.bin"), vec![0u8; 5]).unwrap();
        assert_eq!(
            status_of(&entries, dir.path()),
            ResourceStatus::Partial {
                ready: 1,
                total: 2,
                missing_bytes: 20
            }
        );
        std::fs::write(dir.path().join("b.bin"), vec![0u8; 20]).unwrap();
        assert_eq!(status_of(&entries, dir.path()), ResourceStatus::Ready);
    }

    #[test]
    fn empty_manifests_are_ready() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(status_of(&[], dir.path()), ResourceStatus::Ready);
    }

    #[test]
    fn manifest_entries_reads_paths_and_sizes_only() {
        let entries = manifest_entries(
            r#"[{"path":"m.bin","repo":"o/m","revision":"0123456789abcdef0123456789abcdef01234567","filename":"m.bin","size":7},{"path":"n.bin","size":3,"sha256":"x"}]"#,
        )
        .unwrap();
        assert_eq!(
            entries,
            vec![("m.bin".to_owned(), 7u64), ("n.bin".to_owned(), 3u64)]
        );
        assert!(manifest_entries(r#"[{"path":"m.bin"}]"#).is_err());
    }
}
