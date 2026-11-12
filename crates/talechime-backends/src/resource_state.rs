//! Offline resource readiness shared by cached and transitional native models.
//!
//! Direct Hub resources use pinned size and digest checks in `resources`;
//! the legacy native recipes use their published layout until migrated.
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
    /// Every required entry passed its applicable readiness check.
    Ready,
}

impl ResourceStatus {
    /// Bytes still missing; zero when ready or when resource sizes are unknown.
    pub fn missing_bytes(self) -> u64 {
        match self {
            Self::Missing { missing_bytes } | Self::Partial { missing_bytes, .. } => missing_bytes,
            Self::Ready => 0,
        }
    }
}

/// Single source of readiness: an entry is usable iff present with its pinned
/// size. Inventory aggregates and preparation skips MUST agree on this
/// predicate, otherwise a truncated file reports missing but never heals.
pub(crate) fn entry_ready(path: &Path, size: u64) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && metadata.len() == size)
}

/// Aggregate `(relative path, pinned size)` entries against one directory.
#[cfg(test)]
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

#[cfg(test)]
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
}
