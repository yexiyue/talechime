//! Voice design engines: `VoiceDesigner` implementations over the backend
//! design kernels, plus the static registry the facade enumerates.
//! Order matters: engines reusing prepared narration models come first so the
//! automatic policy prefers zero-download design.

use std::{path::Path, rc::Rc};
use talechime_core::design::{EngineInfo, VoiceDesigner};

#[cfg(feature = "moss")]
mod moss;
#[cfg(feature = "omnivoice")]
mod omnivoice;
#[cfg(feature = "qwen")]
mod qwen;
#[cfg(feature = "voxcpm")]
mod voxcpm;

#[cfg(any(feature = "qwen", feature = "omnivoice", test))]
pub(crate) mod thread;

pub type DesignerHandle = Rc<dyn VoiceDesigner>;

static DESIGNERS: &[fn(crate::downloads::Downloads) -> DesignerHandle] = &[
    #[cfg(feature = "voxcpm")]
    |downloads| Rc::new(voxcpm::Engine(downloads)),
    #[cfg(feature = "omnivoice")]
    |downloads| Rc::new(omnivoice::Engine(downloads)),
    #[cfg(feature = "qwen")]
    |downloads| Rc::new(qwen::Engine(downloads)),
    #[cfg(feature = "moss")]
    |downloads| Rc::new(moss::Engine(downloads)),
];

/// Engine info for every design engine compiled into this build.
pub fn designers(root: &Path, downloads: &crate::downloads::Downloads) -> Vec<EngineInfo> {
    DESIGNERS
        .iter()
        .map(|new| new(downloads.clone()).info(root))
        .collect()
}

/// Exact engine lookup by id.
pub fn designer(id: &str, downloads: &crate::downloads::Downloads) -> Option<DesignerHandle> {
    DESIGNERS
        .iter()
        .map(|new| new(downloads.clone()))
        .find(|engine| engine.id() == id)
}

/// Cheap readiness probe: every manifest file exists with its declared size.
/// Size-only so probing never hashes gigabytes; shares the inventory predicate.
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    test
))]
pub(crate) fn manifest_state(manifests: &[(&'static str, &Path)]) -> (bool, u64) {
    let status = crate::resource_state::status_of_manifests(manifests);
    (
        status == crate::ResourceStatus::Ready,
        status.missing_bytes(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn engine_ids_are_unique_and_reported_for_every_feature() {
        let root = tempfile::tempdir().unwrap();
        let infos = designers(root.path(), &crate::downloads::Downloads::test(root.path()));
        let mut ids: Vec<_> = infos.iter().map(|info| info.id).collect();
        let total = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "engine ids must be unique");
        assert!(!ids.is_empty() || DESIGNERS.is_empty());
    }
    #[test]
    fn manifest_state_detects_missing_files_and_sizes() {
        let root = tempfile::tempdir().unwrap();
        let manifest = r#"[{"path":"model.bin","repo":"o/m","revision":"1234567890123456789012345678901234567890","filename":"model.bin","size":10}]"#;
        std::fs::write(root.path().join("model.bin"), vec![0u8; 4]).unwrap();
        let (ready, missing) = manifest_state(&[(manifest, root.path())]);
        assert!(!ready && missing == 10);
        std::fs::write(root.path().join("model.bin"), vec![0u8; 10]).unwrap();
        let (ready, missing) = manifest_state(&[(manifest, root.path())]);
        assert!(ready && missing == 0);
    }
}
