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

static DESIGNERS: &[fn() -> DesignerHandle] = &[
    #[cfg(feature = "voxcpm")]
    || Rc::new(voxcpm::Engine),
    #[cfg(feature = "omnivoice")]
    || Rc::new(omnivoice::Engine),
    #[cfg(feature = "qwen")]
    || Rc::new(qwen::Engine),
    #[cfg(feature = "moss")]
    || Rc::new(moss::Engine),
];

/// Engine info for every design engine compiled into this build.
pub fn designers(root: &Path) -> Vec<EngineInfo> {
    DESIGNERS.iter().map(|new| new().info(root)).collect()
}

/// Exact engine lookup by id.
pub fn designer(id: &str) -> Option<DesignerHandle> {
    DESIGNERS
        .iter()
        .map(|new| new())
        .find(|engine| engine.id() == id)
}

/// Cheap readiness probe: every manifest file exists with its declared size.
/// Size-only so probing never hashes gigabytes; full hashes still guard prepare.
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    test
))]
pub(crate) fn manifest_state(manifests: &[(&'static str, &Path)]) -> (bool, u64) {
    let mut ready = true;
    let mut missing = 0u64;
    for (manifest, directory) in manifests {
        let parsed: serde_json::Value =
            serde_json::from_str(manifest).expect("embedded manifests parse");
        let resources = parsed.as_array().cloned().unwrap_or_else(|| {
            parsed["resources"]
                .as_array()
                .cloned()
                .expect("embedded manifests are arrays or {resources}")
        });
        for resource in resources {
            let path = resource["path"].as_str().expect("manifest path");
            let size = resource["size"].as_u64().expect("manifest size");
            match std::fs::metadata(directory.join(path)) {
                Ok(metadata) if metadata.len() == size => {}
                _ => {
                    ready = false;
                    missing += size;
                }
            }
        }
    }
    (ready, missing)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn engine_ids_are_unique_and_reported_for_every_feature() {
        let root = tempfile::tempdir().unwrap();
        let infos = designers(root.path());
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
        let manifest = r#"[{"path":"model.bin","sha256":"x","size":10,"url":""}]"#;
        std::fs::write(root.path().join("model.bin"), vec![0u8; 4]).unwrap();
        let (ready, missing) = manifest_state(&[(manifest, root.path())]);
        assert!(!ready && missing == 10);
        std::fs::write(root.path().join("model.bin"), vec![0u8; 10]).unwrap();
        let (ready, missing) = manifest_state(&[(manifest, root.path())]);
        assert!(ready && missing == 0);
    }
}
