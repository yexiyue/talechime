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
pub async fn designers(
    root: &Path,
    downloads: &crate::downloads::Downloads,
) -> anyhow::Result<Vec<EngineInfo>> {
    let mut infos = Vec::with_capacity(DESIGNERS.len());
    for new in DESIGNERS {
        infos.push(new(downloads.clone()).info(root).await?);
    }
    Ok(infos)
}

/// Exact engine lookup by id.
pub fn designer(id: &str, downloads: &crate::downloads::Downloads) -> Option<DesignerHandle> {
    DESIGNERS
        .iter()
        .map(|new| new(downloads.clone()))
        .find(|engine| engine.id() == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn engine_ids_are_unique_and_reported_for_every_feature() {
        let root = tempfile::tempdir().unwrap();
        let infos = designers(root.path(), &crate::downloads::Downloads::test(root.path()))
            .await
            .unwrap();
        let mut ids: Vec<_> = infos.iter().map(|info| info.id).collect();
        let total = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "engine ids must be unique");
        assert!(!ids.is_empty() || DESIGNERS.is_empty());
    }
}
