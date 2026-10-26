//! Opt-in developer-only offline checks. Never downloads or renames files.
pub mod recorder;

/// Request trace context is data-only and can cross an owner-thread boundary.
#[derive(Clone)]
pub struct OwnerContext {
    span: tracing::Span,
}
impl OwnerContext {
    pub fn capture() -> Self {
        Self {
            span: tracing::info_span!(target:"talechime_perf", parent:&tracing::Span::current(), "owner_request"),
        }
    }
    pub fn enter(&self) -> tracing::span::EnteredSpan {
        self.span.clone().entered()
    }
    pub fn span(&self) -> tracing::Span {
        self.span.clone()
    }
}
use std::path::{Component, Path, PathBuf};

#[derive(serde::Serialize)]
pub struct VerifiedFile {
    pub path: PathBuf,
    pub size: u64,
    pub sha256: String,
}

#[derive(serde::Serialize)]
pub struct Receipt {
    pub schema: &'static str,
    pub backend: String,
    pub model: String,
    pub resources_root: PathBuf,
    pub verification: &'static str,
    pub files: Vec<VerifiedFile>,
}

/// Check a pinned manifest without downloading, renaming or publishing resources.
/// Callers must supply the production manifest, not a locally invented oracle.
pub fn manifest(root: &Path, directory: &Path, json: &str) -> anyhow::Result<Vec<VerifiedFile>> {
    let value: serde_json::Value = serde_json::from_str(json)?;
    let resources: Vec<crate::resources::Resource> =
        serde_json::from_value(value.get("resources").unwrap_or(&value).clone())?;
    let canonical_root = root.canonicalize()?;
    let mut files = Vec::new();
    for resource in resources {
        anyhow::ensure!(
            Path::new(&resource.path)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
            "invalid production resource path"
        );
        let path = directory.join(&resource.path);
        anyhow::ensure!(
            path.canonicalize()?.starts_with(&canonical_root),
            "model file escapes resources root"
        );
        crate::resources::verify(&path, &resource)?;
        files.push(VerifiedFile {
            path: path.strip_prefix(root)?.to_owned(),
            size: resource.size,
            sha256: resource.sha256,
        });
    }
    Ok(files)
}

/// Verify the immutable production manifests without invoking preparation.
pub fn verify(registry: &crate::Registry, backend: &str, model: &str) -> anyhow::Result<Receipt> {
    registry.capabilities_for(backend, Some(model))?;
    let root = registry.root();
    let files = match backend {
        #[cfg(feature = "qwen")]
        "qwen" => {
            let selected = crate::qwen::models::Model::parse(Some(model))?;
            let json = match model {
                "0.6b-customvoice" => include_str!("qwen/resources.json"),
                "1.7b-customvoice" => include_str!("qwen/models/1.7b-customvoice.json"),
                "0.6b-base" => include_str!("qwen/models/0.6b-base.json"),
                "1.7b-base" => include_str!("qwen/models/1.7b-base.json"),
                "1.7b-voicedesign" => include_str!("qwen/models/1.7b-voicedesign.json"),
                _ => anyhow::bail!("unknown Qwen model"),
            };
            manifest(root, &selected.directory(root), json)?
        }
        #[cfg(feature = "voxcpm")]
        "voxcpm" => {
            let selected = crate::voxcpm::models::Model::parse(Some(model))?;
            manifest(root, &selected.directory(root), selected.manifest())?
        }
        #[cfg(feature = "omnivoice")]
        "omnivoice" => manifest(
            root,
            &crate::omnivoice::directory(root),
            include_str!("omnivoice/resources.json"),
        )?,
        #[cfg(feature = "cosyvoice3")]
        "cosyvoice3" => manifest(
            root,
            &crate::cosyvoice3::directory(root),
            include_str!("cosyvoice3/resources.json"),
        )?,
        #[cfg(feature = "indextts")]
        "indextts" => manifest(
            root,
            &crate::indextts::directory(root),
            include_str!("indextts/conversion-manifest.json"),
        )?,
        #[cfg(feature = "moss")]
        "moss" => match model {
            #[cfg(feature = "moss-nano-candle")]
            "nano-candle" => manifest(
                root,
                &crate::moss::nano::directory(root),
                include_str!("moss/nano/resources.json"),
            )?,
            "realtime-1.7b" => {
                let mode = crate::moss::candle::Mode::parse(model)?;
                let json = talechime_moss_tts::manifests::REALTIME;
                let mut files = manifest(root, &mode.directory(root), json)?;
                files.extend(manifest(
                    root,
                    &crate::moss::candle::resources::codec_directory(root),
                    talechime_moss_tts::manifests::CODEC,
                )?);
                files
            }
            _ => anyhow::bail!("model is not compiled"),
        },
        _ => Vec::new(),
    };
    anyhow::ensure!(
        !files.is_empty(),
        "blocked: offline production manifest or verified conversion receipt unavailable for {backend}/{model}"
    );
    Ok(Receipt {
        schema: "talechime/performance-resources/1",
        backend: backend.into(),
        model: model.into(),
        resources_root: root.to_owned(),
        verification: if backend == "indextts" {
            "verified_conversion_manifest"
        } else {
            "production_manifest"
        },
        files,
    })
}

/// Load an already verified model using the production owner path, without prepare.
/// The caller must retain the receipt and complete verification before timing load.
pub async fn load(
    registry: &crate::Registry,
    backend: &str,
    model: &str,
    device: talechime_protocol::Device,
) -> anyhow::Result<std::rc::Rc<dyn talechime_core::backend::Backend>> {
    registry.capabilities_for(backend, Some(model))?;
    #[cfg(any(
        feature = "qwen",
        feature = "voxcpm",
        feature = "omnivoice",
        feature = "cosyvoice3",
        feature = "indextts",
        feature = "moss"
    ))]
    {
        anyhow::ensure!(
            registry
                .compiled_devices_for(backend, Some(model))
                .contains(&device),
            "model device is not compiled"
        );
        anyhow::ensure!(
            registry
                .available_devices_for(backend, Some(model))
                .contains(&device),
            "model device is unavailable; no fallback"
        );
    }
    let root = registry.root();
    let loaded: Option<std::rc::Rc<dyn talechime_core::backend::Backend>> = match backend {
        #[cfg(feature = "qwen")]
        "qwen" => Some(std::rc::Rc::new(
            crate::qwen::QwenBackend::load_on(
                crate::qwen::models::Model::parse(Some(model))?.directory(root),
                device,
            )
            .await?,
        )),
        #[cfg(feature = "voxcpm")]
        "voxcpm" => {
            let selected = crate::voxcpm::models::Model::parse(Some(model))?;
            Some(std::rc::Rc::new(
                crate::voxcpm::VoxBackend::load_model_on(
                    selected.directory(root),
                    selected,
                    device,
                )
                .await?,
            ))
        }
        #[cfg(feature = "omnivoice")]
        "omnivoice" => Some(std::rc::Rc::new(
            crate::omnivoice::OmniBackend::load_on(crate::omnivoice::directory(root), device)
                .await?,
        )),
        #[cfg(feature = "cosyvoice3")]
        "cosyvoice3" => Some(std::rc::Rc::new(
            crate::cosyvoice3::CosyBackend::load_on(crate::cosyvoice3::directory(root), device)
                .await?,
        )),
        #[cfg(feature = "indextts")]
        "indextts" => Some(std::rc::Rc::new(
            crate::indextts::IndexBackend::load_on(crate::indextts::directory(root), device)
                .await?,
        )),
        #[cfg(feature = "moss")]
        "moss" => match model {
            #[cfg(feature = "moss-nano-candle")]
            "nano-candle" => Some(std::rc::Rc::new(
                crate::moss::nano::NanoBackend::load_on(root.to_owned(), device).await?,
            )),
            "realtime-1.7b" => Some(std::rc::Rc::new(
                crate::moss::candle::CandleBackend::load_on(
                    root.to_owned(),
                    crate::moss::candle::Mode::parse(model)?,
                    device,
                )
                .await?,
            )),
            _ => None,
        },
        _ => None,
    };
    let _ = (root, device); // Empty-feature catalogue builds have no model arms.
    loaded.ok_or_else(|| anyhow::anyhow!("backend/model is not compiled"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verification_preserves_corrupt_and_missing_resources() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("weights");
        std::fs::write(&path, b"keep me")?;
        let json = r#"[{"path":"weights","url":"https://example.invalid/no-network","size":7,"sha256":"bad"}]"#;
        assert!(manifest(root.path(), root.path(), json).is_err());
        assert_eq!(std::fs::read(&path)?, b"keep me");
        assert_eq!(std::fs::read_dir(root.path())?.count(), 1);
        assert!(
            manifest(
                root.path(),
                root.path(),
                &json.replace("weights", "missing")
            )
            .is_err()
        );
        assert_eq!(std::fs::read_dir(root.path())?.count(), 1);
        Ok(())
    }
}
