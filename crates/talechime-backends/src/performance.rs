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
    let resources: Vec<ManifestFile> =
        serde_json::from_value(value.get("resources").unwrap_or(&value).clone())?;
    verify_resources(root, directory, resources)
}
#[derive(serde::Deserialize)]
struct ManifestFile {
    path: String,
    size: u64,
    #[serde(default)]
    sha256: Option<String>,
}

fn verify_resources(
    root: &Path,
    directory: &Path,
    resources: Vec<ManifestFile>,
) -> anyhow::Result<Vec<VerifiedFile>> {
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
        // Downloaded manifests pin revisions without content hashes; the
        // receipt records the recomputed digest as run evidence.
        let sha256 = crate::resources::sha256_of(&path)?;
        anyhow::ensure!(
            resource
                .sha256
                .as_ref()
                .is_none_or(|expected| *expected == sha256),
            "model checksum mismatch: {}",
            path.display()
        );
        anyhow::ensure!(
            path.metadata()?.len() == resource.size,
            "unexpected size for {}",
            path.display()
        );
        files.push(VerifiedFile {
            path: path.strip_prefix(root)?.to_owned(),
            size: resource.size,
            sha256,
        });
    }
    Ok(files)
}

/// Verify pinned files in the shared Hub cache without downloading them.
#[cfg(any(
    feature = "kokoro",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "cosyvoice3",
    feature = "indextts",
    feature = "moss"
))]
pub async fn verify(
    registry: &crate::Registry,
    backend: &str,
    model: &str,
) -> anyhow::Result<Receipt> {
    registry.resolve(backend, Some(model))?;
    let manifests = crate::registry::model_manifests(backend, model)?;
    let root = registry.downloads().cache_dir().canonicalize()?;
    let mut files = Vec::new();
    for manifest in manifests {
        let paths =
            crate::resources::resolve_manifest(manifest, registry.downloads().clone()).await?;
        let entries: Vec<crate::resources::Resource> = serde_json::from_str(manifest)?;
        for entry in entries {
            let path = paths
                .get(&entry.path)
                .ok_or_else(|| anyhow::anyhow!("missing verified resource"))?;
            files.push(VerifiedFile {
                path: path.strip_prefix(&root)?.to_owned(),
                size: entry.size,
                sha256: entry.sha256,
            });
        }
    }
    Ok(Receipt {
        schema: "talechime/performance-resources/1",
        backend: backend.into(),
        model: model.into(),
        resources_root: root,
        verification: "pinned_hub_manifest",
        files,
    })
}

#[cfg(not(any(
    feature = "kokoro",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "cosyvoice3",
    feature = "indextts",
    feature = "moss"
)))]
pub async fn verify(_: &crate::Registry, _: &str, _: &str) -> anyhow::Result<Receipt> {
    anyhow::bail!("backend/model is not compiled")
}

/// Load the same verified cache resources used by the production registry.
pub async fn load(
    registry: &crate::Registry,
    backend: &str,
    model: &str,
    device: talechime_protocol::Device,
) -> anyhow::Result<std::rc::Rc<dyn talechime_core::backend::Backend>> {
    let model = registry.resolve(backend, Some(model))?;
    anyhow::ensure!(
        model.compiled_devices().contains(&device) && model.available_devices().contains(&device),
        "model device is unavailable; no fallback"
    );
    let (tx, mut rx) = tokio::sync::mpsc::channel(32);
    let drain = async { while rx.recv().await.is_some() {} };
    let load = model.load_existing(
        crate::PrepareOptions {
            device,
            ..Default::default()
        },
        tx,
    );
    let (result, ()) = tokio::join!(load, drain);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "kokoro")]
    #[tokio::test]
    async fn kokoro_rejects_locally_invented_manifests_and_wrong_modes() -> anyhow::Result<()> {
        use sha2::{Digest, Sha256};
        let root = tempfile::tempdir()?;
        let directory = crate::kokoro::directory(root.path());
        std::fs::create_dir_all(directory.join("frontend"))?;
        std::fs::create_dir_all(directory.join("voices"))?;
        let mut files = serde_json::Map::new();
        for path in [
            "model.safetensors",
            "config.json",
            "frontend/frontend.json",
            "frontend/jieba.dict",
            "official-export.json",
            "KOKORO-LICENSE",
            "voices/zf_001.safetensors",
        ] {
            std::fs::write(directory.join(path), b"fixture")?;
            files.insert(
                path.into(),
                serde_json::json!({"bytes":7,"sha256":format!("{:x}",Sha256::digest(b"fixture"))}),
            );
        }
        let voices: serde_json::Value =
            serde_json::from_str(talechime_kokoro::BUILTIN_VOICES_JSON)?;
        let voice = voices
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["id"] == "zf_001")
            .unwrap();
        let manifest = serde_json::json!({
            "format":"kokoro-native-v1", "model_revision":talechime_kokoro::MODEL_REVISION,
            "contextual":false, "files":files,
            "voices":{"zf_001":{"path":"voices/zf_001.safetensors","source_sha256":voice["sha256"]}}
        });
        std::fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec(&manifest)?,
        )?;
        let registry = crate::Registry::with_downloads(
            Some(root.path().join("data")),
            &crate::downloads::DownloadConfig {
                cache_dir: root.path().join("hub"),
                ..Default::default()
            },
        )?;
        assert!(
            verify(&registry, "kokoro", crate::kokoro::MODEL)
                .await
                .is_err()
        );
        assert_eq!(
            std::fs::read(directory.join("model.safetensors"))?,
            b"fixture"
        );
        std::fs::write(directory.join("model.safetensors"), b"corrupt")?;
        assert!(
            verify(&registry, "kokoro", crate::kokoro::MODEL)
                .await
                .is_err()
        );
        assert_eq!(
            std::fs::read(directory.join("model.safetensors"))?,
            b"corrupt"
        );
        Ok(())
    }
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
