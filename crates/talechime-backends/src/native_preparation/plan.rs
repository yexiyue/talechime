use super::{PreparationControl, Recipe};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::Read,
    path::{Component, Path},
};
use tokio::sync::mpsc;

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Input {
    pub path: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub filename: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    /// A fixed public artifact outside the Hub; never inherits Hub credentials.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<&'static str>,
}
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Output {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}
/// Stage facts are separate from network byte progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PreparationPhase {
    /// Fetching missing pinned inputs.
    Downloading,
    /// Applying a versioned native transform.
    Converting,
    /// Checking all generated resources.
    Verifying,
    /// Atomically publishing a complete resource version.
    Publishing,
}
/// A bounded stage update; a zero total means indeterminate work.
#[derive(Clone, Debug, Serialize)]
pub struct PreparationProgress {
    /// Current stage.
    pub phase: PreparationPhase,
    /// Component identity, never a signed URL or token.
    pub component: String,
    /// Completed units within this stage.
    pub completed: u64,
    /// Stage units, not implied download bytes.
    pub total: u64,
}
/// Offline requirements bound to fixed sources, conversion version and cache.
#[derive(Clone, Debug, Serialize)]
pub struct PreparationPlan {
    /// Immutable recipe identity.
    pub recipe: String,
    /// Opaque confirmation identity. Recompute immediately before execution.
    pub fingerprint: String,
    /// All pinned outputs and the publication marker are already valid.
    pub ready: bool,
    /// Input bytes not available in the verified input view or shared cache.
    pub download_bytes: u64,
    /// Conservative additional disk space for missing inputs, outputs and workspace.
    pub peak_additional_bytes: u64,
    /// Full runtime resource size, independent of current cache hits.
    pub output_bytes: u64,
    /// Native transforms whose complete pinned outputs cannot be reused yet.
    pub conversions: Vec<String>,
}
pub(super) fn safe_relative(path: &Path) -> bool {
    !path.as_os_str().is_empty() && path.components().all(|c| matches!(c, Component::Normal(_)))
}
pub(super) fn marker(recipe: &Recipe) -> anyhow::Result<Vec<u8>> {
    Ok(serde_json::to_vec(&(
        1u32,
        recipe.id,
        recipe.inputs,
        recipe.outputs,
    ))?)
}
pub(super) fn marker_matches(path: &Path, expected: &[u8]) -> anyhow::Result<bool> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    if !metadata.is_file() || metadata.len() != expected.len() as u64 {
        return Ok(false);
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(expected.len() as u64 + 1)
        .read_to_end(&mut bytes)?;
    Ok(bytes == expected)
}
pub(super) fn verified(
    path: &Path,
    size: u64,
    expected: &str,
    control: &PreparationControl,
) -> anyhow::Result<bool> {
    control.check()?;
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(e.into()),
    };
    if !metadata.is_file() || metadata.len() != size {
        return Ok(false);
    }
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        control.check()?;
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()) == expected)
}
/// Reuse verified complete components across source-build and distribution recipes.
pub(super) fn cached_input(
    root: &Path,
    downloads: &crate::downloads::Downloads,
    input: &Input,
    control: &PreparationControl,
) -> anyhow::Result<Option<std::path::PathBuf>> {
    let component = root
        .join(".native-preparation/components")
        .join(input.sha256);
    if verified(&component, input.size, input.sha256, control)? {
        return Ok(Some(component));
    }
    if input.url.is_none()
        && let Some(path) = downloads.cached_path(input.repo, input.revision, input.filename)
        && verified(&path, input.size, input.sha256, control)?
    {
        return Ok(Some(path));
    }
    Ok(None)
}
pub(super) fn build(
    recipe: &Recipe,
    root: &Path,
    downloads: &crate::downloads::Downloads,
    control: &PreparationControl,
) -> anyhow::Result<PreparationPlan> {
    anyhow::ensure!(
        root.is_absolute()
            && safe_relative(&recipe.target)
            && safe_relative(Path::new(recipe.id))
            && Path::new(recipe.id).components().count() == 1,
        "invalid preparation target"
    );
    let inputs_root = root.join(".native-preparation/inputs").join(recipe.id);
    let mut missing = 0u64;
    let mut missing_inputs = Vec::new();
    let mut input_growth = 0u64;
    let mut seen = std::collections::BTreeSet::new();
    for input in recipe.inputs {
        if let Some(url) = input.url {
            let url = reqwest::Url::parse(url)?;
            anyhow::ensure!(
                url.scheme() == "https"
                    && url.username().is_empty()
                    && url.password().is_none()
                    && url.query().is_none()
                    && url.fragment().is_none(),
                "resource URL must be fixed public HTTPS without credentials or query"
            );
        }
        anyhow::ensure!(
            safe_relative(Path::new(input.path))
                && safe_relative(Path::new(input.filename))
                && seen.insert(input.path),
            "invalid or duplicate preparation input"
        );
        let revision_pinned = if input.url.is_some() {
            !input.revision.is_empty()
                && input.revision.len() <= 128
                && input
                    .revision
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
        } else {
            input.revision.len() == 40 && input.revision.bytes().all(|b| b.is_ascii_hexdigit())
        };
        anyhow::ensure!(
            revision_pinned
                && input.sha256.len() == 64
                && input.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "unpinned preparation input"
        );
        let present = verified(
            &inputs_root.join(input.path),
            input.size,
            input.sha256,
            control,
        )?;
        let cached = if present {
            None
        } else {
            cached_input(root, downloads, input, control)?
        };
        if !present && cached.as_ref().is_none_or(|path| !same_volume(root, path)) {
            input_growth = input_growth
                .checked_add(input.size)
                .ok_or_else(|| anyhow::anyhow!("resource size overflow"))?;
        }
        if !present && cached.is_none() {
            missing_inputs.push(input.path);
            missing = missing
                .checked_add(input.size)
                .ok_or_else(|| anyhow::anyhow!("resource size overflow"))?;
        }
    }
    let mut output_bytes = 0u64;
    let mut output_names = std::collections::BTreeSet::new();
    for output in recipe.outputs {
        anyhow::ensure!(
            safe_relative(Path::new(output.path))
                && output.path != "preparation.json"
                && !output.path.starts_with("preparation.json/")
                && output_names.insert(output.path)
                && output.sha256.len() == 64
                && output.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid preparation output"
        );
        output_bytes = output_bytes
            .checked_add(output.size)
            .ok_or_else(|| anyhow::anyhow!("resource size overflow"))?;
    }
    anyhow::ensure!(
        !recipe.inputs.is_empty() && !recipe.outputs.is_empty(),
        "empty preparation recipe"
    );
    let marker = marker(recipe)?;
    let mut ready = marker_matches(&root.join(&recipe.target).join("preparation.json"), &marker)?;
    for output in recipe.outputs {
        ready &= verified(
            &root.join(&recipe.target).join(output.path),
            output.size,
            output.sha256,
            control,
        )?;
    }
    if ready {
        missing_inputs.clear();
        missing = 0;
    }
    let peak = input_growth
        .checked_add(output_bytes)
        .and_then(|v| v.checked_add(recipe.workspace_bytes))
        .ok_or_else(|| anyhow::anyhow!("resource size overflow"))?;
    let peak = if ready { 0 } else { peak };
    let mut conversions = Vec::new();
    let mut transforms = std::collections::BTreeSet::new();
    for name in recipe.transform_outputs {
        anyhow::ensure!(
            transforms.insert(*name),
            "duplicate native transform component"
        );
        let pin = recipe
            .outputs
            .iter()
            .find(|pin| pin.path == *name)
            .ok_or_else(|| anyhow::anyhow!("native transform output is not in the recipe"))?;
        if !ready
            && !verified(
                &root.join(".native-preparation/components").join(pin.sha256),
                pin.size,
                pin.sha256,
                control,
            )?
        {
            conversions.push((*name).to_owned());
        }
    }
    missing_inputs.sort_unstable();
    let identity = serde_json::to_vec(&(
        recipe.id,
        &recipe.target,
        recipe.inputs,
        recipe.outputs,
        recipe.workspace_bytes,
        root,
        downloads.cache_dir(),
        downloads.source_identity(),
        ready,
        missing,
        &missing_inputs,
        peak,
        &conversions,
    ))?;
    Ok(PreparationPlan {
        recipe: recipe.id.into(),
        ready,
        fingerprint: format!("{:x}", Sha256::digest(identity)),
        download_bytes: missing,
        peak_additional_bytes: peak,
        output_bytes,
        conversions,
    })
}
pub(super) fn send(
    tx: &mpsc::Sender<PreparationProgress>,
    phase: PreparationPhase,
    component: &str,
    completed: u64,
    total: u64,
) {
    // Slow observers may miss intermediate counters; snapshots/terminal results
    // remain authoritative, and native work never blocks waiting for UI capacity.
    let _ = tx.try_send(PreparationProgress {
        phase,
        component: component.into(),
        completed,
        total,
    });
}

fn same_volume(left: &Path, right: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let device = |path: &Path| {
            path.ancestors().find_map(|parent| {
                std::fs::metadata(parent)
                    .ok()
                    .map(|metadata| metadata.dev())
            })
        };
        matches!((device(left), device(right)), (Some(a), Some(b)) if a == b)
    }
    #[cfg(not(unix))]
    {
        let _ = (left, right);
        false
    }
}
