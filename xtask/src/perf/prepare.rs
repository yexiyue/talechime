//! Read-only resource preflight through a verified production catalogue executable.
use super::{
    build::Build,
    evidence::{VERSION, file_digest, publish_json, read_json},
};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub schema: String,
    pub root: PathBuf,
    pub resources: PathBuf,
    pub catalog: Vec<serde_json::Value>,
    pub path_resolution: String,
    pub resource_acquisition: bool,
}

impl Catalog {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.schema == "talechime/performance-catalog/1",
            "unknown catalog schema"
        );
        ensure!(
            self.path_resolution == "pure_no_migration" && !self.resource_acquisition,
            "catalogue must be offline and pure"
        );
        ensure!(
            self.resources == self.root.join("resources"),
            "catalogue path identity mismatch"
        );
        Ok(())
    }
}

/// Trusted, non-inference helper: files bound memory and polling bounds disk/time.
pub fn capture(command: &mut Command, timeout: Duration) -> anyhow::Result<Vec<u8>> {
    let stdout = tempfile::tempfile()?;
    let stderr = tempfile::tempfile()?;
    let mut child = command
        .stdin(Stdio::null())
        .stdout(stdout.try_clone()?)
        .stderr(stderr.try_clone()?)
        .spawn()?;
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline
            || stdout.metadata()?.len() > 16 * 1024 * 1024
            || stderr.metadata()?.len() > 16 * 1024 * 1024
        {
            child.kill()?;
            child.wait()?;
            anyhow::bail!("catalogue helper exceeded timeout or output budget");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    use std::io::{Read, Seek};
    let mut stdout = stdout;
    let mut stderr = stderr;
    ensure!(
        stdout.metadata()?.len() <= 16 * 1024 * 1024
            && stderr.metadata()?.len() <= 16 * 1024 * 1024,
        "helper output exceeds budget"
    );
    stdout.rewind()?;
    stderr.rewind()?;
    let mut errors = String::new();
    stderr.read_to_string(&mut errors)?;
    ensure!(status.success(), "offline preflight blocked: {errors}");
    let mut bytes = Vec::new();
    stdout.read_to_end(&mut bytes)?;
    Ok(bytes)
}

pub fn prepare(
    build_path: &Path,
    app_root: Option<&Path>,
    backend: Option<&str>,
    model: Option<&str>,
    output: &Path,
) -> anyhow::Result<()> {
    ensure!(!output.exists(), "preparation receipt exists");
    let build: Build = read_json(build_path)?;
    build.verify()?;
    ensure!(
        build.probe == "performance_catalog",
        "prepare requires catalogue executable"
    );
    let mut command = Command::new(&build.executable);
    if let Some(root) = app_root {
        command.arg(root);
    }
    let catalog: Catalog =
        serde_json::from_slice(&capture(&mut command, Duration::from_secs(30))?)?;
    catalog.validate()?;
    let resource_receipt = match (backend, model) {
        (Some(backend), Some(model)) => {
            let mut command = Command::new(&build.executable);
            command.arg(&catalog.root).args([backend, model]);
            let value: serde_json::Value =
                serde_json::from_slice(&capture(&mut command, Duration::from_secs(600))?)?;
            ensure!(
                value["schema"] == "talechime/performance-resources/1"
                    && value["backend"] == backend
                    && value["model"] == model,
                "resource receipt identity mismatch"
            );
            Some(value)
        }
        (None, None) => None,
        _ => anyhow::bail!("backend and model must be supplied together"),
    };
    ensure!(
        file_digest(&build.executable)? == build.binary_sha256,
        "catalogue binary changed during prepare"
    );
    publish_json(
        output,
        &serde_json::json!({
            "schema": VERSION, "catalog": catalog, "build_receipt_sha256": file_digest(build_path)?,
            "binary_sha256": build.binary_sha256, "resource_receipt": resource_receipt,
            "status": if backend.is_some() { "resources_verified" } else { "catalog_only" },
            "execution_verified": false,
            "acquisition": "disabled; missing importer is blocked, never Python fallback"
        }),
    )
}

pub fn load_catalog(path: &Path) -> anyhow::Result<Catalog> {
    let value: serde_json::Value = read_json(path)?;
    ensure!(value["schema"] == VERSION, "unknown preparation schema");
    let catalog: Catalog =
        serde_json::from_value(value.get("catalog").context("catalog missing")?.clone())?;
    catalog.validate()?;
    Ok(catalog)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalog_paths_do_not_create_or_migrate_anything() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let app = root.path().join("custom-root");
        let catalog = Catalog {
            schema: "talechime/performance-catalog/1".into(),
            resources: app.join("resources"),
            root: app.clone(),
            catalog: vec![],
            path_resolution: "pure_no_migration".into(),
            resource_acquisition: false,
        };
        catalog.validate()?;
        assert!(!app.exists());
        assert_eq!(std::fs::read_dir(root.path())?.count(), 0);
        Ok(())
    }
}
