//! Versioned facts and no-clobber publication used by developer tooling.
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Component, Path},
    process::Command,
};

pub const VERSION: &str = "talechime/performance/1";

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn file_digest(path: &Path) -> anyhow::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Paths in evidence are portable relative paths. Existing symlinks cannot escape root.
pub fn contained(root: &Path, relative: &Path) -> anyhow::Result<std::path::PathBuf> {
    ensure!(!relative.as_os_str().is_empty(), "empty evidence path");
    ensure!(
        relative
            .components()
            .all(|part| matches!(part, Component::Normal(_))),
        "non-relative evidence path"
    );
    let root = root.canonicalize()?;
    let path = root.join(relative);
    let mut parent = path.as_path();
    while !parent.exists() {
        parent = parent.parent().context("evidence path has no parent")?;
    }
    ensure!(
        parent.canonicalize()?.starts_with(&root),
        "evidence path escapes root"
    );
    Ok(path)
}

/// A fully flushed staging file is published using an atomic no-clobber hard link.
pub fn publish_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    publish_bytes(path, &bytes)
}

pub fn publish_bytes(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let stage = parent.join(format!(
        ".perf-{}-{}.staging",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&stage)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::hard_link(&stage, path)
            .with_context(|| format!("publish without overwrite: {}", path.display()))?;
        Ok(())
    })();
    let _ = fs::remove_file(stage);
    result
}

pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    const LIMIT: u64 = 16 * 1024 * 1024;
    let file = fs::File::open(path)?;
    ensure!(file.metadata()?.len() <= LIMIT, "metadata exceeds 16 MiB");
    Ok(serde_json::from_reader(file)?)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub revision: String,
    pub dirty_sha256: String,
    pub untracked: Vec<(String, String)>,
}

impl Source {
    pub fn capture(root: &Path) -> anyhow::Result<Self> {
        fn git(root: &Path, args: &[&str]) -> anyhow::Result<Vec<u8>> {
            let out = Command::new("git").current_dir(root).args(args).output()?;
            ensure!(out.status.success(), "git source identity failed");
            Ok(out.stdout)
        }
        let revision = String::from_utf8(git(root, &["rev-parse", "HEAD"])?)?
            .trim()
            .to_owned();
        let dirty_sha256 = digest(&git(root, &["diff", "HEAD", "--binary", "--", "."])?);
        let paths = git(root, &["ls-files", "--others", "--exclude-standard", "-z"])?;
        let mut untracked = Vec::new();
        for bytes in paths.split(|b| *b == 0).filter(|b| !b.is_empty()) {
            let name = std::str::from_utf8(bytes)?.to_owned();
            let path = contained(root, Path::new(&name))?;
            ensure!(
                path.is_file(),
                "untracked identity requires regular file: {name}"
            );
            untracked.push((name, file_digest(&path)?));
        }
        untracked.sort();
        Ok(Self {
            revision,
            dirty_sha256,
            untracked,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Reading {
    pub value: Option<f64>,
    pub unit: String,
    pub scope: String,
    pub source: String,
    pub interval_ms: Option<u64>,
    pub status: String,
    pub reason: Option<String>,
}

impl Reading {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            !self.unit.is_empty() && !self.scope.is_empty() && !self.source.is_empty(),
            "reading semantics missing"
        );
        match self.status.as_str() {
            "available" => ensure!(
                self.value.is_some_and(f64::is_finite) && self.reason.is_none(),
                "invalid available reading"
            ),
            "unsupported" | "unavailable" | "deferred" => ensure!(
                self.value.is_none() && self.reason.as_ref().is_some_and(|r| !r.is_empty()),
                "missing reading requires reason"
            ),
            _ => anyhow::bail!("unknown reading status"),
        }
        ensure!(
            !(self.source.contains("unified") && self.scope == "process_exclusive_vram"),
            "unified memory is not exclusive VRAM"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_identity_includes_dirty_and_untracked_bytes() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let git = |args: &[&str]| -> anyhow::Result<()> {
            let output = Command::new("git")
                .current_dir(root.path())
                .args(args)
                .output()?;
            ensure!(
                output.status.success(),
                "fixture git failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            Ok(())
        };
        git(&["init"])?;
        git(&["config", "user.name", "Fixture"])?;
        git(&["config", "user.email", "fixture@example.invalid"])?;
        fs::write(root.path().join("tracked.rs"), "first")?;
        git(&["add", "tracked.rs"])?;
        git(&["-c", "commit.gpgsign=false", "commit", "-m", "fixture"])?;
        let clean = Source::capture(root.path())?;
        fs::write(root.path().join("tracked.rs"), "changed")?;
        fs::write(root.path().join("untracked.rs"), "one")?;
        let dirty = Source::capture(root.path())?;
        assert_eq!(clean.revision, dirty.revision);
        assert_ne!(clean.dirty_sha256, dirty.dirty_sha256);
        assert_eq!(dirty.untracked.len(), 1);
        fs::write(root.path().join("untracked.rs"), "two")?;
        assert_ne!(dirty.untracked, Source::capture(root.path())?.untracked);
        Ok(())
    }
    #[test]
    fn publication_never_overwrites_and_cleans_stage() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("plan.json");
        publish_json(&path, &vec![1])?;
        assert!(publish_json(&path, &vec![2]).is_err());
        assert_eq!(read_json::<Vec<u32>>(&path)?, vec![1]);
        assert_eq!(fs::read_dir(root.path())?.count(), 1);
        Ok(())
    }
    #[test]
    fn paths_reject_escape() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        for name in ["../bad", "", "/absolute", "a/../../b"] {
            assert!(contained(root.path(), Path::new(name)).is_err(), "{name}");
        }
        assert!(contained(root.path(), Path::new("cases/one.json")).is_ok());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symlink_assets_cannot_escape_report_root() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let outside = tempfile::tempdir()?;
        fs::write(outside.path().join("model.bin"), b"private model")?;
        std::os::unix::fs::symlink(outside.path(), root.path().join("assets"))?;
        assert!(contained(root.path(), Path::new("assets/model.bin")).is_err());
        assert!(contained(root.path(), Path::new("assets/new.json")).is_err());
        Ok(())
    }
    #[test]
    fn missing_readings_cannot_be_zero_or_exclusive_vram() {
        let mut reading = Reading {
            value: None,
            unit: "bytes".into(),
            scope: "device_shared".into(),
            source: "macos_unified".into(),
            interval_ms: Some(500),
            status: "deferred".into(),
            reason: Some("native collector pending".into()),
        };
        assert!(reading.validate().is_ok());
        reading.value = Some(0.0);
        assert!(reading.validate().is_err());
        reading.value = Some(1.0);
        reading.status = "available".into();
        reading.reason = None;
        reading.scope = "process_exclusive_vram".into();
        assert!(reading.validate().is_err());
    }
}
