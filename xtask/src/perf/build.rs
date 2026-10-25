//! Explicit, isolated probe builds located from Cargo artifact messages.
use super::evidence::{Source, VERSION, digest, file_digest, publish_json};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Build {
    pub schema: String,
    pub source: Source,
    pub probe: String,
    pub features: Vec<String>,
    pub executable: PathBuf,
    pub binary_sha256: String,
    pub target: String,
    pub profile: String,
}

impl Build {
    pub fn verify(&self) -> anyhow::Result<()> {
        ensure!(
            self.schema == VERSION && self.profile == "release",
            "unsupported build identity"
        );
        ensure!(
            ["performance_catalog", "performance_probe"].contains(&self.probe.as_str()),
            "unknown performance probe"
        );
        ensure!(
            file_digest(&self.executable)? == self.binary_sha256,
            "probe binary changed"
        );
        Ok(())
    }
}

pub fn artifact(line: &str, probe: &str) -> anyhow::Result<Option<(PathBuf, Vec<String>)>> {
    let value: serde_json::Value = serde_json::from_str(line)?;
    if value["reason"] != "compiler-artifact" || value["target"]["name"] != probe {
        return Ok(None);
    }
    let kinds: Vec<String> = serde_json::from_value(value["target"]["kind"].clone())?;
    if !kinds.iter().any(|kind| kind == "example") {
        return Ok(None);
    }
    ensure!(
        !value["profile"]["test"].as_bool().unwrap_or(true),
        "test artifact is not a measurement executable"
    );
    let path = value["executable"]
        .as_str()
        .context("Cargo artifact has no executable")?;
    let features = serde_json::from_value(value["features"].clone())?;
    Ok(Some((PathBuf::from(path), features)))
}

const DYNAMIC_ORT_FEATURE: &str = "ort/load-dynamic";

fn dynamic_ort_artifact(line: &str) -> anyhow::Result<bool> {
    let value: serde_json::Value = serde_json::from_str(line)?;
    if value["reason"] != "compiler-artifact"
        || value["target"]["name"] != "ort"
        || value["profile"]["test"] != false
    {
        return Ok(false);
    }
    let kinds: Vec<String> = serde_json::from_value(value["target"]["kind"].clone())?;
    let features: Vec<String> = serde_json::from_value(value["features"].clone())?;
    Ok(kinds.iter().any(|kind| kind == "lib")
        && features.iter().any(|feature| feature == "load-dynamic"))
}

pub fn build(probe: &str, mut features: Vec<String>, output: &Path) -> anyhow::Result<()> {
    ensure!(
        ["performance_catalog", "performance_probe"].contains(&probe),
        "unsupported probe name"
    );
    ensure!(!output.exists(), "build receipt already exists");
    features.push("performance-tools".into());
    features.sort();
    features.dedup();
    for feature in &features {
        ensure!(
            feature == DYNAMIC_ORT_FEATURE
                || (!feature.is_empty()
                    && feature
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'-')),
            "invalid feature name"
        );
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let compiler = Command::new("rustc").arg("-vV").output()?;
    ensure!(compiler.status.success(), "cannot determine host target");
    let compiler = std::str::from_utf8(&compiler.stdout)?;
    let target = compiler
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .context("rustc host triple missing")?
        .to_owned();
    let source = Source::capture(root)?;
    let target_dir = root.join("target/perf-builds").join(digest(
        serde_json::to_string(&(&features, &target))?.as_bytes(),
    ));
    let mut command = Command::new("cargo");
    command
        .current_dir(root)
        .args([
            "build",
            "--locked",
            "--release",
            "-p",
            "talechime-backends",
            "--no-default-features",
            "--example",
            probe,
            "--message-format=json-render-diagnostics",
            "--target-dir",
        ])
        .arg(target_dir)
        .arg("--target")
        .arg(&target);
    if !features.is_empty() {
        command.arg("--features").arg(features.join(","));
    }
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stream = BufReader::new(child.stdout.take().context("Cargo stdout unavailable")?);
    let mut found = None;
    let mut dynamic_ort = false;
    let parsed = (|| -> anyhow::Result<()> {
        for line in stream.lines() {
            let line = line?;
            ensure!(
                line.len() <= 4 * 1024 * 1024,
                "Cargo artifact message too large"
            );
            if let Some(item) = artifact(&line, probe)? {
                found = Some(item);
            }
            dynamic_ort |= dynamic_ort_artifact(&line)?;
        }
        Ok(())
    })();
    if parsed.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    parsed?;
    ensure!(status.success(), "probe build failed");
    let (executable, mut effective_features) =
        found.context("Cargo did not emit requested example artifact")?;
    if features
        .iter()
        .any(|feature| feature == DYNAMIC_ORT_FEATURE)
    {
        ensure!(dynamic_ort, "Cargo did not confirm ORT load-dynamic");
    }
    if dynamic_ort {
        effective_features.push(DYNAMIC_ORT_FEATURE.into());
        effective_features.sort();
        effective_features.dedup();
    }
    ensure!(
        features.iter().all(|f| effective_features.contains(f)),
        "requested features missing from artifact"
    );
    ensure!(
        Source::capture(root)? == source,
        "source changed while building; retry with stable sources"
    );
    // Cargo overwrites the same example path on subsequent builds. Receipts
    // must reference a preserved copy so an A/B rebuild cannot destroy A.
    let executable = preserve_executable(&executable, output, probe)?;
    let receipt = Build {
        schema: VERSION.into(),
        source,
        probe: probe.into(),
        features: effective_features,
        binary_sha256: file_digest(&executable)?,
        executable,
        target,
        profile: "release".into(),
    };
    receipt.verify()?;
    publish_json(output, &receipt)
}

fn preserve_executable(source: &Path, receipt: &Path, probe: &str) -> anyhow::Result<PathBuf> {
    let hash = file_digest(source)?;
    let root = receipt.parent().unwrap_or(Path::new("."));
    let directory = root.join("binaries").join(&hash);
    std::fs::create_dir_all(&directory)?;
    let name = format!(
        "{probe}{}",
        source
            .extension()
            .map(|e| format!(".{}", e.to_string_lossy()))
            .unwrap_or_default()
    );
    let destination = directory.join(name);
    if !destination.exists() {
        let stage = tempfile::NamedTempFile::new_in(&directory)?;
        std::fs::copy(source, stage.path())?;
        stage.as_file().sync_all()?;
        ensure!(
            file_digest(stage.path())? == hash,
            "probe changed while preserving binary"
        );
        if let Err(error) = stage.persist_noclobber(&destination) {
            ensure!(
                error.error.kind() == std::io::ErrorKind::AlreadyExists,
                "cannot publish preserved probe: {}",
                error.error
            );
        }
    }
    ensure!(
        file_digest(&destination)? == hash,
        "preserved probe binary changed"
    );
    Ok(destination.canonicalize()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dynamic_ort_requires_its_own_library_artifact() -> anyhow::Result<()> {
        let mut value = serde_json::json!({"reason":"compiler-artifact","target":{"name":"ort","kind":["lib"]},"profile":{"test":false},"features":["cuda","load-dynamic"]});
        assert!(dynamic_ort_artifact(&value.to_string())?);
        value["features"] = serde_json::json!(["cuda"]);
        assert!(!dynamic_ort_artifact(&value.to_string())?);
        value["features"] = serde_json::json!(["load-dynamic"]);
        value["target"]["name"] = "performance_probe".into();
        assert!(!dynamic_ort_artifact(&value.to_string())?);
        value["target"]["name"] = "ort".into();
        value["profile"]["test"] = true.into();
        assert!(!dynamic_ort_artifact(&value.to_string())?);
        value["profile"]["test"] = false.into();
        value["target"]["kind"] = serde_json::json!(["example"]);
        assert!(!dynamic_ort_artifact(&value.to_string())?);
        Ok(())
    }
    #[test]
    fn rebuild_preserves_both_executables_and_detects_tampering() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let source = root.path().join("cargo-probe.exe");
        let receipt = root.path().join("build.json");
        std::fs::write(&source, b"baseline")?;
        let baseline = preserve_executable(&source, &receipt, "performance_probe")?;
        std::fs::write(&source, b"candidate")?;
        let candidate = preserve_executable(&source, &receipt, "performance_probe")?;
        assert_ne!(baseline, candidate);
        assert_eq!(std::fs::read(&baseline)?, b"baseline");
        assert_eq!(std::fs::read(&candidate)?, b"candidate");
        assert_eq!(
            preserve_executable(&source, &receipt, "performance_probe")?,
            candidate
        );
        std::fs::write(candidate, b"tampered")?;
        assert!(preserve_executable(&source, &receipt, "performance_probe").is_err());
        Ok(())
    }
    #[test]
    fn changed_binary_cannot_reuse_build_identity() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let executable = root.path().join("probe");
        std::fs::write(&executable, b"one")?;
        let receipt = Build {
            schema: VERSION.into(),
            source: Source {
                revision: "fixture".into(),
                dirty_sha256: String::new(),
                untracked: vec![],
            },
            probe: "performance_catalog".into(),
            features: vec!["performance-tools".into()],
            binary_sha256: file_digest(&executable)?,
            executable: executable.clone(),
            target: "fixture".into(),
            profile: "release".into(),
        };
        receipt.verify()?;
        std::fs::write(&executable, b"two")?;
        assert!(receipt.verify().is_err());
        Ok(())
    }
    #[test]
    fn only_explicit_non_test_example_artifacts_are_accepted() -> anyhow::Result<()> {
        let mut value = serde_json::json!({"reason":"compiler-artifact","target":{"name":"performance_probe","kind":["example"]},"profile":{"test":false},"executable":"somewhere/probe.exe","features":["qwen-cuda"]});
        assert!(artifact(&value.to_string(), "performance_catalog")?.is_none());
        assert_eq!(
            artifact(&value.to_string(), "performance_probe")?
                .unwrap()
                .1,
            vec!["qwen-cuda"]
        );
        value["profile"]["test"] = true.into();
        assert!(artifact(&value.to_string(), "performance_probe").is_err());
        value["profile"]["test"] = false.into();
        value["executable"] = serde_json::Value::Null;
        assert!(artifact(&value.to_string(), "performance_probe").is_err());
        Ok(())
    }
}
