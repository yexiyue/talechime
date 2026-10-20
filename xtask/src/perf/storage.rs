//! Locks and atomic evidence commits; failed attempts are never overwritten.
use super::{
    evidence::{VERSION, contained, file_digest, publish_json, read_json},
    executor::Capture,
    records::{Artifact, Attempt},
};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    path::{Path, PathBuf},
};

pub struct RunLock {
    _file: File,
}

impl Drop for RunLock {
    fn drop(&mut self) {
        // Release the owner lock even if a concurrent fork temporarily inherited
        // the open file description before CLOEXEC takes effect.
        let _ = self._file.unlock();
    }
}

impl RunLock {
    pub fn acquire(root: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(root)?;
        let path = contained(root, Path::new("run.lock"))?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        file.try_lock().context("another runner owns this run")?;
        Ok(Self { _file: file })
    }
}

pub fn new_attempt(root: &Path) -> anyhow::Result<PathBuf> {
    let attempts = contained(root, Path::new("attempts"))?;
    std::fs::create_dir_all(&attempts)?;
    let id = format!(
        "{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos(),
        std::process::id()
    );
    let path = contained(root, &Path::new("attempts").join(id))?;
    std::fs::create_dir(&path)?;
    Ok(path)
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Commit {
    schema: String,
    attempt_sha256: String,
    logs: Vec<Artifact>,
}

pub fn commit(root: &Path, attempt: &Attempt) -> anyhow::Result<()> {
    attempt.conditions.validate()?;
    attempt.verify_artifacts(root)?;
    ensure!(
        attempt.outcome == "success",
        "failed attempt cannot be committed"
    );
    let capture: Capture = read_json(&contained(root, Path::new("capture.json"))?)?;
    ensure!(
        capture.outcome == "success" && capture.cleanup_verified && capture.errors.is_empty(),
        "process capture is incomplete"
    );
    let mut logs = Vec::new();
    for name in [
        "stdout.jsonl",
        "stderr.log",
        "capture.json",
        "resource-samples.jsonl",
    ] {
        let path = contained(root, Path::new(name))?;
        ensure!(path.is_file(), "required log missing");
        logs.push(Artifact {
            path: name.into(),
            bytes: path.metadata()?.len(),
            sha256: file_digest(&path)?,
            kind: "raw_log".into(),
            derived_from: None,
        });
    }
    let attempt_path = contained(root, Path::new("attempt.json"))?;
    publish_json(&attempt_path, attempt)?;
    let commit = Commit {
        schema: VERSION.into(),
        attempt_sha256: file_digest(&attempt_path)?,
        logs,
    };
    publish_json(&contained(root, Path::new("committed.json"))?, &commit)
}

pub fn recover(root: &Path, expected: &Attempt) -> anyhow::Result<Attempt> {
    let commit: Commit = read_json(&contained(root, Path::new("committed.json"))?)?;
    ensure!(commit.schema == VERSION, "unsupported commit schema");
    let attempt_path = contained(root, Path::new("attempt.json"))?;
    ensure!(
        file_digest(&attempt_path)? == commit.attempt_sha256,
        "committed attempt changed"
    );
    let attempt: Attempt = read_json(&attempt_path)?;
    ensure!(
        attempt.binary_sha256 == expected.binary_sha256
            && attempt.source == expected.source
            && attempt.conditions == expected.conditions
            && attempt.run_id == expected.run_id
            && attempt.case_id == expected.case_id
            && attempt.process_index == expected.process_index
            && attempt.expected_hot_rounds == expected.expected_hot_rounds,
        "resume identity mismatch; create a new attempt/run"
    );
    ensure!(
        attempt.outcome == "success",
        "only successful attempts may be reused"
    );
    attempt.conditions.validate()?;
    attempt.verify_artifacts(root)?;
    for log in commit.logs {
        log.verify(root)?;
    }
    Ok(attempt)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resume_checks_identity_and_all_committed_artifacts() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let mut attempts = super::super::test_support::attempts(&[1.0]);
        let attempt = &mut attempts[0];
        std::fs::write(root.path().join("stdout.jsonl"), "fixture log")?;
        std::fs::write(root.path().join("stderr.log"), "")?;
        std::fs::write(root.path().join("resource-samples.jsonl"), "")?;
        publish_json(
            &root.path().join("capture.json"),
            &Capture {
                outcome: "success".into(),
                exit_code: Some(0),
                cleanup_verified: true,
                errors: vec![],
            },
        )?;
        std::fs::write(root.path().join("audio.wav"), b"synthetic test artifact")?;
        attempt.rounds[1].artifacts.push(Artifact {
            path: "audio.wav".into(),
            sha256: file_digest(&root.path().join("audio.wav"))?,
            bytes: 23,
            kind: "audio_f32".into(),
            derived_from: None,
        });
        commit(root.path(), attempt)?;
        recover(root.path(), attempt)?;
        attempt.binary_sha256 = "f".repeat(64);
        assert!(recover(root.path(), attempt).is_err());
        attempt.binary_sha256 = super::super::evidence::digest(b"binary");
        std::fs::remove_file(root.path().join("audio.wav"))?;
        assert!(recover(root.path(), attempt).is_err());
        assert!(root.path().join("committed.json").is_file());
        Ok(())
    }
    #[test]
    fn run_lock_serializes_writers_and_attempts_preserve_history() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let lock = RunLock::acquire(root.path())?;
        assert!(RunLock::acquire(root.path()).is_err());
        let first = new_attempt(root.path())?;
        std::fs::write(first.join("failure.txt"), "old failure")?;
        let second = new_attempt(root.path())?;
        assert_ne!(first, second);
        assert_eq!(
            std::fs::read_to_string(first.join("failure.txt"))?,
            "old failure"
        );
        // A fork between open and exec can temporarily retain this description.
        let inherited = lock._file.try_clone()?;
        drop(lock);
        RunLock::acquire(root.path())?;
        drop(inherited);
        Ok(())
    }
}
