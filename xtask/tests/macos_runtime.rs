//! Native scheduling must stop when containment evidence fails.
#![cfg(target_os = "macos")]
use std::{os::unix::fs::PermissionsExt, path::Path, process::Command};
use xtask::perf::{
    build::Build,
    evidence::{Source, VERSION, file_digest, publish_json, read_json},
    fixture_job,
    runner::Job,
};

fn fake_build(root: &Path, script: &str) -> anyhow::Result<std::path::PathBuf> {
    let executable = root.join("probe.sh");
    std::fs::write(&executable, script)?;
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700))?;
    let build = Build {
        schema: VERSION.into(),
        source: Source {
            revision: "fixture".into(),
            dirty_sha256: String::new(),
            untracked: vec![],
        },
        probe: "performance_probe".into(),
        features: vec!["performance-tools".into()],
        binary_sha256: file_digest(&executable)?,
        executable,
        target: "aarch64-apple-darwin".into(),
        profile: "release".into(),
    };
    let path = root.join("build.json");
    publish_json(&path, &build)?;
    Ok(path)
}

#[test]
fn escaped_probe_stops_batch_without_success_commit() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let xtask = env!("CARGO_BIN_EXE_xtask").replace('\'', "'\\''");
    let build = fake_build(
        root.path(),
        &format!("#!/bin/sh\nexec '{xtask}' __perf-fixture escape-tree --marker \"$2/tree\"\n"),
    )?;
    let input = root.path().join("input");
    fixture_job::create(&build, &input)?;
    let mut job: Job = read_json(&input.join("job.json"))?;
    job.timeout_seconds = 3;
    std::fs::remove_file(input.join("job.json"))?;
    publish_json(&input.join("job.json"), &job)?;
    let output = root.path().join("run");
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["perf", "run", "--job"])
        .arg(input.join("job.json"))
        .arg("--output")
        .arg(&output)
        .output()?;
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("cleanup unverified"));
    let attempts: Vec<_> = std::fs::read_dir(output.join("attempts"))?.collect::<Result<_, _>>()?;
    assert_eq!(attempts.len(), 1);
    assert!(!attempts[0].path().join("committed.json").exists());
    let capture: serde_json::Value = read_json(&attempts[0].path().join("capture.json"))?;
    assert_eq!(capture["cleanup_verified"], false);
    for missing_capture in [false, true] {
        if missing_capture {
            std::fs::remove_file(attempts[0].path().join("capture.json"))?;
        }
        let resumed = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["perf", "resume", "--job"])
            .arg(input.join("job.json"))
            .arg("--output")
            .arg(&output)
            .output()?;
        assert!(!resumed.status.success());
        assert!(String::from_utf8_lossy(&resumed.stderr).contains("cleanup unverified"));
        assert_eq!(std::fs::read_dir(output.join("attempts"))?.count(), 1);
    }
    Ok(())
}

#[test]
fn cancellation_after_terminal_keeps_last_attempt_uncommitted_and_retryable() -> anyhow::Result<()>
{
    use std::time::{Duration, Instant};
    use xtask::perf::{plan::Plan, records::Attempt};
    let root = tempfile::tempdir()?;
    // Signal after terminal output but before probe exit: the last process must not commit.
    let script = r#"#!/bin/sh
id=${2##*/}
printf '%s' "$RAYON_NUM_THREADS" > "$2/threads"
printf '{"schema":"talechime/performance/1","attempt_id":"%s","sequence":0,"offset_seconds":0.0,"kind":"terminal","payload":{"outcome":"success"}}\n' "$id"
exec /bin/sleep 30
"#;
    let build = fake_build(root.path(), script)?;
    let input = root.path().join("input");
    fixture_job::create(&build, &input)?;
    let mut plan: Plan = read_json(&input.join("plan.json"))?;
    plan.processes = 1;
    plan.hot_rounds = 1;
    plan.warmup_rounds = 0;
    std::fs::remove_file(input.join("plan.json"))?;
    publish_json(&input.join("plan.json"), &plan)?;
    let output = root.path().join("run");
    for resume in [false, true] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["perf", if resume { "resume" } else { "run" }, "--job"])
            .arg(input.join("job.json"))
            .env("RAYON_NUM_THREADS", "99")
            .arg("--output")
            .arg(&output)
            .spawn()?;
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let count = output
                .join("attempts")
                .read_dir()
                .ok()
                .map(|entries| {
                    entries
                        .flatten()
                        .filter(|e| {
                            std::fs::read_to_string(e.path().join("stdout.jsonl"))
                                .is_ok_and(|text| text.contains("terminal"))
                        })
                        .count()
                })
                .unwrap_or(0);
            if count == if resume { 2 } else { 1 } {
                break;
            }
            if Instant::now() >= deadline || child.try_wait()?.is_some() {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!("terminal not observed before signal deadline");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        // SAFETY: Signal only this test-owned runner PID.
        assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGTERM) }, 0);
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                anyhow::bail!("cancel cleanup timeout");
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(!status.success());
    }
    let results: Vec<Attempt> = read_json(&output.join("results.json"))?;
    assert_eq!(results[0].outcome, "cancelled");
    assert!(results[0].cleanup_verified);
    let attempts: Vec<_> = std::fs::read_dir(output.join("attempts"))?.collect::<Result<_, _>>()?;
    assert_eq!(attempts.len(), 2);
    for entry in &attempts {
        assert_eq!(std::fs::read_to_string(entry.path().join("threads"))?, "1");
    }
    assert!(
        attempts
            .iter()
            .all(|entry| !entry.path().join("committed.json").exists())
    );
    Ok(())
}
