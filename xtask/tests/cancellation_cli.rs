//! Explicit fixture workflow smoke, enabled after building the model-free probe.
#![cfg(any(windows, target_os = "macos"))]
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::{
    process::{Child, Command},
    time::{Duration, Instant},
};
#[cfg(windows)]
use windows::Win32::System::{
    Console::{CTRL_BREAK_EVENT, GenerateConsoleCtrlEvent},
    Threading::CREATE_NEW_PROCESS_GROUP,
};
use xtask::perf::{
    evidence::{publish_json, read_json},
    fixture_job,
    plan::Plan,
    records::Attempt,
};

struct ChildGuard(Child);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn cli_ctrl_break_retains_cancelled_attempt_and_stops_scheduling() -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    {
        cancellation(Some(libc::SIGINT))?;
        cancellation(Some(libc::SIGTERM))?;
    }
    #[cfg(windows)]
    cancellation(None)?;
    Ok(())
}

fn cancellation(signal: Option<i32>) -> anyhow::Result<()> {
    let Some(build) = std::env::var_os("TALECHIME_PERF_PROBE_BUILD") else {
        eprintln!(
            "explicit workflow smoke: set TALECHIME_PERF_PROBE_BUILD to a model-free probe receipt"
        );
        return Ok(());
    };
    let root = tempfile::tempdir()?;
    let input = root.path().join("input");
    fixture_job::create(std::path::Path::new(&build), &input)?;
    let mut plan: Plan = read_json(&input.join("plan.json"))?;
    plan.hot_rounds = 100;
    std::fs::remove_file(input.join("plan.json"))?;
    publish_json(&input.join("plan.json"), &plan)?;
    let output = root.path().join("run");
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    #[cfg(windows)]
    command.creation_flags(CREATE_NEW_PROCESS_GROUP.0);
    let mut child = ChildGuard(
        command
            .args(["perf", "run", "--job"])
            .arg(input.join("job.json"))
            .arg("--output")
            .arg(&output)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?,
    );
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        anyhow::ensure!(child.0.try_wait()?.is_none(), "CLI exited before signal");
        let prepared = output
            .join("attempts")
            .read_dir()
            .ok()
            .is_some_and(|entries| {
                entries.flatten().any(|entry| {
                    std::fs::read_to_string(entry.path().join("stdout.jsonl"))
                        .is_ok_and(|text| text.contains("\"prepared\""))
                })
            });
        if prepared {
            break;
        }
        anyhow::ensure!(
            Instant::now() < deadline,
            "probe did not prepare before signal deadline"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    // SAFETY: Target only the new CLI process group created above, never group zero.
    #[cfg(windows)]
    unsafe {
        GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, child.0.id())?;
    }
    #[cfg(windows)]
    let _ = signal;
    #[cfg(target_os = "macos")]
    unsafe {
        anyhow::ensure!(
            libc::kill(child.0.id() as i32, signal.unwrap()) == 0,
            "signal CLI failed"
        );
    }
    let status = loop {
        if let Some(status) = child.0.try_wait()? {
            break status;
        }
        anyhow::ensure!(
            Instant::now() < deadline,
            "CLI failed to clean up after cancellation"
        );
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(!status.success());
    let attempts: Vec<Attempt> = read_json(&output.join("results.json"))?;
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].outcome, "cancelled");
    assert!(attempts[0].cleanup_verified);
    assert!(
        !output
            .join("attempts")
            .join(&attempts[0].attempt_id)
            .join("committed.json")
            .exists()
    );
    Ok(())
}
