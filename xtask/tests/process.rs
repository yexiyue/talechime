//! Cleanup must include descendants and leave other jobs alone.
#![cfg(any(windows, target_os = "macos"))]
use std::sync::atomic::AtomicBool;
use std::{
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use xtask::perf::executor::{Budget, execute};
use xtask::perf::process::Process;

#[test]
fn cleanup_contains_grandchildren_and_preserves_unrelated_processes() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let marker = root.path().join("parent");
    let mut unrelated = Process::spawn(
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["__perf-fixture", "idle"])
            .stdout(Stdio::null())
            .stderr(Stdio::null()),
    )?;
    let mut tree = Process::spawn(
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["__perf-fixture", "tree", "--marker"])
            .arg(&marker)
            .stdout(Stdio::null())
            .stderr(Stdio::null()),
    )?;
    let deadline = Instant::now() + Duration::from_secs(5);
    while !marker.with_extension("grandchild").exists() {
        anyhow::ensure!(Instant::now() < deadline, "grandchild did not start");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(tree.active_processes()? >= 2);
    tree.cleanup()?;
    assert_eq!(tree.active_processes()?, 0);
    assert!(unrelated.try_wait()?.is_none());
    unrelated.cleanup()?;
    Ok(())
}

#[test]
fn bounded_capture_handles_timeout_flood_and_next_success_serially() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let cancel = AtomicBool::new(false);
    let timeout = execute(
        Command::new(env!("CARGO_BIN_EXE_xtask")).args(["__perf-fixture", "idle"]),
        &root.path().join("timeout"),
        "fixture",
        Duration::from_millis(100),
        Budget::default(),
        &cancel,
    )?;
    assert_eq!(timeout.outcome, "timeout");
    assert!(timeout.cleanup_verified);
    cancel.store(true, std::sync::atomic::Ordering::Release);
    let cancelled = execute(
        Command::new(env!("CARGO_BIN_EXE_xtask")).args(["__perf-fixture", "idle"]),
        &root.path().join("cancelled"),
        "fixture",
        Duration::from_secs(5),
        Budget::default(),
        &cancel,
    )?;
    assert_eq!(cancelled.outcome, "cancelled");
    assert!(cancelled.cleanup_verified);
    cancel.store(false, std::sync::atomic::Ordering::Release);
    let flood = execute(
        Command::new(env!("CARGO_BIN_EXE_xtask")).args(["__perf-fixture", "flood"]),
        &root.path().join("flood"),
        "fixture",
        Duration::from_secs(5),
        Budget {
            frame_bytes: 128,
            log_bytes: 1024,
            event_count: 10,
            ..Budget::default()
        },
        &cancel,
    )?;
    assert_eq!(flood.outcome, "failure");
    assert!(flood.cleanup_verified);
    let success = execute(
        Command::new(env!("CARGO_BIN_EXE_xtask")).args(["__perf-fixture", "events"]),
        &root.path().join("success"),
        "fixture",
        Duration::from_secs(5),
        Budget::default(),
        &cancel,
    )?;
    assert_eq!(success.outcome, "success");
    assert!(success.errors.is_empty());
    assert!(root.path().join("timeout/capture.json").is_file());
    assert!(root.path().join("flood/stderr.log").is_file());
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn mac_spawn_failure_and_orphaned_pipe_cleanup_are_bounded() -> anyhow::Result<()> {
    assert!(Process::spawn(&mut Command::new("/no/such/performance-probe")).is_err());
    let root = tempfile::tempdir()?;
    let started = Instant::now();
    let capture = execute(
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["__perf-fixture", "orphan-pipe", "--marker"])
            .arg(root.path().join("parent")),
        &root.path().join("capture"),
        "fixture",
        Duration::from_secs(3),
        Budget::default(),
        &AtomicBool::new(false),
    )?;
    assert_eq!(capture.outcome, "failure");
    assert!(capture.cleanup_verified, "{capture:?}");
    assert!(started.elapsed() < Duration::from_secs(6));
    Ok(())
}

#[cfg(target_os = "macos")]
#[test]
fn observed_group_escape_fails_closed_and_retains_capture() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let capture = execute(
        Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["__perf-fixture", "escape-tree", "--marker"])
            .arg(root.path().join("parent")),
        &root.path().join("capture"),
        "fixture",
        Duration::from_secs(3),
        Budget::default(),
        &AtomicBool::new(false),
    )?;
    assert!(
        root.path().join("parent.escaped").is_file(),
        "{capture:?}; {:?}",
        std::fs::read_to_string(root.path().join("capture/stderr.log"))
    );
    assert!(!capture.cleanup_verified);
    assert_eq!(capture.outcome, "failure");
    assert!(capture.errors.iter().any(|error| error.contains("escaped")));
    assert!(root.path().join("capture/capture.json").is_file());
    Ok(())
}
