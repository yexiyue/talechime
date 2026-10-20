//! Bounded streaming event capture. Failed attempts retain their raw logs.
use super::records::{Event, EventSequence};
use anyhow::ensure;
use serde::{Deserialize, Serialize};
use std::{
    io::{BufRead, BufReader, Read, Write},
    path::Path,
    process::Command,
    sync::atomic::AtomicBool,
    time::Duration,
};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Budget {
    pub frame_bytes: usize,
    pub log_bytes: u64,
    pub event_count: u64,
    pub sample_interval_ms: Option<u64>,
    pub device_sampling: bool,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            frame_bytes: 1024 * 1024,
            log_bytes: 64 * 1024 * 1024,
            event_count: 1_000_000,
            sample_interval_ms: Some(500),
            device_sampling: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Capture {
    pub outcome: String,
    pub exit_code: Option<i32>,
    pub cleanup_verified: bool,
    pub errors: Vec<String>,
}

pub fn events(
    reader: impl Read,
    mut log: impl Write,
    attempt_id: &str,
    budget: Budget,
) -> anyhow::Result<()> {
    ensure!(
        budget.frame_bytes > 0 && budget.log_bytes > 0 && budget.event_count > 0,
        "zero event budget"
    );
    let mut reader = BufReader::new(reader);
    let mut sequence = EventSequence::default();
    let mut total = 0u64;
    let mut count = 0u64;
    loop {
        let mut line = Vec::new();
        let n = reader
            .by_ref()
            .take(budget.frame_bytes as u64 + 1)
            .read_until(b'\n', &mut line)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        count += 1;
        ensure!(
            n <= budget.frame_bytes && total <= budget.log_bytes && count <= budget.event_count,
            "event output budget exceeded"
        );
        log.write_all(&line)?;
        ensure!(line.last() == Some(&b'\n'), "truncated event frame");
        let event: Event = serde_json::from_slice(&line)?;
        sequence.accept(attempt_id, &event)?;
    }
    log.flush()?;
    sequence.finish()
}

pub fn diagnostic(mut reader: impl Read, mut log: impl Write, limit: u64) -> anyhow::Result<()> {
    let mut bytes = [0; 8192];
    let mut total = 0;
    loop {
        let n = reader.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        ensure!(total <= limit, "stderr budget exceeded");
        log.write_all(&bytes[..n])?;
    }
    log.flush()?;
    Ok(())
}

#[cfg(any(windows, target_os = "macos"))]
pub fn execute(
    command: &mut Command,
    directory: &Path,
    attempt_id: &str,
    timeout: Duration,
    budget: Budget,
    cancel: &AtomicBool,
) -> anyhow::Result<Capture> {
    use super::process::Process;
    use anyhow::Context;
    use std::{fs::OpenOptions, process::Stdio, sync::atomic::Ordering, time::Instant};
    ensure!(timeout > Duration::ZERO, "zero process timeout");
    ensure!(
        budget
            .sample_interval_ms
            .is_none_or(|interval| (10..=60_000).contains(&interval)),
        "sampling interval must be between 10 and 60000 ms"
    );
    std::fs::create_dir_all(directory)?;
    let stdout = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("stdout.jsonl"))?;
    let stderr = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("stderr.log"))?;
    let mut process = Process::spawn(
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped()),
    )?;
    let outgoing = process.take_stdout().context("probe stdout missing")?;
    let errors = process.take_stderr().context("probe stderr missing")?;
    let fault = AtomicBool::new(false);
    let sample_stop = AtomicBool::new(false);
    let reader_abort = AtomicBool::new(false);
    #[cfg(target_os = "macos")]
    let outgoing = super::process::InterruptiblePipe::new(outgoing, &reader_abort)?;
    #[cfg(target_os = "macos")]
    let errors = super::process::InterruptiblePipe::new(errors, &reader_abort)?;
    let start = Instant::now();
    let result = std::thread::scope(|scope| -> anyhow::Result<Capture> {
        // Own the guard inside the scope so error unwinding kills descendants
        // before scope joins readers that may still be waiting for pipe EOF.
        let mut process = process;
        struct Stop<'a>(&'a AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let _stop = Stop(&sample_stop);
        let pid = process.id();
        let sample_log = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(directory.join("resource-samples.jsonl"))?;
        let sample_fault = &fault;
        let stopped = &sample_stop;
        let sampler = scope.spawn(move || -> anyhow::Result<()> {
            let mut log = sample_log;
            let mut bytes = 0usize;
            if let Some(interval) = budget.sample_interval_ms {
                while !stopped.load(Ordering::Acquire) {
                    let mut readings = super::sampling::process(pid, interval);
                    if budget.device_sampling {
                        readings.extend(super::sampling::gpu(interval));
                    }
                    let mut line = serde_json::to_vec(&serde_json::json!({"offset_seconds":start.elapsed().as_secs_f64(),"clock":"parent_spawn_complete","readings":readings}))?;
                    line.push(b'\n');
                    bytes += line.len();
                    if bytes as u64 > budget.log_bytes {
                        sample_fault.store(true, Ordering::Release);
                        anyhow::bail!("resource sample budget exceeded");
                    }
                    if let Err(error) = log.write_all(&line) {
                        sample_fault.store(true, Ordering::Release);
                        return Err(error.into());
                    }
                    let next = Instant::now() + Duration::from_millis(interval);
                    while Instant::now() < next && !stopped.load(Ordering::Acquire) {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
            }
            log.sync_all()?;
            Ok(())
        });
        let stdout = scope.spawn(|| {
            let result = events(outgoing, stdout, attempt_id, budget);
            if result.is_err() {
                fault.store(true, Ordering::Release);
            }
            result
        });
        let stderr = scope.spawn(|| {
            let result = diagnostic(errors, stderr, budget.log_bytes);
            if result.is_err() {
                fault.store(true, Ordering::Release);
            }
            result
        });
        let mut outcome = "success";
        let mut exit_code = None;
        let mut lifecycle_errors = Vec::new();
        loop {
            if cancel.load(Ordering::Acquire) {
                outcome = "cancelled";
                break;
            }
            if fault.load(Ordering::Acquire) {
                outcome = "failure";
                break;
            }
            if start.elapsed() >= timeout {
                outcome = "timeout";
                break;
            }
            let status = match process.try_wait() {
                Ok(status) => status,
                Err(error) => {
                    lifecycle_errors.push(error.to_string());
                    outcome = "failure";
                    break;
                }
            };
            if let Some(status) = status {
                exit_code = status.code();
                if !status.success() {
                    outcome = "failure";
                }
                match process.active_processes() {
                    Ok(0) => {}
                    Ok(_) => outcome = "failure",
                    Err(error) => {
                        lifecycle_errors.push(error.to_string());
                        outcome = "failure";
                    }
                }
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        // No subsequent execution may proceed unless this succeeds.
        let cleanup = process.cleanup();
        if cleanup.is_err() {
            reader_abort.store(true, Ordering::Release);
        }
        sample_stop.store(true, Ordering::Release);
        let mut failures = lifecycle_errors;
        for result in [
            stdout
                .join()
                .map_err(|_| anyhow::anyhow!("stdout reader panicked"))?,
            stderr
                .join()
                .map_err(|_| anyhow::anyhow!("stderr reader panicked"))?,
            sampler
                .join()
                .map_err(|_| anyhow::anyhow!("resource sampler panicked"))?,
        ] {
            if let Err(error) = result {
                failures.push(error.to_string());
            }
        }
        if let Err(error) = &cleanup {
            failures.push(format!("cleanup unverified: {error:#}"));
            outcome = "failure";
        }
        if !failures.is_empty() && outcome == "success" {
            outcome = "failure";
        }
        Ok(Capture {
            outcome: outcome.into(),
            exit_code,
            cleanup_verified: cleanup.is_ok(),
            errors: failures,
        })
    });
    if let Ok(capture) = &result {
        super::evidence::publish_json(&directory.join("capture.json"), capture)?;
    }
    result
}

#[cfg(not(any(windows, target_os = "macos")))]
pub fn execute(
    _command: &mut Command,
    _directory: &Path,
    _attempt_id: &str,
    _timeout: Duration,
    _budget: Budget,
    _cancel: &AtomicBool,
) -> anyhow::Result<Capture> {
    anyhow::bail!("deferred: native process lifecycle requires platform implementation")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn oversized_and_missing_terminal_frames_fail_without_unbounded_allocation() {
        let budget = Budget {
            frame_bytes: 100,
            ..Budget::default()
        };
        let mut log = Vec::new();
        assert!(events(&[b'x'; 1024][..], &mut log, "attempt", budget).is_err());
        assert!(log.is_empty());
        assert!(events(&b""[..], &mut log, "attempt", budget).is_err());
        assert!(diagnostic(&[0; 100][..], Vec::new(), 10).is_err());
    }
}
