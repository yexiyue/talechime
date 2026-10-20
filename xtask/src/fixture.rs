//! Hidden subprocess fixtures for ownership tests; no models or network.
use std::{
    io::Write,
    path::Path,
    process::{Command, Stdio},
    time::Duration,
};

pub fn run(mode: &str, marker: Option<&Path>) -> anyhow::Result<()> {
    if let Some(marker) = marker {
        std::fs::write(marker, "started")?;
    }
    match mode {
        "events" => {
            for (sequence, kind) in [(0, "prepared"), (1, "terminal")] {
                println!(
                    "{}",
                    serde_json::json!({"schema": "talechime/performance/1", "attempt_id": "fixture", "sequence": sequence, "offset_seconds": sequence as f64, "kind": kind, "payload": null})
                );
            }
        }
        "idle" => std::thread::sleep(Duration::from_secs(30)),
        "tree" => {
            let marker = marker.ok_or_else(|| anyhow::anyhow!("tree fixture requires marker"))?;
            let mut child = Command::new(std::env::current_exe()?)
                .args(["__perf-fixture", "idle", "--marker"])
                .arg(marker.with_extension("grandchild"))
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()?;
            child.wait()?;
        }
        #[cfg(target_os = "macos")]
        "ignore-term" => {
            // SAFETY: Isolated fixture process intentionally ignores only SIGTERM.
            unsafe {
                libc::signal(libc::SIGTERM, libc::SIG_IGN);
            }
            if let Some(marker) = marker {
                std::fs::write(marker.with_extension("ready"), "ready")?;
            }
            std::thread::sleep(Duration::from_secs(30));
        }
        #[cfg(target_os = "macos")]
        "escape" => {
            std::thread::sleep(Duration::from_millis(300));
            // SAFETY: This fixture is not a group leader; tests own and reap it.
            anyhow::ensure!(unsafe { libc::setsid() } > 0, "fixture setsid failed");
            if let Some(marker) = marker {
                std::fs::write(marker.with_extension("escaped"), "escaped")?;
            }
            std::thread::sleep(Duration::from_secs(30));
        }
        #[cfg(target_os = "macos")]
        "orphan-pipe" | "escape-tree" => {
            let marker = marker.ok_or_else(|| anyhow::anyhow!("marker required"))?;
            let mut child = Command::new(std::env::current_exe()?)
                .args([
                    "__perf-fixture",
                    if mode == "escape-tree" {
                        "escape"
                    } else {
                        "ignore-term"
                    },
                    "--marker",
                ])
                .arg(marker.with_extension("descendant"))
                .spawn()?;
            std::fs::write(marker.with_extension("pid"), child.id().to_string())?;
            if mode == "escape-tree" {
                child.wait()?;
            }
            // orphan-pipe intentionally leaves a descendant holding inherited pipes.
        }
        "flood" => {
            for _ in 0..8192 {
                std::io::stdout().write_all(&[b'x'; 4096])?;
                std::io::stderr().write_all(&[b'y'; 4096])?;
            }
        }
        _ => anyhow::bail!("unknown process fixture"),
    }
    Ok(())
}
