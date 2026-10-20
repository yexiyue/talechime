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
