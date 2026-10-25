//! Model-free CLI regression; the selected probe must support its synthetic fixture.
#![cfg(any(windows, target_os = "macos"))]
use std::process::Command;
use xtask::perf::evidence::file_digest;

#[test]
#[ignore = "explicit fixture workflow; set TALECHIME_PERF_PROBE_BUILD"]
fn replacing_provider_in_place_refuses_resume_before_new_attempt() -> anyhow::Result<()> {
    let build = std::env::var_os("TALECHIME_PERF_PROBE_BUILD")
        .expect("explicit frozen fixture probe receipt required");
    let root = tempfile::tempdir()?;
    let main = root.path().join("onnxruntime.dll");
    let provider = root.path().join("onnxruntime_providers_cuda.dll");
    // These are identity fixtures, never loaded by the synthetic inference probe.
    std::fs::write(&main, b"synthetic-runtime")?;
    std::fs::write(&provider, b"provider-A")?;
    let command = || {
        let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
        command.env("ORT_DYLIB_PATH", &main);
        command
    };
    let input = root.path().join("input");
    let output = command()
        .args(["perf", "fixture-job", "--build"])
        .arg(build)
        .arg("--output")
        .arg(&input)
        .output()?;
    anyhow::ensure!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let run = root.path().join("run");
    let invoke = |mode: &str| {
        command()
            .args(["perf", mode, "--job"])
            .arg(input.join("job.json"))
            .arg("--output")
            .arg(&run)
            .output()
    };
    let output = invoke("run")?;
    anyhow::ensure!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let attempts =
        || -> std::io::Result<usize> { Ok(std::fs::read_dir(run.join("attempts"))?.count()) };
    let count = attempts()?;
    assert!(count > 0);
    let original = file_digest(&run.join("results.json"))?;
    std::fs::write(&provider, b"provider-B")?;
    let rejected = invoke("resume")?;
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("performance flags"));
    assert_eq!(attempts()?, count);
    assert_eq!(file_digest(&run.join("results.json"))?, original);
    std::fs::write(&provider, b"provider-A")?;
    let resumed = invoke("resume")?;
    anyhow::ensure!(
        resumed.status.success(),
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    assert_eq!(attempts()?, count);
    assert_eq!(file_digest(&run.join("results.json"))?, original);
    Ok(())
}
