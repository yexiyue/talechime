//! Entry points must run without Python, model runtimes or audio initialization.
use std::process::Command;

#[test]
fn help_and_doctor_need_no_external_runtime() -> anyhow::Result<()> {
    for args in [vec!["perf", "--help"], vec!["perf", "doctor"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(args)
            .env("PATH", "")
            .output()?;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}
