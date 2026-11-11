use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=RUSTC");
    let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    if let Ok(output) = Command::new(compiler).arg("--version").output()
        && output.status.success()
    {
        println!(
            "cargo:rustc-env=TALECHIME_BUILD_RUSTC={}",
            String::from_utf8_lossy(&output.stdout).trim()
        );
    }
}
