use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/codec_layers/ordered_deconv.cu");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_NVCC");
    let target = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if env::var_os("CARGO_FEATURE_CUDA").is_none()
        || !matches!(target.as_str(), "windows" | "linux")
    {
        return;
    }
    let compiler = env::var_os("CUDA_NVCC")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("CUDA_PATH")
                .or_else(|| env::var_os("CUDA_HOME"))
                .map(|root| {
                    PathBuf::from(root).join("bin").join(if cfg!(windows) {
                        "nvcc.exe"
                    } else {
                        "nvcc"
                    })
                })
        })
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("ordered_deconv.ptx");
    let result = Command::new(&compiler)
        .args([
            "--ptx",
            "-O3",
            "--std=c++17",
            "--gpu-architecture=compute_80",
            "--fmad=true",
            "--ftz=false",
        ])
        .arg("src/codec_layers/ordered_deconv.cu")
        .arg("-o")
        .arg(output)
        .output()
        .unwrap_or_else(|e| panic!("cannot execute CUDA compiler {}: {e}", compiler.display()));
    assert!(
        result.status.success(),
        "ordered decoder CUDA compilation failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}
