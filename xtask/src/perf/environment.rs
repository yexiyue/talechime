//! Non-secret execution identity. Volatile temperatures belong to sample streams.
use super::prepare::capture;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, process::Command, time::Duration};

#[derive(Debug, Serialize, Deserialize)]
pub struct Environment {
    pub machine: String,
    pub os: String,
    pub arch: String,
    pub hardware: String,
    pub driver: String,
    pub flags: BTreeMap<String, Option<String>>,
}

pub fn flags() -> BTreeMap<String, Option<String>> {
    [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUBLAS_WORKSPACE_CONFIG",
        "NVIDIA_TF32_OVERRIDE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "QWEN_CP_GRAPH",
        "QWEN_DECODE_GRAPH",
        "VOXCPM_CUDA_GRAPH",
        "VOXCPM_DECODER_GRAPH",
        "VECLIB_MAXIMUM_THREADS",
    ]
    .into_iter()
    .map(|name| (name.into(), std::env::var(name).ok()))
    .collect()
}

pub fn capture_environment(provider: &str) -> anyhow::Result<Environment> {
    let machine = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unverified_machine".into());
    let cpu = cpu_name()?;
    let count =
        std::env::var("NUMBER_OF_PROCESSORS").unwrap_or_else(|_| "unverified_cpu_count".into());
    let os = os_version()?;
    let mut hardware = format!("{cpu}; logical_processors={count}");
    let mut driver = "not_applicable".to_owned();
    if provider == "cuda" {
        let mut command = Command::new("nvidia-smi");
        command.args([
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ]);
        let bytes = capture(&mut command, Duration::from_secs(2))?;
        let text = std::str::from_utf8(&bytes)?.trim();
        anyhow::ensure!(!text.is_empty(), "GPU identity unavailable");
        hardware.push_str("; GPUs=");
        hardware.push_str(text);
        driver = text
            .lines()
            .map(|line| line.rsplit(',').next().unwrap_or("").trim())
            .collect::<Vec<_>>()
            .join(",");
    }
    Ok(Environment {
        machine,
        os,
        arch: std::env::consts::ARCH.into(),
        hardware,
        driver,
        flags: flags(),
    })
}

#[cfg(windows)]
fn cpu_name() -> anyhow::Result<String> {
    registry_string(
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        "ProcessorNameString",
    )
}

#[cfg(windows)]
fn os_version() -> anyhow::Result<String> {
    let key = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
    Ok(format!(
        "windows:{}:{}:{}",
        registry_string(key, "ProductName")?,
        registry_string(key, "DisplayVersion")?,
        registry_string(key, "BuildLabEx")?
    ))
}

#[cfg(not(windows))]
fn os_version() -> anyhow::Result<String> {
    Ok(std::env::consts::OS.into())
}

#[cfg(windows)]
fn registry_string(key: &str, name: &str) -> anyhow::Result<String> {
    use windows::{
        Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ, RegGetValueW},
        core::PCWSTR,
    };
    let key: Vec<u16> = key.encode_utf16().chain(Some(0)).collect();
    let name: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
    let mut buffer = [0u16; 512];
    let mut bytes = std::mem::size_of_val(&buffer) as u32;
    // SAFETY: Read-only predefined hardware key; fixed typed UTF-16 output buffer.
    unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            PCWSTR(key.as_ptr()),
            PCWSTR(name.as_ptr()),
            RRF_RT_REG_SZ,
            None,
            Some(buffer.as_mut_ptr() as _),
            Some(&mut bytes),
        )
        .ok()?;
    }
    let length = buffer.iter().position(|c| *c == 0).unwrap_or(buffer.len());
    Ok(String::from_utf16(&buffer[..length])?.trim().to_owned())
}

#[cfg(not(windows))]
fn cpu_name() -> anyhow::Result<String> {
    Ok("unverified_native_cpu_collector_deferred".into())
}
