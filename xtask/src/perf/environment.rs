//! Non-secret execution identity. Volatile temperatures belong to sample streams.
use super::prepare::capture;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, process::Command, time::Duration};
mod ort_identity;

#[derive(Debug, Serialize, Deserialize)]
pub struct Environment {
    pub machine: String,
    pub os: String,
    pub arch: String,
    pub hardware: String,
    pub driver: String,
    pub flags: BTreeMap<String, Option<String>>,
}

pub fn flags() -> anyhow::Result<BTreeMap<String, Option<String>>> {
    let mut flags = [
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
    .collect();
    ort_identity::record(&mut flags, std::env::var_os("ORT_DYLIB_PATH").as_deref())?;
    Ok(flags)
}

pub fn capture_environment(provider: &str) -> anyhow::Result<Environment> {
    #[cfg(not(target_os = "macos"))]
    let machine = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unverified_machine".into());
    #[cfg(target_os = "macos")]
    let machine = mac_value("/usr/sbin/sysctl", &["-n", "kern.hostname"])?;
    let cpu = cpu_name()?;
    #[cfg(not(target_os = "macos"))]
    let count =
        std::env::var("NUMBER_OF_PROCESSORS").unwrap_or_else(|_| "unverified_cpu_count".into());
    #[cfg(target_os = "macos")]
    let count = mac_value("/usr/sbin/sysctl", &["-n", "hw.logicalcpu"])?;
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
    #[cfg(target_os = "macos")]
    if provider == "metal" {
        hardware.push_str("; Metal=");
        hardware.push_str(&metal_identity()?);
        driver = os.clone();
    }
    Ok(Environment {
        machine,
        os,
        arch: std::env::consts::ARCH.into(),
        hardware,
        driver,
        flags: flags()?,
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

#[cfg(not(any(windows, target_os = "macos")))]
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

#[cfg(not(any(windows, target_os = "macos")))]
fn cpu_name() -> anyhow::Result<String> {
    Ok("unverified_native_cpu_collector_deferred".into())
}

#[cfg(target_os = "macos")]
fn mac_value(program: &str, arguments: &[&str]) -> anyhow::Result<String> {
    let bytes = capture(
        Command::new(program).args(arguments),
        Duration::from_secs(10),
    )?;
    let text = std::str::from_utf8(&bytes)?.trim();
    anyhow::ensure!(!text.is_empty(), "native identity unavailable: {program}");
    Ok(text.into())
}

#[cfg(target_os = "macos")]
fn cpu_name() -> anyhow::Result<String> {
    mac_value("/usr/sbin/sysctl", &["-n", "machdep.cpu.brand_string"])
}

#[cfg(target_os = "macos")]
fn os_version() -> anyhow::Result<String> {
    Ok(format!(
        "macos:{}:{}",
        mac_value("/usr/bin/sw_vers", &["-productVersion"])?,
        mac_value("/usr/sbin/sysctl", &["-n", "kern.osversion"])?
    ))
}

#[cfg(target_os = "macos")]
fn metal_identity() -> anyhow::Result<String> {
    let bytes = mac_value(
        "/usr/sbin/system_profiler",
        &["-json", "SPDisplaysDataType", "-detailLevel", "mini"],
    )?;
    parse_metal_identity(&bytes)
}

#[cfg(target_os = "macos")]
fn parse_metal_identity(bytes: &str) -> anyhow::Result<String> {
    let parsed: serde_json::Value = serde_json::from_str(bytes)?;
    let devices = parsed["SPDisplaysDataType"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("display identity missing"))?;
    let identities: Vec<_> = devices
        .iter()
        .filter_map(|device| {
            let support = device
                .get("spdisplays_mtlgpufamilysupport")
                .or_else(|| device.get("spdisplays_metal"))?
                .as_str()?;
            if support.contains("unsupported") {
                return None;
            }
            Some(serde_json::json!({
                "model":device["sppci_model"], "vendor":device["spdisplays_vendor"],
                "device_id":device["spdisplays_device-id"], "metal":support
            }))
        })
        .collect();
    anyhow::ensure!(
        !identities.is_empty(),
        "Metal hardware unavailable; no CPU fallback"
    );
    Ok(serde_json::to_string(&identities)?)
}

#[cfg(all(test, target_os = "macos"))]
mod mac_tests {
    #[test]
    fn metal_identity_handles_current_keys_and_excludes_display_serials() -> anyhow::Result<()> {
        let info = r#"{"SPDisplaysDataType":[{"sppci_model":"Apple M4 Pro","spdisplays_mtlgpufamilysupport":"spdisplays_metal4","spdisplays_ndrvs":[{"serial":"private"}]}]}"#;
        let result = super::parse_metal_identity(info)?;
        assert!(result.contains("Apple M4 Pro"));
        assert!(!result.contains("private"));
        assert!(super::parse_metal_identity(r#"{"SPDisplaysDataType":[]}"#).is_err());
        Ok(())
    }
    #[test]
    fn native_identity_is_stable_and_missing_command_fails() -> anyhow::Result<()> {
        let a = super::capture_environment("cpu")?;
        let b = super::capture_environment("cpu")?;
        assert_eq!(a.hardware, b.hardware);
        assert_eq!(a.os, b.os);
        assert!(!a.hardware.contains("unverified"));
        assert!(super::mac_value("/no/such/identity", &[]).is_err());
        Ok(())
    }
}
