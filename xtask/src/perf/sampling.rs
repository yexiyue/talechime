//! Platform readings keep scope, units and unavailable reasons explicit.
use super::{evidence::Reading, prepare::capture};
use std::{collections::BTreeMap, process::Command, time::Duration};

fn missing(scope: &str, source: &str, reason: String, interval: u64) -> Reading {
    Reading {
        value: None,
        unit: "bytes".into(),
        scope: scope.into(),
        source: source.into(),
        interval_ms: Some(interval),
        status: "unavailable".into(),
        reason: Some(reason),
    }
}

#[cfg(windows)]
pub fn process(pid: u32, interval_ms: u64) -> BTreeMap<String, Reading> {
    use std::mem::size_of;
    use windows::Win32::{
        Foundation::CloseHandle,
        System::{
            ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX},
            Threading::{OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ},
        },
    };
    let read = || -> anyhow::Result<PROCESS_MEMORY_COUNTERS_EX> {
        // SAFETY: Query-only handle, typed output buffer, and owned handle closed on all paths.
        unsafe {
            let handle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, false, pid)?;
            let mut counters = PROCESS_MEMORY_COUNTERS_EX {
                cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
                ..Default::default()
            };
            let result =
                K32GetProcessMemoryInfo(handle, &mut counters as *mut _ as _, counters.cb).ok();
            let _ = CloseHandle(handle);
            result?;
            Ok(counters)
        }
    };
    match read() {
        Ok(counters) => [
            ("rss_bytes", counters.WorkingSetSize),
            ("os_peak_working_set_bytes", counters.PeakWorkingSetSize),
            ("private_bytes", counters.PrivateUsage),
        ]
        .into_iter()
        .map(|(name, value)| {
            (
                name.into(),
                Reading {
                    value: Some(value as f64),
                    unit: "bytes".into(),
                    scope: "process".into(),
                    source: "Windows K32GetProcessMemoryInfo".into(),
                    interval_ms: Some(interval_ms),
                    status: "available".into(),
                    reason: None,
                },
            )
        })
        .collect(),
        Err(error) => ["rss_bytes", "os_peak_working_set_bytes", "private_bytes"]
            .into_iter()
            .map(|name| {
                (
                    name.into(),
                    missing(
                        "process",
                        "Windows K32GetProcessMemoryInfo",
                        error.to_string(),
                        interval_ms,
                    ),
                )
            })
            .collect(),
    }
}

#[cfg(not(any(windows, target_os = "macos")))]
pub fn process(_pid: u32, interval_ms: u64) -> BTreeMap<String, Reading> {
    let mut reading = missing(
        "process",
        "native collector",
        "platform collector is deferred".into(),
        interval_ms,
    );
    reading.status = "deferred".into();
    BTreeMap::from([("rss_bytes".into(), reading)])
}

pub fn parse_gpu(text: &str, interval_ms: u64) -> anyhow::Result<BTreeMap<String, Reading>> {
    let mut result = BTreeMap::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        anyhow::ensure!(
            fields.len() == 5 && fields[0].starts_with("GPU-"),
            "unexpected nvidia-smi CSV schema"
        );
        for (index, name, unit, multiplier) in [
            (1, "device_vram_bytes", "bytes", 1024.0 * 1024.0),
            (2, "sm_clock_mhz", "MHz", 1.0),
            (3, "power_watts", "watts", 1.0),
            (4, "temperature_celsius", "celsius", 1.0),
        ] {
            let value = fields[index]
                .parse::<f64>()
                .ok()
                .filter(|n| n.is_finite() && *n >= 0.0)
                .map(|n| n * multiplier);
            let reading = Reading {
                value,
                unit: unit.into(),
                scope: "whole_device".into(),
                source: format!("nvidia-smi:{}", fields[0]),
                interval_ms: Some(interval_ms),
                status: if value.is_some() {
                    "available"
                } else {
                    "unavailable"
                }
                .into(),
                reason: value
                    .is_none()
                    .then(|| format!("unsupported or missing field: {}", fields[index])),
            };
            reading.validate()?;
            result.insert(format!("{}:{name}", fields[0]), reading);
        }
    }
    anyhow::ensure!(!result.is_empty(), "empty nvidia-smi output");
    Ok(result)
}

pub fn gpu(interval_ms: u64) -> BTreeMap<String, Reading> {
    let mut command = Command::new("nvidia-smi");
    command.args([
        "--query-gpu=uuid,memory.used,clocks.sm,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]);
    let result = capture(&mut command, Duration::from_secs(2))
        .and_then(|bytes| parse_gpu(std::str::from_utf8(&bytes)?, interval_ms));
    result.unwrap_or_else(|error| {
        BTreeMap::from([(
            "device_vram_bytes".into(),
            missing("whole_device", "nvidia-smi", error.to_string(), interval_ms),
        )])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_readings_keep_units_scope_and_missing_fields() -> anyhow::Result<()> {
        let result = parse_gpu("GPU-fixture, 100, 2100, [N/A], 42", 500)?;
        assert_eq!(
            result["GPU-fixture:device_vram_bytes"].value,
            Some(100.0 * 1024.0 * 1024.0)
        );
        assert_eq!(
            result["GPU-fixture:device_vram_bytes"].scope,
            "whole_device"
        );
        assert!(result["GPU-fixture:power_watts"].value.is_none());
        assert!(parse_gpu("", 500).is_err());
        assert!(parse_gpu("wrong format", 500).is_err());
        Ok(())
    }
    #[cfg(windows)]
    #[test]
    fn current_process_has_native_rss_and_os_high_water() {
        let readings = process(std::process::id(), 500);
        assert!(readings["rss_bytes"].value.is_some_and(|n| n > 0.0));
        assert!(readings["os_peak_working_set_bytes"].value >= readings["rss_bytes"].value);
        assert!(process(u32::MAX, 500)["rss_bytes"].value.is_none());
    }
}

/// Mac RSS and footprint are different OS measures, never exclusive GPU VRAM.
#[cfg(target_os = "macos")]
pub fn process(pid: u32, interval_ms: u64) -> BTreeMap<String, Reading> {
    // SAFETY: Zeroed output buffer matches RUSAGE_INFO_V4 and is correctly sized.
    let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    let result = unsafe {
        libc::proc_pid_rusage(
            pid as i32,
            libc::RUSAGE_INFO_V4,
            &mut info as *mut _ as *mut libc::rusage_info_t,
        )
    };
    let error = (result != 0).then(|| std::io::Error::last_os_error().to_string());
    let mut readings: BTreeMap<_, _> = [
        ("rss_bytes", info.ri_resident_size),
        ("physical_footprint_bytes", info.ri_phys_footprint),
        (
            "os_peak_physical_footprint_bytes",
            info.ri_lifetime_max_phys_footprint,
        ),
    ]
    .into_iter()
    .map(|(name, value)| {
        let reading = if let Some(error) = &error {
            missing(
                "process",
                "macOS proc_pid_rusage RUSAGE_INFO_V4",
                error.clone(),
                interval_ms,
            )
        } else {
            Reading {
                value: Some(value as f64),
                unit: "bytes".into(),
                scope: "process".into(),
                source: "macOS proc_pid_rusage RUSAGE_INFO_V4".into(),
                interval_ms: Some(interval_ms),
                status: "available".into(),
                reason: None,
            }
        };
        (name.into(), reading)
    })
    .collect();
    readings.insert(
        "os_peak_rss_bytes".into(),
        missing(
            "process",
            "macOS proc_pid_rusage",
            "RUSAGE_INFO_V4 exposes footprint peak, not RSS high-water mark".into(),
            interval_ms,
        ),
    );
    readings.insert(
        "metal_allocation_bytes".into(),
        missing(
            "process_device_allocations",
            "Metal native collector",
            "not implemented; RSS and unified memory are not exclusive VRAM".into(),
            interval_ms,
        ),
    );
    readings
}

#[cfg(all(test, target_os = "macos"))]
mod mac_tests {
    #[test]
    fn native_memory_has_units_and_unavailable_is_not_zero() {
        let mut allocation = vec![0u8; 16 * 1024 * 1024];
        for byte in allocation.iter_mut().step_by(4096) {
            *byte = 1;
        }
        std::hint::black_box(&allocation);
        let readings = super::process(std::process::id(), 100);
        assert!(readings["rss_bytes"].value.unwrap() > 0.0);
        assert_eq!(readings["rss_bytes"].unit, "bytes");
        assert_eq!(readings["rss_bytes"].scope, "process");
        assert!(
            readings["os_peak_physical_footprint_bytes"].value.unwrap()
                >= readings["physical_footprint_bytes"].value.unwrap()
        );
        assert!(readings["os_peak_rss_bytes"].value.is_none());
        assert!(readings["metal_allocation_bytes"].value.is_none());
        let missing = super::process(i32::MAX as u32, 100);
        assert!(missing["rss_bytes"].value.is_none());
        assert!(missing["rss_bytes"].reason.is_some());
    }
}
