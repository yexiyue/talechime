//! Env-gated stage profiling for development runs
//! (`INDEXTTS_PROFILE=1`). Disabled by default; enabled runs synchronize the device at scope
//! boundaries so CUDA stages are attributed to the stage that submitted
//! them instead of the next blocking read.
//!
//! Reports accumulate on the calling thread and print to stderr as one
//! JSON object per [`crate::Model::generate`] call:
//! `{"profile":{"cfm.estimator":[secs,calls],...}}` (seconds, call count).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use candle_core::Device;

static ENABLED: OnceLock<bool> = OnceLock::new();

pub(crate) fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var_os("INDEXTTS_PROFILE").is_some_and(|v| {
            !v.is_empty() && v != "0" && v.to_string_lossy().to_lowercase() != "false"
        })
    })
}

thread_local! {
    static TIMINGS: RefCell<BTreeMap<&'static str, (Duration, usize)>> =
        const { RefCell::new(BTreeMap::new()) };
}

fn add(name: &'static str, secs: Duration) {
    TIMINGS.with(|t| {
        let mut map = t.borrow_mut();
        let entry = map.entry(name).or_insert((Duration::ZERO, 0));
        entry.0 += secs;
        entry.1 += 1;
    });
}

fn take_report() -> Option<serde_json::Value> {
    TIMINGS.with(|t| {
        let map = std::mem::take(&mut *t.borrow_mut());
        (!map.is_empty()).then(|| {
            let stages: BTreeMap<_, _> = map
                .into_iter()
                .map(|(name, (dur, calls))| (name, (dur.as_secs_f64(), calls)))
                .collect();
            serde_json::json!({ "profile": stages })
        })
    })
}

/// Per-call lifetime: clear stale timings on entry and drain on every exit,
/// including an inference error. The owner must precede its timed scopes.
pub(crate) struct Report;

pub(crate) fn report() -> Option<Report> {
    enabled().then(Report::new)
}

impl Report {
    fn new() -> Self {
        TIMINGS.with(|t| t.borrow_mut().clear());
        Self
    }
}

impl Drop for Report {
    fn drop(&mut self) {
        if let Some(report) = take_report() {
            eprintln!("{report}");
        }
    }
}

/// Timed scope; synchronizes `device` (when given) on enter and exit so
/// GPU work lands in the stage that issued it. Without a device, this is
/// host wall time: calls may still block or contain nested synchronized
/// scopes. It is not a measurement of CPU utilization or pure submission.
pub(crate) struct Scope {
    name: &'static str,
    device: Option<Device>,
    start: Instant,
}

pub(crate) fn scope(device: Option<&Device>, name: &'static str) -> Option<Scope> {
    if !enabled() {
        return None;
    }
    if let Some(dev) = device {
        let _ = dev.synchronize();
    }
    Some(Scope {
        name,
        device: device.cloned(),
        start: Instant::now(),
    })
}

impl Drop for Scope {
    fn drop(&mut self) {
        if let Some(dev) = &self.device {
            let _ = dev.synchronize();
        }
        let elapsed = self.start.elapsed();
        add(self.name, elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_is_parseable_and_preserves_counts() {
        let _report = Report::new();
        add("cfm.estimator", Duration::from_millis(125));
        add("cfm.estimator", Duration::from_millis(125));
        let json = take_report().unwrap().to_string();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed["profile"]["cfm.estimator"],
            serde_json::json!([0.25, 2])
        );
        assert!(take_report().is_none());
    }

    #[test]
    fn failed_call_does_not_contaminate_next_report() {
        let failed_call = || -> Result<(), &'static str> {
            let _report = Report::new();
            add("failed", Duration::from_millis(1));
            Err("inference failed")
        };
        assert!(failed_call().is_err());
        assert!(take_report().is_none());
        add("stale", Duration::from_secs(1));
        let _report = Report::new();
        assert!(take_report().is_none());
    }
}
