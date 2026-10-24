//! Optional code dumps and device-synchronized timing.
//! Model-owned canary storage remains with `Stage0RuntimePlan`.

use candle_core::{DType, Device, Tensor};

use crate::error::Result;

/// Diagnostic-only: `OMNIVOICE_CODES_DUMP=<path>` appends the final discrete
/// codes of every generation so two numeric paths can be compared token by
/// token. Never enable in production runs.
pub(super) fn dump_final_tokens(final_tokens: &[Tensor]) {
    let Ok(path) = std::env::var("OMNIVOICE_CODES_DUMP") else {
        return;
    };
    for tokens in final_tokens {
        let Ok(values) = tokens.to_dtype(DType::I64).and_then(|t| t.flatten_all()) else {
            return;
        };
        let Ok(values) = values.to_vec1::<i64>() else {
            return;
        };
        let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        else {
            return;
        };
        use std::io::Write;
        let count = (values.len() as u32).to_le_bytes();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let _ = file.write_all(&count);
        let _ = file.write_all(&bytes);
    }
}

/// Diagnostic-only: `OMNIVOICE_SYNC_PROBE=1` brackets the stage0 forward with
/// device barriers so the measured time reflects GPU execution rather than
/// command submission. Never enable in production runs.
pub(super) fn sync_probe_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("OMNIVOICE_SYNC_PROBE").is_ok_and(|value| value == "1"))
}

/// Diagnostic-only canary: a fixed-shape matmul measured next to the stage0
/// forward so GPU throughput regressions can be separated from
/// model-specific work.
pub(super) fn sync_probe_canary(device: &Device) -> Option<(Tensor, Tensor)> {
    // This GPU canary uses F16, which CPU Accelerate matmul does not support.
    if device.is_cpu() {
        return None;
    }
    let lhs = Tensor::rand(0f32, 1f32, (300, 1024), device)
        .ok()?
        .to_dtype(DType::F16)
        .ok()?;
    let rhs = Tensor::rand(0f32, 1f32, (1024, 1024), device)
        .ok()?
        .to_dtype(DType::F16)
        .ok()?;
    Some((lhs, rhs))
}

/// Diagnostic-only device barrier helper used by `OMNIVOICE_SYNC_PROBE=1`.
pub(super) struct SyncProbe {
    pub(super) enabled: bool,
}

impl SyncProbe {
    pub(super) fn new() -> Self {
        Self {
            enabled: sync_probe_enabled(),
        }
    }

    /// Barrier, then start the clock.
    pub(super) fn begin(&self, device: &Device) -> Result<Option<std::time::Instant>> {
        if !self.enabled {
            return Ok(None);
        }
        device.synchronize()?;
        Ok(Some(std::time::Instant::now()))
    }

    /// Barrier, then report the elapsed time as a stage timing line.
    pub(super) fn end(
        &self,
        device: &Device,
        started: Option<std::time::Instant>,
        label: &str,
    ) -> Result<()> {
        let Some(started) = started else {
            return Ok(());
        };
        device.synchronize()?;
        eprintln!(
            "[stage] name={label} ms={:.2}",
            started.elapsed().as_secs_f64() * 1000.0
        );
        Ok(())
    }
}
