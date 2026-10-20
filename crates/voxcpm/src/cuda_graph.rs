//! Model-local CUDA specializations. PCM delivery and EOS stay outside the graphs.
pub(crate) mod decoder;
#[cfg(test)]
mod decoder_tests;
pub(crate) mod replay;
#[cfg(test)]
mod tests;

use crate::acoustic::Acoustic;
use crate::cuda_graph::replay::Replay;
use candle_core::cuda_backend::CudaDevice;
use candle_core::{DType, Device, Result, Tensor};
use std::collections::HashMap;

const MAX_CAPTURED: usize = 4;

pub(crate) fn enabled(device: &Device) -> bool {
    device.is_cuda() && std::env::var_os("VOXCPM_CUDA_GRAPH").is_none_or(|v| v != "0")
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GraphKey {
    steps: usize,
    cfg: u64,
    dtype: DType,
}

struct Captured {
    replay: Replay,
    used: u64,
}

pub(crate) struct CfmGraphs {
    device: CudaDevice,
    captured: HashMap<GraphKey, Captured>,
    clock: u64,
    disabled: bool,
}

impl CfmGraphs {
    pub fn new(device: &Device) -> Option<Self> {
        if !enabled(device) {
            return None;
        }
        let Device::Cuda(device) = device else {
            return None;
        };
        Some(Self {
            device: device.clone(),
            captured: HashMap::new(),
            clock: 0,
            disabled: false,
        })
    }

    pub fn active(&self) -> bool {
        !self.disabled && !self.captured.is_empty()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &mut self,
        acoustic: &mut Acoustic,
        mu: &Tensor,
        prefix: &Tensor,
        noise: &Tensor,
        steps: usize,
        cfg: f64,
        cancel: &impl Fn() -> bool,
    ) -> Result<Tensor> {
        if cancel() {
            candle_core::bail!("generation cancelled");
        }
        if self.disabled {
            return acoustic.sample(mu, prefix, noise, steps, cfg, cancel);
        }
        let key = GraphKey {
            steps,
            cfg: cfg.to_bits(),
            dtype: mu.dtype(),
        };
        self.clock += 1;
        let attempt = if let Some(entry) = self.captured.get_mut(&key) {
            entry.used = self.clock;
            entry.replay.launch(&[mu, prefix, noise])
        } else {
            self.record(acoustic, mu, prefix, noise, steps, cfg, cancel, key)
        };
        self.finish(attempt, cancel, || {
            acoustic.sample(mu, prefix, noise, steps, cfg, cancel)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        acoustic: &mut Acoustic,
        mu: &Tensor,
        prefix: &Tensor,
        noise: &Tensor,
        steps: usize,
        cfg: f64,
        cancel: &impl Fn() -> bool,
        key: GraphKey,
    ) -> Result<Tensor> {
        let _guard = self.device.enable_cuda_graph_htod_cache();
        let warmup = acoustic.sample(mu, prefix, noise, steps, cfg, cancel)?;
        if cancel() {
            candle_core::bail!("generation cancelled");
        }
        // A different steps count replaces Acoustic's schedule. A graph must own
        // all external tensor addresses it captured, even while cached and idle.
        let parameters = acoustic.schedule_tensors();
        let replay = Replay::record(
            &self.device,
            &[mu, prefix, noise],
            &warmup,
            parameters,
            |args| acoustic.sample(&args[0], &args[1], &args[2], steps, cfg, &|| false),
        )?;
        if self.captured.len() == MAX_CAPTURED {
            let oldest = self
                .captured
                .iter()
                .min_by_key(|(_, entry)| entry.used)
                .map(|(key, _)| *key);
            if let Some(key) = oldest {
                self.captured.remove(&key);
            }
        }
        self.captured.insert(
            key,
            Captured {
                replay,
                used: self.clock,
            },
        );
        // Recording does not execute. The warm-up already computed this frame.
        Ok(warmup)
    }

    fn finish(
        &mut self,
        attempt: Result<Tensor>,
        cancel: &impl Fn() -> bool,
        fallback: impl FnOnce() -> Result<Tensor>,
    ) -> Result<Tensor> {
        if cancel() {
            candle_core::bail!("generation cancelled");
        }
        match attempt {
            Ok(output) => Ok(output),
            Err(error) => {
                tracing::warn!(%error, "voxcpm cuda graph failed; using eager inference");
                self.disabled = true;
                self.captured.clear();
                fallback()
            }
        }
    }
}
