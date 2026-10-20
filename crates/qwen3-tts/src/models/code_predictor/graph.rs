//! CUDA replay for one fixed-shape acoustic code predictor frame.
//!
//! A frame is a two-token prefill followed by `num_code_groups - 2`
//! single-token steps, so every frame issues the same kernels against the same
//! KV addresses. The graph covers the whole loop; the talker step, semantic
//! sampling, EOS and PCM stay outside it.
use super::CodePredictor;
use crate::cuda_graph::replay::Replay;
use crate::models::kv_cache::AnyKVCache;
use anyhow::{Error, Result};
use candle_core::cuda_backend::CudaDevice;
use candle_core::{Device, Tensor};

/// One captured frame per generation, over caches owned by the caller.
///
/// The caches must outlive this object and must not be used eagerly while a
/// replay exists: a replay rewrites every valid KV position in the fixed
/// sequence, but host-side cache bookkeeping is not re-executed.
pub(super) struct AcousticGraph {
    device: CudaDevice,
    // Destroy the graph before the caller releases the caches it captured.
    replay: Option<Replay>,
    disabled: bool,
}

impl AcousticGraph {
    pub(super) fn new(device: &Device) -> Option<Self> {
        // Profiling measures the per-step spans of eager inference; a replay
        // collapses them into one launch and hides the work being qualified.
        if cfg!(feature = "profiling") || std::env::var("QWEN_CP_GRAPH").as_deref() == Ok("0") {
            return None;
        }
        let Device::Cuda(device) = device else {
            return None;
        };
        Some(Self {
            device: device.clone(),
            replay: None,
            disabled: false,
        })
    }

    #[cfg(test)]
    pub(super) fn active(&self) -> bool {
        self.replay.is_some() && !self.disabled
    }

    /// `None` asks the caller to run eager inference for this frame.
    pub(super) fn try_generate(
        &mut self,
        predictor: &CodePredictor,
        caches: &mut [AnyKVCache],
        talker_hidden: &Tensor,
        semantic_embed: &Tensor,
    ) -> Result<Option<Tensor>> {
        if self.disabled {
            return Ok(None);
        }
        let attempt = self.replay.as_ref().map(|replay| {
            replay
                .launch(&[talker_hidden, semantic_embed])
                .map_err(Error::from)
        });
        match attempt {
            Some(Ok(codes)) => return Ok(Some(codes)),
            Some(Err(error)) => {
                // A frame is a pure function of its two inputs and rewrites
                // every KV position it reads, so eager may recompute this frame
                // even though the launch failed part way through.
                tracing::warn!(
                    %error,
                    "qwen code predictor cuda graph failed; using eager inference"
                );
                self.replay = None;
                self.disabled = true;
                return Ok(None);
            }
            None => {}
        }
        self.record(predictor, caches, talker_hidden, semantic_embed)
    }

    fn record(
        &mut self,
        predictor: &CodePredictor,
        caches: &mut [AnyKVCache],
        talker_hidden: &Tensor,
        semantic_embed: &Tensor,
    ) -> Result<Option<Tensor>> {
        let _guard = self.device.enable_cuda_graph_htod_cache();
        // The warm-up is this frame's real result and primes cublas workspaces,
        // kernel modules and parameter uploads that the capture reuses.
        let warmup = predictor.generate_acoustic_codes(talker_hidden, semantic_embed, caches)?;
        let recorded = Replay::record(
            &self.device,
            &[talker_hidden, semantic_embed],
            &warmup,
            vec![],
            // `Replay` speaks Candle's error type; capture diagnostics keep the
            // message only, which is what the fallback warning reports.
            |inputs| {
                predictor
                    .generate_acoustic_codes(&inputs[0], &inputs[1], caches)
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))
            },
        );
        match recorded {
            Ok(replay) => {
                self.replay = Some(replay);
                Ok(Some(warmup))
            }
            Err(error) => {
                tracing::warn!(
                    %error,
                    "qwen code predictor cuda graph capture unavailable; using eager inference"
                );
                self.disabled = true;
                Ok(Some(warmup))
            }
        }
    }
}
