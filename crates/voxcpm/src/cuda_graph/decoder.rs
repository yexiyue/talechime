//! Experimental fixed-frame AudioVAE replay with explicit device-side state.
use super::replay::{Replay, copy_into};
use crate::codec::Codec;
use candle_core::{Device, Result, Tensor};

/// Own one graph per Codec; the Codec and its weights must outlive this object.
/// Output aliases replay scratch and must be consumed before the next decode.
#[derive(Default)]
pub(crate) struct DecoderGraph {
    // Destroy the graph before releasing any state addresses it captured.
    replay: Option<Replay>,
    state: Vec<Option<Tensor>>,
    needs_reset: bool,
    disabled: bool,
}

impl DecoderGraph {
    pub(crate) fn active(&self) -> bool {
        self.replay.is_some() && !self.disabled
    }
    pub(crate) fn for_device(device: &Device, default_enabled: bool) -> Option<Self> {
        (super::enabled(device)
            && std::env::var_os("VOXCPM_DECODER_GRAPH")
                .map_or(default_enabled, |value| value == "1"))
        .then(Self::default)
    }

    /// Call at every request boundary, alongside the owner's Codec reset.
    pub(crate) fn reset(&mut self) {
        self.needs_reset = true;
    }

    pub(crate) fn decode(
        &mut self,
        codec: &mut Codec,
        input: &Tensor,
        cancel: &impl Fn() -> bool,
    ) -> Result<Tensor> {
        if cancel() {
            candle_core::bail!("generation cancelled");
        }
        if input.dims() != [1, 64, 4] {
            candle_core::bail!("decoder graph requires a single [1, 64, 4] frame");
        }
        let Device::Cuda(device) = input.device() else {
            candle_core::bail!("decoder graph requires CUDA input");
        };
        let stream = device.cuda_stream();

        // A failed launch may already have updated persistent state. Until an
        // explicit request reset, prohibit eager continuation of that request.
        if self.disabled && self.replay.is_some() && !self.needs_reset {
            candle_core::bail!("decoder graph state is uncertain; abort and reset the request");
        }
        if self.needs_reset {
            if self.replay.is_some() && !self.disabled {
                let reset = (|| {
                    for state in self.state.iter().flatten() {
                        let zero =
                            Tensor::zeros(state.shape().clone(), state.dtype(), state.device())?;
                        copy_into(&stream, state, &zero)?;
                    }
                    codec.restore_decoder_state(&self.state)
                })();
                if let Err(error) = reset {
                    self.disabled = true;
                    return Err(error);
                }
            } else {
                let empty = vec![None; codec.decoder_state().len()];
                codec.restore_decoder_state(&empty)?;
                self.replay = None;
                self.state.clear();
            }
            self.needs_reset = false;
        }
        if self.disabled {
            return codec.decode(input, cancel);
        }

        let output = if let Some(replay) = &self.replay {
            // Host references stay fixed while kernels copy each new history
            // and overlap tail into the same persistent storage on the stream.
            let attempt = codec
                .restore_decoder_state(&self.state)
                .and_then(|()| replay.launch(&[input]));
            match attempt {
                Ok(output) => output,
                Err(error) => {
                    self.disabled = true;
                    // Never recompute this frame: even a synchronous launch
                    // rejection is not assumed to prove no state writes occurred.
                    // Asynchronous failures also surface at the owner's readback;
                    // the owner must abort that request, never emit failed PCM.
                    tracing::warn!(%error, "AudioVAE replay failed; abort decoder request");
                    return Err(error);
                }
            }
        } else {
            let _guard = device.enable_cuda_graph_htod_cache();
            // The actual frame is the warmup: its state and PCM are already the
            // correct next values. Recording must not execute it a second time.
            let warmup = codec.decode(input, cancel)?;
            self.state = codec.decoder_state();
            if cancel() {
                candle_core::bail!("generation cancelled");
            }
            let parameters = self.state.iter().flatten().cloned().collect();
            let recorded = Replay::record(device, &[input], &warmup, parameters, |inputs| {
                // Always restore host references before leaving capture, even
                // when decode or a copy fails. This releases capture allocations
                // retained by Codec; Replay releases the returned output itself.
                let operation = (|| {
                    let output = codec.decode(&inputs[0], &|| false)?;
                    let produced = codec.decoder_state();
                    if produced.len() != self.state.len() {
                        candle_core::bail!("decoder graph state length changed");
                    }
                    for (persistent, next) in self.state.iter().zip(&produced) {
                        match (persistent, next) {
                            (Some(destination), Some(source)) => {
                                copy_into(&stream, destination, source)?;
                            }
                            (None, None) => {}
                            _ => candle_core::bail!("decoder graph state presence changed"),
                        }
                    }
                    Ok(output)
                })();
                codec.restore_decoder_state(&self.state)?;
                operation
            });
            // Also covers failures before the operation closure was entered.
            codec.restore_decoder_state(&self.state)?;
            match recorded {
                Ok(replay) => self.replay = Some(replay),
                Err(error) => {
                    tracing::warn!(%error, "AudioVAE capture unavailable; keeping warmup and using eager");
                    self.disabled = true;
                }
            }
            warmup
        };
        if cancel() {
            candle_core::bail!("generation cancelled");
        }
        Ok(output)
    }
}
