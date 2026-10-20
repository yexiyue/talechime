//! Model-owned replay of stateless history decoding, serialized with scratch copies.
use super::Decoder12Hz;
use crate::cuda_graph::replay::Replay;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::collections::VecDeque;

const MAX_ENTRIES: usize = 4;
const MAX_FRAMES: usize = 64;

#[derive(Default)]
pub(super) struct DecoderGraphs {
    // Graphs must be destroyed before the decoder's weights.
    entries: VecDeque<(usize, Option<Replay>)>,
    disabled: bool,
}

impl DecoderGraphs {
    #[cfg(test)]
    pub(super) fn recorded_shapes(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.1.is_some())
            .count()
    }

    pub(super) fn decode(&mut self, decoder: &Decoder12Hz, codes: &Tensor) -> Result<Tensor> {
        let (batch, quantizers, frames) = codes.dims3()?;
        if self.disabled
            || cfg!(feature = "profiling")
            || std::env::var("QWEN_DECODE_GRAPH").as_deref() == Ok("0")
            || batch != 1
            || quantizers != 16
            || codes.dtype() != DType::I64
            || frames == 0
            || frames > MAX_FRAMES
        {
            return decoder.decode(codes);
        }
        let Some(index) = self.entries.iter().position(|entry| entry.0 == frames) else {
            if self.entries.len() == MAX_ENTRIES {
                self.entries.pop_front();
            }
            self.entries.push_back((frames, None));
            return decoder.decode(codes);
        };
        let entry = self.entries.remove(index).expect("located graph entry");
        self.entries.push_back(entry);
        if let Some(replay) = &self.entries.back().expect("graph entry").1 {
            // Never expose mutable graph scratch across calls or sessions.
            let result = replay.launch(&[codes]).and_then(|output| output.copy());
            if result.is_err() {
                self.disabled = true;
                self.entries.clear();
                // An asynchronous CUDA failure may invalidate the context.
                // Propagate it instead of promising eager recovery.
            }
            return Ok(result?);
        }
        let Device::Cuda(device) = codes.device() else {
            unreachable!("CUDA entry point")
        };
        let tables = decoder.tables(frames)?;
        let _guard = device.enable_cuda_graph_htod_cache();
        let warmup = decoder.decode(codes)?;
        let record = Replay::record(
            device,
            &[codes],
            &warmup,
            vec![tables.cos.clone(), tables.sin.clone(), tables.mask.clone()],
            |inputs| {
                decoder
                    .decode(&inputs[0])
                    .map_err(|error| candle_core::Error::Msg(error.to_string()))
            },
        );
        match record {
            Ok(replay) => self.entries.back_mut().expect("graph entry").1 = Some(replay),
            Err(error) => {
                self.disabled = true;
                self.entries.clear();
                anyhow::ensure!(
                    !crate::cuda_graph::replay::is_capturing(&device.cuda_stream())?,
                    "decoder capture left the stream active"
                );
                tracing::warn!(%error, "qwen decoder capture unavailable; using eager inference");
            }
        }
        Ok(warmup)
    }
}
