//! Model-owned replay of stateless history decoding, serialized with scratch copies.
use super::Decoder12Hz;
use crate::cuda_graph::replay::Replay;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::collections::VecDeque;

const MAX_ENTRIES: usize = 4;
const MAX_FRAMES: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    #[cfg(test)]
    Codes,
    Vocoder,
}

impl Mode {
    fn run(self, decoder: &Decoder12Hz, input: &Tensor) -> Result<Tensor> {
        match self {
            #[cfg(test)]
            Self::Codes => decoder.decode(input),
            Self::Vocoder => decoder.vocode(input),
        }
    }

    fn accepts(self, channels: usize, dtype: DType) -> bool {
        match self {
            #[cfg(test)]
            Self::Codes => channels == 16 && dtype == DType::I64,
            Self::Vocoder => channels == 1024 && dtype == DType::F32,
        }
    }
}

#[derive(Default)]
pub(super) struct DecoderGraphs {
    // Graphs must be destroyed before the decoder's weights.
    entries: VecDeque<((Mode, usize), Option<Replay>)>,
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

    #[cfg(test)]
    pub(super) fn recorded_frame_sizes(&self) -> Vec<usize> {
        self.entries
            .iter()
            .filter_map(|((_, frames), replay)| replay.as_ref().map(|_| *frames))
            .collect()
    }

    #[cfg(test)]
    pub(super) fn decode(&mut self, decoder: &Decoder12Hz, codes: &Tensor) -> Result<Tensor> {
        self.run(decoder, codes, Mode::Codes)
    }

    pub(super) fn vocode(&mut self, decoder: &Decoder12Hz, features: &Tensor) -> Result<Tensor> {
        self.run(decoder, features, Mode::Vocoder)
    }

    fn run(&mut self, decoder: &Decoder12Hz, input: &Tensor, mode: Mode) -> Result<Tensor> {
        let (batch, channels, frames) = input.dims3()?;
        let key = (mode, frames);
        if self.disabled
            || cfg!(feature = "profiling")
            || std::env::var("QWEN_DECODE_GRAPH").as_deref() == Ok("0")
            || batch != 1
            || !mode.accepts(channels, input.dtype())
            || !input.device().is_cuda()
            || frames == 0
            || frames > MAX_FRAMES
        {
            return mode.run(decoder, input);
        }
        let Some(index) = self.entries.iter().position(|entry| entry.0 == key) else {
            if self.entries.len() == MAX_ENTRIES {
                self.entries.pop_front();
            }
            self.entries.push_back((key, None));
            return mode.run(decoder, input);
        };
        let entry = self.entries.remove(index).expect("located graph entry");
        self.entries.push_back(entry);
        if let Some(replay) = &self.entries.back().expect("graph entry").1 {
            // Never expose mutable graph scratch across calls or sessions.
            let result = replay.launch(&[input]).and_then(|output| output.copy());
            if result.is_err() {
                self.disabled = true;
                self.entries.clear();
                // An asynchronous CUDA failure may invalidate the context.
                // Propagate it instead of promising eager recovery.
            }
            return Ok(result?);
        }
        let Device::Cuda(device) = input.device() else {
            unreachable!("CUDA entry point")
        };
        // Only the codes path reads RoPE/mask tables during capture. Vocoder
        // capture reads model weights and its input, with no transformer state.
        let parameters = match mode {
            #[cfg(test)]
            Mode::Codes => {
                let tables = decoder.tables(frames)?;
                vec![tables.cos.clone(), tables.sin.clone(), tables.mask.clone()]
            }
            Mode::Vocoder => Vec::new(),
        };
        let _guard = device.enable_cuda_graph_htod_cache();
        let warmup = mode.run(decoder, input)?;
        let record = Replay::record(device, &[input], &warmup, parameters, |inputs| {
            mode.run(decoder, &inputs[0])
                .map_err(|error| candle_core::Error::Msg(error.to_string()))
        });
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
