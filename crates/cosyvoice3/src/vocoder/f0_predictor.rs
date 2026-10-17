//! Causal F0 predictor (`cosyvoice/hifigan/f0_predictor.py`).
//!
//! The official inference path runs this module in float64 (the source
//! notes "f0_predictor precision is crucial for causal inference"); we do
//! the same on CPU regardless of the generator device.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::causal_conv::{CausalConv1d, CausalType};
use crate::activations::Elu;

#[derive(Debug)]
pub struct CausalConvRnnF0Predictor {
    condnet: Vec<CausalConv1d>,
    elu: Elu,
    classifier: Linear,
}

impl CausalConvRnnF0Predictor {
    pub fn new(in_channels: usize, cond_channels: usize, vb: VarBuilder) -> Result<Self> {
        // nn.Sequential indices: 0, 2, 4, 6, 8 (conv at evens).
        let first = CausalConv1d::new(
            in_channels,
            cond_channels,
            4,
            1,
            CausalType::Right,
            vb.pp("condnet.0"),
        )?;
        let mut condnet = vec![first];
        for i in 1..5 {
            condnet.push(CausalConv1d::new(
                cond_channels,
                cond_channels,
                3,
                1,
                CausalType::Left,
                vb.pp(format!("condnet.{}", i * 2)),
            )?);
        }
        Ok(Self {
            condnet,
            elu: Elu,
            classifier: candle_nn::linear(cond_channels, 1, vb.pp("classifier"))?,
        })
    }

    /// `mel`: `[B, 80, T]` in f32 on any device; returns `[B, T]` f0 in the
    /// same device/dtype. `finalize` selects the streamed split of the first
    /// look-right conv.
    /// Look-right padding of the first condnet conv (streamed mel trim).
    pub fn first_causal_padding(&self) -> usize {
        self.condnet[0].causal_padding()
    }

    pub fn forward(&self, mel: &Tensor, finalize: bool) -> Result<Tensor> {
        // Float64 on CPU, per the official inference path.
        let x = mel
            .to_device(&candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F64)?;
        let mut x = if finalize {
            self.condnet[0].forward(&x)?
        } else {
            let pad = self.condnet[0].causal_padding();
            if x.dim(2)? > pad {
                let main = x.narrow(2, 0, x.dim(2)? - pad)?;
                let cache = x.narrow(2, x.dim(2)? - pad, pad)?;
                self.condnet[0].forward_with_cache(&main, &cache)?
            } else {
                self.condnet[0].forward(&x)?
            }
        };
        x = self.elu.forward(&x)?;
        for conv in self.condnet.iter().skip(1) {
            x = self.elu.forward(&conv.forward(&x)?)?;
        }
        let x = x.transpose(1, 2)?;
        let f0 = self.classifier.forward(&x)?.squeeze(2)?.abs()?;
        // Metal cannot convert f64 tensors; narrow on the owning CPU first.
        f0.to_dtype(mel.dtype())?.to_device(mel.device())
    }
}
