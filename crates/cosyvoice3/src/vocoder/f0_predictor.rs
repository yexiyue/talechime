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
        let profile = std::env::var("CV3_HIFT_PROFILE").is_ok_and(|v| v != "0");
        let t_all = std::time::Instant::now();
        let t_cond = t_all;
        let t_mv = std::time::Instant::now();
        let x = mel
            .to_device(&candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F64)?;
        let mv_ms = t_mv.elapsed().as_secs_f64() * 1e3;
        let t_first = std::time::Instant::now();
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
        let first_ms = t_first.elapsed().as_secs_f64() * 1e3;
        let t_elu0 = std::time::Instant::now();
        x = self.elu.forward(&x)?;
        let elu0_ms = t_elu0.elapsed().as_secs_f64() * 1e3;
        for conv in self.condnet.iter().skip(1) {
            x = self.elu.forward(&conv.forward(&x)?)?;
        }
        let cond_ms = t_cond.elapsed().as_secs_f64() * 1e3;
        let t_cls = std::time::Instant::now();
        let x = x.transpose(1, 2)?;
        let f0 = self.classifier.forward(&x)?.squeeze(2)?.abs()?;
        // Metal cannot convert f64 tensors; narrow on the owning CPU first.
        let f0 = f0.to_dtype(mel.dtype())?.to_device(mel.device())?;
        if profile {
            eprintln!(
                "[cv3-hift] f0 mv {mv_ms:.0} conv0 {first_ms:.0} elu0 {elu0_ms:.0} rest {:.0} cls {:.0} (total {:.0} ms)",
                cond_ms - mv_ms - first_ms - elu0_ms,
                t_cls.elapsed().as_secs_f64() * 1e3,
                t_all.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(f0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn streamed_prefixes_match_scalar_history_and_final_padding() -> Result<()> {
        let device = Device::Cpu;
        let mut weights = std::collections::HashMap::new();
        for i in 0..5 {
            let kernel: &[f64] = if i == 0 {
                &[0.5, -0.25, 0.125, 0.75]
            } else {
                &[0.25, -0.5, 0.75]
            };
            weights.insert(
                format!("condnet.{}.weight", i * 2),
                Tensor::from_slice(kernel, (1, 1, kernel.len()), &device)?,
            );
            weights.insert(
                format!("condnet.{}.bias", i * 2),
                Tensor::from_slice(&[-0.1f64], 1, &device)?,
            );
        }
        weights.insert(
            "classifier.weight".into(),
            Tensor::from_slice(&[1.5f64], (1, 1), &device)?,
        );
        weights.insert(
            "classifier.bias".into(),
            Tensor::from_slice(&[0.2f64], 1, &device)?,
        );
        let predictor = CausalConvRnnF0Predictor::new(
            1,
            1,
            VarBuilder::from_tensors(weights, DType::F64, &device),
        )?;
        let values: Vec<f64> = (0..85).map(|i| ((i * 17 % 23) as f64 - 11.) / 8.).collect();
        // Scalar reference keeps the first conv's three future frames and
        // all four subsequent convs' left history, including at the seam.
        let mut expected = values.clone();
        for layer in 0..5 {
            let kernel: &[f64] = if layer == 0 {
                &[0.5, -0.25, 0.125, 0.75]
            } else {
                &[0.25, -0.5, 0.75]
            };
            expected = (0..values.len())
                .map(|i| {
                    let v = -0.1
                        + kernel
                            .iter()
                            .enumerate()
                            .map(|(j, w)| {
                                let index = if layer == 0 {
                                    Some(i + j)
                                } else {
                                    (i + j).checked_sub(2)
                                };
                                w * index.and_then(|k| expected.get(k)).copied().unwrap_or(0.)
                            })
                            .sum::<f64>();
                    if v >= 0. { v } else { v.exp() - 1. }
                })
                .collect();
        }
        let expected: Vec<f64> = expected.iter().map(|v| (1.5 * v + 0.2).abs()).collect();
        for frames in [8, 25, 50, 75, 85] {
            for finalize in [false, true] {
                // Final padding is only comparable to the full scalar
                // oracle at the actual end of the utterance.
                if finalize && frames != values.len() {
                    continue;
                }
                let mel = Tensor::from_slice(&values[..frames], (1, 1, frames), &device)?;
                let actual = predictor
                    .forward(&mel, finalize)?
                    .flatten_all()?
                    .to_vec1::<f64>()?;
                assert_eq!(actual.len(), frames - if finalize { 0 } else { 3 });
                for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-12,
                        "frames={frames}, finalize={finalize}, frame={i}: {a} != {b}"
                    );
                }
            }
        }
        Ok(())
    }
}
