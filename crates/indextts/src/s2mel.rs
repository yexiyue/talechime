//! s2mel container (`MyModel` in `indextts/s2mel/modules/commons.py`).
//!
//! Owns the two inference submodules from `s2mel.pth`:
//! `models.length_regulator` and `models.cfm`. `MyModel.__init__` builds
//! `InterpolateRegulator(channels=512, sampling_ratios=[1,1,1,1],
//! is_discrete=false, in_channels=1024, no VQ, no f0)` plus `CFM(args)` with
//! the `DiT` estimator; `load_checkpoint2` reads `state["net"]`, a dict keyed
//! by `models.*` module names, and the converted `s2mel.safetensors` keeps the
//! resulting `models.length_regulator.*` / `models.cfm.*` tensor names
//! verbatim, so this container only re-roots the [`VarBuilder`].

use crate::cfm::Cfm;
use crate::config::S2MelConfig;
use crate::length_regulator::InterpolateRegulator;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

pub struct S2Mel {
    pub length_regulator: InterpolateRegulator,
    pub cfm: Cfm,
}

impl S2Mel {
    pub fn load(vb: &VarBuilder, cfg: &S2MelConfig) -> Result<Self> {
        let models = vb.pp("models");
        Ok(Self {
            length_regulator: InterpolateRegulator::load(
                &models.pp("length_regulator"),
                &cfg.length_regulator,
            )?,
            cfm: Cfm::load(&models.pp("cfm"), cfg)?,
        })
    }
}

/// `sequence_mask(lengths, max_length)` from commons.py: `(B, max_len)` f32
/// 1/0 mask where `mask[b, i] = 1.0` iff `i < lengths[b]` (built on CPU).
pub fn sequence_mask(ylens: &[usize], max_len: usize) -> Result<Tensor> {
    sequence_mask_on(&Device::Cpu, ylens, max_len)
}

/// Device-resident variant of [`sequence_mask`], used by the forward paths
/// that multiply masks into tensors living on an inference device.
pub(crate) fn sequence_mask_on(device: &Device, ylens: &[usize], max_len: usize) -> Result<Tensor> {
    if ylens.is_empty() {
        candle_core::bail!("sequence_mask: ylens must be non-empty");
    }
    if ylens.iter().any(|&l| l > max_len) {
        candle_core::bail!("sequence_mask: length exceeds max_len {max_len}");
    }
    let pos = Tensor::arange(0f32, max_len as f32, device)?.unsqueeze(0)?;
    let lens: Vec<f32> = ylens.iter().map(|&l| l as f32).collect();
    let lens = Tensor::from_vec(lens, (ylens.len(), 1), device)?;
    let mask = pos.broadcast_lt(&lens)?;
    mask.to_dtype(DType::F32)
}

/// Type alias used by callers that keep the reference mel around.
pub type MelFrames = Tensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_mask_matches_python() {
        // torch.arange(max_length)[None, :] < lengths[:, None]
        let mask = sequence_mask(&[3, 5], 5).unwrap();
        assert_eq!(mask.dims(), vec![2, 5]);
        let rows = mask.to_vec2::<f32>().unwrap();
        assert_eq!(rows[0], vec![1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(rows[1], vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn sequence_mask_rejects_overlong_lengths() {
        assert!(sequence_mask(&[6], 5).is_err());
    }
}
