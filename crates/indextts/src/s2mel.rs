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

pub fn sequence_mask(ylens: &[usize], max_len: usize) -> Result<Tensor> {
    sequence_mask_on(&Device::Cpu, ylens, max_len)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_mask_matches_python() {
        let mask = sequence_mask(&[3, 5], 5).unwrap();
        assert_eq!(mask.dims(), vec![2, 5]);
        let rows = mask.to_vec2::<f32>().unwrap();
        assert_eq!(rows[0], vec![1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(rows[1], vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn sequence_mask_out_of_range() {
        assert!(sequence_mask(&[6], 5).is_err());
    }
}
