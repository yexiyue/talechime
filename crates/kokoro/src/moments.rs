//! CPU row moments following PyTorch 2.7.1 `moments_utils.h`'s architecture-matched
//! Welford/cascade reduction. See SOURCE.md and licenses/PYTORCH-LICENSE.

fn add(a: &mut (usize, f32, f32), b: (usize, f32, f32)) {
    let count = a.0 + b.0;
    let ratio = if count == 0 {
        0.0
    } else {
        b.0 as f32 / count as f32
    };
    let delta = b.1 - a.1;
    a.1 += ratio * delta;
    a.2 += b.2 + delta * delta * ratio * a.0 as f32;
    a.0 = count;
}

pub(crate) fn row(values: &[f32]) -> (f32, f32) {
    const LANES: usize = if cfg!(target_arch = "aarch64") { 4 } else { 8 };
    let vectors = values.len() / LANES;
    let chunks = vectors.div_ceil(16);
    let depth = (usize::BITS - chunks.saturating_sub(1).leading_zeros()).max(1) as usize;
    let mut stack = vec![[(0, 0.0f32, 0.0f32); LANES]; depth];
    for chunk in 0..chunks {
        let count = (vectors - chunk * 16).min(16);
        for lane in 0..LANES {
            let mut mean = 0.0;
            let mut m2 = 0.0;
            for j in 0..count {
                let value = values[(chunk * 16 + j) * LANES + lane];
                let delta = value - mean;
                mean += delta * (1.0 / (j + 1) as f32);
                m2 += delta * (value - mean);
            }
            add(&mut stack[0][lane], (count, mean, m2));
        }
        let mut mask = chunk + 1;
        for level in 1..depth {
            if mask & 1 != 0 {
                break;
            }
            let previous = stack[level - 1];
            for (target, source) in stack[level].iter_mut().zip(previous) {
                add(target, source);
            }
            stack[level - 1] = [(0, 0.0, 0.0); LANES];
            mask >>= 1;
        }
    }
    for level in 1..depth {
        let previous = stack[level];
        for (target, source) in stack[0].iter_mut().zip(previous) {
            add(target, source);
        }
    }
    let mut total = (0, 0.0, 0.0);
    for &value in &values[vectors * LANES..] {
        let delta = value - total.1;
        total.0 += 1;
        total.1 += delta / total.0 as f32;
        total.2 += delta * (value - total.1);
    }
    for lane in stack[0] {
        let count = total.0 + lane.0;
        let ratio = if count == 0 {
            0.0
        } else {
            lane.0 as f32 / count as f32
        };
        let delta = lane.1 - total.1;
        total.1 = ratio.mul_add(delta, total.1);
        total.2 += ((delta * delta) * ratio).mul_add(total.0 as f32, lane.2);
        total.0 = count;
    }
    (total.1, total.2 / values.len() as f32)
}

#[cfg(test)]
mod tests {
    #[test]
    fn normalization_matches_independent_pytorch_rows() -> candle_core::Result<()> {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/moments.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let values: Vec<f32> = case["input"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let expected: Vec<f32> = case["normalized"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let x = candle_core::Tensor::from_slice(
                &values,
                (1, values.len()),
                &candle_core::Device::Cpu,
            )?;
            let actual = crate::ops::normalize(&x, 1, 1e-5)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(max < 1e-6, "width {}: max error {max}", values.len());
            let floats = |name: &str| -> Vec<f32> {
                case[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            };
            let weight = floats("weight");
            let bias = floats("bias");
            let expected = floats("affine");
            let w = candle_core::Tensor::from_vec(weight, values.len(), &candle_core::Device::Cpu)?;
            let b = candle_core::Tensor::from_vec(bias, values.len(), &candle_core::Device::Cpu)?;
            let actual = crate::ops::affine_norm(&x, &w, &b, 1e-5)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let max = actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(max < 2e-6, "affine width {}: max error {max}", values.len());
        }
        Ok(())
    }
}
