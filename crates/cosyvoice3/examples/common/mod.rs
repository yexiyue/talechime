use anyhow::Result;
use candle_core::{DType, Tensor};

pub fn compare(name: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    anyhow::ensure!(
        actual.dims() == expected.dims(),
        "{name}: shape {:?} != {:?}",
        actual.dims(),
        expected.dims()
    );
    let a = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let b = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut max = 0f32;
    let mut sum = 0f64;
    for (a, b) in a.iter().zip(&b) {
        anyhow::ensure!(a.is_finite() && b.is_finite(), "{name}: non-finite value");
        let e = (a - b).abs();
        max = max.max(e);
        sum += f64::from(e).powi(2);
    }
    println!(
        "{name}: shape={:?} max_abs={max:.8e} rmse={:.8e} tolerance={tolerance:.2e} {}",
        actual.dims(),
        (sum / a.len() as f64).sqrt(),
        if max <= tolerance { "PASS" } else { "FAIL" }
    );
    anyhow::ensure!(max <= tolerance, "{name}: tolerance exceeded");
    Ok(())
}
