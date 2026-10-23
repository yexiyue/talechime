//! Compare native Metal GQA with explicit KV repetition, without model files.
use std::{error::Error, time::Instant};

use candle_core::{DType, Device, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    let device = Device::new_metal(0)?;
    for seq in [157, 287] {
        let q = Tensor::rand(-1f32, 1f32, (2, 16, seq, 128), &device)?.to_dtype(DType::F16)?;
        let k = Tensor::rand(-1f32, 1f32, (2, 8, seq, 128), &device)?.to_dtype(DType::F16)?;
        let v = Tensor::rand(-1f32, 1f32, (2, seq, 8, 128), &device)?
            .to_dtype(DType::F16)?
            .transpose(1, 2)?;
        let mask =
            Tensor::zeros((2, 1, seq, seq), DType::F16, &device)?.expand((2, 16, seq, seq))?;
        let run = |repeat| -> candle_core::Result<Tensor> {
            let (k, v) = if repeat {
                (
                    candle_transformers::utils::repeat_kv(k.clone(), 2)?,
                    candle_transformers::utils::repeat_kv(v.clone(), 2)?,
                )
            } else {
                (k.clone(), v.clone())
            };
            candle_nn::ops::sdpa(&q, &k, &v, Some(&mask), false, 1.0 / 128f32.sqrt(), 1.0)
        };
        let native = run(false)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let repeated = run(true)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if native != repeated {
            return Err(format!("native GQA differs at sequence length {seq}").into());
        }
        for _ in 0..50 {
            let _ = run(true)?;
            let _ = run(false)?;
        }
        device.synchronize()?;
        for repeat in [true, false, false, true, true, false, false, true] {
            let start = Instant::now();
            for _ in 0..100 {
                let _ = run(repeat)?;
            }
            device.synchronize()?;
            println!(
                "seq={seq} repeat={repeat} us={:.3}",
                start.elapsed().as_secs_f64() * 1e4
            );
        }
    }
    Ok(())
}
