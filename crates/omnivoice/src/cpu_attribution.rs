//! Test-only real-input diagnostics. Never compiled into the inference library.
use std::{collections::HashMap, fs, hint::black_box, path::PathBuf, time::Instant};

use candle_core::{Device, Tensor};
use serde_json::json;

use crate::Result;

pub(crate) fn root() -> Result<PathBuf> {
    std::env::var_os("OMNI_CPU_TRACE_ROOT")
        .map(PathBuf::from)
        .ok_or_else(|| crate::OmniVoiceError::InvalidRequest("set OMNI_CPU_TRACE_ROOT".into()))
}

pub(crate) fn model() -> Result<PathBuf> {
    std::env::var_os("OMNI_CPU_TRACE_MODEL")
        .map(PathBuf::from)
        .ok_or_else(|| crate::OmniVoiceError::InvalidRequest("set OMNI_CPU_TRACE_MODEL".into()))
}

pub(crate) fn save(name: &str, tensors: &[(&str, &Tensor)]) -> Result<()> {
    let root = root()?;
    fs::create_dir_all(&root)?;
    let metadata: serde_json::Map<String, serde_json::Value> = tensors
        .iter()
        .map(|(name, t)| {
            (
                (*name).into(),
                json!({"shape":t.dims(), "strides":t.stride(),
            "offset":t.layout().start_offset(), "dtype":format!("{:?}",t.dtype())}),
            )
        })
        .collect();
    let tensors: HashMap<_, _> = tensors
        .iter()
        .map(|(name, t)| ((*name).to_string(), (*t).clone()))
        .collect();
    let path = root.join(format!("{name}.safetensors"));
    if path.exists() {
        return Err(crate::OmniVoiceError::InvalidRequest(format!(
            "refusing to overwrite {}",
            path.display()
        )));
    }
    candle_core::safetensors::save(&tensors, &path)?;
    fs::write(
        root.join(format!("{name}.layout.json")),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    Ok(())
}

pub(crate) fn load(name: &str) -> Result<HashMap<String, Tensor>> {
    Ok(candle_core::safetensors::load(
        root()?.join(format!("{name}.safetensors")),
        &Device::Cpu,
    )?)
}

pub(crate) fn equal(a: &Tensor, b: &Tensor) -> Result<()> {
    assert_eq!(a.dims(), b.dims());
    let a = a.flatten_all()?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(
        a.iter()
            .zip(&b)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count(),
        0,
        "replay changed bits"
    );
    Ok(())
}

pub(crate) fn bench(case: &str, op: &str, mut run: impl FnMut() -> Result<Tensor>) -> Result<()> {
    black_box(run()?);
    for group in 0..5 {
        let started = Instant::now();
        for _ in 0..5 {
            black_box(run()?);
        }
        println!(
            "CPU_TRACE {}",
            json!({"case":case,"op":op,"group":group,
            "microseconds":started.elapsed().as_secs_f64()*1e6/5.0})
        );
    }
    Ok(())
}

#[test]
fn tensor_archive_preserves_bits_for_noncontiguous_input() -> Result<()> {
    // Avoid process-global environment mutation: exercise the serialization primitives directly.
    let directory = tempfile::tempdir()?;
    let tensor = Tensor::from_vec(vec![0f32, -0f32, 1., -2., 3., 4.], (2, 3), &Device::Cpu)?.t()?;
    let tensors = HashMap::from([("x", tensor.clone())]);
    let path = directory.path().join("tensor.safetensors");
    candle_core::safetensors::save(&tensors, &path)?;
    let loaded = candle_core::safetensors::load(path, &Device::Cpu)?;
    equal(&tensor, &loaded["x"])
}
