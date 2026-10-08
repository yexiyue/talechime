//! Provider selection is explicit; registration never implies measured acceleration.
pub mod calibration;
#[cfg(any(feature = "ort-coreml", feature = "ort-cuda"))]
use ort::ep::ExecutionProvider;
#[cfg(feature = "moss")]
use ort::session::Session;
#[cfg(feature = "moss")]
use std::path::Path;
use tts_protocol::{Device, Event};

pub fn compiled() -> Vec<Device> {
    vec![
        Device::Cpu,
        #[cfg(feature = "ort-coreml")]
        Device::Coreml,
        #[cfg(feature = "ort-cuda")]
        Device::Cuda,
    ]
}
pub fn available() -> Vec<Device> {
    compiled()
        .into_iter()
        .filter(|device| match device {
            Device::Cpu => true,
            #[cfg(feature = "ort-coreml")]
            Device::Coreml => ort::ep::CoreML::default().is_available().unwrap_or(false),
            #[cfg(feature = "ort-cuda")]
            Device::Cuda => {
                ort::ep::CUDA::default().is_available().unwrap_or(false)
                    && std::process::Command::new("nvidia-smi")
                        .args(["--query-gpu=uuid", "--format=csv,noheader"])
                        .output()
                        .is_ok_and(|output| output.status.success() && !output.stdout.is_empty())
            }
            _ => false,
        })
        .collect()
}
pub fn validate(device: Device) -> anyhow::Result<()> {
    if device == Device::Auto {
        return Ok(());
    }
    let available = available();
    anyhow::ensure!(
        available.contains(&device),
        "device {device:?} is unavailable; compiled {:?}, available {:?}",
        compiled(),
        available
    );
    Ok(())
}
pub fn status(component: &str, selected: Device, reason: Option<String>) -> Event {
    Event::DeviceStatus {
        component: component.into(),
        compiled: compiled(),
        available: available(),
        selected,
        reason,
    }
}
#[cfg(feature = "moss")]
pub fn session(path: &Path, device: Device, cache: &Path) -> anyhow::Result<Session> {
    validate(device)?;
    let builder = Session::builder()?
        .with_intra_threads(4)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut builder = match device {
        // Keep the CPU choice explicit, independent of ORT's device policies.
        Device::Cpu => builder
            .with_execution_providers([ort::ep::CPU::default().build().error_on_failure()])
            .map_err(|e| anyhow::anyhow!("{e}"))?,
        Device::Coreml => {
            #[cfg(feature = "ort-coreml")]
            {
                use ort::ep::{
                    CoreML,
                    coreml::{ComputeUnits, ModelFormat},
                };
                use sha2::{Digest, Sha256};
                // MLProgram fails on MOSS dynamic shape partitions with ORT 1.28.
                // Isolate compiled partitions by format and native ORT version.
                let cache = cache.join(format!(
                    "neural-network-ort-{:x}",
                    Sha256::digest(ort::info().as_bytes())
                ));
                std::fs::create_dir_all(&cache)?;
                builder
                    .with_execution_providers([CoreML::default()
                        .with_model_format(ModelFormat::NeuralNetwork)
                        .with_compute_units(ComputeUnits::All)
                        .with_static_input_shapes(true)
                        .with_model_cache_dir(cache.to_string_lossy())
                        .build()
                        .error_on_failure()])
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            }
            #[cfg(not(feature = "ort-coreml"))]
            anyhow::bail!("CoreML feature is not compiled");
        }
        Device::Cuda => {
            #[cfg(feature = "ort-cuda")]
            {
                builder
                    .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            }
            #[cfg(not(feature = "ort-cuda"))]
            anyhow::bail!("CUDA feature is not compiled");
        }
        Device::Metal => anyhow::bail!("Metal requires a Candle adapter; ORT does not support it"),
        Device::Auto => anyhow::bail!("auto requires calibration before constructing a session"),
    };
    let _ = cache;
    builder
        .commit_from_file(path)
        .map_err(|e| anyhow::anyhow!("{e}"))
}
