use super::*;
use crate::devices::calibration;

/// Preparation policy shared by explicit library and automatic CLI selection.
#[derive(Debug, Clone)]
pub struct PrepareOptions {
    pub device: Device,
    pub calibrate: bool,
    pub recovery: bool,
    /// Voice to use for calibration; defaults to the resolved model's default voice.
    pub voice: Option<String>,
}
impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            calibrate: false,
            recovery: false,
            voice: None,
        }
    }
}
impl ResolvedModel {
    pub async fn prepare(
        &self,
        options: PrepareOptions,
        progress: mpsc::Sender<Event>,
    ) -> anyhow::Result<Rc<dyn Backend>> {
        let available = self.available_devices();
        anyhow::ensure!(
            options.device == Device::Auto || available.contains(&options.device),
            "device {:?} is unavailable for {}/{}",
            options.device,
            self.capabilities.backend,
            self.adapter.id()
        );
        let voice = options
            .voice
            .as_deref()
            .unwrap_or(&self.capabilities.default_voice);
        let candidate = available.iter().copied().find(|d| *d != Device::Cpu);
        let mut selected = if options.device == Device::Auto {
            available
                .iter()
                .copied()
                .find(|device| *device == Device::Cpu)
                .or(candidate)
                .ok_or_else(|| anyhow::anyhow!("no available device for this model"))?
        } else {
            options.device
        };
        let key =
            (options.device == Device::Auto && options.calibrate && self.adapter.calibrates())
                .then(|| {
                    calibration::key(
                        "tts",
                        &format!(
                            "{}:{}:{}",
                            self.capabilities.backend,
                            self.adapter.id(),
                            self.revision()
                        ),
                    )
                });
        let cached = key
            .as_deref()
            .and_then(|key| calibration::cached_for(&self.root, "tts", key, &available));
        if let Some(record) = &cached {
            selected = record.device;
        }
        self.adapter.prepare(&self.root, progress.clone()).await?;
        let mut backend = match self
            .adapter
            .load(&self.root, progress.clone(), selected, false)
            .await
        {
            Ok(backend) => backend,
            Err(error)
                if options.device == Device::Auto
                    && selected != Device::Cpu
                    && available.contains(&Device::Cpu) =>
            {
                selected = Device::Cpu;
                let _ = progress
                    .send(self.device_status(
                        selected,
                        Some(format!("provider initialization failed: {error}")),
                    ))
                    .await;
                self.adapter
                    .load(&self.root, progress.clone(), Device::Cpu, false)
                    .await?
            }
            Err(error) => return Err(error),
        };
        if options.device == Device::Auto
            && cached.is_none()
            && available.contains(&Device::Cpu)
            && options.calibrate
            && self.adapter.calibrates()
            && let Some(device) = candidate
        {
            let _ = progress
                .send(self.device_status(
                    Device::Cpu,
                    Some("calibrating: 3 warmups + 5 measurements".into()),
                ))
                .await;
            let evaluation = async {
                let accelerated = self
                    .adapter
                    .load(&self.root, progress.clone(), device, false)
                    .await?;
                let cpu = calibration::synthesis(&*backend, voice).await?;
                let gpu = calibration::synthesis(&*accelerated, voice).await?;
                Ok::<_, anyhow::Error>((accelerated, cpu, gpu))
            }
            .await;
            let measured = evaluation.is_ok();
            let reason = match evaluation {
                Ok((accelerated, cpu, gpu)) => {
                    if gpu.improves(cpu) {
                        selected = device;
                        backend = accelerated;
                    }
                    format!(
                        "CPU {:.1} ms / first {:.1} ms; {device:?} {:.1} ms / first {:.1} ms",
                        cpu.total_ms, cpu.first_ms, gpu.total_ms, gpu.first_ms
                    )
                }
                Err(error) => format!("accelerator calibration failed: {error}"),
            };
            if measured && let Some(key) = key {
                calibration::save(
                    &self.root,
                    "tts",
                    &calibration::Record {
                        key,
                        device: selected,
                        reason: reason.clone(),
                    },
                )?;
            }
            let _ = progress
                .send(self.device_status(selected, Some(reason)))
                .await;
        } else {
            let _ = progress
                .send(self.device_status(selected, cached.map(|r| r.reason)))
                .await;
        }
        if options.device == Device::Auto
            && options.recovery
            && selected != Device::Cpu
            && self.adapter.supports_recovery()
        {
            drop(backend);
            backend = self
                .adapter
                .load(&self.root, progress, selected, true)
                .await?;
        }
        Ok(backend)
    }
}

#[cfg(test)]
#[path = "preparation_tests.rs"]
mod tests;
