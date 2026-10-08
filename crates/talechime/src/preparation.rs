//! Composition and calibration belong to the program, not the model-free core.
use crate::resources::Resources;
use std::rc::Rc;
use tokio::sync::mpsc;
use tts_core::backend::Backend;
use tts_protocol::{Config, Event};
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
))]
mod synthesis;

#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
))]
pub fn validate_device(
    device: tts_protocol::Device,
    backend: &str,
    model: Option<&str>,
    resources: &Resources,
) -> anyhow::Result<()> {
    if device == tts_protocol::Device::Auto {
        return Ok(());
    }
    let available = resources.available_devices_for(backend, model);
    anyhow::ensure!(
        available.contains(&device),
        "device {device:?} is unavailable for {backend}; compiled {:?}, available {:?}",
        resources.compiled_devices_for(backend, model),
        available
    );
    Ok(())
}

pub async fn prepare(
    resources: Resources,
    config: Config,
    progress: mpsc::Sender<Event>,
) -> anyhow::Result<Rc<dyn Backend>> {
    #[cfg(not(any(
        feature = "moss",
        feature = "qwen",
        feature = "voxcpm",
        feature = "omnivoice",
    )))]
    {
        let _ = (resources, config, progress);
        anyhow::bail!("no synthesis backend is compiled");
    }
    #[cfg(any(
        feature = "moss",
        feature = "qwen",
        feature = "voxcpm",
        feature = "omnivoice",
    ))]
    {
        validate_device(
            config.tts_device,
            &config.backend,
            config.model.as_deref(),
            &resources,
        )?;
        let backend = synthesis::prepare(&resources, &config, &progress).await?;
        let _ = progress
            .send(Event::ResourceState {
                stage: "模型就绪".into(),
                resource: config.backend.clone(),
            })
            .await;
        Ok(backend)
    }
}
