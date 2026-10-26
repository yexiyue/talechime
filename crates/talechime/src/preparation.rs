//! CLI configuration selects policy; adapters own preparation and recovery.
use std::rc::Rc;
use talechime_backends::Registry;
use talechime_core::backend::Backend;
use talechime_protocol::{Config, Event};
use tokio::sync::mpsc;
pub fn validate_device(
    device: talechime_protocol::Device,
    backend: &str,
    model: Option<&str>,
    resources: &Registry,
) -> anyhow::Result<()> {
    if device == talechime_protocol::Device::Auto {
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
    resources: Registry,
    config: Config,
    progress: mpsc::Sender<Event>,
) -> anyhow::Result<Rc<dyn Backend>> {
    let model = resources.resolve(&config.backend, config.model.as_deref())?;
    let backend = model
        .prepare(
            talechime_backends::PrepareOptions {
                device: config.tts_device,
                calibrate: true,
                recovery: true,
                voice: Some(config.voice),
            },
            progress.clone(),
        )
        .await?;
    let _ = progress
        .send(Event::ResourceState {
            stage: "模型就绪".into(),
            resource: config.backend,
        })
        .await;
    Ok(backend)
}

/// Explicit preparation bundle; enabling readback here never enables the gate by itself.
pub struct PreparedModels {
    pub backend: Rc<dyn Backend>,
    pub verifier: Option<Rc<talechime::Verifier>>,
}
pub async fn prepare_models(
    resources: Registry,
    config: Config,
    progress: mpsc::Sender<Event>,
    readback: bool,
) -> anyhow::Result<PreparedModels> {
    #[cfg(not(feature = "asr"))]
    if readback {
        anyhow::bail!("readback models require the asr build feature");
    }
    let directory = resources.root().to_path_buf();
    let backend = prepare(resources, config, progress.clone()).await?;
    #[cfg(feature = "asr")]
    let verifier = if readback {
        Some(
            talechime_backends::asr::prepare(
                &talechime_backends::asr::ReadbackModelOptions::new(directory),
                progress,
            )
            .await?,
        )
    } else {
        None
    };
    #[cfg(not(feature = "asr"))]
    let verifier = {
        let _ = (directory, progress);
        None
    };
    Ok(PreparedModels { backend, verifier })
}
