//! Model discovery without preparing resources or opening audio.
use crate::{Capabilities, Device, EngineError};
use std::path::PathBuf;

/// One compiled model, its voice/parameter capabilities and usable devices.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
    pub capabilities: Capabilities,
    /// Devices available on this host, distinct from compiled device support.
    pub available_devices: Vec<Device>,
}

/// List models and their current voice catalogs without downloading weights.
/// An empty list means no synthesis backend was compiled into this build.
pub fn model_catalog(resources: impl Into<PathBuf>) -> Result<Vec<ModelInfo>, EngineError> {
    let registry =
        talechime_backends::Registry::new(Some(resources.into())).map_err(EngineError::Prepare)?;
    let models = registry.catalog().map_err(EngineError::Prepare)?;
    Ok(models
        .into_iter()
        .map(|capabilities| {
            let available_devices = registry
                .available_devices_for(&capabilities.backend, capabilities.model.as_deref());
            ModelInfo {
                capabilities,
                available_devices,
            }
        })
        .collect())
}
