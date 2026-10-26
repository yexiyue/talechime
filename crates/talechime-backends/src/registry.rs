//! Static backend registration and one preparation path for library and CLI hosts.
use std::{
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    rc::Rc,
};
use talechime_core::backend::Backend;
use talechime_protocol::{Capabilities, Config, Device, Event};
use tokio::sync::mpsc;

#[cfg(feature = "cosyvoice3")]
mod cosyvoice3;
#[cfg(feature = "indextts")]
mod indextts;
#[cfg(feature = "moss")]
mod moss;
#[cfg(feature = "omnivoice")]
mod omnivoice;
mod preparation;
#[cfg(feature = "qwen")]
mod qwen;
#[cfg(feature = "voxcpm")]
mod voxcpm;

pub use preparation::PrepareOptions;
type Preparing<'a> = Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>>;
type Loading<'a> = Pin<Box<dyn Future<Output = anyhow::Result<Rc<dyn Backend>>> + 'a>>;
trait ModelAdapter {
    fn id(&self) -> &'static str;
    fn capabilities(&self, root: &Path) -> anyhow::Result<Capabilities>;
    fn compiled_devices(&self) -> Vec<Device>;
    fn available_devices(&self) -> Vec<Device>;
    fn revision(&self) -> &'static str;
    fn calibrates(&self) -> bool {
        true
    }
    fn supports_recovery(&self) -> bool {
        false
    }
    fn prepare<'a>(&'a self, root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a>;
    fn load<'a>(
        &'a self,
        root: &'a Path,
        progress: mpsc::Sender<Event>,
        device: Device,
        recovery: bool,
    ) -> Loading<'a>;
}
struct Registration {
    id: &'static str,
    default_model: &'static str,
    models: fn() -> Vec<Rc<dyn ModelAdapter>>,
}
static ADAPTERS: &[Registration] = &[
    #[cfg(feature = "moss")]
    moss::REGISTRATION,
    #[cfg(feature = "voxcpm")]
    voxcpm::REGISTRATION,
    #[cfg(feature = "omnivoice")]
    omnivoice::REGISTRATION,
    #[cfg(feature = "cosyvoice3")]
    cosyvoice3::REGISTRATION,
    #[cfg(feature = "indextts")]
    indextts::REGISTRATION,
    #[cfg(feature = "qwen")]
    qwen::REGISTRATION,
];

/// Exact model selection; every subsequent preparation step uses the same adapter.
#[derive(Clone)]
pub struct ResolvedModel {
    adapter: Rc<dyn ModelAdapter>,
    root: PathBuf,
    capabilities: Capabilities,
}
impl ResolvedModel {
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
    pub fn revision(&self) -> &'static str {
        self.adapter.revision()
    }
    pub fn compiled_devices(&self) -> Vec<Device> {
        self.adapter.compiled_devices()
    }
    pub fn available_devices(&self) -> Vec<Device> {
        self.adapter.available_devices()
    }
    pub fn device_status(&self, selected: Device, reason: Option<String>) -> Event {
        Event::DeviceStatus {
            component: "tts".into(),
            compiled: self.compiled_devices(),
            available: self.available_devices(),
            selected,
            reason,
        }
    }
}

#[derive(Clone)]
pub struct Registry {
    root: PathBuf,
}
impl Registry {
    pub fn new(root: Option<PathBuf>) -> anyhow::Result<Self> {
        Ok(Self {
            root: match root {
                Some(root) => root,
                None => talechime_core::paths::AppPaths::user_default()?.resources(),
            },
        })
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn feature_exists(feature: &str) -> bool {
        matches!(
            feature,
            "moss"
                | "moss-candle"
                | "moss-candle-cuda"
                | "moss-candle-metal"
                | "moss-nano-candle"
                | "moss-nano-candle-cuda"
                | "moss-nano-candle-metal"
                | "qwen"
                | "qwen-cuda"
                | "qwen-metal"
                | "voxcpm"
                | "voxcpm-cuda"
                | "voxcpm-metal"
                | "omnivoice"
                | "omnivoice-cuda"
                | "omnivoice-metal"
                | "cosyvoice3"
                | "cosyvoice3-cuda"
                | "cosyvoice3-metal"
                | "indextts"
                | "indextts-cuda-cudnn"
                | "indextts-metal"
        )
    }

    pub fn resolve(&self, backend: &str, model: Option<&str>) -> anyhow::Result<ResolvedModel> {
        let registration = ADAPTERS
            .iter()
            .find(|entry| entry.id == backend)
            .ok_or_else(|| anyhow::anyhow!("backend {backend} is unavailable in this build"))?;
        let id = model.unwrap_or(registration.default_model);
        let adapter = (registration.models)()
            .into_iter()
            .find(|adapter| adapter.id() == id)
            .ok_or_else(|| anyhow::anyhow!("unsupported or retired model {backend}/{id}; supported models in this build: {}", (registration.models)().iter().map(|model| model.id()).collect::<Vec<_>>().join(", ")))?;
        let mut capabilities = adapter.capabilities(&self.root)?;
        capabilities.model = Some(adapter.id().into());
        capabilities.compiled_devices = adapter.compiled_devices();
        Ok(ResolvedModel {
            adapter,
            capabilities,
            root: self.root.clone(),
        })
    }
    pub fn catalog(&self) -> anyhow::Result<Vec<Capabilities>> {
        let mut catalog = Vec::new();
        for registration in ADAPTERS {
            for model in (registration.models)() {
                let mut caps = model.capabilities(&self.root)?;
                caps.model = Some(model.id().into());
                caps.compiled_devices = model.compiled_devices();
                catalog.push(caps);
            }
        }
        Ok(catalog)
    }
    pub fn capabilities(&self, backend: &str) -> anyhow::Result<Capabilities> {
        self.capabilities_for(backend, None)
    }
    pub fn capabilities_for(
        &self,
        backend: &str,
        model: Option<&str>,
    ) -> anyhow::Result<Capabilities> {
        Ok(self.resolve(backend, model)?.capabilities)
    }
    pub fn compiled_devices_for(&self, backend: &str, model: Option<&str>) -> Vec<Device> {
        self.resolve(backend, model)
            .map_or_else(|_| vec![], |model| model.compiled_devices())
    }
    pub fn available_devices_for(&self, backend: &str, model: Option<&str>) -> Vec<Device> {
        self.resolve(backend, model)
            .map_or_else(|_| vec![], |model| model.available_devices())
    }
    pub fn device_status_for(
        &self,
        backend: &str,
        model: Option<&str>,
        selected: Device,
        reason: Option<String>,
    ) -> Event {
        Event::DeviceStatus {
            component: "tts".into(),
            compiled: self.compiled_devices_for(backend, model),
            available: self.available_devices_for(backend, model),
            selected,
            reason,
        }
    }
    /// Application policy for a new settings file, independent of adapter registration.
    pub fn default_config(&self) -> anyhow::Result<Config> {
        #[cfg(feature = "qwen")]
        if let Some(device) = crate::qwen::available_devices()
            .into_iter()
            .find(|d| matches!(d, Device::Cuda | Device::Metal))
        {
            let caps = self.capabilities_for("qwen", Some("1.7b-customvoice"))?;
            return Ok(Config {
                backend: caps.backend,
                model: caps.model,
                voice: caps.default_voice,
                tts_device: device,
                ..Default::default()
            });
        }
        if let Ok(model) = self.resolve("moss", None) {
            let device = if model.compiled_devices().contains(&Device::Cpu) {
                Device::Cpu
            } else {
                Device::Auto
            };
            let caps = model.capabilities;
            return Ok(Config {
                backend: caps.backend,
                model: caps.model,
                voice: caps.default_voice,
                tts_device: device,
                ..Default::default()
            });
        }
        anyhow::bail!(
            "no default synthesis backend is compiled; select an available backend explicitly"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn registered_models_resolve_defaults_and_exact_identities_without_writes() -> anyhow::Result<()>
    {
        let root = tempfile::tempdir()?;
        let registry = Registry::new(Some(root.path().to_path_buf()))?;
        let mut identities = std::collections::HashSet::new();
        for registration in ADAPTERS {
            let default = registry.resolve(registration.id, None)?;
            assert_eq!(default.adapter.id(), registration.default_model);
            assert!(
                registry
                    .resolve(registration.id, Some("unknown-model"))
                    .is_err()
            );
            for adapter in (registration.models)() {
                let model = registry.resolve(registration.id, Some(adapter.id()))?;
                assert!(identities.insert((registration.id, adapter.id())));
                assert!(!model.revision().is_empty());
                assert_eq!(model.capabilities().backend, registration.id);
                assert_eq!(model.capabilities().model.as_deref(), Some(adapter.id()));
                assert_eq!(
                    model.capabilities().compiled_devices,
                    model.compiled_devices()
                );
            }
        }
        assert_eq!(registry.catalog()?.len(), identities.len());
        assert!(registry.resolve("unknown-backend", None).is_err());
        assert_eq!(std::fs::read_dir(root.path())?.count(), 0);
        Ok(())
    }
    #[cfg(feature = "moss")]
    #[test]
    fn moss_catalog_contains_only_retained_models_and_rejects_retired_ids() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let registry = Registry::new(Some(root.path().into()))?;
        let models = registry
            .catalog()?
            .into_iter()
            .filter(|caps| caps.backend == "moss")
            .map(|caps| caps.model.unwrap())
            .collect::<Vec<_>>();
        let expected = if cfg!(feature = "moss-nano-candle") {
            vec!["nano-candle", "realtime-1.7b"]
        } else {
            vec!["realtime-1.7b"]
        };
        assert_eq!(models, expected);
        for retired in ["nano", "local-1b", "local-1.7b"] {
            let error = match registry.resolve("moss", Some(retired)) {
                Ok(_) => panic!("retired model accepted"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("unsupported or retired model"));
            assert!(error.to_string().contains(retired));
        }
        assert_eq!(std::fs::read_dir(root.path())?.count(), 0);
        Ok(())
    }
}
