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
#[cfg(feature = "kokoro")]
mod kokoro;
#[cfg(feature = "moss")]
mod moss;
#[cfg(feature = "omnivoice")]
mod omnivoice;
mod preparation;
mod reference;
use reference::ReferenceEntry;
pub use reference::ReferenceVoice;
#[cfg(feature = "qwen")]
mod qwen;
#[cfg(feature = "voxcpm")]
mod voxcpm;

pub use preparation::PrepareOptions;
type Preparing<'a> = Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>>;
type Loading<'a> = Pin<Box<dyn Future<Output = anyhow::Result<Rc<dyn Backend>>> + 'a>>;
type Installing<'a> = Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>>;
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
    /// Physical reference payload; materialization ownership is tracked separately.
    fn reference_entry(&self, _root: &Path, _id: &str) -> anyhow::Result<ReferenceEntry> {
        anyhow::bail!("this model does not support reference voices")
    }
    /// Validate and install a reference recording for later synthesis.
    /// Directory-backed stores complete synchronously in milliseconds.
    fn install_reference(
        &self,
        _root: &Path,
        _wav: &Path,
        _id: &str,
        _name: &str,
        _transcript: &str,
    ) -> anyhow::Result<()> {
        anyhow::bail!("this model does not support reference voices")
    }
    /// Asynchronous install for adapters needing background work (codec encoding).
    fn install_reference_async<'a>(
        &'a self,
        root: &'a Path,
        wav: &'a Path,
        id: &'a str,
        name: &'a str,
        transcript: &'a str,
    ) -> Installing<'a> {
        Box::pin(async move { self.install_reference(root, wav, id, name, transcript) })
    }
}
struct Registration {
    id: &'static str,
    default_model: &'static str,
    models: fn() -> Vec<Rc<dyn ModelAdapter>>,
}
static ADAPTERS: &[Registration] = &[
    #[cfg(feature = "kokoro")]
    kokoro::REGISTRATION,
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
    /// Engine info for every design engine compiled into this build.
    pub fn designers(&self) -> Vec<talechime_core::design::EngineInfo> {
        crate::design::designers(&self.root)
    }
    /// Materialize a library asset; unmarked custom references are never replaced.
    pub async fn install_reference(
        &self,
        backend: &str,
        model: Option<&str>,
        voice: ReferenceVoice<'_>,
    ) -> anyhow::Result<()> {
        let adapter = self.adapter(backend, model)?;
        reference::install(adapter.as_ref(), &self.root, voice).await
    }
    /// Check both the asset fingerprint and the installed payload integrity.
    pub fn reference_fresh(
        &self,
        backend: &str,
        model: Option<&str>,
        voice_id: &str,
        fingerprint: &str,
    ) -> bool {
        talechime_core::voices::library::validate_id(voice_id).is_ok()
            && self
                .adapter(backend, model)
                .ok()
                .and_then(|adapter| {
                    adapter
                        .reference_entry(&self.root, &format!("custom:{voice_id}"))
                        .ok()
                })
                .is_some_and(|entry| entry.fresh(voice_id, fingerprint))
    }
    pub fn feature_exists(feature: &str) -> bool {
        matches!(
            feature,
            "kokoro"
                | "kokoro-metal"
                | "kokoro-g2pw"
                | "kokoro-cuda"
                | "moss"
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

    fn adapter(&self, backend: &str, model: Option<&str>) -> anyhow::Result<Rc<dyn ModelAdapter>> {
        let registration = ADAPTERS
            .iter()
            .find(|entry| entry.id == backend)
            .ok_or_else(|| anyhow::anyhow!("backend {backend} is unavailable in this build"))?;
        let id = model.unwrap_or(registration.default_model);
        let adapter = (registration.models)()
            .into_iter()
            .find(|adapter| adapter.id() == id)
            .ok_or_else(|| anyhow::anyhow!("unsupported or retired model {backend}/{id}; supported models in this build: {}", (registration.models)().iter().map(|model| model.id()).collect::<Vec<_>>().join(", ")))?;
        Ok(adapter)
    }
    pub fn resolve(&self, backend: &str, model: Option<&str>) -> anyhow::Result<ResolvedModel> {
        let adapter = self.adapter(backend, model)?;
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
    #[cfg(feature = "qwen")]
    #[test]
    fn qwen_catalog_hides_the_design_generator_and_rejects_it_for_narration() -> anyhow::Result<()>
    {
        let root = tempfile::tempdir()?;
        let registry = Registry::new(Some(root.path().into()))?;
        let models = registry
            .catalog()?
            .into_iter()
            .filter(|caps| caps.backend == "qwen")
            .map(|caps| caps.model.unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            models,
            vec![
                "0.6b-customvoice",
                "1.7b-customvoice",
                "0.6b-base",
                "1.7b-base"
            ]
        );
        let error = match registry.resolve("qwen", Some("1.7b-voicedesign")) {
            Ok(_) => panic!("design-only model accepted for narration"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("unsupported or retired model"));
        assert!(error.to_string().contains("1.7b-voicedesign"));
        assert_eq!(std::fs::read_dir(root.path())?.count(), 0);
        Ok(())
    }
    /// One second of quiet mono audio; valid for reference validation.
    #[cfg(feature = "qwen")]
    fn write_reference_wav(path: &std::path::Path, seconds: u32) {
        write_reference_wav_pitch(path, seconds, 400.0)
    }
    #[cfg(feature = "qwen")]
    fn write_reference_wav_pitch(path: &std::path::Path, seconds: u32, frequency: f32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for index in 0..seconds * 24000 {
            let value =
                ((index as f32 / 24000.0) * frequency * std::f32::consts::PI).sin() * 8000.0;
            writer.write_sample(value as i16).unwrap();
        }
        writer.finalize().unwrap();
    }
    /// The reference.wav location for a qwen base model voice, via public APIs.
    #[cfg(feature = "qwen")]
    fn talechime_backends_qwen_voice_wav(root: &std::path::Path, id: &str) -> std::path::PathBuf {
        use crate::qwen::{models::Model, voice_store};
        voice_store(&Model::Base17.directory(root), Model::Base17)
            .unwrap()
            .path(&format!("custom:{id}"))
            .unwrap()
            .join("reference.wav")
    }
    #[cfg(feature = "qwen")]
    fn reference_asset(id: &str, wav: &[u8]) -> talechime_core::voices::library::VoiceAsset {
        use talechime_core::voices::library::{VoiceAsset, VoiceSource, fingerprint};
        VoiceAsset {
            id: id.into(),
            name: "主角".into(),
            transcript: "试音转写".into(),
            description: None,
            source: VoiceSource::Imported,
            created_at_unix: 0,
            fingerprint: fingerprint("试音转写", None, wav),
        }
    }
    #[cfg(feature = "qwen")]
    #[tokio::test]
    async fn install_reference_materializes_reuses_and_refreshes() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let registry = Registry::new(Some(root.path().into()))?;
        let wav = root.path().join("reference.wav");
        write_reference_wav(&wav, 1);
        let bytes = std::fs::read(&wav)?;
        let asset = reference_asset("hero", &bytes);
        let voice = ReferenceVoice {
            asset: &asset,
            wav: &bytes,
        };
        registry
            .install_reference("qwen", Some("1.7b-base"), voice)
            .await?;
        assert!(registry.reference_fresh("qwen", Some("1.7b-base"), "hero", &asset.fingerprint));
        let voice_wav = talechime_backends_qwen_voice_wav(root.path(), "hero");
        let before = std::fs::read(&voice_wav)?;
        registry
            .install_reference("qwen", Some("1.7b-base"), voice)
            .await?;
        assert_eq!(std::fs::read(&voice_wav)?, before);
        write_reference_wav_pitch(&wav, 1, 800.0);
        let changed = std::fs::read(&wav)?;
        let changed_asset = reference_asset("hero", &changed);
        registry
            .install_reference(
                "qwen",
                Some("1.7b-base"),
                ReferenceVoice {
                    asset: &changed_asset,
                    wav: &changed,
                },
            )
            .await?;
        assert!(registry.reference_fresh(
            "qwen",
            Some("1.7b-base"),
            "hero",
            &changed_asset.fingerprint
        ));
        assert_ne!(std::fs::read(&voice_wav)?, before);
        // Damage must not make catalog enumeration prevent repairing the entry.
        std::fs::write(voice_wav.parent().unwrap().join("voice.json"), b"broken")?;
        assert!(!registry.reference_fresh(
            "qwen",
            Some("1.7b-base"),
            "hero",
            &changed_asset.fingerprint
        ));
        registry
            .install_reference(
                "qwen",
                Some("1.7b-base"),
                ReferenceVoice {
                    asset: &changed_asset,
                    wav: &changed,
                },
            )
            .await?;
        assert!(registry.reference_fresh(
            "qwen",
            Some("1.7b-base"),
            "hero",
            &changed_asset.fingerprint
        ));
        Ok(())
    }
    #[cfg(feature = "qwen")]
    #[tokio::test]
    async fn install_reference_rejects_duration_limit_and_non_cloning_models() -> anyhow::Result<()>
    {
        let root = tempfile::tempdir()?;
        let registry = Registry::new(Some(root.path().into()))?;
        let wav = root.path().join("long.wav");
        write_reference_wav(&wav, 20);
        let bytes = std::fs::read(&wav)?;
        let asset = reference_asset("long", &bytes);
        let voice = ReferenceVoice {
            asset: &asset,
            wav: &bytes,
        };
        let error = registry
            .install_reference("qwen", Some("1.7b-base"), voice)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("1..15 seconds"));
        let error = registry
            .install_reference("qwen", Some("1.7b-customvoice"), voice)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("0.6b-base or 1.7b-base"));
        Ok(())
    }
}
