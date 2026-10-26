//! Voice assets for hosts: design engines, one cross-model library, and
//! explicit or automatic materialization into synthesis models.
//!
//! The automatic tier ([`VoiceStudio::design`], library voice ids accepted by
//! `Engine` synthesis entries) is a thin composition of the explicit tier
//! ([`VoiceStudio::design_with`], [`VoiceStudio::install_voice_for`]); both
//! share the same internal path. Errors are display-safe: automatic-tier
//! messages never name concrete backend models.

use crate::Event;
use std::path::{Path, PathBuf};
use talechime_backends::Registry;
use talechime_core::design::EngineInfo;
use talechime_core::voices::library::{VoiceAsset, VoiceLibrary, VoiceSource};
use talechime_protocol::Device;

/// Default spoken trial text recorded into designed references.
pub const DEFAULT_TRIAL_TEXT: &str = "你好，欢迎收听今天的故事。山风吹过松林，星光照亮归途。";

/// Library voice identity; distinct type so hosts cannot mix it with
/// model-local `custom:` strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct VoiceId(String);
impl VoiceId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl TryFrom<String> for VoiceId {
    type Error = VoiceStudioError;
    fn try_from(id: String) -> Result<Self, Self::Error> {
        talechime_core::voices::library::validate_id(&id).map_err(VoiceStudioError::from)?;
        Ok(Self(id))
    }
}
impl std::str::FromStr for VoiceId {
    type Err = VoiceStudioError;
    fn from_str(id: &str) -> Result<Self, Self::Err> {
        Self::try_from(id.to_owned())
    }
}
impl From<VoiceId> for String {
    fn from(id: VoiceId) -> Self {
        id.0
    }
}
impl std::ops::Deref for VoiceId {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}
impl std::fmt::Display for VoiceId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Voice studio failure. Automatic design hides native model details in its source.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct VoiceStudioError {
    message: String,
    #[source]
    source: Option<anyhow::Error>,
}
impl VoiceStudioError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }
}
impl From<anyhow::Error> for VoiceStudioError {
    fn from(error: anyhow::Error) -> Self {
        Self {
            message: error.to_string(),
            source: Some(error),
        }
    }
}

/// Explicit design request; defaults come from the chosen engine.
#[derive(Debug, Clone)]
pub struct DesignOptions {
    pub description: String,
    /// Explicit engine id; automatic policy picks when absent.
    pub engine: Option<String>,
    /// Library id; a unique slug is generated when absent.
    pub id: Option<String>,
    /// Display name; derived from the description when absent.
    pub name: Option<String>,
    /// Spoken trial text recorded into the reference.
    pub text: Option<String>,
    pub device: Device,
    pub seed: Option<u64>,
}
impl DesignOptions {
    /// The automatic tier's exact request shape.
    pub fn auto(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            engine: None,
            id: None,
            name: None,
            text: None,
            device: Device::Auto,
            seed: None,
        }
    }
}

/// Default automatic engine policy: prefer engines reusing already-prepared
/// narration models (zero extra download), then any ready engine, then a dedicated design engine with a usable device.
pub(crate) fn pick_engine(infos: &[EngineInfo]) -> Option<&EngineInfo> {
    let usable = |info: &&EngineInfo| !info.devices.is_empty();
    infos
        .iter()
        .filter(usable)
        .find(|info| info.ready && info.reuses_synthesis_model)
        .or_else(|| infos.iter().filter(usable).find(|info| info.ready))
        .or_else(|| {
            infos
                .iter()
                .filter(usable)
                .find(|info| !info.reuses_synthesis_model)
        })
}

#[derive(Clone)]
pub struct VoiceStudio {
    registry: Registry,
    library: VoiceLibrary,
    /// Injected engines overriding the compiled static registry (tests).
    designers: Option<Vec<talechime_backends::design::DesignerHandle>>,
}
impl VoiceStudio {
    /// Open the studio over caller-owned roots; defaults live under the app
    /// root (`~/.talechime`), never inside model resources.
    pub fn open(
        resources: Option<PathBuf>,
        voices: Option<PathBuf>,
    ) -> Result<Self, VoiceStudioError> {
        let resources = match resources {
            Some(root) => root,
            None => talechime_core::paths::AppPaths::user_default()
                .map_err(|error| VoiceStudioError::new(error.to_string()))?
                .resources(),
        };
        let voices = match voices {
            Some(root) => root,
            None => talechime_core::paths::AppPaths::user_default()
                .map_err(|error| VoiceStudioError::new(error.to_string()))?
                .voices(),
        };
        Self::at(resources, voices)
    }
    pub fn at(
        resources: impl Into<PathBuf>,
        voices: impl Into<PathBuf>,
    ) -> Result<Self, VoiceStudioError> {
        Ok(Self {
            registry: Registry::new(Some(resources.into())).map_err(VoiceStudioError::from)?,
            library: VoiceLibrary::open(voices).map_err(VoiceStudioError::from)?,
            designers: None,
        })
    }
    /// Replace the design engine list. Hosts embed custom engines the same way
    /// they inject custom synthesis backends; tests use it to avoid downloads.
    pub fn set_designers(&mut self, engines: Vec<talechime_backends::design::DesignerHandle>) {
        self.designers = Some(engines);
    }
    pub fn list(&self) -> Result<Vec<VoiceAsset>, VoiceStudioError> {
        self.library.list().map_err(VoiceStudioError::from)
    }
    pub fn get(&self, id: &str) -> Result<VoiceAsset, VoiceStudioError> {
        self.library.get(id).map_err(VoiceStudioError::from)
    }
    pub fn rename(&self, id: &str, name: &str) -> Result<VoiceAsset, VoiceStudioError> {
        self.library
            .rename(id, name)
            .map_err(VoiceStudioError::from)
    }
    pub fn remove(&self, id: &str) -> Result<(), VoiceStudioError> {
        self.library.remove(id).map_err(VoiceStudioError::from)
    }
    /// Export one voice as a single self-contained file.
    pub fn export(&self, id: &str, path: &Path) -> Result<(), VoiceStudioError> {
        self.library
            .export(id, path)
            .map_err(VoiceStudioError::from)
    }
    /// Import a bundle file; `new_id` overrides the recorded id.
    pub fn import_bundle(
        &self,
        path: &Path,
        new_id: Option<&str>,
    ) -> Result<VoiceAsset, VoiceStudioError> {
        self.library
            .import_bundle(path, new_id)
            .map_err(VoiceStudioError::from)
    }
    /// Install a reference recording as a new library voice.
    pub fn import_voice(
        &self,
        id: Option<&str>,
        name: &str,
        wav: &Path,
        transcript: &str,
    ) -> Result<VoiceAsset, VoiceStudioError> {
        let bytes = std::fs::read(wav).map_err(|error| {
            VoiceStudioError::new(format!("cannot read {}: {error}", wav.display()))
        })?;
        let id = self.unique_id(id)?;
        self.library
            .create(&id, name, &bytes, transcript, None, VoiceSource::Imported)
            .map_err(VoiceStudioError::from)
    }
    /// Design engines compiled into this build with readiness and cost.
    pub fn engines(&self) -> Vec<EngineInfo> {
        match &self.designers {
            Some(engines) => engines
                .iter()
                .map(|engine| engine.info(self.registry.root()))
                .collect(),
            None => self.registry.designers(),
        }
    }
    /// Automatic design: policy-chosen engine, engine-default text and device.
    pub async fn design(
        &self,
        description: impl Into<String>,
    ) -> Result<VoiceAsset, VoiceStudioError> {
        self.design_with(DesignOptions::auto(description), |_| {})
            .await
    }
    /// Explicit design with a closure progress sink. See [`Self::design_with_channel`].
    pub async fn design_with(
        &self,
        options: DesignOptions,
        mut progress: impl FnMut(Event),
    ) -> Result<VoiceAsset, VoiceStudioError> {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(16);
        let design = self.design_with_channel(options, sender);
        tokio::pin!(design);
        let mut progress_open = true;
        loop {
            tokio::select! {
                result = &mut design => {
                    while let Ok(event) = receiver.try_recv() {
                        progress(event);
                    }
                    return result;
                }
                event = receiver.recv(), if progress_open => {
                    match event {
                        Some(event) => progress(event),
                        None => progress_open = false,
                    }
                }
            }
        }
    }
    /// Explicit design reporting progress into a bounded channel with backpressure.
    /// Dropping the returned future cancels the job without saving a voice.
    pub async fn design_with_channel(
        &self,
        options: DesignOptions,
        progress: tokio::sync::mpsc::Sender<Event>,
    ) -> Result<VoiceAsset, VoiceStudioError> {
        let automatic = options.engine.is_none();
        let infos: Vec<_> = self
            .engines()
            .into_iter()
            .filter(|info| {
                !automatic
                    || options.device == Device::Auto
                    || info.devices.contains(&options.device)
            })
            .collect();
        let chosen = match options.engine.as_deref() {
            Some(id) => {
                let found = infos.iter().find(|info| info.id == id);
                found.ok_or_else(|| {
                    VoiceStudioError::new(format!(
                        "unknown design engine {id}; available: {}",
                        infos.iter().map(|info| info.id).collect::<Vec<_>>().join(", ")
                    ))
                })?
            }
            None => pick_engine(&infos).ok_or_else(|| {
                VoiceStudioError::new(
                    "no design engine is available in this build (check compile features or device support)",
                )
            })?,
        };
        let engine = match &self.designers {
            Some(engines) => engines
                .iter()
                .find(|engine| engine.id() == chosen.id)
                .cloned()
                .ok_or_else(|| VoiceStudioError::new("design engine is unavailable"))?,
            None => talechime_backends::design::designer(chosen.id).ok_or_else(|| {
                VoiceStudioError::new("design engine is unavailable in this build")
            })?,
        };
        let text = options
            .text
            .clone()
            .unwrap_or_else(|| DEFAULT_TRIAL_TEXT.to_owned());
        if let Some(id) = options.id.as_deref() {
            talechime_core::voices::library::validate_id(id).map_err(VoiceStudioError::from)?;
            self.unique_id(Some(id))?;
        }
        let name = options.name.as_deref().unwrap_or(&options.description);
        if name.trim().is_empty() || name.len() > 256 && options.name.is_some() {
            return Err(VoiceStudioError::new(
                "voice name must contain 1..256 bytes",
            ));
        }
        // Input limits are enforced before any weights load or download starts.
        chosen
            .inputs
            .validate(&text, &options.description)
            .map_err(VoiceStudioError::from)?;
        let device = match options.device {
            Device::Auto => chosen
                .devices
                .iter()
                .find(|device| **device != Device::Cpu)
                .or_else(|| chosen.devices.first())
                .cloned()
                .ok_or_else(|| {
                    VoiceStudioError::new(format!(
                        "design engine {} has no usable device",
                        chosen.id
                    ))
                })?,
            requested => {
                if !chosen.devices.contains(&requested) {
                    return Err(VoiceStudioError::new(format!(
                        "device {requested:?} is unavailable for design engine {}",
                        chosen.id
                    )));
                }
                requested
            }
        };
        let root = self.registry.root().to_path_buf();
        engine
            .prepare(&root, progress)
            .await
            .map_err(|error| design_error(automatic, error))?;
        let output = engine
            .design(
                &root,
                talechime_core::design::DesignRequest {
                    text: text.clone(),
                    description: options.description.clone(),
                    device,
                    seed: options.seed,
                },
            )
            .await
            .map_err(|error| design_error(automatic, error))?;
        let id = self.unique_id(options.id.as_deref())?;
        let name = options
            .name
            .clone()
            .unwrap_or_else(|| default_name(&options.description));
        self.library
            .create(
                &id,
                &name,
                &output.wav,
                &text,
                Some(options.description.as_str()),
                VoiceSource::Designed(engine.id().to_owned()),
            )
            .map_err(VoiceStudioError::from)
    }
    /// Explicitly materialize a library voice into a synthesis model.
    /// Async because one backend encodes references in the background.
    pub async fn install_voice_for(
        &self,
        backend: &str,
        model: Option<&str>,
        id: &str,
    ) -> Result<(), VoiceStudioError> {
        self.materialize(backend, model, id).await.map(|_| ())
    }

    /// The single materialization path for explicit installs and every synthesis entry.
    pub(crate) async fn materialize(
        &self,
        backend: &str,
        model: Option<&str>,
        voice_id: &str,
    ) -> Result<String, VoiceStudioError> {
        let (asset, wav) = self
            .library
            .load(voice_id)
            .map_err(VoiceStudioError::from)?;
        self.registry
            .install_reference(
                backend,
                model,
                talechime_backends::ReferenceVoice {
                    asset: &asset,
                    wav: &wav,
                },
            )
            .await
            .map_err(VoiceStudioError::from)?;
        Ok(format!("custom:{}", asset.id))
    }

    /// Resolve a model voice or materialize a library id, preserving failures.
    pub async fn resolve_voice(
        &self,
        caps: &talechime_protocol::Capabilities,
        voice: &str,
    ) -> Result<String, VoiceStudioError> {
        if caps.voices.iter().any(|available| available == voice) {
            return Ok(voice.to_owned());
        }
        self.materialize(&caps.backend, caps.model.as_deref(), voice)
            .await
    }

    /// Materialize selected library voices and freeze their model ids and aliases.
    /// Successful installs extend the capability snapshot taken before first use.
    pub async fn voice_snapshot(
        &self,
        caps: &talechime_protocol::Capabilities,
        voices: &[String],
    ) -> Result<talechime_core::VoiceSnapshot, VoiceStudioError> {
        let mut aliases = std::collections::BTreeMap::new();
        let mut resolved = Vec::with_capacity(voices.len());
        let mut capabilities = caps.clone();
        for voice in voices {
            let target = self.resolve_voice(caps, voice).await?;
            if !capabilities.voices.contains(&target) {
                capabilities.voices.push(target.clone());
            }
            aliases.insert(voice.clone(), target.clone());
            resolved.push(target);
        }
        talechime_core::VoiceSnapshot::new(
            &caps.backend,
            caps.model.as_deref(),
            &capabilities,
            resolved,
        )
        .and_then(|snapshot| snapshot.with_aliases(aliases))
        .map_err(|error| VoiceStudioError::new(error.to_string()))
    }
    fn unique_id(&self, requested: Option<&str>) -> Result<String, VoiceStudioError> {
        match requested {
            Some(id) => {
                if self.library.exists(id) {
                    return Err(VoiceStudioError::new(format!(
                        "voice {id} already exists; choose another ID or remove it first"
                    )));
                }
                Ok(id.to_owned())
            }
            None => {
                let mut candidate = "voice".to_owned();
                let mut counter = 2u32;
                while self.library.exists(&candidate) {
                    candidate = format!("voice-{counter}");
                    counter += 1;
                }
                Ok(candidate)
            }
        }
    }
}

fn design_error(automatic: bool, error: anyhow::Error) -> VoiceStudioError {
    if automatic {
        VoiceStudioError {
            message: "voice design failed; check device availability and resources".into(),
            source: Some(error),
        }
    } else {
        VoiceStudioError::from(error)
    }
}

fn default_name(description: &str) -> String {
    let trimmed = description.trim();
    let taken: String = trimmed.chars().take(24).collect();
    if taken.is_empty() {
        "设计音色".to_owned()
    } else {
        taken
    }
}

#[cfg(test)]
mod tests {
    fn test_wav() -> Vec<u8> {
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&38u32.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&24000u32.to_le_bytes());
        wav.extend_from_slice(&48000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&2u32.to_le_bytes());
        wav.extend_from_slice(&0i16.to_le_bytes());
        wav
    }

    use super::*;
    use talechime_core::design::InputConstraints;

    /// Test-only design engine: no weights, instant reference output.
    struct FakeDesigner;
    impl talechime_core::design::VoiceDesigner for FakeDesigner {
        fn id(&self) -> &'static str {
            "fake-designer"
        }
        fn info(&self, _root: &Path) -> EngineInfo {
            EngineInfo {
                id: "fake-designer",
                name: "Fake Designer",
                devices: vec![Device::Cpu],
                inputs: InputConstraints {
                    text_max_chars: 100,
                    description_max_chars: None,
                    description_max_bytes: 4096,
                    forbid_parentheses: false,
                },
                extra_download_bytes: 0,
                ready: true,
                reuses_synthesis_model: false,
            }
        }
        fn prepare<'a>(
            &'a self,
            _root: &'a Path,
            progress: tokio::sync::mpsc::Sender<Event>,
        ) -> talechime_core::design::Preparing<'a> {
            Box::pin(async move {
                let _ = progress
                    .send(Event::ResourceState {
                        stage: "校验".into(),
                        resource: "fake/designer".into(),
                    })
                    .await;
                Ok(())
            })
        }
        fn design<'a>(
            &'a self,
            _root: &'a Path,
            _request: talechime_core::design::DesignRequest,
        ) -> talechime_core::design::Designing<'a> {
            Box::pin(async move {
                // Minimal valid RIFF: header plus one silent sample frame.
                let wav = test_wav();
                Ok(talechime_core::design::DesignOutput { wav })
            })
        }
    }

    struct PendingDesigner;
    impl talechime_core::design::VoiceDesigner for PendingDesigner {
        fn id(&self) -> &'static str {
            "fake-designer"
        }
        fn info(&self, root: &Path) -> EngineInfo {
            FakeDesigner.info(root)
        }
        fn prepare<'a>(
            &'a self,
            _root: &'a Path,
            _progress: tokio::sync::mpsc::Sender<Event>,
        ) -> talechime_core::design::Preparing<'a> {
            Box::pin(async { Ok(()) })
        }
        fn design<'a>(
            &'a self,
            _root: &'a Path,
            _request: talechime_core::design::DesignRequest,
        ) -> talechime_core::design::Designing<'a> {
            Box::pin(std::future::pending())
        }
    }

    #[tokio::test]
    async fn closed_progress_does_not_starve_cancellation_or_publish_partial_audio() {
        let root = tempfile::tempdir().unwrap();
        let mut studio = VoiceStudio::at(root.path().join("res"), root.path().join("lib")).unwrap();
        studio.set_designers(vec![std::rc::Rc::new(PendingDesigner)]);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), studio.design("描述"))
                .await
                .is_err()
        );
        assert!(studio.list().unwrap().is_empty());
    }

    #[test]
    fn voice_ids_validate_before_deserialization_and_coerce_to_voice_arguments() {
        let id: VoiceId = "hero_1".parse().unwrap();
        assert_eq!(&*id, "hero_1");
        assert_eq!(serde_json::to_string(&id).unwrap(), "\"hero_1\"");
        assert!(serde_json::from_str::<VoiceId>("\"../escape\"").is_err());
        assert!(serde_json::from_str::<VoiceId>("\"custom:hero\"").is_err());
    }

    fn info(id: &'static str, ready: bool, reuse: bool, devices: usize) -> EngineInfo {
        EngineInfo {
            id,
            name: id,
            devices: vec![Device::Cpu; devices],
            inputs: InputConstraints {
                text_max_chars: 100,
                description_max_chars: None,
                description_max_bytes: 4096,
                forbid_parentheses: false,
            },
            extra_download_bytes: 0,
            ready,
            reuses_synthesis_model: reuse,
        }
    }

    #[test]
    fn automatic_policy_prefers_ready_shared_model_engines() {
        let infos = vec![
            info("vox", false, true, 1),
            info("qwen", true, false, 1),
            info("omni", true, true, 1),
        ];
        assert_eq!(pick_engine(&infos).unwrap().id, "omni");
        let infos = vec![info("vox", false, true, 1), info("qwen", true, false, 1)];
        assert_eq!(pick_engine(&infos).unwrap().id, "qwen");
        let infos = vec![info("vox", false, true, 1), info("moss", false, false, 0)];
        assert!(pick_engine(&infos).is_none());
        assert!(pick_engine(&[info("moss", true, false, 0)]).is_none());
    }

    #[test]
    fn slug_generation_avoids_collisions_and_requested_ids_are_reserved() {
        let root = tempfile::tempdir().unwrap();
        let studio = VoiceStudio::at(root.path().join("res"), root.path().join("lib")).unwrap();
        assert_eq!(studio.unique_id(None).unwrap(), "voice");
        let wav = test_wav();
        studio
            .library
            .create("voice", "v", &wav, "转写", None, VoiceSource::Imported)
            .unwrap();
        assert_eq!(studio.unique_id(None).unwrap(), "voice-2");
        assert!(studio.unique_id(Some("voice")).is_err());
        assert_eq!(studio.unique_id(Some("hero")).unwrap(), "hero");
    }

    #[test]
    fn default_name_trims_long_descriptions() {
        assert_eq!(default_name("  低沉男声  "), "低沉男声");
        let long = "字".repeat(40);
        assert_eq!(default_name(&long).chars().count(), 24);
        assert_eq!(default_name("   "), "设计音色");
    }

    #[tokio::test]
    async fn design_with_injected_engine_saves_and_reports_progress() {
        crate::run_local(async {
            let root = tempfile::tempdir().unwrap();
            let mut studio =
                VoiceStudio::at(root.path().join("res"), root.path().join("lib")).unwrap();
            studio.set_designers(vec![std::rc::Rc::new(FakeDesigner)]);
            let mut events = Vec::new();
            let asset = studio
                .design_with(DesignOptions::auto("低沉温和的中年男声"), |event| {
                    events.push(event)
                })
                .await
                .unwrap();
            assert_eq!(asset.id, "voice");
            assert_eq!(asset.name, "低沉温和的中年男声");
            assert_eq!(asset.transcript, DEFAULT_TRIAL_TEXT);
            assert_eq!(
                asset.source,
                VoiceSource::Designed("fake-designer".to_owned())
            );
            assert!(matches!(events.as_slice(), [Event::ResourceState { .. }]));
            // Automatic design is exactly the explicit tier with defaults.
            let auto = studio.design("另一个声音").await.unwrap();
            assert_eq!(auto.id, "voice-2");
            // Explicit unknown engine lists what exists.
            let error = studio
                .design_with(
                    DesignOptions {
                        engine: Some("nope".into()),
                        ..DesignOptions::auto("描述")
                    },
                    |_| {},
                )
                .await
                .unwrap_err()
                .to_string();
            assert!(error.contains("unknown design engine nope"), "{error}");
            assert!(error.contains("fake-designer"), "{error}");
        })
        .await;
    }

    #[tokio::test]
    async fn design_rejects_bad_input_before_preparing() {
        crate::run_local(async {
            let root = tempfile::tempdir().unwrap();
            let mut studio =
                VoiceStudio::at(root.path().join("res"), root.path().join("lib")).unwrap();
            studio.set_designers(vec![std::rc::Rc::new(FakeDesigner)]);
            // The constraints error can only be produced by the pre-prepare check.
            let error = studio
                .design_with(
                    DesignOptions {
                        text: Some(String::new()),
                        ..DesignOptions::auto("描述")
                    },
                    |_| {},
                )
                .await
                .unwrap_err();
            assert!(error.to_string().contains("1..100"), "{}", error);
        })
        .await;
    }

    #[cfg(not(any(
        feature = "qwen",
        feature = "moss",
        feature = "voxcpm",
        feature = "omnivoice"
    )))]
    #[tokio::test]
    async fn automatic_design_error_is_display_safe_without_engines() {
        let root = tempfile::tempdir().unwrap();
        let studio = VoiceStudio::at(root.path().join("res"), root.path().join("lib")).unwrap();
        let error = studio.design("描述").await.unwrap_err().to_string();
        assert!(error.contains("no design engine"), "{error}");
        assert!(
            !error.contains("qwen") && !error.contains("moss"),
            "{error}"
        );
    }
}
