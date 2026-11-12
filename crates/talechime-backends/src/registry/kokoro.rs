use super::*;
use crate::kokoro as model;

// Inference identity is independent of the v3 verified resource directory.
const INFERENCE_REVISION: &str = "01e7505-runtime-f32-v3-text-v4-loqui-neural-0.4.0";

pub(super) const REGISTRATION: Registration = Registration {
    id: "kokoro",
    default_model: model::MODEL,
    models: || vec![Rc::new(Adapter(model::MODEL))],
};
struct Adapter(&'static str);
impl ModelAdapter for Adapter {
    fn manifests(&self) -> Vec<&'static str> {
        vec![model::resources::HUB_MANIFEST]
    }
    fn id(&self) -> &'static str {
        self.0
    }
    fn capabilities(&self, _root: &Path) -> anyhow::Result<Capabilities> {
        model::cached_capabilities(self.0)
    }
    fn compiled_devices(&self) -> Vec<Device> {
        model::compiled_devices()
    }
    fn available_devices(&self) -> Vec<Device> {
        model::available_devices()
    }
    fn revision(&self) -> &'static str {
        INFERENCE_REVISION
    }
    fn calibrates(&self) -> bool {
        false
    }
    fn resource_status<'a>(
        &'a self,
        _root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
    ) -> Inventory<'a> {
        Box::pin(async move {
            crate::resources::cached_manifest_status(
                model::resources::HUB_MANIFEST,
                downloads.clone(),
            )
            .await
        })
    }
    fn prepare<'a>(
        &'a self,
        _root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
        progress: mpsc::Sender<Event>,
    ) -> Preparing<'a> {
        Box::pin(async move {
            crate::resources::prepare_cached(
                "kokoro",
                serde_json::from_str(model::resources::HUB_MANIFEST)?,
                downloads,
                progress,
            )
            .await?;
            Ok(())
        })
    }
    fn load<'a>(
        &'a self,
        _root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
        _progress: mpsc::Sender<Event>,
        device: Device,
        _recovery: bool,
    ) -> Loading<'a> {
        Box::pin(async move {
            let files = crate::resources::resolve_manifest(
                model::resources::HUB_MANIFEST,
                downloads.clone(),
            )
            .await?;
            Ok(
                Rc::new(model::KokoroBackend::load_cached(files, self.0, device).await?)
                    as Rc<dyn Backend>,
            )
        })
    }
}
