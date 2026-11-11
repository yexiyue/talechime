use super::*;
use crate::kokoro as model;

// Inference identity is independent of the v3 verified resource directory.
const INFERENCE_REVISION: &str =
    "01e7505-runtime-f32-v3-text-v3-espeak-1.52.0-4870adfa25b1a32b4361592f1be8a40337c58d6c";

pub(super) const REGISTRATION: Registration = Registration {
    id: "kokoro",
    default_model: model::MODEL,
    models: || {
        vec![
            Rc::new(Adapter(model::MODEL)),
            #[cfg(feature = "kokoro-g2pw")]
            Rc::new(Adapter(model::CONTEXT_MODEL)),
        ]
    },
};
struct Adapter(&'static str);
impl ModelAdapter for Adapter {
    fn native_recipe(&self) -> Option<crate::native_preparation::Recipe> {
        model::native_recipe(self.0)
    }
    fn resource_preparation(&self) -> ResourcePreparation {
        ResourcePreparation::Native
    }
    fn id(&self) -> &'static str {
        self.0
    }
    fn capabilities(&self, root: &Path) -> anyhow::Result<Capabilities> {
        model::capabilities_for(&model::directory_for(root, self.0), self.0)
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
    fn resource_status(&self, root: &Path) -> crate::resource_state::ResourceStatus {
        match model::native_recipe(self.0) {
            Some(recipe) => recipe.status(root),
            None => model::resources::conversion_status(&model::directory_for(root, self.0)),
        }
    }
    fn prepare<'a>(
        &'a self,
        root: &'a Path,
        _downloads: &'a crate::downloads::Downloads,
        progress: mpsc::Sender<Event>,
    ) -> Preparing<'a> {
        Box::pin(async move {
            model::resources::prepare_for(&model::directory_for(root, self.0), self.0, progress)
                .await
        })
    }
    fn load<'a>(
        &'a self,
        root: &'a Path,
        _progress: mpsc::Sender<Event>,
        device: Device,
        _recovery: bool,
    ) -> Loading<'a> {
        Box::pin(async move {
            Ok(Rc::new(
                model::KokoroBackend::load_model_on(
                    model::directory_for(root, self.0),
                    self.0,
                    device,
                )
                .await?,
            ) as Rc<dyn Backend>)
        })
    }
}
