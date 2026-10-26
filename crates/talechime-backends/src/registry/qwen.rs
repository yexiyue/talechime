use super::*;
use crate::qwen::{self, models::Model};
pub(super) const REGISTRATION: Registration = Registration {
    id: "qwen",
    default_model: "0.6b-customvoice",
    models: || {
        [
            Model::Custom06,
            Model::Custom17,
            Model::Base06,
            Model::Base17,
            Model::Design17,
        ]
        .into_iter()
        .map(|model| Rc::new(Adapter(model)) as Rc<dyn ModelAdapter>)
        .collect()
    },
};
struct Adapter(Model);
impl ModelAdapter for Adapter {
    fn id(&self) -> &'static str {
        self.0.id()
    }
    fn capabilities(&self, root: &Path) -> anyhow::Result<Capabilities> {
        qwen::capabilities_at(&self.0.directory(root), self.0)
    }
    fn compiled_devices(&self) -> Vec<Device> {
        qwen::compiled_devices()
    }
    fn available_devices(&self) -> Vec<Device> {
        qwen::available_devices()
    }
    fn revision(&self) -> &'static str {
        self.0.revision()
    }
    fn supports_recovery(&self) -> bool {
        true
    }
    fn prepare<'a>(&'a self, root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a> {
        Box::pin(async move {
            qwen::resources::prepare_model(&self.0.directory(root), self.0, progress).await
        })
    }
    fn load<'a>(
        &'a self,
        root: &'a Path,
        progress: mpsc::Sender<Event>,
        device: Device,
        recovery: bool,
    ) -> Loading<'a> {
        Box::pin(async move {
            let directory = self.0.directory(root);
            let backend = if recovery && device != Device::Cpu {
                qwen::QwenBackend::load_with_recovery(directory, device, Some(progress)).await?
            } else {
                qwen::QwenBackend::load_on(directory, device).await?
            };
            Ok(Rc::new(backend) as Rc<dyn Backend>)
        })
    }
}
