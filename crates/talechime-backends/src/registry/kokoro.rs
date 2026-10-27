use super::*;
use crate::kokoro as model;

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
        model::resources::REVISION
    }
    fn calibrates(&self) -> bool {
        false
    }
    fn prepare<'a>(&'a self, root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a> {
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
