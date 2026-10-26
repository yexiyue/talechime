use super::*;
use crate::cosyvoice3 as model;
pub(super) const REGISTRATION: Registration = Registration {
    id: "cosyvoice3",
    default_model: model::MODEL,
    models: || vec![Rc::new(Adapter)],
};
struct Adapter;
impl ModelAdapter for Adapter {
    fn id(&self) -> &'static str {
        model::MODEL
    }
    fn capabilities(&self, root: &Path) -> anyhow::Result<Capabilities> {
        model::capabilities(&model::directory(root))
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
    fn reference_entry(&self, root: &Path, id: &str) -> anyhow::Result<ReferenceEntry> {
        Ok(ReferenceEntry::Directory(
            model::voice_store(&model::directory(root))?.path(id)?,
        ))
    }
    fn install_reference(
        &self,
        root: &Path,
        wav: &Path,
        id: &str,
        name: &str,
        transcript: &str,
    ) -> anyhow::Result<()> {
        crate::reference::load(wav, 24000)?;
        let store = model::voice_store(&model::directory(root))?;
        store.import(id, name, wav, transcript, None)?;
        Ok(())
    }

    fn calibrates(&self) -> bool {
        false
    }
    fn prepare<'a>(&'a self, root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a> {
        Box::pin(async move { model::resources::prepare(&model::directory(root), progress).await })
    }
    fn load<'a>(
        &'a self,
        root: &'a Path,
        _progress: mpsc::Sender<Event>,
        device: Device,
        _recovery: bool,
    ) -> Loading<'a> {
        Box::pin(async move {
            let directory = model::directory(root);
            Ok(Rc::new(model::CosyBackend::load_on(directory, device).await?) as Rc<dyn Backend>)
        })
    }
}
