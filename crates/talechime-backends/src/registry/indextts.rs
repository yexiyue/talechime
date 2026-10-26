use super::*;
use crate::indextts as model;
pub(super) const REGISTRATION: Registration = Registration {
    id: "indextts",
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
        let reference = crate::reference::load(wav, 22_050)?;
        anyhow::ensure!(
            reference.len() <= 15 * 22_050,
            "IndexTTS reference must be at most 15 seconds"
        );
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
            Ok(Rc::new(model::IndexBackend::load_on(directory, device).await?) as Rc<dyn Backend>)
        })
    }
}
