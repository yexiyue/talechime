use super::*;
use crate::cosyvoice3 as model;
pub(super) const REGISTRATION: Registration = Registration {
    id: "cosyvoice3",
    default_model: model::MODEL,
    models: || vec![Rc::new(Adapter)],
};
struct Adapter;
impl ModelAdapter for Adapter {
    fn native_recipe(&self) -> Option<crate::native_preparation::Recipe> {
        Some(model::native_recipe())
    }
    fn resource_preparation(&self) -> ResourcePreparation {
        ResourcePreparation::Native
    }
    fn id(&self) -> &'static str {
        model::MODEL
    }
    fn capabilities(&self, root: &Path) -> anyhow::Result<Capabilities> {
        model::capabilities(root)
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
            model::voice_store(root)?.path(id)?,
        ))
    }
    fn install_reference(
        &self,
        root: &Path,
        _downloads: &crate::downloads::Downloads,
        wav: &Path,
        id: &str,
        name: &str,
        transcript: &str,
    ) -> anyhow::Result<()> {
        crate::reference::load(wav, 24000)?;
        let store = model::voice_store(root)?;
        store.import(id, name, wav, transcript, None)?;
        Ok(())
    }

    fn calibrates(&self) -> bool {
        false
    }
    fn resource_status(&self, root: &Path) -> crate::resource_state::ResourceStatus {
        model::native_recipe().status(root)
    }
    fn prepare<'a>(
        &'a self,
        root: &'a Path,
        _downloads: &'a crate::downloads::Downloads,
        progress: mpsc::Sender<Event>,
    ) -> Preparing<'a> {
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
            Ok(
                Rc::new(model::CosyBackend::load_on(root.to_owned(), device).await?)
                    as Rc<dyn Backend>,
            )
        })
    }
}
