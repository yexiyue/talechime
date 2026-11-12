use super::*;
use crate::omnivoice as model;
pub(super) const REGISTRATION: Registration = Registration {
    id: "omnivoice",
    default_model: model::MODEL,
    models: || vec![Rc::new(Adapter)],
};
struct Adapter;
impl ModelAdapter for Adapter {
    fn manifests(&self) -> Vec<&'static str> {
        vec![model::resources::MANIFEST]
    }
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
        _downloads: &crate::downloads::Downloads,
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
        true
    }
    fn resource_status<'a>(
        &'a self,
        _root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
    ) -> Inventory<'a> {
        Box::pin(crate::resources::cached_manifest_status(
            model::resources::MANIFEST,
            downloads.clone(),
        ))
    }
    fn prepare<'a>(
        &'a self,
        _root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
        progress: mpsc::Sender<Event>,
    ) -> Preparing<'a> {
        Box::pin(async move { model::resources::prepare(downloads, progress).await })
    }
    fn load<'a>(
        &'a self,
        root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
        _progress: mpsc::Sender<Event>,
        device: Device,
        _recovery: bool,
    ) -> Loading<'a> {
        Box::pin(async move {
            let directory = model::directory(root);
            let files =
                crate::resources::resolve_manifest(model::resources::MANIFEST, downloads.clone())
                    .await?
                    .into_iter()
                    .map(|(name, path)| (name.into(), path))
                    .collect();
            Ok(
                Rc::new(model::OmniBackend::load_source_on(directory, files, device).await?)
                    as Rc<dyn Backend>,
            )
        })
    }
}
