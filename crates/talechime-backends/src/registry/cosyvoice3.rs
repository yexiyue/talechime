use super::*;
use crate::cosyvoice3 as model;
pub(super) const REGISTRATION: Registration = Registration {
    id: "cosyvoice3",
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
    fn resource_status<'a>(
        &'a self,
        _root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
    ) -> Inventory<'a> {
        Box::pin(async move {
            crate::resources::cached_manifest_status(model::resources::MANIFEST, downloads.clone())
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
                "cosyvoice3",
                serde_json::from_str(model::resources::MANIFEST)?,
                downloads,
                progress,
            )
            .await?;
            Ok(())
        })
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
            let files =
                crate::resources::resolve_manifest(model::resources::MANIFEST, downloads.clone())
                    .await?;
            Ok(
                Rc::new(model::CosyBackend::load_cached(root.to_owned(), files, device).await?)
                    as Rc<dyn Backend>,
            )
        })
    }
}
