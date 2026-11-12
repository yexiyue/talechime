use super::*;
use crate::voxcpm::{self, models::Model};
pub(super) const REGISTRATION: Registration = Registration {
    id: "voxcpm",
    default_model: "2b-q8_0",
    models: || {
        [Model::Q8, Model::OriginalBf16]
            .into_iter()
            .filter(|model| !model.compiled_devices().is_empty())
            .map(|model| Rc::new(Adapter(model)) as Rc<dyn ModelAdapter>)
            .collect()
    },
};
struct Adapter(Model);
impl ModelAdapter for Adapter {
    fn manifests(&self) -> Vec<&'static str> {
        vec![self.0.manifest()]
    }
    fn id(&self) -> &'static str {
        self.0.id()
    }
    fn capabilities(&self, root: &Path) -> anyhow::Result<Capabilities> {
        voxcpm::capabilities_for(&self.0.directory(root), self.0)
    }
    fn compiled_devices(&self) -> Vec<Device> {
        self.0.compiled_devices()
    }
    fn available_devices(&self) -> Vec<Device> {
        voxcpm::available_devices()
            .into_iter()
            .filter(|d| self.0.compiled_devices().contains(d))
            .collect()
    }
    fn revision(&self) -> &'static str {
        self.0.calibration_revision()
    }
    fn reference_entry(&self, root: &Path, id: &str) -> anyhow::Result<ReferenceEntry> {
        Ok(ReferenceEntry::Directory(
            voxcpm::voice_store_for(&self.0.directory(root), self.0)?.path(id)?,
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
        crate::reference::load(wav, 16000)?;
        let store = voxcpm::voice_store_for(&self.0.directory(root), self.0)?;
        store.import(id, name, wav, transcript, None)?;
        Ok(())
    }

    fn resource_status<'a>(
        &'a self,
        _root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
    ) -> Inventory<'a> {
        Box::pin(crate::resources::cached_manifest_status(
            self.0.manifest(),
            downloads.clone(),
        ))
    }
    fn prepare<'a>(
        &'a self,
        _root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
        progress: mpsc::Sender<Event>,
    ) -> Preparing<'a> {
        Box::pin(async move { voxcpm::resources::prepare_model(self.0, downloads, progress).await })
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
            let directory = self.0.directory(root);
            let paths =
                crate::resources::resolve_manifest(self.0.manifest(), downloads.clone()).await?;
            let files = voxcpm::models::ModelFiles::from_resolved(self.0, paths)?;
            Ok(Rc::new(
                voxcpm::VoxBackend::load_model_files_on(directory, self.0, files, device).await?,
            ) as Rc<dyn Backend>)
        })
    }
}
