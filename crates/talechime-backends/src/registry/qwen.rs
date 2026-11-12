use super::*;
use crate::qwen::{self, models::Model};
pub(super) const REGISTRATION: Registration = Registration {
    id: "qwen",
    default_model: "0.6b-customvoice",
    models: || {
        // 1.7b-voicedesign stays an internal design-only generator: the voices
        // command prepares it on demand and never offers it for narration.
        [
            Model::Custom06,
            Model::Custom17,
            Model::Base06,
            Model::Base17,
        ]
        .into_iter()
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
    fn reference_entry(&self, root: &Path, id: &str) -> anyhow::Result<ReferenceEntry> {
        anyhow::ensure!(
            self.0.is_base(),
            "qwen reference voices require model 0.6b-base or 1.7b-base"
        );
        Ok(ReferenceEntry::Directory(
            qwen::voice_store(&self.0.directory(root), self.0)?.path(id)?,
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
        anyhow::ensure!(
            self.0.is_base(),
            "qwen reference voices require model 0.6b-base or 1.7b-base"
        );
        qwen::validate_wav(wav)?;
        let store = qwen::voice_store(&self.0.directory(root), self.0)?;
        store.import(id, name, wav, transcript, None)?;
        Ok(())
    }

    fn supports_recovery(&self) -> bool {
        true
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
        Box::pin(async move { qwen::resources::prepare_model(self.0, downloads, progress).await })
    }
    fn load<'a>(
        &'a self,
        root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
        progress: mpsc::Sender<Event>,
        device: Device,
        recovery: bool,
    ) -> Loading<'a> {
        Box::pin(async move {
            let directory = self.0.directory(root);
            let paths =
                crate::resources::resolve_manifest(self.0.manifest(), downloads.clone()).await?;
            let files = Model::resolved_files(paths)?;
            let recovery = (recovery && device != Device::Cpu).then_some(progress);
            let backend =
                qwen::QwenBackend::load_source(directory, self.0, files, device, recovery).await?;
            Ok(Rc::new(backend) as Rc<dyn Backend>)
        })
    }
}
