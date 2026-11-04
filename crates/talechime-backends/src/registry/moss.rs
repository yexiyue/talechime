use super::*;
use crate::moss;
pub(super) const REGISTRATION: Registration = Registration {
    id: "moss",
    default_model: if cfg!(feature = "moss-nano-candle") {
        "nano-candle"
    } else {
        "realtime-1.7b"
    },
    models: || {
        vec![
            #[cfg(feature = "moss-nano-candle")]
            Rc::new(Adapter::Nano),
            Rc::new(Adapter::Realtime),
        ]
    },
};
enum Adapter {
    #[cfg(feature = "moss-nano-candle")]
    Nano,
    Realtime,
}
impl ModelAdapter for Adapter {
    fn id(&self) -> &'static str {
        match self {
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => "nano-candle",
            Self::Realtime => "realtime-1.7b",
        }
    }
    fn capabilities(&self, root: &Path) -> anyhow::Result<Capabilities> {
        match self {
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => moss::nano::capabilities(root),
            Self::Realtime => moss::candle::capabilities(root, moss::candle::Mode::Realtime),
        }
    }
    fn compiled_devices(&self) -> Vec<Device> {
        match self {
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => moss::nano::compiled_devices(),
            Self::Realtime => moss::candle::compiled_devices(),
        }
    }
    fn available_devices(&self) -> Vec<Device> {
        match self {
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => moss::nano::available_devices(),
            Self::Realtime => moss::candle::available_devices(),
        }
    }
    fn revision(&self) -> &'static str {
        match self {
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => moss::nano::REVISION,
            Self::Realtime => moss::candle::Mode::Realtime.revision(),
        }
    }
    fn reference_entry(&self, root: &Path, id: &str) -> anyhow::Result<ReferenceEntry> {
        match self {
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => Ok(ReferenceEntry::File(
                moss::voices::VoiceStore::new(&moss::nano::directory(root)).path(id)?,
            )),
            Self::Realtime => Ok(ReferenceEntry::Directory(
                moss::candle::voice_store(
                    &moss::candle::Mode::Realtime.directory(root),
                    moss::candle::Mode::Realtime,
                )?
                .path(id)?,
            )),
        }
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
        match self {
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => anyhow::bail!(
                "MOSS Nano encodes references in the background; use the asynchronous install"
            ),
            Self::Realtime => {
                let reference = crate::reference::load(wav, 24000)?;
                anyhow::ensure!(
                    reference.len() <= 240000,
                    "MOSS trial reference must be at most 10 seconds"
                );
                let mode = moss::candle::Mode::Realtime;
                let store = moss::candle::voice_store(&mode.directory(root), mode)?;
                store.import(id, name, wav, transcript, None)?;
                Ok(())
            }
        }
    }
    fn install_reference_async<'a>(
        &'a self,
        root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
        wav: &'a Path,
        id: &'a str,
        name: &'a str,
        transcript: &'a str,
    ) -> Installing<'a> {
        let root = root.to_path_buf();
        let downloads = downloads.clone();
        let wav = wav.to_path_buf();
        let id = id.to_string();
        let name = name.to_string();
        let transcript = transcript.to_string();
        Box::pin(async move {
            match self {
                #[cfg(feature = "moss-nano-candle")]
                Self::Nano => {
                    let _ = transcript;
                    moss::nano::import_voice(root, &downloads, id, name, wav).await?;
                    Ok(())
                }
                Self::Realtime => {
                    self.install_reference(&root, &downloads, &wav, &id, &name, &transcript)
                }
            }
        })
    }
    fn resource_status(&self, root: &Path) -> crate::resource_state::ResourceStatus {
        match self {
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => crate::resource_state::status_of_manifests(&[(
                moss::nano::MANIFEST,
                &moss::nano::directory(root),
            )]),
            Self::Realtime => {
                let mode = moss::candle::Mode::Realtime;
                crate::resource_state::status_of_manifests(&[
                    (
                        talechime_moss_tts::manifests::REALTIME,
                        &mode.directory(root),
                    ),
                    (
                        talechime_moss_tts::manifests::CODEC,
                        &moss::candle::resources::codec_directory(root),
                    ),
                ])
            }
        }
    }
    fn prepare<'a>(
        &'a self,
        root: &'a Path,
        downloads: &'a crate::downloads::Downloads,
        progress: mpsc::Sender<Event>,
    ) -> Preparing<'a> {
        Box::pin(async move {
            match self {
                #[cfg(feature = "moss-nano-candle")]
                Self::Nano => moss::nano::prepare(root, downloads, progress).await,
                Self::Realtime => {
                    moss::candle::resources::prepare(
                        root,
                        moss::candle::Mode::Realtime,
                        downloads,
                        progress,
                    )
                    .await
                }
            }
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
            Ok(match self {
                #[cfg(feature = "moss-nano-candle")]
                Self::Nano => {
                    Rc::new(moss::nano::NanoBackend::load_on(root.to_path_buf(), device).await?)
                        as Rc<dyn Backend>
                }
                Self::Realtime => Rc::new(
                    moss::candle::CandleBackend::load_on(
                        root.to_path_buf(),
                        moss::candle::Mode::Realtime,
                        device,
                    )
                    .await?,
                ) as Rc<dyn Backend>,
            })
        })
    }
}
