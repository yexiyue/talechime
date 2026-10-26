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
    fn prepare<'a>(&'a self, root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a> {
        Box::pin(async move {
            match self {
                #[cfg(feature = "moss-nano-candle")]
                Self::Nano => moss::nano::prepare(root, progress).await,
                Self::Realtime => {
                    moss::candle::resources::prepare(root, moss::candle::Mode::Realtime, progress)
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
