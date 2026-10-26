use super::*;
use crate::moss;
pub(super) const REGISTRATION: Registration = Registration {
    id: "moss",
    default_model: if cfg!(feature = "moss-nano-candle") {
        "nano-candle"
    } else {
        "nano"
    },
    models: || {
        vec![
            #[cfg(feature = "moss-nano-candle")]
            Rc::new(Adapter::Nano),
            Rc::new(Adapter::Onnx),
            #[cfg(feature = "moss-candle")]
            Rc::new(Adapter::Candle(moss::candle::Mode::Local)),
            #[cfg(feature = "moss-candle")]
            Rc::new(Adapter::Candle(moss::candle::Mode::Realtime)),
        ]
    },
};
enum Adapter {
    Onnx,
    #[cfg(feature = "moss-nano-candle")]
    Nano,
    #[cfg(feature = "moss-candle")]
    Candle(moss::candle::Mode),
}
impl ModelAdapter for Adapter {
    fn id(&self) -> &'static str {
        match self {
            Self::Onnx => "nano",
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => "nano-candle",
            #[cfg(feature = "moss-candle")]
            Self::Candle(mode) => mode.id(),
        }
    }
    fn capabilities(&self, root: &Path) -> anyhow::Result<Capabilities> {
        match self {
            Self::Onnx => moss::capabilities(&root.join("moss")),
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => moss::nano::capabilities(root),
            #[cfg(feature = "moss-candle")]
            Self::Candle(mode) => moss::candle::capabilities(root, *mode),
        }
    }
    fn compiled_devices(&self) -> Vec<Device> {
        match self {
            Self::Onnx => crate::devices::compiled(),
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => moss::nano::compiled_devices(),
            #[cfg(feature = "moss-candle")]
            Self::Candle(_) => moss::candle::compiled_devices(),
        }
    }
    fn available_devices(&self) -> Vec<Device> {
        match self {
            Self::Onnx => crate::devices::available(),
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => moss::nano::available_devices(),
            #[cfg(feature = "moss-candle")]
            Self::Candle(_) => moss::candle::available_devices(),
        }
    }
    fn revision(&self) -> &'static str {
        match self {
            Self::Onnx => moss::resources::REVISION,
            #[cfg(feature = "moss-nano-candle")]
            Self::Nano => moss::nano::REVISION,
            #[cfg(feature = "moss-candle")]
            Self::Candle(mode) => mode.revision(),
        }
    }
    fn supports_recovery(&self) -> bool {
        matches!(self, Self::Onnx)
    }
    fn prepare<'a>(&'a self, root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a> {
        Box::pin(async move {
            match self {
                Self::Onnx => moss::resources::prepare(&root.join("moss"), progress).await,
                #[cfg(feature = "moss-nano-candle")]
                Self::Nano => moss::nano::prepare(root, progress).await,
                #[cfg(feature = "moss-candle")]
                Self::Candle(mode) => moss::candle::resources::prepare(root, *mode, progress).await,
            }
        })
    }
    fn load<'a>(
        &'a self,
        root: &'a Path,
        progress: mpsc::Sender<Event>,
        device: Device,
        recovery: bool,
    ) -> Loading<'a> {
        Box::pin(async move {
            Ok(match self {
                Self::Onnx => {
                    let directory = root.join("moss");
                    let backend = if recovery && device != Device::Cpu {
                        moss::MossBackend::load_with_recovery(directory, device, Some(progress))
                            .await?
                    } else {
                        moss::MossBackend::load_on(directory, device).await?
                    };
                    Rc::new(backend) as Rc<dyn Backend>
                }
                #[cfg(feature = "moss-nano-candle")]
                Self::Nano => {
                    Rc::new(moss::nano::NanoBackend::load_on(root.to_path_buf(), device).await?)
                        as Rc<dyn Backend>
                }
                #[cfg(feature = "moss-candle")]
                Self::Candle(mode) => Rc::new(
                    moss::candle::CandleBackend::load_on(root.to_path_buf(), *mode, device).await?,
                ) as Rc<dyn Backend>,
            })
        })
    }
}
