use super::*;
use std::cell::RefCell;
use talechime_core::backend::{BackendError, SegmentRequest, Streaming};

fn capabilities() -> Capabilities {
    Capabilities {
        backend: "fixture".into(),
        model: Some("one".into()),
        model_name: String::new(),
        voices: vec!["voice".into()],
        default_voice: "voice".into(),
        voice_names: Default::default(),
        compiled_devices: vec![Device::Cpu, Device::Cuda],
        native_streaming: true,
        cloning: false,
        style: false,
        pronunciation: false,

        parameters: vec![],
    }
}
struct Prepared;
impl Backend for Prepared {
    fn capabilities(&self) -> Capabilities {
        capabilities()
    }
    fn stream<'a>(&'a self, _: SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async { Err(BackendError::Synthesis("unused fixture synthesis".into())) })
    }
}
struct Trace {
    loads: RefCell<Vec<(Device, bool)>>,
    prepares: std::cell::Cell<usize>,
}
struct Adapter {
    trace: Rc<Trace>,
    fail_gpu: bool,
}
impl ModelAdapter for Adapter {
    fn id(&self) -> &'static str {
        "one"
    }
    fn revision(&self) -> &'static str {
        "revision-one"
    }
    fn capabilities(&self, _: &Path) -> anyhow::Result<Capabilities> {
        Ok(capabilities())
    }
    fn compiled_devices(&self) -> Vec<Device> {
        vec![Device::Cpu, Device::Cuda]
    }
    fn available_devices(&self) -> Vec<Device> {
        self.compiled_devices()
    }
    fn supports_recovery(&self) -> bool {
        true
    }
    fn prepare<'a>(&'a self, _: &'a Path, _: mpsc::Sender<Event>) -> Preparing<'a> {
        Box::pin(async move {
            self.trace.prepares.set(self.trace.prepares.get() + 1);
            Ok(())
        })
    }
    fn load<'a>(
        &'a self,
        _: &'a Path,
        _: mpsc::Sender<Event>,
        device: Device,
        recovery: bool,
    ) -> Loading<'a> {
        Box::pin(async move {
            self.trace.loads.borrow_mut().push((device, recovery));
            anyhow::ensure!(
                !(device == Device::Cuda && self.fail_gpu),
                "fixture GPU initialization failed"
            );
            Ok(Rc::new(Prepared) as Rc<dyn Backend>)
        })
    }
}
fn model(root: &Path, fail_gpu: bool) -> (ResolvedModel, Rc<Trace>) {
    let loads = Rc::new(Trace {
        loads: RefCell::new(vec![]),
        prepares: std::cell::Cell::new(0),
    });
    (
        ResolvedModel {
            adapter: Rc::new(Adapter {
                trace: loads.clone(),
                fail_gpu,
            }),
            root: root.into(),
            capabilities: capabilities(),
        },
        loads,
    )
}
fn cache(model: &ResolvedModel, device: Device) {
    let key = calibration::key(
        "tts",
        &format!(
            "{}:{}:{}",
            model.capabilities.backend,
            model.adapter.id(),
            model.revision()
        ),
    );
    calibration::save(
        &model.root,
        "tts",
        &calibration::Record {
            key,
            device,
            reason: "fixture cache".into(),
        },
    )
    .unwrap();
}
#[tokio::test]
async fn explicit_failure_never_falls_back_or_enables_recovery() {
    let root = tempfile::tempdir().unwrap();
    let (model, loads) = model(root.path(), true);
    let (tx, _rx) = mpsc::channel(16);
    assert!(
        model
            .prepare(
                PrepareOptions {
                    device: Device::Cuda,
                    calibrate: true,
                    recovery: true,
                    voice: None
                },
                tx
            )
            .await
            .is_err()
    );
    assert_eq!(*loads.loads.borrow(), [(Device::Cuda, false)]);
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
}
#[tokio::test]
async fn auto_without_calibration_uses_cpu_without_writing_or_reading_cache() {
    let root = tempfile::tempdir().unwrap();
    let (model, loads) = model(root.path(), false);
    let (tx, _rx) = mpsc::channel(16);
    model
        .prepare(
            PrepareOptions {
                device: Device::Auto,
                ..Default::default()
            },
            tx,
        )
        .await
        .unwrap();
    assert_eq!(*loads.loads.borrow(), [(Device::Cpu, false)]);
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
}
#[tokio::test]
async fn failed_calibration_retains_cpu_and_does_not_publish_a_cache() {
    let root = tempfile::tempdir().unwrap();
    let (model, loads) = model(root.path(), true);
    let (tx, mut rx) = mpsc::channel(16);
    model
        .prepare(
            PrepareOptions {
                device: Device::Auto,
                calibrate: true,
                ..Default::default()
            },
            tx,
        )
        .await
        .unwrap();
    assert_eq!(
        *loads.loads.borrow(),
        [(Device::Cpu, false), (Device::Cuda, false)]
    );
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    assert!(std::iter::from_fn(|| rx.try_recv().ok()).any(|event| matches!(event, Event::DeviceStatus { selected: Device::Cpu, reason: Some(reason), .. } if reason.contains("calibration failed"))));
}
#[tokio::test]
async fn cached_accelerator_failure_falls_back_but_explicit_selection_does_not() {
    let root = tempfile::tempdir().unwrap();
    let (model, loads) = model(root.path(), true);
    cache(&model, Device::Cuda);
    let (tx, _rx) = mpsc::channel(16);
    model
        .prepare(
            PrepareOptions {
                device: Device::Auto,
                calibrate: true,
                recovery: true,
                voice: None,
            },
            tx,
        )
        .await
        .unwrap();
    assert_eq!(
        *loads.loads.borrow(),
        [(Device::Cuda, false), (Device::Cpu, false)]
    );
}
#[tokio::test]
async fn cached_accelerator_uses_adapter_owned_recovery_only_when_requested() {
    let root = tempfile::tempdir().unwrap();
    let (model, loads) = model(root.path(), false);
    cache(&model, Device::Cuda);
    for recovery in [false, true] {
        loads.loads.borrow_mut().clear();
        let (tx, _rx) = mpsc::channel(16);
        model
            .prepare(
                PrepareOptions {
                    device: Device::Auto,
                    calibrate: true,
                    recovery,
                    voice: None,
                },
                tx,
            )
            .await
            .unwrap();
        let mut expected = vec![(Device::Cuda, false)];
        if recovery {
            expected.push((Device::Cuda, true));
        }
        assert_eq!(*loads.loads.borrow(), expected);
        assert_eq!(loads.prepares.get(), if recovery { 2 } else { 1 });
    }
}
