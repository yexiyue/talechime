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
    prepares: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}
struct Adapter {
    trace: Rc<Trace>,
    fail_gpu: bool,
    ready: bool,
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
    fn resource_status(&self, _: &Path) -> crate::resource_state::ResourceStatus {
        if self.ready {
            crate::ResourceStatus::Ready
        } else {
            crate::ResourceStatus::Missing { missing_bytes: 1 }
        }
    }
    fn prepare<'a>(
        &'a self,
        _: &'a Path,
        _: &crate::downloads::Downloads,
        _: mpsc::Sender<Event>,
    ) -> Preparing<'a> {
        let prepares = self.trace.prepares.clone();
        Box::pin(async move {
            use std::sync::atomic::Ordering;
            prepares.fetch_add(1, Ordering::SeqCst);
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
    model_ready(root, fail_gpu, true)
}
fn model_ready(root: &Path, fail_gpu: bool, ready: bool) -> (ResolvedModel, Rc<Trace>) {
    let loads = Rc::new(Trace {
        loads: RefCell::new(vec![]),
        prepares: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    });
    (
        ResolvedModel {
            adapter: Rc::new(Adapter {
                trace: loads.clone(),
                fail_gpu,
                ready,
            }),
            root: root.into(),
            downloads: crate::downloads::Downloads::test(root),
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
        assert_eq!(
            loads.prepares.load(std::sync::atomic::Ordering::SeqCst),
            if recovery { 2 } else { 1 }
        );
    }
}

#[tokio::test]
async fn loading_existing_never_runs_resource_preparation() {
    let root = tempfile::tempdir().unwrap();
    let (model, trace) = model(root.path(), false);
    let (tx, _rx) = mpsc::channel(16);
    model
        .load_existing(PrepareOptions::default(), tx)
        .await
        .unwrap();
    assert_eq!(trace.prepares.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(*trace.loads.borrow(), [(Device::Cpu, false)]);
}

#[tokio::test]
async fn missing_resources_never_prepare_or_load() {
    let root = tempfile::tempdir().unwrap();
    let (model, trace) = model_ready(root.path(), false, false);
    let (tx, _rx) = mpsc::channel(16);
    assert!(
        model
            .load_existing(PrepareOptions::default(), tx)
            .await
            .is_err()
    );
    assert_eq!(trace.prepares.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(trace.loads.borrow().is_empty());
}

#[tokio::test]
async fn native_plan_and_confirmed_execution_use_the_same_resolved_recipe() {
    use crate::native_preparation::{Input, Output, Recipe};
    struct Native;
    impl ModelAdapter for Native {
        fn native_recipe(&self) -> Option<Recipe> {
            const HASH: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
            Some(Recipe {
                id: "fixture-native-v1",
                target: "fixture/models/one/native-v1".into(),
                inputs: &[Input {
                    url: None,
                    path: "source.bin",
                    repo: "fixture/model",
                    revision: "1111111111111111111111111111111111111111",
                    filename: "source.bin",
                    size: 3,
                    sha256: HASH,
                }],
                outputs: &[Output {
                    path: "model.bin",
                    size: 3,
                    sha256: HASH,
                }],
                transform_outputs: &["model.bin"],
                workspace_bytes: 0,
                transform: |work| {
                    work.component("model.bin", |input, output, _| {
                        std::fs::copy(input.join("source.bin"), output)?;
                        Ok(())
                    })
                },
            })
        }
        fn id(&self) -> &'static str {
            "one"
        }
        fn revision(&self) -> &'static str {
            "fixture-native-v1"
        }
        fn capabilities(&self, _: &Path) -> anyhow::Result<Capabilities> {
            Ok(capabilities())
        }
        fn compiled_devices(&self) -> Vec<Device> {
            vec![Device::Cpu]
        }
        fn available_devices(&self) -> Vec<Device> {
            vec![Device::Cpu]
        }
        fn resource_status(&self, _: &Path) -> crate::ResourceStatus {
            crate::ResourceStatus::Missing { missing_bytes: 3 }
        }
        fn prepare<'a>(
            &'a self,
            _: &'a Path,
            _: &'a crate::downloads::Downloads,
            _: mpsc::Sender<Event>,
        ) -> Preparing<'a> {
            Box::pin(async { anyhow::bail!("unused") })
        }
        fn load<'a>(
            &'a self,
            _: &'a Path,
            _: mpsc::Sender<Event>,
            _: Device,
            _: bool,
        ) -> Loading<'a> {
            Box::pin(async { anyhow::bail!("unused") })
        }
    }
    let root = tempfile::tempdir().unwrap();
    let inputs = root
        .path()
        .join(".native-preparation/inputs/fixture-native-v1");
    std::fs::create_dir_all(&inputs).unwrap();
    std::fs::write(inputs.join("source.bin"), b"abc").unwrap();
    let model = ResolvedModel {
        adapter: Rc::new(Native),
        root: root.path().into(),
        downloads: crate::downloads::Downloads::test(root.path()),
        capabilities: capabilities(),
    };
    let plan = model.native_preparation_plan().await.unwrap().unwrap();
    assert!(!plan.ready);
    assert_eq!(plan.download_bytes, 0);
    let (tx, _) = mpsc::channel(8);
    let (files, _) = mpsc::channel(8);
    model
        .prepare_native_resources(
            plan.fingerprint,
            crate::PreparationControl::default(),
            tx,
            files,
        )
        .await
        .unwrap();
    let ready = model.native_preparation_plan().await.unwrap().unwrap();
    assert!(ready.ready);
    assert_eq!(ready.peak_additional_bytes, 0);
    assert_eq!(
        std::fs::read(root.path().join("fixture/models/one/native-v1/model.bin")).unwrap(),
        b"abc"
    );
}
