//! The facade exposes model-independent transfer without constructing a Registry.
use talechime::{
    ExportTarget, ImportAction, ImportDecision, ImportStatus, TransferControl, TransferLimits,
    VoiceId, VoiceLibrary, VoiceSource,
};
fn fixtures() -> (tempfile::TempDir, VoiceLibrary) {
    let root = tempfile::tempdir().unwrap();
    let lib = VoiceLibrary::open(root.path().join("source")).unwrap();
    let mut bytes = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(
        &mut bytes,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    writer.write_sample(1i16).unwrap();
    writer.finalize().unwrap();
    for id in ["reader", "hero"] {
        lib.create(id, id, bytes.get_ref(), "你好", None, VoiceSource::Imported)
            .unwrap();
    }
    (root, lib)
}
#[test]
fn facade_transfers_a_batch_with_no_model_runtime_or_network() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<VoiceLibrary>();
    send_sync::<TransferControl>();
    let (root, lib) = fixtures();
    let path = root.path().join("all.voice");
    lib.export_bundle(
        &[
            "reader".parse::<VoiceId>().unwrap(),
            "hero".parse().unwrap(),
        ],
        &path,
        ExportTarget::CreateNew,
        TransferLimits::default(),
        &TransferControl::default(),
    )
    .unwrap();
    let target = VoiceLibrary::open(root.path().join("dest")).unwrap();
    let p = target
        .inspect_import(
            &[path],
            TransferLimits::default(),
            &TransferControl::default(),
        )
        .unwrap();
    let decisions = p
        .entries()
        .iter()
        .map(|e| ImportDecision {
            entry: e.index,
            action: ImportAction::Import,
        })
        .collect();
    let report = target.import_selected(
        target.plan_import(p, decisions).unwrap(),
        &TransferControl::default(),
    );
    assert!(
        report
            .items
            .iter()
            .all(|i| i.status == ImportStatus::Imported)
    );
}
#[cfg(feature = "cli")]
#[test]
fn cli_accepts_multiple_ids_and_packages_without_synthesis_configuration() {
    let (root, lib) = fixtures();
    let source = root.path().join("cli-source");
    let dest = root.path().join("cli-dest");
    std::fs::create_dir(&source).unwrap();
    std::fs::create_dir(&dest).unwrap();
    let source_library = VoiceLibrary::open(source.join("voices-v2")).unwrap();
    for a in lib.list().unwrap() {
        source_library
            .create(
                a.id(),
                a.name(),
                &lib.read_wav(a.id()).unwrap(),
                a.transcript(),
                a.description(),
                a.source().clone(),
            )
            .unwrap();
    }
    let output = root.path().join("all.voice");
    let exported = std::process::Command::new(env!("CARGO_BIN_EXE_talechime"))
        .arg("--config")
        .arg(source.join("config.json"))
        .args(["voices", "export", "reader", "hero", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        exported.status.success(),
        "{}",
        String::from_utf8_lossy(&exported.stderr)
    );
    for extra in [vec!["--inspect"], vec![]] {
        let imported = std::process::Command::new(env!("CARGO_BIN_EXE_talechime"))
            .arg("--config")
            .arg(dest.join("config.json"))
            .args(["voices", "import-packages"])
            .arg(&output)
            .arg(&output)
            .args(extra)
            .output()
            .unwrap();
        assert!(
            imported.status.success(),
            "{}",
            String::from_utf8_lossy(&imported.stderr)
        );
    }
    assert_eq!(
        VoiceLibrary::open(dest.join("voices-v2"))
            .unwrap()
            .list()
            .unwrap()
            .len(),
        2
    );
    assert!(!source.join("config.json").exists());
    assert!(!dest.join("config.json").exists());
}
