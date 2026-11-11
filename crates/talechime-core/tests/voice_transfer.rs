//! Public, model-free transfer contracts using synthetic audio only.
use std::{io::Write, path::PathBuf};
use talechime_core::voices::*;

#[test]
fn full_batch_roundtrip_uses_bounded_file_descriptors() {
    let root = tempfile::tempdir().unwrap();
    let source = VoiceLibrary::open(root.path().join("source")).unwrap();
    let destination = VoiceLibrary::open(root.path().join("destination")).unwrap();
    let ids: Vec<String> = (0..256).map(|n| format!("voice-{n}")).collect();
    for (n, id) in ids.iter().enumerate() {
        voice(&source, id, n as i16 + 1);
    }
    let package = export(
        &source,
        &ids.iter().map(String::as_str).collect::<Vec<_>>(),
        root.path().join("all.voice"),
    );
    let plan = plan(&destination, &[package], ImportAction::Import);
    let report = destination.import_selected(plan, &TransferControl::default());
    assert_eq!(report.items.len(), 256);
    assert!(
        report
            .items
            .iter()
            .all(|item| matches!(item.status, ImportStatus::Imported))
    );
    assert_eq!(destination.inventory().unwrap().assets.len(), 256);
}
fn wav(seed: i16) -> Vec<u8> {
    let mut out = std::io::Cursor::new(Vec::new());
    let mut w = hound::WavWriter::new(
        &mut out,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for sample in [seed, -seed, seed, -seed] {
        w.write_sample(sample).unwrap();
    }
    w.finalize().unwrap();
    out.into_inner()
}
fn voice(library: &VoiceLibrary, id: &str, seed: i16) -> VoiceAsset {
    library
        .create(
            id,
            "声音",
            &wav(seed),
            "你好",
            Some("温和"),
            VoiceSource::Imported,
        )
        .unwrap()
}
fn export(library: &VoiceLibrary, ids: &[&str], path: PathBuf) -> PathBuf {
    library
        .export_bundle(
            &ids.iter().map(|id| id.parse().unwrap()).collect::<Vec<_>>(),
            &path,
            ExportTarget::CreateNew,
            TransferLimits::default(),
            &TransferControl::default(),
        )
        .unwrap();
    path
}
fn plan(library: &VoiceLibrary, paths: &[PathBuf], action: ImportAction) -> ImportPlan {
    let preview = library
        .inspect_import(
            paths,
            TransferLimits::default(),
            &TransferControl::default(),
        )
        .unwrap();
    let decisions = preview
        .entries()
        .iter()
        .map(|e| ImportDecision {
            entry: e.index,
            action,
        })
        .collect();
    library.plan_import(preview, decisions).unwrap()
}
#[test]
fn batch_roundtrip_preserves_identity_provenance_and_created_time() {
    let from = tempfile::tempdir().unwrap();
    let to = tempfile::tempdir().unwrap();
    let left = VoiceLibrary::open(from.path().join("assets")).unwrap();
    let right = VoiceLibrary::open(to.path()).unwrap();
    let first = voice(&left, "one", 100);
    voice(&left, "two", 200);
    let path = export(&left, &["one", "two"], from.path().join("all.voice"));
    let report = right.import_selected(
        plan(&right, &[path], ImportAction::Import),
        &TransferControl::default(),
    );
    assert!(
        report
            .items
            .iter()
            .all(|r| r.status == ImportStatus::Imported)
    );
    let copied = right.get("one").unwrap();
    assert_eq!(copied.asset_digest(), first.asset_digest());
    assert_eq!(copied.created_at_unix(), first.created_at_unix());
    assert_eq!(copied.source(), first.source());
    assert_eq!(right.read_wav("one").unwrap(), wav(100));
    assert_eq!(right.list().unwrap().len(), 2);
}
#[test]
fn reference_identity_deserializes_only_valid_ids() {
    assert!(serde_json::from_str::<VoiceId>("\"../evil\"").is_err());
    assert_eq!("hero_1".parse::<VoiceId>().unwrap().as_str(), "hero_1");
}
#[test]
fn metadata_changes_do_not_change_reference_fingerprint() {
    let root = tempfile::tempdir().unwrap();
    let lib = VoiceLibrary::open(root.path()).unwrap();
    let old = voice(&lib, "one", 100);
    let renamed = lib.rename("one", "新名字").unwrap();
    assert_eq!(old.fingerprint(), renamed.fingerprint());
    assert_ne!(old.asset_digest(), renamed.asset_digest());
    let described = lib.describe("one", Some("低沉男声，语速偏慢")).unwrap();
    assert_eq!(described.fingerprint(), old.fingerprint());
    assert_ne!(described.asset_digest(), renamed.asset_digest());
    assert_eq!(described.description(), Some("低沉男声，语速偏慢"));
    let cleared = lib.describe("one", None).unwrap();
    assert_eq!(cleared.description(), None);
}
#[test]
fn old_records_are_refused_and_left_untouched() {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join("old");
    std::fs::create_dir(&directory).unwrap();
    let data = b"{\"format\":\"talechime/voice/1\"}";
    std::fs::write(directory.join("voice.json"), data).unwrap();
    let lib = VoiceLibrary::open(root.path()).unwrap();
    assert!(matches!(
        lib.get("old"),
        Err(VoiceLibraryError::UnsupportedFormat(_))
    ));
    assert_eq!(std::fs::read(directory.join("voice.json")).unwrap(), data);
}
#[test]
fn inventory_keeps_valid_assets_when_one_record_is_damaged() {
    let root = tempfile::tempdir().unwrap();
    let lib = VoiceLibrary::open(root.path()).unwrap();
    voice(&lib, "good", 1);
    std::fs::create_dir(root.path().join("broken")).unwrap();
    let inventory = lib.inventory().unwrap();
    assert_eq!(inventory.assets.len(), 1);
    assert_eq!(inventory.diagnostics[0].id, "broken");
}
#[test]
fn frozen_preview_ignores_replaced_input_and_honors_selection() {
    let root = tempfile::tempdir().unwrap();
    let source = VoiceLibrary::open(root.path().join("source")).unwrap();
    voice(&source, "one", 1);
    voice(&source, "two", 2);
    let path = export(&source, &["one", "two"], root.path().join("all.voice"));
    let dest = VoiceLibrary::open(root.path().join("dest")).unwrap();
    let preview = dest
        .inspect_import(
            std::slice::from_ref(&path),
            TransferLimits::default(),
            &TransferControl::default(),
        )
        .unwrap();
    assert!(!dest.root().exists());
    std::fs::write(&path, b"changed by another process").unwrap();
    let plan = dest
        .plan_import(
            preview,
            vec![ImportDecision {
                entry: 0,
                action: ImportAction::Import,
            }],
        )
        .unwrap();
    assert_eq!(
        dest.import_selected(plan, &TransferControl::default())
            .items[0]
            .status,
        ImportStatus::Imported
    );
    assert_eq!(dest.list().unwrap().len(), 1);
    assert_eq!(dest.read_wav("one").unwrap(), wav(1));
}
#[test]
fn identity_conflicts_require_decision_and_new_identity_replays_are_idempotent() {
    let root = tempfile::tempdir().unwrap();
    let source = VoiceLibrary::open(root.path().join("source")).unwrap();
    let dest = VoiceLibrary::open(root.path().join("dest")).unwrap();
    voice(&source, "hero", 1);
    voice(&dest, "hero", 2);
    let path = export(&source, &["hero"], root.path().join("one.voice"));
    let preview = dest
        .inspect_import(
            std::slice::from_ref(&path),
            TransferLimits::default(),
            &TransferControl::default(),
        )
        .unwrap();
    assert!(
        preview.entries()[0]
            .conflicts
            .contains(&ImportConflict::IdentityConflict)
    );
    assert!(
        dest.plan_import(
            preview,
            vec![ImportDecision {
                entry: 0,
                action: ImportAction::Import
            }]
        )
        .is_err()
    );
    let first = dest.import_selected(
        plan(
            &dest,
            std::slice::from_ref(&path),
            ImportAction::ImportAsNew,
        ),
        &TransferControl::default(),
    );
    let second = dest.import_selected(
        plan(&dest, &[path], ImportAction::ImportAsNew),
        &TransferControl::default(),
    );
    assert_eq!(second.items[0].status, ImportStatus::AlreadyPresent);
    assert_eq!(first.items[0].local_id, second.items[0].local_id);
    assert_eq!(dest.list().unwrap().len(), 2);
    assert_eq!(dest.read_wav("hero").unwrap(), wav(2));
}
#[test]
fn conflicts_between_packages_are_detected_before_writing() {
    let root = tempfile::tempdir().unwrap();
    let a = VoiceLibrary::open(root.path().join("a")).unwrap();
    let b = VoiceLibrary::open(root.path().join("b")).unwrap();
    voice(&a, "hero", 1);
    voice(&b, "hero", 2);
    let paths = [
        export(&a, &["hero"], root.path().join("a.voice")),
        export(&b, &["hero"], root.path().join("b.voice")),
    ];
    let dest = VoiceLibrary::open(root.path().join("dest")).unwrap();
    let p = dest
        .inspect_import(
            &paths,
            TransferLimits::default(),
            &TransferControl::default(),
        )
        .unwrap();
    assert!(
        p.entries()
            .iter()
            .all(|e| e.conflicts.contains(&ImportConflict::IdentityConflict))
    );
    assert!(!dest.root().exists());
}
#[test]
fn identical_packages_merge_sources_without_duplicate_records() {
    let root = tempfile::tempdir().unwrap();
    let source = VoiceLibrary::open(root.path().join("a")).unwrap();
    voice(&source, "hero", 1);
    let path = export(&source, &["hero"], root.path().join("a.voice"));
    let dest = VoiceLibrary::open(root.path().join("dest")).unwrap();
    let preview = dest
        .inspect_import(
            &[path.clone(), path],
            TransferLimits::default(),
            &TransferControl::default(),
        )
        .unwrap();
    assert_eq!(preview.entries().len(), 1);
    assert_eq!(preview.entries()[0].sources.len(), 2);
}
#[test]
fn target_occupied_after_preview_is_not_overwritten() {
    let root = tempfile::tempdir().unwrap();
    let source = VoiceLibrary::open(root.path().join("a")).unwrap();
    voice(&source, "hero", 1);
    let path = export(&source, &["hero"], root.path().join("a.voice"));
    let dest = VoiceLibrary::open(root.path().join("dest")).unwrap();
    let plan = plan(&dest, &[path], ImportAction::Import);
    voice(&dest, "hero", 2);
    let report = dest.import_selected(plan, &TransferControl::default());
    assert_eq!(report.items[0].status, ImportStatus::Failed);
    assert_eq!(dest.read_wav("hero").unwrap(), wav(2));
}
#[test]
fn cancellation_between_publications_preserves_completed_items() {
    let root = tempfile::tempdir().unwrap();
    let source = VoiceLibrary::open(root.path().join("a")).unwrap();
    voice(&source, "one", 1);
    voice(&source, "two", 2);
    let path = export(&source, &["one", "two"], root.path().join("a.voice"));
    let dest = VoiceLibrary::open(root.path().join("dest")).unwrap();
    let holder = std::sync::Arc::new(std::sync::Mutex::new(None::<TransferControl>));
    let h = holder.clone();
    let control = TransferControl::with_progress(move |p| {
        if matches!(p.stage, TransferStage::Committing) && p.completed == 1 {
            h.lock().unwrap().as_ref().unwrap().cancel()
        }
    });
    *holder.lock().unwrap() = Some(control.clone());
    let report = dest.import_selected(plan(&dest, &[path], ImportAction::Import), &control);
    assert_eq!(report.items[0].status, ImportStatus::Imported);
    assert_eq!(report.items[1].status, ImportStatus::Cancelled);
    assert_eq!(dest.list().unwrap().len(), 1);
}
#[test]
fn complete_export_is_atomic_and_existing_target_requires_explicit_replacement() {
    let root = tempfile::tempdir().unwrap();
    let lib = VoiceLibrary::open(root.path().join("lib")).unwrap();
    voice(&lib, "hero", 1);
    let target = root.path().join("existing.voice");
    std::fs::write(&target, b"keep").unwrap();
    let ids = vec!["hero".parse().unwrap()];
    let call = |target_mode| {
        lib.export_bundle(
            &ids,
            &target,
            target_mode,
            TransferLimits::default(),
            &TransferControl::default(),
        )
    };
    assert!(matches!(
        call(ExportTarget::CreateNew),
        Err(VoiceLibraryError::Conflict(_))
    ));
    assert_eq!(std::fs::read(&target).unwrap(), b"keep");
    call(ExportTarget::ReplaceConfirmed).unwrap();
    assert_ne!(std::fs::read(&target).unwrap(), b"keep");
}
#[test]
fn cancelled_and_corrupt_exports_preserve_previous_file() {
    let root = tempfile::tempdir().unwrap();
    let lib = VoiceLibrary::open(root.path().join("lib")).unwrap();
    voice(&lib, "hero", 1);
    let target = root.path().join("existing.voice");
    std::fs::write(&target, b"keep").unwrap();
    let control = TransferControl::default();
    control.cancel();
    assert!(matches!(
        lib.export_bundle(
            &["hero".parse().unwrap()],
            &target,
            ExportTarget::ReplaceConfirmed,
            TransferLimits::default(),
            &control
        ),
        Err(VoiceLibraryError::Cancelled)
    ));
    std::fs::write(lib.root().join("hero/reference.wav"), wav(2)).unwrap();
    assert!(
        lib.export_bundle(
            &["hero".parse().unwrap()],
            &target,
            ExportTarget::ReplaceConfirmed,
            TransferLimits::default(),
            &TransferControl::default()
        )
        .is_err()
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"keep");
}
#[test]
fn tightened_limits_and_invalid_inputs_never_publish_assets() {
    let root = tempfile::tempdir().unwrap();
    let lib = VoiceLibrary::open(root.path().join("lib")).unwrap();
    voice(&lib, "hero", 1);
    let path = export(&lib, &["hero"], root.path().join("one.voice"));
    let dest = VoiceLibrary::open(root.path().join("dest")).unwrap();
    for limits in [
        TransferLimits {
            max_input_bytes: 1,
            ..Default::default()
        },
        TransferLimits {
            max_wav_bytes: 1,
            ..Default::default()
        },
        TransferLimits {
            max_manifest_bytes: 1,
            ..Default::default()
        },
        TransferLimits {
            max_assets: 257,
            ..Default::default()
        },
    ] {
        assert!(
            dest.inspect_import(
                std::slice::from_ref(&path),
                limits,
                &TransferControl::default()
            )
            .is_err()
        );
        assert!(!dest.root().exists());
    }
}
#[test]
fn archive_rejects_legacy_unknown_fields_extra_entries_and_truncated_audio() {
    let root = tempfile::tempdir().unwrap();
    let lib = VoiceLibrary::open(root.path().join("source")).unwrap();
    voice(&lib, "hero", 1);
    let original = export(&lib, &["hero"], root.path().join("original.voice"));
    for corruption in ["legacy", "unknown", "extra", "truncated", "hash", "missing"] {
        let mut archive = zip::ZipArchive::new(std::fs::File::open(&original).unwrap()).unwrap();
        let mut data = String::new();
        std::io::Read::read_to_string(&mut archive.by_name("manifest.json").unwrap(), &mut data)
            .unwrap();
        let mut manifest: serde_json::Value = serde_json::from_str(&data).unwrap();
        match corruption {
            "legacy" => manifest["format"] = "talechime/voice-bundle/1".into(),
            "unknown" => manifest["surprise"] = true.into(),
            "hash" => manifest["assets"][0]["fingerprint"] = "bad".into(),
            _ => {}
        }
        let path = root.path().join(format!("{corruption}.voice"));
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        zip.start_file("manifest.json", options).unwrap();
        zip.write_all(&serde_json::to_vec(&manifest).unwrap())
            .unwrap();
        if corruption != "missing" {
            zip.start_file("voices/hero/reference.wav", options)
                .unwrap();
            let mut audio = wav(1);
            if corruption == "truncated" {
                audio.pop();
            }
            zip.write_all(&audio).unwrap();
        }
        if corruption == "extra" {
            zip.start_file("../evil", options).unwrap();
            zip.write_all(b"bad").unwrap();
        }
        zip.finish().unwrap();
        let dest = VoiceLibrary::open(root.path().join("dest")).unwrap();
        assert!(
            dest.inspect_import(
                &[path],
                TransferLimits::default(),
                &TransferControl::default()
            )
            .is_err(),
            "{corruption}"
        );
        assert!(!dest.root().exists());
    }
}

#[test]
fn physical_duplicate_zip_entries_and_excessive_directory_count_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    let source = VoiceLibrary::open(root.path().join("source")).unwrap();
    voice(&source, "hero", 1);
    let original = export(&source, &["hero"], root.path().join("original.voice"));
    let mut archive = zip::ZipArchive::new(std::fs::File::open(&original).unwrap()).unwrap();
    let mut manifest = Vec::new();
    std::io::Read::read_to_end(
        &mut archive.by_name("manifest.json").unwrap(),
        &mut manifest,
    )
    .unwrap();
    let target = root.path().join("duplicate.voice");
    let mut writer = zip::ZipWriter::new(std::fs::File::create(&target).unwrap());
    let options = zip::write::SimpleFileOptions::default();
    writer.start_file("manifest.json", options).unwrap();
    writer.write_all(&manifest).unwrap();
    for name in ["voices/hero/reference.wav", "voices/her0/reference.wav"] {
        writer.start_file(name, options).unwrap();
        writer.write_all(&wav(1)).unwrap();
    }
    writer.finish().unwrap();
    let mut bytes = std::fs::read(&target).unwrap();
    for i in 0..bytes.len() - 3 {
        if &bytes[i..i + 4] == b"her0" {
            bytes[i + 3] = b'o';
        }
    }
    std::fs::write(&target, bytes).unwrap();
    let dest = VoiceLibrary::open(root.path().join("dest")).unwrap();
    let error = dest
        .inspect_import(
            &[target],
            TransferLimits::default(),
            &TransferControl::default(),
        )
        .err()
        .unwrap();
    assert!(error.to_string().contains("duplicate ZIP"));
    let mut bytes = std::fs::read(original).unwrap();
    let n = bytes.len();
    bytes[n - 14..n - 10].copy_from_slice(&[255; 4]);
    let bomb = root.path().join("many.voice");
    std::fs::write(&bomb, bytes).unwrap();
    assert!(
        dest.inspect_import(
            &[bomb],
            TransferLimits::default(),
            &TransferControl::default()
        )
        .is_err()
    );
    assert!(!dest.root().exists());
}
#[test]
fn metadata_conflicts_and_duplicate_content_are_distinct_from_names() {
    let root = tempfile::tempdir().unwrap();
    let source = VoiceLibrary::open(root.path().join("source")).unwrap();
    let dest = VoiceLibrary::open(root.path().join("dest")).unwrap();
    voice(&source, "hero", 1);
    voice(&dest, "hero", 1);
    dest.rename("hero", "本地名字").unwrap();
    voice(&dest, "other", 1);
    let path = export(&source, &["hero"], root.path().join("one.voice"));
    let preview = dest
        .inspect_import(
            std::slice::from_ref(&path),
            TransferLimits::default(),
            &TransferControl::default(),
        )
        .unwrap();
    let conflicts = &preview.entries()[0].conflicts;
    assert!(conflicts.contains(&ImportConflict::MetadataDifference));
    assert!(conflicts.contains(&ImportConflict::DuplicateContent));
    assert!(conflicts.contains(&ImportConflict::SameName));
    assert!(!conflicts.contains(&ImportConflict::IdentityConflict));
    let report = dest.import_selected(
        plan(&dest, &[path], ImportAction::Import),
        &TransferControl::default(),
    );
    assert_eq!(report.items[0].status, ImportStatus::AlreadyPresent);
    assert_eq!(dest.get("hero").unwrap().name(), "本地名字");
}
#[test]
fn package_count_manifest_total_and_audio_total_limits_are_enforced() {
    let root = tempfile::tempdir().unwrap();
    let source = VoiceLibrary::open(root.path().join("source")).unwrap();
    voice(&source, "hero", 1);
    let path = export(&source, &["hero"], root.path().join("one.voice"));
    let dest = VoiceLibrary::open(root.path().join("dest")).unwrap();
    for limits in [
        TransferLimits {
            max_packages: 1,
            ..Default::default()
        },
        TransferLimits {
            max_total_manifest_bytes: 1,
            ..Default::default()
        },
        TransferLimits {
            max_audio_bytes: 1,
            ..Default::default()
        },
    ] {
        assert!(
            dest.inspect_import(
                &[path.clone(), path.clone()],
                limits,
                &TransferControl::default()
            )
            .is_err()
        );
    }
    let control = TransferControl::default();
    control.cancel();
    assert!(matches!(
        dest.inspect_import(&[path], TransferLimits::default(), &control),
        Err(VoiceLibraryError::Input { .. })
    ));
    assert!(!dest.root().exists());
}

#[test]
fn export_cannot_replace_an_internal_asset_record() {
    let root = tempfile::tempdir().unwrap();
    let library = VoiceLibrary::open(root.path()).unwrap();
    voice(&library, "hero", 1);
    let target = root.path().join("hero/voice.json");
    let previous = std::fs::read(&target).unwrap();
    assert!(matches!(
        library.export_bundle(
            &["hero".parse().unwrap()],
            &target,
            ExportTarget::ReplaceConfirmed,
            TransferLimits::default(),
            &TransferControl::default()
        ),
        Err(VoiceLibraryError::Conflict(_))
    ));
    assert_eq!(std::fs::read(&target).unwrap(), previous);
    assert!(library.get("hero").is_ok());
}
