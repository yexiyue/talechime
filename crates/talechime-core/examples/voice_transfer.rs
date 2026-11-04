//! Run without models or network: cargo run -p talechime-core --example voice_transfer
use talechime_core::voices::{
    ExportTarget, ImportAction, ImportDecision, TransferControl, TransferLimits, VoiceLibrary,
    VoiceSource,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = tempfile::tempdir()?;
    let source = VoiceLibrary::open(root.path().join("source"))?;
    for (id, seed) in [("reader", 100i16), ("hero", 200)] {
        let mut bytes = std::io::Cursor::new(Vec::new());
        let mut wav = hound::WavWriter::new(
            &mut bytes,
            hound::WavSpec {
                channels: 1,
                sample_rate: 24000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for sample in [seed, -seed, seed, -seed] {
            wav.write_sample(sample)?;
        }
        wav.finalize()?;
        source.create(
            id,
            id,
            &bytes.into_inner(),
            "你好",
            None,
            VoiceSource::Imported,
        )?;
    }
    let output = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root.path().join("packages"));
    std::fs::create_dir_all(&output)?;
    for n in 0..18 {
        let recording = source.read_wav("reader")?;
        source.create(
            &format!("extra-{n}"),
            &format!("声音 {n}"),
            &recording,
            &format!("参考文本 {n}"),
            None,
            VoiceSource::Imported,
        )?;
    }
    let all = source.inventory()?.assets;
    for (name, assets) in [("first", &all[..10]), ("second", &all[10..])] {
        let ids = assets
            .iter()
            .map(|a| a.id().parse())
            .collect::<Result<Vec<_>, _>>()?;
        source.export_bundle(
            &ids,
            &output.join(format!("{name}.voice")),
            ExportTarget::ReplaceConfirmed,
            TransferLimits::default(),
            &TransferControl::default(),
        )?;
    }
    let package = output.join("voices.voice");
    source.export_bundle(
        &["reader".parse()?, "hero".parse()?],
        &package,
        ExportTarget::CreateNew,
        TransferLimits::default(),
        &TransferControl::default(),
    )?;
    let target = VoiceLibrary::open(root.path().join("target"))?;
    let preview = target.inspect_import(
        &[package],
        TransferLimits::default(),
        &TransferControl::default(),
    )?;
    let decisions = preview
        .entries()
        .iter()
        .map(|e| ImportDecision {
            entry: e.index,
            action: ImportAction::Import,
        })
        .collect();
    let plan = target.plan_import(preview, decisions)?;
    let report = target.import_selected(plan, &TransferControl::default());
    for item in report.items {
        println!("{:?}: {:?}", item.status, item.local_id);
    }
    assert_eq!(target.inventory()?.assets.len(), 2);
    Ok(())
}
