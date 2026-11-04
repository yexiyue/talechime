//! Synthetic boundary fixtures, never real speech or model data.
//! cargo run -p talechime-core --example voice_transfer_stress -- /tmp/voice-transfer-boundaries
use talechime_core::voices::{
    ExportTarget, TransferControl, TransferLimits, VoiceLibrary, VoiceSource,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .ok_or("provide an output directory")?;
    std::fs::create_dir_all(&output)?;
    let root = tempfile::tempdir()?;
    let library = VoiceLibrary::open(root.path().join("voices"))?;
    let mut bytes = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(
        &mut bytes,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for _ in 0..(63 * 1024 * 1024 / 2) {
        writer.write_sample(1i16)?;
    }
    writer.finalize()?;
    library.create(
        "near-audio-limit",
        "大小边界样本",
        bytes.get_ref(),
        "合成测试信号，不是真人录音",
        None,
        VoiceSource::Imported,
    )?;
    let control = TransferControl::default();
    let limits = TransferLimits::default();
    library.export_bundle(
        &["near-audio-limit".parse()?],
        &output.join("near-audio-limit.voice"),
        ExportTarget::ReplaceConfirmed,
        limits.clone(),
        &control,
    )?;
    drop(bytes);
    // Rebuild a minimal valid WAV rather than truncating a declared long recording.
    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(
        &mut cursor,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    writer.write_sample(1i16)?;
    writer.finalize()?;
    let mut ids = Vec::new();
    for n in 0..256 {
        let id = format!("count-{n:03}");
        library.create(
            &id,
            &format!("数量边界 {n}"),
            cursor.get_ref(),
            "合成测试信号",
            (n < 12).then_some("筛选范围十二项"),
            VoiceSource::Imported,
        )?;
        ids.push(id.parse()?);
    }
    library.export_bundle(
        &ids[..50],
        &output.join("filter-fifty.voice"),
        ExportTarget::ReplaceConfirmed,
        limits.clone(),
        &control,
    )?;
    library.export_bundle(
        &ids,
        &output.join("near-count-limit.voice"),
        ExportTarget::ReplaceConfirmed,
        limits,
        &control,
    )?;
    library.create(
        "hero",
        "冲突音色",
        cursor.get_ref(),
        "不同的合成参考信号",
        None,
        VoiceSource::Imported,
    )?;
    library.export_bundle(
        &["hero".parse()?],
        &output.join("conflict.voice"),
        ExportTarget::ReplaceConfirmed,
        TransferLimits::default(),
        &control,
    )?;
    println!("{}", output.display());
    Ok(())
}
