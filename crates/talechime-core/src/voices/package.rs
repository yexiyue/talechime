//! Private, strict current-version archive codec. No archive paths are extracted.
use super::{
    TransferControl, TransferLimits, VoiceLibraryError as Error,
    store::{self, AssetData, Snapshot},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
};
const FORMAT: &str = "talechime/voice-bundle/2";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    format: String,
    assets: Vec<AssetData>,
}

pub(super) fn read(
    path: &Path,
    limits: &TransferLimits,
    control: &TransferControl,
    bytes: &mut u64,
    manifest_bytes: &mut u64,
    input_bytes: &mut u64,
) -> Result<Vec<Snapshot>, Error> {
    control.check()?;
    // Freeze the compressed input too: enforce the actual read budget, not a racy path stat.
    let mut source = File::open(path)?;
    let mut frozen = store::temporary()?;
    let mut buffer = [0u8; 65536];
    loop {
        control.check()?;
        let n = source.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        *input_bytes = input_bytes
            .checked_add(n as u64)
            .ok_or_else(|| Error::LimitExceeded("input bytes".into()))?;
        if *input_bytes > limits.max_input_bytes {
            return Err(Error::LimitExceeded("input bytes".into()));
        }
        frozen.write_all(&buffer[..n])?;
    }
    frozen.flush()?;
    validate_directory(frozen.as_file_mut(), limits.max_assets + 1)?;
    let mut archive = zip::ZipArchive::new(frozen.reopen()?).map_err(invalid)?;
    if archive.len() > limits.max_assets * 2 + 1 {
        return Err(Error::LimitExceeded("archive entries".into()));
    }
    let mut names = HashSet::new();
    for i in 0..archive.len() {
        let entry = archive.by_index(i).map_err(invalid)?;
        if entry.is_dir() || entry.is_symlink() || !names.insert(entry.name().to_owned()) {
            return Err(Error::InvalidPackage(
                "duplicate or non-file archive entry".into(),
            ));
        }
    }
    let data = store::bounded_bytes(
        archive.by_name("manifest.json").map_err(invalid)?,
        limits.max_manifest_bytes,
    )?;
    *manifest_bytes = manifest_bytes
        .checked_add(data.len() as u64)
        .ok_or_else(|| Error::LimitExceeded("manifests".into()))?;
    if *manifest_bytes > limits.max_total_manifest_bytes {
        return Err(Error::LimitExceeded("manifests".into()));
    }
    let header: serde_json::Value = serde_json::from_slice(&data).map_err(invalid)?;
    if header.get("format").and_then(|v| v.as_str()) != Some(FORMAT) {
        return Err(Error::UnsupportedFormat(
            header
                .get("format")
                .map(ToString::to_string)
                .unwrap_or_default(),
        ));
    }
    let manifest: Manifest = serde_json::from_slice(&data).map_err(invalid)?;
    if manifest.assets.is_empty() || manifest.assets.len() > limits.max_assets {
        return Err(Error::LimitExceeded("asset count".into()));
    }
    let mut ids = HashSet::new();
    let mut expected = HashSet::from(["manifest.json".to_owned()]);
    let mut result = Vec::new();
    let scratch = store::Scratch::new()?;
    for data in manifest.assets {
        control.check()?;
        let asset = data.validated(0)?;
        if !ids.insert(asset.id().to_owned()) {
            return Err(Error::InvalidPackage("duplicate asset ID".into()));
        }
        if asset.wav_bytes() > limits.max_wav_bytes {
            return Err(Error::LimitExceeded("reference WAV".into()));
        }
        *bytes = bytes
            .checked_add(asset.wav_bytes())
            .ok_or_else(|| Error::LimitExceeded("audio total".into()))?;
        if *bytes > limits.max_audio_bytes {
            return Err(Error::LimitExceeded("audio total".into()));
        }
        let name = format!("voices/{}/reference.wav", asset.id());
        expected.insert(name.clone());
        let mut source = archive.by_name(&name).map_err(invalid)?;
        if source.size() != asset.wav_bytes() {
            return Err(Error::ContentMismatch);
        }
        let mut wav = scratch.file()?;
        store::copy_checked(&mut source, &mut wav, &asset, control)?;
        result.push(Snapshot {
            asset,
            wav: scratch.freeze(wav),
        });
    }
    if expected != names {
        return Err(Error::InvalidPackage(
            "undeclared or missing archive entry".into(),
        ));
    }
    Ok(result)
}
pub(super) fn write(
    file: File,
    assets: &[Snapshot],
    limits: &TransferLimits,
    control: &TransferControl,
) -> Result<(), Error> {
    let manifest = Manifest {
        format: FORMAT.into(),
        assets: assets.iter().map(|s| AssetData::from(&s.asset)).collect(),
    };
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(invalid)?;
    if bytes.len() as u64 > limits.max_manifest_bytes {
        return Err(Error::LimitExceeded("manifest".into()));
    }
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    zip.start_file("manifest.json", options).map_err(invalid)?;
    zip.write_all(&bytes)?;
    let mut buffer = [0; 65536];
    let mut written = 0u64;
    for (i, snapshot) in assets.iter().enumerate() {
        control.check()?;
        zip.start_file(
            format!("voices/{}/reference.wav", snapshot.asset.id()),
            options,
        )
        .map_err(invalid)?;
        let mut source = File::open(snapshot.wav.path())?;
        loop {
            control.check()?;
            let n = source.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            zip.write_all(&buffer[..n])?;
            written += n as u64;
        }
        control.emit(super::TransferStage::Writing, i + 1, assets.len(), written);
    }
    let mut file = zip.finish().map_err(invalid)?;
    file.flush()?;
    file.sync_all()?;
    if file.metadata()?.len() > limits.max_input_bytes {
        return Err(Error::LimitExceeded("output package".into()));
    }
    Ok(())
}
fn invalid(e: impl std::fmt::Display) -> Error {
    Error::InvalidPackage(e.to_string())
}

/// Bound the central directory before zip allocates its metadata. Zip's indexed
/// reader deduplicates names, so uniqueness must be checked on the physical records.
fn validate_directory(file: &mut File, max_entries: usize) -> Result<(), Error> {
    let len = file.metadata()?.len();
    let tail_len = len.min(65557) as usize;
    file.seek(SeekFrom::End(-(tail_len as i64)))?;
    let mut tail = vec![0u8; tail_len];
    file.read_exact(&mut tail)?;
    let position = (0..tail.len().saturating_sub(21))
        .rev()
        .find(|&i| {
            tail[i..i + 4] == [0x50, 0x4b, 0x05, 0x06]
                && i + 22 + usize::from(u16::from_le_bytes([tail[i + 20], tail[i + 21]]))
                    == tail.len()
        })
        .ok_or_else(|| Error::InvalidPackage("missing ZIP directory".into()))?;
    let end = &tail[position..];
    let short = |i| u16::from_le_bytes([end[i], end[i + 1]]);
    let long = |i| u32::from_le_bytes([end[i], end[i + 1], end[i + 2], end[i + 3]]);
    let count = usize::from(short(10));
    let directory_size = u64::from(long(12));
    let directory_offset = u64::from(long(16));
    if count > max_entries || directory_size > 4 * 1024 * 1024 {
        return Err(Error::LimitExceeded("ZIP directory".into()));
    }
    if short(4) != 0
        || short(6) != 0
        || short(8) != short(10)
        || count == 0
        || directory_offset.checked_add(directory_size)
            != Some(len - tail_len as u64 + position as u64)
    {
        return Err(Error::InvalidPackage(
            "unsupported or excessive ZIP directory".into(),
        ));
    }
    file.seek(SeekFrom::Start(directory_offset))?;
    let mut directory = file.take(directory_size);
    let mut names = HashSet::new();
    for _ in 0..count {
        let mut header = [0u8; 46];
        directory.read_exact(&mut header)?;
        if header[..4] != [0x50, 0x4b, 0x01, 0x02] {
            return Err(Error::InvalidPackage("invalid ZIP directory entry".into()));
        }
        let name_len = usize::from(u16::from_le_bytes([header[28], header[29]]));
        let extra = usize::from(u16::from_le_bytes([header[30], header[31]]));
        let comment = usize::from(u16::from_le_bytes([header[32], header[33]]));
        if name_len > 128 || extra > 4096 || comment > 1024 {
            return Err(Error::LimitExceeded("ZIP entry metadata".into()));
        }
        let mut name = vec![0; name_len];
        directory.read_exact(&mut name)?;
        if !names.insert(name) {
            return Err(Error::InvalidPackage("duplicate ZIP entry".into()));
        }
        std::io::copy(
            &mut directory.by_ref().take((extra + comment) as u64),
            &mut std::io::sink(),
        )?;
    }
    if directory.limit() != 0 {
        return Err(Error::InvalidPackage("ZIP directory size mismatch".into()));
    }
    Ok(())
}
