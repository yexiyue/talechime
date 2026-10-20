//! Export only manifest-approved report assets, never recurse over model directories.
use super::{evidence::contained, report::verify_report};
use anyhow::{Context, ensure};
use std::{io::Write, path::Path};

pub fn pack(report: &Path, output: &Path) -> anyhow::Result<()> {
    ensure!(!output.exists(), "archive already exists");
    let manifest = verify_report(report)?;
    let mut total = 0u64;
    for artifact in &manifest.artifacts {
        total = total
            .checked_add(artifact.bytes)
            .context("archive size overflow")?;
    }
    ensure!(
        total <= 8 * 1024 * 1024 * 1024,
        "archive exceeds 8 GiB export budget"
    );
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut stage = tempfile::NamedTempFile::new_in(parent)?;
    {
        let mut archive = zip::ZipWriter::new(stage.as_file_mut());
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        for artifact in &manifest.artifacts {
            let source = contained(report, &artifact.path)?;
            let name = artifact
                .path
                .components()
                .map(|part| part.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            archive.start_file(name, options)?;
            std::io::copy(&mut std::fs::File::open(source)?, &mut archive)?;
        }
        archive.start_file("manifest.json", options)?;
        archive.write_all(&serde_json::to_vec_pretty(&manifest)?)?;
        archive.finish()?;
    }
    stage.as_file().sync_all()?;
    {
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let mut archive = zip::ZipArchive::new(stage.reopen()?)?;
        for artifact in &manifest.artifacts {
            let name = artifact
                .path
                .components()
                .map(|part| part.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            let mut entry = archive.by_name(&name)?;
            let mut hash = Sha256::new();
            let mut buffer = [0; 65536];
            let mut bytes = 0u64;
            loop {
                let n = entry.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
                bytes += n as u64;
            }
            ensure!(
                bytes == artifact.bytes && format!("{:x}", hash.finalize()) == artifact.sha256,
                "archive asset changed during export"
            );
        }
    }
    stage.persist_noclobber(output)?;
    Ok(())
}
