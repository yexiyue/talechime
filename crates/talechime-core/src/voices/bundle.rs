//! Single-file voice bundles for moving library assets between machines.
//!
//! A bundle is a zip container holding `manifest.json` (format version plus the
//! full [`VoiceAsset`] record) and the verbatim `reference.wav`. Readers accept
//! only the current format version and fail explicitly on anything else.

use super::library::{VoiceAsset, VoiceLibrary};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::Path;

pub const BUNDLE_FORMAT: &str = "talechime/voice-bundle/1";

#[derive(Serialize, Deserialize)]
struct BundleManifest {
    format: String,
    asset: VoiceAsset,
}

fn write_bundle(path: &Path, asset: &VoiceAsset, wav: &[u8]) -> anyhow::Result<()> {
    let manifest = BundleManifest {
        format: BUNDLE_FORMAT.into(),
        asset: asset.clone(),
    };
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("bundle path needs a parent directory"))?;
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    let file = tempfile::NamedTempFile::new_in(parent)?;
    {
        let mut archive = zip::ZipWriter::new(file.reopen()?);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        archive.start_file("manifest.json", options)?;
        serde_json::to_writer_pretty(&mut archive, &manifest)?;
        archive.start_file("reference.wav", options)?;
        archive.write_all(wav)?;
        archive.finish()?;
    }
    file.persist(path).map_err(|error| error.error)?;
    Ok(())
}

fn read_bundle(path: &Path) -> anyhow::Result<(VoiceAsset, Vec<u8>)> {
    let file = std::fs::File::open(path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    let mut manifest_entry = archive
        .by_name("manifest.json")
        .map_err(|_| anyhow::anyhow!("voice bundle is missing manifest.json"))?;
    let mut manifest_bytes = Vec::new();
    manifest_entry.read_to_end(&mut manifest_bytes)?;
    drop(manifest_entry);
    let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes)?;
    anyhow::ensure!(
        manifest.format == BUNDLE_FORMAT,
        "unsupported voice bundle format {} (expected {BUNDLE_FORMAT}); refusing to import",
        manifest.format
    );
    let mut wav_entry = archive
        .by_name("reference.wav")
        .map_err(|_| anyhow::anyhow!("voice bundle is missing reference.wav"))?;
    let mut wav = Vec::new();
    wav_entry.read_to_end(&mut wav)?;
    let computed = super::library::fingerprint(
        &manifest.asset.transcript,
        manifest.asset.description.as_deref(),
        &wav,
    );
    anyhow::ensure!(
        computed == manifest.asset.fingerprint,
        "voice bundle contents do not match its manifest; refusing to import"
    );
    Ok((manifest.asset, wav))
}

impl VoiceLibrary {
    /// Export one voice as a single self-contained file.
    pub fn export(&self, id: &str, path: &Path) -> anyhow::Result<()> {
        let (asset, wav) = self.load(id)?;
        write_bundle(path, &asset, &wav)
    }

    /// Import a bundle; `new_id` overrides the recorded ID (required on collision).
    /// Source provenance is preserved; the content fingerprint is recomputed and
    /// must match the manifest. Creation time records the import.
    pub fn import_bundle(&self, path: &Path, new_id: Option<&str>) -> anyhow::Result<VoiceAsset> {
        let (mut asset, wav) = read_bundle(path)?;
        if let Some(new_id) = new_id {
            asset.id = new_id.into();
        }
        self.create(
            &asset.id,
            &asset.name,
            &wav,
            &asset.transcript,
            asset.description.as_deref(),
            asset.source,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::library::VoiceSource;
    use super::super::library::tests::wav_fixture;
    use super::*;

    #[test]
    fn export_accepts_a_bare_relative_filename() {
        let root = tempfile::tempdir().unwrap();
        let library = VoiceLibrary::open(root.path()).unwrap();
        library
            .create(
                "v",
                "名字",
                &wav_fixture(),
                "转写",
                None,
                VoiceSource::Imported,
            )
            .unwrap();
        // A unique caller-owned file keeps this test independent of other tests
        // without changing the process-wide working directory.
        let output = tempfile::NamedTempFile::new_in(".").unwrap();
        let relative = Path::new(output.path().file_name().unwrap());
        assert!(relative.parent().unwrap().as_os_str().is_empty());
        library.export("v", relative).unwrap();
        let (asset, wav) = read_bundle(relative).unwrap();
        assert_eq!(asset.id, "v");
        assert_eq!(wav, wav_fixture());
    }

    #[test]
    fn export_rejects_corrupt_snapshots_and_lock_contention_without_overwrite() {
        let root = tempfile::tempdir().unwrap();
        let library = VoiceLibrary::open(root.path()).unwrap();
        library
            .create(
                "v",
                "名字",
                &wav_fixture(),
                "转写",
                None,
                VoiceSource::Imported,
            )
            .unwrap();
        let output = root.path().join("v.voice");
        library.export("v", &output).unwrap();
        let original = std::fs::read(&output).unwrap();
        let lock = crate::storage::lock(&root.path().join(".lock")).unwrap();
        assert!(library.export("v", &output).is_err());
        drop(lock);
        let mut changed = wav_fixture();
        changed[44] ^= 1;
        std::fs::write(root.path().join("v/reference.wav"), changed).unwrap();
        assert!(
            library
                .export("v", &output)
                .unwrap_err()
                .to_string()
                .contains("fingerprint")
        );
        assert_eq!(std::fs::read(output).unwrap(), original);
    }

    #[test]
    fn unknown_bundle_version_is_rejected_without_creating_a_record() {
        let root = tempfile::tempdir().unwrap();
        let library = VoiceLibrary::open(root.path().join("library")).unwrap();
        let bundle = root.path().join("future.voice");
        let mut archive = zip::ZipWriter::new(std::fs::File::create(&bundle).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        archive.start_file("manifest.json", options).unwrap();
        let wav = wav_fixture();
        let asset = VoiceAsset {
            id: "v".into(),
            name: "名字".into(),
            transcript: "转写".into(),
            description: None,
            source: VoiceSource::Imported,
            created_at_unix: 0,
            fingerprint: super::super::library::fingerprint("转写", None, &wav),
        };
        serde_json::to_writer(
            &mut archive,
            &BundleManifest {
                format: "talechime/voice-bundle/999".into(),
                asset,
            },
        )
        .unwrap();
        archive.start_file("reference.wav", options).unwrap();
        archive.write_all(&wav).unwrap();
        archive.finish().unwrap();
        assert!(
            library
                .import_bundle(&bundle, None)
                .unwrap_err()
                .to_string()
                .contains("unsupported voice bundle format")
        );
        assert!(!library.root().exists());
    }

    #[test]
    fn bundle_roundtrips_across_libraries() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let left = VoiceLibrary::open(first.path()).unwrap();
        let right = VoiceLibrary::open(second.path()).unwrap();
        let wav = wav_fixture();
        left.create(
            "hero",
            "主角",
            &wav,
            "试音转写",
            Some("低沉温和"),
            VoiceSource::Designed("an-engine".into()),
        )
        .unwrap();
        let bundle = first.path().join("hero.voice");
        left.export("hero", &bundle).unwrap();
        let asset = right.import_bundle(&bundle, None).unwrap();
        assert_eq!(asset.id, "hero");
        assert_eq!(right.read_wav("hero").unwrap(), wav);
        assert_eq!(asset.source, VoiceSource::Designed("an-engine".into()));
        // Collision forces an explicit new ID.
        let error = right.import_bundle(&bundle, None).unwrap_err();
        assert!(error.to_string().contains("already exists"));
        let renamed = right.import_bundle(&bundle, Some("hero-2")).unwrap();
        assert_eq!(renamed.id, "hero-2");
    }

    #[test]
    fn tampered_or_foreign_bundles_fail_explicitly() {
        let root = tempfile::tempdir().unwrap();
        let library = VoiceLibrary::open(root.path()).unwrap();
        let wav = wav_fixture();
        library
            .create("v", "名字", &wav, "转写", None, VoiceSource::Imported)
            .unwrap();
        let bundle = root.path().join("v.voice");
        library.export("v", &bundle).unwrap();
        // Tamper: flip a WAV byte inside a rewritten zip with the same manifest.
        let (asset, mut wav) = read_bundle(&bundle).unwrap();
        wav[10] ^= 0xff;
        let tampered = root.path().join("tampered.voice");
        write_bundle(&tampered, &asset, &wav).unwrap();
        assert!(
            library
                .import_bundle(&tampered, None)
                .unwrap_err()
                .to_string()
                .contains("do not match")
        );
        // Foreign format is refused, not guessed.
        let foreign = root.path().join("foreign.json");
        std::fs::write(&foreign, "{}").unwrap();
        assert!(library.import_bundle(&foreign, None).is_err());
    }
}
