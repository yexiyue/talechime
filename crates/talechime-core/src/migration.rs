//! Explicit executable migration; library path resolution remains side effect free.
use crate::paths::AppPaths;
use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};
use tts_protocol::{Capabilities, Config};

/// Copy-only migration from the retired worker's locations under one home.
#[derive(Debug, Clone)]
pub struct LegacyMigration {
    home: PathBuf,
    paths: AppPaths,
}

impl LegacyMigration {
    pub fn from_home(home: impl AsRef<Path>) -> Self {
        Self {
            home: home.as_ref().to_owned(),
            paths: AppPaths::from_home(home),
        }
    }

    /// Small startup settings. Existing new configuration always wins.
    pub fn config(&self, defaults: &Config, catalog: &[Capabilities]) -> std::io::Result<()> {
        let Some(source) = self.source(".novel/tts_config.json")? else {
            return Ok(());
        };
        self.run("config", || {
            if std::fs::symlink_metadata(self.paths.config()).is_ok() { return Ok(()) }
            let mut bytes = Vec::new();
            File::open(&source)?.take(2 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
            if bytes.len() > 2 * 1024 * 1024 { return Err(std::io::Error::other("legacy configuration exceeds 2 MiB")); }
            let raw: serde_json::Value = serde_json::from_slice(&bytes)?;
            let mut config: Config = serde_json::from_value(raw.clone())?;
            crate::config::validate_playback(config.volume, config.speed).map_err(std::io::Error::other)?;
            if raw.get("backend").is_none() || matches!(config.backend.as_str(), "kokoro" | "zipvoice") {
                config.backend.clone_from(&defaults.backend);
                config.model.clone_from(&defaults.model);
                config.voice.clone_from(&defaults.voice);
                config.tts_device = defaults.tts_device;
                config.style = None;
                config.revision = config.revision.checked_add(1).ok_or_else(|| std::io::Error::other("legacy revision overflow"))?;
                eprintln!("legacy retired backend migrated to {} / {:?}; original configuration retained at {}", config.backend, config.model, source.display());
                bytes = serde_json::to_vec_pretty(&config)?;
                bytes.push(b'\n');
            }
            let caps = catalog.iter().find(|caps| caps.matches(&config.backend, config.model.as_deref()))
                .ok_or_else(|| std::io::Error::other("legacy backend/model is unavailable in this build"))?;
            crate::config::validate_style(config.style.as_deref(), caps).map_err(std::io::Error::other)?;
            if let Some(id) = config.voice.strip_prefix("custom:") {
                if !caps.cloning || !valid_voice_id(id) { return Err(std::io::Error::other("legacy custom voice is unsupported or invalid")); }
            } else {
                crate::config::validate_voice(&config.voice, caps).map_err(std::io::Error::other)?;
            }
            publish(&self.paths.config(), |output| output.write_all(&bytes))
        })
    }

    pub fn checkpoints(&self) -> std::io::Result<()> {
        self.tree(
            "checkpoints",
            ".novel/tts/checkpoints",
            &self.paths.checkpoints(),
            None,
        )
    }

    /// Bounded complete voice bundles make legacy custom voices visible in Hello.
    /// Model weights and encoding caches remain deferred until preparation.
    pub fn voices(&self, coded_revision: Option<&str>) -> std::io::Result<()> {
        self.voices_budget(coded_revision, 64 * 1024 * 1024, None)
    }

    /// Preparation retries remaining voice bundles with cancellation and no
    /// aggregate startup budget; each file retains the same validation/limit.
    pub fn prepare_voices(
        &self,
        coded_revision: Option<&str>,
        cancelled: &AtomicBool,
    ) -> std::io::Result<()> {
        self.voices_budget(coded_revision, u64::MAX, Some(cancelled))
    }

    fn voices_budget(
        &self,
        coded_revision: Option<&str>,
        mut remaining: u64,
        cancelled: Option<&AtomicBool>,
    ) -> std::io::Result<()> {
        let Some(source) = self.source(".novel-tts")? else {
            return Ok(());
        };
        self.run("voices", || {
            scan_voices(
                &source,
                &self.paths.resources(),
                0,
                &mut remaining,
                coded_revision,
                cancelled,
            )
        })
    }

    /// Potentially large copies belong in actual resource preparation, not Hello.
    pub fn resources(&self) -> std::io::Result<()> {
        self.tree("resources", ".novel-tts", &self.paths.resources(), None)
    }

    /// Cancellation is checked between files and every 1 MiB of a large copy.
    pub fn resources_with_cancellation(&self, cancelled: &AtomicBool) -> std::io::Result<()> {
        self.tree(
            "resources",
            ".novel-tts",
            &self.paths.resources(),
            Some(cancelled),
        )
    }

    fn tree(
        &self,
        stage: &str,
        relative: &str,
        destination: &Path,
        cancelled: Option<&AtomicBool>,
    ) -> std::io::Result<()> {
        let Some(source) = self.source(relative)? else {
            return Ok(());
        };
        self.run(stage, || {
            copy_tree(&source, destination, cancelled, stage == "checkpoints")
        })
    }

    // Inspect every legacy path component, so a symlinked root or ancestor is
    // skipped just like symlinks inside the copied tree.
    fn source(&self, relative: &str) -> std::io::Result<Option<PathBuf>> {
        let mut path = self.home.clone();
        for component in Path::new(relative).components() {
            path.push(component);
            match std::fs::symlink_metadata(&path) {
                Ok(metadata) if metadata.file_type().is_symlink() => return Ok(None),
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => return Err(error),
            }
        }
        Ok(Some(path))
    }

    fn run(&self, stage: &str, work: impl FnOnce() -> std::io::Result<()>) -> std::io::Result<()> {
        directory(self.paths.root())?;
        let marker = self
            .paths
            .root()
            .join(format!(".legacy-{stage}-v1.complete"));
        if std::fs::symlink_metadata(&marker).is_ok() {
            return Ok(());
        }
        let lock_path = if stage == "config" {
            self.paths.config().with_extension("json.lock")
        } else {
            self.paths.root().join(format!(".legacy-{stage}.lock"))
        };
        if std::fs::symlink_metadata(&lock_path).is_ok_and(|meta| !meta.file_type().is_file()) {
            return Err(std::io::Error::other(
                "migration lock is not a regular file",
            ));
        }
        let lock = File::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)?;
        lock.try_lock().map_err(std::io::Error::other)?;
        if std::fs::symlink_metadata(&marker).is_ok() {
            return Ok(());
        }
        work()?;
        publish(&marker, |file| {
            file.write_all(b"talechime-legacy-migration-v1\n")
        })
    }
}

fn directory(path: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_dir() => Ok(()),
        Ok(_) => Err(std::io::Error::other(format!(
            "migration target is not a regular directory: {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => std::fs::create_dir(path),
        Err(error) => Err(error),
    }
}

fn copy_tree(
    source: &Path,
    target: &Path,
    cancelled: Option<&AtomicBool>,
    checkpoints: bool,
) -> std::io::Result<()> {
    check_cancelled(cancelled)?;
    if std::fs::symlink_metadata(source.join("voice.json")).is_ok() {
        // A voice is one bundle. Never fill missing files in a bundle that the
        // new application already owns, even if its metadata matches the ID.
        return Ok(());
    }
    directory(target)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let destination = target.join(entry.file_name());
        if kind.is_dir() {
            // Only the validated voice-bundle path may import voices.
            if entry.file_name() == "voices" {
                continue;
            }
            // Existing foreign links/files are preserved, never followed.
            if std::fs::symlink_metadata(&destination).is_ok_and(|meta| !meta.file_type().is_dir())
            {
                continue;
            }
            copy_tree(&entry.path(), &destination, cancelled, checkpoints)?;
        } else if kind.is_file() && std::fs::symlink_metadata(&destination).is_err() {
            if checkpoints {
                if entry
                    .path()
                    .extension()
                    .is_none_or(|extension| extension != "json")
                {
                    continue;
                }
                validate_checkpoint(&entry.path())?;
            }
            publish(&destination, |output| {
                copy_file(&entry.path(), output, cancelled)
            })?;
        }
    }
    Ok(())
}

fn valid_voice_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_'))
}

fn bounded_bytes(path: &Path, limit: u64, remaining: &mut u64) -> std::io::Result<Vec<u8>> {
    let size = std::fs::symlink_metadata(path)?;
    if !size.file_type().is_file() || size.len() > limit || size.len() > *remaining {
        return Err(std::io::Error::other(
            "legacy voice exceeds startup migration budget or is not a regular file",
        ));
    }
    let mut bytes = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit || bytes.len() as u64 > *remaining {
        return Err(std::io::Error::other("legacy voice grew during migration"));
    }
    *remaining -= bytes.len() as u64;
    Ok(bytes)
}

fn scan_voices(
    source: &Path,
    target: &Path,
    depth: usize,
    remaining: &mut u64,
    coded_revision: Option<&str>,
    cancelled: Option<&AtomicBool>,
) -> std::io::Result<()> {
    check_cancelled(cancelled)?;
    if depth > 12 {
        return Ok(());
    }
    if std::fs::symlink_metadata(target).is_ok_and(|meta| !meta.file_type().is_dir()) {
        return Ok(());
    }
    let metadata = source.join("voice.json");
    if std::fs::symlink_metadata(&metadata).is_ok_and(|meta| meta.file_type().is_file()) {
        if std::fs::symlink_metadata(target).is_ok() {
            return Ok(());
        }
        let voice_bytes = bounded_bytes(&metadata, 2 * 1024 * 1024, remaining)?;
        let voice: crate::voices::Voice = serde_json::from_slice(&voice_bytes)?;
        let id = voice
            .id
            .strip_prefix("custom:")
            .ok_or_else(|| std::io::Error::other("invalid legacy voice ID"))?;
        if !valid_voice_id(id)
            || source.file_name().and_then(|name| name.to_str())
                != Some(format!("voice_{id}").as_str())
            || voice.transcript.trim().is_empty()
            || source
                .parent()
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                != Some(voice.revision.as_str())
            || source
                .parent()
                .and_then(Path::parent)
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                != Some(voice.model.as_str())
            || source
                .parent()
                .and_then(Path::parent)
                .and_then(Path::parent)
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                != Some(voice.backend.as_str())
        {
            return Err(std::io::Error::other("invalid legacy voice bundle"));
        }
        let deferred_reference = *remaining > 64 * 1024 * 1024;
        let reference = if deferred_reference {
            None
        } else {
            Some(bounded_bytes(
                &source.join("reference.wav"),
                32 * 1024 * 1024,
                remaining,
            )?)
        };
        let parent = target
            .parent()
            .ok_or_else(|| std::io::Error::other("voice has no parent"))?;
        let _lock = crate::storage::lock(&parent.join(".lock"))?;
        if std::fs::symlink_metadata(target).is_ok() {
            return Ok(());
        }
        let staged = tempfile::tempdir_in(parent)?;
        std::fs::write(staged.path().join("voice.json"), voice_bytes)?;
        if let Some(reference) = reference {
            std::fs::write(staged.path().join("reference.wav"), reference)?;
        } else {
            let reference_path = source.join("reference.wav");
            if !std::fs::symlink_metadata(&reference_path)?
                .file_type()
                .is_file()
            {
                return Err(std::io::Error::other(
                    "voice reference is not a regular file",
                ));
            }
            let mut output = File::create(staged.path().join("reference.wav"))?;
            copy_file(&reference_path, &mut output, cancelled)?;
            output.sync_all()?;
        }
        std::fs::rename(staged.path(), target)?;
        return Ok(());
    }
    directory(target)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let destination = target.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            scan_voices(
                &entry.path(),
                &destination,
                depth + 1,
                remaining,
                coded_revision,
                cancelled,
            )?;
        } else if entry.file_type()?.is_file()
            && source.file_name().is_some_and(|name| name == "voices")
            && entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "json")
            && std::fs::symlink_metadata(&destination).is_err()
        {
            let Some(revision) = coded_revision else {
                continue;
            };
            let bytes = bounded_bytes(&entry.path(), 2 * 1024 * 1024, remaining)?;
            let voice: serde_json::Value = serde_json::from_slice(&bytes)?;
            let id = voice["id"]
                .as_str()
                .and_then(|id| id.strip_prefix("custom:"))
                .ok_or_else(|| std::io::Error::other("invalid legacy coded voice"))?;
            let codes: Vec<Vec<i32>> = serde_json::from_value(voice["codes"].clone())?;
            if !valid_voice_id(id)
                || voice["revision"].as_str() != Some(revision)
                || voice["name"]
                    .as_str()
                    .is_none_or(|name| name.trim().is_empty() || name.len() > 256)
                || codes.is_empty()
                || codes.len() > 375
                || codes.iter().any(|row| {
                    row.len() != 16 || row.iter().any(|value| !(0..1024).contains(value))
                })
                || entry.path().file_name().and_then(|name| name.to_str())
                    != Some(format!("{id}.json").as_str())
            {
                return Err(std::io::Error::other("invalid legacy coded voice"));
            }
            let _lock = crate::storage::lock(&target.join(".lock"))?;
            publish(&destination, |output| output.write_all(&bytes))?;
        }
    }
    Ok(())
}

fn validate_checkpoint(path: &Path) -> std::io::Result<()> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(64 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 64 * 1024 {
        return Err(std::io::Error::other("legacy checkpoint exceeds 64 KiB"));
    }
    let checkpoint: crate::checkpoint::Checkpoint = serde_json::from_slice(&bytes)?;
    let expected = format!(
        "{}.json",
        tts_protocol::text_hash(&serde_json::to_string(&checkpoint.source)?)
    );
    if checkpoint.schema_version != 1
        || checkpoint.text_hash.len() != 64
        || !checkpoint
            .text_hash
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || path.file_name().and_then(|name| name.to_str()) != Some(expected.as_str())
    {
        return Err(std::io::Error::other(format!(
            "invalid legacy checkpoint: {}",
            path.display()
        )));
    }
    Ok(())
}

fn check_cancelled(cancelled: Option<&AtomicBool>) -> std::io::Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "legacy migration cancelled",
        ));
    }
    Ok(())
}

fn copy_file(
    source: &Path,
    output: &mut File,
    cancelled: Option<&AtomicBool>,
) -> std::io::Result<()> {
    let mut source = File::open(source)?;
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        check_cancelled(cancelled)?;
        let count = source.read(&mut buffer)?;
        if count == 0 {
            return Ok(());
        }
        output.write_all(&buffer[..count])?;
    }
}

fn publish(
    target: &Path,
    write: impl FnOnce(&mut File) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let parent = target
        .parent()
        .ok_or_else(|| std::io::Error::other("migration target has no parent"))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    write(temporary.as_file_mut())?;
    temporary.as_file().sync_all()?;
    match temporary.persist_noclobber(target) {
        Ok(_) => Ok(()),
        Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error.error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn catalog() -> Vec<Capabilities> {
        vec![serde_json::from_value(serde_json::json!({"backend":"moss", "model":"nano-candle", "voices":["Weiguo"], "default_voice":"Weiguo", "native_streaming":false, "style":false,"cloning":true,"pronunciation":false})).unwrap()]
    }
    fn defaults() -> Config {
        Config {
            model: Some("nano-candle".into()),
            ..Default::default()
        }
    }
    fn old_config(home: &Path, text: &str) -> PathBuf {
        let parent = home.join(".novel");
        std::fs::create_dir_all(&parent).unwrap();
        let path = parent.join("tts_config.json");
        std::fs::write(&path, text).unwrap();
        path
    }
    #[test]
    fn retired_and_pre_backend_preferences_convert_without_touching_sources() {
        for backend in ["", ",\"backend\":\"kokoro\"", ",\"backend\":\"zipvoice\""] {
            let home = tempfile::tempdir().unwrap();
            let text = format!(
                "{{\"volume\":0.7,\"speed\":1.3,\"auto_play\":true,\"voice\":\"Zf001\",\"unknown\":{{\"keep\":42}}{backend}}}"
            );
            let source = old_config(home.path(), &text);
            let migration = LegacyMigration::from_home(home.path());
            migration.config(&defaults(), &catalog()).unwrap();
            let config = crate::config::ConfigStore::new(migration.paths.config())
                .load()
                .unwrap();
            assert_eq!(config.backend, "moss");
            assert_eq!(config.model.as_deref(), Some("nano-candle"));
            assert_eq!(config.voice, "Weiguo");
            assert_eq!(config.volume, 0.7);
            assert_eq!(config.speed, 1.3);
            assert!(config.auto_play);
            assert_eq!(config.extra["unknown"]["keep"], 42);
            assert_eq!(std::fs::read_to_string(source).unwrap(), text);
        }
    }
    #[test]
    fn existing_new_data_wins_and_success_marker_prevents_reimport() {
        let home = tempfile::tempdir().unwrap();
        let source = home.path().join(".novel-tts/moss");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("voice"), b"old").unwrap();
        std::fs::write(source.join("weights"), b"model bytes").unwrap();
        let migration = LegacyMigration::from_home(home.path());
        let target = migration.paths.resources().join("moss");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join("voice"), b"new").unwrap();
        migration.resources().unwrap();
        assert_eq!(std::fs::read(target.join("voice")).unwrap(), b"new");
        assert_eq!(
            std::fs::read(target.join("weights")).unwrap(),
            b"model bytes"
        );
        assert!(source.join("weights").exists());
        std::fs::write(source.join("later"), b"late").unwrap();
        migration.resources().unwrap();
        assert!(!target.join("later").exists());
    }
    #[test]
    fn malformed_config_is_retryable_and_new_config_is_preserved() {
        let home = tempfile::tempdir().unwrap();
        let source = old_config(home.path(), "invalid");
        let migration = LegacyMigration::from_home(home.path());
        assert!(migration.config(&defaults(), &catalog()).is_err());
        assert!(!migration.paths.config().exists());
        assert!(
            !migration
                .paths
                .root()
                .join(".legacy-config-v1.complete")
                .exists()
        );
        let text = "{\"backend\":\"moss\",\"model\":\"nano-candle\",\"voice\":\"custom:reader\"}";
        std::fs::write(&source, text).unwrap();
        migration.config(&defaults(), &catalog()).unwrap();
        assert_eq!(
            std::fs::read_to_string(migration.paths.config()).unwrap(),
            text
        );
        std::fs::write(&source, "invalid again").unwrap();
        migration.config(&defaults(), &catalog()).unwrap();
        assert_eq!(
            std::fs::read_to_string(migration.paths.config()).unwrap(),
            text
        );
    }
    #[test]
    fn cancellation_does_not_commit_a_completion_marker_and_retry_copies_exact_checkpoints() {
        let home = tempfile::tempdir().unwrap();
        let source = home.path().join(".novel-tts");
        std::fs::create_dir(&source).unwrap();
        std::fs::write(source.join("model"), b"fake weight").unwrap();
        let checkpoint = home.path().join(".novel/tts/checkpoints");
        std::fs::create_dir_all(&checkpoint).unwrap();
        let source_id = tts_protocol::SourceId {
            namespace: "reader".into(),
            book: "书".into(),
            chapter: "章".into(),
        };
        crate::checkpoint::CheckpointStore::new(&checkpoint)
            .save(&source_id, "甲乙", 3, false)
            .unwrap();
        let name = format!(
            "{}.json",
            tts_protocol::text_hash(&serde_json::to_string(&source_id).unwrap())
        );
        let original = std::fs::read(checkpoint.join(&name)).unwrap();
        let migration = LegacyMigration::from_home(home.path());
        assert!(
            migration
                .resources_with_cancellation(&AtomicBool::new(true))
                .is_err()
        );
        assert!(
            !migration
                .paths
                .root()
                .join(".legacy-resources-v1.complete")
                .exists()
        );
        migration.resources().unwrap();
        migration.checkpoints().unwrap();
        assert_eq!(
            std::fs::read(migration.paths.checkpoints().join(&name)).unwrap(),
            original
        );
        assert_eq!(
            std::fs::read(migration.paths.resources().join("model")).unwrap(),
            b"fake weight"
        );
    }
    #[test]
    fn startup_voice_bundles_are_complete_and_never_mix_with_new_bundles() {
        let home = tempfile::tempdir().unwrap();
        let old = home.path().join(".novel-tts/backend");
        std::fs::create_dir_all(&old).unwrap();
        let wav = home.path().join("reference.wav");
        std::fs::write(&wav, b"old audio").unwrap();
        let store = crate::voices::VoiceStore::new(&old, "moss", "local", "rev").unwrap();
        store
            .import("custom:reader", "Reader", &wav, "准确转写", None)
            .unwrap();
        store
            .import("custom:owned", "Old", &wav, "旧转写", None)
            .unwrap();
        let migration = LegacyMigration::from_home(home.path());
        let target = crate::voices::VoiceStore::new(
            &migration.paths.resources().join("backend"),
            "moss",
            "local",
            "rev",
        )
        .unwrap();
        let owned = target.path("custom:owned").unwrap();
        std::fs::create_dir_all(&owned).unwrap();
        std::fs::write(owned.join("voice.json"), b"new user metadata").unwrap();
        migration.voices(Some("rev")).unwrap();
        assert_eq!(target.load("custom:reader").unwrap().name, "Reader");
        assert_eq!(
            std::fs::read(target.path("custom:reader").unwrap().join("reference.wav")).unwrap(),
            b"old audio"
        );
        migration.resources().unwrap();
        assert_eq!(
            std::fs::read(owned.join("voice.json")).unwrap(),
            b"new user metadata"
        );
        assert!(!owned.join("reference.wav").exists());
    }
    #[test]
    fn large_voice_references_are_deferred_then_streamed_during_preparation() {
        let home = tempfile::tempdir().unwrap();
        let old = home.path().join(".novel-tts/backend");
        let store = crate::voices::VoiceStore::new(&old, "moss", "local", "rev").unwrap();
        let bundle = store.path("custom:reader").unwrap();
        std::fs::create_dir_all(&bundle).unwrap();
        std::fs::write(
            bundle.join("voice.json"),
            serde_json::to_vec(&crate::voices::Voice {
                id: "custom:reader".into(),
                name: "Reader".into(),
                backend: "moss".into(),
                model: "local".into(),
                revision: "rev".into(),
                transcript: "转写".into(),
                description: None,
            })
            .unwrap(),
        )
        .unwrap();
        let size = 33 * 1024 * 1024;
        File::create(bundle.join("reference.wav"))
            .unwrap()
            .set_len(size)
            .unwrap();
        let migration = LegacyMigration::from_home(home.path());
        assert!(migration.voices(Some("rev")).is_err());
        migration
            .prepare_voices(Some("rev"), &AtomicBool::new(false))
            .unwrap();
        let target = crate::voices::VoiceStore::new(
            &migration.paths.resources().join("backend"),
            "moss",
            "local",
            "rev",
        )
        .unwrap();
        assert_eq!(
            std::fs::metadata(target.path("custom:reader").unwrap().join("reference.wav"))
                .unwrap()
                .len(),
            size
        );
    }
    #[test]
    fn bad_coded_voice_never_poison_the_new_catalog() {
        let home = tempfile::tempdir().unwrap();
        let source = home.path().join(".novel-tts/moss/voices");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(
            source.join("reader.json"),
            br#"{"id":"custom:reader","name":"Reader","revision":"rev","codes":[]}"#,
        )
        .unwrap();
        let migration = LegacyMigration::from_home(home.path());
        assert!(migration.voices(Some("rev")).is_err());
        migration.resources().unwrap();
        assert!(
            !migration
                .paths
                .resources()
                .join("moss/voices/reader.json")
                .exists()
        );
        assert!(
            !migration
                .paths
                .root()
                .join(".legacy-voices-v1.complete")
                .exists()
        );
    }
    #[test]
    fn malformed_checkpoint_is_not_published() {
        let home = tempfile::tempdir().unwrap();
        let source = home.path().join(".novel/tts/checkpoints");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("bad.json"), b"bad").unwrap();
        let migration = LegacyMigration::from_home(home.path());
        assert!(migration.checkpoints().is_err());
        assert!(!migration.paths.checkpoints().join("bad.json").exists());
        assert!(
            !migration
                .paths
                .root()
                .join(".legacy-checkpoints-v1.complete")
                .exists()
        );
        assert!(source.join("bad.json").exists());
    }
    #[test]
    fn held_transaction_lock_defers_migration_and_atomic_publish_never_clobbers() {
        let home = tempfile::tempdir().unwrap();
        old_config(home.path(), "{}");
        let migration = LegacyMigration::from_home(home.path());
        directory(migration.paths.root()).unwrap();
        let lock = File::options()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(migration.paths.config().with_extension("json.lock"))
            .unwrap();
        lock.lock().unwrap();
        assert!(migration.config(&defaults(), &catalog()).is_err());
        assert!(!migration.paths.config().exists());
        drop(lock);
        migration.config(&defaults(), &catalog()).unwrap();
        let target = home.path().join("race");
        publish(&target, |temporary| {
            temporary.write_all(b"old")?;
            std::fs::write(&target, b"new")
        })
        .unwrap();
        assert_eq!(std::fs::read(target).unwrap(), b"new");
    }
    #[cfg(unix)]
    #[test]
    fn symlinked_sources_and_destinations_never_escape_the_home() {
        use std::os::unix::fs::symlink;
        let home = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("private"), b"private").unwrap();
        symlink(outside.path(), home.path().join(".novel-tts")).unwrap();
        let migration = LegacyMigration::from_home(home.path());
        migration.resources().unwrap();
        assert!(!migration.paths.root().exists());
        std::fs::remove_file(home.path().join(".novel-tts")).unwrap();
        std::fs::create_dir(home.path().join(".novel-tts")).unwrap();
        symlink(
            outside.path().join("private"),
            home.path().join(".novel-tts/linked"),
        )
        .unwrap();
        std::fs::write(home.path().join(".novel-tts/regular"), b"old").unwrap();
        std::fs::create_dir(migration.paths.root()).unwrap();
        symlink(outside.path(), migration.paths.resources()).unwrap();
        assert!(migration.resources().is_err());
        assert!(!outside.path().join("regular").exists());
        std::fs::remove_file(migration.paths.resources()).unwrap();
        migration.resources().unwrap();
        assert!(!migration.paths.resources().join("linked").exists());
        assert!(migration.paths.resources().join("regular").exists());
    }
}
