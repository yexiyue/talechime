//! Separate eSpeak NG CLI invocation and Misaki 0.9.4 US IPA mapping.
use candle_core::Result;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    io::{Read, Write},
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

const SOURCE_COMMIT: &str = "4870adfa25b1a32b4361592f1be8a40337c58d6c";
const SOURCE_SHA: &str = "bb4338102ff3b49a81423da8a1a158b420124b055b60fa76cfb4b18677130a23";
const EXECUTABLE: &str = if cfg!(windows) {
    "espeak-ng.exe"
} else {
    "espeak-ng"
};
const PHONEMIZER: &str = if cfg!(windows) {
    "kokoro-espeak-phonemize.exe"
} else {
    "kokoro-espeak-phonemize"
};

#[derive(Deserialize)]
struct BundleManifest {
    schema: String,
    source_commit: String,
    source_sha256: String,
    phonemizer_protocol: String,
    platform: String,
    machine: String,
    files: Vec<BundleFile>,
}

#[derive(Deserialize)]
struct BundleFile {
    path: String,
    size: u64,
    sha256: String,
}

/// Configuration for the separately distributed eSpeak program.
pub type EspeakConfig = EspeakOptions;

/// Explicit paths to a host-verified eSpeak NG 1.52.0 program and its data.
#[derive(Clone, Debug)]
pub struct EspeakOptions {
    /// Absolute path to the separately distributed CLI, verified by the host.
    pub executable: PathBuf,
    /// Absolute path to its verified `espeak-ng-data` directory.
    pub data_dir: PathBuf,
    /// Maximum time for a single word conversion.
    pub timeout: Duration,
}

impl EspeakOptions {
    /// Verify a host-distributed 1.52.0 bundle and use a five-second per-word timeout.
    pub fn from_bundle(root: &std::path::Path) -> Result<Self> {
        Self::from_bundle_cancellable(root, &|| false)
    }

    pub(super) fn from_bundle_cancellable(
        root: &std::path::Path,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Self> {
        if !root.is_absolute() || !std::fs::symlink_metadata(root).is_ok_and(|m| m.is_dir()) {
            candle_core::bail!("Kokoro English fallback bundle is unavailable");
        }
        let manifest_path = root.join("manifest.json");
        let metadata =
            std::fs::symlink_metadata(&manifest_path).map_err(candle_core::Error::wrap)?;
        if !metadata.is_file() || metadata.len() > 2 * 1024 * 1024 {
            candle_core::bail!("invalid eSpeak manifest");
        }
        let bytes = std::fs::read(manifest_path).map_err(candle_core::Error::wrap)?;
        let manifest: BundleManifest =
            serde_json::from_slice(&bytes).map_err(candle_core::Error::wrap)?;
        if manifest.schema != "talechime-espeak-bundle-v1"
            || manifest.source_commit != SOURCE_COMMIT
            || manifest.source_sha256 != SOURCE_SHA
            || manifest.phonemizer_protocol != "talechime-espeak-text-to-phonemes-v1"
        {
            candle_core::bail!("unsupported eSpeak bundle identity");
        }
        let platform = match std::env::consts::OS {
            "macos" => "darwin",
            other => other,
        };
        let machine = match manifest.machine.to_ascii_lowercase().as_str() {
            "arm64" => "aarch64",
            "amd64" => "x86_64",
            _ => manifest.machine.as_str(),
        };
        if !manifest.platform.eq_ignore_ascii_case(platform) || machine != std::env::consts::ARCH {
            candle_core::bail!("eSpeak bundle platform does not match this host");
        }
        let mut seen = HashSet::new();
        for file in &manifest.files {
            if cancelled() {
                candle_core::bail!("Kokoro English pronunciation cancelled");
            }
            let relative = std::path::Path::new(&file.path);
            if file.path.contains('\\')
                || relative
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
                || relative.as_os_str().is_empty()
                || !seen.insert(file.path.as_str())
                || file.sha256.len() != 64
                || !file.sha256.bytes().all(|c| c.is_ascii_hexdigit())
            {
                candle_core::bail!("invalid eSpeak bundle manifest path or digest");
            }
            let mut path = root.to_path_buf();
            for component in relative.components() {
                path.push(component.as_os_str());
                if std::fs::symlink_metadata(&path)
                    .map_err(candle_core::Error::wrap)?
                    .file_type()
                    .is_symlink()
                {
                    candle_core::bail!("eSpeak bundle contains a symbolic link");
                }
            }
            let mut input = std::fs::File::open(&path).map_err(candle_core::Error::wrap)?;
            if !input
                .metadata()
                .map_err(candle_core::Error::wrap)?
                .is_file()
                || input.metadata().map_err(candle_core::Error::wrap)?.len() != file.size
            {
                candle_core::bail!("eSpeak bundle file size mismatch");
            }
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 65536];
            loop {
                if cancelled() {
                    candle_core::bail!("Kokoro English pronunciation cancelled");
                }
                let size = input.read(&mut buffer).map_err(candle_core::Error::wrap)?;
                if size == 0 {
                    break;
                }
                hash.update(&buffer[..size]);
            }
            if format!("{:x}", hash.finalize()) != file.sha256 {
                candle_core::bail!("eSpeak bundle file digest mismatch");
            }
        }
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            if cancelled() {
                candle_core::bail!("Kokoro English pronunciation cancelled");
            }
            for entry in std::fs::read_dir(directory).map_err(candle_core::Error::wrap)? {
                let entry = entry.map_err(candle_core::Error::wrap)?;
                let path = entry.path();
                let kind = entry.file_type().map_err(candle_core::Error::wrap)?;
                if kind.is_dir() {
                    pending.push(path);
                } else if kind.is_file() {
                    let relative = path.strip_prefix(root).map_err(candle_core::Error::wrap)?;
                    let portable = relative
                        .components()
                        .map(|part| part.as_os_str().to_str())
                        .collect::<Option<Vec<_>>>()
                        .ok_or_else(|| {
                            candle_core::Error::Msg("non-UTF-8 eSpeak bundle path".into())
                        })?
                        .join("/");
                    if relative != std::path::Path::new("manifest.json")
                        && !seen.contains(portable.as_str())
                    {
                        candle_core::bail!("eSpeak bundle contains an unlisted file");
                    }
                } else {
                    candle_core::bail!("eSpeak bundle contains an unsupported file");
                }
            }
        }
        if ![
            EXECUTABLE,
            PHONEMIZER,
            "espeak-ng-data/en_dict",
            "espeak-ng-data/intonations",
            "espeak-ng-data/lang/gmw/en",
            "espeak-ng-data/lang/gmw/en-US",
            "espeak-ng-data/phondata",
            "espeak-ng-data/phonindex",
            "espeak-ng-data/phontab",
            "source/COPYING",
            "source/kokoro_espeak_phonemize.c",
        ]
        .iter()
        .all(|path| seen.contains(path))
            || !manifest.files.iter().any(|file| {
                file.path == "source/espeak-ng-1.52.0.tar.gz" && file.sha256 == SOURCE_SHA
            })
        {
            candle_core::bail!("eSpeak bundle is incomplete");
        }
        Ok(Self {
            executable: root.join(PHONEMIZER),
            data_dir: root.join("espeak-ng-data"),
            timeout: Duration::from_secs(5),
        })
    }
    pub(super) fn pronounce(&self, word: &str, cancelled: &dyn Fn() -> bool) -> Result<String> {
        if cancelled() {
            candle_core::bail!("Kokoro English pronunciation cancelled");
        }
        // Keep the complete word plus LF below the minimum pipe capacity, so
        // a stalled child cannot block the cancellation loop during write.
        if word.is_empty() || word.len() > 255 || word.contains(['\n', '\r', '\0']) {
            candle_core::bail!("invalid English fallback input");
        }
        if !self.executable.is_absolute()
            || !self.data_dir.is_absolute()
            || self
                .data_dir
                .file_name()
                .is_none_or(|name| name != "espeak-ng-data")
            || !self.executable.is_file()
            || !self.data_dir.is_dir()
        {
            candle_core::bail!("Kokoro English fallback resources are unavailable");
        }
        // eSpeak 1.52 stores the data path in a fixed-size C buffer. Reject
        // truncation before it can fall back to a different system data set.
        let path_limit = if cfg!(windows) { 230 } else { 160 };
        if self.data_dir.as_os_str().as_encoded_bytes().len() >= path_limit {
            candle_core::bail!("Kokoro English fallback data path is too long");
        }
        let parent = self
            .data_dir
            .parent()
            .ok_or_else(|| candle_core::Error::Msg("invalid eSpeak data path".into()))?;
        if self.timeout.is_zero() {
            candle_core::bail!("invalid eSpeak timeout");
        }
        let mut data_argument = std::ffi::OsString::from("--path=");
        data_argument.push(parent.as_os_str());
        let started = Instant::now();
        let mut child = Command::new(&self.executable)
            .arg(data_argument)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .env_remove("ESPEAK_DATA_PATH")
            .env_remove("HOME")
            .spawn()
            .map_err(|e| {
                candle_core::Error::Msg(format!("failed to start Kokoro English fallback: {e}"))
            })?;
        let stdout = child.stdout.take().expect("piped stdout");
        let reader = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            stdout.take(65537).read_to_end(&mut bytes).map(|_| bytes)
        });
        let input_result = child
            .stdin
            .take()
            .expect("piped stdin")
            .write_all(format!("{word}\n").as_bytes());
        if let Err(e) = input_result {
            let _ = child.kill();
            let _ = child.wait();
            let _ = reader.join();
            return Err(candle_core::Error::Msg(format!(
                "failed to write English fallback input: {e}"
            )));
        }
        let status = loop {
            if cancelled() || started.elapsed() >= self.timeout {
                let was_cancelled = cancelled();
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                if was_cancelled {
                    candle_core::bail!("Kokoro English pronunciation cancelled");
                }
                candle_core::bail!("Kokoro English fallback timed out");
            }
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = reader.join();
                    return Err(candle_core::Error::Msg(format!(
                        "English fallback process failed: {e}"
                    )));
                }
            }
        };
        let bytes = reader
            .join()
            .map_err(|_| candle_core::Error::Msg("English fallback output reader failed".into()))?
            .map_err(candle_core::Error::wrap)?;
        if !status.success() || bytes.len() > 65536 {
            candle_core::bail!("Kokoro English fallback failed or exceeded output budget");
        }
        let ipa = String::from_utf8(bytes).map_err(candle_core::Error::wrap)?;
        let phonemes = misaki_us(&ipa);
        if phonemes.trim().is_empty() {
            candle_core::bail!("Kokoro English fallback produced no phonemes");
        }
        Ok(phonemes)
    }
}

fn misaki_us(ipa: &str) -> String {
    let mut value = ipa.trim().to_owned();
    for (from, to) in [
        ("ʔˌn\u{329}", "ʔn"),
        ("ʔn\u{329}", "ʔn"),
        ("a^ɪ", "I"),
        ("a^ʊ", "W"),
        ("d^ʒ", "ʤ"),
        ("e^ɪ", "A"),
        ("t^ʃ", "ʧ"),
        ("ɔ^ɪ", "Y"),
        ("ə^l", "ᵊl"),
        ("ʲo", "jo"),
        ("ʲə", "jə"),
        ("ɚ", "əɹ"),
        ("ʲ", ""),
        ("r", "ɹ"),
        ("x", "k"),
        ("ç", "k"),
        ("ɐ", "ə"),
        ("ɬ", "l"),
        ("\u{303}", ""),
        ("e", "A"),
    ] {
        value = value.replace(from, to);
    }
    static SYLLABIC: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    value = SYLLABIC
        .get_or_init(|| regex::Regex::new(r"(\S)\x{329}").expect("static regex"))
        .replace_all(&value, "ᵊ$1")
        .replace('\u{329}', "");
    for (from, to) in [
        ("o^ʊ", "O"),
        ("ɜːɹ", "ɜɹ"),
        ("ɜː", "ɜɹ"),
        ("ɪə", "iə"),
        ("ː", ""),
        ("o", "ɔ"),
        ("ɾ", "T"),
        ("ʔ", "t"),
        ("^", ""),
    ] {
        value = value.replace(from, to);
    }
    value
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    #[ignore = "requires the separately built protocol-v1 eSpeak bundle"]
    fn standalone_phonemizer_matches_official_function_words() -> Result<()> {
        let root = std::env::var_os("TALECHIME_ESPEAK_DIR")
            .ok_or_else(|| candle_core::Error::Msg("set TALECHIME_ESPEAK_DIR".into()))?;
        let options = EspeakOptions::from_bundle(&PathBuf::from(root))?;
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/english-function-words-espeak.json"
        ))
        .map_err(candle_core::Error::wrap)?;
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 20);
        for case in cases {
            let text = case["text"].as_str().unwrap();
            assert_eq!(
                options.pronounce(text, &|| false)?,
                case["phonemes"],
                "{text}"
            );
        }
        Ok(())
    }

    fn mock(script: &str, timeout: Duration) -> (tempfile::TempDir, EspeakOptions) {
        let root = tempfile::tempdir().unwrap();
        let exe = root.path().join("espeak-ng");
        std::fs::write(&exe, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
        let data = root.path().join("espeak-ng-data");
        std::fs::create_dir(&data).unwrap();
        let options = EspeakOptions {
            executable: exe,
            data_dir: data,
            timeout,
        };
        (root, options)
    }

    #[test]
    fn writes_terminated_input_and_maps_output() {
        let (_root, options) = mock(
            "IFS= read -r word || exit 3; [ \"$word\" = zorbulate ] || exit 4; printf 'a^ɪ\\n'",
            Duration::from_secs(2),
        );
        assert_eq!(options.pronounce("zorbulate", &|| false).unwrap(), "I");
        assert!(options.pronounce("bad\nword", &|| false).is_err());
    }

    #[test]
    fn propagates_missing_exit_timeout_and_cancellation() {
        let (_root, options) = mock("exit 7", Duration::from_secs(2));
        assert!(options.pronounce("word", &|| false).is_err());
        let (_root, options) = mock("exec /bin/sleep 5", Duration::from_millis(50));
        let start = Instant::now();
        assert!(
            options
                .pronounce("word", &|| false)
                .unwrap_err()
                .to_string()
                .contains("timed out")
        );
        assert!(start.elapsed() < Duration::from_secs(2));
        let start = Instant::now();
        assert!(
            options
                .pronounce("word", &|| start.elapsed() > Duration::from_millis(50))
                .unwrap_err()
                .to_string()
                .contains("cancelled")
        );
        assert!(start.elapsed() < Duration::from_secs(2));
        std::fs::remove_file(&options.executable).unwrap();
        assert!(
            options
                .pronounce("word", &|| false)
                .unwrap_err()
                .to_string()
                .contains("unavailable")
        );
    }

    #[test]
    fn bundle_rejects_invalid_manifest_and_traversal() {
        let root = tempfile::tempdir().unwrap();
        let machine = std::env::consts::ARCH;
        let platform = if cfg!(target_os = "macos") {
            "Darwin"
        } else {
            std::env::consts::OS
        };
        let manifest = |path: &str| {
            serde_json::json!({
                "schema":"talechime-espeak-bundle-v1", "source_commit":SOURCE_COMMIT,
                "source_sha256":SOURCE_SHA, "phonemizer_protocol":"talechime-espeak-text-to-phonemes-v1", "platform":platform, "machine":machine,
                "files":[{"path":path, "size":0, "sha256":"0".repeat(64)}]
            })
        };
        let mut missing_protocol = manifest("espeak-ng");
        missing_protocol
            .as_object_mut()
            .unwrap()
            .remove("phonemizer_protocol");
        std::fs::write(
            root.path().join("manifest.json"),
            missing_protocol.to_string(),
        )
        .unwrap();
        assert!(EspeakConfig::from_bundle(root.path()).is_err());
        std::fs::write(
            root.path().join("manifest.json"),
            manifest("../escape").to_string(),
        )
        .unwrap();
        assert!(
            EspeakConfig::from_bundle(root.path())
                .unwrap_err()
                .to_string()
                .contains("manifest")
        );
        std::fs::write(
            root.path().join("manifest.json"),
            manifest("espeak-ng").to_string(),
        )
        .unwrap();
        std::fs::write(root.path().join("espeak-ng"), []).unwrap();
        assert!(
            EspeakConfig::from_bundle(root.path())
                .unwrap_err()
                .to_string()
                .contains("digest")
        );
        assert!(
            EspeakOptions::from_bundle_cancellable(root.path(), &|| true)
                .unwrap_err()
                .to_string()
                .contains("cancelled")
        );
    }
}
