//! Identity of an explicitly selected ORT library and its sibling EP libraries.
use super::super::evidence::{digest, file_digest};
use anyhow::{Context, ensure};
use std::{collections::BTreeMap, ffi::OsStr, path::Path};

pub(super) fn record(
    flags: &mut BTreeMap<String, Option<String>>,
    requested: Option<&OsStr>,
) -> anyhow::Result<()> {
    let Some(requested) = requested.filter(|value| !value.is_empty()) else {
        // Preserve historical static-build identities when no library was selected.
        return Ok(());
    };
    let path = Path::new(requested);
    ensure!(path.is_absolute(), "ORT_DYLIB_PATH must be absolute");
    ensure!(
        path.is_file(),
        "ORT_DYLIB_PATH must select an existing file"
    );
    let literal = requested
        .to_str()
        .context("ORT_DYLIB_PATH is not Unicode")?;
    let mut libraries = BTreeMap::from([("runtime".to_owned(), file_digest(path)?)]);
    for entry in std::fs::read_dir(path.parent().context("ORT library has no parent")?)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if is_provider(name) {
            ensure!(entry.path().is_file(), "ORT provider is not a file: {name}");
            libraries.insert(name.to_owned(), file_digest(&entry.path())?);
        }
    }
    flags.insert("ORT_DYLIB_PATH".into(), Some(literal.to_owned()));
    flags.insert(
        "ORT_RUNTIME_BUNDLE_SHA256".into(),
        Some(digest(&serde_json::to_vec(&libraries)?)),
    );
    Ok(())
}

fn is_provider(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    (name.starts_with("onnxruntime_providers_") || name.starts_with("libonnxruntime_providers_"))
        && (name.ends_with(".dll")
            || name.ends_with(".so")
            || name.ends_with(".dylib")
            || name.split_once(".so.").is_some_and(|(_, version)| {
                !version.is_empty() && version.bytes().all(|v| v.is_ascii_digit() || v == b'.')
            }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(path: &Path) -> anyhow::Result<BTreeMap<String, Option<String>>> {
        let mut result = BTreeMap::new();
        record(&mut result, Some(path.as_os_str()))?;
        Ok(result)
    }

    #[test]
    fn same_path_replacement_and_provider_inventory_change_identity() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let main = root.path().join("onnxruntime.dll");
        let provider = root.path().join("onnxruntime_providers_cuda.dll");
        std::fs::write(&main, b"runtime")?;
        std::fs::write(&provider, b"provider-A")?;
        let baseline = snapshot(&main)?;
        std::fs::write(&provider, b"provider-B")?;
        assert_ne!(snapshot(&main)?, baseline);
        std::fs::write(&provider, b"provider-A")?;
        assert_eq!(snapshot(&main)?, baseline);
        std::fs::write(&main, b"changed-runtime")?;
        assert_ne!(snapshot(&main)?, baseline);
        std::fs::write(&main, b"runtime")?;
        let shared = root.path().join("onnxruntime_providers_shared.dll");
        std::fs::write(&shared, b"shared")?;
        assert_ne!(snapshot(&main)?, baseline);
        std::fs::remove_file(shared)?;
        assert_eq!(snapshot(&main)?, baseline);
        std::fs::write(root.path().join("unrelated.log"), b"log")?;
        assert_eq!(snapshot(&main)?, baseline);
        std::fs::remove_file(provider)?;
        assert_ne!(snapshot(&main)?, baseline);
        Ok(())
    }

    #[test]
    fn absent_selection_is_compatible_but_invalid_explicit_paths_fail() -> anyhow::Result<()> {
        let mut flags = BTreeMap::from([("existing".into(), None)]);
        let original = flags.clone();
        record(&mut flags, None)?;
        record(&mut flags, Some(OsStr::new("")))?;
        assert_eq!(flags, original);
        assert!(snapshot(Path::new("relative.dll")).is_err());
        let root = tempfile::tempdir()?;
        assert!(snapshot(&root.path().join("missing.dll")).is_err());
        assert!(snapshot(root.path()).is_err());
        for name in [
            "onnxruntime_providers_cuda.dll",
            "libonnxruntime_providers_cuda.so.1.28",
            "libonnxruntime_providers_shared.dylib",
        ] {
            assert!(is_provider(name));
        }
        assert!(!is_provider("onnxruntime_providers_cuda.so.backup"));
        Ok(())
    }
}
