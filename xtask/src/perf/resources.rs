//! Copy-only benchmark resource preparation. Measurement never downloads or converts.
use super::evidence::{contained, digest, file_digest, publish_json, read_json};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::Read,
    path::{Component, Path, PathBuf},
    time::Duration,
};

pub const RESOURCE_SCHEMA: &str = "talechime/performance-resources/1";
pub const PLAN_SCHEMA: &str = "talechime/benchmark-resource-plan/1";
pub const IMPORT_SCHEMA: &str = "talechime/benchmark-resource-import/1";
pub const MAX_RESOURCE_BYTES: u64 = 32 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResourceIdentity {
    pub backend: String,
    pub model: String,
    pub revision: String,
    /// Production destination relative to the resolved resources root.
    pub destination: String,
    /// Explicit local source; empty selects the pinned `url` during download.
    #[serde(default)]
    pub source: PathBuf,
    #[serde(default)]
    pub url: String,
    pub size: u64,
    pub sha256: String,
    #[serde(default)]
    pub completeness: String,
}

impl ResourceIdentity {
    pub fn validate(&self) -> anyhow::Result<()> {
        for (name, value) in [
            ("backend", &self.backend),
            ("model", &self.model),
            ("revision", &self.revision),
        ] {
            ensure!(
                !value.is_empty()
                    && value.len() <= 128
                    && value
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
                    && value != "."
                    && value != "..",
                "invalid resource identity component: {name}"
            );
        }
        let parts: Vec<Component<'_>> = Path::new(&self.destination).components().collect();
        ensure!(
            !parts.is_empty()
                && parts
                    .iter()
                    .all(|part| matches!(part, Component::Normal(_))),
            "resource destination must be relative without traversal"
        );
        let bound = parts
            .first()
            .is_some_and(|first| first.as_os_str() == self.backend.as_str())
            || (parts.len() >= 3
                && parts[0].as_os_str() == "voices"
                && parts[2].as_os_str() == self.backend.as_str());
        ensure!(
            bound,
            "resource destination must stay inside its backend namespace"
        );
        ensure!(
            self.sha256.len() == 64
                && self.sha256.bytes().all(|b| b.is_ascii_hexdigit())
                && self.size > 0
                && self.size <= MAX_RESOURCE_BYTES,
            "resource file identity is incomplete"
        );
        ensure!(
            self.completeness.is_empty()
                || ["production_manifest", "conversion_receipt", "incomplete"]
                    .contains(&self.completeness.as_str()),
            "unknown resource completeness identity"
        );
        if self.source.as_os_str().is_empty() {
            ensure!(
                self.url.starts_with("https://") || self.url.starts_with("http://"),
                "entries without a local source require a pinned https URL"
            );
        }
        Ok(())
    }

    pub fn destination(&self, root: &Path) -> anyhow::Result<PathBuf> {
        self.validate()?;
        contained(root, Path::new(&self.destination))
    }
}

fn safe_source(source: &Path) -> anyhow::Result<PathBuf> {
    let path = source
        .canonicalize()
        .with_context(|| format!("source not found: {}", source.display()))?;
    ensure!(path.is_file(), "source is not a regular file");
    Ok(path)
}

fn identity(source: &Path, size: u64) -> anyhow::Result<String> {
    let mut file = File::open(source)?;
    ensure!(file.metadata()?.len() == size, "source size changed");
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Inventory explicit local sources without reading beyond their own subtrees.
pub fn inventory(entries: &[ResourceIdentity]) -> anyhow::Result<serde_json::Value> {
    ensure!(!entries.is_empty(), "resource plan is empty");
    for entry in entries {
        entry.validate()?;
    }
    let mut items = Vec::new();
    let mut planned_bytes = 0u64;
    for entry in entries {
        let mut item = serde_json::json!({
            "resource":entry,
            "source_size":serde_json::Value::Null,
            "source_sha256":serde_json::Value::Null,
            "status":"missing_download",
            "reason":"explicit local source absent; use the pinned download phase",
        });
        if !entry.source.as_os_str().is_empty() {
            let source = safe_source(&entry.source)?;
            let actual_size = source.metadata()?.len();
            let actual_sha = identity(&source, actual_size)?;
            item["source"] = source.to_string_lossy().into_owned().into();
            item["source_size"] = actual_size.into();
            item["source_sha256"] = actual_sha.clone().into();
            if actual_size != entry.size || actual_sha != entry.sha256 {
                item["status"] = "unverified".into();
                item["reason"] = "source identity differs from the pinned resource identity".into();
            } else if entry.completeness != "production_manifest" {
                item["status"] = "receipt_only".into();
                item["reason"] = "conversion receipt is incomplete and cannot substitute for production integrity".into();
            } else {
                item["status"] = "ready".into();
                item["reason"] = serde_json::Value::Null;
                planned_bytes += entry.size;
            }
        }
        items.push(item);
    }
    Ok(serde_json::json!({
        "schema":IMPORT_SCHEMA,
        "entries":items,
        "planned_bytes":planned_bytes,
        "ready_count":items.iter().filter(|item| item["status"] == "ready").count(),
        "receipt_only_count":items.iter().filter(|item| item["status"] == "receipt_only").count(),
        "total_count":items.len(),
    }))
}

fn resource_root(root: &Path) -> anyhow::Result<PathBuf> {
    let root = root
        .canonicalize()
        .with_context(|| format!("resources root not found: {}", root.display()))?;
    ensure!(
        root.file_name().is_none_or(|name| name != "target"),
        "benchmark storage cannot be a build target directory"
    );
    Ok(root)
}

fn id_lock(root: &Path, entry: &ResourceIdentity) -> anyhow::Result<File> {
    let directory = entry
        .destination(root)?
        .parent()
        .context("destination has no parent")?
        .to_owned();
    std::fs::create_dir_all(&directory)?;
    let path = directory.join(".resource-import.lock");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    file.try_lock().map_err(|error| {
        anyhow::anyhow!("another importer owns this resource identity: {error}")
    })?;
    Ok(file)
}

fn staging_path(parent: &Path, kind: &str) -> anyhow::Result<PathBuf> {
    Ok(parent.join(format!(
        ".resource-{kind}-{}-{}.staging",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    )))
}

fn stage(root: &Path, entry: &ResourceIdentity, source: &Path) -> anyhow::Result<PathBuf> {
    let destination = entry.destination(root)?;
    let parent = destination
        .parent()
        .context("destination has no parent")?
        .to_owned();
    std::fs::create_dir_all(&parent)?;
    let stage = staging_path(&parent, "import")?;
    let result = (|| -> anyhow::Result<()> {
        std::fs::copy(source, &stage)?;
        let size = stage.metadata()?.len();
        ensure!(
            size == entry.size && identity(&stage, entry.size)? == entry.sha256,
            "staged resource changed; existing data remains untouched"
        );
        Ok(())
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&stage);
        return Err(error);
    }
    Ok(stage)
}

fn publish(root: &Path, entry: &ResourceIdentity, stage: PathBuf) -> anyhow::Result<bool> {
    let destination = entry.destination(root)?;
    let published = match std::fs::hard_link(&stage, &destination) {
        Ok(()) => true,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
        Err(error) => {
            let _ = std::fs::remove_file(&stage);
            return Err(error.into());
        }
    };
    let _ = std::fs::remove_file(&stage);
    Ok(published)
}

fn remove_stale_staging(parent: &Path) -> anyhow::Result<()> {
    let parent = parent
        .parent()
        .context("destination has no parent directory")?;
    if parent.is_dir() {
        for entry in std::fs::read_dir(parent)? {
            let path = entry?.path();
            let name = path
                .file_name()
                .context("staging entry has no filename")?
                .to_string_lossy()
                .into_owned();
            if (name.starts_with(".resource-import-") || name.starts_with(".resource-download-"))
                && name.ends_with(".staging")
            {
                std::fs::remove_file(path)?;
            }
        }
    }
    Ok(())
}

fn plan_digest(entries: &[ResourceIdentity]) -> anyhow::Result<String> {
    let mut canonical = entries.to_vec();
    canonical.sort_by(|a, b| a.destination.cmp(&b.destination));
    Ok(digest(&serde_json::to_vec(&canonical)?))
}

fn complete(root: &Path, entries: &[ResourceIdentity]) -> anyhow::Result<serde_json::Value> {
    ensure!(!entries.is_empty(), "resource plan is empty");
    for entry in entries {
        ensure!(
            entry.completeness == "production_manifest" && verified_destination(root, entry)?,
            "resource completion requires verified production files"
        );
    }
    let marker = root.join(format!(
        ".benchmark-import-complete-{}.json",
        plan_digest(entries)?
    ));
    let receipt = serde_json::json!({
        "schema":RESOURCE_SCHEMA,
        "status":"verified",
        "resources_root":root,
        "files":entries,
    });
    if marker.exists() {
        let existing: serde_json::Value = read_json(&marker)?;
        ensure!(existing == receipt, "resource completion marker changed");
    } else {
        publish_json(&marker, &receipt)?;
    }
    Ok(receipt)
}

fn verified_destination(root: &Path, entry: &ResourceIdentity) -> anyhow::Result<bool> {
    let destination = entry.destination(root)?;
    Ok(destination.exists()
        && destination.metadata()?.len() == entry.size
        && file_digest(&destination)? == entry.sha256)
}

fn item_status(receipt: &serde_json::Value, entry: &ResourceIdentity) -> anyhow::Result<String> {
    let expected = serde_json::to_value(entry)?;
    let item = receipt["entries"]
        .as_array()
        .context("inventory entries missing")?
        .iter()
        .find(|item| item["resource"] == expected)
        .context("inventory entry missing")?;
    Ok(item["status"]
        .as_str()
        .context("inventory status missing")?
        .to_owned())
}

/// Publish verified inventory entries. Matching destinations are reused; conflicts stop the import.
pub fn import(inventory_path: &Path, root: &Path, output: &Path) -> anyhow::Result<()> {
    ensure!(!output.exists(), "import receipt already exists");
    let receipt: serde_json::Value = read_json(inventory_path)?;
    ensure!(
        receipt["schema"] == IMPORT_SCHEMA,
        "unknown resource inventory schema"
    );
    let entries: Vec<ResourceIdentity> = receipt["entries"]
        .as_array()
        .context("inventory entries missing")?
        .iter()
        .map(|item| serde_json::from_value(item["resource"].clone()))
        .collect::<Result<_, _>>()?;
    ensure!(!entries.is_empty(), "resource plan is empty");
    for entry in &entries {
        entry.validate()?;
        ensure!(
            entry.completeness == "production_manifest",
            "only production manifests may be imported; conversion receipts are provenance only"
        );
        let status = item_status(&receipt, entry)?;
        ensure!(
            status != "unverified",
            "inventory contains an unverified source; no destination data changed"
        );
        ensure!(
            status != "missing_download",
            "missing pinned resource; run the explicit download phase first"
        );
    }
    let root = resource_root(root)?;
    let mut published = Vec::new();
    let mut reused = Vec::new();
    for entry in &entries {
        let _identity_lock = id_lock(&root, entry)?;
        if verified_destination(&root, entry)? {
            reused.push(serde_json::to_value(entry)?);
            continue;
        }
        let stage_path = stage(&root, entry, &entry.source)?;
        let published_file = publish(&root, entry, stage_path)?;
        remove_stale_staging(&entry.destination(&root)?)?;
        ensure!(
            verified_destination(&root, entry)?,
            "destination conflicts with pinned identity; existing data is preserved"
        );
        if published_file {
            published.push(serde_json::to_value(entry)?);
        } else {
            reused.push(serde_json::to_value(entry)?);
        }
    }
    let complete_receipt = complete(&root, &entries)?;
    publish_json(
        output,
        &serde_json::json!({
            "schema":IMPORT_SCHEMA,
            "resources_root":root,
            "published":published,
            "reused":reused,
            "receipt":complete_receipt,
        }),
    )
}

fn download_to_stage(url: &str, stage: &Path, expected_size: u64) -> anyhow::Result<()> {
    use std::io::Write;
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(30 * 60))
        .build()?;
    let mut response = client.get(url).send()?.error_for_status()?;
    let mut file = File::create(stage)?;
    file.sync_all()?;
    let mut buffer = [0u8; 65536];
    let mut downloaded = 0u64;
    loop {
        let count = response.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        downloaded += u64::try_from(count)?;
        ensure!(
            downloaded <= expected_size,
            "download exceeds the pinned resource byte budget"
        );
        file.write_all(&buffer[..count])?;
    }
    file.sync_all()?;
    ensure!(
        downloaded == expected_size,
        "downloaded size differs from the pinned manifest"
    );
    Ok(())
}

/// Explicit preparation only: download missing pinned production weights, copy-only publication.
pub fn download(plan_path: &Path, root: &Path, offline: bool, output: &Path) -> anyhow::Result<()> {
    ensure!(!output.exists(), "download receipt already exists");
    let plan: serde_json::Value = read_json(plan_path)?;
    ensure!(
        plan["schema"] == PLAN_SCHEMA,
        "unknown resource plan schema"
    );
    let entries: Vec<ResourceIdentity> = plan["files"]
        .as_array()
        .context("plan files missing")?
        .iter()
        .map(|file| {
            Ok(ResourceIdentity {
                backend: plan["backend"].as_str().context("backend missing")?.into(),
                model: plan["model"].as_str().context("model missing")?.into(),
                revision: plan["revision"]
                    .as_str()
                    .context("revision missing")?
                    .into(),
                destination: file["destination"]
                    .as_str()
                    .context("file destination missing")?
                    .into(),
                source: PathBuf::new(),
                url: file["url"].as_str().context("file URL missing")?.into(),
                size: file["size"].as_u64().context("file size missing")?,
                sha256: file["sha256"]
                    .as_str()
                    .context("file digest missing")?
                    .into(),
                completeness: file["completeness"]
                    .as_str()
                    .or_else(|| plan["completeness"].as_str())
                    .unwrap_or("production_manifest")
                    .into(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    ensure!(!entries.is_empty(), "resource plan is empty");
    for entry in &entries {
        entry.validate()?;
        ensure!(
            entry.completeness == "production_manifest",
            "only production manifests may be downloaded; conversion receipts are provenance only"
        );
    }
    let root = resource_root(root)?;
    let mut published = Vec::new();
    let mut reused = Vec::new();
    for entry in &entries {
        if verified_destination(&root, entry)? {
            reused.push(serde_json::to_value(entry)?);
            continue;
        }
        ensure!(
            !offline,
            "offline mode requested and pinned resource is missing: {}",
            entry.destination
        );
        let destination = entry.destination(&root)?;
        let parent = destination
            .parent()
            .context("destination has no parent")?
            .to_owned();
        let _identity_lock = id_lock(&root, entry)?;
        if verified_destination(&root, entry)? {
            reused.push(serde_json::to_value(entry)?);
            continue;
        }
        std::fs::create_dir_all(&parent)?;
        let stage = staging_path(&parent, "download")?;
        let result = (|| -> anyhow::Result<()> {
            download_to_stage(&entry.url, &stage, entry.size)?;
            ensure!(
                identity(&stage, entry.size)? == entry.sha256,
                "downloaded digest differs from the pinned manifest"
            );
            Ok(())
        })();
        if let Err(error) = result {
            let _ = std::fs::remove_file(&stage);
            return Err(error);
        }
        let published_file = publish(&root, entry, stage)?;
        remove_stale_staging(&destination)?;
        ensure!(
            verified_destination(&root, entry)?,
            "destination conflicts with pinned identity; existing data is preserved"
        );
        if published_file {
            published.push(serde_json::to_value(entry)?);
        } else {
            reused.push(serde_json::to_value(entry)?);
        }
    }
    let complete_receipt = complete(&root, &entries)?;
    publish_json(
        output,
        &serde_json::json!({
            "schema":IMPORT_SCHEMA,
            "kind":"download",
            "resources_root":root,
            "offline":offline,
            "published":published,
            "reused":reused,
            "receipt":complete_receipt,
        }),
    )
}

pub fn receipt_identity(receipt: &serde_json::Value) -> anyhow::Result<String> {
    ensure!(
        receipt["schema"] == RESOURCE_SCHEMA,
        "unknown resource receipt schema"
    );
    ensure!(
        receipt["status"] == "verified",
        "resource receipt is not fully verified"
    );
    Ok(digest(&serde_json::to_vec(receipt)?))
}
#[cfg(test)]
mod tests {
    use super::*;

    fn entry(root: &std::path::Path, name: &str, bytes: &[u8]) -> (ResourceIdentity, PathBuf) {
        let source = root.join(name);
        std::fs::write(&source, bytes).unwrap();
        (
            ResourceIdentity {
                backend: "backend".into(),
                model: "model".into(),
                revision: "revision".into(),
                destination: format!("backend/models/model/revision/{name}.bin"),
                source: source.clone(),
                url: String::new(),
                size: bytes.len() as u64,
                sha256: digest(bytes),
                completeness: "production_manifest".into(),
            },
            source,
        )
    }

    fn write_plan(path: &Path, entries: &[ResourceIdentity]) -> anyhow::Result<serde_json::Value> {
        let receipt = inventory(entries)?;
        publish_json(path, &receipt)?;
        Ok(receipt)
    }

    #[test]
    fn traversal_and_bad_identity_are_rejected() {
        let root = tempfile::tempdir().unwrap();
        let (mut entry, source) = entry(root.path(), "one", b"one");
        entry.destination = "../escape".into();
        assert!(entry.validate().is_err());
        entry.destination = "other/models/model/revision/one.bin".into();
        assert!(entry.validate().is_err());
        entry.destination = "backend/models/model/revision/one.bin".into();
        entry.backend = "../backend".into();
        assert!(entry.validate().is_err());
        entry.backend = "backend".into();
        entry.sha256 = "bad".into();
        assert!(entry.validate().is_err());
        entry.sha256 = digest(b"one");
        assert!(entry.validate().is_ok());
        assert!(safe_source(&source).is_ok());
        entry.source = PathBuf::new();
        assert!(entry.validate().is_err());
        entry.url = "https://example.invalid/model.bin".into();
        assert!(entry.validate().is_ok());
    }

    #[test]
    fn inventory_rejects_identity_and_incomplete_receipts() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let (entry, source) = entry(root.path(), "model", b"same");
        let mut different = entry.clone();
        different.source = root.path().join("different");
        std::fs::write(&different.source, b"changed")?;
        let receipt = inventory(&[different])?;
        assert_eq!(receipt["ready_count"], 0);
        assert_eq!(receipt["entries"][0]["status"], "unverified");
        let receipt = inventory(std::slice::from_ref(&entry))?;
        assert_eq!(receipt["ready_count"], 1);
        let mut incomplete = entry.clone();
        incomplete.completeness = "conversion_receipt".into();
        incomplete.sha256 = digest(b"same");
        let receipt = inventory(&[incomplete])?;
        assert_eq!(receipt["ready_count"], 0);
        assert_eq!(receipt["receipt_only_count"], 1);
        assert!(
            receipt["entries"][0]["reason"]
                .as_str()
                .unwrap()
                .contains("cannot substitute")
        );
        let _ = source;
        Ok(())
    }

    #[test]
    fn import_reuses_matches_and_preserves_sources() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let output_root = tempfile::tempdir()?;
        let (first, first_source) = entry(root.path(), "one", b"same");
        let (second, second_source) = entry(root.path(), "two", b"other");
        let plan_path = output_root.path().join("plan.json");
        write_plan(&plan_path, &[first.clone(), second.clone()])?;
        let receipt_path = output_root.path().join("receipt.json");
        let seed_root = tempfile::tempdir()?;
        let seeded_first = first.destination(seed_root.path())?;
        std::fs::create_dir_all(
            seeded_first
                .parent()
                .expect("destination parent should exist"),
        )?;
        std::fs::write(&seeded_first, b"same")?;
        import(&plan_path, seed_root.path(), &receipt_path)?;
        let receipt: serde_json::Value = read_json(&receipt_path)?;
        assert_eq!(receipt["published"].as_array().unwrap().len(), 1);
        assert_eq!(receipt["reused"].as_array().unwrap().len(), 1);
        assert_eq!(receipt["receipt"]["status"], "verified");
        assert!(first_source.is_file() && second_source.is_file());
        assert_eq!(std::fs::read(first_source)?, b"same");
        let first_identity = receipt_identity(&receipt["receipt"])?;
        let retry_path = output_root.path().join("retry.json");
        import(&plan_path, seed_root.path(), &retry_path)?;
        let retry: serde_json::Value = read_json(&retry_path)?;
        assert_eq!(retry["published"].as_array().unwrap().len(), 0);
        assert_eq!(retry["reused"].as_array().unwrap().len(), 2);
        assert_eq!(receipt_identity(&retry["receipt"])?, first_identity);
        Ok(())
    }

    #[test]
    fn incomplete_import_rejects_entire_plan_before_publication() -> anyhow::Result<()> {
        let sources = tempfile::tempdir()?;
        let root = tempfile::tempdir()?;
        let output = tempfile::tempdir()?;
        let (first, _) = entry(sources.path(), "production", b"verified");
        let (mut incomplete, _) = entry(sources.path(), "converted", b"converted");
        incomplete.completeness = "conversion_receipt".into();
        let plan = output.path().join("inventory.json");
        write_plan(&plan, &[first, incomplete])?;
        let receipt = output.path().join("receipt.json");
        let error = import(&plan, root.path(), &receipt).unwrap_err();
        assert!(error.to_string().contains("only production manifests"));
        assert_eq!(std::fs::read_dir(root.path())?.count(), 0);
        assert!(!receipt.exists());
        Ok(())
    }

    #[test]
    fn completion_rechecks_files_and_existing_marker() -> anyhow::Result<()> {
        let sources = tempfile::tempdir()?;
        let root = tempfile::tempdir()?;
        let (entry, source) = entry(sources.path(), "one", b"same");
        let entries = [entry];
        let destination = entries[0].destination(root.path())?;
        std::fs::create_dir_all(destination.parent().unwrap())?;
        std::fs::copy(source, &destination)?;
        let root = root.path().canonicalize()?;
        let original = complete(&root, &entries)?;
        assert_eq!(complete(&root, &entries)?, original);
        std::fs::write(&destination, b"changed")?;
        assert!(complete(&root, &entries).is_err());
        std::fs::write(&destination, b"same")?;
        let marker = root.join(format!(
            ".benchmark-import-complete-{}.json",
            plan_digest(&entries)?
        ));
        std::fs::write(marker, b"{}")?;
        assert!(complete(&root, &entries).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn destination_rejects_symlink_escape_before_creating_directories() -> anyhow::Result<()> {
        let sources = tempfile::tempdir()?;
        let root = tempfile::tempdir()?;
        let outside = tempfile::tempdir()?;
        let (entry, _) = entry(sources.path(), "one", b"same");
        std::os::unix::fs::symlink(outside.path(), root.path().join("backend"))?;
        assert!(entry.destination(root.path()).is_err());
        assert!(id_lock(root.path(), &entry).is_err());
        assert_eq!(std::fs::read_dir(outside.path())?.count(), 0);
        Ok(())
    }

    #[test]
    fn conflicting_destination_preserves_originals_and_stops() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let output_root = tempfile::tempdir()?;
        let (entry, source) = entry(root.path(), "one", b"same");
        let plan_path = output_root.path().join("plan.json");
        write_plan(&plan_path, std::slice::from_ref(&entry))?;
        let seed_root = tempfile::tempdir()?;
        let destination = entry.destination(seed_root.path())?;
        std::fs::create_dir_all(destination.parent().expect("destination parent"))?;
        std::fs::write(&destination, b"existing other data")?;
        let receipt_path = output_root.path().join("receipt.json");
        assert!(import(&plan_path, seed_root.path(), &receipt_path).is_err());
        assert_eq!(std::fs::read(&destination)?, b"existing other data");
        assert_eq!(std::fs::read(&source)?, b"same");
        assert!(!receipt_path.exists());
        let stale = std::fs::read_dir(destination.parent().unwrap())?
            .filter(|item| {
                item.as_ref()
                    .map(|item| item.file_name().to_string_lossy().ends_with(".staging"))
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(stale, 0);
        Ok(())
    }

    #[test]
    fn identity_lock_blocks_concurrent_publish() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let output_root = tempfile::tempdir()?;
        let (entry, source) = entry(root.path(), "one", b"same");
        let plan_path = output_root.path().join("plan.json");
        write_plan(&plan_path, std::slice::from_ref(&entry))?;
        let seed_root = tempfile::tempdir()?;
        let destination = entry.destination(seed_root.path())?;
        std::fs::create_dir_all(destination.parent().expect("destination parent"))?;
        let lock = id_lock(seed_root.path(), &entry)?;
        let receipt_path = output_root.path().join("receipt.json");
        let result = import(&plan_path, seed_root.path(), &receipt_path);
        drop(lock);
        assert!(
            result
                .expect_err("locked importer must fail")
                .to_string()
                .contains("another importer owns this resource identity")
        );
        assert_eq!(std::fs::read(&source)?, b"same");
        assert!(!destination.exists());
        Ok(())
    }

    #[test]
    fn interrupted_staging_is_cleaned_on_retry() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let output_root = tempfile::tempdir()?;
        let (entry, _source) = entry(root.path(), "one", b"same");
        let plan_path = output_root.path().join("plan.json");
        write_plan(&plan_path, std::slice::from_ref(&entry))?;
        let seed_root = tempfile::tempdir()?;
        let destination = entry.destination(seed_root.path())?;
        std::fs::create_dir_all(destination.parent().expect("destination parent"))?;
        let stale = destination
            .parent()
            .unwrap()
            .join(".resource-import-1-2.staging");
        std::fs::write(&stale, b"interrupted copy")?;
        let receipt_path = output_root.path().join("receipt.json");
        import(&plan_path, seed_root.path(), &receipt_path)?;
        assert!(!stale.exists());
        assert_eq!(std::fs::read(&destination)?, b"same");
        Ok(())
    }

    fn server(body: &'static [u8]) -> (String, std::thread::JoinHandle<()>) {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/resource", listener.local_addr().unwrap());
        let task = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            loop {
                let mut byte = [0];
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
                if request.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                )
                .unwrap();
            stream.write_all(body).unwrap();
        });
        (url, task)
    }

    fn download_entry(
        root: &std::path::Path,
        name: &str,
        bytes: &[u8],
        url: &str,
    ) -> ResourceIdentity {
        let _ = root;
        ResourceIdentity {
            backend: "backend".into(),
            model: "model".into(),
            revision: "revision".into(),
            destination: format!("backend/models/model/revision/{name}.bin"),
            source: PathBuf::new(),
            url: url.to_owned(),
            size: bytes.len() as u64,
            sha256: digest(bytes),
            completeness: "production_manifest".into(),
        }
    }

    fn write_download_plan(path: &Path, entries: &[ResourceIdentity]) -> anyhow::Result<()> {
        publish_json(
            path,
            &serde_json::json!({
                "schema":PLAN_SCHEMA,
                "backend":"backend",
                "model":"model",
                "revision":"revision",
                "completeness":"production_manifest",
                "files":entries.iter().map(|entry| serde_json::json!({
                    "destination":entry.destination,
                    "url":entry.url,
                    "size":entry.size,
                    "sha256":entry.sha256,
                    "completeness":entry.completeness,
                })).collect::<Vec<_>>(),
            }),
        )
    }

    #[test]
    fn download_rejects_digest_mismatch_and_cleans_staging() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let output_root = tempfile::tempdir()?;
        let (url, task) = server(b"changed bytes");
        let entry = download_entry(root.path(), "model", b"expected bytes", &url);
        let plan_path = output_root.path().join("plan.json");
        write_download_plan(&plan_path, std::slice::from_ref(&entry))?;
        let receipt_path = output_root.path().join("receipt.json");
        assert!(download(&plan_path, root.path(), false, &receipt_path).is_err());
        task.join().unwrap();
        assert!(!receipt_path.exists());
        assert!(!entry.destination(root.path())?.exists());
        let destination_parent = entry
            .destination(root.path())?
            .parent()
            .expect("destination parent should exist")
            .to_owned();
        let leftovers = std::fs::read_dir(&destination_parent)?
            .filter(|item| {
                item.as_ref()
                    .map(|item| {
                        let name = item.file_name().to_string_lossy().into_owned();
                        name != ".resource-import.lock" && !name.ends_with(".staging")
                    })
                    .unwrap_or(true)
            })
            .count();
        assert_eq!(leftovers, 0);
        Ok(())
    }

    #[test]
    fn download_stops_at_declared_size_budget() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let stage = root.path().join("download.staging");
        let (url, task) = server(b"larger than declared");
        let error = download_to_stage(&url, &stage, 4).unwrap_err();
        task.join().unwrap();
        assert!(error.to_string().contains("byte budget"));
        assert!(stage.metadata()?.len() <= 4);
        Ok(())
    }

    #[test]
    fn download_reuses_verified_destinations_offline() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let output_root = tempfile::tempdir()?;
        let entry = download_entry(
            root.path(),
            "model",
            b"offline reuse",
            "https://example.invalid/never",
        );
        let plan_path = output_root.path().join("plan.json");
        write_download_plan(&plan_path, std::slice::from_ref(&entry))?;
        let destination = entry.destination(root.path())?;
        std::fs::create_dir_all(destination.parent().expect("destination parent"))?;
        std::fs::write(&destination, b"offline reuse")?;
        let receipt_path = output_root.path().join("receipt.json");
        download(&plan_path, root.path(), true, &receipt_path)?;
        let receipt: serde_json::Value = read_json(&receipt_path)?;
        assert_eq!(receipt["reused"].as_array().unwrap().len(), 1);
        assert_eq!(receipt["published"].as_array().unwrap().len(), 0);
        assert_eq!(std::fs::read(&destination)?, b"offline reuse");
        Ok(())
    }

    #[test]
    fn receipt_only_entries_cannot_be_downloaded() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let output_root = tempfile::tempdir()?;
        let mut entry = download_entry(
            root.path(),
            "model",
            b"converted",
            "https://example.invalid/model",
        );
        entry.completeness = "conversion_receipt".into();
        let plan_path = output_root.path().join("plan.json");
        write_download_plan(&plan_path, &[entry])?;
        let receipt_path = output_root.path().join("receipt.json");
        assert!(download(&plan_path, root.path(), false, &receipt_path).is_err());
        assert!(!root.path().join("backend").exists());
        Ok(())
    }
}
