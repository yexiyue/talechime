//! Batch planning and publication without any model or host runtime dependency.
use super::{VoiceAsset, VoiceId, VoiceLibrary, VoiceLibraryError as Error, package, store};
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Debug, Clone)]
pub struct TransferLimits {
    pub max_packages: usize,
    pub max_assets: usize,
    pub max_manifest_bytes: u64,
    pub max_total_manifest_bytes: u64,
    pub max_wav_bytes: u64,
    pub max_audio_bytes: u64,
    pub max_input_bytes: u64,
}
impl Default for TransferLimits {
    fn default() -> Self {
        Self {
            max_packages: 32,
            max_assets: 256,
            max_manifest_bytes: 4 * 1024 * 1024,
            max_total_manifest_bytes: 16 * 1024 * 1024,
            max_wav_bytes: 64 * 1024 * 1024,
            max_audio_bytes: 1024 * 1024 * 1024,
            max_input_bytes: 1024 * 1024 * 1024,
        }
    }
}
impl TransferLimits {
    fn validate(&self) -> Result<(), Error> {
        let hard = Self::default();
        if self.max_packages == 0
            || self.max_assets == 0
            || self.max_packages > hard.max_packages
            || self.max_assets > hard.max_assets
            || self.max_manifest_bytes > hard.max_manifest_bytes
            || self.max_total_manifest_bytes > hard.max_total_manifest_bytes
            || self.max_wav_bytes > hard.max_wav_bytes
            || self.max_audio_bytes > hard.max_audio_bytes
            || self.max_input_bytes > hard.max_input_bytes
        {
            return Err(Error::LimitExceeded("invalid limits".into()));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, Serialize)]
pub enum TransferStage {
    Checking,
    Planning,
    Writing,
    Committing,
    Completed,
}
#[derive(Debug, Clone, Serialize)]
pub struct TransferProgress {
    pub stage: TransferStage,
    pub completed: usize,
    pub total: usize,
    pub bytes: u64,
}
#[derive(Clone, Default)]
pub struct TransferControl {
    cancelled: Arc<AtomicBool>,
    progress: Option<Arc<dyn Fn(TransferProgress) + Send + Sync>>,
}
impl TransferControl {
    pub fn with_progress(callback: impl Fn(TransferProgress) + Send + Sync + 'static) -> Self {
        Self {
            progress: Some(Arc::new(callback)),
            ..Self::default()
        }
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
    pub(super) fn check(&self) -> Result<(), Error> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
    pub(super) fn emit(&self, stage: TransferStage, completed: usize, total: usize, bytes: u64) {
        if let Some(callback) = &self.progress {
            callback(TransferProgress {
                stage,
                completed,
                total,
                bytes,
            });
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ImportConflict {
    AlreadyPresent,
    MetadataDifference,
    IdentityConflict,
    SameName,
    DuplicateContent,
}
/// Source identity in the caller's package list, independent of archive paths.
#[derive(Debug, Clone, Serialize)]
pub struct ImportSource {
    pub package_index: usize,
    pub voice_id: VoiceId,
}
#[derive(Debug, Clone, Serialize)]
pub struct ImportEntry {
    pub index: usize,
    pub asset: VoiceAsset,
    pub sources: Vec<ImportSource>,
    pub conflicts: Vec<ImportConflict>,
    pub existing_id: Option<String>,
    pub existing_asset: Option<VoiceAsset>,
}
pub struct ImportPreview {
    pub(super) root: PathBuf,
    pub(super) snapshots: Vec<store::Snapshot>,
    entries: Vec<ImportEntry>,
}
impl ImportPreview {
    pub fn entries(&self) -> &[ImportEntry] {
        &self.entries
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportAction {
    Import,
    ImportAsNew,
    Skip,
}
#[derive(Debug, Clone)]
pub struct ImportDecision {
    pub entry: usize,
    pub action: ImportAction,
}
struct PlannedEntry {
    snapshot: store::Snapshot,
    index: usize,
    action: ImportAction,
    target: VoiceId,
    origin: String,
    sources: Vec<ImportSource>,
}
pub struct ImportPlan {
    root: PathBuf,
    entries: Vec<PlannedEntry>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ImportStatus {
    Imported,
    AlreadyPresent,
    Skipped,
    Failed,
    Cancelled,
    NotAttempted,
}
#[derive(Debug)]
pub struct ImportItemResult {
    pub name: String,
    pub entry: usize,
    pub sources: Vec<ImportSource>,
    pub local_id: Option<String>,
    pub status: ImportStatus,
    pub error: Option<Error>,
}
#[derive(Debug, Default)]
pub struct ImportReport {
    pub items: Vec<ImportItemResult>,
}
#[derive(Debug, Clone, Copy)]
pub enum ExportTarget {
    CreateNew,
    ReplaceConfirmed,
}
#[derive(Debug)]
pub struct ExportReport {
    pub voices: usize,
    pub bytes: u64,
}
fn origin(a: &VoiceAsset) -> String {
    format!("{}:{}", a.id(), a.asset_digest())
}

impl VoiceLibrary {
    /// Validate every package before any publication; the returned preview owns frozen references.
    pub fn inspect_import(
        &self,
        paths: &[PathBuf],
        limits: TransferLimits,
        control: &TransferControl,
    ) -> Result<ImportPreview, Error> {
        limits.validate()?;
        if paths.is_empty() || paths.len() > limits.max_packages {
            return Err(Error::LimitExceeded("package count".into()));
        }
        let mut input = 0u64;
        for path in paths {
            input = input
                .checked_add(std::fs::metadata(path)?.len())
                .ok_or_else(|| Error::LimitExceeded("input bytes".into()))?;
        }
        if input > limits.max_input_bytes {
            return Err(Error::LimitExceeded("input bytes".into()));
        }
        let mut input_bytes = 0;
        let mut bytes = 0;
        let mut manifests = 0;
        let mut snapshots = Vec::<store::Snapshot>::new();
        let mut sources = Vec::<Vec<ImportSource>>::new();
        let mut seen = HashMap::<String, usize>::new();
        let mut count = 0;
        for (p, path) in paths.iter().enumerate() {
            let items = package::read(
                path,
                &limits,
                control,
                &mut bytes,
                &mut manifests,
                &mut input_bytes,
            )
            .map_err(|error| Error::Input {
                index: p,
                error: Box::new(error),
            })?;
            count += items.len();
            if count > limits.max_assets {
                return Err(Error::LimitExceeded("asset count".into()));
            }
            for snapshot in items {
                let source = ImportSource {
                    package_index: p,
                    voice_id: snapshot.asset.id.clone(),
                };
                if let Some(&index) = seen.get(&origin(&snapshot.asset)) {
                    sources[index].push(source);
                } else {
                    let index = snapshots.len();
                    seen.insert(origin(&snapshot.asset), index);
                    snapshots.push(snapshot);
                    sources.push(vec![source]);
                }
            }
            control.emit(TransferStage::Checking, p + 1, paths.len(), bytes);
        }
        let inventory = self.inventory()?;
        let unreadable: HashSet<_> = inventory
            .diagnostics
            .iter()
            .map(|d| d.id.as_str())
            .collect();
        let mut entries = Vec::new();
        for (index, snapshot) in snapshots.iter().enumerate() {
            let a = &snapshot.asset;
            let mut conflicts = Vec::new();
            let mut existing_id = None;
            let mut existing_asset = None;
            if unreadable.contains(a.id()) {
                conflicts.push(ImportConflict::IdentityConflict);
                existing_id = Some(a.id().into());
            }
            for local in &inventory.assets {
                if local.id() == a.id() {
                    existing_id = Some(local.id().to_owned());
                    existing_asset = Some(local.clone());
                    if local.fingerprint() != a.fingerprint() {
                        conflicts.push(ImportConflict::IdentityConflict);
                    } else if local.asset_digest() != a.asset_digest() {
                        conflicts.push(ImportConflict::MetadataDifference);
                    } else {
                        conflicts.push(ImportConflict::AlreadyPresent);
                    }
                } else if local.fingerprint() == a.fingerprint() {
                    conflicts.push(ImportConflict::DuplicateContent);
                }
                if local.id() != a.id() && local.name() == a.name() {
                    conflicts.push(ImportConflict::SameName);
                }
            }
            for (other, s) in snapshots
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
            {
                if s.asset.id() == a.id() {
                    let kind = if s.asset.fingerprint() != a.fingerprint() {
                        ImportConflict::IdentityConflict
                    } else {
                        ImportConflict::MetadataDifference
                    };
                    conflicts.push(kind);
                } else if s.asset.fingerprint() == a.fingerprint() {
                    conflicts.push(ImportConflict::DuplicateContent);
                }
                if other < index && s.asset.name() == a.name() {
                    conflicts.push(ImportConflict::SameName);
                }
            }
            if self.root.exists() {
                let _lock = store::lock(self)?;
                if let Some(local) = store::find_origin(self, &origin(a))? {
                    existing_id = Some(local.id().into());
                    existing_asset = Some(local.clone());
                    conflicts.retain(|c| *c != ImportConflict::IdentityConflict);
                    conflicts.push(ImportConflict::AlreadyPresent);
                }
            }
            entries.push(ImportEntry {
                index,
                asset: a.clone(),
                sources: sources[index].clone(),
                conflicts,
                existing_id,
                existing_asset,
            });
        }
        control.check()?;
        Ok(ImportPreview {
            root: self.root.clone(),
            snapshots,
            entries,
        })
    }
    /// Consume a validated preview. Only explicitly selected entries enter the plan.
    pub fn plan_import(
        &self,
        preview: ImportPreview,
        decisions: Vec<ImportDecision>,
    ) -> Result<ImportPlan, Error> {
        if preview.root != self.root {
            return Err(Error::StalePreview);
        }
        let mut selected = HashMap::new();
        for d in decisions {
            if d.entry >= preview.entries.len() || selected.insert(d.entry, d.action).is_some() {
                return Err(Error::InvalidPackage("invalid selection".into()));
            }
        }
        if selected.is_empty() {
            return Err(Error::InvalidPackage("empty selection".into()));
        }
        let existing = self.inventory()?.assets;
        let mut used: HashSet<String> = existing.iter().map(|a| a.id().into()).collect();
        for entry in &preview.entries {
            used.insert(entry.asset.id().into());
        }
        let mut planned_ids = HashSet::new();
        let mut entries = Vec::new();
        for (index, snapshot) in preview.snapshots.into_iter().enumerate() {
            let Some(action) = selected.remove(&index) else {
                continue;
            };
            let entry = &preview.entries[index];
            if action == ImportAction::Import
                && entry.conflicts.contains(&ImportConflict::IdentityConflict)
            {
                return Err(Error::Conflict(format!(
                    "voice {} requires a decision",
                    snapshot.asset.id()
                )));
            }
            let target = if let Some(existing) = &entry.existing_id
                && !entry.conflicts.contains(&ImportConflict::IdentityConflict)
            {
                existing.parse()?
            } else if action == ImportAction::ImportAsNew {
                let mut n = 1u64;
                loop {
                    let id = format!("import-{}-{n}", &snapshot.asset.asset_digest()[..16]);
                    if used.insert(id.clone()) {
                        break id.parse()?;
                    }
                    n += 1;
                }
            } else {
                snapshot.asset.id.clone()
            };
            if action != ImportAction::Skip && !planned_ids.insert(target.to_string()) {
                return Err(Error::Conflict(
                    "multiple selected entries target one ID".into(),
                ));
            }
            let key = origin(&snapshot.asset);
            entries.push(PlannedEntry {
                snapshot,
                index,
                action,
                target,
                origin: key,
                sources: entry.sources.clone(),
            });
        }
        Ok(ImportPlan {
            root: self.root.clone(),
            entries,
        })
    }
    /// Publish one item at a time; the report always accounts for every selected entry.
    pub fn import_selected(&self, plan: ImportPlan, control: &TransferControl) -> ImportReport {
        let total = plan.entries.len();
        let mut report = ImportReport::default();
        let mut stopped = false;
        for item in plan.entries {
            let mut result = ImportItemResult {
                name: item.snapshot.asset.name().to_owned(),
                entry: item.index,
                sources: item.sources,
                local_id: None,
                status: ImportStatus::NotAttempted,
                error: None,
            };
            if plan.root != self.root {
                result.status = ImportStatus::Failed;
                result.error = Some(Error::StalePreview);
            } else if item.action == ImportAction::Skip {
                result.status = ImportStatus::Skipped;
            } else if stopped {
            } else if control.is_cancelled() {
                result.status = ImportStatus::Cancelled;
            } else {
                let mut a = item.snapshot.asset;
                a.id = item.target;
                let digest = a.compute_digest();
                let publication = digest.and_then(|digest| {
                    a.asset_digest = digest;
                    store::publish(
                        self,
                        a,
                        item.snapshot.wav.path(),
                        Some(item.origin),
                        control,
                    )
                });
                match publication {
                    Ok((a, created)) => {
                        result.name = a.name().into();
                        result.local_id = Some(a.id().into());
                        result.status = if created {
                            ImportStatus::Imported
                        } else {
                            ImportStatus::AlreadyPresent
                        };
                    }
                    Err(Error::Cancelled) => result.status = ImportStatus::Cancelled,
                    Err(e) => {
                        stopped = !matches!(e, Error::Conflict(_));
                        result.status = ImportStatus::Failed;
                        result.error = Some(e);
                    }
                }
            }
            report.items.push(result);
            control.emit(TransferStage::Committing, report.items.len(), total, 0);
        }
        control.emit(TransferStage::Completed, total, total, 0);
        report
    }
    /// Freeze selected assets and atomically publish exactly one complete package.
    pub fn export_bundle(
        &self,
        ids: &[VoiceId],
        path: &Path,
        target: ExportTarget,
        limits: TransferLimits,
        control: &TransferControl,
    ) -> Result<ExportReport, Error> {
        limits.validate()?;
        control.check()?;
        if ids.is_empty() || ids.len() > limits.max_assets {
            return Err(Error::LimitExceeded("export asset count".into()));
        }
        let mut unique = HashSet::new();
        let mut snapshots = Vec::new();
        let scratch = store::Scratch::new()?;
        let mut total = 0u64;
        for id in ids {
            control.check()?;
            if !unique.insert(id.as_str()) {
                return Err(Error::Conflict("duplicate export ID".into()));
            }
            let _lock = store::lock(self)?;
            let snapshot = store::snapshot_in(self, id, &scratch)?;
            total = total
                .checked_add(snapshot.asset.wav_bytes())
                .ok_or_else(|| Error::LimitExceeded("audio total".into()))?;
            if snapshot.asset.wav_bytes() > limits.max_wav_bytes || total > limits.max_audio_bytes {
                return Err(Error::LimitExceeded("audio total".into()));
            }
            snapshots.push(snapshot);
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        if parent
            .canonicalize()?
            .starts_with(self.root.canonicalize()?)
        {
            return Err(Error::Conflict(
                "export target is inside the voice library".into(),
            ));
        }
        let file = tempfile::NamedTempFile::new_in(parent)?;
        package::write(file.reopen()?, &snapshots, &limits, control)?;
        let bytes = file.as_file().metadata()?.len();
        control.check()?;
        match target {
            ExportTarget::CreateNew => file.persist_noclobber(path),
            ExportTarget::ReplaceConfirmed => file.persist(path),
        }
        .map_err(|e| {
            if e.error.kind() == std::io::ErrorKind::AlreadyExists {
                Error::Conflict("export target exists".into())
            } else {
                e.error.into()
            }
        })?;
        Ok(ExportReport {
            voices: snapshots.len(),
            bytes,
        })
    }
}
