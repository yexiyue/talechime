use super::{
    PreparationControl, PreparationError, PreparationPhase, PreparationPlan, PreparationProgress,
    Recipe, plan,
};
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};
use tokio::sync::mpsc;
mod cleanup;

pub(super) struct Transaction {
    pub inputs: PathBuf,
    pub components: PathBuf,
    pub staging: tempfile::TempDir,
    target: PathBuf,
    root: PathBuf,
    _lease: File,
}
impl Transaction {
    pub fn new(root: &Path, recipe: &Recipe, plan: &PreparationPlan) -> anyhow::Result<Self> {
        let managed = root.join(".native-preparation");
        directory(&managed)?;
        let lock_path = managed.join("lease");
        if let Ok(metadata) = std::fs::symlink_metadata(&lock_path) {
            anyhow::ensure!(metadata.is_file(), "invalid preparation lease");
        }
        let lease = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)?;
        lease.try_lock().map_err(|_| PreparationError::Busy)?;
        cleanup::remove_abandoned(&managed, recipe)?;
        let available = std::fs::File::open(root)?.metadata()?;
        anyhow::ensure!(available.is_dir(), "preparation root is not a directory");
        anyhow::ensure!(
            fs2::available_space(root)? >= plan.peak_additional_bytes,
            PreparationError::Space
        );
        let inputs = managed.join("inputs").join(recipe.id);
        directory(&managed.join("inputs"))?;
        directory(&inputs)?;
        let components = managed.join("components");
        directory(&components)?;
        let staging = tempfile::Builder::new()
            .prefix(&format!("staging-{}-", recipe.id))
            .tempdir_in(&managed)?;
        let mut marker = File::create_new(staging.path().join("preparation.json"))?;
        marker.write_all(&plan::marker(recipe)?)?;
        marker.sync_all()?;
        Ok(Self {
            _lease: lease,
            inputs,
            components,
            staging,
            target: root.join(&recipe.target),
            root: root.to_owned(),
        })
    }
    pub fn publish(
        self,
        recipe: &Recipe,
        control: &PreparationControl,
        progress: &mpsc::Sender<PreparationProgress>,
    ) -> anyhow::Result<()> {
        for (index, file) in recipe.outputs.iter().enumerate() {
            control.check()?;
            plan::send(
                progress,
                PreparationPhase::Verifying,
                file.path,
                index as u64,
                recipe.outputs.len() as u64,
            );
            let path = self.staging.path().join(file.path);
            anyhow::ensure!(
                plan::verified(&path, file.size, file.sha256, control)?,
                PreparationError::Integrity(file.path.into())
            );
            File::open(path)?.sync_all()?;
        }
        let manifest = plan::marker(recipe)?;
        anyhow::ensure!(
            plan::marker_matches(&self.staging.path().join("preparation.json"), &manifest)?,
            PreparationError::Integrity("staging publication marker changed".into())
        );
        sync_directory(self.staging.path())?;
        control.check()?;
        plan::send(progress, PreparationPhase::Publishing, recipe.id, 0, 1);
        control.commit(|| {
            let mut checked = self.root.clone();
            for component in recipe
                .target
                .parent()
                .into_iter()
                .flat_map(Path::components)
            {
                checked.push(component);
                directory(&checked)?;
            }
            if let Ok(metadata) = std::fs::symlink_metadata(&self.target) {
                anyhow::ensure!(metadata.is_dir(), "invalid published resource directory");
            }
            if self.target.exists() {
                anyhow::ensure!(
                    plan::marker_matches(&self.target.join("preparation.json"), &manifest)?,
                    PreparationError::Integrity(
                        "existing resource version differs; it was preserved".into()
                    )
                );
                for file in recipe.outputs {
                    anyhow::ensure!(
                        plan::verified(
                            &self.target.join(file.path),
                            file.size,
                            file.sha256,
                            control
                        )?,
                        PreparationError::Integrity(
                            "existing resource version differs; it was preserved".into()
                        )
                    );
                }
                return Ok(());
            }
            std::fs::create_dir_all(
                self.target
                    .parent()
                    .ok_or_else(|| anyhow::anyhow!("missing resource parent"))?,
            )?;
            std::fs::rename(self.staging.path(), &self.target)?;
            Ok(())
        })?;
        // The rename has committed. Late cancellation cannot undo it even if
        // the following durability check fails; report that distinction.
        sync_directory(self.target.parent().ok_or_else(|| anyhow::anyhow!("missing resource parent"))?)
            .map_err(|error| anyhow::anyhow!("resource version published, but directory durability could not be confirmed: {error}"))
    }
}
fn sync_directory(path: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

// Every caller supplies a child of an already checked directory. Reject links
// before creating files; preparation never follows an existing managed alias.
pub(super) fn directory(path: &Path) -> anyhow::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => anyhow::ensure!(metadata.is_dir(), "invalid preparation directory"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(path)?
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}
