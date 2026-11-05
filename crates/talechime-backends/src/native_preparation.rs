//! Fixed component preparation; model-specific transforms stay with their adapters.
mod control;
mod plan;
mod publication;
mod workspace;
pub(crate) use workspace::Workspace;
#[cfg(any(test, feature = "cosyvoice3", feature = "kokoro", feature = "indextts"))]
mod tensors;
#[cfg(any(test, feature = "cosyvoice3", feature = "kokoro"))]
mod weight_norm;
#[cfg(any(test, feature = "cosyvoice3", feature = "indextts"))]
pub(crate) use tensors::Archive;
#[cfg(feature = "kokoro")]
pub(crate) use tensors::SafeArchive;
#[cfg(any(test, feature = "cosyvoice3", feature = "kokoro", feature = "indextts"))]
pub(crate) use tensors::{TensorSpec, write as write_tensors};
#[cfg(any(test, feature = "cosyvoice3", feature = "kokoro"))]
pub(crate) use weight_norm::fold as fold_weight_norm;
#[cfg(all(test, feature = "cosyvoice3", feature = "indextts"))]
mod acceptance;
#[cfg(test)]
mod tests;

pub use control::PreparationControl;
pub(crate) use plan::{Input, Output};
pub use plan::{PreparationPhase, PreparationPlan, PreparationProgress};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;

/// A typed failure independent of a host's UI and task persistence.
#[derive(Debug, thiserror::Error)]
pub enum PreparationError {
    /// The caller cancelled preparation and all owned work has stopped.
    #[error("resource preparation cancelled")]
    Cancelled,
    /// Inputs or target no longer match the user's confirmed plan.
    #[error("resource preparation plan changed; confirm again")]
    PlanChanged,
    /// Another preparation owns the resource namespace.
    #[error("resource preparation is busy")]
    Busy,
    /// A pinned input or generated resource failed validation.
    #[error("resource integrity check failed: {0}")]
    Integrity(String),
    /// Storage cannot hold the worst-case working set.
    #[error("insufficient disk space for resource preparation")]
    Space,
}

pub(crate) type Transform = fn(&Workspace<'_>) -> anyhow::Result<()>;

#[derive(Clone)]
pub(crate) struct Recipe {
    pub id: &'static str,
    pub target: PathBuf,
    pub inputs: &'static [Input],
    pub outputs: &'static [Output],
    pub transform_outputs: &'static [&'static str],
    pub workspace_bytes: u64,
    pub transform: Transform,
}
impl Recipe {
    /// Consume verified finished files while keeping native source recipes reproducible.
    #[cfg(any(test, feature = "kokoro", feature = "cosyvoice3", feature = "indextts"))]
    pub(crate) fn distributed(
        mut self,
        id: &'static str,
        inputs: &'static [Input],
        target: PathBuf,
    ) -> Self {
        self.id = id;
        self.inputs = inputs;
        self.target = target;
        self.transform_outputs = &[];
        self.workspace_bytes = 0;
        self.transform = |work| work.copy_all();
        self
    }
    pub async fn verify_existing(
        &self,
        directory: PathBuf,
        control: PreparationControl,
    ) -> anyhow::Result<()> {
        let recipe = self.clone();
        control::worker(control, move |control| {
            recipe.verify_files(&directory, &control)
        })
        .await
    }
    pub fn verify_files(
        &self,
        directory: &Path,
        control: &PreparationControl,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            std::fs::symlink_metadata(directory)?.is_dir(),
            "invalid published resource directory"
        );
        anyhow::ensure!(
            plan::marker_matches(&directory.join("preparation.json"), &plan::marker(self)?)?,
            PreparationError::Integrity("resource publication recipe differs".into())
        );
        for output in self.outputs {
            anyhow::ensure!(
                plan::verified(
                    &directory.join(output.path),
                    output.size,
                    output.sha256,
                    control
                )?,
                PreparationError::Integrity(output.path.into())
            );
        }
        Ok(())
    }
    #[cfg(any(test, feature = "kokoro", feature = "cosyvoice3", feature = "indextts"))]
    pub fn status(&self, root: &Path) -> crate::ResourceStatus {
        let ready = self
            .outputs
            .iter()
            .filter(|file| {
                crate::resource_state::entry_ready(
                    &root.join(&self.target).join(file.path),
                    file.size,
                )
            })
            .count();
        let marker_ready = plan::marker(self)
            .and_then(|expected| {
                plan::marker_matches(&root.join(&self.target).join("preparation.json"), &expected)
            })
            .unwrap_or(false);
        let missing_bytes = self
            .outputs
            .iter()
            .filter(|file| {
                !crate::resource_state::entry_ready(
                    &root.join(&self.target).join(file.path),
                    file.size,
                )
            })
            .map(|file| file.size)
            .sum();
        if ready == self.outputs.len() && marker_ready {
            crate::ResourceStatus::Ready
        } else if ready == 0 {
            crate::ResourceStatus::Missing { missing_bytes }
        } else {
            crate::ResourceStatus::Partial {
                ready,
                total: self.outputs.len() + usize::from(!marker_ready),
                missing_bytes,
            }
        }
    }

    #[cfg(test)]
    pub fn plan(
        &self,
        root: &Path,
        downloads: &crate::downloads::Downloads,
    ) -> anyhow::Result<PreparationPlan> {
        plan::build(self, root, downloads, &PreparationControl::default())
    }
    pub async fn execute(
        self,
        root: PathBuf,
        downloads: crate::downloads::Downloads,
        confirmed: String,
        control: PreparationControl,
        progress: mpsc::Sender<PreparationProgress>,
        file_progress: mpsc::Sender<talechime_protocol::Event>,
    ) -> anyhow::Result<()> {
        let plan = plan_recipe(
            self.clone(),
            root.clone(),
            downloads.clone(),
            control.clone(),
        )
        .await?;
        anyhow::ensure!(plan.fingerprint == confirmed, PreparationError::PlanChanged);
        control.check()?;
        if plan.ready {
            self.verify_existing(root.join(&self.target), control.clone())
                .await?;
            return control.commit(|| Ok(()));
        }
        let transaction = publication::Transaction::new(&root, &self, &plan)?;
        // A competing publisher/cache operation may have completed after the
        // first preview. Recheck while this execution holds the publication lease.
        let locked = plan_recipe(
            self.clone(),
            root.clone(),
            downloads.clone(),
            control.clone(),
        )
        .await?;
        anyhow::ensure!(
            locked.fingerprint == confirmed,
            PreparationError::PlanChanged
        );
        for input in self.inputs {
            control.check()?;
            let dest = transaction.inputs.join(input.path);
            let mut parent = transaction.inputs.clone();
            for part in Path::new(input.path)
                .parent()
                .into_iter()
                .flat_map(Path::components)
            {
                parent.push(part);
                publication::directory(&parent)?;
            }
            let local_path = dest.clone();
            let local_downloads = downloads.clone();
            let local_root = root.clone();
            let reused = control::worker(control.clone(), move |control| {
                if plan::verified(&local_path, input.size, input.sha256, &control)? {
                    return Ok(true);
                }
                if let Some(cached) =
                    plan::cached_input(&local_root, &local_downloads, input, &control)?
                {
                    if local_path.exists() {
                        anyhow::ensure!(
                            std::fs::symlink_metadata(&local_path)?.is_file(),
                            "invalid preparation input view"
                        );
                        std::fs::remove_file(&local_path)?;
                    }
                    link_or_copy(&cached, &local_path, &control)?;
                    return Ok(true);
                }
                Ok(false)
            })
            .await?;
            if reused {
                continue;
            }
            plan::send(
                &progress,
                PreparationPhase::Downloading,
                input.path,
                0,
                self.inputs.len() as u64,
            );
            if dest.exists() {
                // Remove only our rejected input view, never its shared cache blob.
                anyhow::ensure!(
                    std::fs::symlink_metadata(&dest)?.is_file(),
                    "invalid preparation input view"
                );
                std::fs::remove_file(&dest)?;
            }
            let mut valid = false;
            for refresh in [false, true] {
                control.check()?;
                let bridge = crate::downloads::FileProgressBridge::new(
                    file_progress.clone(),
                    input.path.into(),
                    input.size,
                );
                let transfer = async {
                    if refresh {
                        downloads
                            .refresh(&dest, input.repo, input.revision, input.filename, bridge)
                            .await
                    } else {
                        downloads
                            .ensure(&dest, input.repo, input.revision, input.filename, bridge)
                            .await
                    }
                };
                let outcome = {
                    tokio::pin!(transfer);
                    tokio::select! {
                        result = &mut transfer => result,
                        _ = control.cancelled() => Err(PreparationError::Cancelled.into()),
                    }
                };
                anyhow::ensure!(
                    !downloads.checkpoint_failed(),
                    "download recovery checkpoint could not be persisted"
                );
                outcome?;
                let path = dest.clone();
                valid = control::worker(control.clone(), move |control| {
                    plan::verified(&path, input.size, input.sha256, &control)
                })
                .await?;
                if valid {
                    break;
                }
                std::fs::remove_file(&dest)?;
            }
            anyhow::ensure!(valid, PreparationError::Integrity(input.path.into()));
        }
        control::worker(control, move |control| {
            plan::send(
                &progress,
                if self.transform_outputs.is_empty() {
                    PreparationPhase::Verifying
                } else {
                    PreparationPhase::Converting
                },
                self.id,
                0,
                0,
            );
            let workspace = Workspace::new(
                &transaction.inputs,
                transaction.staging.path(),
                &transaction.components,
                &self,
                &control,
                &progress,
            );
            (self.transform)(&workspace)?;
            workspace.finish()?;
            control.check()?;
            transaction.publish(&self, &control, &progress)
        })
        .await
    }
}

pub(crate) async fn plan_recipe(
    recipe: Recipe,
    root: PathBuf,
    downloads: crate::downloads::Downloads,
    control: PreparationControl,
) -> anyhow::Result<PreparationPlan> {
    control::worker(control, move |control| {
        plan::build(&recipe, &root, &downloads, &control)
    })
    .await
}

pub(crate) async fn prepare_recipe(
    recipe: Recipe,
    root: PathBuf,
    downloads: crate::downloads::Downloads,
    events: mpsc::Sender<talechime_protocol::Event>,
) -> anyhow::Result<()> {
    let control = PreparationControl::default();
    let plan = plan_recipe(
        recipe.clone(),
        root.clone(),
        downloads.clone(),
        control.clone(),
    )
    .await?;
    let (tx, mut rx) = mpsc::channel::<PreparationProgress>(32);
    let stage_events = events.clone();
    let forward = async move {
        while let Some(update) = rx.recv().await {
            let stage = match update.phase {
                PreparationPhase::Downloading => "下载",
                PreparationPhase::Converting => "转换",
                PreparationPhase::Verifying => "校验",
                PreparationPhase::Publishing => "发布",
            };
            if stage_events
                .send(talechime_protocol::Event::ResourceState {
                    stage: stage.into(),
                    resource: update.component,
                })
                .await
                .is_err()
            {
                break;
            }
        }
    };
    let (result, ()) = tokio::join!(
        recipe.execute(root, downloads, plan.fingerprint, control, tx, events),
        forward
    );
    result
}

pub(crate) fn link_or_copy(
    source: &Path,
    target: &Path,
    control: &PreparationControl,
) -> anyhow::Result<()> {
    use std::io::{Read, Write};
    control.check()?;
    if std::fs::hard_link(source, target).is_ok() {
        return Ok(());
    }
    let mut source = std::fs::File::open(source)?;
    let mut target = std::fs::File::create_new(target)?;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        control.check()?;
        let n = source.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        target.write_all(&buffer[..n])?;
    }
    Ok(())
}
