//! Reuse complete, pinned components across retries and model variants.
use super::{
    Output, PreparationControl, PreparationError, PreparationPhase, PreparationProgress, Recipe,
    link_or_copy, plan, publication,
};
use std::{cell::RefCell, collections::BTreeSet, path::Path};
use tokio::sync::mpsc;

pub(crate) struct Workspace<'a> {
    input: &'a Path,
    control: &'a PreparationControl,
    output: &'a Path,
    components: &'a Path,
    recipe: &'a Recipe,
    progress: &'a mpsc::Sender<PreparationProgress>,
    completed: RefCell<BTreeSet<&'static str>>,
}
impl<'a> Workspace<'a> {
    pub(super) fn new(
        input: &'a Path,
        output: &'a Path,
        components: &'a Path,
        recipe: &'a Recipe,
        control: &'a PreparationControl,
        progress: &'a mpsc::Sender<PreparationProgress>,
    ) -> Self {
        Self {
            input,
            output,
            components,
            recipe,
            control,
            progress,
            completed: RefCell::new(BTreeSet::new()),
        }
    }
    pub fn component(
        &self,
        name: &str,
        build: impl FnOnce(&Path, &Path, &PreparationControl) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        self.control.check()?;
        let pin: &Output = self
            .recipe
            .outputs
            .iter()
            .find(|file| file.path == name)
            .ok_or_else(|| anyhow::anyhow!("component is not part of the compiled recipe"))?;
        plan::send(
            self.progress,
            if self.recipe.transform_outputs.is_empty() {
                PreparationPhase::Verifying
            } else {
                PreparationPhase::Converting
            },
            name,
            self.completed.borrow().len() as u64,
            self.recipe.outputs.len() as u64,
        );
        let dest = self.output.join(name);
        let mut parent = self.output.to_owned();
        for part in Path::new(name)
            .parent()
            .into_iter()
            .flat_map(Path::components)
        {
            parent.push(part);
            publication::directory(&parent)?;
        }
        anyhow::ensure!(
            !self.completed.borrow().contains(pin.path),
            "component prepared twice"
        );
        let cached = self.components.join(pin.sha256);
        if plan::verified(&cached, pin.size, pin.sha256, self.control)? {
            link_or_copy(&cached, &dest, self.control)?;
        } else {
            build(self.input, &dest, self.control)?;
            anyhow::ensure!(
                plan::verified(&dest, pin.size, pin.sha256, self.control)?,
                PreparationError::Integrity(name.into())
            );
            std::fs::File::open(&dest)?.sync_all()?;
            self.control.check()?;
            if let Ok(meta) = std::fs::symlink_metadata(&cached) {
                anyhow::ensure!(meta.is_file(), "invalid preparation component cache");
                std::fs::remove_file(&cached)?;
            }
            // Optional cache links add no disk copy. Unsupported hardlinks simply
            // leave this component uncached; the verified staged output is valid.
            let _ = std::fs::hard_link(&dest, &cached);
        }
        self.completed.borrow_mut().insert(pin.path);
        plan::send(
            self.progress,
            if self.recipe.transform_outputs.is_empty() {
                PreparationPhase::Verifying
            } else {
                PreparationPhase::Converting
            },
            name,
            self.completed.borrow().len() as u64,
            self.recipe.outputs.len() as u64,
        );
        Ok(())
    }
    pub(super) fn finish(&self) -> anyhow::Result<()> {
        for pin in self.recipe.outputs {
            if !self.completed.borrow().contains(pin.path) {
                self.component(pin.path, |_, _, _| {
                    anyhow::bail!("required component was not produced")
                })?;
            }
        }
        Ok(())
    }
    #[cfg(any(test, feature = "kokoro", feature = "cosyvoice3", feature = "indextts"))]
    pub fn copy_all(&self) -> anyhow::Result<()> {
        for output in self.recipe.outputs {
            self.copy(output.path)?;
        }
        Ok(())
    }
    #[cfg(any(test, feature = "kokoro", feature = "cosyvoice3", feature = "indextts"))]
    pub fn copy(&self, name: &str) -> anyhow::Result<()> {
        self.component(name, |input, dest, control| {
            link_or_copy(&input.join(name), dest, control)
        })
    }
    #[cfg(any(feature = "kokoro", feature = "cosyvoice3", feature = "indextts"))]
    pub fn write(&self, name: &str, bytes: &[u8]) -> anyhow::Result<()> {
        self.component(name, |_, dest, _| {
            use std::io::Write;
            let mut file = std::fs::File::create_new(dest)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            Ok(())
        })
    }
}
