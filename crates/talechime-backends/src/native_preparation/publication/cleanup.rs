//! Cleanup is called only while the global preparation lease is held.
use crate::native_preparation::{Recipe, plan};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

pub(super) fn remove_abandoned(managed: &Path, recipe: &Recipe) -> anyhow::Result<()> {
    let prefix = format!("staging-{}-", recipe.id);
    let marker = plan::marker(recipe)?;
    let mut files = BTreeSet::from([PathBuf::from("preparation.json")]);
    let mut directories = BTreeSet::new();
    for file in recipe.outputs {
        files.insert(PathBuf::from(file.path));
        let mut parent = Path::new(file.path).parent();
        while let Some(path) = parent.filter(|path| !path.as_os_str().is_empty()) {
            directories.insert(path.to_owned());
            parent = path.parent();
        }
    }
    for entry in std::fs::read_dir(managed)? {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with(&prefix) || !entry.file_type()?.is_dir()
        {
            continue;
        }
        let path = entry.path();
        if plan::marker_matches(&path.join("preparation.json"), &marker)?
            && known_tree(&path, &path, &files, &directories)?
        {
            std::fs::remove_dir_all(path)?;
        }
    }
    Ok(())
}
fn known_tree(
    root: &Path,
    current: &Path,
    files: &BTreeSet<PathBuf>,
    directories: &BTreeSet<PathBuf>,
) -> anyhow::Result<bool> {
    for entry in std::fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let relative = path.strip_prefix(root)?;
        let kind = entry.file_type()?;
        if kind.is_file() {
            if !files.contains(relative) {
                return Ok(false);
            }
        } else if kind.is_dir() {
            if !directories.contains(relative) || !known_tree(root, &path, files, directories)? {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }
    }
    Ok(true)
}
