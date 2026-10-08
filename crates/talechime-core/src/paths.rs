//! Application-owned locations; resolving paths never creates or migrates files.
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct AppPaths {
    root: PathBuf,
}

impl AppPaths {
    pub fn from_home(home: impl AsRef<Path>) -> Self {
        Self {
            root: home.as_ref().join(".talechime"),
        }
    }

    pub fn user_default() -> std::io::Result<Self> {
        std::env::var_os("HOME")
            .filter(|home| !home.is_empty())
            .map(PathBuf::from)
            .or_else(dirs::home_dir)
            .map(Self::from_home)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "home directory unavailable")
            })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn config(&self) -> PathBuf {
        self.root.join("config.json")
    }
    pub fn checkpoints(&self) -> PathBuf {
        self.root.join("checkpoints")
    }
    pub fn resources(&self) -> PathBuf {
        self.root.join("resources")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{checkpoint::CheckpointStore, config::ConfigStore};

    #[test]
    fn new_layout_does_not_read_or_migrate_old_state() {
        let home = tempfile::tempdir().unwrap();
        let old = home.path().join(".novel");
        std::fs::create_dir(&old).unwrap();
        let old_config = old.join("tts_config.json");
        std::fs::write(&old_config, "old, invalid configuration").unwrap();
        let paths = AppPaths::from_home(home.path());
        assert_eq!(paths.resources(), home.path().join(".talechime/resources"));
        assert_eq!(
            paths.checkpoints(),
            home.path().join(".talechime/checkpoints")
        );
        ConfigStore::new(paths.config()).load().unwrap();
        let source = tts_protocol::SourceId {
            namespace: "cli".into(),
            book: "book.txt".into(),
            chapter: "file".into(),
        };
        assert!(
            CheckpointStore::new(paths.checkpoints())
                .load(&source, "text")
                .unwrap()
                .is_none()
        );
        assert!(!paths.root().exists());
        assert_eq!(
            std::fs::read_to_string(old_config).unwrap(),
            "old, invalid configuration"
        );
    }
}
