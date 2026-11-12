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
        self.root.join("data")
    }
    pub fn cache(&self) -> PathBuf {
        self.root.join("cache")
    }
    pub fn voices(&self) -> PathBuf {
        Self::voice_library_at(self.resources())
    }
    /// Resolve the current asset namespace under an explicitly selected app/config root.
    pub fn voice_library_at(root: impl AsRef<Path>) -> PathBuf {
        root.as_ref().join("voices-v2")
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
        assert_eq!(paths.resources(), home.path().join(".talechime/data"));
        assert_eq!(paths.cache(), home.path().join(".talechime/cache"));
        assert_eq!(
            paths.checkpoints(),
            home.path().join(".talechime/checkpoints")
        );
        ConfigStore::new(paths.config()).load().unwrap();
        let source = talechime_protocol::SourceId {
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
        assert_eq!(
            paths.voices(),
            home.path().join(".talechime/data/voices-v2")
        );
        assert!(!paths.root().exists());
        assert_eq!(
            std::fs::read_to_string(old_config).unwrap(),
            "old, invalid configuration"
        );
    }
}
