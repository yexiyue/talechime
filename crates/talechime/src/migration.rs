//! Executable-only legacy migration; errors leave sources and current data intact.
use crate::resources::Resources;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tts_core::migration::LegacyMigration;
use tts_protocol::{Config, Event};

pub fn startup(migration: &LegacyMigration, resources: &Resources) {
    #[cfg(feature = "moss")]
    let coded_revision = Some(tts_backends::moss::resources::REVISION);
    #[cfg(not(feature = "moss"))]
    let coded_revision = None;
    warn(migration.voices(coded_revision));
    let defaults = resources.default_config().or_else(|_| {
        let caps = resources
            .catalog()?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no backend compiled"))?;
        Ok::<_, anyhow::Error>(Config {
            backend: caps.backend,
            model: caps.model,
            voice: caps.default_voice,
            ..Default::default()
        })
    });
    match defaults {
        Ok(defaults) => match resources.catalog() {
            Ok(catalog) => warn(migration.config(&defaults, &catalog)),
            Err(error) => eprintln!("legacy configuration migration unavailable: {error}"),
        },
        Err(error) => eprintln!("legacy configuration migration unavailable: {error}"),
    }
    warn(migration.checkpoints());
}

fn warn(result: std::io::Result<()>) {
    if let Err(error) = result {
        eprintln!("legacy migration deferred; original and existing new data retained: {error}");
    }
}

#[derive(Clone)]
pub struct CopyWork {
    cancelled: Arc<AtomicBool>,
    closed: Arc<AtomicBool>,
    changed: Arc<tokio::sync::Notify>,
}
impl Default for CopyWork {
    fn default() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            closed: Arc::new(AtomicBool::new(true)),
            changed: Arc::new(tokio::sync::Notify::new()),
        }
    }
}
impl CopyWork {
    pub async fn cancel_and_wait(&self) {
        self.cancelled.store(true, Ordering::Release);
        loop {
            let changed = self.changed.notified();
            if self.closed.load(Ordering::Acquire) {
                return;
            }
            changed.await;
        }
    }
}

pub async fn resources(
    migration: Option<LegacyMigration>,
    progress: Option<&tokio::sync::mpsc::Sender<Event>>,
    work: CopyWork,
) {
    let Some(migration) = migration else { return };
    if let Some(progress) = progress {
        let _ = progress
            .send(Event::ResourceState {
                stage: "迁移旧模型与音色资源".into(),
                resource: "legacy resources".into(),
            })
            .await;
    }
    struct Cancellation(CopyWork);
    impl Drop for Cancellation {
        fn drop(&mut self) {
            self.0.cancelled.store(true, Ordering::Release);
        }
    }
    struct Closed(CopyWork);
    impl Drop for Closed {
        fn drop(&mut self) {
            self.0.closed.store(true, Ordering::Release);
            self.0.changed.notify_one();
        }
    }
    let _cancellation = Cancellation(work.clone());
    work.closed.store(false, Ordering::Release);
    let closed = Closed(work.clone());
    match tokio::task::spawn_blocking(move || {
        let _closed = closed;
        #[cfg(feature = "moss")]
        let coded_revision = Some(tts_backends::moss::resources::REVISION);
        #[cfg(not(feature = "moss"))]
        let coded_revision = None;
        warn(migration.prepare_voices(coded_revision, &work.cancelled));
        migration.resources_with_cancellation(&work.cancelled)
    })
    .await
    {
        Ok(result) => warn(result),
        Err(error) => eprintln!("legacy resources migration deferred: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn cancelled_copy_releases_its_owner_before_control_returns() {
        let home = tempfile::tempdir().unwrap();
        let old = home.path().join(".novel-tts/models");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::File::create(old.join("fake.bin"))
            .unwrap()
            .set_len(128 * 1024 * 1024)
            .unwrap();
        let migration = LegacyMigration::from_home(home.path());
        let work = CopyWork::default();
        let pending = work.clone();
        let task = tokio::spawn(async move {
            resources(Some(migration), None, pending).await;
        });
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;
        tokio::time::timeout(std::time::Duration::from_secs(3), work.cancel_and_wait())
            .await
            .unwrap();
        assert!(work.closed.load(Ordering::Acquire));
        let lock = home.path().join(".talechime/.legacy-resources.lock");
        if let Ok(file) = std::fs::File::options().read(true).write(true).open(lock) {
            file.try_lock().unwrap();
        }
        assert!(old.join("fake.bin").exists());
    }
}
