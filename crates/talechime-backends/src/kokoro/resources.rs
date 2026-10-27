//! Integrity-checked local conversion; never overwrites or migrates unsupported data.
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Component, Path},
    thread::JoinHandle,
};
use talechime_protocol::Event;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub const REVISION: &str = "01e7505-native-v1";
#[derive(Deserialize)]
pub(super) struct Voice {
    pub path: String,
    pub source_sha256: String,
}
#[derive(Deserialize)]
pub(super) struct File {
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Deserialize)]
pub(super) struct Manifest {
    format: String,
    model_revision: String,
    #[serde(default)]
    pub contextual: bool,
    pub voices: BTreeMap<String, Voice>,
    files: BTreeMap<String, File>,
}
#[derive(Deserialize)]
struct OfficialVoice {
    id: String,
    sha256: String,
}

pub(super) fn read(directory: &Path) -> anyhow::Result<Manifest> {
    let manifest: Manifest =
        serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
    anyhow::ensure!(
        manifest.format == "kokoro-native-v1"
            && manifest.model_revision == "01e7505bd6a7a2ac4975463114c3a7650a9f7218",
        "unsupported Kokoro conversion format or model revision"
    );
    let official: Vec<OfficialVoice> = serde_json::from_str(talechime_kokoro::BUILTIN_VOICES_JSON)?;
    anyhow::ensure!(
        !manifest.voices.is_empty(),
        "Kokoro bundle contains no voices"
    );
    for (id, voice) in &manifest.voices {
        anyhow::ensure!(
            id.starts_with("zf_") || id.starts_with("zm_"),
            "Kokoro frontend currently supports Mandarin voices"
        );
        anyhow::ensure!(
            official
                .iter()
                .any(|v| v.id == *id && v.sha256 == voice.source_sha256),
            "unrecognized official Kokoro voice identity"
        );
        anyhow::ensure!(
            voice.path == format!("voices/{id}.safetensors")
                && manifest.files.contains_key(&voice.path),
            "invalid Kokoro voice resource path"
        );
    }
    for required in [
        "model.safetensors",
        "config.json",
        "frontend/frontend.json",
        "frontend/jieba.dict",
        "official-export.json",
        "KOKORO-LICENSE",
    ] {
        anyhow::ensure!(
            manifest.files.contains_key(required),
            "Kokoro manifest omits {required}"
        );
    }
    if manifest.contextual {
        for required in [
            "context/context.json",
            "context/g2pw.onnx",
            "context/vocab.txt",
            "context/G2PW-LICENSE",
            "context/BERT-SOURCE.txt",
        ] {
            anyhow::ensure!(
                manifest.files.contains_key(required),
                "Kokoro manifest omits {required}"
            );
        }
    }
    for path in manifest.files.keys() {
        anyhow::ensure!(
            !path.is_empty()
                && Path::new(path)
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
            "unsafe Kokoro resource path"
        );
    }
    Ok(manifest)
}

pub(super) fn read_for(directory: &Path, model: &str) -> anyhow::Result<Manifest> {
    let manifest = read(directory)?;
    anyhow::ensure!(
        manifest.contextual == (model == "1.1-zh-g2pw"),
        "Kokoro bundle pronunciation mode does not match selected model"
    );
    Ok(manifest)
}

/// Reuse production format/voice checks before the offline performance audit.
#[cfg(feature = "performance-tools")]
pub(crate) fn performance_resources(
    directory: &Path,
    model: &str,
) -> anyhow::Result<Vec<crate::resources::Resource>> {
    Ok(read_for(directory, model)?
        .files
        .into_iter()
        .map(|(path, file)| crate::resources::Resource {
            path,
            size: file.bytes,
            sha256: file.sha256,
            url: String::new(),
        })
        .collect())
}

pub(super) fn verify_cancellable(
    directory: &Path,
    cancelled: &dyn Fn() -> bool,
) -> anyhow::Result<Manifest> {
    let manifest = read(directory)?;
    verify_files(directory, &manifest, cancelled)?;
    Ok(manifest)
}

fn verify_files(
    directory: &Path,
    manifest: &Manifest,
    cancelled: &dyn Fn() -> bool,
) -> anyhow::Result<()> {
    for (path, file) in &manifest.files {
        let resource = crate::resources::Resource {
            path: path.clone(),
            url: String::new(),
            sha256: file.sha256.clone(),
            size: file.bytes,
        };
        crate::resources::verify_cancellable(
            &directory.join(&resource.path),
            &resource,
            cancelled,
        )?;
    }
    Ok(())
}

struct VerificationTask {
    shutdown: CancellationToken,
    thread: Option<JoinHandle<()>>,
    result: oneshot::Receiver<anyhow::Result<()>>,
}

impl VerificationTask {
    fn spawn(
        verify: impl FnOnce(CancellationToken) -> anyhow::Result<()> + Send + 'static,
    ) -> std::io::Result<Self> {
        let shutdown = CancellationToken::new();
        let stop = shutdown.clone();
        let (answer, result) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name("kokoro-verify".into())
            .spawn(move || {
                let _ = answer.send(verify(stop));
            })?;
        Ok(Self {
            shutdown,
            thread: Some(thread),
            result,
        })
    }
}

impl Drop for VerificationTask {
    fn drop(&mut self) {
        self.shutdown.cancel();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub async fn prepare(directory: &Path, progress: mpsc::Sender<Event>) -> anyhow::Result<()> {
    prepare_for(directory, super::MODEL, progress).await
}
pub async fn prepare_for(
    directory: &Path,
    model: &str,
    progress: mpsc::Sender<Event>,
) -> anyhow::Result<()> {
    let _ = progress
        .send(Event::ResourceState {
            stage: "校验".into(),
            resource: format!("kokoro/{model}"),
        })
        .await;
    let verification_directory = directory.to_owned();
    let model = model.to_owned();
    let verified = async {
        let mut task = VerificationTask::spawn(move |stop| {
            let manifest = read_for(&verification_directory, &model)?;
            verify_files(&verification_directory, &manifest, &|| stop.is_cancelled())
        })?;
        (&mut task.result).await?
    }
    .await;
    verified.map_err(|error| anyhow::anyhow!(
        "Kokoro local conversion verification failed: {error}; prepare the pinned official runtime bundle in {} (see docs/kokoro.md)", directory.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[tokio::test]
    async fn cancelled_preparation_joins_verification_before_returning() {
        let finished = Arc::new(AtomicBool::new(false));
        let worker_finished = finished.clone();
        let (started, running) = oneshot::channel();
        let preparation = tokio::spawn(async move {
            let mut task = VerificationTask::spawn(move |stop| {
                let _ = started.send(());
                while !stop.is_cancelled() {
                    std::thread::park_timeout(std::time::Duration::from_millis(1));
                }
                worker_finished.store(true, Ordering::Release);
                Ok(())
            })
            .unwrap();
            (&mut task.result).await.unwrap().unwrap();
        });
        running.await.unwrap();
        preparation.abort();
        assert!(preparation.await.unwrap_err().is_cancelled());
        assert!(finished.load(Ordering::Acquire));
    }
}
