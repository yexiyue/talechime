//! Integrity-checked local conversion; never overwrites or migrates unsupported data.
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Component, Path},
};
use talechime_protocol::Event;
use tokio::sync::mpsc;

pub const REVISION: &str = "01e7505-runtime-f32-v3";
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
    use std::io::Read;
    let path = directory.join("manifest.json");
    anyhow::ensure!(
        std::fs::symlink_metadata(&path)?.is_file(),
        "invalid Kokoro manifest file"
    );
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(512 * 1024)
        .read_to_end(&mut bytes)?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        bytes == super::native::manifest(manifest.contextual),
        "Kokoro manifest differs from the compiled native recipe"
    );
    let marker = super::native::publication_marker(manifest.contextual)?;
    let path = directory.join("preparation.json");
    let meta = std::fs::symlink_metadata(&path)?;
    anyhow::ensure!(
        meta.is_file() && meta.len() == marker.len() as u64,
        "Kokoro resources were not published by the native preparation recipe"
    );
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(marker.len() as u64 + 1)
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes == marker, "Kokoro publication recipe differs");
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
) -> anyhow::Result<Vec<crate::resources::LocalResource>> {
    Ok(read_for(directory, model)?
        .files
        .into_iter()
        .map(|(path, file)| crate::resources::LocalResource {
            path,
            size: file.bytes,
            sha256: file.sha256,
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

/// Offline readiness from the on-disk conversion manifest.
pub(crate) fn conversion_status(directory: &Path) -> crate::resource_state::ResourceStatus {
    match read(directory) {
        Ok(manifest) => {
            let entries: Vec<(String, u64)> = manifest
                .files
                .iter()
                .map(|(path, file)| (path.clone(), file.bytes))
                .collect();
            crate::resource_state::status_of(&entries, directory)
        }
        Err(_) => crate::resource_state::ResourceStatus::Missing { missing_bytes: 0 },
    }
}

fn verify_files(
    directory: &Path,
    manifest: &Manifest,
    cancelled: &dyn Fn() -> bool,
) -> anyhow::Result<()> {
    for (path, file) in &manifest.files {
        let resource = crate::resources::LocalResource {
            path: path.clone(),
            sha256: file.sha256.clone(),
            size: file.bytes,
        };
        crate::resources::verify_local_cancellable(directory, &resource, cancelled)?;
    }
    Ok(())
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
    let recipe = super::native_recipe(model)
        .ok_or_else(|| anyhow::anyhow!("unsupported Kokoro native model"))?;
    recipe.verify_existing(directory.to_owned(), crate::PreparationControl::default()).await
        .map_err(|error| anyhow::anyhow!("Kokoro native resource verification failed: {error}; prepare the native resource recipe explicitly"))
}
