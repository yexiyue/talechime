//! Integrity-checked local conversion; never overwrites or migrates unsupported data.
use serde::Deserialize;
use std::collections::BTreeMap;

pub const REVISION: &str = "01e7505-runtime-f32-v3";
#[derive(Deserialize)]
pub(super) struct Voice {
    pub path: String,
    #[cfg(test)]
    pub source_sha256: String,
}
#[cfg(test)]
#[derive(Deserialize)]
pub(super) struct File {
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Deserialize)]
pub(super) struct Manifest {
    pub voices: BTreeMap<String, Voice>,
    #[cfg(test)]
    files: BTreeMap<String, File>,
}

pub(crate) const HUB_MANIFEST: &str = include_str!("distribution-context-inputs.json");

pub(super) fn compiled_manifest() -> anyhow::Result<Manifest> {
    Ok(serde_json::from_slice(include_bytes!(
        "native-context-manifest.json"
    ))?)
}
#[cfg(test)]
#[derive(Deserialize)]
struct OfficialVoice {
    id: String,
    sha256: String,
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn fixed_hub_resources_cover_the_compiled_model_and_voices() {
        let manifest = compiled_manifest().unwrap();
        let resources: Vec<crate::resources::Resource> =
            serde_json::from_str(HUB_MANIFEST).unwrap();
        let files: BTreeMap<_, _> = resources
            .into_iter()
            .map(|item| (item.path.clone(), item))
            .collect();
        for (name, expected) in &manifest.files {
            let actual = files.get(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(actual.size, expected.bytes, "{name}");
            assert_eq!(actual.sha256, expected.sha256, "{name}");
        }
        let official: Vec<OfficialVoice> =
            serde_json::from_str(talechime_kokoro::BUILTIN_VOICES_JSON).unwrap();
        assert!(!manifest.voices.is_empty());
        for (id, voice) in manifest.voices {
            assert!(
                official
                    .iter()
                    .any(|item| item.id == id && item.sha256 == voice.source_sha256)
            );
            assert!(files.contains_key(&voice.path));
        }
    }
}
