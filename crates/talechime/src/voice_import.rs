//! Reference voice import without the CLI or protocol session.
//!
//! Hosts embedding the library can install cloning references before preparing
//! an engine; validation matches the CLI `voices import` command exactly.

use crate::ModelOptions;
use std::path::Path;

/// Library-level voice import failure; the message is safe to display.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct VoiceImportError {
    message: String,
}

fn fail(error: anyhow::Error) -> VoiceImportError {
    VoiceImportError {
        message: error.to_string(),
    }
}

fn unsupported(backend: &str) -> VoiceImportError {
    VoiceImportError {
        message: format!("{backend} does not support library voice import; use the CLI"),
    }
}

#[cfg(any(
    not(feature = "qwen"),
    not(feature = "voxcpm"),
    not(feature = "omnivoice"),
    not(feature = "indextts")
))]
fn not_compiled(backend: &str) -> VoiceImportError {
    VoiceImportError {
        message: format!("{backend} adapter is not compiled into this build"),
    }
}

#[cfg(any(
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "indextts"
))]
fn custom_id(id: &str) -> String {
    format!("custom:{}", id.trim_start_matches("custom:"))
}

/// Import a WAV reference voice for a cloning-capable backend without loading weights.
///
/// `options` carries the same backend/model/resources selection passed to
/// `Engine::prepare`. `transcript` must match the reference audio exactly; the
/// id is stored under the `custom:` namespace and may not collide with built-ins.
pub fn import_reference_voice(
    options: &ModelOptions,
    id: &str,
    name: &str,
    wav: &Path,
    transcript: &str,
) -> Result<(), VoiceImportError> {
    // Named parameters stay meaningful in builds without any cloning adapter.
    let _ = (id, name, wav, transcript);
    match options.backend.as_str() {
        "qwen" => {
            #[cfg(feature = "qwen")]
            {
                let model = tts_backends::qwen::models::Model::parse(options.model.as_deref())
                    .map_err(fail)?;
                if !model.is_base() {
                    return Err(VoiceImportError {
                        message: "qwen reference voices require model 0.6b-base or 1.7b-base"
                            .into(),
                    });
                }
                tts_backends::qwen::validate_wav(wav).map_err(fail)?;
                let store = tts_backends::qwen::voice_store(
                    &model.directory(options.resources.as_path()),
                    model,
                )
                .map_err(fail)?;
                store
                    .import(&custom_id(id), name, wav, transcript, None)
                    .map_err(fail)?;
                Ok(())
            }
            #[cfg(not(feature = "qwen"))]
            Err(not_compiled("qwen"))
        }
        "voxcpm" => {
            #[cfg(feature = "voxcpm")]
            {
                let model = tts_backends::voxcpm::models::Model::parse(options.model.as_deref())
                    .map_err(fail)?;
                tts_backends::reference::load(wav, 16000).map_err(fail)?;
                let store = tts_backends::voxcpm::voice_store_for(
                    &model.directory(options.resources.as_path()),
                    model,
                )
                .map_err(fail)?;
                store
                    .import(&custom_id(id), name, wav, transcript, None)
                    .map_err(fail)?;
                Ok(())
            }
            #[cfg(not(feature = "voxcpm"))]
            Err(not_compiled("voxcpm"))
        }
        "omnivoice" => {
            #[cfg(feature = "omnivoice")]
            {
                tts_backends::reference::load(wav, 24000).map_err(fail)?;
                let store = tts_backends::omnivoice::voice_store(
                    &tts_backends::omnivoice::directory(options.resources.as_path()),
                )
                .map_err(fail)?;
                store
                    .import(&custom_id(id), name, wav, transcript, None)
                    .map_err(fail)?;
                Ok(())
            }
            #[cfg(not(feature = "omnivoice"))]
            Err(not_compiled("omnivoice"))
        }
        "indextts" => {
            #[cfg(feature = "indextts")]
            {
                if options.model.as_deref().is_some_and(|model| model != "2.5") {
                    return Err(VoiceImportError {
                        message: "IndexTTS model must be 2.5".into(),
                    });
                }
                let reference = tts_backends::reference::load(wav, 22_050).map_err(fail)?;
                if reference.len() > 15 * 22_050 {
                    return Err(VoiceImportError {
                        message: "IndexTTS reference must be at most 15 seconds".into(),
                    });
                }
                let store = tts_backends::indextts::voice_store(
                    &tts_backends::indextts::directory(options.resources.as_path()),
                )
                .map_err(fail)?;
                store
                    .import(&custom_id(id), name, wav, transcript, None)
                    .map_err(fail)?;
                Ok(())
            }
            #[cfg(not(feature = "indextts"))]
            Err(not_compiled("indextts"))
        }
        backend => Err(unsupported(backend)),
    }
}

/// List installed voice ids and display names without loading model weights.
pub fn list_voices(options: &ModelOptions) -> Result<Vec<(String, String)>, VoiceImportError> {
    let registry = tts_backends::Registry::new(Some(options.resources.clone())).map_err(fail)?;
    let caps = registry
        .capabilities_for(&options.backend, options.model.as_deref())
        .map_err(fail)?;
    Ok(caps
        .voices
        .into_iter()
        .map(|id| {
            (
                id.clone(),
                caps.voice_names
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| id.clone()),
            )
        })
        .collect())
}
