//! VoiceDesign creates one reference recording; Base is used for subsequent narration.
use super::{MAX_FRAMES, runtime};
use std::path::PathBuf;
use talechime_protocol::Device;
use talechime_qwen3_tts::{AudioBuffer, Language, SynthesisOptions};

/// Model-specific VoiceDesign conditions; the shared designer defaults to Auto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    pub language: Language,
    pub seed: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            language: Language::Auto,
            seed: None,
        }
    }
}

/// Load the fixed VoiceDesign files from the configured Hub cache.
pub async fn reference(
    downloads: &crate::downloads::Downloads,
    device: Device,
    text: String,
    description: String,
    output: PathBuf,
    options: Options,
) -> anyhow::Result<()> {
    let paths = crate::resources::resolve_manifest(
        super::models::Model::Design17.manifest(),
        downloads.clone(),
    )
    .await?;
    let files = super::models::Model::resolved_files(paths)?;
    reference_source(files, device, text, description, output, options).await
}
pub(crate) async fn reference_source(
    source: talechime_qwen3_tts::ModelFiles,
    device: Device,
    text: String,
    description: String,
    output: PathBuf,
    options: Options,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !description.trim().is_empty() && description.len() <= 4096,
        "provide a voice description up to 4096 bytes"
    );
    anyhow::ensure!(
        !text.trim().is_empty() && text.chars().count() <= 100,
        "design reference text must contain 1..100 characters"
    );
    anyhow::ensure!(
        super::models::Model::detect_file(&source.config)? == super::models::Model::Design17,
        "voice design requires 1.7b-voicedesign weights"
    );
    let (completed, receiver) = tokio::sync::oneshot::channel();
    let thread = std::thread::Builder::new()
        .name("qwen-voice-design".into())
        .spawn(move || {
            let result = (|| -> anyhow::Result<()> {
                let model = runtime::load(&source, device)?;
                let mut stream = model.synthesize_voice_design_streaming(
                    &text,
                    &description,
                    options.language,
                    SynthesisOptions {
                        max_length: MAX_FRAMES,
                        chunk_frames: 20,
                        seed: Some(options.seed.unwrap_or(42)),
                        ..Default::default()
                    },
                )?;
                let mut samples = Vec::new();
                while let Some(chunk) = stream.next_chunk_with_cancel(|| completed.is_closed())? {
                    samples.extend(chunk.samples);
                }
                anyhow::ensure!(!completed.is_closed(), "voice design cancelled");
                super::runtime::validate_completion(stream.is_done(), !samples.is_empty())?;
                let audio = AudioBuffer::new(samples, 24000);
                super::validate_reference(&audio)?;
                audio.save(output)?;
                Ok(())
            })();
            // GPU weights are released before the caller can load Base.
            let _ = completed.send(result);
        })?;
    let mut owner = crate::design::thread::DesignThread::new(receiver, thread);
    (&mut owner.receiver)
        .await
        .map_err(|_| anyhow::anyhow!("voice design thread exited"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn design_defaults_to_auto_and_accepts_explicit_language_and_seed() {
        assert_eq!(
            Options::default(),
            Options {
                language: Language::Auto,
                seed: None
            }
        );
        let options = Options {
            language: Language::Japanese,
            seed: Some(7),
        };
        assert_eq!(
            (options.language, options.seed),
            (Language::Japanese, Some(7))
        );
    }
}
