//! Create one tagged reference; subsequent narration uses its cached clone prompt.
use std::{path::PathBuf, sync::Arc};
use talechime_protocol::Device;
/// Load the fixed VoiceDesign files from the configured Hub cache.
pub async fn reference(
    downloads: &crate::downloads::Downloads,
    device: Device,
    text: String,
    description: String,
    output: PathBuf,
    seed: Option<u64>,
) -> anyhow::Result<()> {
    let files = crate::resources::resolve_manifest(super::resources::MANIFEST, downloads.clone())
        .await?
        .into_iter()
        .map(|(name, path)| (name.into(), path))
        .collect();
    reference_source(files, device, text, description, output, seed).await
}
pub(crate) async fn reference_source(
    source: super::ModelFiles,
    device: Device,
    text: String,
    description: String,
    output: PathBuf,
    seed: Option<u64>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !text.trim().is_empty() && text.chars().count() <= 100,
        "design reference must contain 1..100 characters"
    );
    anyhow::ensure!(
        !description.trim().is_empty() && description.len() <= 4096,
        "provide a voice description up to 4096 bytes"
    );
    let (completed, receiver) = tokio::sync::oneshot::channel::<()>();
    let completed = Arc::new(completed);
    // Receiver drop is observed directly at diffusion and decode boundaries.
    let (result, received) = tokio::sync::oneshot::channel();
    let thread = std::thread::Builder::new()
        .name("omni-voice-design".into())
        .spawn(move || {
            let generated = (|| -> anyhow::Result<()> {
                let pipeline = super::runtime::pipeline(source, device, seed.unwrap_or(42))?;
                let signal = completed.clone();
                pipeline.set_cancellation_probe(Some(
                    talechime_omnivoice::stage0_model::CancellationProbe(Arc::new(move || {
                        signal.is_closed()
                    })),
                ));
                let request =
                    talechime_omnivoice::contracts::GenerationRequest::new_text_only(text)
                        .with_instruct(description);
                let mut generated = pipeline.generate(&request)?;
                anyhow::ensure!(
                    !completed.is_closed() && generated.len() == 1,
                    "voice design cancelled or returned an invalid batch"
                );
                let audio = generated.remove(0);
                let pcm = talechime_core::backend::Pcm {
                    samples: audio.samples,
                    sample_rate: audio.sample_rate,
                    channels: 1,
                };
                pcm.duration_ms()?;
                let mut wav = hound::WavWriter::create(
                    output,
                    hound::WavSpec {
                        channels: 1,
                        sample_rate: pcm.sample_rate,
                        bits_per_sample: 32,
                        sample_format: hound::SampleFormat::Float,
                    },
                )?;
                for sample in pcm.samples {
                    wav.write_sample(sample)?;
                }
                wav.finalize()?;
                Ok(())
            })();
            let _ = result.send(generated);
        })?;
    let _owner = crate::design::thread::DesignThread::new(receiver, thread);
    received
        .await
        .map_err(|_| anyhow::anyhow!("Omni voice design thread exited"))?
}
