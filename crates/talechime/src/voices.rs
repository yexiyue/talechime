use clap::Subcommand;
use std::path::PathBuf;
use talechime_backends::Registry;
#[derive(Subcommand)]
pub enum VoiceCommand {
    /// List voices of the selected backend without loading a model.
    List,
    /// Encode a WAV reference once. ID accepts ASCII letters, digits, '-' and '_'.
    Import {
        id: String,
        #[arg(long)]
        name: String,
        /// Exact transcript of the reference WAV (required for cloned voices).
        #[arg(long)]
        text: Option<String>,
        wav: PathBuf,
    },
    /// Delete an imported voice; built-in voices cannot be deleted.
    Remove { id: String },
    /// Create a reusable reference (Qwen/Vox description; Omni supported voice tags).
    Design {
        id: String,
        #[arg(long)]
        name: String,
        #[arg(long)]
        description: String,
        #[arg(
            long,
            default_value = "你好，欢迎收听今天的故事。山风吹过松林，星光照亮归途。"
        )]
        text: String,
    },
}
pub async fn run(
    command: VoiceCommand,
    resources: Registry,
    config: &talechime_protocol::Config,
) -> anyhow::Result<()> {
    let backend = config.backend.as_str();
    let selected = resources.resolve(backend, config.model.as_deref())?;
    if matches!(command, VoiceCommand::List) {
        let caps = resources.capabilities_for(backend, config.model.as_deref())?;
        for id in caps.voices {
            println!(
                "{}\t{}",
                id,
                caps.voice_names
                    .get(&id)
                    .map_or(id.as_str(), String::as_str)
            );
        }
        return Ok(());
    }
    #[cfg(any(
        feature = "cosyvoice3",
        feature = "indextts",
        feature = "voxcpm",
        feature = "omnivoice",
        feature = "moss"
    ))]
    if let Some((store, rate)) = shared_store(&resources, config)? {
        match command {
            VoiceCommand::List => unreachable!(),
            VoiceCommand::Remove { id } => {
                store.remove(&format!("custom:{}", id.trim_start_matches("custom:")))?
            }
            VoiceCommand::Import {
                id,
                name,
                wav,
                text,
            } => {
                let text = text.ok_or_else(|| {
                    anyhow::anyhow!("{backend} cloning requires --text matching the reference WAV")
                })?;
                let reference = talechime_backends::reference::load(&wav, rate)?;
                anyhow::ensure!(
                    backend != "indextts" || reference.len() <= 15 * rate as usize,
                    "IndexTTS reference must be at most 15 seconds"
                );
                anyhow::ensure!(
                    backend != "moss" || reference.len() <= 240000,
                    "MOSS trial reference must be at most 10 seconds"
                );
                store.import(
                    &format!("custom:{}", id.trim_start_matches("custom:")),
                    &name,
                    &wav,
                    &text,
                    None,
                )?;
            }
            #[cfg(any(feature = "voxcpm", feature = "omnivoice", feature = "moss"))]
            VoiceCommand::Design {
                id,
                name,
                text,
                description,
            } => {
                let device = if config.tts_device == talechime_protocol::Device::Auto {
                    resources
                        .available_devices_for(backend, config.model.as_deref())
                        .into_iter()
                        .find(|d| *d != talechime_protocol::Device::Cpu)
                        .unwrap_or(talechime_protocol::Device::Cpu)
                } else {
                    config.tts_device
                };
                anyhow::ensure!(
                    resources
                        .available_devices_for(backend, config.model.as_deref())
                        .contains(&device),
                    "requested design device is unavailable"
                );
                let output = tempfile::NamedTempFile::new()?;
                let (progress, mut updates) = tokio::sync::mpsc::channel(16);
                let logger = tokio::task::spawn_local(async move {
                    while let Some(event) = updates.recv().await {
                        if let talechime_protocol::Event::ModelProgress {
                            resource,
                            downloaded,
                            total,
                        } = event
                        {
                            eprintln!("{resource}: {downloaded}/{total}");
                        }
                    }
                });
                let result = async {
                    match backend {
                        #[cfg(feature = "moss")]
                        "moss" => {
                            talechime_backends::moss::candle::resources::prepare_design(
                                resources.root(),
                                progress,
                            )
                            .await?;
                            talechime_backends::moss::candle::design::reference(
                                resources.root().into(),
                                device,
                                text.clone(),
                                description.clone(),
                                output.path().into(),
                            )
                            .await?;
                        }
                        #[cfg(feature = "omnivoice")]
                        "omnivoice" => {
                            let directory =
                                talechime_backends::omnivoice::directory(resources.root());
                            talechime_backends::omnivoice::resources::prepare(&directory, progress)
                                .await?;
                            talechime_backends::omnivoice::design::reference(
                                directory,
                                device,
                                text.clone(),
                                description.clone(),
                                output.path().into(),
                            )
                            .await?;
                        }
                        #[cfg(feature = "voxcpm")]
                        "voxcpm" => {
                            let model = talechime_backends::voxcpm::models::Model::parse(
                                config.model.as_deref(),
                            )?;
                            let directory = model.directory(resources.root());
                            talechime_backends::voxcpm::resources::prepare_model(
                                &directory, model, progress,
                            )
                            .await?;
                            talechime_backends::voxcpm::design::reference_model(
                                directory,
                                model,
                                device,
                                text.clone(),
                                description.clone(),
                                output.path().into(),
                            )
                            .await?;
                        }
                        _ => anyhow::bail!("{backend} does not support voice design"),
                    }
                    talechime_backends::reference::load(output.path(), rate)?;
                    store.import(
                        &format!("custom:{}", id.trim_start_matches("custom:")),
                        &name,
                        output.path(),
                        &text,
                        Some(description),
                    )?;
                    Ok::<_, anyhow::Error>(())
                }
                .await;
                logger.abort();
                result?;
            }
            #[cfg(not(any(feature = "voxcpm", feature = "omnivoice", feature = "moss")))]
            VoiceCommand::Design { .. } => {
                anyhow::bail!("{backend} does not support voice design")
            }
        }
        return Ok(());
    }
    #[cfg(feature = "qwen")]
    if backend == "qwen" {
        use talechime_backends::qwen::{models::Model, validate_wav, voice_store};
        if let VoiceCommand::Design {
            id,
            name,
            description,
            text,
        } = command
        {
            let design = Model::Design17;
            let directory = design.directory(resources.root());
            let device = if config.tts_device == talechime_protocol::Device::Auto {
                talechime_backends::qwen::available_devices()
                    .into_iter()
                    .find(|d| *d != talechime_protocol::Device::Cpu)
                    .unwrap_or(talechime_protocol::Device::Cpu)
            } else {
                config.tts_device
            };
            anyhow::ensure!(
                talechime_backends::qwen::available_devices().contains(&device),
                "requested design device is unavailable"
            );
            let (progress, mut updates) = tokio::sync::mpsc::channel(16);
            let logger = tokio::task::spawn_local(async move {
                while let Some(event) = updates.recv().await {
                    if let talechime_protocol::Event::ModelProgress {
                        resource,
                        downloaded,
                        total,
                    } = event
                    {
                        eprintln!("{resource}: {downloaded}/{total}");
                    }
                }
            });
            let output = tempfile::NamedTempFile::new()?;
            let result = async {
                talechime_backends::qwen::resources::prepare_model(&directory, design, progress)
                    .await?;
                talechime_backends::qwen::design::reference(
                    directory,
                    device,
                    text.clone(),
                    description.clone(),
                    output.path().into(),
                )
                .await?;
                let base = Model::Base17;
                voice_store(&base.directory(resources.root()), base)?.import(
                    &format!("custom:{}", id.trim_start_matches("custom:")),
                    &name,
                    output.path(),
                    &text,
                    Some(description),
                )?;
                Ok::<_, anyhow::Error>(())
            }
            .await;
            logger.abort();
            result?;
            eprintln!(
                "Saved voice {id}; select --model 1.7b-base --voice custom:{id} for narration"
            );
            return Ok(());
        }
        let model = Model::parse(config.model.as_deref())?;
        anyhow::ensure!(
            model.is_base(),
            "reference voices require --model 0.6b-base or 1.7b-base"
        );
        let store = voice_store(&model.directory(resources.root()), model)?;
        match command {
            VoiceCommand::List => unreachable!(),
            VoiceCommand::Design { .. } => unreachable!(),
            VoiceCommand::Remove { id } => {
                store.remove(&format!("custom:{}", id.trim_start_matches("custom:")))?
            }
            VoiceCommand::Import {
                id,
                name,
                wav,
                text,
            } => {
                let text = text.ok_or_else(|| {
                    anyhow::anyhow!(
                        "Qwen cloning requires --text with the reference WAV transcript"
                    )
                })?;
                validate_wav(&wav)?;
                store.import(
                    &format!("custom:{}", id.trim_start_matches("custom:")),
                    &name,
                    &wav,
                    &text,
                    None,
                )?;
                eprintln!(
                    "Imported reference; the model-specific prompt is cached on first synthesis"
                );
            }
        }
        return Ok(());
    }
    anyhow::ensure!(
        backend == "moss",
        "{backend} supports preset voices only; voice import/remove requires the MOSS backend"
    );
    #[cfg(feature = "moss-nano-candle")]
    {
        anyhow::ensure!(
            selected.capabilities().model.as_deref() == Some("nano-candle"),
            "MOSS Nano Candle is not selected"
        );
        let directory = talechime_backends::moss::nano::directory(resources.root());
        match command {
            VoiceCommand::List => unreachable!("list handled before model-specific operations"),
            VoiceCommand::Design { .. } => {
                anyhow::bail!("voice design is not supported by {backend}")
            }
            VoiceCommand::Remove { id } => {
                talechime_backends::moss::voices::VoiceStore::new(&directory)
                    .remove(&format!("custom:{}", id.trim_start_matches("custom:")))?
            }
            VoiceCommand::Import { id, name, wav, .. } => {
                let id = format!("custom:{}", id.trim_start_matches("custom:"));
                let (progress, mut rx) = tokio::sync::mpsc::channel(16);
                let logger = tokio::task::spawn_local(async move {
                    while let Some(event) = rx.recv().await {
                        if let talechime_protocol::Event::ModelProgress {
                            resource,
                            downloaded,
                            total,
                        } = event
                        {
                            eprintln!("{resource}: {downloaded}/{total}");
                        }
                    }
                });
                let result = async {
                    talechime_backends::moss::nano::prepare(resources.root(), progress).await?;
                    talechime_backends::moss::nano::import_voice(
                        resources.root().into(),
                        id.clone(),
                        name,
                        wav,
                    )
                    .await
                }
                .await;
                logger.abort();
                result?;
                eprintln!("Imported {id}");
            }
        }
        Ok(())
    }
    #[cfg(not(feature = "moss-nano-candle"))]
    {
        let _ = (command, resources, selected);
        anyhow::bail!("voice import requires the moss Cargo feature")
    }
}
#[cfg(any(
    feature = "cosyvoice3",
    feature = "indextts",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "moss"
))]
fn shared_store(
    resources: &Registry,
    config: &talechime_protocol::Config,
) -> anyhow::Result<Option<(talechime_core::voices::VoiceStore, u32)>> {
    Ok(match config.backend.as_str() {
        #[cfg(feature = "indextts")]
        "indextts" => Some((
            talechime_backends::indextts::voice_store(&talechime_backends::indextts::directory(
                resources.root(),
            ))?,
            22_050,
        )),
        #[cfg(feature = "cosyvoice3")]
        "cosyvoice3" => Some((
            talechime_backends::cosyvoice3::voice_store(
                &talechime_backends::cosyvoice3::directory(resources.root()),
            )?,
            24000,
        )),
        #[cfg(feature = "moss")]
        "moss" => {
            let selected = resources.resolve("moss", config.model.as_deref())?;
            if selected.capabilities().model.as_deref() == Some("realtime-1.7b") {
                let mode = talechime_backends::moss::candle::Mode::Realtime;
                Some((
                    talechime_backends::moss::candle::voice_store(
                        &mode.directory(resources.root()),
                        mode,
                    )?,
                    24000,
                ))
            } else {
                None
            }
        }
        #[cfg(feature = "voxcpm")]
        "voxcpm" => {
            let model = talechime_backends::voxcpm::models::Model::parse(config.model.as_deref())?;
            Some((
                talechime_backends::voxcpm::voice_store_for(
                    &model.directory(resources.root()),
                    model,
                )?,
                16000,
            ))
        }
        #[cfg(feature = "omnivoice")]
        "omnivoice" => Some((
            talechime_backends::omnivoice::voice_store(&talechime_backends::omnivoice::directory(
                resources.root(),
            ))?,
            24000,
        )),
        _ => None,
    })
}
