//! Model-independent voice asset commands over the shared library.
//! These never read or write the backend/model selection in the config file.
use clap::Subcommand;
use std::path::PathBuf;
use talechime::{DEFAULT_TRIAL_TEXT, DesignOptions, VoiceStudio};
use talechime_protocol::{Device, Event};

#[derive(Subcommand)]
pub enum VoiceCommand {
    /// List library voices with their source.
    List,
    /// List design engines: device support, readiness and download cost.
    Engines,
    /// Design a voice from a text description into the library.
    Design {
        #[arg(long)]
        name: String,
        #[arg(long)]
        description: String,
        /// Library ID; auto-generated when omitted.
        #[arg(long)]
        id: Option<String>,
        /// Spoken trial text recorded into the reference.
        #[arg(long, default_value = DEFAULT_TRIAL_TEXT)]
        text: String,
        /// Design engine id; see `engines`. Default: automatic policy.
        #[arg(long)]
        engine: Option<String>,
        #[arg(long, value_parser = crate::parse_device)]
        device: Option<Device>,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Import a reference recording with its exact transcript.
    Import {
        wav: PathBuf,
        #[arg(long)]
        name: String,
        #[arg(long)]
        text: String,
        /// Library ID; auto-generated when omitted.
        #[arg(long)]
        id: Option<String>,
    },
    /// Rename a library voice.
    Rename {
        id: String,
        #[arg(long)]
        name: String,
    },
    /// Delete a library voice.
    Remove { id: String },
    /// Export one voice as a single self-contained file.
    Export {
        id: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Import a voice bundle file exported on this or another machine.
    ImportBundle {
        path: PathBuf,
        /// Override the recorded ID (required on collision).
        #[arg(long)]
        id: Option<String>,
    },
}

pub async fn run(command: VoiceCommand, studio: VoiceStudio) -> anyhow::Result<()> {
    match command {
        VoiceCommand::List => {
            let voices = studio.list()?;
            if voices.is_empty() {
                println!("voice library is empty; design or import a voice first");
            }
            for asset in voices {
                let source = match asset.source {
                    talechime::VoiceSource::Designed(engine) => format!("designed by {engine}"),
                    talechime::VoiceSource::Imported => "imported".to_owned(),
                };
                println!("{}\t{}\t{}", asset.id, asset.name, source);
            }
        }
        VoiceCommand::Engines => {
            let engines = studio.engines();
            if engines.is_empty() {
                println!("no design engine is compiled into this build");
            }
            for info in engines {
                println!(
                    "{}\t{}\t{}\t{}\t{}",
                    info.id,
                    info.name,
                    devices(&info.devices),
                    if info.ready {
                        "ready"
                    } else {
                        "needs preparation"
                    },
                    if info.extra_download_bytes == 0 {
                        "no extra download".to_owned()
                    } else {
                        format!("download {} MB", info.extra_download_bytes / 1_048_576)
                    },
                );
            }
        }
        VoiceCommand::Design {
            name,
            description,
            id,
            text,
            engine,
            device,
            seed,
        } => {
            let options = DesignOptions {
                description,
                engine,
                id,
                name: Some(name),
                text: Some(text),
                device: device.unwrap_or(Device::Auto),
                seed,
            };
            let asset = studio.design_with(options, |event| report(&event)).await?;
            eprintln!(
                "Saved voice {}; read with --voice {} (first use materializes it into the model)",
                asset.id, asset.id
            );
        }
        VoiceCommand::Import {
            wav,
            name,
            text,
            id,
        } => {
            let asset = studio.import_voice(id.as_deref(), &name, &wav, &text)?;
            println!("Imported {}", asset.id);
        }
        VoiceCommand::Rename { id, name } => {
            let asset = studio.rename(&id, &name)?;
            println!("Renamed {} to {}", asset.id, asset.name);
        }
        VoiceCommand::Remove { id } => {
            studio.remove(&id)?;
            println!("Removed {id}");
        }
        VoiceCommand::Export { id, output } => {
            let path = output.unwrap_or_else(|| format!("{id}.voice").into());
            studio.export(&id, &path)?;
            println!("Exported {} to {}", id, path.display());
        }
        VoiceCommand::ImportBundle { path, id } => {
            let asset = studio.import_bundle(&path, id.as_deref())?;
            println!("Imported {}", asset.id);
        }
    }
    Ok(())
}

fn devices(list: &[Device]) -> String {
    list.iter()
        .map(|device| match device {
            Device::Auto => "auto",
            Device::Cpu => "cpu",
            Device::Coreml => "coreml",
            Device::Metal => "metal",
            Device::Cuda => "cuda",
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn report(event: &Event) {
    if let Event::ModelProgress {
        resource,
        downloaded,
        total,
    } = event
    {
        eprintln!("{resource}: {downloaded}/{total}");
    }
}
