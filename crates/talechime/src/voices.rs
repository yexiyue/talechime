//! Model-independent voice asset commands over the shared library.
//! These never read or write the backend/model selection in the config file.
use clap::Subcommand;
use std::path::PathBuf;
use talechime::{
    DEFAULT_TRIAL_TEXT, DesignOptions, ExportTarget, ImportAction, ImportConflict, ImportDecision,
    TransferControl, TransferLimits, VoiceStudio,
};
use talechime_protocol::{Device, Event};

#[derive(Subcommand)]
pub enum VoiceCommand {
    /// List library voices with their source.
    List,
    /// List design engines: device support, readiness and download cost.
    Engines,
    /// Design a voice from a text description into the library.
    Design {
        /// Display name; defaults to the first 24 characters of the description.
        #[arg(long)]
        name: Option<String>,
        /// Describe the voice to create.
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
        /// auto, cpu, metal or cuda; constrained by the chosen design engine.
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
    /// Export selected library IDs into one package.
    Export {
        #[arg(required = true, num_args = 1..)]
        ids: Vec<String>,
        #[arg(long)]
        output: PathBuf,
        /// Explicitly replace an existing output file.
        #[arg(long)]
        replace: bool,
    },
    /// Inspect and optionally import one or more current-format packages.
    ImportPackages {
        #[arg(required = true, num_args = 1..)]
        paths: Vec<PathBuf>,
        #[arg(long)]
        inspect: bool,
        /// Import conflicting identities with new IDs; existing voices are never overwritten.
        #[arg(long)]
        as_new: bool,
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
                let source = match asset.source() {
                    talechime::VoiceSource::Designed(engine) => format!("designed by {engine}"),
                    talechime::VoiceSource::Imported => "imported".to_owned(),
                };
                println!("{}\t{}\t{}", asset.id(), asset.name(), source);
            }
        }
        VoiceCommand::Engines => {
            let engines = studio.engines().await?;
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
                name,
                text: Some(text),
                device: device.unwrap_or(Device::Auto),
                seed,
            };
            let asset = studio.design_with(options, |event| report(&event)).await?;
            println!(
                "Saved voice {}; read with --voice {} (first use materializes it into the model)",
                asset.id(),
                asset.id()
            );
        }
        VoiceCommand::Import {
            wav,
            name,
            text,
            id,
        } => {
            let asset = studio.import_voice(id.as_deref(), &name, &wav, &text)?;
            println!("Imported {}", asset.id());
        }
        VoiceCommand::Rename { id, name } => {
            let asset = studio.rename(&id, &name)?;
            println!("Renamed {} to {}", asset.id(), asset.name());
        }
        VoiceCommand::Remove { id } => {
            studio.remove(&id)?;
            println!("Removed {id}");
        }
        VoiceCommand::Export {
            ids,
            output,
            replace,
        } => {
            let ids = ids
                .into_iter()
                .map(|id| id.parse())
                .collect::<Result<Vec<_>, _>>()?;
            let report = studio.library().export_bundle(
                &ids,
                &output,
                if replace {
                    ExportTarget::ReplaceConfirmed
                } else {
                    ExportTarget::CreateNew
                },
                TransferLimits::default(),
                &TransferControl::default(),
            )?;
            println!("Exported {} voices ({} bytes)", report.voices, report.bytes);
        }
        VoiceCommand::ImportPackages {
            paths,
            inspect,
            as_new,
        } => {
            let preview = studio.library().inspect_import(
                &paths,
                TransferLimits::default(),
                &TransferControl::default(),
            )?;
            for entry in preview.entries() {
                println!(
                    "{}\t{}\t{:?}",
                    entry.asset.id(),
                    entry.asset.name(),
                    entry.conflicts
                );
            }
            if inspect {
                return Ok(());
            }
            let decisions = preview
                .entries()
                .iter()
                .map(|entry| ImportDecision {
                    entry: entry.index,
                    action: if as_new && entry.conflicts.contains(&ImportConflict::IdentityConflict)
                    {
                        ImportAction::ImportAsNew
                    } else {
                        ImportAction::Import
                    },
                })
                .collect();
            let plan = studio.library().plan_import(preview, decisions)?;
            let report = studio
                .library()
                .import_selected(plan, &TransferControl::default());
            let failed = report.items.iter().any(|item| item.error.is_some());
            for item in report.items {
                println!(
                    "{}\t{:?}\t{}",
                    item.entry,
                    item.status,
                    item.local_id.unwrap_or_default()
                );
            }
            anyhow::ensure!(!failed, "one or more voices failed to import");
        }
    }
    Ok(())
}

fn devices(list: &[Device]) -> String {
    if list.is_empty() {
        return "unavailable".into();
    }
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
