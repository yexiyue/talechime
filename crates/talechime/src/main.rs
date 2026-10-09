mod app_session;
mod cli;
#[cfg(test)]
#[path = "../tests/library/fixture.rs"]
#[allow(dead_code)] // Binary adapter tests use only normal/long fixture modes.
mod fixture;
mod plan_input;
mod preparation;
mod protocol;
mod resources;
mod runtime;
#[cfg(feature = "asr")]
mod verify;
mod voices;

use clap::Parser;
use std::path::PathBuf;

#[derive(clap::Subcommand)]
enum Commands {
    /// Compare existing PCM16 WAV to UTF-8 source without synthesis or playback.
    #[cfg(feature = "asr")]
    Verify {
        text: PathBuf,
        audio: PathBuf,
        #[arg(long)]
        report: PathBuf,
    },
    Voices {
        #[command(subcommand)]
        command: voices::VoiceCommand,
    },
}

#[derive(clap::ValueEnum, Clone, Copy, Default)]
enum VerifyMode {
    #[default]
    Off,
    Report,
    Gate,
}

#[derive(Parser)]
#[command(
    version,
    subcommand_negates_reqs = true,
    about = "Read a UTF-8 file aloud, or serve JSON Lines on stdin/stdout"
)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,
    /// Optional readback strategy; default off. Gate permits one resynthesis.
    #[arg(long, value_enum, default_value = "off", conflicts_with_all = ["protocol", "plan"])]
    verify: VerifyMode,
    /// Write per-attempt JSONL reports to this explicit path.
    #[arg(long, conflicts_with = "protocol")]
    verification_report: Option<PathBuf>,
    /// Prepare the ASR model group with prepare_model in JSON Lines mode.
    #[arg(long, requires = "protocol")]
    readback_models: bool,
    #[arg(long)]
    backend: Option<String>,
    #[arg(long)]
    voice: Option<String>,
    /// Model ID within the selected backend (see the protocol model catalog).
    #[arg(long)]
    model: Option<String>,
    /// Speaking style for Qwen 1.7B CustomVoice; an empty value clears it.
    #[arg(long)]
    style: Option<String>,
    #[arg(long, value_parser = parse_device)]
    tts_device: Option<tts_protocol::Device>,
    /// File to read. Interactive controls: space=pause/resume, s=stop, q=exit.
    #[arg(required_unless_present = "protocol", conflicts_with = "protocol")]
    file: Option<PathBuf>,
    /// Machine mode; never takes over the terminal.
    #[arg(long)]
    protocol: bool,
    /// Complete JSON plan (source text comes from the file and must match exactly).
    #[arg(long, conflicts_with_all = ["protocol", "voice", "style", "after_chapter"])]
    plan: Option<PathBuf>,
    /// Generate and verify the whole chapter on disk before playback.
    #[arg(long, conflicts_with = "protocol")]
    after_chapter: bool,
    /// Explicitly restart this file from its beginning, ignoring a stored checkpoint.
    #[arg(long, conflicts_with = "protocol")]
    restart: bool,
    /// Override the listening configuration path.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Override the model directory (does not copy or delete existing models).
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Override the independent listening checkpoint directory.
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    #[cfg(feature = "asr")]
    if let Some(Commands::Verify {
        text,
        audio,
        report,
    }) = &args.command
    {
        let resources = resources::Resources::new(args.model_dir.clone())?;
        return talechime::run_local(verify::run(text, audio, report, resources.root())).await;
    }
    let config = match args.config {
        Some(path) => tts_core::config::ConfigStore::new(path),
        None => tts_core::config::ConfigStore::user_default()?,
    };
    let checkpoints = match args.checkpoint_dir {
        Some(path) => tts_core::checkpoint::CheckpointStore::new(path),
        None => tts_core::checkpoint::CheckpointStore::user_default()?,
    };
    let resources = resources::Resources::new(args.model_dir)?;
    let config = if config.path().exists() {
        config
    } else {
        let defaults = match args.backend.as_deref() {
            Some(backend) => {
                let caps = resources.capabilities_for(backend, args.model.as_deref())?;
                tts_protocol::Config {
                    backend: caps.backend,
                    model: caps.model,
                    voice: caps.default_voice,
                    ..Default::default()
                }
            }
            None => resources.default_config()?,
        };
        config.with_defaults(defaults)
    };
    if args.command.is_none()
        && (args.model.is_some()
            || args.style.is_some()
            || args.backend.is_some()
            || args.voice.is_some()
            || args.tts_device.is_some())
    {
        let current = config.load()?;
        let target = args.backend.as_deref().unwrap_or(&current.backend);
        let model = args.model.as_deref().or_else(|| {
            (target == current.backend)
                .then_some(current.model.as_deref())
                .flatten()
        });
        let caps = resources.capabilities_for(target, model)?;
        let model_changed = resources
            .capabilities_for(&current.backend, current.model.as_deref())
            .is_ok_and(|old| old.model != caps.model);
        let voice = args.voice.or_else(|| {
            (target != current.backend || model_changed).then(|| caps.default_voice.clone())
        });
        let tts_device = args.tts_device.or_else(|| {
            args.backend
                .as_ref()
                .filter(|id| *id != &current.backend)
                .map(|_| tts_protocol::Device::Auto)
        });
        #[cfg(any(
            feature = "moss",
            feature = "qwen",
            feature = "voxcpm",
            feature = "omnivoice",
        ))]
        if tts_device.is_some() || model_changed {
            preparation::validate_device(
                tts_device.unwrap_or(current.tts_device),
                target,
                model,
                &resources,
            )?;
        }
        config.update(
            &tts_protocol::ConfigPatch {
                expected_revision: current.revision,
                backend: args.backend.clone(),
                model: args.model.clone(),
                style: args.style.clone(),
                tts_device,
                voice,
                ..Default::default()
            },
            &caps,
        )?;
    }
    talechime::run_local(async move {
        if let Some(Commands::Voices { command }) = args.command {
            let mut selection = config.load()?;
            if let Some(backend) = args.backend {
                if backend != selection.backend {
                    selection.model = None;
                }
                selection.backend = backend;
            }
            if let Some(model) = args.model {
                selection.model = Some(model);
            }
            if let Some(device) = args.tts_device {
                selection.tts_device = device;
            }
            voices::run(command, resources, &selection).await
        } else if args.protocol {
            protocol::run(config, checkpoints, resources, args.readback_models).await
        } else {
            cli::run(
                args.file.ok_or_else(|| {
                    anyhow::anyhow!(
                        "provide a UTF-8 file, --protocol, or voices command; see --help"
                    )
                })?,
                config,
                checkpoints,
                resources,
                cli::CliOptions {
                    restore_checkpoint: !args.restart,
                    plan_file: args.plan,
                    after_chapter: args.after_chapter,
                    verification: tts_protocol::VerificationOptions {
                        policy: match args.verify {
                            VerifyMode::Off => tts_protocol::VerificationPolicy::Off,
                            VerifyMode::Report => tts_protocol::VerificationPolicy::ReportOnly,
                            VerifyMode::Gate => tts_protocol::VerificationPolicy::Gate {
                                max_retries: 1,
                                strict_suspect: false,
                            },
                        },
                        ..Default::default()
                    },
                    report_file: args.verification_report,
                },
            )
            .await
        }
    })
    .await
}

fn parse_device(value: &str) -> Result<tts_protocol::Device, String> {
    match value {
        "auto" => Ok(tts_protocol::Device::Auto),
        "cpu" => Ok(tts_protocol::Device::Cpu),
        "coreml" => Ok(tts_protocol::Device::Coreml),
        "metal" => Ok(tts_protocol::Device::Metal),
        "cuda" => Ok(tts_protocol::Device::Cuda),
        _ => Err("expected auto/cpu/coreml/cuda/metal".into()),
    }
}

#[cfg(test)]
mod argument_tests {
    use super::*;
    #[test]
    fn plan_and_playback_flags_do_not_change_machine_mode_or_voice_selection() {
        assert!(
            Args::try_parse_from(["talechime", "chapter.txt", "--after-chapter"])
                .unwrap()
                .after_chapter
        );
        assert_eq!(
            Args::try_parse_from(["talechime", "chapter.txt", "--plan", "chapter.json"])
                .unwrap()
                .plan,
            Some(PathBuf::from("chapter.json"))
        );
        for args in [
            vec!["talechime", "--protocol", "--after-chapter"],
            vec![
                "talechime",
                "chapter.txt",
                "--plan",
                "p.json",
                "--voice",
                "A",
            ],
            vec![
                "talechime",
                "chapter.txt",
                "--plan",
                "p.json",
                "--after-chapter",
            ],
        ] {
            assert!(Args::try_parse_from(args).is_err());
        }
    }
}
