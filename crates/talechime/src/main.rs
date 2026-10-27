mod app_session;
mod cli;
#[cfg(test)]
#[path = "../tests/library/fixture.rs"]
#[allow(dead_code)] // Binary adapter tests use only normal/long fixture modes.
mod fixture;
mod models;
mod plan_input;
mod preparation;
mod protocol;
use talechime_backends::Registry;
mod runtime;
#[cfg(feature = "asr")]
mod verify;
mod voices;

use clap::Parser;
use std::path::PathBuf;

#[derive(clap::Subcommand)]
enum Commands {
    /// List synthesis models, devices, built-in voices and generation parameters. No downloads.
    Models {
        #[arg(long)]
        backend: Option<String>,
        #[arg(long)]
        model: Option<String>,
        /// Emit the complete capability catalog as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Compare existing PCM16 WAV to UTF-8 source without synthesis or playback.
    #[cfg(feature = "asr")]
    Verify {
        text: PathBuf,
        audio: PathBuf,
        #[arg(long)]
        report: PathBuf,
    },
    /// Design, import and manage the cross-model voice library.
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
    about = "Read text aloud, manage voices, or serve JSON Lines",
    after_help = "Examples:
  talechime models
  talechime chapter.txt --backend moss --tts-device cpu
  talechime voices list
  talechime voices design --description 'A warm, low male voice'
  talechime --protocol"
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
    /// Synthesis backend; discover available models with `models`.
    #[arg(long)]
    backend: Option<String>,
    /// Built-in voice (`models`) or library ID (`voices list`). Saved for later reads.
    #[arg(long)]
    voice: Option<String>,
    /// Model ID within the selected backend; see `models`.
    #[arg(long)]
    model: Option<String>,
    /// Speaking style supported by the selected model; an empty value clears it.
    #[arg(long)]
    style: Option<String>,
    /// Synthesis device: auto, cpu, metal, cuda or coreml; availability is model-specific.
    #[arg(long, value_parser = parse_device)]
    tts_device: Option<talechime_protocol::Device>,
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
    /// Generation parameter for the prepared model, repeatable (name=value).
    #[arg(long, value_parser = parse_param, conflicts_with = "protocol")]
    param: Vec<(String, talechime_protocol::ParamValue)>,
    /// Pin sampling for a reproducible run; default draws fresh randomness.
    #[arg(long, conflicts_with = "protocol")]
    seed: Option<u64>,
    /// Override the listening configuration path.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    /// Override the model directory (does not copy or delete existing models).
    #[arg(long, global = true)]
    model_dir: Option<PathBuf>,
    /// Override the independent listening checkpoint directory.
    #[arg(long, global = true)]
    checkpoint_dir: Option<PathBuf>,
}

/// Parse name=value where the value is an int, float or quoted-by-shape text.
fn parse_param(raw: &str) -> Result<(String, talechime_protocol::ParamValue), String> {
    let (name, value) = raw
        .split_once('=')
        .ok_or_else(|| format!("expected name=value, got {raw}"))?;
    if name.is_empty() {
        return Err("parameter name must not be empty".into());
    }
    let value = if let Ok(int) = value.parse::<i64>() {
        talechime_protocol::ParamValue::Int(int)
    } else if let Ok(float) = value.parse::<f64>() {
        talechime_protocol::ParamValue::Float(float)
    } else {
        talechime_protocol::ParamValue::Text(value.to_owned())
    };
    Ok((name.to_owned(), value))
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
        let resources = Registry::new(args.model_dir.clone())?;
        return talechime::run_local(verify::run(text, audio, report, resources.root())).await;
    }
    let config = match args.config {
        Some(path) => talechime_core::config::ConfigStore::new(path),
        None => talechime_core::config::ConfigStore::user_default()?,
    };
    let checkpoints = match args.checkpoint_dir {
        Some(path) => talechime_core::checkpoint::CheckpointStore::new(path),
        None => talechime_core::checkpoint::CheckpointStore::user_default()?,
    };
    let resources = Registry::new(args.model_dir)?;
    if let Some(Commands::Models {
        backend,
        model,
        json,
    }) = args.command
    {
        return models::run(
            resources.root(),
            backend.or(args.backend).as_deref(),
            model.or(args.model).as_deref(),
            json,
        );
    }
    if let Some(Commands::Voices { command }) = args.command {
        // Asset operations depend on the config location, not synthesis settings.
        let voices = config
            .path()
            .parent()
            .map(|root| root.join("voices"))
            .ok_or_else(|| anyhow::anyhow!("config path needs a parent directory"))?;
        let studio = talechime::VoiceStudio::at(resources.root(), voices)?;
        return talechime::run_local(voices::run(command, studio)).await;
    }
    let plan = match args.plan.as_deref() {
        Some(path) => Some(plan_input::read_file(path).await?),
        None => None,
    };
    if let Some(plan) = &plan {
        anyhow::ensure!(
            args.backend
                .as_ref()
                .is_none_or(|backend| backend == &plan.backend)
                && args
                    .model
                    .as_ref()
                    .is_none_or(|model| Some(model) == plan.model.as_ref()),
            "CLI backend/model must match the plan"
        );
    }
    let selected_backend = args
        .backend
        .as_deref()
        .or_else(|| plan.as_ref().map(|plan| plan.backend.as_str()));
    let selected_model = args
        .model
        .as_deref()
        .or_else(|| plan.as_ref().and_then(|plan| plan.model.as_deref()));
    let config = if config.path().exists() {
        config
    } else {
        let defaults = match selected_backend {
            Some(backend) => {
                let caps = resources.capabilities_for(backend, selected_model)?;
                talechime_protocol::Config {
                    backend: caps.backend,
                    model: caps.model,
                    voice: plan
                        .as_ref()
                        .and_then(|plan| plan.voices.first())
                        .cloned()
                        .unwrap_or(caps.default_voice),
                    ..Default::default()
                }
            }
            None if args.protocol && resources.catalog()?.is_empty() => Default::default(),
            None => resources.default_config()?,
        };
        config.with_defaults(defaults)
    };
    // Reject retired identities before a CLI override can persist a replacement.
    let stored = config.load()?;
    if cfg!(feature = "moss") && stored.backend == "moss" {
        resources.resolve(&stored.backend, stored.model.as_deref())?;
    }
    if args.command.is_none()
        && (plan.is_some()
            || args.model.is_some()
            || args.style.is_some()
            || args.backend.is_some()
            || args.voice.is_some()
            || args.tts_device.is_some())
    {
        let current = config.load()?;
        let target = selected_backend.unwrap_or(&current.backend);
        let model = selected_model.or_else(|| {
            (target == current.backend)
                .then_some(current.model.as_deref())
                .flatten()
        });
        let mut caps = resources.capabilities_for(target, model)?;
        let model_changed = resources
            .capabilities_for(&current.backend, current.model.as_deref())
            .is_ok_and(|old| old.model != caps.model);
        let voice = args
            .voice
            .or_else(|| plan.as_ref().and_then(|plan| plan.voices.first()).cloned())
            .or_else(|| {
                (target != current.backend || model_changed).then(|| caps.default_voice.clone())
            });
        let studio = preparation::voice_studio(&config, &resources)?;
        preparation::include_library_voice(
            &mut caps,
            voice.as_deref().unwrap_or(&current.voice),
            &studio,
        )?;
        let tts_device = args.tts_device.or_else(|| {
            selected_backend
                .filter(|id| *id != current.backend)
                .map(|_| talechime_protocol::Device::Auto)
        });
        if tts_device.is_some() || model_changed {
            preparation::validate_device(
                tts_device.unwrap_or(current.tts_device),
                target,
                model,
                &resources,
            )?;
        }
        config.update(
            &talechime_protocol::ConfigPatch {
                expected_revision: current.revision,
                backend: selected_backend.map(str::to_owned),
                model: selected_model.map(str::to_owned),
                style: args.style.clone(),
                tts_device,
                voice,
                ..Default::default()
            },
            &caps,
        )?;
    }
    talechime::run_local(async move {
        if args.protocol {
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
                    plan,
                    after_chapter: args.after_chapter,

                    params: args.param,
                    seed: args.seed,
                    verification: talechime_protocol::VerificationOptions {
                        policy: match args.verify {
                            VerifyMode::Off => talechime_protocol::VerificationPolicy::Off,
                            VerifyMode::Report => {
                                talechime_protocol::VerificationPolicy::ReportOnly
                            }
                            VerifyMode::Gate => talechime_protocol::VerificationPolicy::Gate {
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

fn parse_device(value: &str) -> Result<talechime_protocol::Device, String> {
    match value {
        "auto" => Ok(talechime_protocol::Device::Auto),
        "cpu" => Ok(talechime_protocol::Device::Cpu),
        "coreml" => Ok(talechime_protocol::Device::Coreml),
        "metal" => Ok(talechime_protocol::Device::Metal),
        "cuda" => Ok(talechime_protocol::Device::Cuda),
        _ => Err("expected auto/cpu/coreml/cuda/metal".into()),
    }
}

#[cfg(test)]
mod argument_tests {
    use super::*;
    #[test]
    fn removed_reference_strategy_flags_are_rejected() {
        for args in [
            vec!["talechime", "chapter.txt", "--no-continuation"],
            vec![
                "talechime",
                "chapter.txt",
                "--continuation-boundary",
                "scene",
            ],
            vec!["talechime", "chapter.txt", "--continuation-span", "45"],
        ] {
            assert!(Args::try_parse_from(args).is_err());
        }
    }

    #[test]
    fn param_flags_parse_typed_values_and_seed() {
        let args = Args::try_parse_from([
            "talechime",
            "chapter.txt",
            "--param",
            "steps=32",
            "--param",
            "temperature=0.8",
            "--param",
            "language=english",
            "--seed",
            "7",
        ])
        .unwrap();
        assert_eq!(
            args.param,
            vec![
                ("steps".to_string(), talechime_protocol::ParamValue::Int(32)),
                (
                    "temperature".to_string(),
                    talechime_protocol::ParamValue::Float(0.8)
                ),
                (
                    "language".to_string(),
                    talechime_protocol::ParamValue::Text("english".into())
                ),
            ]
        );
        assert_eq!(args.seed, Some(7));
        assert!(Args::try_parse_from(["talechime", "chapter.txt", "--param", "steps"]).is_err());
        assert!(Args::try_parse_from(["talechime", "chapter.txt", "--param", "=5"]).is_err());
        assert!(
            !Args::try_parse_from(["talechime", "chapter.txt"])
                .unwrap()
                .param
                .iter()
                .any(|(name, _)| name == "steps")
        );
    }
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
            vec!["talechime", "--protocol", "--seed", "7"],
            vec!["talechime", "--protocol", "--param", "steps=32"],
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
