mod app_session;
mod cli;
#[cfg(test)]
#[path = "../tests/library/fixture.rs"]
#[allow(dead_code)] // Binary adapter tests use only normal/long fixture modes.
mod fixture;
mod migration;
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
    /// Speaking style supported by the selected model; an empty value clears it.
    #[arg(long)]
    style: Option<String>,
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
    /// Disable previous-segment conditioning, including when a plan enables it.
    #[arg(long, conflicts_with = "protocol")]
    no_continuation: bool,
    /// Structure that resets the continuation chain (paragraph|scene|none);
    /// defaults to the plan's policy, then paragraph gaps.
    #[arg(
        long,
        value_parser = parse_continuation_boundary,
        conflicts_with = "protocol"
    )]
    continuation_boundary: Option<talechime_protocol::ContinuationBoundary>,
    /// Chained seconds before the voice re-anchors (default 45).
    #[arg(long, conflicts_with = "protocol")]
    continuation_span: Option<u64>,
    /// Explicitly restart this file from its beginning, ignoring a stored checkpoint.
    #[arg(long, conflicts_with = "protocol")]
    restart: bool,
    /// Generation parameter for the prepared model, repeatable (name=value).
    #[arg(long, value_parser = parse_param)]
    param: Vec<(String, talechime_protocol::ParamValue)>,
    /// Pin sampling for a reproducible run; default draws fresh randomness.
    #[arg(long)]
    seed: Option<u64>,
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

fn parse_continuation_boundary(
    raw: &str,
) -> Result<talechime_protocol::ContinuationBoundary, String> {
    serde_json::from_value::<talechime_protocol::ContinuationBoundary>(serde_json::json!(raw))
        .map_err(|_| format!("expected paragraph, scene or none, got {raw}"))
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
    let legacy_migration =
        if args.config.is_none() && args.checkpoint_dir.is_none() && args.model_dir.is_none() {
            let paths = talechime_core::paths::AppPaths::user_default()?;
            let home = paths
                .root()
                .parent()
                .ok_or_else(|| anyhow::anyhow!("application home unavailable"))?;
            let migration = talechime_core::migration::LegacyMigration::from_home(home);
            migration::startup(&migration, &resources::Resources::new(None)?);
            Some(migration)
        } else {
            None
        };
    if !args.protocol {
        migration::resources(
            legacy_migration.clone(),
            None,
            migration::CopyWork::default(),
        )
        .await;
    }
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
        Some(path) => talechime_core::config::ConfigStore::new(path),
        None => talechime_core::config::ConfigStore::user_default()?,
    };
    let checkpoints = match args.checkpoint_dir {
        Some(path) => talechime_core::checkpoint::CheckpointStore::new(path),
        None => talechime_core::checkpoint::CheckpointStore::user_default()?,
    };
    let resources = resources::Resources::new(args.model_dir)?;
    let config = if config.path().exists() {
        config
    } else {
        let defaults = match args.backend.as_deref() {
            Some(backend) => {
                let caps = resources.capabilities_for(backend, args.model.as_deref())?;
                talechime_protocol::Config {
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
                .map(|_| talechime_protocol::Device::Auto)
        });
        #[cfg(any(
            feature = "moss",
            feature = "qwen",
            feature = "voxcpm",
            feature = "cosyvoice3",
            feature = "indextts",
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
            &talechime_protocol::ConfigPatch {
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
            protocol::run(
                config,
                checkpoints,
                resources,
                args.readback_models,
                legacy_migration,
            )
            .await
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
                    no_continuation: args.no_continuation,
                    continuation_boundary: args.continuation_boundary,
                    continuation_span: args.continuation_span,
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
    fn continuation_flag_defaults_on_and_can_override_a_plan() {
        assert!(
            !Args::try_parse_from(["talechime", "chapter.txt"])
                .unwrap()
                .no_continuation
        );
        assert!(
            Args::try_parse_from([
                "talechime",
                "chapter.txt",
                "--plan",
                "p.json",
                "--no-continuation"
            ])
            .unwrap()
            .no_continuation
        );
        assert!(Args::try_parse_from(["talechime", "--protocol", "--no-continuation"]).is_err());
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
