//! Rust performance workflow. Heavy model execution is delegated to probes.
pub mod build;
pub mod cancellation;
pub mod compare;
pub mod environment;
pub mod evidence;
pub mod executor;
pub mod fixture_job;
pub mod overhead;
pub mod pack;
pub mod plan;
pub mod prepare;
pub mod process;
pub mod profiling;
pub mod records;
pub mod report;
pub mod runner;
pub mod sampling;
pub mod statistics;
pub mod storage;
#[cfg(test)]
pub(crate) mod test_support;

use clap::{Args as ClapArgs, Subcommand};
use std::path::PathBuf;

#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect optional native tools; missing tools do not block baseline runs.
    ProfileTools,
    /// Attach an externally collected native profile to a profile-mode run.
    AttachProfile {
        #[arg(long)]
        run: PathBuf,
        #[arg(long,value_parser=["nsight_systems","wpr"])]
        tool: String,
        #[arg(long)]
        version: String,
        #[arg(long)]
        artifact: PathBuf,
        #[arg(last = true, required = true)]
        invocation: Vec<String>,
    },
    /// Calibrate off/basic/detailed observation costs for the same frozen case.
    Overhead {
        #[arg(long)]
        job: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Export verified report assets in an offline ZIP, excluding models.
    Pack {
        #[arg(long)]
        report: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Generate a portable offline report and copy only approved evidence assets.
    Report {
        #[arg(long)]
        run: PathBuf,
        #[arg(long, default_value = "results.json")]
        results: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Create a model-free smoke job, explicitly labelled as synthetic evidence.
    FixtureJob {
        #[arg(long)]
        build: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long,default_value="baseline",value_parser=["off","baseline","trace","profile"])]
        observation: String,
    },
    /// Execute a frozen case serially using a previously built probe.
    Run {
        #[arg(long)]
        job: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Reuse only audited compatible commits; preserve every older attempt.
    Resume {
        #[arg(long)]
        job: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Compare raw per-case attempt arrays under an explicit policy.
    Compare {
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Resolve production paths and optionally verify a prepared model offline.
    Prepare {
        #[arg(long)]
        build: PathBuf,
        #[arg(long)]
        app_root: Option<PathBuf>,
        #[arg(long, requires = "model")]
        backend: Option<String>,
        #[arg(long, requires = "backend")]
        model: Option<String>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Build an isolated release probe and publish its immutable receipt.
    Build {
        #[arg(long, default_value = "performance_catalog")]
        probe: String,
        #[arg(long, value_delimiter = ',')]
        features: Vec<String>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Inspect tool availability without loading a model or changing resources.
    Doctor,
    /// Validate a frozen plan, rejecting unknown versions and invalid states.
    Check { plan: PathBuf },
    /// Freeze an explicit platform plan. Existing output is never overwritten.
    Plan {
        #[arg(long, value_enum, default_value = "windows")]
        platform: plan::Platform,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        machine: String,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long,default_value="baseline",value_parser=["off","baseline","trace","profile"])]
        observation: String,
    },
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let _handler = matches!(
        args.command,
        Command::Run { .. } | Command::Resume { .. } | Command::Overhead { .. }
    )
    .then(cancellation::install)
    .transpose()?;
    match args.command {
        Command::ProfileTools => {
            println!("{}", serde_json::to_string_pretty(&profiling::tools())?);
            Ok(())
        }
        Command::AttachProfile {
            run,
            tool,
            version,
            artifact,
            invocation,
        } => profiling::attach(&run, &tool, &version, invocation, &artifact),
        Command::Overhead { job, output } => overhead::measure(&job, &output),
        Command::Pack { report, output } => pack::pack(&report, &output),
        Command::Report {
            run,
            results,
            output,
        } => report::report(&run, &results, &output),
        Command::FixtureJob {
            build,
            output,
            observation,
        } => fixture_job::create_mode(&build, &output, &observation),
        Command::Run { job, output } => execute_job(&job, &output, false),
        Command::Resume { job, output } => execute_job(&job, &output, true),
        Command::Compare {
            baseline,
            candidate,
            policy,
            output,
        } => {
            let baseline: Vec<records::Attempt> = evidence::read_json(&baseline)?;
            let candidate: Vec<records::Attempt> = evidence::read_json(&candidate)?;
            let policy = evidence::read_json(&policy)?;
            let result = compare::compare(&baseline, &candidate, policy)?;
            evidence::publish_json(&output, &result)?;
            println!("{}", result.verdict);
            Ok(())
        }
        Command::Prepare {
            build,
            app_root,
            backend,
            model,
            output,
        } => prepare::prepare(
            &build,
            app_root.as_deref(),
            backend.as_deref(),
            model.as_deref(),
            &output,
        ),
        Command::Build {
            probe,
            features,
            output,
        } => build::build(&probe, features, &output),
        Command::Check { plan } => {
            let plan: plan::Plan = evidence::read_json(&plan)?;
            plan.validate()?;
            println!("valid: {} cases", plan.cases.len());
            Ok(())
        }
        Command::Doctor => {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "schema": evidence::VERSION,
                    "host_os": std::env::consts::OS,
                    "host_arch": std::env::consts::ARCH,
                    "environment": environment::capture_environment("cpu")?,
                    "python_required": false,
                "model_execution": "native tool available on Windows/macOS; per-model validation separate",
                    "macos_native": {"lifecycle": if cfg!(target_os = "macos") {"available"} else {"not_this_host"}, "rss": if cfg!(target_os = "macos") {"available"} else {"not_this_host"}, "models":"unverified; explicit readiness and smoke required", "instruments":"deferred", "metal_allocation":"unavailable"},
                    "product_executable": "talechime"
                }))?
            );
            Ok(())
        }
        Command::Plan {
            platform,
            output,
            machine,
            catalog,
            observation,
        } => {
            let mut plan = plan::Plan::new(platform, machine)?;
            plan.observation = observation;
            if let Some(path) = catalog {
                plan.apply_catalog(&prepare::load_catalog(&path)?)?;
            }
            plan.validate()?;
            evidence::publish_json(&output, &plan)?;
            println!("{}", output.display());
            Ok(())
        }
    }
}

fn execute_job(
    job: &std::path::Path,
    output: &std::path::Path,
    resume: bool,
) -> anyhow::Result<()> {
    let path = runner::run(job, output, resume)?;
    println!("{}", path.display());
    let attempts: Vec<records::Attempt> = evidence::read_json(&path)?;
    anyhow::ensure!(
        !cancellation::requested()
            && !attempts.is_empty()
            && attempts
                .iter()
                .all(|attempt| attempt.outcome == "success" && attempt.cleanup_verified),
        "run contains unsuccessful attempts; inspect retained results"
    );
    Ok(())
}
