//! Lightweight repository tools. Model and audio dependencies belong to probes.
use xtask::perf;
mod fixture;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Talechime developer tools; not a product executable")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(name = "__perf-fixture", hide = true)]
    Fixture {
        mode: String,
        #[arg(long)]
        marker: Option<std::path::PathBuf>,
    },
    /// Plan and inspect attributable performance evidence.
    Perf(perf::Args),
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Fixture { mode, marker } => fixture::run(&mode, marker.as_deref()),
        Command::Perf(args) => perf::run(args),
    }
}
