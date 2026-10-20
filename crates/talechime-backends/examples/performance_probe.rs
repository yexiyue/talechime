//! Offline production-path measurement. Parent owns isolation, deadlines and evidence commit.
#[path = "performance_support/observed.rs"]
mod observed;
#[path = "performance_support/probe.rs"]
mod probe;
#[path = "performance_support/telemetry.rs"]
mod telemetry;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    tokio::task::LocalSet::new().run_until(probe::main()).await
}
