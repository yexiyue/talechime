//! Offline production-path measurement. Parent owns isolation, deadlines and evidence commit.
#[path = "performance_support/observed.rs"]
mod observed;
#[path = "performance_support/probe.rs"]
mod probe;
#[path = "performance_support/telemetry.rs"]
mod telemetry;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    if std::env::args().nth(1).as_deref() == Some("--execution-info") {
        let cpu = if candle_core::utils::has_accelerate() {
            "accelerate"
        } else {
            "rust"
        };
        println!(
            "{}",
            serde_json::json!({"schema":"talechime/performance-execution/1",
            "os":std::env::consts::OS,"arch":std::env::consts::ARCH,
            "cpu_math":format!("build:candle-{cpu};ort-if-selected;operator-usage-unmeasured"),
            "metal_math":format!("build:candle-metal;host-cpu-{cpu};operator-usage-unmeasured"),
            "timing_contract":"production_synthesis_stream_consumer_start_to_completed/1",
            "thread_observation":"rayon configured; VECLIB_MAXIMUM_THREADS inherited frozen setting; actual operator threads unmeasured"})
        );
        return Ok(());
    }
    tokio::task::LocalSet::new().run_until(probe::main()).await
}
