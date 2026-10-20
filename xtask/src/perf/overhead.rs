//! Matched observation modes; output identity precedes overhead claims.
use super::{
    evidence::{VERSION, publish_json, read_json},
    plan::Plan,
    records::Attempt,
    runner::{Job, run},
    statistics::{Metric, summarize},
};
use anyhow::ensure;
use std::path::Path;

pub fn measure(template: &Path, output: &Path) -> anyhow::Result<()> {
    ensure!(!output.exists(), "overhead output already exists");
    let template: Job = read_json(template)?;
    let plan: Plan = read_json(&template.plan)?;
    std::fs::create_dir_all(output)?;
    let root = output.canonicalize()?;
    let mut data = Vec::new();
    let mut reference_hashes = None;
    let mut reference_mean = None;
    let mut all_equal = true;
    let mut all_complete = true;
    for mode in ["off", "baseline", "trace"] {
        ensure!(
            !super::cancellation::requested(),
            "observation calibration cancelled; partial evidence retained"
        );
        let input = root.join(format!("{mode}-input"));
        std::fs::create_dir(&input)?;
        let mut plan_value = serde_json::to_value(&plan)?;
        plan_value["observation"] = mode.into();
        publish_json(&input.join("plan.json"), &plan_value)?;
        let mut job: Job = serde_json::from_value(serde_json::to_value(&template)?)?;
        job.plan = input.join("plan.json");
        job.conditions.observation = mode.into();
        if mode == "off" {
            job.budget.sample_interval_ms = None;
        }
        job.conditions
            .observation_config
            .insert("capture_budget".into(), serde_json::to_string(&job.budget)?);
        publish_json(&input.join("job.json"), &job)?;
        let results = run(&input.join("job.json"), &root.join(mode), false)?;
        let attempts: Vec<Attempt> = read_json(&results)?;
        let summary = summarize(&attempts, plan.processes as usize, Metric::SynthesisSeconds)?;
        let hashes = attempts
            .iter()
            .flat_map(|a| {
                a.rounds
                    .iter()
                    .filter(|r| !r.warmup)
                    .map(|r| r.output_sha256.clone())
            })
            .collect::<Vec<_>>();
        if reference_hashes.is_none() {
            reference_hashes = Some(hashes.clone());
            reference_mean = summary.mean_of_process_means;
        }
        let equal =
            reference_hashes.as_ref() == Some(&hashes) && hashes.iter().all(Option::is_some);
        all_equal &= equal;
        all_complete &= summary.complete;
        let overhead = if equal && summary.complete {
            reference_mean
                .filter(|n| *n > 0.0)
                .zip(summary.mean_of_process_means)
                .map(|(a, b)| (b - a) / a)
        } else {
            None
        };
        data.push(serde_json::json!({"mode":mode,"results":format!("{mode}/results.json"),"summary":summary,"output_equivalent_to_off":equal,"relative_overhead":overhead}));
    }
    ensure!(
        !super::cancellation::requested(),
        "observation calibration cancelled; partial evidence retained"
    );
    publish_json(
        &root.join("overhead.json"),
        &serde_json::json!({"schema":VERSION,"observations":data,"execution_order":["off","baseline","trace"],"output_preserving":all_equal,"complete":all_complete,"scope":if template.conditions.backend=="fixture"{"synthetic_fixture_only"}else{"real_model_descriptive"},"real_model_overhead_acceptance":"unverified","interpretation":"independent-process descriptive calibration; inspect raw distributions and environmental samples; no general overhead guarantee"}),
    )
}
