//! Explicit compatibility and practical-effect policy before regression verdicts.
use super::{
    evidence::VERSION,
    records::Attempt,
    statistics::{Metric, Summary, summarize},
};
use anyhow::ensure;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub schema: String,
    pub metric: Metric,
    pub lower_is_better: bool,
    pub absolute_threshold: f64,
    pub relative_threshold: f64,
    pub expected_processes: usize,
    pub require_equal_output: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Comparison {
    pub schema: String,
    pub verdict: String,
    pub reasons: Vec<String>,
    pub policy: Policy,
    pub baseline: Summary,
    pub candidate: Summary,
    pub delta: Option<f64>,
    pub relative_delta: Option<f64>,
    pub output_equivalent: bool,
}

pub fn compare(
    baseline: &[Attempt],
    candidate: &[Attempt],
    policy: Policy,
) -> anyhow::Result<Comparison> {
    ensure!(
        policy.schema == VERSION,
        "unsupported comparison policy schema"
    );
    ensure!(
        policy.absolute_threshold.is_finite()
            && policy.absolute_threshold > 0.0
            && policy.relative_threshold.is_finite()
            && policy.relative_threshold > 0.0,
        "policy requires finite positive thresholds"
    );
    let a = summarize(baseline, policy.expected_processes, policy.metric)?;
    let b = summarize(candidate, policy.expected_processes, policy.metric)?;
    let mut verdict = "inconclusive";
    let mut reasons = Vec::new();
    let mut delta = None;
    let mut relative_delta = None;
    let mut output_equivalent = false;
    if let (Some(first), Some(second)) = (baseline.first(), candidate.first()) {
        if let Err(error) = first.conditions.validate() {
            reasons.push(format!("baseline conditions unverified: {error}"));
        }
        if let Err(error) = second.conditions.validate() {
            reasons.push(format!("candidate conditions unverified: {error}"));
        }
        let old = serde_json::to_value(&first.conditions)?;
        let new = serde_json::to_value(&second.conditions)?;
        for (key, value) in old.as_object().unwrap() {
            if new.get(key) != Some(value) {
                reasons.push(format!("incompatible condition: {key}"));
            }
        }
        if !matches!(first.conditions.observation.as_str(), "baseline" | "off")
            || !matches!(second.conditions.observation.as_str(), "baseline" | "off")
        {
            reasons.push("diagnostic observation mode cannot produce baseline verdict".into());
        }
        let hashes = |attempts: &[Attempt]| {
            let mut processes: Vec<_> = attempts.iter().collect();
            processes.sort_by_key(|a| a.process_index);
            processes
                .into_iter()
                .flat_map(|a| {
                    a.rounds
                        .iter()
                        .filter(|r| !r.warmup)
                        .map(|r| r.output_sha256.clone())
                })
                .collect::<Vec<_>>()
        };
        let old_hashes = hashes(baseline);
        let new_hashes = hashes(candidate);
        output_equivalent = !old_hashes.is_empty()
            && old_hashes.iter().all(Option::is_some)
            && old_hashes == new_hashes;
    }
    if !reasons.is_empty() {
        verdict = "incomparable";
    } else if !a.complete || !b.complete {
        verdict = "incomplete";
        reasons.push("required processes or hot rounds are incomplete".into());
    } else if policy.require_equal_output && !output_equivalent {
        reasons.push("PCM output identity differs; no output-preserving speed claim".into());
    } else if let (Some(old), Some(new)) = (a.mean_of_process_means, b.mean_of_process_means) {
        let difference = new - old;
        delta = Some(difference);
        if old > 0.0 {
            relative_delta = Some(difference / old);
        }
        let practically_equal = difference.abs() < policy.absolute_threshold
            || relative_delta.is_some_and(|d| d.abs() < policy.relative_threshold);
        let separated = b.process_min.unwrap() > a.process_max.unwrap()
            || b.process_max.unwrap() < a.process_min.unwrap();
        if practically_equal {
            verdict = "pass";
        } else if separated && relative_delta.is_some() {
            let worsened = if policy.lower_is_better {
                difference > 0.0
            } else {
                difference < 0.0
            };
            verdict = if worsened { "regression" } else { "pass" };
        } else {
            reasons.push("process ranges overlap or reference mean is zero".into());
        }
    }
    Ok(Comparison {
        schema: VERSION.into(),
        verdict: verdict.into(),
        reasons,
        policy,
        baseline: a,
        candidate: b,
        delta,
        relative_delta,
        output_equivalent,
    })
}

#[cfg(test)]
mod tests {
    use super::super::test_support::attempts;
    use super::*;
    fn policy() -> Policy {
        Policy {
            schema: VERSION.into(),
            metric: Metric::SynthesisSeconds,
            lower_is_better: true,
            absolute_threshold: 0.1,
            relative_threshold: 0.1,
            expected_processes: 3,
            require_equal_output: true,
        }
    }
    #[test]
    fn code_changes_are_allowed_but_platform_changes_are_incomparable() -> anyhow::Result<()> {
        let baseline = attempts(&[1.0; 3]);
        let mut candidate = attempts(&[1.5; 3]);
        for attempt in &mut candidate {
            attempt.source.revision = "optimized".into();
            attempt.binary_sha256 = "b".repeat(64);
        }
        assert_eq!(
            compare(&baseline, &candidate, policy())?.verdict,
            "regression"
        );
        for attempt in &mut candidate {
            attempt.conditions.os = "macos-fixture".into();
            attempt.conditions.provider = "metal".into();
        }
        let result = compare(&baseline, &candidate, policy())?;
        assert_eq!(result.verdict, "incomparable");
        assert!(result.reasons.iter().any(|r| r.ends_with("provider")));
        Ok(())
    }
    #[test]
    fn verdicts_preserve_coverage_variation_and_output_evidence() -> anyhow::Result<()> {
        let baseline = attempts(&[1.0; 3]);
        assert_eq!(
            compare(&baseline, &attempts(&[0.5; 3]), policy())?.verdict,
            "pass"
        );
        assert_eq!(
            compare(&baseline, &attempts(&[1.5; 2]), policy())?.verdict,
            "incomplete"
        );
        assert_eq!(
            compare(&baseline, &attempts(&[0.8, 1.5, 1.7]), policy())?.verdict,
            "inconclusive"
        );
        let mut different_pcm = attempts(&[0.5; 3]);
        for attempt in &mut different_pcm {
            attempt.rounds[1].output_sha256 = Some("f".repeat(64));
        }
        assert_eq!(
            compare(&baseline, &different_pcm, policy())?.verdict,
            "inconclusive"
        );
        for attempt in &mut different_pcm {
            attempt.conditions.observation = "trace".into();
        }
        assert_eq!(
            compare(&baseline, &different_pcm, policy())?.verdict,
            "incomparable"
        );
        Ok(())
    }

    #[test]
    fn uninstrumented_off_runs_are_comparable_without_relabeling() -> anyhow::Result<()> {
        let mut baseline = attempts(&[1.0; 3]);
        let mut candidate = attempts(&[0.5; 3]);
        for attempt in baseline.iter_mut().chain(&mut candidate) {
            attempt.conditions.observation = "off".into();
        }
        let result = compare(&baseline, &candidate, policy())?;
        assert_eq!(result.verdict, "pass");
        assert!(result.output_equivalent);
        assert_eq!(result.relative_delta, Some(-0.5));

        // Accept each mode, but never erase differences between frozen conditions.
        for attempt in &mut candidate {
            attempt.conditions.observation = "baseline".into();
        }
        let result = compare(&baseline, &candidate, policy())?;
        assert_eq!(result.verdict, "incomparable");
        assert!(
            result
                .reasons
                .iter()
                .any(|reason| reason.ends_with("observation"))
        );
        Ok(())
    }

    #[test]
    fn matching_diagnostic_modes_still_cannot_produce_baseline_verdicts() -> anyhow::Result<()> {
        for mode in ["trace", "profile"] {
            let mut baseline = attempts(&[1.0; 3]);
            let mut candidate = attempts(&[0.5; 3]);
            for attempt in baseline.iter_mut().chain(&mut candidate) {
                attempt.conditions.observation = mode.into();
            }
            let result = compare(&baseline, &candidate, policy())?;
            assert_eq!(result.verdict, "incomparable");
            assert!(
                result
                    .reasons
                    .iter()
                    .any(|reason| reason.contains("diagnostic observation"))
            );
        }
        Ok(())
    }
}
