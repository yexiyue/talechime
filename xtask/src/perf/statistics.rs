//! Preserve round/process hierarchy; never substitute pooled or fastest-round timing.
use super::records::{Attempt, Round};
use anyhow::ensure;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Rtf,
    SynthesisSeconds,
    FirstPcmSeconds,
}

impl Metric {
    pub fn value(self, round: &Round) -> Option<f64> {
        let value = match self {
            Self::Rtf => round.rtf(),
            Self::SynthesisSeconds => round.synthesis_seconds,
            Self::FirstPcmSeconds => round.first_pcm_seconds,
        };
        value.filter(|n| n.is_finite() && *n >= 0.0)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Summary {
    pub metric: Metric,
    pub complete: bool,
    pub expected_processes: usize,
    pub observed_processes: usize,
    pub successful_hot_rounds: usize,
    pub expected_hot_rounds: usize,
    pub process_means: Vec<f64>,
    pub mean_of_process_means: Option<f64>,
    pub process_min: Option<f64>,
    pub process_max: Option<f64>,
    pub warmup_rounds_excluded: usize,
}

pub fn summarize(
    attempts: &[Attempt],
    expected_processes: usize,
    metric: Metric,
) -> anyhow::Result<Summary> {
    ensure!(
        expected_processes > 0,
        "expected process count must be positive"
    );
    let mut indices = BTreeSet::new();
    let mut means = Vec::new();
    let mut successful = 0;
    let mut expected_hot = 0;
    let mut warmups = 0;
    let mut complete = attempts.len() == expected_processes;
    for attempt in attempts {
        attempt.validate()?;
        ensure!(
            indices.insert(attempt.process_index),
            "duplicate process index"
        );
        ensure!(
            (attempt.process_index as usize) < expected_processes,
            "process index exceeds plan"
        );
        if let Some(first) = attempts.first() {
            ensure!(
                first.conditions == attempt.conditions
                    && first.case_id == attempt.case_id
                    && first.source == attempt.source
                    && first.binary_sha256 == attempt.binary_sha256
                    && first.expected_hot_rounds == attempt.expected_hot_rounds,
                "mixed measurement identities in one case"
            );
        }
        expected_hot += attempt.expected_hot_rounds as usize;
        let mut values = Vec::new();
        for round in &attempt.rounds {
            if round.warmup {
                warmups += 1;
                continue;
            }
            if round.outcome == "success"
                && let Some(value) = metric.value(round)
            {
                values.push(value);
            }
        }
        successful += values.len();
        complete &= attempt.outcome == "success"
            && attempt.cleanup_verified
            && values.len() == attempt.expected_hot_rounds as usize;
        if !values.is_empty() {
            means.push(values.iter().sum::<f64>() / values.len() as f64);
        }
    }
    // Missing processes still contribute their planned expected hot rounds.
    if let Some(first) = attempts.first() {
        expected_hot +=
            expected_processes.saturating_sub(attempts.len()) * first.expected_hot_rounds as usize;
    }
    let mean = (!means.is_empty()).then(|| means.iter().sum::<f64>() / means.len() as f64);
    let min = means.iter().copied().reduce(f64::min);
    let max = means.iter().copied().reduce(f64::max);
    Ok(Summary {
        metric,
        complete,
        expected_processes,
        observed_processes: attempts.len(),
        successful_hot_rounds: successful,
        expected_hot_rounds: expected_hot,
        process_means: means,
        mean_of_process_means: mean,
        process_min: min,
        process_max: max,
        warmup_rounds_excluded: warmups,
    })
}

#[cfg(test)]
mod tests {
    use super::super::test_support::attempts;
    use super::*;
    #[test]
    fn warmup_exclusion_process_means_and_missing_processes_remain_visible() -> anyhow::Result<()> {
        let input = attempts(&[1.0, 2.0, 3.0]);
        let result = summarize(&input, 3, Metric::Rtf)?;
        assert!(result.complete);
        assert_eq!(result.process_means, [0.5, 1.0, 1.5]);
        assert_eq!(result.mean_of_process_means, Some(1.0));
        assert_eq!(result.warmup_rounds_excluded, 3);
        let result = summarize(&input[..2], 3, Metric::Rtf)?;
        assert!(!result.complete);
        assert_eq!(
            (result.successful_hot_rounds, result.expected_hot_rounds),
            (2, 3)
        );
        Ok(())
    }
}
