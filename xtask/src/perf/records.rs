//! Authoritative raw observations, never reconstructed from aggregate histograms.
use super::evidence::{Reading, Source, VERSION, contained, file_digest};
use anyhow::ensure;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Conditions {
    pub machine: String,
    pub os: String,
    pub arch: String,
    pub hardware: String,
    pub driver: String,
    pub provider: String,
    pub math_backend: String,
    pub precision: String,
    pub features: Vec<String>,
    pub threads: BTreeMap<String, u32>,
    pub backend: String,
    pub model: String,
    pub resource_sha256: String,
    pub corpus_sha256: String,
    pub conditioning_sha256: String,
    pub parameters: BTreeMap<String, String>,
    pub mode: String,
    /// Why first PCM has the observed latency: native streaming or whole segment.
    pub delivery_boundary: String,
    pub observation: String,
    pub timing_contract: String,
    #[serde(default)]
    pub observation_config: BTreeMap<String, String>,
    #[serde(default)]
    pub environment: BTreeMap<String, Option<String>>,
}

impl Conditions {
    pub fn validate(&self) -> anyhow::Result<()> {
        for (name, value) in [
            ("machine", &self.machine),
            ("os", &self.os),
            ("hardware", &self.hardware),
            ("provider", &self.provider),
            ("math_backend", &self.math_backend),
            ("precision", &self.precision),
            ("backend", &self.backend),
            ("model", &self.model),
            ("delivery_boundary", &self.delivery_boundary),
            ("timing_contract", &self.timing_contract),
        ] {
            ensure!(
                !value.is_empty() && !value.starts_with("unverified"),
                "unverified comparison condition: {name}"
            );
        }
        for value in [
            &self.resource_sha256,
            &self.corpus_sha256,
            &self.conditioning_sha256,
        ] {
            ensure!(
                value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid condition digest"
            );
        }
        self.rayon_threads()?;
        if self.os.starts_with("macos")
            && let Some(Some(value)) = self.environment.get("VECLIB_MAXIMUM_THREADS")
        {
            ensure!(
                value.bytes().all(|byte| byte.is_ascii_digit())
                    && value.parse::<i32>().is_ok_and(|count| count > 0),
                "VECLIB_MAXIMUM_THREADS must be a positive 32-bit integer"
            );
        }
        ensure!(
            ["baseline", "trace", "profile", "off"].contains(&self.observation.as_str()),
            "invalid observation profile"
        );
        Ok(())
    }

    pub fn rayon_threads(&self) -> anyhow::Result<u32> {
        ensure!(
            self.threads.len() == 1,
            "freeze exactly one supported thread setting: rayon"
        );
        let count = self
            .threads
            .get("rayon")
            .copied()
            .filter(|count| *count > 0)
            .ok_or_else(|| anyhow::anyhow!("explicit positive rayon thread count required"))?;
        Ok(count)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub name: String,
    pub parent: Option<String>,
    pub boundary: String,
    pub location: String,
    pub reading: Reading,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: std::path::PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub kind: String,
    pub derived_from: Option<String>,
}

impl Artifact {
    pub fn verify(&self, root: &Path) -> anyhow::Result<()> {
        let path = contained(root, &self.path)?;
        ensure!(
            path.is_file() && path.metadata()?.len() == self.bytes,
            "missing or changed artifact: {}",
            self.path.display()
        );
        ensure!(
            file_digest(&path)? == self.sha256,
            "artifact checksum mismatch"
        );
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Round {
    pub index: u32,
    pub warmup: bool,
    pub outcome: String,
    pub synthesis_seconds: Option<f64>,
    pub audio_seconds: Option<f64>,
    pub first_pcm_seconds: Option<f64>,
    #[serde(default)]
    pub delivery_boundary: Option<String>,
    pub output_sha256: Option<String>,
    pub observations: Vec<Observation>,
    pub artifacts: Vec<Artifact>,
}

impl Round {
    pub fn rtf(&self) -> Option<f64> {
        match (self.synthesis_seconds, self.audio_seconds) {
            (Some(time), Some(audio))
                if time.is_finite() && time >= 0.0 && audio.is_finite() && audio > 0.0 =>
            {
                let ratio = time / audio;
                ratio.is_finite().then_some(ratio)
            }
            _ => None,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            [
                "success",
                "failure",
                "timeout",
                "cancelled",
                "truncated",
                "blocked"
            ]
            .contains(&self.outcome.as_str()),
            "invalid round outcome"
        );
        if self.outcome == "success" {
            ensure!(
                self.rtf().is_some(),
                "successful round requires valid timings and nonempty PCM"
            );
            ensure!(
                self.output_sha256
                    .as_ref()
                    .is_some_and(|h| h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit())),
                "successful round requires output digest"
            );
        }
        if let Some(first) = self.first_pcm_seconds {
            ensure!(
                first.is_finite()
                    && first >= 0.0
                    && self.synthesis_seconds.is_some_and(|total| first <= total),
                "invalid first PCM boundary"
            );
            ensure!(
                self.delivery_boundary
                    .as_ref()
                    .is_some_and(|boundary| !boundary.is_empty()),
                "successful first PCM requires a delivery boundary"
            );
        }
        for observation in &self.observations {
            ensure!(
                !observation.boundary.is_empty() && !observation.location.is_empty(),
                "observation boundary missing"
            );
            observation.reading.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attempt {
    pub schema: String,
    pub run_id: String,
    pub case_id: String,
    pub attempt_id: String,
    pub process_index: u32,
    pub source: Source,
    pub binary_sha256: String,
    pub conditions: Conditions,
    pub expected_hot_rounds: u32,
    pub rounds: Vec<Round>,
    pub outcome: String,
    pub cleanup_verified: bool,
    pub trace_complete: bool,
    pub dropped_trace_records: Option<u64>,
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
}

impl Attempt {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.schema == VERSION, "unsupported attempt schema");
        ensure!(
            !self.run_id.is_empty() && !self.case_id.is_empty() && !self.attempt_id.is_empty(),
            "attempt correlation missing"
        );
        ensure!(self.expected_hot_rounds > 0, "missing repetition budget");
        for (index, round) in self.rounds.iter().enumerate() {
            ensure!(round.index as usize == index, "round sequence gap");
            round.validate()?;
        }
        if self.outcome == "success" {
            ensure!(
                self.cleanup_verified && self.rounds.iter().all(|r| r.outcome == "success"),
                "attempt success contradicts terminal evidence"
            );
            ensure!(
                self.rounds.iter().filter(|r| !r.warmup).count()
                    == self.expected_hot_rounds as usize,
                "attempt hot-round coverage incomplete"
            );
        }
        Ok(())
    }

    pub fn verify_artifacts(&self, root: &Path) -> anyhow::Result<()> {
        self.validate()?;
        for round in &self.rounds {
            for artifact in &round.artifacts {
                artifact.verify(root)?;
            }
        }
        for artifact in &self.artifacts {
            artifact.verify(root)?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub schema: String,
    pub attempt_id: String,
    pub sequence: u64,
    pub offset_seconds: f64,
    pub kind: String,
    pub payload: serde_json::Value,
}

/// Finite stream verification state. The caller enforces frame and disk budgets.
#[derive(Default)]
pub struct EventSequence {
    next: u64,
    last_offset: f64,
    terminal: bool,
}

impl EventSequence {
    pub fn accept(&mut self, expected_attempt: &str, event: &Event) -> anyhow::Result<()> {
        ensure!(!self.terminal, "event after terminal");
        ensure!(
            event.schema == VERSION && event.attempt_id == expected_attempt,
            "event identity mismatch"
        );
        ensure!(event.sequence == self.next, "event sequence gap");
        ensure!(
            event.offset_seconds.is_finite() && event.offset_seconds >= self.last_offset,
            "invalid monotonic offset"
        );
        ensure!(
            ["prepared", "round", "observation", "terminal"].contains(&event.kind.as_str()),
            "unknown event kind"
        );
        self.terminal = event.kind == "terminal";
        self.next += 1;
        self.last_offset = event.offset_seconds;
        Ok(())
    }

    pub fn finish(self) -> anyhow::Result<()> {
        ensure!(
            self.terminal,
            "disconnected probe without terminal evidence"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mac_accelerate_thread_control_rejects_invalid_frozen_values() {
        let mut conditions = super::super::test_support::attempts(&[1.0])
            .remove(0)
            .conditions;
        conditions.os = "macos:fixture:build".into();
        for value in ["", "0", "-1", "abc", "1.5", "2147483648"] {
            conditions
                .environment
                .insert("VECLIB_MAXIMUM_THREADS".into(), Some(value.into()));
            assert!(conditions.validate().is_err(), "accepted {value:?}");
        }
        for value in [None, Some("1".into()), Some("12".into())] {
            conditions
                .environment
                .insert("VECLIB_MAXIMUM_THREADS".into(), value);
            assert!(conditions.validate().is_ok());
        }
    }
    #[test]
    fn thread_identity_requires_supported_explicit_effective_setting() {
        let mut attempts = super::super::test_support::attempts(&[1.0]);
        let conditions = &mut attempts[0].conditions;
        assert_eq!(conditions.rayon_threads().unwrap(), 1);
        for threads in [
            BTreeMap::new(),
            BTreeMap::from([("rayon".into(), 0)]),
            BTreeMap::from([("unknown".into(), 1)]),
        ] {
            conditions.threads = threads;
            assert!(conditions.validate().is_err());
        }
    }
    #[test]
    fn event_gaps_and_disconnects_are_not_success() {
        let mut sequence = EventSequence::default();
        let mut event = Event {
            schema: VERSION.into(),
            attempt_id: "a".into(),
            sequence: 1,
            offset_seconds: 0.0,
            kind: "round".into(),
            payload: serde_json::Value::Null,
        };
        assert!(sequence.accept("a", &event).is_err());
        event.sequence = 0;
        assert!(sequence.accept("a", &event).is_ok());
        assert!(sequence.finish().is_err());
        let mut sequence = EventSequence::default();
        event.kind = "terminal".into();
        sequence.accept("a", &event).unwrap();
        assert!(sequence.accept("a", &event).is_err());
        assert!(sequence.finish().is_ok());
    }
    #[test]
    fn rtf_uses_actual_audio_and_rejects_zero() {
        let mut round = Round {
            index: 0,
            warmup: false,
            outcome: "success".into(),
            synthesis_seconds: Some(3.0),
            audio_seconds: Some(6.0),
            first_pcm_seconds: Some(1.0),
            delivery_boundary: Some("native_streaming_first_pcm".into()),
            output_sha256: Some("a".repeat(64)),
            observations: vec![],
            artifacts: vec![],
        };
        assert_eq!(round.rtf(), Some(0.5));
        round.validate().unwrap();
        round.audio_seconds = Some(0.0);
        assert!(round.validate().is_err());
        round.audio_seconds = Some(6.0);
        round.first_pcm_seconds = Some(4.0);
        assert!(round.validate().is_err());
    }
}
