//! Host-supplied generation parameters and the seed policy for one execution.
//!
//! Parameters cross the [`crate::backend::Backend`] boundary as a validated
//! name-to-value map; typed per-model builders live in talechime-backends.
//! Nothing is silently ignored: parameters the prepared model did not declare,
//! wrong types and out-of-range values are explicit session-start errors.

use std::collections::BTreeMap;
use tts_protocol::{ParamValue, ParameterSpec};

/// Generation parameters for one execution, validated against the prepared
/// model's declared catalog before any synthesis starts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationParams(BTreeMap<String, ParamValue>);

impl GenerationParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, value: ParamValue) {
        self.0.insert(name.into(), value);
    }

    pub fn get(&self, name: &str) -> Option<&ParamValue> {
        self.0.get(name)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &ParamValue)> {
        self.0.iter().map(|(name, value)| (name.as_str(), value))
    }

    /// Explicit validation: unknown parameters, type mismatches and
    /// out-of-range values are errors naming the accepted catalog.
    pub fn validate(&self, specs: &[ParameterSpec]) -> Result<(), String> {
        let mut errors = Vec::new();
        for (name, value) in &self.0 {
            let Some(spec) = specs.iter().find(|spec| spec.name == *name) else {
                let catalog = if specs.is_empty() {
                    "none".to_string()
                } else {
                    specs
                        .iter()
                        .map(|spec| spec.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                errors.push(format!(
                    "unknown generation parameter '{name}'; this model declares: {catalog}"
                ));
                continue;
            };
            if let Err(error) = spec.validate(value) {
                errors.push(error);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}

/// Sampling seed policy owned by the session producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SeedPolicy {
    /// Fresh randomness for every synthesis attempt, including gated retries.
    #[default]
    Auto,
    /// Deterministically derived per synthesis call and attempt so probe runs
    /// reproduce while gated retries can still escape a sampling error.
    Pinned(u64),
}

impl SeedPolicy {
    /// Resolve the concrete seed for one synthesis call. `call` counts every
    /// backend synthesis request in this execution; `attempt` is the gated
    /// retry count of the current segment.
    pub fn resolve(self, call: u64, attempt: u8) -> u64 {
        match self {
            SeedPolicy::Auto => rand::random(),
            SeedPolicy::Pinned(seed) => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::hash::DefaultHasher::new();
                seed.hash(&mut hasher);
                call.hash(&mut hasher);
                attempt.hash(&mut hasher);
                hasher.finish()
            }
        }
    }
}

impl From<Option<u64>> for SeedPolicy {
    fn from(seed: Option<u64>) -> Self {
        seed.map_or(SeedPolicy::Auto, SeedPolicy::Pinned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tts_protocol::ParamKind;

    fn int_spec(name: &str, min: i64, max: i64) -> ParameterSpec {
        ParameterSpec {
            name: name.into(),
            kind: ParamKind::Int { min, max },
            default: ParamValue::Int(min),
            description: String::new(),
        }
    }

    #[test]
    fn validation_names_the_catalog_and_range() {
        let specs = vec![int_spec("steps", 2, 128)];
        let mut params = GenerationParams::new();
        params.insert("steps", ParamValue::Int(10));
        assert!(params.validate(&specs).is_ok());
        params.insert("steps", ParamValue::Int(200));
        assert_eq!(
            params.validate(&specs).unwrap_err(),
            "parameter 'steps' expects int in 2..=128, got int (200)"
        );
        params.insert("steps", ParamValue::Int(64));
        params.insert("temperature", ParamValue::Float(0.9));
        let error = params.validate(&specs).unwrap_err();
        assert!(
            error.contains("unknown generation parameter 'temperature'"),
            "{error}"
        );
        assert!(error.contains("steps"), "{error}");
    }

    #[test]
    fn empty_catalog_rejects_everything() {
        let mut params = GenerationParams::new();
        params.insert("steps", ParamValue::Int(10));
        assert_eq!(
            params.validate(&[]).unwrap_err(),
            "unknown generation parameter 'steps'; this model declares: none"
        );
    }

    #[test]
    fn pinned_resolution_reproduces_and_varies() {
        let policy = SeedPolicy::Pinned(7);
        assert_eq!(policy.resolve(3, 0), policy.resolve(3, 0));
        assert_ne!(policy.resolve(3, 0), policy.resolve(4, 0));
        assert_ne!(policy.resolve(3, 0), policy.resolve(3, 1));
        assert_ne!(
            SeedPolicy::Pinned(7).resolve(1, 0),
            SeedPolicy::Pinned(8).resolve(1, 0)
        );
        assert_eq!(SeedPolicy::from(None), SeedPolicy::Auto);
        assert_eq!(SeedPolicy::from(Some(9)), SeedPolicy::Pinned(9));
    }
}
