//! Platform-scoped plans; a catalogue row is never proof of runtime support.
use super::evidence::{Source, VERSION};
use anyhow::ensure;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Platform {
    Windows,
    Macos,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema: String,
    pub platform: Platform,
    pub machine: String,
    pub source: Source,
    pub processes: u32,
    pub warmup_rounds: u32,
    pub hot_rounds: u32,
    pub observation: String,
    pub cases: Vec<Case>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub backend: String,
    pub model: String,
    pub device: String,
    pub math_backend: String,
    pub status: String,
    pub reason: String,
}

impl Platform {
    pub fn host() -> anyhow::Result<Self> {
        match std::env::consts::OS {
            "windows" => Ok(Self::Windows),
            "macos" => Ok(Self::Macos),
            _ => anyhow::bail!("native execution unavailable on this platform"),
        }
    }
}

impl Plan {
    pub fn apply_catalog(&mut self, catalog: &super::prepare::Catalog) -> anyhow::Result<()> {
        catalog.validate()?;
        for case in &mut self.cases {
            if case.status == "not_applicable" {
                continue;
            }
            let entry = catalog
                .catalog
                .iter()
                .find(|entry| entry["backend"] == case.backend && entry["model"] == case.model);
            let compiled = entry
                .and_then(|entry| entry["compiled_devices"].as_array())
                .is_some_and(|devices| devices.iter().any(|device| device == &case.device));
            if !compiled {
                case.status = "missing_feature".into();
                case.reason =
                    "model/device absent from compiled production catalogue; no fallback".into();
            }
        }
        Ok(())
    }
    pub fn new(platform: Platform, machine: String) -> anyhow::Result<Self> {
        ensure!(!machine.trim().is_empty(), "machine identifier is required");
        let mut cases = Vec::new();
        let devices = match platform {
            Platform::Windows => ["cpu", "cuda"],
            Platform::Macos => ["cpu", "metal"],
        };
        for (backend, models) in [
            (
                "moss",
                &["nano-candle", "nano", "local-1.7b", "realtime-1.7b"][..],
            ),
            (
                "qwen",
                &[
                    "0.6b-base",
                    "1.7b-base",
                    "0.6b-customvoice",
                    "1.7b-customvoice",
                    "1.7b-voicedesign",
                ][..],
            ),
            ("voxcpm", &["2b-q8_0", "2b-bf16"][..]),
            ("omnivoice", &["0.6b"][..]),
            ("cosyvoice3", &["0.5b"][..]),
            ("indextts", &["2.5"][..]),
        ] {
            for model in models {
                for device in devices {
                    let unsupported = (device == "cpu"
                        && ["local-1.7b", "realtime-1.7b", "2b-bf16"].contains(model))
                        || (device == "metal" && *model == "2b-bf16");
                    let (status, reason) = if unsupported {
                        (
                            "not_applicable",
                            "product does not expose this device/model combination",
                        )
                    } else {
                        (
                            "unverified",
                            "requires compiled probe catalog, prepared resources and device verification",
                        )
                    };
                    cases.push(Case {
                        backend: backend.into(),
                        model: (*model).into(),
                        device: device.into(),
                        math_backend: if platform == Platform::Macos && device == "cpu" {
                            "unverified_accelerate_or_rust"
                        } else {
                            "unverified"
                        }
                        .into(),
                        status: status.into(),
                        reason: reason.into(),
                    });
                }
            }
        }
        Ok(Self {
            schema: VERSION.into(),
            platform,
            machine,
            source: Source::capture(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .unwrap(),
            )?,
            processes: 3,
            warmup_rounds: 1,
            hot_rounds: 5,
            observation: "baseline".into(),
            cases,
        })
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.schema == VERSION,
            "unsupported performance schema: {}",
            self.schema
        );
        ensure!(
            !self.machine.trim().is_empty(),
            "machine identifier missing"
        );
        ensure!(
            self.processes > 0 && self.hot_rounds > 0,
            "invalid repetitions"
        );
        ensure!(
            ["off", "baseline", "trace", "profile"].contains(&self.observation.as_str()),
            "unknown observation mode"
        );
        for case in &self.cases {
            ensure!(
                [
                    "not_applicable",
                    "deferred",
                    "unverified",
                    "missing_feature",
                    "ready",
                    "blocked"
                ]
                .contains(&case.status.as_str()),
                "unknown case status"
            );
            ensure!(!case.reason.is_empty(), "case requires evidence or reason");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn windows_matrix_keeps_all_combinations_without_claiming_readiness() -> anyhow::Result<()> {
        let plan = Plan::new(Platform::Windows, "fixture".into())?;
        plan.validate()?;
        assert_eq!(plan.cases.len(), 28);
        assert_eq!(
            plan.cases
                .iter()
                .filter(|c| c.status == "not_applicable")
                .count(),
            3
        );
        assert_eq!(
            plan.cases
                .iter()
                .filter(|c| c.status == "unverified")
                .count(),
            25
        );
        Ok(())
    }
    #[test]
    fn mac_plan_keeps_models_unverified_and_unknown_schema_is_rejected() -> anyhow::Result<()> {
        let mut plan = Plan::new(Platform::Macos, "mac-fixture".into())?;
        plan.validate()?;
        assert_eq!(plan.cases.len(), 28);
        assert!(
            plan.cases
                .iter()
                .all(|c| c.status == "unverified" || c.status == "not_applicable")
        );
        assert!(
            plan.cases
                .iter()
                .filter(|c| c.device == "cpu")
                .all(|c| c.math_backend.contains("accelerate"))
        );
        plan.cases[0].status = "ready".into();
        assert!(plan.validate().is_ok());
        plan.cases[0].status = "missing_feature".into();
        assert!(plan.validate().is_ok());
        plan.schema = "talechime/performance/999".into();
        assert!(plan.validate().is_err());
        Ok(())
    }
    #[test]
    fn missing_features_keep_the_frozen_matrix() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let mut plan = Plan::new(Platform::Windows, "fixture".into())?;
        let catalog = super::super::prepare::Catalog {
            schema: "talechime/performance-catalog/1".into(),
            root: root.path().into(),
            resources: root.path().join("resources"),
            catalog: vec![],
            path_resolution: "pure_no_migration".into(),
            resource_acquisition: false,
        };
        plan.apply_catalog(&catalog)?;
        let mut mac = Plan::new(Platform::Macos, "mac-fixture".into())?;
        mac.apply_catalog(&catalog)?;
        assert_eq!(mac.cases.len(), 28);
        assert_eq!(
            mac.cases
                .iter()
                .filter(|c| c.status == "missing_feature")
                .count(),
            24
        );
        assert_eq!(plan.cases.len(), 28);
        assert_eq!(
            plan.cases
                .iter()
                .filter(|c| c.status == "missing_feature")
                .count(),
            25
        );
        Ok(())
    }
}
