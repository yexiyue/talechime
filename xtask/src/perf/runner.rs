//! Serial attempts for a frozen case; raw results are committed only after audit.
use super::{
    build::Build,
    evidence::{VERSION, contained, digest, file_digest, publish_bytes, publish_json, read_json},
    executor::{Budget, Capture, execute},
    plan::{Plan, Platform},
    records::{Artifact, Attempt, Conditions, Event, Round},
    storage::{RunLock, commit, new_attempt, recover},
};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub schema: String,
    pub plan: PathBuf,
    pub build: PathBuf,
    pub case_id: String,
    pub conditions: Conditions,
    pub request: serde_json::Value,
    pub timeout_seconds: u64,
    pub conditioning_files: Vec<Artifact>,
    #[serde(default)]
    pub budget: Budget,
}

/// Input changes always require a different run directory, including during resume.
pub fn run(job_path: &Path, root: &Path, resume: bool) -> anyhow::Result<PathBuf> {
    ensure!(
        cfg!(any(windows, target_os = "macos")),
        "deferred: native lifecycle has not been implemented on this platform"
    );
    let job: Job = read_json(job_path)?;
    ensure!(
        job.schema == VERSION && !job.case_id.is_empty(),
        "invalid job identity"
    );
    ensure!(
        job.timeout_seconds > 0 && job.timeout_seconds <= 14400,
        "process timeout must be within four hours"
    );
    job.conditions.validate()?;
    let rayon_threads = job.conditions.rayon_threads()?;
    let environment = super::environment::capture_environment(&job.conditions.provider)?;
    ensure!(
        environment.machine == job.conditions.machine
            && environment.os == job.conditions.os
            && environment.arch == job.conditions.arch
            && environment.hardware == job.conditions.hardware
            && environment.driver == job.conditions.driver
            && environment.flags == job.conditions.environment,
        "actual machine/driver/performance flags differ from frozen conditions"
    );
    ensure!(
        job.conditions.observation_config.get("capture_budget")
            == Some(&serde_json::to_string(&job.budget)?),
        "capture configuration differs from frozen observation identity"
    );
    let plan: Plan = read_json(&job.plan)?;
    plan.validate()?;
    ensure!(
        plan.platform == Platform::host()? && plan.machine == job.conditions.machine,
        "job machine/platform differs from frozen plan"
    );
    let build: Build = read_json(&job.build)?;
    build.verify()?;
    ensure!(
        build.probe == "performance_probe",
        "run requires a measurement probe"
    );
    ensure!(
        build.source == plan.source,
        "plan and build source identities differ; recreate the plan for this build"
    );
    ensure!(
        job.conditions.features == build.features,
        "effective build features differ from conditions"
    );
    ensure!(
        plan.observation == job.conditions.observation,
        "observation profile mismatch"
    );
    ensure!(
        job.request["backend"] == job.conditions.backend
            && job.request["model"] == job.conditions.model
            && job.request["device"] == job.conditions.provider,
        "request/backend/device identity mismatch"
    );
    let fixture = job.request["fixture"] == true;
    #[cfg(target_os = "macos")]
    if !fixture {
        let info: serde_json::Value = serde_json::from_slice(&super::prepare::capture(
            Command::new(&build.executable).arg("--execution-info"),
            Duration::from_secs(10),
        )?)?;
        validate_mac_execution_info(&info, &job.conditions)?;
    }
    let resource_root = PathBuf::from(
        job.request["resources_root"]
            .as_str()
            .context("resources root missing")?,
    );
    if job.request["voice"]
        .as_str()
        .is_some_and(|v| v.starts_with("custom:"))
    {
        ensure!(
            !job.conditioning_files.is_empty(),
            "custom voice requires frozen reference/metadata receipts"
        );
    }
    let conditioning_identity = digest(&serde_json::to_vec(
        &serde_json::json!({"voice":job.request["voice"],"style":job.request["style"],"files":job.conditioning_files}),
    )?);
    ensure!(
        conditioning_identity == job.conditions.conditioning_sha256,
        "conditioning identity mismatch"
    );
    if fixture {
        ensure!(
            job.conditions.backend == "fixture"
                && job.conditions.model == "deterministic"
                && job.conditions.provider == "cpu",
            "fixture cannot impersonate production evidence"
        );
    } else {
        let case = plan
            .cases
            .iter()
            .find(|c| {
                c.backend == job.conditions.backend
                    && c.model == job.conditions.model
                    && c.device == job.conditions.provider
            })
            .context("case absent from frozen plan")?;
        ensure!(
            ["ready", "unverified"].contains(&case.status.as_str()),
            "case is {}: {}",
            case.status,
            case.reason
        );
    }
    let corpus_path = PathBuf::from(
        job.request["corpus"]
            .as_str()
            .context("corpus path missing")?,
    )
    .canonicalize()?;
    ensure!(
        file_digest(&corpus_path)? == job.conditions.corpus_sha256
            && job.request["corpus_sha256"] == job.conditions.corpus_sha256,
        "corpus identity mismatch"
    );
    let identity = digest(&serde_json::to_vec(
        &serde_json::json!({"job":job,"plan":plan,"build":build}),
    )?);
    let _lock = RunLock::acquire(root)?;
    let root = root.canonicalize()?;
    let run_file = root.join("run.json");
    let corpus_snapshot = root.join("corpus.txt");
    if resume {
        let existing: serde_json::Value = read_json(&run_file)?;
        ensure!(
            existing["schema"] == VERSION && existing["identity"] == identity,
            "resume inputs changed; choose a new run"
        );
        ensure!(
            file_digest(&corpus_snapshot)? == job.conditions.corpus_sha256,
            "immutable corpus snapshot missing or changed"
        );
        verify_previous_cleanup(&root)?;
    } else {
        publish_bytes(&corpus_snapshot, &std::fs::read(&corpus_path)?)?;
        publish_json(
            &run_file,
            &serde_json::json!({"schema":VERSION,"identity":identity,"plan":plan,"job":job,"build":build,"listening_acceptance":"pending"}),
        )?;
    }
    let mut results = Vec::new();
    for process_index in 0..plan.processes {
        if super::cancellation::requested() {
            break;
        }
        for artifact in &job.conditioning_files {
            artifact.verify(&resource_root)?;
        }
        let mut expected = Attempt {
            schema: VERSION.into(),
            run_id: identity.clone(),
            case_id: job.case_id.clone(),
            attempt_id: String::new(),
            process_index,
            source: build.source.clone(),
            binary_sha256: build.binary_sha256.clone(),
            conditions: job.conditions.clone(),
            expected_hot_rounds: plan.hot_rounds,
            rounds: vec![],
            outcome: "failure".into(),
            cleanup_verified: false,
            trace_complete: false,
            dropped_trace_records: None,
            artifacts: vec![],
        };
        ensure!(
            super::environment::flags() == job.conditions.environment,
            "performance flags changed between processes"
        );
        if resume && let Some(saved) = find_committed(&root, &expected)? {
            results.push(saved);
            continue;
        }
        build.verify()?;
        ensure!(
            file_digest(&corpus_path)? == job.conditions.corpus_sha256,
            "corpus changed between processes"
        );
        let directory = new_attempt(&root)?;
        expected.attempt_id = directory
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let mut request = job.request.clone();
        request["schema"] = "talechime/performance-request/1".into();
        request["attempt_id"] = expected.attempt_id.clone().into();
        request["warmup_rounds"] = plan.warmup_rounds.into();
        request["hot_rounds"] = plan.hot_rounds.into();
        request["observation_mode"] = job.conditions.observation.clone().into();
        request["corpus"] = corpus_snapshot.to_string_lossy().to_string().into();
        let request_path = directory.join("request.json");
        publish_json(&request_path, &request)?;
        let mut command = Command::new(&build.executable);
        command.arg(&request_path).arg(&directory);
        command.env("RAYON_NUM_THREADS", rayon_threads.to_string());
        #[cfg(target_os = "macos")]
        if let Some(Some(value)) = job.conditions.environment.get("VECLIB_MAXIMUM_THREADS") {
            command.env("VECLIB_MAXIMUM_THREADS", value);
        }
        let capture = execute(
            &mut command,
            &directory,
            &expected.attempt_id,
            Duration::from_secs(job.timeout_seconds),
            job.budget,
            &super::cancellation::REQUESTED,
        )?;
        expected.outcome = capture.outcome.clone();
        expected.cleanup_verified = capture.cleanup_verified;
        let mut terminal_success = false;
        let mut preflight_valid = false;
        let mut preparation = None;
        for line in BufReader::new(std::fs::File::open(directory.join("stdout.jsonl"))?).lines() {
            let line = line?;
            let Ok(event) = serde_json::from_str::<Event>(&line) else {
                break;
            };
            if event.kind == "round" {
                match serde_json::from_value::<Round>(event.payload) {
                    Ok(round) => expected.rounds.push(round),
                    Err(_) => {
                        expected.outcome = "failure".into();
                        break;
                    }
                }
            } else if event.kind == "prepared" {
                preparation = Some(event.payload.clone());
                preflight_valid = digest(&serde_json::to_vec(&event.payload["resource_receipt"])?)
                    == job.conditions.resource_sha256;
                preflight_valid &= job.conditions.mode
                    == if event.payload["capabilities"]["native_streaming"] == true {
                        "native_stream"
                    } else {
                        "segment_delivery"
                    };
                let mut effective = std::collections::BTreeMap::new();
                if let Some(defaults) = event.payload["capabilities"]["parameters"].as_array() {
                    for parameter in defaults {
                        let name = parameter["name"]
                            .as_str()
                            .context("parameter name missing")?;
                        effective.insert(
                            format!("generation.{name}"),
                            parameter["default"].to_string(),
                        );
                    }
                }
                if let Some(parameters) = job.request["params"].as_object() {
                    for (name, value) in parameters {
                        effective.insert(format!("generation.{name}"), value.to_string());
                    }
                }
                for name in ["voice", "style", "seed", "continuation"] {
                    effective.insert(format!("request.{name}"), job.request[name].to_string());
                }
                preflight_valid &= effective == job.conditions.parameters;
            } else if event.kind == "terminal" {
                terminal_success = event.payload["outcome"] == "success";
                if event.payload["outcome"] == "timeout" && expected.outcome != "cancelled" {
                    expected.outcome = "timeout".into();
                }
            }
        }
        if expected.outcome == "success" && (!terminal_success || !preflight_valid) {
            expected.outcome = "failure".into();
        }
        if let Some(preparation) = preparation {
            publish_json(&directory.join("preparation.json"), &preparation)?;
        }
        for (name, kind) in [
            ("preparation.json", "preparation"),
            ("metrics.json", "metrics"),
            ("diagnostics.json", "diagnostic_state"),
            ("trace.json", "diagnostic_trace"),
        ] {
            let path = directory.join(name);
            if path.is_file() {
                expected.artifacts.push(Artifact {
                    path: name.into(),
                    sha256: file_digest(&path)?,
                    bytes: path.metadata()?.len(),
                    kind: kind.into(),
                    derived_from: None,
                });
            }
        }
        if directory.join("diagnostics.json").is_file() {
            let diagnostics: serde_json::Value = read_json(&directory.join("diagnostics.json"))?;
            expected.trace_complete = diagnostics["trace_complete"].as_bool().unwrap_or(false);
            expected.dropped_trace_records = diagnostics["dropped_trace_records"].as_u64();
        }
        if super::cancellation::requested() {
            expected.outcome = "cancelled".into();
        }
        if expected.outcome == "success" {
            commit(&directory, &expected)?;
        } else {
            publish_json(&directory.join("attempt.json"), &expected)?;
            publish_json(
                &directory.join("failure.json"),
                &serde_json::json!({"preflight_identity_verified":preflight_valid,"terminal_success":terminal_success,"capture":capture}),
            )?;
        }
        let cleanup_verified = expected.cleanup_verified;
        results.push(expected);
        ensure!(
            cleanup_verified,
            "cleanup unverified; partial evidence retained; refuse next measurement"
        );
    }
    let suffix = if resume {
        format!(
            "results-resumed-{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        )
    } else {
        "results.json".into()
    };
    let results_path = root.join(suffix);
    publish_json(&results_path, &results)?;
    ensure!(
        !super::cancellation::requested(),
        "run cancelled; partial results retained at {}",
        results_path.display()
    );
    Ok(results_path)
}

fn attempt_directories(root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let directory = contained(root, Path::new("attempts"))?;
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut attempts = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = contained(root, &Path::new("attempts").join(entry.file_name()))?;
        if path.is_dir() {
            attempts.push(path);
        }
    }
    attempts.sort();
    Ok(attempts)
}

fn verify_previous_cleanup(root: &Path) -> anyhow::Result<()> {
    for attempt in attempt_directories(root)? {
        let capture: Capture = read_json(&contained(&attempt, Path::new("capture.json"))?)
            .with_context(|| {
                format!(
                    "cleanup unverified: prior attempt {} has no valid capture; refuse resume",
                    attempt.display()
                )
            })?;
        ensure!(
            capture.cleanup_verified,
            "cleanup unverified: prior attempt {}; refuse resume until cleanup is independently verified",
            attempt.display()
        );
    }
    Ok(())
}

fn find_committed(root: &Path, expected: &Attempt) -> anyhow::Result<Option<Attempt>> {
    for attempt in attempt_directories(root)? {
        if attempt.join("committed.json").is_file()
            && let Ok(saved) = recover(&attempt, expected)
        {
            return Ok(Some(saved));
        }
    }
    Ok(None)
}

#[cfg(target_os = "macos")]
fn validate_mac_execution_info(
    info: &serde_json::Value,
    conditions: &Conditions,
) -> anyhow::Result<()> {
    ensure!(
        info["schema"] == "talechime/performance-execution/1"
            && info["os"] == "macos"
            && info["arch"] == conditions.arch,
        "probe execution metadata mismatch"
    );
    let math = match conditions.provider.as_str() {
        "cpu" => &info["cpu_math"],
        "metal" => &info["metal_math"],
        _ => anyhow::bail!("unsupported Mac provider; no fallback"),
    };
    ensure!(
        math == &conditions.math_backend && info["timing_contract"] == conditions.timing_contract,
        "effective probe math/timing contract differs from frozen conditions"
    );
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod mac_tests {
    #[test]
    fn mismatched_math_build_and_provider_are_rejected() -> anyhow::Result<()> {
        let mut conditions = super::super::test_support::attempts(&[1.0])
            .remove(0)
            .conditions;
        let info = serde_json::json!({"schema":"talechime/performance-execution/1", "os":"macos", "arch":conditions.arch,
            "cpu_math":"build:candle-rust", "metal_math":"build:candle-metal", "timing_contract":conditions.timing_contract});
        assert!(super::validate_mac_execution_info(&info, &conditions).is_err());
        conditions.math_backend = "build:candle-rust".into();
        super::validate_mac_execution_info(&info, &conditions)?;
        conditions.provider = "metal".into();
        assert!(super::validate_mac_execution_info(&info, &conditions).is_err());
        conditions.provider = "cuda".into();
        assert!(super::validate_mac_execution_info(&info, &conditions).is_err());
        Ok(())
    }
}
