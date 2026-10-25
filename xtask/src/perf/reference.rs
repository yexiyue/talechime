//! Explicit reference adapters share native containment without impersonating production probes.
use super::{
    evidence::{VERSION, file_digest, publish_json, read_json},
    executor::{Budget, execute},
    records::{Artifact, Event, Round},
    storage::RunLock,
};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    path: PathBuf,
    sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Job {
    schema: String,
    executable: PathBuf,
    executable_sha256: String,
    arguments: Vec<String>,
    inputs: Vec<Input>,
    environment: BTreeMap<String, String>,
    request: serde_json::Value,
    processes: u32,
    warmup_rounds: u32,
    hot_rounds: u32,
    timeout_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    minimum_available_commit_bytes: Option<u64>,
}

fn commit_preflight(directory: &Path, minimum: Option<u64>) -> anyhow::Result<()> {
    let Some(minimum) = minimum else {
        return Ok(());
    };
    #[cfg(windows)]
    let observed = super::sampling::system_available_private_bytes();
    #[cfg(not(windows))]
    let observed: anyhow::Result<u64> = Err(anyhow::anyhow!(
        "explicit commit budget preflight is currently Windows-only"
    ));
    let (available, reason) = match observed {
        Ok(bytes) => (Some(bytes), None),
        Err(error) => (None, Some(error.to_string())),
    };
    let ready = available.is_some_and(|bytes| bytes >= minimum);
    publish_json(
        &directory.join("commit-preflight.json"),
        &serde_json::json!({
            "scope":"system_commit", "unit":"bytes", "source":"native commit availability",
            "minimum_available_commit_bytes":minimum, "available_commit_bytes":available,
            "status":if ready { "ready" } else { "blocked" }, "reason":reason,
            "policy":"explicit per-job budget; snapshot only, not a reservation or runtime limit"
        }),
    )?;
    ensure!(
        ready,
        "reference commit budget unmet: available={available:?}, required={minimum}; no reference process started"
    );
    Ok(())
}

impl Job {
    fn verify_inputs(&self) -> anyhow::Result<()> {
        ensure!(
            file_digest(&self.executable)? == self.executable_sha256,
            "reference interpreter changed"
        );
        for input in &self.inputs {
            ensure!(
                file_digest(&input.path)? == input.sha256,
                "reference input changed: {}",
                input.path.display()
            );
        }
        Ok(())
    }
}

fn rounds(path: &Path, root: &Path, warmup: u32, hot: u32) -> anyhow::Result<Vec<Round>> {
    let mut result = Vec::new();
    let mut terminal = false;
    for line in BufReader::new(std::fs::File::open(path)?).lines() {
        let event: Event = serde_json::from_str(&line?)?;
        if event.kind == "round" {
            let round: Round = serde_json::from_value(event.payload)?;
            round.validate()?;
            ensure!(
                round.index as usize == result.len(),
                "reference round sequence mismatch"
            );
            ensure!(
                round.warmup == (round.index < warmup),
                "reference warmup mismatch"
            );
            ensure!(round.outcome == "success", "reference round failed");
            ensure!(round.output_sha256.is_some(), "reference PCM hash missing");
            ensure!(
                !round.artifacts.is_empty(),
                "reference audio evidence missing"
            );
            for artifact in &round.artifacts {
                artifact.verify(root)?;
            }
            result.push(round);
        } else if event.kind == "terminal" {
            terminal = event.payload["outcome"] == "success";
        }
    }
    ensure!(
        terminal && result.len() == (warmup + hot) as usize,
        "reference terminal/coverage incomplete"
    );
    Ok(result)
}

pub fn run(job_path: &Path, output: &Path) -> anyhow::Result<()> {
    run_mode(job_path, output, false)
}

pub fn run_diagnostic(job_path: &Path, output: &Path) -> anyhow::Result<()> {
    run_mode(job_path, output, true)
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Diagnostic {
    summary: serde_json::Value,
    artifacts: Vec<Artifact>,
}

fn diagnostic_result(path: &Path, root: &Path) -> anyhow::Result<Diagnostic> {
    let mut result = None;
    let mut terminal = false;
    for line in BufReader::new(std::fs::File::open(path)?).lines() {
        let event: Event = serde_json::from_str(&line?)?;
        ensure!(
            event.kind != "round",
            "diagnostic must not emit TTS timing rounds"
        );
        if event.kind == "observation" && event.payload.get("diagnostic_result").is_some() {
            ensure!(
                result.is_none() && !terminal,
                "duplicate or late diagnostic result"
            );
            let diagnostic: Diagnostic =
                serde_json::from_value(event.payload["diagnostic_result"].clone())?;
            ensure!(
                diagnostic.summary.is_object() && !diagnostic.artifacts.is_empty(),
                "diagnostic requires a structured summary and artifacts"
            );
            for artifact in &diagnostic.artifacts {
                ensure!(
                    artifact.kind.starts_with("diagnostic_"),
                    "diagnostic artifact kind required"
                );
                artifact.verify(root)?;
            }
            result = Some(diagnostic);
        } else if event.kind == "terminal" {
            ensure!(
                !terminal && result.is_some(),
                "diagnostic result must precede terminal"
            );
            ensure!(
                event.payload["outcome"] == "success",
                "diagnostic terminal failed"
            );
            terminal = true;
        }
    }
    ensure!(terminal, "diagnostic terminal missing");
    result.context("diagnostic result missing")
}

fn run_mode(job_path: &Path, output: &Path, diagnostic: bool) -> anyhow::Result<()> {
    let job: Job = read_json(job_path)?;
    ensure!(job.schema == VERSION, "unsupported reference job schema");
    ensure!(
        job.minimum_available_commit_bytes != Some(0),
        "zero commit budget is invalid"
    );
    ensure!(
        (1..=8).contains(&job.processes)
            && job.warmup_rounds <= 10
            && (1..=100).contains(&job.hot_rounds),
        "invalid reference repetitions"
    );
    ensure!(
        (1..=14400).contains(&job.timeout_seconds),
        "invalid reference timeout"
    );
    ensure!(
        !diagnostic || (job.warmup_rounds == 0 && job.hot_rounds == 1),
        "diagnostic requires zero protocol warmups and one invocation per process"
    );
    ensure!(
        !job.inputs.is_empty(),
        "reference source/input identities required"
    );
    ensure!(
        !output.exists(),
        "reference output already exists; retain prior evidence"
    );
    let _lock = RunLock::acquire(output)?;
    let output = output.canonicalize()?;
    publish_json(&output.join("job.json"), &job)?;
    let mut summaries = Vec::new();
    for index in 0..job.processes {
        ensure!(!super::cancellation::requested(), "reference cancelled");
        job.verify_inputs()?;
        let directory = output.join(format!("process-{index}"));
        std::fs::create_dir(&directory)?;
        commit_preflight(&directory, job.minimum_available_commit_bytes)?;
        if index == 0 {
            publish_json(
                &output.join("environment.json"),
                &super::environment::capture_environment("cuda")?,
            )?;
        }
        let attempt = format!("reference-{index}");
        let request = serde_json::json!({"attempt_id":attempt,"warmup_rounds":job.warmup_rounds,"hot_rounds":job.hot_rounds,"request":job.request});
        let request_path = directory.join("request.json");
        publish_json(&request_path, &request)?;
        let mut command = Command::new(&job.executable);
        command
            .args(&job.arguments)
            .arg(&request_path)
            .arg(&directory)
            .envs(&job.environment);
        let capture = execute(
            &mut command,
            &directory,
            &attempt,
            Duration::from_secs(job.timeout_seconds),
            Budget {
                device_sampling: true,
                ..Budget::default()
            },
            &super::cancellation::REQUESTED,
        )?;
        ensure!(
            capture.outcome == "success" && capture.cleanup_verified,
            "reference failed; retained at {}",
            directory.display()
        );
        if diagnostic {
            let measured = diagnostic_result(&directory.join("stdout.jsonl"), &directory)?;
            job.verify_inputs()?;
            publish_json(&directory.join("diagnostic-result.json"), &measured)?;
            summaries.push(
                serde_json::json!({"process":index,"diagnostic":measured,"cleanup_verified":true}),
            );
            continue;
        }
        let measured = rounds(
            &directory.join("stdout.jsonl"),
            &directory,
            job.warmup_rounds,
            job.hot_rounds,
        )
        .context("reference result audit failed")?;
        job.verify_inputs()?;
        publish_json(&directory.join("rounds.json"), &measured)?;
        summaries
            .push(serde_json::json!({"process":index,"rounds":measured,"cleanup_verified":true}));
    }
    publish_json(
        &output.join("results.json"),
        &serde_json::json!({"schema":VERSION,"kind":if diagnostic {"external_diagnostic_not_tts_performance"} else {"external_reference_not_production"},"listening_acceptance":if diagnostic {"not_applicable"} else {"pending"},"processes":summaries}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(windows)]
    #[test]
    fn commit_budget_blocks_before_reference_process_creation() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let executable = std::env::current_exe()?;
        let source = root.path().join("source.txt");
        std::fs::write(&source, b"fixture input")?;
        let mut job = Job {
            schema: VERSION.into(),
            executable_sha256: file_digest(&executable)?,
            executable,
            arguments: vec!["must-not-launch".into()],
            inputs: vec![Input {
                sha256: file_digest(&source)?,
                path: source,
            }],
            environment: BTreeMap::new(),
            request: serde_json::json!({"fixture":true}),
            processes: 1,
            warmup_rounds: 0,
            hot_rounds: 1,
            timeout_seconds: 5,
            minimum_available_commit_bytes: None,
        };
        let legacy = serde_json::to_value(&job)?;
        assert!(legacy.get("minimum_available_commit_bytes").is_none());
        assert!(
            serde_json::from_value::<Job>(legacy)?
                .minimum_available_commit_bytes
                .is_none()
        );
        job.minimum_available_commit_bytes = Some(u64::MAX);
        let path = root.path().join("job.json");
        publish_json(&path, &job)?;
        let output = root.path().join("run");
        let error = run(&path, &output).unwrap_err();
        assert!(
            error.to_string().contains("commit budget unmet"),
            "{error:#}"
        );
        let check: serde_json::Value = read_json(&output.join("process-0/commit-preflight.json"))?;
        assert_eq!(check["status"], "blocked");
        assert!(check["available_commit_bytes"].as_u64().is_some());
        assert!(!output.join("process-0/request.json").exists());
        assert!(!output.join("process-0/stdout.jsonl").exists());
        assert!(!output.join("results.json").exists());
        let ready = root.path().join("ready");
        std::fs::create_dir(&ready)?;
        commit_preflight(&ready, Some(1))?;
        let check: serde_json::Value = read_json(&ready.join("commit-preflight.json"))?;
        assert_eq!(check["status"], "ready");
        Ok(())
    }

    #[test]
    fn reference_audit_rejects_changed_audio_and_wrong_warmup() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let audio = root.path().join("audio.wav");
        std::fs::write(&audio, b"fixture audio bytes")?;
        let round = serde_json::json!({"index":0,"warmup":false,"outcome":"success","synthesis_seconds":0.5,"audio_seconds":1.0,"first_pcm_seconds":0.5,"delivery_boundary":"segment_delivery_first_pcm","output_sha256":file_digest(&audio)?,"observations":[],"artifacts":[{"path":"audio.wav","sha256":file_digest(&audio)?,"bytes":std::fs::metadata(&audio)?.len(),"kind":"audio_f32","derived_from":null}]});
        let frame = |sequence, kind, payload| serde_json::json!({"schema":VERSION,"attempt_id":"reference-0","sequence":sequence,"offset_seconds":sequence as f64,"kind":kind,"payload":payload});
        let path = root.path().join("events.jsonl");
        std::fs::write(
            &path,
            format!(
                "{}\n{}\n",
                frame(0, "round", round),
                frame(1, "terminal", serde_json::json!({"outcome":"success"}))
            ),
        )?;
        assert_eq!(rounds(&path, root.path(), 0, 1)?.len(), 1);
        assert!(rounds(&path, root.path(), 1, 1).is_err());
        std::fs::write(audio, b"changed audio bytes")?;
        assert!(rounds(&path, root.path(), 0, 1).is_err());
        Ok(())
    }
    #[test]
    fn terminal_only_is_not_a_successful_measurement() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("events.jsonl");
        std::fs::write(
            &path,
            "{\"schema\":\"talechime/performance/1\",\"attempt_id\":\"r\",\"sequence\":0,\"offset_seconds\":0,\"kind\":\"terminal\",\"payload\":{\"outcome\":\"success\"}}\n",
        )?;
        assert!(rounds(&path, root.path(), 1, 5).is_err());
        Ok(())
    }

    #[test]
    fn diagnostic_audit_accepts_numeric_artifacts_without_fake_audio() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let tensor = root.path().join("embedding.json");
        std::fs::write(&tensor, b"[0.25,0.5]")?;
        let payload = serde_json::json!({"summary":{"component":"speaker_embedding"},"artifacts":[{
            "path":"embedding.json","sha256":file_digest(&tensor)?,"bytes":std::fs::metadata(&tensor)?.len(),
            "kind":"diagnostic_tensor","derived_from":null}]});
        let frame = |sequence, kind, payload| serde_json::json!({"schema":VERSION,"attempt_id":"reference-0","sequence":sequence,"offset_seconds":sequence as f64,"kind":kind,"payload":payload});
        let path = root.path().join("events.jsonl");
        std::fs::write(
            &path,
            format!(
                "{}\n{}\n",
                frame(
                    0,
                    "observation",
                    serde_json::json!({"diagnostic_result":payload})
                ),
                frame(1, "terminal", serde_json::json!({"outcome":"success"}))
            ),
        )?;
        let mut sequence = super::super::records::EventSequence::default();
        for line in std::fs::read_to_string(&path)?.lines() {
            sequence.accept("reference-0", &serde_json::from_str(line)?)?;
        }
        assert_eq!(diagnostic_result(&path, root.path())?.artifacts.len(), 1);
        assert!(rounds(&path, root.path(), 0, 1).is_err());
        std::fs::write(tensor, b"changed")?;
        assert!(diagnostic_result(&path, root.path()).is_err());
        Ok(())
    }

    #[test]
    fn diagnostic_audit_rejects_rounds_and_terminal_only() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("events.jsonl");
        for kind in ["round", "terminal"] {
            std::fs::write(
                &path,
                format!(
                    "{}\n",
                    serde_json::json!({"schema":VERSION,"attempt_id":"reference-0","sequence":0,"offset_seconds":0.,"kind":kind,"payload":{"outcome":"success"}})
                ),
            )?;
            assert!(diagnostic_result(&path, root.path()).is_err());
        }
        Ok(())
    }
}
