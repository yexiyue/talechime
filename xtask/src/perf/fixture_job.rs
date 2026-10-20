//! Create an explicitly synthetic end-to-end job, never a production model result.
use super::{
    build::Build,
    evidence::{VERSION, digest, publish_json, read_json},
    plan::{Plan, Platform},
    records::Conditions,
    runner::Job,
};
use anyhow::ensure;
use std::{collections::BTreeMap, path::Path};

pub fn create(build_path: &Path, output: &Path) -> anyhow::Result<()> {
    create_mode(build_path, output, "baseline")
}

pub fn create_mode(build_path: &Path, output: &Path, mode: &str) -> anyhow::Result<()> {
    ensure!(
        ["off", "baseline", "trace", "profile"].contains(&mode),
        "unknown observation mode"
    );
    let build: Build = read_json(build_path)?;
    build.verify()?;
    ensure!(
        build.probe == "performance_probe",
        "fixture job requires measurement probe"
    );
    ensure!(!output.exists(), "fixture input directory already exists");
    std::fs::create_dir_all(output)?;
    let output = output.canonicalize()?;
    let corpus = output.join("corpus.txt");
    let bytes = "你好，欢迎使用性能测试夹具。".as_bytes();
    std::fs::write(&corpus, bytes)?;
    let environment = super::environment::capture_environment("cpu")?;
    let mut plan = Plan::new(Platform::host()?, environment.machine.clone())?;
    // The executed immutable binary, rather than later edits to the workspace,
    // owns the source identity of this intentionally synthetic smoke plan.
    plan.source = build.source.clone();
    plan.observation = mode.into();
    let plan_path = output.join("plan.json");
    publish_json(&plan_path, &plan)?;
    let request = serde_json::json!({"resources_root":output.join("resources"),"backend":"fixture","model":"deterministic","device":"cpu","corpus":corpus,"corpus_sha256":digest(bytes),"voice":"fixture","style":null,"continuation":false,"seed":42,"params":{},"max_pcm_bytes":4*1024*1024,"fixture":true});
    let mut parameters = BTreeMap::new();
    for name in ["voice", "style", "seed", "continuation"] {
        parameters.insert(format!("request.{name}"), request[name].to_string());
    }
    let conditions = Conditions {
        machine: environment.machine,
        os: environment.os,
        arch: environment.arch,
        hardware: environment.hardware,
        driver: environment.driver,
        provider: "cpu".into(),
        math_backend: "synthetic".into(),
        precision: "f32".into(),
        features: build.features,
        threads: BTreeMap::from([("rayon".into(), 1)]),
        backend: "fixture".into(),
        model: "deterministic".into(),
        resource_sha256: digest(&serde_json::to_vec(
            &serde_json::json!({"verification":"synthetic_fixture"}),
        )?),
        corpus_sha256: digest(bytes),
        conditioning_sha256: digest(&serde_json::to_vec(
            &serde_json::json!({"voice":"fixture","style":null,"files":[]}),
        )?),
        parameters,
        mode: "native_stream".into(),
        delivery_boundary: "native_streaming_first_pcm".into(),
        observation: mode.into(),
        timing_contract: "production_synthesis_stream_consumer_start_to_completed/1".into(),
        observation_config: BTreeMap::from([(
            "capture_budget".into(),
            serde_json::to_string(&super::executor::Budget {
                sample_interval_ms: if mode == "off" { None } else { Some(500) },
                ..Default::default()
            })?,
        )]),
        environment: environment.flags,
    };
    let job = Job {
        schema: VERSION.into(),
        plan: plan_path,
        build: build_path.canonicalize()?,
        case_id: "synthetic-fixture".into(),
        conditions,
        request,
        timeout_seconds: 30,
        conditioning_files: vec![],
        budget: super::executor::Budget {
            sample_interval_ms: if mode == "off" { None } else { Some(500) },
            ..Default::default()
        },
    };
    publish_json(&output.join("job.json"), &job)
}
