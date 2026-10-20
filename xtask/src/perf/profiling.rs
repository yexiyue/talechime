//! Optional native artifacts; this module never starts global WPR recording.
use super::{
    evidence::{VERSION, contained, file_digest, publish_json, read_json},
    prepare::capture,
    records::Artifact,
};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use std::{path::Path, process::Command, time::Duration};

pub fn tools() -> serde_json::Value {
    let mut tools = Vec::new();
    for (name, program, argument) in [
        ("nsight_systems", "nsys", "--version"),
        ("windows_performance_recorder", "wpr", "-help"),
    ] {
        let mut command = Command::new(program);
        command.arg(argument);
        let result = match capture(&mut command, Duration::from_secs(2)) {
            Ok(bytes) => {
                let version = String::from_utf8_lossy(&bytes)
                    .chars()
                    .take(2048)
                    .collect::<String>();
                serde_json::json!({"tool":name,"status":"available","version_output":version,"query":[program,argument]})
            }
            Err(error) => {
                serde_json::json!({"tool":name,"status":"unavailable","version_output":null,"reason":error.to_string(),"baseline_blocked":false})
            }
        };
        tools.push(result);
    }
    serde_json::json!({"schema":VERSION,"tools":tools,"instruments":{"status":"deferred","reason":"Mac native adapter and machine verification pending"}})
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attachment {
    pub schema: String,
    pub run_identity: String,
    pub tool: String,
    pub version: String,
    pub invocation: Vec<String>,
    pub provenance: String,
    pub artifact: Artifact,
}

pub fn attach(
    run: &Path,
    tool: &str,
    version: &str,
    invocation: Vec<String>,
    source: &Path,
) -> anyhow::Result<()> {
    ensure!(
        ["nsight_systems", "wpr"].contains(&tool),
        "native tool adapter unsupported; Instruments is deferred"
    );
    ensure!(
        !version.trim().is_empty() && !invocation.is_empty(),
        "native tool version and invocation are required"
    );
    let run = run.canonicalize()?;
    let info: serde_json::Value = read_json(&run.join("run.json"))?;
    ensure!(
        info["schema"] == VERSION && info["job"]["conditions"]["observation"] == "profile",
        "native profiles must belong to a separate profile-mode run"
    );
    let extension = if tool == "wpr" { "etl" } else { "nsys-rep" };
    ensure!(
        source.extension().and_then(|s| s.to_str()) == Some(extension),
        "native artifact extension differs from tool"
    );
    let bytes = source.metadata()?.len();
    ensure!(
        bytes > 0 && bytes <= 8 * 1024 * 1024 * 1024,
        "native artifact exceeds budget or is empty"
    );
    let hash = file_digest(source)?;
    let relative = format!("profiles/{hash}.{extension}");
    let target = contained(&run, Path::new(&relative))?;
    std::fs::create_dir_all(target.parent().context("profile parent missing")?)?;
    let mut stage = tempfile::NamedTempFile::new_in(target.parent().unwrap())?;
    std::io::copy(&mut std::fs::File::open(source)?, stage.as_file_mut())?;
    stage.as_file().sync_all()?;
    ensure!(
        file_digest(stage.path())? == hash,
        "native artifact changed during import"
    );
    stage.persist_noclobber(&target)?;
    let attachment=Attachment {schema:VERSION.into(),run_identity:info["identity"].as_str().context("run identity missing")?.into(),tool:tool.into(),version:version.into(),invocation,provenance:"explicit external tool artifact; version/invocation declared by caller; native timing interpretation requires tool review".into(),artifact:Artifact {path:relative.into(),sha256:hash.clone(),bytes,kind:"native_profile".into(),derived_from:None}};
    publish_json(&run.join(format!("profiles/{hash}.json")), &attachment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_profiles_require_separate_mode_and_preserve_provenance() -> anyhow::Result<()> {
        let run = tempfile::tempdir()?;
        let source = tempfile::tempdir()?;
        let artifact = source.path().join("sample.etl");
        std::fs::write(&artifact, b"opaque external profile")?;
        let info = |mode| serde_json::json!({"schema":VERSION,"identity":"fixture-run","job":{"conditions":{"observation":mode}}});
        publish_json(&run.path().join("run.json"), &info("baseline"))?;
        assert!(
            attach(
                run.path(),
                "wpr",
                "fixture-version",
                vec!["wpr".into()],
                &artifact
            )
            .is_err()
        );
        std::fs::remove_file(run.path().join("run.json"))?;
        publish_json(&run.path().join("run.json"), &info("profile"))?;
        assert!(attach(run.path(), "wpr", "", vec![], &artifact).is_err());
        attach(
            run.path(),
            "wpr",
            "fixture-version",
            vec!["wpr".into(), "-stop".into()],
            &artifact,
        )?;
        let hash = file_digest(&artifact)?;
        let receipt: Attachment = read_json(&run.path().join(format!("profiles/{hash}.json")))?;
        assert_eq!(receipt.version, "fixture-version");
        assert_eq!(receipt.run_identity, "fixture-run");
        receipt.artifact.verify(run.path())?;
        assert!(
            attach(
                run.path(),
                "wpr",
                "fixture-version",
                vec!["wpr".into()],
                &artifact
            )
            .is_err()
        );
        Ok(())
    }
}
