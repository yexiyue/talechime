//! Portable offline evidence report, built entirely in Rust.
use super::{
    evidence::{VERSION, contained, digest, file_digest, publish_bytes, publish_json, read_json},
    records::{Artifact, Attempt, Conditions},
    statistics::{Metric, summarize},
};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use std::io::BufRead;
use std::{collections::BTreeMap, path::Path};

const ECHARTS_VERSION: &str = "6.0.0";
const ECHARTS_SHA256: &str = "baa8dfe7e1d9336b98e8986ba7e20ea15e7cdbea1ef42a59d59478632fa45a1d";
const ECHARTS_LICENSE_SHA256: &str =
    "634293835b43a6dd2094fa39182a3d9a6b9ca43b7fdb9ac354e8037af2a3093a";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub artifacts: Vec<Artifact>,
    pub listening_acceptance: String,
}

#[derive(Serialize)]
struct ChartPoint {
    case: String,
    metric: &'static str,
    value: Option<f64>,
    unit: &'static str,
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn record(root: &Path, path: &str, kind: &str) -> anyhow::Result<Artifact> {
    let full = contained(root, Path::new(path))?;
    Ok(Artifact {
        path: path.into(),
        sha256: file_digest(&full)?,
        bytes: full.metadata()?.len(),
        kind: kind.into(),
        derived_from: None,
    })
}

fn copy_asset(source: &Path, destination: &Path, expected: &Artifact) -> anyhow::Result<()> {
    ensure!(
        source.metadata()?.len() == expected.bytes && file_digest(source)? == expected.sha256,
        "report input asset changed"
    );
    let parent = destination.parent().context("asset parent missing")?;
    std::fs::create_dir_all(parent)?;
    let mut stage = tempfile::NamedTempFile::new_in(parent)?;
    std::io::copy(&mut std::fs::File::open(source)?, stage.as_file_mut())?;
    stage.as_file().sync_all()?;
    ensure!(
        file_digest(stage.path())? == expected.sha256,
        "asset changed during copy"
    );
    stage.persist_noclobber(destination)?;
    Ok(())
}

fn csv(text: &str) -> String {
    format!("\"{}\"", text.replace('"', "\"\""))
}

fn asset_extension(asset: &Artifact) -> anyhow::Result<&str> {
    Ok(match asset.kind.as_str() {
        "audio_f32" | "audio_pcm16" => "wav",
        "source_ranges" | "preparation" | "metrics" | "diagnostic_state" | "diagnostic_trace"
        | "profile_receipt" => "json",
        "raw_samples" => "jsonl",
        "native_profile" => asset
            .path
            .extension()
            .and_then(|s| s.to_str())
            .filter(|s| ["etl", "nsys-rep"].contains(s))
            .context("unsupported native profile")?,
        _ => anyhow::bail!("unapproved evidence artifact"),
    })
}

fn evidence_asset(
    source_root: &Path,
    output: &Path,
    asset: &Artifact,
    manifest: &mut Manifest,
    copied: &mut BTreeMap<String, String>,
) -> anyhow::Result<String> {
    let extension = asset_extension(asset)?;
    let relative = format!("assets/{}.{}", asset.sha256, extension);
    asset.verify(source_root)?;
    if !copied.contains_key(&relative) {
        copy_asset(
            &contained(source_root, &asset.path)?,
            &contained(output, Path::new(&relative))?,
            asset,
        )?;
        manifest
            .artifacts
            .push(record(output, &relative, &asset.kind)?);
        copied.insert(relative.clone(), asset.sha256.clone());
    }
    Ok(relative)
}
fn number(value: Option<f64>) -> String {
    value.map_or_else(|| "unavailable".into(), |v| format!("{v:.6}"))
}

fn resource_table(path: &Path) -> anyhow::Result<String> {
    let reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut maxima: BTreeMap<String, (Option<f64>, String)> = BTreeMap::new();
    for line in reader.lines() {
        let line = line?;
        ensure!(line.len() <= 1024 * 1024, "resource sample frame too large");
        let sample: serde_json::Value = serde_json::from_str(&line)?;
        for (name, value) in sample["readings"]
            .as_object()
            .context("invalid resource sample")?
        {
            let reading: super::evidence::Reading = serde_json::from_value(value.clone())?;
            reading.validate()?;
            let key = format!(
                "{name} / {} / {} / {}",
                reading.unit, reading.scope, reading.source
            );
            let entry = maxima.entry(key).or_insert((None, String::new()));
            if let Some(value) = reading.value {
                entry.0 = Some(entry.0.map_or(value, |previous| previous.max(value)));
            } else {
                entry.1 = reading.reason.unwrap_or_default();
            }
            ensure!(maxima.len() <= 256, "too many resource sample series");
        }
    }
    let mut html = String::from(
        "<li>Resource maxima within this attempt; OS high-water retains OS semantics; other maxima are sampled, not true peaks.<table><tr><th>Metric / unit / scope / source</th><th>Maximum</th><th>Unavailable sample reason</th></tr>",
    );
    for (name, (value, reason)) in maxima {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td></tr>",
            escape(&name),
            number(value),
            escape(&reason)
        ));
    }
    html.push_str("</table></li>");
    Ok(html)
}

fn resource_metrics(
    run: &Path,
    attempts: &[Attempt],
    metric: &str,
    scope: &str,
) -> anyhow::Result<BTreeMap<String, Option<f64>>> {
    let mut samples = Vec::new();
    for attempt in attempts {
        let path = run
            .join("attempts")
            .join(&attempt.attempt_id)
            .join("resource-samples.jsonl");
        if !path.is_file() {
            samples.push(BTreeMap::new());
            continue;
        }
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut maxima: BTreeMap<String, Option<f64>> = BTreeMap::new();
        for line in reader.lines() {
            let sample: serde_json::Value = serde_json::from_str(&line?)?;
            for (name, reading) in sample["readings"]
                .as_object()
                .context("invalid resource sample")?
            {
                if name != metric
                    && !(scope == "whole_device"
                        && name
                            .split_once(':')
                            .is_some_and(|(_, suffix)| suffix == metric))
                {
                    continue;
                }
                let reading: super::evidence::Reading = serde_json::from_value(reading.clone())?;
                reading.validate()?;
                if reading.scope == scope {
                    let maximum = maxima.entry(name.clone()).or_default();
                    if let Some(value) = reading.value {
                        *maximum = Some(maximum.map_or(value, |previous| previous.max(value)));
                    }
                }
            }
        }
        samples.push(maxima);
    }
    let mut names: std::collections::BTreeSet<_> = samples
        .iter()
        .flat_map(|sample| sample.keys().cloned())
        .collect();
    if names.is_empty() {
        names.insert(metric.to_owned());
    }
    Ok(names
        .into_iter()
        .map(|name| {
            let value = samples
                .iter()
                .map(|sample| sample.get(&name).copied().flatten())
                .collect::<Option<Vec<_>>>()
                .and_then(|values| values.into_iter().reduce(f64::max));
            (name, value)
        })
        .collect())
}

fn copy_echarts(output: &Path, library_root: &Path) -> anyhow::Result<()> {
    let script_name = format!("echarts-{ECHARTS_VERSION}.min.js");
    let library_root = library_root.canonicalize()?;
    let script = library_root.join(&script_name);
    let license = library_root.join("LICENSE");
    let script_destination = contained(output, Path::new(&script_name))?;
    let license_destination = contained(output, Path::new("echarts-LICENSE"))?;
    for (source, expected, destination) in [
        (&script, ECHARTS_SHA256, script_destination),
        (&license, ECHARTS_LICENSE_SHA256, license_destination),
    ] {
        ensure!(source.is_file(), "bundled ECharts asset missing");
        let digest = file_digest(source)
            .with_context(|| format!("hash bundled ECharts asset {}", source.display()))?;
        ensure!(
            digest == expected,
            "bundled ECharts identity changed: {} {} != {}",
            source.display(),
            digest,
            expected
        );
        std::fs::copy(source, &destination)
            .with_context(|| format!("copy {}", destination.display()))?;
        ensure!(
            file_digest(&destination)? == expected,
            "ECharts copy changed"
        );
    }
    Ok(())
}

fn planned_chart_label(planned: &serde_json::Value) -> String {
    [
        planned["backend"].as_str().unwrap_or("unknown backend"),
        planned["model"].as_str().unwrap_or("unknown model"),
        planned["device"].as_str().unwrap_or("unknown device"),
    ]
    .join(" / ")
}

fn planned_matches(planned: &serde_json::Value, platform: &str, conditions: &Conditions) -> bool {
    planned["backend"].as_str() == Some(conditions.backend.as_str())
        && planned["model"].as_str() == Some(conditions.model.as_str())
        && planned["device"].as_str() == Some(conditions.provider.as_str())
        && conditions.os.split('-').next() == Some(platform)
}

fn chart_label(case: &str, conditions: &Conditions) -> String {
    format!(
        "{} / {} / {} / {} / {case}",
        conditions.backend, conditions.model, conditions.provider, conditions.mode
    )
}

fn chart_json(rows: &[ChartPoint]) -> anyhow::Result<String> {
    Ok(serde_json::to_string(rows)
        .context("serialize chart data")?
        .replace('<', "\\u003c"))
}

fn mean_if_complete(values: &[Option<f64>]) -> Option<f64> {
    if values.is_empty() || values.iter().any(Option::is_none) {
        return None;
    }
    let sum = values.iter().filter_map(|value| *value).sum::<f64>();
    Some(sum / values.len() as f64)
}

pub fn report(run: &Path, results: &Path, output: &Path) -> anyhow::Result<()> {
    let library_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("docs/legal/echarts");
    ensure!(!output.exists(), "report output already exists");
    let run = run.canonicalize()?;
    let results = contained(&run, results)?;
    let attempts: Vec<Attempt> = read_json(&results)?;
    let run_info: serde_json::Value = read_json(&run.join("run.json"))?;
    ensure!(run_info["schema"] == VERSION, "unknown run schema");
    let expected = run_info["plan"]["processes"]
        .as_u64()
        .context("planned process count missing")? as usize;
    let mut groups: BTreeMap<String, Vec<Attempt>> = BTreeMap::new();
    for attempt in attempts {
        attempt.validate()?;
        groups
            .entry(attempt.case_id.clone())
            .or_default()
            .push(attempt);
    }
    std::fs::create_dir_all(output)?;
    let output = output.canonicalize()?;
    let mut manifest = Manifest {
        schema: VERSION.into(),
        artifacts: vec![],
        listening_acceptance: "pending".into(),
    };
    let mut table = String::new();
    let mut details = String::new();
    let mut csv_output = String::from(
        "case,platform,provider,metric,mean_of_process_means,successful_hot_rounds,expected_hot_rounds,complete\n",
    );
    let mut chart_rows = Vec::new();
    let mut copied: BTreeMap<String, String> = BTreeMap::new();
    let mut all_results = Vec::new();
    let planned_cases = run_info["plan"]["cases"]
        .as_array()
        .context("planned cases missing")?
        .clone();
    let platform = run_info["plan"]["platform"].as_str().unwrap_or("");
    for planned in &planned_cases {
        let label = planned_chart_label(planned);
        let attempted = groups
            .values()
            .flatten()
            .any(|attempt| planned_matches(planned, platform, &attempt.conditions));
        if !attempted {
            for metric in ["RTF", "First PCM"] {
                chart_rows.push(ChartPoint {
                    case: label.clone(),
                    metric,
                    value: None,
                    unit: if metric == "RTF" { "ratio" } else { "seconds" },
                });
            }
        }
    }
    for (case, mut processes) in groups {
        processes.sort_by_key(|a| a.process_index);
        let first = &processes[0];
        let label = chart_label(&case, &first.conditions);
        details.push_str(&format!(
            "<section><h2>{}</h2><p>{} · {} · listening_acceptance=pending</p>",
            escape(&case),
            escape(&label),
            escape(&first.conditions.os)
        ));
        let mut representative = None;
        let mut load_values = Vec::new();
        let mut reference_values = Vec::new();
        for metric in [
            Metric::Rtf,
            Metric::SynthesisSeconds,
            Metric::FirstPcmSeconds,
        ] {
            let summary = summarize(&processes, expected, metric)?;
            let name = format!("{metric:?}");
            table.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}/{}</td><td>{}</td></tr>",
                escape(&case),
                escape(&label),
                name,
                number(summary.mean_of_process_means),
                summary.successful_hot_rounds,
                summary.expected_hot_rounds,
                if summary.complete {
                    "complete"
                } else {
                    "incomplete"
                }
            ));
            csv_output.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                csv(&case),
                csv(&first.conditions.os),
                csv(&first.conditions.provider),
                csv(&name),
                summary
                    .mean_of_process_means
                    .map_or(String::new(), |n| n.to_string()),
                summary.successful_hot_rounds,
                summary.expected_hot_rounds,
                summary.complete
            ));
            if metric == Metric::Rtf {
                chart_rows.push(ChartPoint {
                    case: label.clone(),
                    metric: "RTF",
                    value: summary.mean_of_process_means,
                    unit: "ratio",
                });
            }
        }
        for attempt in &processes {
            let attempt_dir = contained(&run, &Path::new("attempts").join(&attempt.attempt_id))?;
            let audited: Attempt = if attempt.outcome == "success" {
                super::storage::recover(&attempt_dir, attempt)?
            } else {
                read_json(&attempt_dir.join("attempt.json"))?
            };
            ensure!(
                serde_json::to_value(&audited)? == serde_json::to_value(attempt)?,
                "report results differ from saved attempt evidence"
            );
            details.push_str(&format!(
                "<details><summary>process {} · {}</summary><ul>",
                attempt.process_index,
                escape(&attempt.outcome)
            ));
            details.push_str(&format!(
                "<li>trace_complete={} · dropped={}</li>",
                attempt.trace_complete,
                attempt
                    .dropped_trace_records
                    .map_or_else(|| "unavailable".into(), |n| n.to_string())
            ));
            for asset in &attempt.artifacts {
                let relative =
                    evidence_asset(&attempt_dir, &output, asset, &mut manifest, &mut copied)?;
                details.push_str(&format!(
                    "<li><a href=\"{relative}\">{}</a></li>",
                    escape(&asset.kind)
                ));
                if asset.kind == "preparation" {
                    let preparation: serde_json::Value =
                        read_json(&contained(&attempt_dir, &asset.path)?)?;
                    load_values.push(preparation["load_seconds"].as_f64());
                    reference_values.push(preparation["reference_prepare_seconds"].as_f64());
                    for phase in [
                        "verify_seconds",
                        "load_seconds",
                        "reference_prepare_seconds",
                    ] {
                        details.push_str(&format!(
                            "<li>{}: {} seconds</li>",
                            phase,
                            number(preparation[phase].as_f64())
                        ));
                    }
                    details.push_str(&format!(
                        "<li>Reference preparation: {}</li>",
                        escape(
                            preparation["reference_prepare_reason"]
                                .as_str()
                                .unwrap_or("see preparation receipt")
                        )
                    ));
                }
            }
            if attempt_dir.join("resource-samples.jsonl").is_file() {
                let asset = record(&attempt_dir, "resource-samples.jsonl", "raw_samples")?;
                let relative =
                    evidence_asset(&attempt_dir, &output, &asset, &mut manifest, &mut copied)?;
                details.push_str(&format!("<li><a href=\"{relative}\">Resource samples: process RSS/private bytes and optional whole-device GPU readings; unavailable values retain reasons</a></li>"));
                details.push_str(&resource_table(
                    &attempt_dir.join("resource-samples.jsonl"),
                )?);
            }
            for round in &attempt.rounds {
                details.push_str(&format!(
                    "<li>round {} · {} · {} · synthesis={}s · audio={}s<ul>",
                    round.index,
                    if round.warmup { "warmup" } else { "hot" },
                    escape(&round.outcome),
                    number(round.synthesis_seconds),
                    number(round.audio_seconds)
                ));
                for asset in &round.artifacts {
                    ensure!(
                        ["audio_f32", "audio_pcm16", "source_ranges"]
                            .contains(&asset.kind.as_str()),
                        "unapproved report artifact kind"
                    );
                    let relative =
                        evidence_asset(&attempt_dir, &output, asset, &mut manifest, &mut copied)?;
                    if representative.is_none()
                        && !round.warmup
                        && round.outcome == "success"
                        && asset.kind == "audio_pcm16"
                        && asset.path.to_string_lossy().contains("-full-")
                    {
                        representative = Some(relative.clone());
                    }
                    details.push_str(&format!(
                        "<li><a href=\"{}\">{}</a></li>",
                        relative,
                        escape(&asset.path.to_string_lossy())
                    ));
                }
                for observation in &round.observations {
                    details.push_str(&format!(
                        "<li>{}: {} {}</li>",
                        escape(&observation.name),
                        number(observation.reading.value),
                        escape(
                            observation
                                .reading
                                .reason
                                .as_deref()
                                .unwrap_or(&observation.reading.unit)
                        )
                    ));
                }
                details.push_str("</ul></li>");
            }
            details.push_str("</ul></details>");
        }
        if let Some(audio) = representative {
            details.push_str(&format!("<p>Representative: first successful process / first successful hot round</p><audio controls preload=\"none\" src=\"{audio}\"></audio>"));
        }
        chart_rows.push(ChartPoint {
            case: label.clone(),
            metric: "Load",
            value: mean_if_complete(&load_values),
            unit: "seconds",
        });
        chart_rows.push(ChartPoint {
            case: label.clone(),
            metric: "Reference preparation",
            value: mean_if_complete(&reference_values),
            unit: "seconds",
        });
        chart_rows.push(ChartPoint {
            case: label.clone(),
            metric: "First PCM",
            value: summarize(&processes, expected, Metric::FirstPcmSeconds)?.mean_of_process_means,
            unit: "seconds",
        });
        for (reading, scope, metric) in [
            ("rss_bytes", "process", "RSS peak"),
            (
                "device_vram_bytes",
                "whole_device",
                "Whole-device VRAM peak",
            ),
        ] {
            for (name, value) in resource_metrics(&run, &processes, reading, scope)? {
                chart_rows.push(ChartPoint {
                    case: if scope == "whole_device" {
                        format!("{label} / {name} / whole_device")
                    } else {
                        label.clone()
                    },
                    metric,
                    value,
                    unit: "bytes",
                });
            }
        }
        details.push_str("</section>");
        all_results.extend(processes);
    }
    let matrix = serde_json::to_string_pretty(&run_info["plan"]["cases"])?;
    let profiles = run.join("profiles");
    if profiles.is_dir() {
        for entry in std::fs::read_dir(&profiles)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let receipt: super::profiling::Attachment = read_json(&path)?;
            ensure!(
                receipt.schema == VERSION
                    && receipt.run_identity
                        == run_info["identity"]
                            .as_str()
                            .context("missing run identity")?,
                "profile identity mismatch"
            );
            let relative =
                evidence_asset(&run, &output, &receipt.artifact, &mut manifest, &mut copied)?;
            let receipt_asset = record(
                &run,
                &format!(
                    "profiles/{}",
                    path.file_name()
                        .context("missing profile filename")?
                        .to_string_lossy()
                ),
                "profile_receipt",
            )?;
            let provenance =
                evidence_asset(&run, &output, &receipt_asset, &mut manifest, &mut copied)?;
            details.push_str(&format!("<p>External native profile: <a href=\"{relative}\">{}</a> · <a href=\"{provenance}\">declared version/invocation</a>. Native timing requires tool review.</p>", escape(&receipt.tool)));
        }
    }
    let script_name = format!("echarts-{ECHARTS_VERSION}.min.js");
    let chart_script = r#"<script>
const rows=CHART_DATA_JSON;
const groups=[
  {id:'chart-rtf',metrics:['RTF'],title:'RTF (lower is faster)',markLine:1},
  {id:'chart-first-pcm',metrics:['First PCM'],title:'First PCM'},
  {id:'chart-load',metrics:['Load','Reference preparation'],title:'Model load / reference preparation'},
  {id:'chart-memory',metrics:['RSS peak','Whole-device VRAM peak'],title:'Sampled memory peaks (distinct scopes)'}
];
function points(metric){return rows.filter(row=>row.metric===metric);}
function cases(datasets){return [...new Set(datasets.flat().map(point=>point.case))];}
for(const group of groups){
  const datasets=group.metrics.map(points);
  echarts.init(document.getElementById(group.id)).setOption({
    title:{text:group.title,left:8},
    tooltip:{trigger:'axis'},
    grid:{left:180,right:60,bottom:50},
    dataZoom:[{type:'inside'},{type:'slider'}],
    legend:group.metrics.length>1?{}:undefined,
    xAxis:{type:'value',name:datasets.flat()[0]?.unit ?? 'unavailable'},
    yAxis:{type:'category',data:cases(datasets)},
    series:datasets.map((data,index)=>({
      type:'bar',name:group.metrics[index],data:data.map(point=>[point.value,point.case]),
      connectNulls:true,itemStyle:{color:index===0?'#196b5c':'#b76525'},
      ...(group.markLine?{markLine:{symbol:'none',lineStyle:{type:'dashed'},data:[{xAxis:group.markLine}]}}:{})
    }))
  });
}
</script>"#
        .replace(
            "CHART_DATA_JSON",
            &chart_json(&chart_rows)?,
        );
    let html = format!(
        "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Talechime performance evidence</title><style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#1c302c;background:#fafbf9}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #ccd5d1}}section{{margin:40px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}audio{{width:100%;max-width:600px}}a{{color:#125e50}}summary{{cursor:pointer}}.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:24px}}.chart{{height:360px;background:#fff;border:1px solid #ccd5d1}}</style><main><h1>Talechime performance evidence</h1><p>Generation evidence · listening_acceptance=pending. Synthetic fixture results are not production model baselines. Platforms and providers remain separate; speed does not establish content or quality equivalence.</p><p><a href=\"results.json\">Raw rounds</a> · <a href=\"results.csv\">CSV</a> · <a href=\"corpus.txt\">Corpus snapshot</a> · <a href=\"echarts-LICENSE\">ECharts license</a></p><div class=\"charts\"><div id=\"chart-rtf\" class=\"chart\" aria-label=\"RTF by case\"></div><div id=\"chart-first-pcm\" class=\"chart\" aria-label=\"First PCM by case\"></div><div id=\"chart-load\" class=\"chart\" aria-label=\"Load and reference cost\"></div><div id=\"chart-memory\" class=\"chart\" aria-label=\"RSS and whole-device VRAM peaks\"></div></div><div style=\"overflow-x:auto\"><table><thead><tr><th>Case</th><th>Execution</th><th>Metric</th><th>Mean</th><th>Hot coverage</th><th>State</th></tr></thead><tbody>{table}</tbody></table></div>{details}<details><summary>Frozen matrix (including deferred, missing and not applicable rows)</summary><pre>{}</pre></details></main><script src=\"{script_name}\"></script>{chart_script}",
        escape(&matrix)
    );
    copy_echarts(&output, &library_root)?;
    publish_bytes(&output.join("index.html"), html.as_bytes())?;
    publish_json(&output.join("results.json"), &all_results)?;
    publish_bytes(&output.join("results.csv"), csv_output.as_bytes())?;
    let corpus = run.join("corpus.txt");
    let corpus_bytes = std::fs::read(&corpus)?;
    if let Some(attempt) = all_results.first() {
        ensure!(
            digest(&corpus_bytes) == attempt.conditions.corpus_sha256,
            "corpus snapshot digest mismatch"
        );
    }
    publish_bytes(&output.join("corpus.txt"), &corpus_bytes)?;
    for (path, kind) in [
        ("index.html", "report"),
        (&script_name, "chart_library"),
        ("echarts-LICENSE", "chart_license"),
        ("results.json", "raw_results"),
        ("results.csv", "table"),
        ("corpus.txt", "source_snapshot"),
    ] {
        manifest.artifacts.push(record(&output, path, kind)?);
    }
    publish_json(&output.join("manifest.json"), &manifest)
}

pub fn verify_report(root: &Path) -> anyhow::Result<Manifest> {
    let manifest: Manifest = read_json(&root.join("manifest.json"))?;
    ensure!(
        manifest.schema == VERSION && manifest.listening_acceptance == "pending",
        "unsupported report manifest"
    );
    for artifact in &manifest.artifacts {
        let name = artifact.path.to_string_lossy();
        let approved = match artifact.kind.as_str() {
            "report" => name == "index.html",
            "chart_library" => name == format!("echarts-{ECHARTS_VERSION}.min.js"),
            "chart_license" => name == "echarts-LICENSE",
            "raw_results" => name == "results.json",
            "table" => name == "results.csv",
            "source_snapshot" => name == "corpus.txt",
            "audio_f32" | "audio_pcm16" | "source_ranges" | "raw_samples" | "preparation"
            | "metrics" | "diagnostic_state" | "diagnostic_trace" | "profile_receipt"
            | "native_profile" => {
                let extension = asset_extension(artifact)?;
                name == format!("assets/{}.{}", artifact.sha256, extension)
                    && artifact.sha256.len() == 64
                    && artifact.sha256.bytes().all(|b| b.is_ascii_hexdigit())
            }
            _ => false,
        };
        ensure!(approved, "archive contains a non-report artifact");
        artifact.verify(root)?;
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn labels_are_escaped_and_csv_quotes_are_preserved() {
        assert_eq!(escape("<script>\"&'"), "&lt;script&gt;&quot;&amp;&#39;");
        assert_eq!(csv("a,\"b\""), "\"a,\"\"b\"\"\"");
    }

    #[test]
    fn chart_data_cannot_close_script_and_cases_keep_distinct_conditions() -> anyhow::Result<()> {
        let attempts = super::super::test_support::attempts(&[1.0]);
        let mut conditions = attempts[0].conditions.clone();
        conditions.model = "</script><script>alert('x')</script>".into();
        let first = chart_label("short-case", &conditions);
        let second = chart_label("multi-case", &conditions);
        assert_ne!(first, second);
        conditions.mode = "continuation".into();
        assert_ne!(first, chart_label("short-case", &conditions));
        let json = chart_json(&[ChartPoint {
            case: first.clone(),
            metric: "RTF",
            value: Some(1.0),
            unit: "ratio",
        }])?;
        assert!(!json.contains('<'));
        assert!(json.contains("\\u003c/script>"));
        let decoded: serde_json::Value = serde_json::from_str(&json)?;
        assert_eq!(decoded[0]["case"], first);
        Ok(())
    }

    #[test]
    fn planned_coverage_matches_model_device_and_platform_without_case_ids() {
        let attempts = super::super::test_support::attempts(&[1.0]);
        let conditions = &attempts[0].conditions;
        let mut planned =
            serde_json::json!({"backend":"fixture", "model":"deterministic", "device":"cpu"});
        assert!(planned_matches(&planned, "windows", conditions));
        assert!(!planned_matches(&planned, "macos", conditions));
        for field in ["backend", "model", "device"] {
            let previous = planned[field].clone();
            planned[field] = "different".into();
            assert!(!planned_matches(&planned, "windows", conditions));
            planned[field] = previous;
        }
    }

    #[test]
    fn memory_charts_keep_rss_separate_and_retain_every_gpu() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let attempts = super::super::test_support::attempts(&[1.0, 2.0]);
        let reading = |value: Option<f64>, scope: &str| {
            serde_json::json!({
                "value":value, "unit":"bytes", "scope":scope, "source":"fixture", "interval_ms":500,
                "status":if value.is_some() { "available" } else { "unavailable" },
                "reason":if value.is_none() { Some("unsupported") } else { None },
            })
        };
        for (index, attempt) in attempts.iter().enumerate() {
            let directory = root.path().join("attempts").join(&attempt.attempt_id);
            std::fs::create_dir_all(&directory)?;
            let sample = serde_json::json!({"readings":{
                "os_peak_rss_bytes":reading(None, "process"),
                "rss_bytes":reading(Some(100.0 + index as f64), "process"),
                "GPU-A:device_vram_bytes":reading(Some(200.0 + index as f64), "whole_device"),
                "GPU-B:device_vram_bytes":reading(Some(500.0), "whole_device"),
                "GPU-C:device_vram_bytes":reading(None, "whole_device"),
            }});
            publish_bytes(
                &directory.join("resource-samples.jsonl"),
                format!("{sample}\n").as_bytes(),
            )?;
        }
        let rss = resource_metrics(root.path(), &attempts, "rss_bytes", "process")?;
        assert_eq!(rss, BTreeMap::from([("rss_bytes".into(), Some(101.0))]));
        let gpu = resource_metrics(root.path(), &attempts, "device_vram_bytes", "whole_device")?;
        assert_eq!(gpu.len(), 3);
        assert_eq!(gpu["GPU-A:device_vram_bytes"], Some(201.0));
        assert_eq!(gpu["GPU-B:device_vram_bytes"], Some(500.0));
        assert_eq!(gpu["GPU-C:device_vram_bytes"], None);
        std::fs::remove_file(
            root.path()
                .join("attempts")
                .join(&attempts[1].attempt_id)
                .join("resource-samples.jsonl"),
        )?;
        assert!(
            resource_metrics(root.path(), &attempts, "rss_bytes", "process")?["rss_bytes"]
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn report_keeps_failed_rounds_missing_values_and_coverage() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let run = root.path().join("run");
        std::fs::create_dir_all(run.join("attempts"))?;
        let mut attempts = super::super::test_support::attempts(&[1.0, 2.0, 3.0]);
        attempts[2].outcome = "failure".into();
        attempts[2].rounds[1].outcome = "failure".into();
        attempts[2].rounds[1].first_pcm_seconds = None;
        for attempt in &attempts {
            let directory = run.join("attempts").join(&attempt.attempt_id);
            std::fs::create_dir(&directory)?;
            if attempt.outcome == "success" {
                for name in ["stdout.jsonl", "stderr.log", "resource-samples.jsonl"] {
                    publish_bytes(&directory.join(name), b"")?;
                }
                publish_json(
                    &directory.join("capture.json"),
                    &super::super::executor::Capture {
                        outcome: "success".into(),
                        exit_code: Some(0),
                        cleanup_verified: true,
                        errors: vec![],
                        job_peak_memory: None,
                    },
                )?;
                super::super::storage::commit(&directory, attempt)?;
            } else {
                publish_json(&directory.join("attempt.json"), attempt)?;
            }
        }
        publish_bytes(&run.join("corpus.txt"), b"corpus")?;
        publish_json(
            &run.join("run.json"),
            &serde_json::json!({"schema":VERSION,"plan":{"platform":"windows","processes":3,"cases":[
                {"backend":"fixture","model":"deterministic","device":"cpu"},
                {"backend":"fixture","model":"deterministic","device":"cuda"}
            ]}}),
        )?;
        publish_json(&run.join("results.json"), &attempts)?;
        let output = root.path().join("report");
        report(&run, Path::new("results.json"), &output)?;
        verify_report(&output)?;
        let html = std::fs::read_to_string(output.join("index.html"))?;
        assert!(html.contains("incomplete") && html.contains("2/3") && html.contains("failure"));
        assert!(html.contains("listening_acceptance=pending"));
        assert!(html.contains("echarts.init(document.getElementById(group.id))"));
        assert!(html.contains("markLine"));
        let chart_data = html
            .split("const rows=")
            .nth(1)
            .unwrap()
            .split(";\nconst groups=")
            .next()
            .unwrap();
        let points: Vec<serde_json::Value> = serde_json::from_str(chart_data)?;
        let rtf = points
            .iter()
            .filter(|point| point["metric"] == "RTF")
            .collect::<Vec<_>>();
        assert_eq!(
            rtf.len(),
            2,
            "only the unattempted CUDA row and measured CPU case remain"
        );
        assert!(
            rtf.iter()
                .any(|point| point["case"] == "fixture / deterministic / cuda"
                    && point["value"].is_null())
        );
        assert!(rtf.iter().any(
            |point| point["case"].as_str().unwrap().contains("fixture-case")
                && point["value"].is_number()
        ));
        let raw: Vec<Attempt> = read_json(&output.join("results.json"))?;
        assert!(raw[2].rounds[1].first_pcm_seconds.is_none());
        Ok(())
    }

    #[test]
    fn diagnostic_assets_are_verified_and_archive_allowlist_excludes_models() -> anyhow::Result<()>
    {
        let input = tempfile::tempdir()?;
        let output = tempfile::tempdir()?;
        publish_bytes(&input.path().join("metrics.json"), b"{}")?;
        let asset = record(input.path(), "metrics.json", "metrics")?;
        let mut manifest = Manifest {
            schema: VERSION.into(),
            artifacts: vec![],
            listening_acceptance: "pending".into(),
        };
        evidence_asset(
            input.path(),
            output.path(),
            &asset,
            &mut manifest,
            &mut BTreeMap::new(),
        )?;
        publish_json(&output.path().join("manifest.json"), &manifest)?;
        verify_report(output.path())?;
        std::fs::write(input.path().join("metrics.json"), b"changed")?;
        assert!(
            evidence_asset(
                input.path(),
                output.path(),
                &asset,
                &mut manifest,
                &mut BTreeMap::new()
            )
            .is_err()
        );
        manifest.artifacts[0].kind = "model_weights".into();
        std::fs::remove_file(output.path().join("manifest.json"))?;
        publish_json(&output.path().join("manifest.json"), &manifest)?;
        assert!(verify_report(output.path()).is_err());
        Ok(())
    }
}
