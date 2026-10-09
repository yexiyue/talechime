//! Explicit real-model corpus acceptance. Inputs/audio/results stay in ignored local directories.
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use talechime::*;

type Wave = hound::WavWriter<BufWriter<fs::File>>;
struct Export {
    range: TextRange,
    writer: Wave,
}
impl Export {
    fn new(path: &Path, audio: &SpeechAudio) -> anyhow::Result<Self> {
        let pcm = audio.pcm();
        let writer = hound::WavWriter::create(
            path,
            hound::WavSpec {
                channels: pcm.channels,
                sample_rate: pcm.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?;
        Ok(Self {
            range: audio.range(),
            writer,
        })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 7 || args.len() == 8,
        "readback_corpus BACKEND MODEL cpu|metal RESOURCES CORPUS_JSON OUTPUT_DIR [MODES: off,report,gate,strict]"
    );
    run_local(run(&args)).await
}

async fn run(args: &[String]) -> anyhow::Result<()> {
    let modes: Vec<&str> = args
        .get(7)
        .map_or("off,report,gate", String::as_str)
        .split(',')
        .collect();
    anyhow::ensure!(
        modes
            .iter()
            .all(|mode| matches!(*mode, "off" | "report" | "gate" | "strict")),
        "invalid mode list"
    );
    let cases: Vec<Value> = serde_json::from_slice(&fs::read(&args[5])?)?;
    let output = PathBuf::from(&args[6]);
    fs::create_dir_all(&output)?;
    let mut model = ModelOptions::new(&args[1], &args[4]);
    model.model = Some(args[2].clone());
    model.device = match args[3].as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        _ => anyhow::bail!("invalid device"),
    };
    let started = Instant::now();
    let mut engine = Engine::prepare(model, |_| {}).await?;
    let tts_load_s = started.elapsed().as_secs_f64();
    let asr_load_s = if modes.iter().any(|mode| *mode != "off") {
        let started = Instant::now();
        let verifier = prepare_readback(ReadbackModelOptions::new(&args[4]), |_| {}).await?;
        engine.set_verifier(verifier)?;
        started.elapsed().as_secs_f64()
    } else {
        0.0
    };
    let caps = engine.capabilities()?;
    let mut summary = Vec::new();
    for case in cases {
        let id = case["id"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("missing case id"))?;
        anyhow::ensure!(!id.contains(['/', '\\']) && id != "..", "invalid case id");
        let text = fs::read_to_string(
            case["file"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("missing text file"))?,
        )?;
        anyhow::ensure!(
            case["sha256"].as_str() == Some(text_hash(&text).as_str()),
            "source hash mismatch"
        );
        for (mode, policy) in [
            ("off", VerificationPolicy::Off),
            ("report", VerificationPolicy::ReportOnly),
            (
                "gate",
                VerificationPolicy::Gate {
                    max_retries: 1,
                    strict_suspect: false,
                },
            ),
            (
                "strict",
                VerificationPolicy::Gate {
                    max_retries: 1,
                    strict_suspect: true,
                },
            ),
        ] {
            if !modes.contains(&mode) {
                continue;
            }
            let directory = output.join(id).join(mode);
            fs::create_dir_all(&directory)?;
            let mut reports = BufWriter::new(fs::File::create(directory.join("reports.jsonl"))?);
            let started = Instant::now();
            eprintln!("START {id} {mode} chars={}", text.chars().count());
            let mut stream = engine.synthesize_verified(
                text.clone(),
                &caps.default_voice,
                None,
                VerificationOptions {
                    policy,
                    ..Default::default()
                },
            )?;
            let mut first_pcm_s = None;
            let mut audio_s = 0.0;
            let mut segment: Option<Export> = None;
            let mut segment_count = 0;
            let mut verdicts = [0usize; 4];
            let mut retries = 0;
            let mut cache_hits = 0;
            let mut failure = None;
            let mut delivered_end = 0;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(SynthesisItem::Verification(report)) => {
                        let at = match report.verdict {
                            VerificationVerdict::Passed => 0,
                            VerificationVerdict::Suspect => 1,
                            VerificationVerdict::ConfirmedError => 2,
                            VerificationVerdict::Unverified => 3,
                        };
                        verdicts[at] += 1;
                        retries += usize::from(report.attempt > 0);
                        cache_hits += usize::from(report.cache_hit);
                        writeln!(reports, "{}", serde_json::to_string(&report)?)?;
                        reports.flush()?;
                        eprintln!(
                            "REPORT {id} {mode} {}..{} {:?} attempt={} elapsed={:.1}s",
                            report.range.start,
                            report.range.end,
                            report.verdict,
                            report.attempt,
                            started.elapsed().as_secs_f64()
                        );
                    }
                    Ok(SynthesisItem::Audio(audio)) => {
                        first_pcm_s.get_or_insert_with(|| started.elapsed().as_secs_f64());
                        if segment
                            .as_ref()
                            .is_none_or(|segment| segment.range != audio.range())
                        {
                            if let Some(segment) = segment.take() {
                                segment.writer.finalize()?;
                            }
                            segment_count += 1;
                            segment = Some(Export::new(
                                &directory.join(format!("{segment_count:04}.wav")),
                                &audio,
                            )?);
                        }
                        let pcm = audio.pcm();
                        audio_s +=
                            pcm.samples.len() as f64 / pcm.channels as f64 / pcm.sample_rate as f64;
                        for &sample in &pcm.samples {
                            segment
                                .as_mut()
                                .expect("active segment")
                                .writer
                                .write_sample(sample)?;
                        }
                        delivered_end = audio.range().end;
                    }
                    Err(error) => {
                        failure = Some(error.to_string());
                        break;
                    }
                }
            }
            if let Some(segment) = segment.take() {
                segment.writer.finalize()?;
            }
            reports.flush()?;
            let elapsed_s = started.elapsed().as_secs_f64();
            let result = json!({"id":id,"mode":mode,"chars":text.chars().count(),"source_bytes":text.len(),
                "text_hash":text_hash(&text),"model":caps.model,"backend":caps.backend,"voice":caps.default_voice,
                "device":args[3],"state":format!("{:?}",stream.state()),"failure":failure,"elapsed_s":elapsed_s,
                "first_pcm_s":first_pcm_s,"audio_s":audio_s,"rtf":if audio_s>0.0 {Some(elapsed_s/audio_s)} else {None},
                "delivered_end":delivered_end,"segments":segment_count,"verdicts":{"passed":verdicts[0],"suspect":verdicts[1],
                    "confirmed_error":verdicts[2],"unverified":verdicts[3]},"retries":retries,"cache_hits":cache_hits});
            fs::write(
                directory.join("summary.json"),
                serde_json::to_vec_pretty(&result)?,
            )?;
            eprintln!(
                "END {id} {mode} {} segments={segment_count} elapsed={elapsed_s:.1}s",
                result["state"]
            );
            summary.push(result);
            fs::write(
                output.join("summary.json"),
                serde_json::to_vec_pretty(&json!({"tts_load_s":tts_load_s,
                "asr_load_s":asr_load_s,"runs":summary}))?,
            )?;
        }
    }
    engine.close().await?;
    Ok(())
}
