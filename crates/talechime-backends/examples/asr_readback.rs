//! Explicit local real-model verification; ordinary tests never run or download this.
use serde::Deserialize;
use std::{path::Path, rc::Rc, sync::Arc, time::Instant};
use talechime_backends::asr;
use tts_core::{SourceSnapshot, backend::Pcm, verification::*};
use tts_protocol::{SourceId, TextRange};
#[derive(Deserialize)]
struct Case {
    id: String,
    file: String,
    text: String,
}
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 5,
        "usage: asr_readback QWEN_DIR SENSEVOICE_DIR CORPUS_JSON OUTPUT_JSON"
    );
    let started = Instant::now();
    let primary = asr::load_qwen(Path::new(&args[1])).await?;
    let reviewer = asr::load_sensevoice(Path::new(&args[2]), 4).await?;
    let verifier = Verifier::new(primary.clone(), reviewer.clone())?;
    let load_s = started.elapsed().as_secs_f64();
    let cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let mut rows = Vec::new();
    let mut cancellation_s = None;
    for case in cases {
        let reader = hound::WavReader::open(&case.file)?;
        let spec = reader.spec();
        anyhow::ensure!(
            spec.sample_format == hound::SampleFormat::Int && spec.bits_per_sample == 16,
            "probe requires PCM16 WAV"
        );
        let samples = reader
            .into_samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?;
        let audio = Arc::new(ReadbackAudio {
            samples: samples.clone(),
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        });
        if cancellation_s.is_none() {
            let start = Instant::now();
            let request = primary.request(audio.clone());
            let mut completion = request.completion.expect("native request receipt");
            anyhow::ensure!(
                tokio::time::timeout(std::time::Duration::from_millis(1), request.future)
                    .await
                    .is_err(),
                "cancellation probe unexpectedly completed"
            );
            completion.wait_for(|done| *done).await?;
            cancellation_s = Some(start.elapsed().as_secs_f64());
        }
        let start = Instant::now();
        let qwen = primary.transcribe(audio.clone()).await;
        let qwen_s = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let sense = reviewer.transcribe(audio).await;
        let sense_s = start.elapsed().as_secs_f64();
        let source = SourceSnapshot::new(
            SourceId {
                namespace: "probe".into(),
                book: "".into(),
                chapter: case.id.clone(),
            },
            case.text.clone(),
            &tts_protocol::text_hash(&case.text),
        )?;
        let pcm = Pcm {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        };
        // Inject the independently recorded transcripts to exercise production comparison without another model run.
        let report = if let (Ok(a), Ok(b)) = (&qwen, &sense) {
            struct Recorded(RecognizerIdentity, String);
            impl Recognizer for Recorded {
                fn identity(&self) -> RecognizerIdentity {
                    self.0.clone()
                }
                fn transcribe(&self, _: Arc<ReadbackAudio>) -> Recognition<'_> {
                    Box::pin(async { Ok(self.1.clone()) })
                }
            }
            let readback = Verifier::new(
                Rc::new(Recorded(primary.identity(), a.clone())),
                Rc::new(Recorded(reviewer.identity(), b.clone())),
            )?;
            Some(
                readback
                    .report(
                        ReadbackRequest {
                            source: &source,
                            range: TextRange {
                                start: 0,
                                end: case.text.len(),
                            },
                            spoken_text: &case.text,
                            backend: "probe",
                            model: None,
                            voice: "fixed",
                            style: None,
                            attempt: 0,
                        },
                        &pcm,
                        &VerificationOptions::default(),
                    )
                    .await?,
            )
        } else {
            None
        };
        eprintln!("{} qwen={qwen:?} sense={sense:?}", case.id);
        rows.push(serde_json::json!({"id":case.id,"qwen":qwen.as_ref().ok(),"qwen_error":qwen.err().map(|e|e.to_string()),
            "sensevoice":sense.as_ref().ok(),"sensevoice_error":sense.err().map(|e|e.to_string()),"qwen_s":qwen_s,"sensevoice_s":sense_s,"report":report}));
        std::fs::write(
            &args[4],
            serde_json::to_vec_pretty(
                &serde_json::json!({"load_s":load_s,"cancellation_settled_s":cancellation_s,"rows":rows}),
            )?,
        )?;
    }
    drop(verifier);
    Ok(())
}
