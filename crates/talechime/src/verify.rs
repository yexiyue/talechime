//! Report-only command with explicit input/output files, no player or TTS preparation.
use std::path::Path;
use talechime::*;
pub async fn run(text: &Path, audio: &Path, report: &Path, resources: &Path) -> anyhow::Result<()> {
    use tokio::io::AsyncReadExt;
    let mut bytes = Vec::new();
    tokio::fs::File::open(text)
        .await?
        .take(8193)
        .read_to_end(&mut bytes)
        .await?;
    anyhow::ensure!(
        bytes.len() <= 8192,
        "readback source file too large (one segment, max 512 characters)"
    );
    let text = String::from_utf8(bytes)?;
    anyhow::ensure!(
        text.chars().count() <= 512,
        "readback accepts one actual segment, max 512 characters"
    );
    let reader = hound::WavReader::open(audio)?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.sample_format == hound::SampleFormat::Int
            && spec.bits_per_sample == 16
            && spec.sample_rate > 0
            && spec.channels > 0,
        "verify requires framed PCM16 WAV"
    );
    anyhow::ensure!(
        reader.duration() as u64 <= spec.sample_rate as u64 * 30
            && reader.len() as usize <= 4 * 1024 * 1024,
        "readback audio exceeds 30s/16MiB bound"
    );
    let samples = reader
        .into_samples::<i16>()
        .map(|x| x.map(|x| x as f32 / 32768.0))
        .collect::<Result<Vec<_>, _>>()?;
    let verifier = prepare_readback(ReadbackModelOptions::new(resources), |event| {
        if let Event::ResourceState { stage, resource } = event {
            eprintln!("{stage} {resource}");
        }
    })
    .await?;
    let source = SourceSnapshot::new(
        SourceId {
            namespace: "readback-file".into(),
            book: String::new(),
            chapter: String::new(),
        },
        text.clone(),
        &text_hash(&text),
    )?;
    let report_value = verifier
        .report(
            ReadbackRequest {
                source: &source,
                range: TextRange {
                    start: 0,
                    end: text.len(),
                },
                spoken_text: &text,
                backend: "external",
                model: None,
                voice: "external",
                style: None,
                attempt: 0,
            },
            &Pcm {
                samples,
                sample_rate: spec.sample_rate,
                channels: spec.channels,
            },
            &VerificationOptions::default(),
        )
        .await?;
    tokio::fs::write(report, serde_json::to_vec_pretty(&report_value)?).await?;
    eprintln!("回读 {:?} · {}", report_value.verdict, report.display());
    Ok(())
}
