use super::*;
use std::io::{BufWriter, Cursor, Seek, SeekFrom};

#[test]
fn buffered_wav_preserves_bytes_and_no_clobber() -> anyhow::Result<()> {
    let directory = tempfile::tempdir()?;
    let samples = [-1.5, -1., -0.0, 0., 0.25, 1., 1.5, f32::MIN_POSITIVE];
    for pcm16 in [false, true] {
        let path = directory.path().join(format!("{pcm16}.wav"));
        let mut reference = Cursor::new(Vec::new());
        write_wav_to(&mut reference, &samples, 48000, 2, pcm16)?;
        write_wav(&path, &samples, 48000, 2, pcm16)?;
        let bytes = std::fs::read(&path)?;
        assert_eq!(bytes, reference.into_inner());
        assert!(write_wav(&path, &[0.; 8], 48000, 2, pcm16).is_err());
        assert_eq!(std::fs::read(path)?, bytes);
    }
    Ok(())
}

struct FailingWriter {
    inner: Cursor<Vec<u8>>,
    fail_write: bool,
}
impl Write for FailingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.fail_write {
            Err(std::io::Error::other("injected write failure"))
        } else {
            self.inner.write(bytes)
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Err(std::io::Error::other("injected flush failure"))
    }
}
impl Seek for FailingWriter {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(position)
    }
}

#[test]
fn buffered_wav_propagates_write_flush_and_nonfinite_errors() {
    for fail_write in [false, true] {
        let sink = FailingWriter {
            inner: Cursor::new(Vec::new()),
            fail_write,
        };
        assert!(write_wav_to(BufWriter::new(sink), &[0.25; 8], 48000, 2, false).is_err());
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(write_wav_to(Cursor::new(Vec::new()), &[value; 2], 48000, 2, false).is_err());
    }
}

#[test]
#[ignore = "explicit pinned local WAV and new output directory; export cost only, no inference"]
fn real_audio_export_cost_and_byte_equivalence() -> anyhow::Result<()> {
    let required = |name| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .ok_or_else(|| anyhow::anyhow!("{name} required"))
    };
    let input = required("TALECHIME_PERF_EXPORT_INPUT")?;
    let output = required("TALECHIME_PERF_EXPORT_OUTPUT")?;
    let expected = std::env::var("TALECHIME_PERF_EXPORT_INPUT_SHA256")?;
    let bytes = std::fs::read(input)?;
    ensure!(hash(&bytes) == expected, "input WAV identity changed");
    let mut reader = hound::WavReader::new(Cursor::new(bytes))?;
    let spec = reader.spec();
    ensure!(spec.sample_format == hound::SampleFormat::Float && spec.bits_per_sample == 32);
    let samples = reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?;
    ensure!(!samples.is_empty() && samples.len() <= 16 * 1024 * 1024);
    ensure!(samples.iter().all(|v| v.is_finite()));
    std::fs::create_dir(&output)?;
    let mut rows = Vec::new();
    for round in 0..6 {
        for pcm16 in [false, true] {
            let mut elapsed = [0.; 2];
            let paths = [
                output.join(format!("{round}-{pcm16}-direct.wav")),
                output.join(format!("{round}-{pcm16}-buffered.wav")),
            ];
            let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
            for index in order {
                let started = Instant::now();
                if index == 0 {
                    let file = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&paths[index])?;
                    write_wav_to(file, &samples, spec.sample_rate, spec.channels, pcm16)?;
                } else {
                    write_wav(
                        &paths[index],
                        &samples,
                        spec.sample_rate,
                        spec.channels,
                        pcm16,
                    )?;
                }
                elapsed[index] = started.elapsed().as_secs_f64();
            }
            let direct = std::fs::read(&paths[0])?;
            let buffered = std::fs::read(&paths[1])?;
            assert_eq!(direct, buffered);
            rows.push(serde_json::json!({
                "round":round,"warmup":round==0,"pcm16":pcm16,"order":order,
                "direct_seconds":elapsed[0],"buffered_seconds":elapsed[1],
                "bytes":direct.len(),"sha256":hash(&direct)
            }));
        }
    }
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.join("results.json"))?;
    serde_json::to_writer_pretty(
        file,
        &serde_json::json!({
            "kind":"artifact_export_cost_not_inference",
            "input_sha256":expected,"samples":samples.len(),"sample_rate":spec.sample_rate,
            "channels":spec.channels,"rounds":rows
        }),
    )?;
    Ok(())
}
