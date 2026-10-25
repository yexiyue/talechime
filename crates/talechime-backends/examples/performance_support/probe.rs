use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};
use talechime_core::{SynthesisOptions, SynthesisState, SynthesisStream, backend::Backend};
use tracing::Instrument;
#[cfg(test)]
#[path = "wav_tests.rs"]
mod wav_tests;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    attempt_id: String,
    resources_root: PathBuf,
    backend: String,
    model: String,
    device: talechime_protocol::Device,
    corpus: PathBuf,
    corpus_sha256: String,
    voice: String,
    style: Option<String>,
    continuation: bool,
    seed: u64,
    params: std::collections::BTreeMap<String, talechime_protocol::ParamValue>,
    warmup_rounds: u32,
    hot_rounds: u32,
    max_pcm_bytes: usize,
    fixture: bool,
    #[serde(default = "request_timeout")]
    request_timeout_seconds: u64,
    #[serde(default = "observation_mode")]
    observation_mode: String,
    #[serde(default = "artifact_budget")]
    max_artifact_bytes: u64,
}

fn artifact_budget() -> u64 {
    8 * 1024 * 1024 * 1024
}

fn observation_mode() -> String {
    "baseline".into()
}

fn request_timeout() -> u64 {
    1800
}

#[derive(Debug)]
struct RequestTimeout;
impl std::fmt::Display for RequestTimeout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("request timeout; stream cancellation requested")
    }
}
impl std::error::Error for RequestTimeout {}

struct Events {
    attempt: String,
    sequence: u64,
    started: Instant,
}
impl Events {
    fn send(&mut self, kind: &str, payload: impl Serialize) -> anyhow::Result<()> {
        let message = serde_json::json!({"schema":"talechime/performance/1", "attempt_id":self.attempt,"sequence":self.sequence,"offset_seconds":self.started.elapsed().as_secs_f64(),"kind":kind,"payload":payload});
        let stdout = std::io::stdout();
        let mut writer = stdout.lock();
        serde_json::to_writer(&mut writer, &message)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        self.sequence += 1;
        Ok(())
    }
}

pub async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let request = PathBuf::from(args.next().context("request JSON required")?);
    let output = PathBuf::from(args.next().context("attempt directory required")?);
    ensure!(
        args.next().is_none(),
        "usage: performance_probe REQUEST_JSON OUTPUT_DIR"
    );
    ensure!(
        request.metadata()?.len() <= 1024 * 1024,
        "request exceeds budget"
    );
    let request: Request = serde_json::from_reader(std::fs::File::open(request)?)?;
    ensure!(
        request.schema == "talechime/performance-request/1",
        "unsupported request schema"
    );
    ensure!(
        request.hot_rounds > 0 && request.hot_rounds <= 1000 && request.warmup_rounds <= 1000,
        "invalid repetition budget"
    );
    ensure!(
        request.max_pcm_bytes > 0 && request.max_pcm_bytes <= 512 * 1024 * 1024,
        "PCM budget must be within 512 MiB"
    );
    ensure!(
        request.request_timeout_seconds > 0 && request.request_timeout_seconds <= 1800,
        "request timeout must be within 30 minutes"
    );
    ensure!(
        request.max_artifact_bytes > 0 && request.max_artifact_bytes <= artifact_budget(),
        "artifact budget must be within 8 GiB per process"
    );
    let rounds = u64::from(request.warmup_rounds) + u64::from(request.hot_rounds);
    let bound = rounds * (request.max_pcm_bytes as u64 * 3 + 16 * 1024 * 1024) + 32 * 1024 * 1024;
    ensure!(
        bound <= request.max_artifact_bytes,
        "declared PCM/repetition budgets exceed artifact budget; freeze a smaller retention limit or a separate run"
    );
    std::fs::create_dir_all(&output)?;
    let mut events = Events {
        attempt: request.attempt_id.clone(),
        sequence: 0,
        started: Instant::now(),
    };
    let telemetry = crate::telemetry::Telemetry::install(&request.observation_mode)?;
    let span = tracing::info_span!(target:"talechime_perf","attempt",attempt_id=%request.attempt_id,backend=%request.backend,model=%request.model);
    let result = run(&request, &output, &mut events, &telemetry)
        .instrument(span)
        .await;
    if result.is_err() {
        metrics::counter!("talechime.errors","stage"=>"attempt").increment(1);
    }
    telemetry.finish(&output, result.is_ok())?;
    match result {
        Ok(()) => events.send("terminal", serde_json::json!({"outcome":"success"})),
        Err(error) => {
            events.send(
                "terminal",
                serde_json::json!({"outcome":if error.downcast_ref::<RequestTimeout>().is_some() { "timeout" } else { "failure" },"error":error.to_string()}),
            )?;
            Err(error)
        }
    }
}

async fn run(
    request: &Request,
    output: &Path,
    events: &mut Events,
    telemetry: &crate::telemetry::Telemetry,
) -> anyhow::Result<()> {
    ensure!(
        request.corpus.metadata()?.len() <= 1024 * 1024,
        "corpus exceeds 1 MiB"
    );
    let bytes = std::fs::read(&request.corpus)?;
    ensure!(
        hash(&bytes) == request.corpus_sha256,
        "corpus identity changed"
    );
    let text = std::str::from_utf8(&bytes)?.to_owned();
    ensure!(!text.trim().is_empty(), "empty corpus");
    let verify_started = Instant::now();
    let registry = talechime_backends::Registry::new(Some(request.resources_root.clone()))?;
    let receipt = if request.fixture {
        ensure!(
            request.backend == "fixture"
                && request.model == "deterministic"
                && request.device == talechime_protocol::Device::Cpu,
            "fixture cannot impersonate a real model or GPU"
        );
        serde_json::json!({"verification":"synthetic_fixture"})
    } else {
        serde_json::to_value(talechime_backends::performance::verify(
            &registry,
            &request.backend,
            &request.model,
        )?)?
    };
    let verify_seconds = verify_started.elapsed().as_secs_f64();
    let load_started = Instant::now();
    let backend: Rc<dyn Backend> = if request.fixture {
        Rc::new(Fixture)
    } else {
        talechime_backends::performance::load(
            &registry,
            &request.backend,
            &request.model,
            request.device,
        )
        .await?
    };
    let load_seconds = load_started.elapsed().as_secs_f64();
    let caps = backend.capabilities();
    ensure!(
        caps.voices.iter().any(|v| v == &request.voice),
        "voice not in production catalog"
    );
    parameters(request)
        .validate(&caps.parameters)
        .map_err(anyhow::Error::msg)?;
    let reference_warmup_seconds = prepare_reference(request, backend.clone()).await?;
    events.send("prepared", serde_json::json!({"verify_seconds":verify_seconds,"load_seconds":load_seconds,"resource_receipt":receipt,"capabilities":caps,"reference_prepare_seconds":null,"reference_prepare_reason":"reference preparation cannot be isolated from load or synthesis; optional short synthesis is separately reported","reference_warmup_seconds":reference_warmup_seconds}))?;
    for index in 0..request.warmup_rounds + request.hot_rounds {
        let span = tracing::info_span!(target:"talechime_perf","perf_round",round=index);
        let round = round(
            request,
            output,
            &text,
            backend.clone(),
            index,
            telemetry,
            span.clone(),
        )
        .instrument(span)
        .await?;
        events.send("round", round)?;
    }
    drop(backend);
    Ok(())
}

/// A single untimed short synthesis populates reproducibly keyed reference caches.
/// Its full cost is reported separately; it is not a pure reference-encoding metric.
async fn prepare_reference(
    request: &Request,
    backend: Rc<dyn Backend>,
) -> anyhow::Result<Option<f64>> {
    if backend.cache_identity().is_none() {
        return Ok(None);
    }
    let started = Instant::now();
    let mut stream = SynthesisStream::start_with_options(
        backend,
        SHORT_CORPUS,
        &request.voice,
        request.style.clone(),
        None,
        SynthesisOptions {
            continuation: request.continuation,
            params: parameters(request),
            seed: talechime_core::SeedPolicy::Pinned(request.seed),
            ..Default::default()
        },
    )?;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(request.request_timeout_seconds),
        async {
            let mut received = false;
            while let Some(block) = stream.recv().await {
                block?.pcm().duration_ms()?;
                received = true;
            }
            ensure!(
                received && stream.state() == SynthesisState::Completed,
                "reference warmup did not complete with valid PCM"
            );
            Ok::<(), anyhow::Error>(())
        },
    )
    .await;
    stream.cancel().await;
    result.map_err(|_| RequestTimeout)??;
    Ok(Some(started.elapsed().as_secs_f64()))
}

async fn round(
    request: &Request,
    output: &Path,
    text: &str,
    backend: Rc<dyn Backend>,
    index: u32,
    telemetry: &crate::telemetry::Telemetry,
    parent: tracing::Span,
) -> anyhow::Result<serde_json::Value> {
    let stage = if index < request.warmup_rounds {
        "warmup"
    } else {
        "hot"
    };
    let delivery_boundary = {
        let caps = backend.capabilities();
        if caps.native_streaming {
            "native_streaming_first_pcm"
        } else {
            "segment_delivery_first_pcm"
        }
    };
    metrics::counter!("talechime.requests","stage"=>stage).increment(1);
    let expected_ranges = expected_ranges(backend.as_ref(), text).await?;
    let backend = Rc::new(crate::observed::Observed {
        inner: backend,
        parent,
        calls: std::cell::Cell::new(0),
    });
    let started = Instant::now();
    let offset = telemetry.offset_at(started);
    let mut stream = SynthesisStream::start_with_options(
        backend,
        text.to_owned(),
        &request.voice,
        request.style.clone(),
        None,
        SynthesisOptions {
            continuation: request.continuation,
            params: parameters(request),
            seed: talechime_core::SeedPolicy::Pinned(request.seed),
            ..Default::default()
        },
    )?;
    let mut samples = Vec::new();
    let mut format = None;
    let mut first = None;
    let mut observations = Vec::new();
    let mut chunks = Vec::new();
    let mut segments: Vec<(talechime_protocol::TextRange, usize, usize)> = Vec::new();
    let deadline = tokio::time::Instant::now()
        + std::time::Duration::from_secs(request.request_timeout_seconds);
    let completion = async {
        loop {
            let block = tokio::select! {
                block = stream.recv() => block,
                _ = tokio::time::sleep_until(deadline) => {
                    return Err(RequestTimeout.into());
                }
            };
            let Some(block) = block else {
                break;
            };
            let block = block?;
            let time = started.elapsed().as_secs_f64();
            first.get_or_insert(time);
            let pcm = block.pcm();
            pcm.duration_ms()?;
            let current = (pcm.sample_rate, pcm.channels);
            ensure!(
                format.is_none_or(|value| value == current),
                "format changed within request"
            );
            format = Some(current);
            ensure!(
                samples.len() + pcm.samples.len() <= request.max_pcm_bytes / 4,
                "PCM retention budget exceeded"
            );
            let range = block.range();
            ensure!(
                chunks.len() < 65536,
                "chunk evidence budget exceeded (65536 per request)"
            );
            chunks.push((
                time,
                samples.len(),
                samples.len() + pcm.samples.len(),
                range,
            ));
            if segments.last().is_none_or(|(r, _, _)| *r != range) {
                ensure!(
                    segments.len() < 1024,
                    "segment evidence budget exceeded (1024 per request)"
                );
                segments.push((range, samples.len(), samples.len()));
            }
            samples.extend_from_slice(&pcm.samples);
            segments.last_mut().unwrap().2 = samples.len();
        }
        ensure!(
            stream.state() == SynthesisState::Completed,
            "generation did not complete normally"
        );
        let actual_ranges = segments
            .iter()
            .map(|(range, _, _)| *range)
            .collect::<Vec<_>>();
        ensure!(
            actual_ranges == expected_ranges,
            "consumer ranges differ from production segmentation"
        );
        ensure!(
            first.is_some_and(|first| first > 0.0),
            "first PCM boundary was not observed"
        );
        Ok::<(), anyhow::Error>(())
    }
    .await;
    if let Err(error) = completion {
        stream.cancel().await;
        return Err(error);
    }
    let elapsed = started.elapsed().as_secs_f64();
    metrics::histogram!("talechime.request_seconds","stage"=>stage).record(elapsed);
    let (sample_rate, channels) = format.context("empty PCM")?;
    let audio_seconds = samples.len() as f64 / sample_rate as f64 / channels as f64;
    // All audio serialization and hashing are outside the measured boundary.
    let mut artifacts = Vec::new();
    for (suffix, start, end, derived) in
        std::iter::once(("full".to_owned(), 0, samples.len(), None)).chain(
            segments
                .iter()
                .enumerate()
                .map(|(i, (_, start, end))| (format!("segment-{i}"), *start, *end, Some("full"))),
        )
    {
        for pcm16 in [false, true] {
            let name = format!(
                "round-{index}-{suffix}-{}.wav",
                if pcm16 { "pcm16" } else { "f32" }
            );
            write_wav(
                &output.join(&name),
                &samples[start..end],
                sample_rate,
                channels,
                pcm16,
            )?;
            let bytes = std::fs::read(output.join(&name))?;
            let derived_from = if pcm16 {
                Some(format!("round-{index}-{suffix}-f32.wav"))
            } else {
                derived.map(|_| format!("round-{index}-full-f32.wav"))
            };
            artifacts.push(serde_json::json!({"path":name,"sha256":hash(&bytes),"bytes":bytes.len(),"kind":if pcm16 {"audio_pcm16"} else {"audio_f32"},"derived_from":derived_from}));
        }
    }
    let mut pcm_hash = Sha256::new();
    pcm_hash.update(sample_rate.to_le_bytes());
    pcm_hash.update(channels.to_le_bytes());
    for sample in &samples {
        pcm_hash.update(sample.to_le_bytes());
    }
    let (raw, dropped) = telemetry.raw(index);
    let reserved = raw
        .iter()
        .filter(|event| event.name == "audio_budget_reserved_seconds")
        .filter_map(|event| event.value)
        .reduce(f64::max);
    observations.push(serde_json::json!({"name":"audio_budget_peak_reserved_seconds","parent":"request","boundary":"peak outstanding time permits; includes pending and active/consumer-retained audio","location":"core_prefetch_budget","reading":{"value":reserved,"unit":"seconds","scope":"pending_and_active_audio","source":"prefetch semaphore milliseconds / 1000","interval_ms":null,"status":if reserved.is_some(){"available"}else{"unavailable"},"reason":reserved.is_none().then_some("budget observations unavailable or disabled")}}));
    for (kind, name) in [
        ("owner_pcm_ready", "owner_first_pcm"),
        ("adapter_channel_enqueue", "adapter_first_pcm"),
    ] {
        let value = raw
            .iter()
            .filter(|r| r.name == kind)
            .map(|r| r.offset_seconds - offset)
            .filter(|n| *n >= 0.0)
            .reduce(f64::min);
        observations.push(serde_json::json!({"name":name,"parent":"request","boundary":"round start to owner PCM ready / channel enqueue; host wall time","location":if kind=="owner_pcm_ready"{"owner"}else{"adapter_channel"},"reading":{"value":value,"unit":"seconds","scope":"request","source":"correlated_owner_event","interval_ms":null,"status":if value.is_some(){"available"}else{"unavailable"},"reason":value.is_none().then_some("owner observation unavailable or collection disabled")}}));
    }
    let waits = raw
        .iter()
        .filter(|event| event.name == "channel_wait")
        .collect::<Vec<_>>();
    let wait_value =
        (!waits.is_empty() && dropped == 0 && waits.iter().all(|event| event.value.is_some()))
            .then(|| waits.iter().filter_map(|event| event.value).sum::<f64>());
    observations.push(serde_json::json!({"name":"channel_wait","parent":"request","boundary":"sum of bounded sends/reserves; included in request wall time","location":"owner","reading":{"value":wait_value,"unit":"seconds","scope":"request_owner_channels","source":"raw_owner_events; not aggregated histogram","interval_ms":null,"status":if wait_value.is_some(){"available"}else{"unavailable"},"reason":wait_value.is_none().then_some("owner durations unavailable, disabled or dropped")}}));
    observations.push(observation("queue_audio_seconds",None,Some("exact queue-only audio ownership is not exposed; retained consumer audio must not be counted as queued")));
    let ranges = segments.iter().map(|(range, _, _)| serde_json::json!({"range":range,"text_sha256":hash(&text.as_bytes()[range.start..range.end])})).collect::<Vec<_>>();
    let ranges_name = format!("round-{index}-ranges.json");
    let ranges_bytes = serde_json::to_vec_pretty(
        &serde_json::json!({"segments":ranges,"sample_rate":sample_rate,"channels":channels,"f32":"original finite interleaved samples","pcm16":"clamp[-1,1], multiply 32767, round to nearest i16","chunks":chunks,"chunk_fields":["consumer_offset_seconds","sample_start","sample_end","source_range"],"raw_owner_events":raw,"dropped_owner_events":dropped}),
    )?;
    std::fs::write(output.join(&ranges_name), &ranges_bytes)?;
    artifacts.push(serde_json::json!({"path":ranges_name,"sha256":hash(&ranges_bytes),"bytes":ranges_bytes.len(),"kind":"source_ranges","derived_from":null}));
    Ok(
        serde_json::json!({"index":index,"warmup":index<request.warmup_rounds,"outcome":"success","synthesis_seconds":elapsed,"audio_seconds":audio_seconds,"first_pcm_seconds":first,"delivery_boundary":delivery_boundary,"output_sha256":format!("{:x}",pcm_hash.finalize()),"observations":observations,"artifacts":artifacts}),
    )
}

fn observation(name: &str, value: Option<f64>, reason: Option<&str>) -> serde_json::Value {
    serde_json::json!({"name":name,"parent":"request","boundary":"consumer receive; host wall time","location":"consumer","reading":{"value":value,"unit":"seconds","scope":"request","source":"monotonic_host_clock","interval_ms":null,"status":if value.is_some(){"available"}else{"unavailable"},"reason":reason}})
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_wav(
    path: &Path,
    samples: &[f32],
    rate: u32,
    channels: u16,
    pcm16: bool,
) -> anyhow::Result<()> {
    ensure!(!path.exists(), "audio artifact already exists");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    write_wav_to(
        std::io::BufWriter::with_capacity(64 * 1024, file),
        samples,
        rate,
        channels,
        pcm16,
    )
}

fn write_wav_to(
    output: impl std::io::Write + std::io::Seek,
    samples: &[f32],
    rate: u32,
    channels: u16,
    pcm16: bool,
) -> anyhow::Result<()> {
    let mut writer = hound::WavWriter::new(
        output,
        hound::WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: if pcm16 { 16 } else { 32 },
            sample_format: if pcm16 {
                hound::SampleFormat::Int
            } else {
                hound::SampleFormat::Float
            },
        },
    )?;
    for sample in samples {
        ensure!(sample.is_finite(), "non-finite PCM");
        if pcm16 {
            writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
        } else {
            writer.write_sample(*sample)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

fn parameters(request: &Request) -> talechime_core::params::GenerationParams {
    let mut params = talechime_core::params::GenerationParams::new();
    for (name, value) in &request.params {
        params.insert(name, value.clone());
    }
    params
}

const SHORT_CORPUS: &str = "你好，欢迎使用叙铃。";

async fn expected_ranges(
    backend: &dyn Backend,
    text: &str,
) -> anyhow::Result<Vec<talechime_protocol::TextRange>> {
    let mut ranges = Vec::new();
    let mut source = text;
    while let Some(segment) = backend
        .next_segment(source)
        .await
        .map_err(anyhow::Error::msg)?
    {
        let offset = source.as_ptr() as usize - text.as_ptr() as usize;
        let range = talechime_protocol::TextRange {
            start: offset + segment.start,
            end: offset + segment.end,
        };
        anyhow::ensure!(
            range.start < range.end && range.end <= text.len() && range.is_valid(text),
            "backend produced an invalid production range"
        );
        ranges.push(range);
        source = &text[range.end..];
    }
    anyhow::ensure!(
        !ranges.is_empty(),
        "production segmentation produced no synthesis segments"
    );
    Ok(ranges)
}

struct Fixture;
impl Backend for Fixture {
    fn capabilities(&self) -> talechime_protocol::Capabilities {
        talechime_protocol::Capabilities {
            backend: "fixture".into(),
            model: Some("deterministic".into()),
            voices: vec!["fixture".into()],
            default_voice: "fixture".into(),
            model_name: "Deterministic performance fixture".into(),
            voice_names: Default::default(),
            compiled_devices: vec![talechime_protocol::Device::Cpu],
            native_streaming: true,
            style: false,
            cloning: false,
            pronunciation: false,
            continuation: false,
            parameters: vec![],
        }
    }
    fn stream<'a>(
        &'a self,
        _request: talechime_core::backend::SegmentRequest<'a>,
    ) -> talechime_core::backend::Streaming<'a> {
        Box::pin(async {
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            let span = tracing::Span::current();
            tokio::task::spawn_local(async move {
                for _ in 0..4 {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    tracing::info!(target:"talechime_perf",event="owner_pcm_ready");
                    metrics::counter!("talechime.chunks","stage"=>"fixture_owner").increment(1);
                    let waiting=Instant::now();
                    let sent=tx
                        .send(Ok(talechime_core::backend::AudioChunk::Pcm(
                            talechime_core::backend::Pcm {
                                samples: vec![0.125; 2400],
                                sample_rate: 24000,
                                channels: 1,
                            },
                        )))
                        .await;
                    let seconds=waiting.elapsed().as_secs_f64();
                    metrics::histogram!("talechime.channel_wait_seconds","stage"=>"fixture_owner").record(seconds);
                    tracing::info!(target:"talechime_perf",event="channel_wait",seconds);
                    if sent.is_err() {
                        return;
                    }
                    tracing::info!(target:"talechime_perf",event="adapter_channel_enqueue");
                }
                let _ = tx.send(Ok(talechime_core::backend::AudioChunk::End)).await;
            }.instrument(span));
            Ok(rx)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    #[derive(Clone, Copy)]
    enum Ending {
        Success,
        Truncated,
        Error,
        NonFinite,
        Hang,
    }
    struct Controlled {
        ending: Ending,
        closed: Arc<AtomicBool>,
    }
    impl Backend for Controlled {
        fn capabilities(&self) -> talechime_protocol::Capabilities {
            Fixture.capabilities()
        }
        fn stream<'a>(
            &'a self,
            _request: talechime_core::backend::SegmentRequest<'a>,
        ) -> talechime_core::backend::Streaming<'a> {
            let ending = self.ending;
            let closed = self.closed.clone();
            Box::pin(async move {
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                tokio::task::spawn_local(async move {
                    struct Closed(Arc<AtomicBool>);
                    impl Drop for Closed {
                        fn drop(&mut self) {
                            self.0.store(true, Ordering::Release);
                        }
                    }
                    let _closed = Closed(closed);
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    let samples = if matches!(ending, Ending::NonFinite) {
                        vec![f32::NAN; 8]
                    } else {
                        vec![0.125; 8]
                    };
                    if tx
                        .send(Ok(talechime_core::backend::AudioChunk::Pcm(
                            talechime_core::backend::Pcm {
                                samples,
                                sample_rate: 24000,
                                channels: 1,
                            },
                        )))
                        .await
                        .is_err()
                    {
                        return;
                    }
                    match ending {
                        Ending::Success | Ending::NonFinite => {
                            let _ = tx.send(Ok(talechime_core::backend::AudioChunk::End)).await;
                        }
                        Ending::Error => {
                            let _ = tx
                                .send(Err(talechime_core::backend::BackendError::Synthesis(
                                    "injected".into(),
                                )))
                                .await;
                        }
                        Ending::Truncated => {}
                        Ending::Hang => tx.closed().await,
                    }
                });
                Ok(rx)
            })
        }
    }
    struct Cached {
        inner: Controlled,
        calls: std::cell::Cell<u32>,
    }
    impl Backend for Cached {
        fn capabilities(&self) -> talechime_protocol::Capabilities {
            self.inner.capabilities()
        }
        fn cache_identity(&self) -> Option<String> {
            Some("fixture-cache".into())
        }
        fn stream<'a>(
            &'a self,
            request: talechime_core::backend::SegmentRequest<'a>,
        ) -> talechime_core::backend::Streaming<'a> {
            self.calls.set(self.calls.get() + 1);
            self.inner.stream(request)
        }
    }
    #[tokio::test(flavor = "current_thread")]
    async fn reference_preparation_is_separate_and_not_repeated_in_timed_rounds()
    -> anyhow::Result<()> {
        tokio::task::LocalSet::new()
            .run_until(async {
                let backend = Rc::new(Cached {
                    inner: Controlled {
                        ending: Ending::Success,
                        closed: Arc::new(AtomicBool::new(false)),
                    },
                    calls: std::cell::Cell::new(0),
                });
                let request = request();
                let preparation = prepare_reference(&request, backend.clone()).await?;
                assert!(preparation.is_some_and(|seconds| seconds > 0.0));
                assert_eq!(backend.calls.get(), 1);
                let telemetry = crate::telemetry::Telemetry::install("off")?;
                let output = tempfile::tempdir()?;
                for index in 0..2 {
                    round(
                        &request,
                        output.path(),
                        "fixture.",
                        backend.clone(),
                        index,
                        &telemetry,
                        tracing::Span::none(),
                    )
                    .await?;
                }
                assert_eq!(
                    backend.calls.get(),
                    3,
                    "only one synthesis per measured round"
                );
                let closed = Arc::new(AtomicBool::new(false));
                let hung = Rc::new(Cached {
                    inner: Controlled {
                        ending: Ending::Hang,
                        closed: closed.clone(),
                    },
                    calls: std::cell::Cell::new(0),
                });
                let error = prepare_reference(&request, hung).await.unwrap_err();
                assert!(error.downcast_ref::<RequestTimeout>().is_some());
                tokio::time::timeout(std::time::Duration::from_secs(1), async {
                    while !closed.load(Ordering::Acquire) {
                        tokio::task::yield_now().await;
                    }
                })
                .await?;
                Ok::<(), anyhow::Error>(())
            })
            .await
    }
    fn request() -> Request {
        serde_json::from_value(serde_json::json!({"schema":"talechime/performance-request/1","attempt_id":"fixture","resources_root":"not_used","backend":"fixture","model":"deterministic","device":"cpu","corpus":"not_used","corpus_sha256":hash(b"fixture."),"voice":"fixture","style":null,"continuation":false,"seed":42,"params":{},"warmup_rounds":1,"hot_rounds":1,"max_pcm_bytes":1024,"fixture":true,"request_timeout_seconds":1,"observation_mode":"off"})).unwrap()
    }
    #[tokio::test(flavor = "current_thread")]
    async fn invalid_streams_and_pcm_budget_do_not_become_success() -> anyhow::Result<()> {
        tokio::task::LocalSet::new()
            .run_until(async {
                let telemetry = crate::telemetry::Telemetry::install("off")?;
                for ending in [
                    Ending::Truncated,
                    Ending::Error,
                    Ending::NonFinite,
                    Ending::Hang,
                ] {
                    let output = tempfile::tempdir()?;
                    let closed = Arc::new(AtomicBool::new(false));
                    let backend = Rc::new(Controlled {
                        ending,
                        closed: closed.clone(),
                    });
                    assert!(
                        round(
                            &request(),
                            output.path(),
                            "fixture.",
                            backend,
                            0,
                            &telemetry,
                            tracing::Span::none()
                        )
                        .await
                        .is_err()
                    );
                    tokio::time::timeout(std::time::Duration::from_secs(1), async {
                        while !closed.load(Ordering::Acquire) {
                            tokio::task::yield_now().await;
                        }
                    })
                    .await?;
                    assert!(!output.path().join("round-0-full-f32.wav").exists());
                }
                let mut request = request();
                request.max_pcm_bytes = 4;
                let output = tempfile::tempdir()?;
                assert!(
                    round(
                        &request,
                        output.path(),
                        "fixture.",
                        Rc::new(Controlled {
                            ending: Ending::Success,
                            closed: Arc::new(AtomicBool::new(false))
                        }),
                        0,
                        &telemetry,
                        tracing::Span::none()
                    )
                    .await
                    .is_err()
                );
                Ok::<(), anyhow::Error>(())
            })
            .await
    }
    #[tokio::test(flavor = "current_thread")]
    async fn delayed_backend_exports_finite_f32_and_explicit_pcm16_after_timing()
    -> anyhow::Result<()> {
        tokio::task::LocalSet::new()
            .run_until(async {
                let telemetry = crate::telemetry::Telemetry::install("off")?;
                let output = tempfile::tempdir()?;
                let result = round(
                    &request(),
                    output.path(),
                    "fixture.",
                    Rc::new(Controlled {
                        ending: Ending::Success,
                        closed: Arc::new(AtomicBool::new(false)),
                    }),
                    0,
                    &telemetry,
                    tracing::Span::none(),
                )
                .await?;
                assert!(result["synthesis_seconds"].as_f64().unwrap() >= 0.020);
                let mut f32 = hound::WavReader::open(output.path().join("round-0-full-f32.wav"))?;
                assert_eq!(f32.spec().sample_format, hound::SampleFormat::Float);
                assert!(f32.samples::<f32>().all(|s| s.unwrap().is_finite()));
                let pcm16 = hound::WavReader::open(output.path().join("round-0-full-pcm16.wav"))?;
                assert_eq!(pcm16.spec().bits_per_sample, 16);
                assert!(
                    result["artifacts"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|a| a["kind"] == "audio_pcm16"
                            && a["derived_from"] == "round-0-full-f32.wav")
                );
                Ok::<(), anyhow::Error>(())
            })
            .await
    }
    #[tokio::test(flavor = "current_thread")]
    async fn multi_segment_success_preserves_production_ranges_and_finite_pcm() -> anyhow::Result<()>
    {
        tokio::task::LocalSet::new()
            .run_until(async {
                let telemetry = crate::telemetry::Telemetry::install("off")?;
                let output = tempfile::tempdir()?;
                let text = format!("{SHORT_CORPUS}\n{SHORT_CORPUS}");
                let result = round(
                    &request(),
                    output.path(),
                    &text,
                    Rc::new(Controlled {
                        ending: Ending::Success,
                        closed: Arc::new(AtomicBool::new(false)),
                    }),
                    0,
                    &telemetry,
                    tracing::Span::none(),
                )
                .await?;
                let ranges = result["artifacts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|artifact| artifact["kind"] == "source_ranges")
                    .count();
                assert_eq!(ranges, 1);
                let ranges: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
                    output.path().join("round-0-ranges.json"),
                )?)?;
                assert_eq!(ranges["segments"].as_array().unwrap().len(), 2);
                assert_eq!(ranges["segments"][0]["range"]["start"], 0);
                assert_eq!(ranges["segments"][1]["range"]["end"], text.as_bytes().len());
                assert!(result["output_sha256"].as_str().unwrap().len() == 64);
                Ok::<(), anyhow::Error>(())
            })
            .await
    }
}
