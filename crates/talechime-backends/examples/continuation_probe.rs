//! Explicit, pinned real-model comparison. Outputs and references belong in ignored directories.
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};
use tts_core::{
    SpeechContext, SynthesisOptions, SynthesisState, SynthesisStream,
    backend::{AudioChunk, Backend, Pcm, Segmentation, Streaming},
};
use tts_protocol::{Capabilities, Device};

#[derive(serde::Serialize)]
struct Metric {
    text: String,
    continued: bool,
    first_pcm_ms: Option<u128>,
    elapsed_ms: u128,
    audio_seconds: f64,
}
#[cfg(feature = "moss")]
enum SeededMoss {
    Onnx(talechime_backends::moss::MossBackend),
    #[cfg(feature = "moss-nano-candle")]
    Candle(talechime_backends::moss::nano::NanoBackend),
}
#[cfg(feature = "moss")]
impl SeededMoss {
    fn backend(&self) -> &dyn Backend {
        match self {
            Self::Onnx(backend) => backend,
            #[cfg(feature = "moss-nano-candle")]
            Self::Candle(backend) => backend,
        }
    }
}
#[cfg(feature = "moss")]
impl Backend for SeededMoss {
    fn capabilities(&self) -> Capabilities {
        self.backend().capabilities()
    }
    fn stream<'a>(&'a self, text: &'a str, voice: &'a str) -> Streaming<'a> {
        self.stream_with_context(text, voice, None, None)
    }
    fn stream_with_context<'a>(
        &'a self,
        text: &'a str,
        voice: &'a str,
        _style: Option<&'a str>,
        context: Option<&'a SpeechContext>,
    ) -> Streaming<'a> {
        match self {
            Self::Onnx(backend) => {
                Box::pin(backend.stream_seeded_with_context(text, voice, Some(42), context))
            }
            #[cfg(feature = "moss-nano-candle")]
            Self::Candle(backend) => {
                Box::pin(backend.stream_seeded_with_context(text, voice, 42, context))
            }
        }
    }
    fn next_segment<'a>(
        &'a self,
        text: &'a str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        Option<tts_core::text::TextSegment>,
                        tts_core::backend::BackendError,
                    >,
                > + 'a,
        >,
    > {
        self.backend().next_segment(text)
    }
    fn paragraph_end(&self, text: &str, remaining: &str) -> bool {
        self.backend().paragraph_end(text, remaining)
    }
    fn select_voice(&self, voice: &str) {
        self.backend().select_voice(voice)
    }
    fn observe_duration(&self, text: &str, voice: &str, seconds: f64) {
        self.backend().observe_duration(text, voice, seconds)
    }
}
struct Observed {
    inner: Rc<dyn Backend>,
    metrics: Rc<RefCell<Vec<Metric>>>,
}
impl Backend for Observed {
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
    fn stream<'a>(&'a self, text: &'a str, voice: &'a str) -> Streaming<'a> {
        self.stream_with_context(text, voice, None, None)
    }
    fn stream_with_context<'a>(
        &'a self,
        text: &'a str,
        voice: &'a str,
        style: Option<&'a str>,
        context: Option<&'a SpeechContext>,
    ) -> Streaming<'a> {
        Box::pin(async move {
            let started = Instant::now();
            let mut incoming = self
                .inner
                .stream_with_context(text, voice, style, context)
                .await?;
            let metrics = self.metrics.clone();
            let text = text.to_owned();
            let continued = context.is_some();
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            tokio::task::spawn_local(async move {
                let mut first = None;
                let mut seconds = 0.0;
                loop {
                    let chunk = tokio::select! {_=tx.closed()=>break,chunk=incoming.recv()=>chunk};
                    let Some(chunk) = chunk else {
                        break;
                    };
                    let ended = matches!(chunk, Ok(AudioChunk::End));
                    if let Ok(AudioChunk::Pcm(pcm)) = &chunk {
                        first.get_or_insert_with(|| started.elapsed().as_millis());
                        seconds +=
                            pcm.samples.len() as f64 / pcm.sample_rate as f64 / pcm.channels as f64;
                    }
                    if ended {
                        metrics.borrow_mut().push(Metric {
                            text: text.clone(),
                            continued,
                            first_pcm_ms: first,
                            elapsed_ms: started.elapsed().as_millis(),
                            audio_seconds: seconds,
                        });
                    }
                    if tx.send(chunk).await.is_err() || ended {
                        break;
                    }
                }
            });
            Ok(rx)
        })
    }
    fn next_segment<'a>(
        &'a self,
        text: &'a str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        Option<tts_core::text::TextSegment>,
                        tts_core::backend::BackendError,
                    >,
                > + 'a,
        >,
    > {
        self.inner.next_segment(text)
    }
    fn segments<'a>(&'a self, text: &'a str) -> Segmentation<'a> {
        self.inner.segments(text)
    }
    fn paragraph_end(&self, text: &str, remaining: &str) -> bool {
        self.inner.paragraph_end(text, remaining)
    }
    fn select_voice(&self, voice: &str) {
        self.inner.select_voice(voice)
    }
    fn observe_duration(&self, text: &str, voice: &str, seconds: f64) {
        self.inner.observe_duration(text, voice, seconds)
    }
}
fn wav(path: &Path, pcm: &Pcm) -> anyhow::Result<()> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: pcm.channels,
            sample_rate: pcm.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )?;
    for &sample in &pcm.samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}
fn wav16(path: &Path, pcm: &Pcm) -> anyhow::Result<()> {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: pcm.channels,
            sample_rate: pcm.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for &sample in &pcm.samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    tokio::task::LocalSet::new().run_until(run()).await
}
async fn run() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 8,
        "backend verified-model-directory output-directory cpu|metal|cuda voice on|off corpus.txt"
    );
    let directory = PathBuf::from(&args[2]);
    let output = PathBuf::from(&args[3]);
    std::fs::create_dir_all(&output)?;
    let device = match args[4].as_str() {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        "cuda" => Device::Cuda,
        _ => anyhow::bail!("invalid device"),
    };
    let continuation = match args[6].as_str() {
        "on" => true,
        "off" => false,
        _ => anyhow::bail!("on or off required"),
    };
    let text = std::fs::read_to_string(&args[7])?;
    let (progress, mut events) = tokio::sync::mpsc::channel(16);
    let preparation = tokio::task::spawn_local(async move {
        while let Some(event) = events.recv().await {
            eprintln!("{event:?}");
        }
    });
    let started = Instant::now();
    let backend: Rc<dyn Backend> = match args[1].as_str() {
        #[cfg(feature = "moss")]
        "moss" => {
            talechime_backends::moss::resources::prepare(&directory, progress.clone()).await?;
            Rc::new(SeededMoss::Onnx(
                talechime_backends::moss::MossBackend::load_on(directory, device).await?,
            ))
        }
        #[cfg(feature = "moss-nano-candle")]
        "moss-candle" => {
            talechime_backends::moss::nano::prepare(&directory, progress.clone()).await?;
            Rc::new(SeededMoss::Candle(
                talechime_backends::moss::nano::NanoBackend::load_on(directory, device).await?,
            ))
        }
        #[cfg(feature = "qwen")]
        "qwen" | "qwen-base17" => {
            let model = if args[1] == "qwen-base17" {
                talechime_backends::qwen::models::Model::Base17
            } else {
                talechime_backends::qwen::models::Model::Base06
            };
            talechime_backends::qwen::resources::prepare_model(&directory, model, progress.clone())
                .await?;
            Rc::new(talechime_backends::qwen::QwenBackend::load_on(directory, device).await?)
        }
        #[cfg(feature = "voxcpm")]
        "voxcpm" | "voxcpm-bf16" => {
            let model = if args[1] == "voxcpm-bf16" {
                talechime_backends::voxcpm::models::Model::OriginalBf16
            } else {
                talechime_backends::voxcpm::models::Model::Q8
            };
            talechime_backends::voxcpm::resources::prepare_model(
                &directory,
                model,
                progress.clone(),
            )
            .await?;
            Rc::new(
                talechime_backends::voxcpm::VoxBackend::load_model_on(directory, model, device)
                    .await?,
            )
        }
        #[cfg(feature = "moss-candle")]
        "moss-local" | "moss-realtime" => {
            let mode = if args[1] == "moss-local" {
                talechime_backends::moss::candle::Mode::Local
            } else {
                talechime_backends::moss::candle::Mode::Realtime
            };
            talechime_backends::moss::candle::resources::prepare(
                &directory,
                mode,
                progress.clone(),
            )
            .await?;
            Rc::new(
                talechime_backends::moss::candle::CandleBackend::load_on(directory, mode, device)
                    .await?,
            )
        }
        #[cfg(feature = "omnivoice")]
        "omnivoice" => {
            talechime_backends::omnivoice::resources::prepare(&directory, progress.clone()).await?;
            Rc::new(talechime_backends::omnivoice::OmniBackend::load_on(directory, device).await?)
        }
        _ => anyhow::bail!("backend not compiled"),
    };
    drop(progress);
    preparation.await?;
    let load_ms = started.elapsed().as_millis();
    let caps = backend.capabilities();
    let metrics = Rc::new(RefCell::new(vec![]));
    let observed = Rc::new(Observed {
        inner: backend,
        metrics: metrics.clone(),
    });
    let started = Instant::now();
    let mut stream = SynthesisStream::start_with_options(
        observed,
        text.clone(),
        &args[5],
        None,
        None,
        SynthesisOptions {
            continuation,
            ..Default::default()
        },
    )?;
    let mut audio: Option<Pcm> = None;
    let mut segments: Vec<(tts_protocol::TextRange, Pcm)> = vec![];
    while let Some(block) = stream.recv().await {
        let block = match block {
            Ok(block) => block,
            Err(error) => {
                std::fs::write(
                    output.join("error-summary.json"),
                    serde_json::to_vec_pretty(&*metrics.borrow())?,
                )?;
                if let Some(pcm) = &audio {
                    wav(&output.join("partial.wav"), pcm)?;
                }
                return Err(error.into());
            }
        };
        let pcm = block.pcm();
        let range = block.range();
        if segments.last().is_none_or(|(r, _)| *r != range) {
            segments.push((
                range,
                Pcm {
                    samples: vec![],
                    sample_rate: pcm.sample_rate,
                    channels: pcm.channels,
                },
            ));
        }
        segments
            .last_mut()
            .unwrap()
            .1
            .samples
            .extend_from_slice(&pcm.samples);
        let collected = audio.get_or_insert_with(|| Pcm {
            samples: vec![],
            sample_rate: pcm.sample_rate,
            channels: pcm.channels,
        });
        anyhow::ensure!(
            (collected.sample_rate, collected.channels) == (pcm.sample_rate, pcm.channels),
            "format changed"
        );
        collected.samples.extend_from_slice(&pcm.samples);
    }
    anyhow::ensure!(
        stream.state() == SynthesisState::Completed,
        "incomplete generation"
    );
    let elapsed_ms = started.elapsed().as_millis();
    let audio = audio.ok_or_else(|| anyhow::anyhow!("empty generation"))?;
    wav(&output.join("comparison.wav"), &audio)?;
    let mut corpus = vec![];
    for (i, (range, pcm)) in segments.iter().enumerate() {
        let path = output.join(format!("segment-{:03}.wav", i + 1));
        wav(&path, pcm)?;
        let readback = output.join(format!("readback-{:03}.wav", i + 1));
        wav16(&readback, pcm)?;
        corpus.push(serde_json::json!({
            "id": format!("{}-{}-{}", args[1], args[6], i + 1),
            "text": text[range.start..range.end],
            "file": readback.canonicalize()?,
            "range": range,
        }));
    }
    let seconds = audio.samples.len() as f64 / audio.sample_rate as f64 / audio.channels as f64;
    let result = serde_json::json!({
        "backend": caps.backend,
        "model": caps.model,
        "device": format!("{device:?}"),
        "voice": args[5],
        "continuation": continuation,
        "load_and_verify_ms": load_ms,
        "elapsed_ms": elapsed_ms,
        "audio_seconds": seconds,
        "rtf": elapsed_ms as f64 / 1000.0 / seconds,
        "segments": *metrics.borrow(),
        "source_hash": tts_protocol::text_hash(&text),
        "completed": true,
        "seed": 42,
        "listening_acceptance": "pending",
    });
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    std::fs::write(
        output.join("readback-corpus.json"),
        serde_json::to_vec_pretty(&corpus)?,
    )?;
    println!("{result}");
    Ok(())
}
