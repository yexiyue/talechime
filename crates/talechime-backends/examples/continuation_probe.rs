//! Explicit, pinned real-model comparison. Outputs and references belong in ignored directories.
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};
use talechime_core::{
    SynthesisOptions, SynthesisState, SynthesisStream,
    backend::{AudioChunk, Backend, Pcm, Segmentation, SegmentationRequest, Streaming},
};
use talechime_protocol::{Capabilities, Device};

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
    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        match self {
            Self::Onnx(backend) => Box::pin(backend.stream_seeded_with_context(
                request.text,
                request.voice,
                Some(request.seed),
                request.context,
            )),
            #[cfg(feature = "moss-nano-candle")]
            Self::Candle(backend) => Box::pin(backend.stream_seeded_with_context(
                request.text,
                request.voice,
                request.seed,
                request.context,
            )),
        }
    }
    fn next_segment<'a>(&'a self, request: SegmentationRequest<'a>) -> Segmentation<'a> {
        self.backend().next_segment(request)
    }
    fn paragraph_end(&self, text: &str, remaining: &str) -> bool {
        self.backend().paragraph_end(text, remaining)
    }
}
struct Observed {
    inner: Rc<dyn Backend>,
    metrics: Rc<RefCell<Vec<Metric>>>,
    fixed_anchor: bool,
    anchor: Rc<RefCell<Option<talechime_core::SpeechContext>>>,
    line_segments: bool,
}
impl Backend for Observed {
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        Box::pin(async move {
            let anchor = self.anchor.borrow().clone();
            let request = talechime_core::backend::SegmentRequest {
                text: request.text,
                voice: request.voice,
                style: request.style,
                seed: request.seed,
                params: request.params,
                context: if self.fixed_anchor {
                    anchor.as_ref()
                } else {
                    request.context
                },
            };
            let started = Instant::now();
            let text = request.text.to_owned();
            let continued = request.context.is_some();
            let mut incoming = self.inner.stream(request).await?;
            let metrics = self.metrics.clone();
            let anchor_store = self.anchor.clone();
            let capture_anchor = self.fixed_anchor && anchor.is_none();
            let (tx, rx) = tokio::sync::mpsc::channel(1);
            tokio::task::spawn_local(async move {
                let mut first = None;
                let mut seconds = 0.0;
                let mut reference: Option<Pcm> = None;
                let mut oversized = false;
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
                        if capture_anchor && !oversized {
                            if seconds > 20.0 {
                                oversized = true;
                                reference = None;
                            } else {
                                let collected = reference.get_or_insert_with(|| Pcm {
                                    samples: vec![],
                                    sample_rate: pcm.sample_rate,
                                    channels: pcm.channels,
                                });
                                collected.samples.extend_from_slice(&pcm.samples);
                            }
                        }
                    }
                    if ended {
                        if let Some(pcm) = reference.take() {
                            match talechime_core::SpeechContext::new(text.clone(), pcm) {
                                Ok(context) => *anchor_store.borrow_mut() = Some(context),
                                Err(error) => eprintln!("fixed anchor rejected: {error}"),
                            }
                        }
                        eprintln!(
                            "segment {} completed: {seconds:.2}s",
                            metrics.borrow().len() + 1
                        );
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
    fn next_segment<'a>(&'a self, request: SegmentationRequest<'a>) -> Segmentation<'a> {
        if !self.line_segments {
            return self.inner.next_segment(request);
        }
        let text = request.text;
        Box::pin(async move {
            let start = text.len() - text.trim_start().len();
            if start == text.len() {
                return Ok(None);
            }
            let line_end = text[start..].find('\n').map_or(text.len(), |n| start + n);
            let end = text[..line_end].trim_end().len();
            Ok(Some(talechime_core::text::TextSegment {
                text: text[start..end].to_owned(),
                start,
                end,
            }))
        })
    }

    fn paragraph_end(&self, text: &str, remaining: &str) -> bool {
        self.inner.paragraph_end(text, remaining)
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
        args.len() >= 8,
        "backend model-dir output-dir cpu|metal|cuda voice on|off corpus.txt [--seed n] [--param name=value]..."
    );
    let mut seed: u64 = 42;
    let mut fixed_anchor = false;
    let mut dual_reference = false;
    let mut moss_reference_anchor = false;
    let mut line_segments = false;
    let mut continuation_span = None;
    let mut continuation_boundary = talechime_protocol::ContinuationBoundary::Paragraph;
    let mut style: Option<String> = None;
    let mut params = talechime_core::params::GenerationParams::new();
    let mut extra = args[8..].iter();
    while let Some(flag) = extra.next() {
        match flag.as_str() {
            "--dual-reference" => dual_reference = true,
            "--moss-reference-anchor" => moss_reference_anchor = true,
            "--fixed-anchor" => fixed_anchor = true,
            "--line-segments" => line_segments = true,
            "--span" => {
                continuation_span = Some(
                    extra
                        .next()
                        .and_then(|value| value.parse().ok())
                        .ok_or_else(|| anyhow::anyhow!("--span requires seconds"))?,
                )
            }
            "--boundary" => {
                continuation_boundary = serde_json::from_value(serde_json::Value::String(
                    extra
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--boundary requires paragraph|scene|none"))?
                        .clone(),
                ))?
            }
            "--seed" => {
                seed = extra
                    .next()
                    .and_then(|value| value.parse().ok())
                    .ok_or_else(|| anyhow::anyhow!("--seed requires a number"))?;
            }
            "--style" => {
                style = Some(
                    extra
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--style requires a description"))?
                        .clone(),
                );
            }
            "--param" => {
                let entry = extra
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--param requires name=value"))?;
                let (name, value) = entry
                    .split_once('=')
                    .ok_or_else(|| anyhow::anyhow!("--param requires name=value"))?;
                params.insert(
                    name,
                    serde_json::from_str::<talechime_protocol::ParamValue>(value)
                        .map_err(|e| anyhow::anyhow!("invalid param value: {e}"))?,
                );
            }
            other => anyhow::bail!("unknown flag {other}"),
        }
    }

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
    anyhow::ensure!(
        !fixed_anchor || continuation,
        "fixed anchor requires continuation on"
    );
    anyhow::ensure!(
        !dual_reference || (args[1] == "voxcpm" && continuation && !fixed_anchor),
        "dual reference requires voxcpm, continuation on and rolling context"
    );
    anyhow::ensure!(
        !dual_reference || cfg!(feature = "experimental-dual-reference"),
        "dual reference experiment feature is not compiled"
    );
    anyhow::ensure!(
        !moss_reference_anchor || (args[1] == "moss-realtime" && continuation && !fixed_anchor),
        "MOSS reference anchor requires realtime, continuation on and rolling context"
    );
    anyhow::ensure!(
        !moss_reference_anchor || cfg!(feature = "experimental-moss-reference-anchor"),
        "MOSS reference anchor experiment feature is not compiled"
    );
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
            let backend =
                talechime_backends::voxcpm::VoxBackend::load_model_on(directory, model, device)
                    .await?;
            #[cfg(feature = "experimental-dual-reference")]
            let backend = if dual_reference {
                backend.with_dual_reference()
            } else {
                backend
            };
            Rc::new(backend)
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
            let backend =
                talechime_backends::moss::candle::CandleBackend::load_on(directory, mode, device)
                    .await?;
            #[cfg(feature = "experimental-moss-reference-anchor")]
            let backend = if moss_reference_anchor {
                backend.with_reference_anchor()?
            } else {
                backend
            };
            Rc::new(backend)
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
    if let Err(error) = params.validate(&caps.parameters) {
        anyhow::bail!("{error}");
    }
    let metrics = Rc::new(RefCell::new(vec![]));
    let anchor = Rc::new(RefCell::new(None));
    let observed = Rc::new(Observed {
        inner: backend,
        metrics: metrics.clone(),
        fixed_anchor,
        anchor: anchor.clone(),
        line_segments,
    });
    let started = Instant::now();
    let mut stream = SynthesisStream::start_with_options(
        observed,
        text.clone(),
        &args[5],
        style.clone(),
        None,
        SynthesisOptions {
            continuation,
            continuation_span,
            continuation_boundary,
            params: params.clone(),
            seed: talechime_core::SeedPolicy::Pinned(seed),
            ..Default::default()
        },
    )?;
    let mut audio: Option<Pcm> = None;
    let mut segments: Vec<(talechime_protocol::TextRange, Pcm)> = vec![];
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
    if let Some(context) = anchor.borrow().as_ref() {
        wav(&output.join("fixed-anchor.wav"), context.pcm())?;
        std::fs::write(output.join("fixed-anchor.txt"), context.text())?;
    }
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
        "style": style,
        "continuation": continuation,
        "fixed_anchor": fixed_anchor,
        "dual_reference": dual_reference,
        "moss_reference_anchor": moss_reference_anchor,
        "line_segments": line_segments,
        "continuation_span": continuation_span,
        "continuation_boundary": continuation_boundary,
        "load_and_verify_ms": load_ms,
        "elapsed_ms": elapsed_ms,
        "audio_seconds": seconds,
        "rtf": elapsed_ms as f64 / 1000.0 / seconds,
        "segments": *metrics.borrow(),
        "source_hash": talechime_protocol::text_hash(&text),
        "completed": true,
        "seed": seed,
        "params": params
            .iter()
            .map(|(name, value)| (name.to_string(), serde_json::to_value(value).unwrap()))
            .collect::<serde_json::Map<String, serde_json::Value>>(),
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
