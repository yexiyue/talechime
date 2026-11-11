//! Export the exact adapter stream; model files must already be verified locally.
mod segmentation;
use std::{path::PathBuf, rc::Rc, time::Instant};
use talechime_core::backend::{AudioChunk, Backend};
#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    tokio::task::LocalSet::new().run_until(run()).await
}
async fn run() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).is_some_and(|arg| arg == "--identity") {
        let features = [
            ("qwen", cfg!(feature = "qwen")),
            ("qwen-metal", cfg!(feature = "qwen-metal")),
            ("qwen-cuda", cfg!(feature = "qwen-cuda")),
            ("voxcpm", cfg!(feature = "voxcpm")),
            ("voxcpm-metal", cfg!(feature = "voxcpm-metal")),
            ("voxcpm-cuda", cfg!(feature = "voxcpm-cuda")),
            ("kokoro", cfg!(feature = "kokoro")),
            ("kokoro-g2pw", cfg!(feature = "kokoro-g2pw")),
            ("kokoro-metal", cfg!(feature = "kokoro-metal")),
            ("kokoro-cuda", cfg!(feature = "kokoro-cuda")),
            ("cosyvoice3", cfg!(feature = "cosyvoice3")),
            ("cosyvoice3-metal", cfg!(feature = "cosyvoice3-metal")),
            ("cosyvoice3-cuda", cfg!(feature = "cosyvoice3-cuda")),
            ("indextts", cfg!(feature = "indextts")),
            ("indextts-metal", cfg!(feature = "indextts-metal")),
            ("indextts-cuda-cudnn", cfg!(feature = "indextts-cuda-cudnn")),
            ("omnivoice", cfg!(feature = "omnivoice")),
            ("omnivoice-metal", cfg!(feature = "omnivoice-metal")),
            ("omnivoice-cuda", cfg!(feature = "omnivoice-cuda")),
            ("moss-nano-candle", cfg!(feature = "moss-nano-candle")),
            (
                "moss-nano-candle-metal",
                cfg!(feature = "moss-nano-candle-metal"),
            ),
            (
                "moss-nano-candle-cuda",
                cfg!(feature = "moss-nano-candle-cuda"),
            ),
            ("moss-candle", cfg!(feature = "moss-candle")),
            ("moss-candle-metal", cfg!(feature = "moss-candle-metal")),
            ("moss-candle-cuda", cfg!(feature = "moss-candle-cuda")),
        ];
        let features: Vec<_> = features
            .into_iter()
            .filter_map(|(name, enabled)| enabled.then_some(name))
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "package_version": env!("CARGO_PKG_VERSION"),
                "text_diagnostics": true,
                "text_semantics_revision": talechime_backends::TEXT_SEMANTICS_REVISION,
                "rustc": option_env!("TALECHIME_BUILD_RUSTC"),
                "features": features,
                "target_os": std::env::consts::OS,
                "target_arch": std::env::consts::ARCH,
            })
        );
        return Ok(());
    }
    anyhow::ensure!(
        args.len() >= 7,
        "backend directory output.wav cpu|cuda|metal text voice"
    );
    let directory = PathBuf::from(&args[2]);
    let text = if let Some(path) = args[5].strip_prefix('@') {
        std::fs::read_to_string(path)?
    } else {
        args[5].clone()
    };
    let device = match args[4].as_str() {
        "cpu" => talechime_protocol::Device::Cpu,
        "cuda" => talechime_protocol::Device::Cuda,
        "metal" => talechime_protocol::Device::Metal,
        _ => anyhow::bail!("invalid device"),
    };
    if matches!(
        args[1].as_str(),
        "qwen-design" | "moss-design" | "voxcpm-design" | "omnivoice-design"
    ) {
        let options = design_options(&args[1], &probe_params()?)?;
        return design_probe(
            &args[1],
            directory,
            PathBuf::from(&args[3]),
            device,
            &text,
            options,
        )
        .await;
    }
    let load = || async {
        let backend: Rc<dyn Backend> = match args[1].as_str() {
            #[cfg(feature = "moss-nano-candle")]
            "moss-nano" => Rc::new(
                talechime_backends::moss::nano::NanoBackend::load_on(directory.clone(), device)
                    .await?,
            ),
            #[cfg(feature = "kokoro")]
            "kokoro" => Rc::new(
                talechime_backends::kokoro::KokoroBackend::load_model_on(
                    directory.clone(),
                    &std::env::var("TALECHIME_KOKORO_MODEL")
                        .unwrap_or_else(|_| "1.1-zh-g2pw".into()),
                    device,
                )
                .await?,
            ),
            #[cfg(feature = "cosyvoice3")]
            "cosyvoice3" => Rc::new(
                talechime_backends::cosyvoice3::CosyBackend::load_on(directory.clone(), device)
                    .await?,
            ),
            #[cfg(feature = "indextts")]
            "indextts" => Rc::new(
                talechime_backends::indextts::IndexBackend::load_on(directory.clone(), device)
                    .await?,
            ),
            #[cfg(feature = "moss-candle")]
            "moss" => Rc::new(
                talechime_backends::moss::candle::CandleBackend::load_on(
                    directory.clone(),
                    talechime_backends::moss::candle::Mode::parse(
                        &std::env::var("MOSS_MODEL").unwrap_or_else(|_| "realtime-1.7b".into()),
                    )?,
                    device,
                )
                .await?,
            ),
            #[cfg(feature = "qwen")]
            "qwen" => Rc::new(
                talechime_backends::qwen::QwenBackend::load_on(directory.clone(), device).await?,
            ),
            #[cfg(feature = "voxcpm")]
            "voxcpm" => Rc::new(
                talechime_backends::voxcpm::VoxBackend::load_model_on(
                    directory.clone(),
                    probe_voxcpm_model()?,
                    device,
                )
                .await?,
            ),
            #[cfg(feature = "omnivoice")]
            "omnivoice" => Rc::new(
                talechime_backends::omnivoice::OmniBackend::load_on(directory.clone(), device)
                    .await?,
            ),
            _ => anyhow::bail!("backend not compiled"),
        };
        Ok::<_, anyhow::Error>(backend)
    };
    if std::env::var_os("NOVEL_TTS_PROBE_CANCEL_LOAD").is_some() {
        let started = Instant::now();
        let cancelled = tokio::time::timeout(std::time::Duration::from_millis(100), load()).await;
        println!(
            "{}",
            serde_json::json!({"cancelled_loading":cancelled.is_err(),"cancel_load_and_release_ms":started.elapsed().as_millis()})
        );
        // A completed initializer is owned and released before the next load as well.
        drop(cancelled);
    }
    let started = Instant::now();
    let backend = load().await?;
    let load_ms = started.elapsed().as_millis();
    let capabilities = backend.capabilities();
    println!(
        "{}",
        serde_json::json!({"loaded_model":capabilities.model,"loaded_backend":capabilities.backend})
    );
    if std::env::var_os("NOVEL_TTS_PROBE_CANCEL").is_some() {
        let params = probe_params()?;
        let mut stream = backend
            .stream(talechime_core::backend::SegmentRequest {
                text: &text,
                voice: &args[6],
                style: None,

                seed: 42,
                params: &params,
            })
            .await?;
        anyhow::ensure!(
            matches!(stream.recv().await.transpose()?, Some(AudioChunk::Pcm(_))),
            "no PCM before cancel"
        );
        drop(stream);
    }
    if std::env::var_os("NOVEL_TTS_PROBE_CANCEL_EARLY").is_some() {
        let params = probe_params()?;
        let stream = backend
            .stream(talechime_core::backend::SegmentRequest {
                text: &text,
                voice: &args[6],
                style: None,

                seed: 42,
                params: &params,
            })
            .await?;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        drop(stream);
    }
    let started = Instant::now();
    let style = std::env::var("NOVEL_TTS_PROBE_STYLE").ok();
    let mut first = None;
    let mut audio = None;
    let segments = segmentation::collect(&*backend, &text, &args[6]).await?;
    anyhow::ensure!(!segments.is_empty(), "empty input");
    let mut boundaries = Vec::new();
    for segment in &segments {
        if std::env::var("TALECHIME_TEXT_DIAGNOSTICS").is_ok_and(|value| value == "1") {
            let source = text
                .get(segment.start..segment.end)
                .ok_or_else(|| anyhow::anyhow!("invalid source UTF-8 range"))?;
            eprintln!(
                "TALECHIME_TEXT_DIAGNOSTIC {}",
                serde_json::json!({"backend":args[1],"phase":"backend_input","source_start":segment.start,"source_end":segment.end,"source_text":source,"execution_text":segment.text})
            );
        }
        let offset = audio
            .as_ref()
            .map_or(0, |pcm: &talechime_core::backend::Pcm| pcm.samples.len());
        let params = probe_params()?;
        let mut stream = backend
            .stream(talechime_core::backend::SegmentRequest {
                text: &segment.text,
                voice: &args[6],
                style: style.as_deref(),

                seed: 42,
                params: &params,
            })
            .await?;
        let mut ended = false;
        while let Some(chunk) = stream.recv().await {
            match chunk? {
                AudioChunk::Pcm(pcm) => {
                    pcm.duration_ms()?;
                    first.get_or_insert_with(|| started.elapsed().as_millis());
                    if let Some(samples) = &mut audio {
                        let samples: &mut talechime_core::backend::Pcm = samples;
                        anyhow::ensure!(
                            samples.sample_rate == pcm.sample_rate
                                && samples.channels == pcm.channels,
                            "PCM format changed"
                        );
                        samples.samples.extend(pcm.samples);
                    } else {
                        audio = Some(pcm);
                    }
                }
                AudioChunk::End => {
                    ended = true;
                    break;
                }
            }
        }
        anyhow::ensure!(ended, "stream did not end normally");
        let end = audio.as_ref().map_or(0, |pcm| pcm.samples.len());
        anyhow::ensure!(end > offset, "segment produced no PCM");
        boundaries.push((offset, end, &segment.text, segment.start, segment.end));
    }
    let audio = audio.ok_or_else(|| anyhow::anyhow!("empty PCM"))?;
    let generate_ms = started.elapsed().as_millis();
    let duration = audio.samples.len() as f64 / audio.sample_rate as f64 / audio.channels as f64;
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    write_wav(std::path::Path::new(&args[3]), spec, &audio.samples)?;
    let output = PathBuf::from(&args[3]);
    let directory = output.with_file_name(format!(
        "{}-segments",
        output.file_stem().unwrap().to_string_lossy()
    ));
    std::fs::create_dir_all(&directory)?;
    let mut index = Vec::new();
    for (number, (start, end, text, source_start, source_end)) in boundaries.iter().enumerate() {
        let filename = format!("{:03}.wav", number + 1);
        write_wav(
            &directory.join(&filename),
            spec,
            &audio.samples[*start..*end],
        )?;
        index.push(
            serde_json::json!({"file":filename,"text":text,"source_start":source_start,"source_end":source_end,"start_sample":start,"end_sample":end}),
        );
    }
    std::fs::write(
        directory.join("index.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    println!(
        "{}",
        serde_json::json!({"backend":args[1],"device":format!("{device:?}"),"load_ms":load_ms,"first_pcm_ms":first,"generate_ms":generate_ms,"audio_seconds":duration,"segments":segments.len(),"rtf":generate_ms as f64 / 1000.0 / duration,"stream_end":true,"model_eos":"not_exposed"})
    );
    if std::env::var_os("NOVEL_TTS_PROBE_DROP_EARLY").is_some() {
        let params = probe_params()?;
        let stream = backend
            .stream(talechime_core::backend::SegmentRequest {
                text: &text,
                voice: &args[6],
                style: None,

                seed: 42,
                params: &params,
            })
            .await?;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        drop(stream);
        let cancelled = Instant::now();
        drop(backend);
        println!(
            "{}",
            serde_json::json!({"cancel_and_release_ms":cancelled.elapsed().as_millis()})
        );
    }
    Ok(())
}

async fn design_probe(
    backend: &str,
    directory: PathBuf,
    output: PathBuf,
    device: talechime_protocol::Device,
    text: &str,
    options: Option<talechime_backends::qwen::design::Options>,
) -> anyhow::Result<()> {
    let description = std::env::var("NOVEL_TTS_PROBE_STYLE")?;
    if std::env::var("TALECHIME_TEXT_DIAGNOSTICS").is_ok_and(|value| value == "1") {
        eprintln!(
            "TALECHIME_TEXT_DIAGNOSTIC {}",
            serde_json::json!({"backend":backend, "phase":"backend_input", "source_start":0,
                "source_end":text.len(), "source_text":text, "execution_text":text})
        );
    }
    let reference = output.with_file_name("design-reference.wav");
    anyhow::ensure!(
        !output.exists() && !reference.exists(),
        "refuse existing design output"
    );
    let started = Instant::now();
    let model = match backend {
        #[cfg(feature = "qwen")]
        "qwen-design" => {
            talechime_backends::qwen::design::reference(
                directory,
                device,
                text.to_owned(),
                description,
                reference.clone(),
                options.ok_or_else(|| anyhow::anyhow!("missing Qwen design options"))?,
            )
            .await?;
            "1.7b-voicedesign"
        }
        #[cfg(feature = "moss-candle")]
        "moss-design" => {
            talechime_backends::moss::candle::design::reference(
                directory,
                device,
                text.to_owned(),
                description,
                reference.clone(),
                Some(42),
            )
            .await?;
            "voice-design-1.7b"
        }
        #[cfg(feature = "voxcpm")]
        "voxcpm-design" => {
            let variant = probe_voxcpm_model()?;
            talechime_backends::voxcpm::design::reference_model(
                directory,
                variant,
                device,
                text.to_owned(),
                description,
                reference.clone(),
                Some(42),
            )
            .await?;
            variant.id()
        }
        #[cfg(feature = "omnivoice")]
        "omnivoice-design" => {
            talechime_backends::omnivoice::design::reference(
                directory,
                device,
                text.to_owned(),
                description,
                reference.clone(),
                Some(42),
            )
            .await?;
            "0.6b"
        }
        _ => anyhow::bail!("design backend not compiled"),
    };
    let audio = talechime_qwen3_tts::AudioBuffer::load(&reference)?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    write_wav(&output, spec, &audio.samples)?;
    let segments = output.with_file_name("audio-segments");
    std::fs::create_dir(&segments)?;
    std::fs::write(
        segments.join("index.json"),
        serde_json::to_vec_pretty(&[serde_json::json!({
            "text":text, "source_start":0, "source_end":text.len(), "start_sample":0, "end_sample":audio.samples.len(),
        })])?,
    )?;
    println!(
        "{}",
        serde_json::json!({"loaded_model":model, "api_completed":true, "model_eos":"not_exposed", "generate_ms":started.elapsed().as_millis()})
    );
    Ok(())
}

#[cfg(feature = "voxcpm")]
fn probe_voxcpm_model() -> anyhow::Result<talechime_backends::voxcpm::models::Model> {
    let id = std::env::var("TALECHIME_PROBE_MODEL").ok();
    talechime_backends::voxcpm::models::Model::parse(id.as_deref())
}

fn design_options(
    backend: &str,
    params: &talechime_core::params::GenerationParams,
) -> anyhow::Result<Option<talechime_backends::qwen::design::Options>> {
    match backend {
        "qwen-design" => {
            for (name, _) in params.iter() {
                anyhow::ensure!(
                    name == "language",
                    "unsupported Qwen design parameter: {name}"
                );
            }
            let language = match params.get("language") {
                None => talechime_qwen3_tts::Language::Auto,
                Some(talechime_protocol::ParamValue::Text(value)) => value.parse()?,
                Some(_) => anyhow::bail!("Qwen design language must be text"),
            };
            Ok(Some(talechime_backends::qwen::design::Options {
                language,
                seed: Some(42),
            }))
        }
        "moss-design" | "voxcpm-design" | "omnivoice-design" => {
            anyhow::ensure!(
                params.is_empty(),
                "this design backend does not accept generation parameters"
            );
            Ok(None)
        }
        _ => anyhow::bail!("unknown design backend"),
    }
}

#[cfg(test)]
mod design_parameter_tests {
    use super::*;
    use talechime_core::params::GenerationParams;
    use talechime_protocol::ParamValue;
    use talechime_qwen3_tts::Language;

    #[test]
    fn qwen_design_forwards_explicit_language_and_defaults_to_auto() {
        let mut params = GenerationParams::new();
        assert_eq!(
            design_options("qwen-design", &params)
                .unwrap()
                .unwrap()
                .language,
            Language::Auto
        );
        params.insert("language", ParamValue::Text("ja".into()));
        let options = design_options("qwen-design", &params).unwrap().unwrap();
        assert_eq!(options.language, Language::Japanese);
        assert_eq!(options.seed, Some(42));
        params.insert("temperature", ParamValue::Float(0.5));
        assert!(design_options("qwen-design", &params).is_err());
    }

    #[test]
    fn design_rejects_invalid_language_and_moss_parameters() {
        let mut params = GenerationParams::new();
        assert!(design_options("moss-design", &params).unwrap().is_none());
        params.insert("language", ParamValue::Text("not-a-language".into()));
        assert!(design_options("qwen-design", &params).is_err());
        assert!(design_options("moss-design", &params).is_err());
        params.insert("language", ParamValue::Int(1));
        assert!(design_options("qwen-design", &params).is_err());
    }
}

fn write_wav(path: &std::path::Path, spec: hound::WavSpec, samples: &[f32]) -> anyhow::Result<()> {
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

fn probe_params() -> anyhow::Result<talechime_core::params::GenerationParams> {
    let mut params = talechime_core::params::GenerationParams::new();
    if let Ok(json) = std::env::var("TALECHIME_PROBE_PARAMS") {
        let values: std::collections::BTreeMap<String, talechime_protocol::ParamValue> =
            serde_json::from_str(&json)?;
        for (name, value) in values {
            params.insert(name, value);
        }
    }
    Ok(params)
}
