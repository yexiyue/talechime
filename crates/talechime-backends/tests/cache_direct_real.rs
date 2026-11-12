//! Explicit real-model check; ordinary tests never download weights.
#[cfg(feature = "qwen")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires the pinned Qwen 1.7B CustomVoice files in an explicitly selected Hub cache"]
async fn qwen_loads_and_synthesizes_directly_from_cached_files() -> anyhow::Result<()> {
    use talechime_core::backend::{AudioChunk, SegmentRequest};
    use talechime_protocol::Device;
    let cache = std::env::var_os("TALECHIME_TEST_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .expect("set TALECHIME_TEST_HUB_CACHE to an absolute existing cache");
    anyhow::ensure!(cache.is_absolute(), "cache must be absolute");
    let data = tempfile::tempdir()?;
    let registry = talechime_backends::Registry::with_downloads(
        Some(data.path().join("application-data")),
        &talechime_backends::downloads::DownloadConfig {
            cache_dir: cache,
            endpoint: Some("http://127.0.0.1:9".into()),
            token: None,
        },
    )?;
    let model = registry.resolve("qwen", Some("1.7b-customvoice"))?;
    anyhow::ensure!(
        registry
            .resource_status("qwen", Some("1.7b-customvoice"))
            .await?
            == talechime_backends::ResourceStatus::Ready,
        "pinned files not ready"
    );
    let (tx, mut events) = tokio::sync::mpsc::channel(32);
    let drain = async { while events.recv().await.is_some() {} };
    let run = async {
        let backend = model
            .load_existing(
                talechime_backends::PrepareOptions {
                    device: Device::Cpu,
                    ..Default::default()
                },
                tx,
            )
            .await?;
        let params = Default::default();
        let mut audio = backend
            .stream(SegmentRequest {
                text: "你好，欢迎回来。",
                voice: "uncle_fu",
                style: None,
                seed: 42,
                params: &params,
            })
            .await?;
        let mut samples = 0usize;
        let mut peak = 0f32;
        let mut ended = false;
        while let Some(chunk) = audio.recv().await {
            match chunk? {
                AudioChunk::Pcm(pcm) => {
                    anyhow::ensure!(
                        pcm.sample_rate == 24000
                            && pcm.channels == 1
                            && pcm.samples.iter().all(|x| x.is_finite()),
                        "invalid PCM"
                    );
                    samples += pcm.samples.len();
                    peak = pcm
                        .samples
                        .iter()
                        .fold(peak, |value, sample| value.max(sample.abs()));
                }
                AudioChunk::End => {
                    ended = true;
                    break;
                }
            }
        }
        anyhow::ensure!(
            ended && samples > 0 && peak > 0.0001,
            "audio did not complete or was silent"
        );
        println!("Qwen 1.7B CustomVoice CPU: {samples} samples at 24000 Hz; peak {peak:.4}");
        Ok::<_, anyhow::Error>(())
    };
    let (result, ()) = tokio::join!(run, drain);
    result?;
    anyhow::ensure!(
        !data.path().join("application-data/qwen/models").exists(),
        "created a model directory"
    );
    Ok(())
}

#[cfg(feature = "kokoro")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires the pinned Kokoro files in an explicitly selected Hub cache"]
async fn kokoro_loads_and_synthesizes_directly_from_cached_files() -> anyhow::Result<()> {
    use talechime_core::backend::{AudioChunk, SegmentRequest};
    use talechime_protocol::Device;
    let cache = std::env::var_os("TALECHIME_TEST_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .expect("set TALECHIME_TEST_HUB_CACHE to an absolute existing cache");
    anyhow::ensure!(cache.is_absolute(), "cache must be absolute");
    let data = tempfile::tempdir()?;
    let registry = talechime_backends::Registry::with_downloads(
        Some(data.path().join("application-data")),
        &talechime_backends::downloads::DownloadConfig {
            cache_dir: cache,
            endpoint: Some("http://127.0.0.1:9".into()),
            token: None,
        },
    )?;
    let model = registry.resolve("kokoro", Some("1.1-zh-g2pw"))?;
    anyhow::ensure!(
        registry
            .resource_status("kokoro", Some("1.1-zh-g2pw"))
            .await?
            == talechime_backends::ResourceStatus::Ready,
        "pinned files not ready"
    );
    let (tx, mut events) = tokio::sync::mpsc::channel(32);
    let drain = async { while events.recv().await.is_some() {} };
    let run = async {
        let backend = model
            .load_existing(
                talechime_backends::PrepareOptions {
                    device: Device::Cpu,
                    ..Default::default()
                },
                tx,
            )
            .await?;
        let params = Default::default();
        let mut audio = backend
            .stream(SegmentRequest {
                text: "你好，欢迎回来。",
                voice: "zf_001",
                style: None,
                seed: 42,
                params: &params,
            })
            .await?;
        let mut samples = 0usize;
        let mut peak = 0f32;
        let mut ended = false;
        while let Some(chunk) = audio.recv().await {
            match chunk? {
                AudioChunk::Pcm(pcm) => {
                    anyhow::ensure!(
                        pcm.sample_rate == 24000
                            && pcm.channels == 1
                            && pcm.samples.iter().all(|x| x.is_finite()),
                        "invalid PCM"
                    );
                    samples += pcm.samples.len();
                    peak = pcm
                        .samples
                        .iter()
                        .fold(peak, |value, sample| value.max(sample.abs()));
                }
                AudioChunk::End => {
                    ended = true;
                    break;
                }
            }
        }
        anyhow::ensure!(
            ended && samples > 0 && peak > 0.0001,
            "audio did not complete or was silent"
        );
        println!("Kokoro CPU: {samples} samples at 24000 Hz; peak {peak:.4}");
        Ok::<_, anyhow::Error>(())
    };
    let (result, ()) = tokio::join!(run, drain);
    result?;
    anyhow::ensure!(
        !data.path().join("application-data/kokoro/models").exists(),
        "created a model directory"
    );
    Ok(())
}

#[cfg(feature = "moss-nano-candle")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires pinned Nano and codec files in an explicitly selected Hub cache"]
async fn moss_nano_loads_and_synthesizes_directly_from_cached_files() -> anyhow::Result<()> {
    use talechime_core::backend::{AudioChunk, SegmentRequest};
    use talechime_protocol::Device;
    let cache = std::env::var_os("TALECHIME_TEST_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .expect("set TALECHIME_TEST_HUB_CACHE to an absolute existing cache");
    anyhow::ensure!(cache.is_absolute(), "cache must be absolute");
    let data = tempfile::tempdir()?;
    let registry = talechime_backends::Registry::with_downloads(
        Some(data.path().join("application-data")),
        &talechime_backends::downloads::DownloadConfig {
            cache_dir: cache,
            endpoint: Some("http://127.0.0.1:9".into()),
            token: None,
        },
    )?;
    let model = registry.resolve("moss", Some("nano-candle"))?;
    anyhow::ensure!(
        registry
            .resource_status("moss", Some("nano-candle"))
            .await?
            == talechime_backends::ResourceStatus::Ready,
        "pinned files not ready"
    );
    let (tx, mut events) = tokio::sync::mpsc::channel(32);
    let drain = async { while events.recv().await.is_some() {} };
    let run = async {
        let backend = model
            .load_existing(
                talechime_backends::PrepareOptions {
                    device: Device::Cpu,
                    ..Default::default()
                },
                tx,
            )
            .await?;
        let params = Default::default();
        let mut audio = backend
            .stream(SegmentRequest {
                text: "你好，欢迎回来。",
                voice: "Weiguo",
                style: None,
                seed: 42,
                params: &params,
            })
            .await?;
        let mut samples = 0usize;
        let mut peak = 0f32;
        let mut ended = false;
        while let Some(chunk) = audio.recv().await {
            match chunk? {
                AudioChunk::Pcm(pcm) => {
                    anyhow::ensure!(
                        pcm.sample_rate == 48000
                            && pcm.channels == 2
                            && pcm.samples.iter().all(|x| x.is_finite()),
                        "invalid PCM"
                    );
                    samples += pcm.samples.len();
                    peak = pcm
                        .samples
                        .iter()
                        .fold(peak, |value, sample| value.max(sample.abs()));
                }
                AudioChunk::End => {
                    ended = true;
                    break;
                }
            }
        }
        anyhow::ensure!(
            ended && samples > 0 && peak > 0.0001,
            "audio did not complete or was silent"
        );
        println!("MOSS Nano CPU: {samples} interleaved samples at 48000 Hz stereo; peak {peak:.4}");
        Ok::<_, anyhow::Error>(())
    };
    let (result, ()) = tokio::join!(run, drain);
    result?;
    anyhow::ensure!(
        !data.path().join("application-data/moss/models").exists(),
        "created a model directory"
    );
    Ok(())
}

#[cfg(feature = "moss-nano-candle")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires pinned Nano codec files in an explicitly selected Hub cache"]
async fn moss_nano_reference_import_writes_only_to_application_data() -> anyhow::Result<()> {
    let source = std::path::PathBuf::from(
        std::env::var_os("TALECHIME_TEST_HUB_CACHE").expect("set TALECHIME_TEST_HUB_CACHE"),
    );
    let root = tempfile::tempdir()?;
    let revision = "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68";
    let repo = "models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano";
    let target = root
        .path()
        .join("hub")
        .join(repo)
        .join("snapshots")
        .join(revision);
    std::fs::create_dir_all(&target)?;
    for name in [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
    ] {
        let original = source
            .join(repo)
            .join("snapshots")
            .join(revision)
            .join(name)
            .canonicalize()?;
        std::fs::hard_link(original, target.join(name))?;
    }
    let wav = root.path().join("reference.wav");
    let mut writer = hound::WavWriter::create(
        &wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for i in 0..16000 {
        writer.write_sample(((i as f32 * 0.05).sin() * 8000.0) as i16)?;
    }
    writer.finalize()?;
    let data = root.path().join("data");
    let downloads = talechime_backends::downloads::Downloads::new(
        &talechime_backends::downloads::DownloadConfig {
            cache_dir: root.path().join("hub"),
            endpoint: Some("http://127.0.0.1:9".into()),
            token: None,
        },
    )?;
    talechime_backends::moss::nano::import_voice(
        data.clone(),
        &downloads,
        "custom:fixture".into(),
        "Fixture".into(),
        wav,
    )
    .await?;
    let voice = data
        .join("moss/nano-candle")
        .join(talechime_backends::moss::nano::REVISION)
        .join("voices/fixture.json");
    anyhow::ensure!(
        voice.is_file(),
        "reference codes were not saved to application data"
    );
    anyhow::ensure!(
        !data.join("moss/models").exists(),
        "created a model directory"
    );
    Ok(())
}

#[cfg(any(feature = "cosyvoice3", feature = "indextts"))]
async fn cached_reference_model_roundtrip(
    backend: &str,
    model_id: &str,
    sample_rate: u32,
) -> anyhow::Result<()> {
    use talechime_core::backend::{AudioChunk, SegmentRequest};
    use talechime_protocol::Device;
    let cache = std::env::var_os("TALECHIME_TEST_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .expect("set TALECHIME_TEST_HUB_CACHE to an absolute isolated cache");
    anyhow::ensure!(cache.is_absolute(), "cache must be absolute");
    let temp = tempfile::tempdir()?;
    let data = temp.path().join("application-data");
    let wav = temp.path().join("synthetic-reference.wav");
    let mut writer = hound::WavWriter::create(
        &wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for i in 0..48000 {
        let sample =
            ((i as f32 * 2.0 * std::f32::consts::PI * 220.0 / 24000.0).sin() * 3500.0) as i16;
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    let voice = "custom:fixture";
    match backend {
        #[cfg(feature = "cosyvoice3")]
        "cosyvoice3" => talechime_backends::cosyvoice3::voice_store(&data)?.import(
            voice,
            "synthetic",
            &wav,
            "你好。",
            None,
        )?,
        #[cfg(feature = "indextts")]
        "indextts" => talechime_backends::indextts::voice_store(&data)?.import(
            voice,
            "synthetic",
            &wav,
            "你好。",
            None,
        )?,
        _ => anyhow::bail!("unsupported backend"),
    }
    let registry = talechime_backends::Registry::with_storage_paths(
        data.clone(),
        temp.path().join("application-cache"),
        &talechime_backends::downloads::DownloadConfig {
            cache_dir: cache,
            endpoint: Some("http://127.0.0.1:9".into()),
            token: None,
        },
        Default::default(),
    )?;
    let model = registry.resolve(backend, Some(model_id))?;
    anyhow::ensure!(
        registry.resource_status(backend, Some(model_id)).await?
            == talechime_backends::ResourceStatus::Ready,
        "pinned files not ready"
    );
    let (tx, mut events) = tokio::sync::mpsc::channel(32);
    let drain = async { while events.recv().await.is_some() {} };
    let run = async {
        let engine = model
            .load_existing(
                talechime_backends::PrepareOptions {
                    device: Device::Cpu,
                    ..Default::default()
                },
                tx,
            )
            .await?;
        let params = Default::default();
        let mut stream = engine
            .stream(SegmentRequest {
                text: "你好。",
                voice,
                style: None,
                seed: 42,
                params: &params,
            })
            .await?;
        let mut samples = 0usize;
        let mut peak = 0f32;
        let mut ended = false;
        while let Some(chunk) = stream.recv().await {
            match chunk? {
                AudioChunk::Pcm(pcm) => {
                    anyhow::ensure!(
                        pcm.sample_rate == sample_rate
                            && pcm.channels == 1
                            && pcm.samples.iter().all(|x| x.is_finite()),
                        "invalid PCM"
                    );
                    samples += pcm.samples.len();
                    peak = pcm.samples.iter().fold(peak, |peak, x| peak.max(x.abs()));
                }
                AudioChunk::End => {
                    ended = true;
                    break;
                }
            }
        }
        anyhow::ensure!(
            ended && samples > 0 && peak > 0.0001,
            "audio did not complete or was silent"
        );
        println!(
            "{backend} CPU: {samples} samples at {sample_rate} Hz; peak {peak:.4}; synthetic reference"
        );
        Ok::<_, anyhow::Error>(())
    };
    let (result, ()) = tokio::join!(run, drain);
    result?;
    anyhow::ensure!(
        !data.join(backend).join("models").exists(),
        "created a model directory"
    );
    Ok(())
}

#[cfg(feature = "cosyvoice3")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires explicitly prepared isolated pinned Hub cache and CPU inference"]
async fn cosyvoice3_loads_and_synthesizes_directly_from_cached_files() -> anyhow::Result<()> {
    cached_reference_model_roundtrip("cosyvoice3", "0.5b", 24000).await
}

#[cfg(feature = "indextts")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "requires explicitly prepared isolated pinned Hub cache and CPU inference"]
async fn indextts_loads_and_synthesizes_directly_from_cached_files() -> anyhow::Result<()> {
    cached_reference_model_roundtrip("indextts", "2.5", 22050).await
}

#[cfg(feature = "cosyvoice3")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "downloads published WeText files into an explicitly selected isolated cache"]
async fn cosyvoice3_downloads_published_frontend_to_isolated_hub_cache() -> anyhow::Result<()> {
    let cache = std::path::PathBuf::from(
        std::env::var_os("TALECHIME_TEST_HUB_CACHE").expect("isolated cache"),
    );
    anyhow::ensure!(cache.is_absolute(), "cache must be absolute");
    let data = tempfile::tempdir()?;
    let registry = talechime_backends::Registry::with_storage_paths(
        data.path().join("data"),
        data.path().join("app-cache"),
        &talechime_backends::downloads::DownloadConfig {
            cache_dir: cache,
            endpoint: None,
            token: None,
        },
        Default::default(),
    )?;
    let status = registry.resource_status("cosyvoice3", Some("0.5b")).await?;
    anyhow::ensure!(
        matches!(status, talechime_backends::ResourceStatus::Partial { .. }),
        "expected partially populated cache: {status:?}"
    );
    let model = registry.resolve("cosyvoice3", Some("0.5b"))?;
    let (tx, mut events) = tokio::sync::mpsc::channel(32);
    let drain = async { while events.recv().await.is_some() {} };
    let (result, ()) = tokio::join!(model.prepare_resources(tx), drain);
    result?;
    anyhow::ensure!(
        registry.resource_status("cosyvoice3", Some("0.5b")).await?
            == talechime_backends::ResourceStatus::Ready,
        "published resources are not ready"
    );
    anyhow::ensure!(
        !data.path().join("data").exists(),
        "created application data while preparing models"
    );
    Ok(())
}

#[cfg(feature = "indextts")]
#[tokio::test(flavor = "current_thread")]
#[ignore = "downloads published IndexTTS frontend files into an explicitly selected isolated cache"]
async fn indextts_downloads_published_frontend_to_isolated_hub_cache() -> anyhow::Result<()> {
    let cache = std::path::PathBuf::from(
        std::env::var_os("TALECHIME_TEST_HUB_CACHE").expect("isolated cache"),
    );
    anyhow::ensure!(cache.is_absolute(), "cache must be absolute");
    let data = tempfile::tempdir()?;
    let registry = talechime_backends::Registry::with_storage_paths(
        data.path().join("data"),
        data.path().join("app-cache"),
        &talechime_backends::downloads::DownloadConfig {
            cache_dir: cache,
            endpoint: None,
            token: None,
        },
        Default::default(),
    )?;
    let status = registry.resource_status("indextts", Some("2.5")).await?;
    anyhow::ensure!(
        matches!(status, talechime_backends::ResourceStatus::Partial { .. }),
        "expected partially populated cache: {status:?}"
    );
    let model = registry.resolve("indextts", Some("2.5"))?;
    let (tx, mut events) = tokio::sync::mpsc::channel(32);
    let drain = async { while events.recv().await.is_some() {} };
    let (result, ()) = tokio::join!(model.prepare_resources(tx), drain);
    result?;
    anyhow::ensure!(
        registry.resource_status("indextts", Some("2.5")).await?
            == talechime_backends::ResourceStatus::Ready,
        "published resources are not ready"
    );
    anyhow::ensure!(
        !data.path().join("data").exists(),
        "created application data while preparing models"
    );
    Ok(())
}
