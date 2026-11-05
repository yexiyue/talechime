//! Opt-in acceptance against explicitly prepared models and generated reference audio.
#[tokio::test]
#[ignore = "loads multi-GB prepared resources and synthesizes a new WAV; no downloads"]
async fn reference_backend_real_synthesis() -> anyhow::Result<()> {
    use std::{path::PathBuf, rc::Rc};
    use talechime_core::{
        backend::{Backend, SegmentRequest},
        params::GenerationParams,
    };
    use talechime_protocol::Device;
    let root =
        PathBuf::from(std::env::var_os("TALECHIME_NATIVE_ENGINE_ROOT").expect("set engine root"));
    let family = std::env::var("TALECHIME_NATIVE_BACKEND").expect("set backend");
    let device = match std::env::var("TALECHIME_NATIVE_DEVICE")
        .as_deref()
        .unwrap_or("cpu")
    {
        "cpu" => Device::Cpu,
        "metal" => Device::Metal,
        _ => anyhow::bail!("unsupported test device"),
    };
    let reference = PathBuf::from(
        std::env::var_os("TALECHIME_NATIVE_REFERENCE").expect("set generated reference WAV"),
    );
    let output =
        PathBuf::from(std::env::var_os("TALECHIME_NATIVE_AUDIO_OUTPUT").expect("set new WAV path"));
    anyhow::ensure!(
        root.is_absolute() && reference.is_absolute() && output.is_absolute() && !output.exists(),
        "explicit absolute paths required"
    );
    let started = std::time::Instant::now();
    let store = match family.as_str() {
        #[cfg(feature = "cosyvoice3")]
        "cosyvoice3" => crate::cosyvoice3::voice_store(&root)?,
        #[cfg(feature = "indextts")]
        "indextts" => crate::indextts::voice_store(&root)?,
        _ => anyhow::bail!("backend not compiled"),
    };
    let voice = "custom:native_acceptance";
    if !store.path(voice)?.exists() {
        store.import(
            voice,
            "Generated acceptance reference",
            &reference,
            "春天来了，院子里的花开了。",
            None,
        )?;
    }
    let backend: Rc<dyn Backend> = match family.as_str() {
        #[cfg(feature = "cosyvoice3")]
        "cosyvoice3" => Rc::new(crate::cosyvoice3::CosyBackend::load_on(root, device).await?),
        #[cfg(feature = "indextts")]
        "indextts" => Rc::new(crate::indextts::IndexBackend::load_on(root, device).await?),
        _ => anyhow::bail!("backend not compiled"),
    };
    let loaded = started.elapsed();
    let params = GenerationParams::new();
    let pcm = backend
        .synthesize(SegmentRequest {
            text: "你好，欢迎回来。",
            voice,
            style: None,
            seed: 7,
            params: &params,
        })
        .await?;
    let duration = pcm.duration_ms()?;
    anyhow::ensure!(
        duration > 0 && pcm.samples.iter().all(|s| s.is_finite()),
        "invalid PCM"
    );
    let rms = (pcm
        .samples
        .iter()
        .map(|s| f64::from(*s).powi(2))
        .sum::<f64>()
        / pcm.samples.len() as f64)
        .sqrt();
    anyhow::ensure!(rms > 0.001, "silent PCM");
    let spec = hound::WavSpec {
        channels: pcm.channels,
        sample_rate: pcm.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut wav = hound::WavWriter::new(std::fs::File::create_new(&output)?, spec)?;
    for sample in &pcm.samples {
        wav.write_sample(*sample)?;
    }
    wav.finalize()?;
    anyhow::ensure!(hound::WavReader::open(&output)?.duration() > 0, "empty WAV");
    println!(
        "backend={family} device={device:?} load={loaded:?} total={:?} samples={} duration_ms={duration} rms={rms:.6}; listening not evaluated",
        started.elapsed(),
        pcm.samples.len()
    );
    Ok(())
}

#[cfg(feature = "kokoro")]
#[tokio::test]
#[ignore = "explicit real published Kokoro files; new isolated root and unavailable HTTP source"]
async fn published_resources_prepare_offline_without_any_input_cache() -> anyhow::Result<()> {
    let source = std::path::PathBuf::from(
        std::env::var_os("TALECHIME_NATIVE_ENGINE_ROOT").expect("set prepared root"),
    );
    let isolated = tempfile::tempdir_in(
        source
            .parent()
            .ok_or_else(|| anyhow::anyhow!("root has no parent"))?,
    )?;
    let recipe = crate::kokoro::native_recipe(crate::kokoro::MODEL).unwrap();
    let published = isolated.path().join(&recipe.target);
    std::fs::create_dir_all(&published)?;
    for name in recipe
        .outputs
        .iter()
        .map(|pin| pin.path)
        .chain(std::iter::once("preparation.json"))
    {
        let dest = published.join(name);
        std::fs::create_dir_all(dest.parent().unwrap())?;
        std::fs::hard_link(source.join(&recipe.target).join(name), dest)?;
    }
    let cache = isolated.path().join("empty-offline-hf-cache");
    let downloads = crate::downloads::Downloads::with_options(
        &crate::downloads::DownloadConfig {
            cache_dir: cache.clone(),
            endpoint: Some("http://127.0.0.1:9".into()),
            token: None,
        },
        crate::downloads::DownloadOptions {
            transport: crate::downloads::DownloadTransport::Http,
            ..Default::default()
        },
    )?;
    let control = super::PreparationControl::default();
    let plan = super::plan_recipe(
        recipe.clone(),
        isolated.path().into(),
        downloads.clone(),
        control.clone(),
    )
    .await?;
    anyhow::ensure!(
        plan.ready
            && plan.download_bytes == 0
            && plan.peak_additional_bytes == 0
            && plan.conversions.is_empty(),
        "published offline plan is not ready"
    );
    let (stages, _) = tokio::sync::mpsc::channel(8);
    let (files, _) = tokio::sync::mpsc::channel(8);
    recipe
        .execute(
            isolated.path().into(),
            downloads,
            plan.fingerprint,
            control,
            stages,
            files,
        )
        .await?;
    anyhow::ensure!(
        !cache.exists() && !isolated.path().join(".native-preparation").exists(),
        "ready-only preparation created a transfer workspace"
    );
    println!(
        "published Kokoro ready without any input cache, unavailable HTTP source, zero download/conversion/workspace"
    );
    Ok(())
}
