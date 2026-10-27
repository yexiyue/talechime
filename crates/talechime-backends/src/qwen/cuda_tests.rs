//! Explicit local CUDA owner-state tests. Ordinary tests never load these models.
use super::*;
use std::time::Duration;
use talechime_core::{
    backend::{AudioChunk, SegmentRequest},
    params::GenerationParams,
};
use talechime_protocol::ParamValue;

const TEXT: &str = "你好，欢迎收听今天的故事。";

struct TemporaryReference {
    store: Option<talechime_core::voices::VoiceStore>,
    id: String,
}

impl TemporaryReference {
    fn import(directory: &std::path::Path, model: models::Model) -> anyhow::Result<Self> {
        use sha2::{Digest, Sha256};
        let path = PathBuf::from(
            std::env::var_os("TALECHIME_QWEN_STATE_REFERENCE")
                .ok_or_else(|| anyhow::anyhow!("TALECHIME_QWEN_STATE_REFERENCE required"))?,
        );
        let expected = std::env::var("TALECHIME_QWEN_STATE_REFERENCE_SHA256")?;
        anyhow::ensure!(
            format!("{:x}", Sha256::digest(std::fs::read(&path)?)) == expected,
            "cross-reference fixture changed"
        );
        validate_wav(&path)?;
        let store = voice_store(directory, model)?;
        let id = format!(
            "custom:cuda-reference-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        );
        store.import(&id, "temporary generated CUDA reference", &path, TEXT, None)?;
        Ok(Self {
            store: Some(store),
            id,
        })
    }

    fn cache(&self) -> anyhow::Result<PathBuf> {
        Ok(self
            .store
            .as_ref()
            .unwrap()
            .path(&self.id)?
            .join("prompt-encoder-rope-v2.json"))
    }

    fn remove(mut self) -> anyhow::Result<()> {
        let store = self.store.take().unwrap();
        let directory = store.path(&self.id)?;
        store.remove(&self.id)?;
        anyhow::ensure!(!directory.exists(), "temporary reference directory remains");
        Ok(())
    }
}

impl Drop for TemporaryReference {
    fn drop(&mut self) {
        if let Some(store) = &self.store
            && let Err(error) = store.remove(&self.id)
        {
            eprintln!("temporary reference cleanup failed: {error}");
        }
    }
}

fn request<'a>(voice: &'a str, params: &'a GenerationParams) -> SegmentRequest<'a> {
    SegmentRequest {
        text: TEXT,
        voice,
        style: None,

        seed: talechime_core::SeedPolicy::Pinned(42).resolve(0, 0),
        params,
    }
}

async fn collect(
    backend: &QwenBackend,
    voice: &str,
    params: &GenerationParams,
) -> anyhow::Result<Vec<u32>> {
    collect_request(backend, request(voice, params)).await
}

async fn collect_request(
    backend: &QwenBackend,
    request: SegmentRequest<'_>,
) -> anyhow::Result<Vec<u32>> {
    let mut stream = backend.stream(request).await?;
    let mut bits = Vec::new();
    let mut ended = false;
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(30), stream.recv()).await? {
        match chunk? {
            AudioChunk::Pcm(pcm) => {
                pcm.duration_ms()?;
                anyhow::ensure!((pcm.sample_rate, pcm.channels) == (24000, 1));
                anyhow::ensure!(pcm.samples.iter().all(|v| v.is_finite()));
                bits.extend(pcm.samples.iter().map(|v| v.to_bits()));
            }
            AudioChunk::End => {
                ended = true;
                break;
            }
        }
    }
    anyhow::ensure!(ended && !bits.is_empty(), "incomplete Qwen request");
    Ok(bits)
}

async fn verify(model: models::Model, voice: &str, alternate: Option<&str>) -> anyhow::Result<()> {
    let root = PathBuf::from(
        std::env::var_os("TALECHIME_QWEN_STATE_RESOURCES")
            .ok_or_else(|| anyhow::anyhow!("TALECHIME_QWEN_STATE_RESOURCES required"))?,
    );
    let directory = model.directory(&root);
    let mut alternate_reference = if model.is_base() {
        Some(TemporaryReference::import(&directory, model)?)
    } else {
        None
    };
    let mut backend = QwenBackend::load_on(directory, Device::Cuda).await?;
    let mut params = GenerationParams::new();
    params.insert(
        "repetition_penalty",
        ParamValue::Float(if model.is_base() { 1.5 } else { 1.05 }),
    );
    params.insert("chunk_frames", ParamValue::Int(20));
    params.insert("max_duration", ParamValue::Float(30.));
    let expected = collect(&backend, voice, &params).await?;

    // Cancel before consuming any output, then reuse the same owner.
    drop(backend.stream(request(voice, &params)).await?);
    assert_eq!(collect(&backend, voice, &params).await?, expected);

    let mut stream = backend.stream(request(voice, &params)).await?;
    assert_eq!(stream.max_capacity(), 1);
    anyhow::ensure!(matches!(stream.recv().await, Some(Ok(AudioChunk::Pcm(_)))));
    tokio::time::timeout(Duration::from_secs(30), async {
        while stream.len() != 1 {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await?;
    // This observes a full queue, not proof of an actively blocked send.
    drop(stream);
    assert_eq!(collect(&backend, voice, &params).await?, expected);

    assert!(
        backend
            .stream(request("custom:missing-state-test-voice", &params))
            .await
            .is_err()
    );
    assert_eq!(collect(&backend, voice, &params).await?, expected);

    // A valid one-second request budget must fail after partial generation,
    // without publishing End or corrupting the next full request's state.
    let mut limited = params.clone();
    limited.insert("max_duration", ParamValue::Float(1.));
    let mut stream = backend.stream(request(voice, &limited)).await?;
    let mut budget_error = false;
    let mut ended = false;
    let mut partial_samples = 0;
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(30), stream.recv()).await? {
        match chunk {
            Ok(AudioChunk::Pcm(pcm)) => {
                pcm.duration_ms()?;
                anyhow::ensure!(pcm.samples.iter().all(|v| v.is_finite()));
                partial_samples += pcm.samples.len();
            }
            Ok(AudioChunk::End) => {
                ended = true;
                break;
            }
            Err(error) => {
                anyhow::ensure!(
                    error.to_string().contains("frame limit"),
                    "unexpected failure: {error}"
                );
                budget_error = true;
                break;
            }
        }
    }
    // A failed request must close, rather than append a success End after Err.
    anyhow::ensure!(
        tokio::time::timeout(Duration::from_secs(5), stream.recv())
            .await?
            .is_none(),
        "unexpected audio or End after frame-budget failure"
    );
    drop(stream);
    anyhow::ensure!(
        budget_error && !ended && partial_samples > 0,
        "budget did not fail after partial PCM without End"
    );
    assert_eq!(collect(&backend, voice, &params).await?, expected);
    if let Some(alternate) = alternate {
        let other = collect(&backend, alternate, &params).await?;
        assert_ne!(
            other, expected,
            "alternate voice must produce a distinct sample"
        );
        assert_eq!(collect(&backend, voice, &params).await?, expected);
        eprintln!(
            "{}: {voice} -> {alternate} -> {voice} preserves original PCM",
            model.id()
        );
    }
    if let Some(reference) = alternate_reference.take() {
        let cache = reference.cache()?;
        anyhow::ensure!(
            !cache.exists(),
            "new reference unexpectedly has a prompt cache"
        );
        let other = collect(&backend, &reference.id, &params).await?;
        assert_ne!(
            other, expected,
            "distinct reference should affect this fixture"
        );
        let cache_bytes = std::fs::read(&cache)?;
        anyhow::ensure!(!cache_bytes.is_empty());
        assert_eq!(collect(&backend, voice, &params).await?, expected);
        assert_eq!(collect(&backend, &reference.id, &params).await?, other);
        drop(backend.stream(request(&reference.id, &params)).await?);
        assert_eq!(collect(&backend, voice, &params).await?, expected);
        let mut stream = backend.stream(request(&reference.id, &params)).await?;
        anyhow::ensure!(matches!(stream.recv().await, Some(Ok(AudioChunk::Pcm(_)))));
        tokio::time::timeout(Duration::from_secs(30), async {
            while stream.len() != 1 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await?;
        drop(stream);
        assert_eq!(collect(&backend, &reference.id, &params).await?, other);
        assert_eq!(collect(&backend, voice, &params).await?, expected);
        assert_eq!(std::fs::read(&cache)?, cache_bytes);
        let original_owner = backend.thread.as_ref().unwrap().thread().id();
        drop(backend);
        // Only the test-owned copy is hidden; the pinned input fixture stays intact.
        // A new owner cannot re-encode it and must load the serialized prompt.
        let waveform = cache.parent().unwrap().join("reference.wav");
        let hidden_waveform = cache.parent().unwrap().join("reference.offline.wav");
        std::fs::rename(&waveform, &hidden_waveform)?;
        backend = QwenBackend::load_on(model.directory(&root), Device::Cuda).await?;
        let new_owner = backend.thread.as_ref().unwrap().thread().id();
        assert_ne!(new_owner, original_owner);
        assert_eq!(collect(&backend, &reference.id, &params).await?, other);
        assert_eq!(collect(&backend, voice, &params).await?, expected);
        assert_eq!(std::fs::read(&cache)?, cache_bytes);
        anyhow::ensure!(!waveform.exists());
        std::fs::rename(hidden_waveform, waveform)?;
        eprintln!(
            "{}: fresh owner {original_owner:?} -> {new_owner:?} reloads cached B without reference WAV and restores A bitwise",
            model.id()
        );
        reference.remove()?;
        eprintln!(
            "{}: second reference {} raw PCM bits preserves A/B/A and unread/full-queue cancellation; temporary store removed",
            model.id(),
            other.len()
        );
    }
    if model == models::Model::Custom17 {
        let styled_request = || SegmentRequest {
            style: Some("用温柔而平静的语气讲述。"),
            ..request(voice, &params)
        };
        let styled = collect_request(&backend, styled_request()).await?;
        assert_ne!(styled, expected, "style must affect this fixed request");
        assert_eq!(collect_request(&backend, styled_request()).await?, styled);
        drop(backend.stream(styled_request()).await?);
        assert_eq!(collect(&backend, voice, &params).await?, expected);
        let mut stream = backend.stream(styled_request()).await?;
        anyhow::ensure!(matches!(stream.recv().await, Some(Ok(AudioChunk::Pcm(_)))));
        tokio::time::timeout(Duration::from_secs(30), async {
            while stream.len() != 1 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await?;
        drop(stream);
        assert_eq!(collect_request(&backend, styled_request()).await?, styled);
        assert_eq!(collect(&backend, voice, &params).await?, expected);
        eprintln!(
            "{}: styled PCM {} bits repeat after unread/full-queue cancellation; plain PCM restored",
            model.id(),
            styled.len()
        );
    }
    eprintln!(
        "{}: {} raw PCM bits preserved after unread/full-queue cancellation, invalid voice and partial frame-budget failure ({} partial samples, no End)",
        model.id(),
        expected.len(),
        partial_samples
    );
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "explicit local CUDA Base06 model and custom:acceptance06; no downloads"]
async fn real_cuda_base06_state_reuse() -> anyhow::Result<()> {
    tokio::task::LocalSet::new()
        .run_until(async {
            tokio::time::timeout(
                Duration::from_secs(180),
                verify(models::Model::Base06, "custom:acceptance06", None),
            )
            .await?
        })
        .await
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "explicit local CUDA Base17 model and custom:acceptance; no downloads"]
async fn real_cuda_base17_state_reuse() -> anyhow::Result<()> {
    tokio::task::LocalSet::new()
        .run_until(async {
            tokio::time::timeout(
                Duration::from_secs(180),
                verify(models::Model::Base17, "custom:acceptance", None),
            )
            .await?
        })
        .await
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "explicit local CUDA Custom06 model; no downloads"]
async fn real_cuda_custom06_state_and_voice_switch() -> anyhow::Result<()> {
    tokio::task::LocalSet::new()
        .run_until(async {
            tokio::time::timeout(
                Duration::from_secs(180),
                verify(models::Model::Custom06, "uncle_fu", Some("serena")),
            )
            .await?
        })
        .await
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "explicit local CUDA Custom17 model; no downloads"]
async fn real_cuda_custom17_state_and_voice_switch() -> anyhow::Result<()> {
    tokio::task::LocalSet::new()
        .run_until(async {
            tokio::time::timeout(
                Duration::from_secs(180),
                verify(models::Model::Custom17, "uncle_fu", Some("serena")),
            )
            .await?
        })
        .await
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "explicit local CUDA VoiceDesign model and new output directory; no downloads"]
async fn real_cuda_voice_design_production_repeat() -> anyhow::Result<()> {
    let root = PathBuf::from(
        std::env::var_os("TALECHIME_QWEN_STATE_RESOURCES")
            .ok_or_else(|| anyhow::anyhow!("TALECHIME_QWEN_STATE_RESOURCES required"))?,
    );
    let output = PathBuf::from(
        std::env::var_os("TALECHIME_QWEN_DESIGN_OUTPUT")
            .ok_or_else(|| anyhow::anyhow!("TALECHIME_QWEN_DESIGN_OUTPUT required"))?,
    );
    std::fs::create_dir(&output)?;
    let mut expected = None;
    for index in 0..2 {
        let path = output.join(format!("reference-{index}.wav"));
        tokio::time::timeout(
            Duration::from_secs(180),
            design::reference(
                models::Model::Design17.directory(&root),
                Device::Cuda,
                TEXT.to_owned(),
                "年轻女性，声音清晰自然，用平静温柔的语气讲述。".to_owned(),
                path.clone(),
            ),
        )
        .await??;
        let audio = talechime_qwen3_tts::AudioBuffer::load(&path)?;
        validate_reference(&audio)?;
        let bits: Vec<_> = audio.samples.iter().map(|value| value.to_bits()).collect();
        if let Some(expected) = &expected {
            assert_eq!(&bits, expected);
        } else {
            expected = Some(bits);
        }
    }
    eprintln!(
        "VoiceDesign: two independent production loads preserve {} exported PCM bits",
        expected.unwrap().len()
    );
    Ok(())
}
