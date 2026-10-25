//! Explicit local CUDA owner-state tests. Ordinary tests never load these models.
use super::*;
use std::time::Duration;
use talechime_core::{
    backend::{AudioChunk, SegmentRequest},
    params::GenerationParams,
};
use talechime_protocol::ParamValue;

const TEXT: &str = "你好，欢迎收听今天的故事。";

fn request<'a>(voice: &'a str, params: &'a GenerationParams) -> SegmentRequest<'a> {
    SegmentRequest {
        text: TEXT,
        voice,
        style: None,
        context: None,
        seed: talechime_core::SeedPolicy::Pinned(42).resolve(0, 0),
        params,
    }
}

async fn collect(
    backend: &VoxBackend,
    voice: &str,
    params: &GenerationParams,
) -> anyhow::Result<Vec<u32>> {
    let mut stream = backend.stream(request(voice, params)).await?;
    let mut bits = Vec::new();
    let mut ended = false;
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(30), stream.recv()).await? {
        match chunk? {
            AudioChunk::Pcm(pcm) => {
                pcm.duration_ms()?;
                anyhow::ensure!((pcm.sample_rate, pcm.channels) == (48000, 1));
                anyhow::ensure!(pcm.samples.iter().all(|v| v.is_finite()));
                bits.extend(pcm.samples.iter().map(|v| v.to_bits()));
            }
            AudioChunk::End => {
                ended = true;
                break;
            }
        }
    }
    anyhow::ensure!(ended && !bits.is_empty(), "incomplete Vox request");
    Ok(bits)
}

async fn verify(model: models::Model, voice: &str, alternate: Option<&str>) -> anyhow::Result<()> {
    let root = PathBuf::from(
        std::env::var_os("TALECHIME_VOX_STATE_RESOURCES")
            .ok_or_else(|| anyhow::anyhow!("TALECHIME_VOX_STATE_RESOURCES required"))?,
    );
    let backend = VoxBackend::load_model_on(model.directory(&root), model, Device::Cuda).await?;
    let params = GenerationParams::new();
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
        "budget did not fail after partial PCM without End: error={budget_error}, ended={ended}, partial_samples={partial_samples}, normal_samples={}",
        expected.len()
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
    eprintln!(
        "{}: {} raw PCM bits preserved after unread/full-queue cancellation, invalid voice and partial frame-budget failure ({} partial samples, no End)",
        model.id(),
        expected.len(),
        partial_samples
    );
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "explicit local CUDA Q8 model and custom:acceptance; no downloads"]
async fn real_cuda_q8_owner_state() -> anyhow::Result<()> {
    tokio::task::LocalSet::new()
        .run_until(async {
            tokio::time::timeout(
                Duration::from_secs(180),
                verify(models::Model::Q8, "custom:acceptance", Some("narrator")),
            )
            .await?
        })
        .await
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "explicit local CUDA BF16 model; no downloads"]
async fn real_cuda_bf16_owner_state() -> anyhow::Result<()> {
    tokio::task::LocalSet::new()
        .run_until(async {
            tokio::time::timeout(
                Duration::from_secs(180),
                verify(models::Model::OriginalBf16, "narrator", None),
            )
            .await?
        })
        .await
}
