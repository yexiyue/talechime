//! Explicit 1.7B owner-state gates; never download resources in ordinary tests.
use super::*;
use std::time::Duration;
use talechime_core::{
    backend::{AudioChunk, SegmentRequest},
    params::GenerationParams,
};

const TEXT: &str = "你好，欢迎使用叙铃。";

fn request(params: &GenerationParams) -> SegmentRequest<'_> {
    SegmentRequest {
        text: TEXT,
        voice: "narrator",
        style: None,

        seed: talechime_core::SeedPolicy::Pinned(42).resolve(0, 0),
        params,
    }
}

async fn collect(backend: &CandleBackend, params: &GenerationParams) -> anyhow::Result<Vec<u32>> {
    collect_request(backend, request(params)).await
}

async fn collect_request(
    backend: &CandleBackend,
    request: SegmentRequest<'_>,
) -> anyhow::Result<Vec<u32>> {
    let mut stream = backend.stream(request).await?;
    let mut bits = Vec::new();
    let mut ended = false;
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(60), stream.recv()).await? {
        match chunk? {
            AudioChunk::Pcm(pcm) => {
                pcm.duration_ms()?;
                anyhow::ensure!((pcm.sample_rate, pcm.channels) == (24000, 1));
                anyhow::ensure!(pcm.samples.iter().all(|sample| sample.is_finite()));
                bits.extend(pcm.samples.into_iter().map(f32::to_bits));
            }
            AudioChunk::End => {
                ended = true;
                break;
            }
        }
    }
    anyhow::ensure!(ended && !bits.is_empty(), "incomplete MOSS request");
    Ok(bits)
}

async fn check(mode: Mode) -> anyhow::Result<()> {
    tokio::task::LocalSet::new()
        .run_until(async {
            let root = std::env::var_os("TALECHIME_MOSS17_CUDA_ROOT")
                .ok_or_else(|| anyhow::anyhow!("explicit local MOSS resource root required"))?;
            let backend = CandleBackend::load_on(root.into(), mode, Device::Cuda).await?;
            let params = GenerationParams::new();
            let expected = collect(&backend, &params).await?;
            drop(backend.stream(request(&params)).await?);
            assert_eq!(collect(&backend, &params).await?, expected);

            let mut stream = backend.stream(request(&params)).await?;
            assert_eq!(stream.max_capacity(), 1);
            assert!(matches!(
                tokio::time::timeout(Duration::from_secs(60), stream.recv()).await?,
                Some(Ok(AudioChunk::Pcm(_)))
            ));
            tokio::time::timeout(Duration::from_secs(60), async {
                while stream.len() != 1 {
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
            })
            .await?;
            drop(stream);
            assert_eq!(collect(&backend, &params).await?, expected);

            let limited = params::MossParams::default()
                .max_duration_seconds(1.0)
                .to_generation_params();
            let mut stream = backend.stream(request(&limited)).await?;
            let mut failed = false;
            while let Some(chunk) =
                tokio::time::timeout(Duration::from_secs(60), stream.recv()).await?
            {
                match chunk {
                    Err(error) => {
                        anyhow::ensure!(error.to_string().contains("frame limit reached before EOS"));
                        failed = true;
                    }
                    Ok(AudioChunk::End) => anyhow::bail!("frame-limit failure emitted End"),
                    Ok(AudioChunk::Pcm(_)) => {}
                }
            }
            anyhow::ensure!(failed, "expected actual inference frame-limit failure");
            assert_eq!(collect(&backend, &params).await?, expected);
            eprintln!(
                "{}: {} identical PCM samples after unconsumed/full-queue cancellation and actual inference failure; normal End present, failed End absent",
                mode.id(), expected.len()
            );
            Ok(())
        })
        .await
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "explicit local MOSS Realtime 1.7B CUDA resources; no downloads"]
async fn real_cuda_realtime_owner_cancel_failure_and_reuse() -> anyhow::Result<()> {
    check(Mode::Realtime).await
}

struct TemporaryVoice {
    store: Option<talechime_core::voices::VoiceStore>,
    id: String,
}

impl Drop for TemporaryVoice {
    fn drop(&mut self) {
        if let Some(store) = &self.store {
            let _ = store.remove(&self.id);
        }
    }
}

async fn check_fixed_reference(mode: Mode) -> anyhow::Result<()> {
    tokio::task::LocalSet::new()
        .run_until(async {
            let root = PathBuf::from(
                std::env::var_os("TALECHIME_MOSS17_CUDA_ROOT")
                    .ok_or_else(|| anyhow::anyhow!("explicit local MOSS resource root required"))?,
            );
            let params = GenerationParams::new();
            // Generate our own complete reference; never use private recordings.
            let backend = CandleBackend::load_on(root.clone(), mode, Device::Cuda).await?;
            let narrator = collect(&backend, &params).await?;
            drop(backend);
            let samples: Vec<f32> = narrator.iter().copied().map(f32::from_bits).collect();
            let temporary = tempfile::tempdir()?;
            let wav = temporary.path().join("generated-reference.wav");
            let mut writer = hound::WavWriter::create(
                &wav,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 24000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )?;
            for sample in samples {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
            let store = voice_store(&mode.directory(&root), mode)?;
            let id = format!(
                "custom:cuda-gate-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_nanos()
            );
            let voice_path = store.path(&id)?;
            anyhow::ensure!(!voice_path.exists());
            store.import(&id, "generated CUDA gate reference", &wav, TEXT, None)?;
            let mut voice = TemporaryVoice {
                store: Some(store),
                id,
            };
            let backend = CandleBackend::load_on(root, mode, Device::Cuda).await?;
            let clone_request = || SegmentRequest {
                voice: &voice.id,
                ..request(&params)
            };
            let cloned = collect_request(&backend, clone_request()).await?;
            let cache_path = voice_path.join("moss-prompt.json");
            let cache = std::fs::read(&cache_path)?;
            assert_eq!(collect_request(&backend, clone_request()).await?, cloned);
            assert_eq!(std::fs::read(&cache_path)?, cache);
            assert_eq!(collect(&backend, &params).await?, narrator);

            let mut stream = backend.stream(clone_request()).await?;
            anyhow::ensure!(matches!(
                tokio::time::timeout(Duration::from_secs(60), stream.recv()).await?,
                Some(Ok(AudioChunk::Pcm(_)))
            ));
            drop(stream);
            assert_eq!(collect_request(&backend, clone_request()).await?, cloned);
            assert_eq!(std::fs::read(&cache_path)?, cache);
            assert_eq!(collect(&backend, &params).await?, narrator);

            drop(backend);
            voice.store.take().unwrap().remove(&voice.id)?;
            anyhow::ensure!(!voice_path.exists(), "test voice cleanup failed");
            Ok(())
        })
        .await
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "explicit local MOSS Realtime 1.7B CUDA fixed reference gate; no downloads"]
async fn real_cuda_realtime_fixed_reference_reuse() -> anyhow::Result<()> {
    check_fixed_reference(Mode::Realtime).await
}
