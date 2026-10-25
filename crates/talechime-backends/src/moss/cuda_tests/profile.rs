//! Native Runtime trace control, isolated from product configuration and ordinary tests.
use crate::moss::{
    diagnostics::{GenerationEnd, GenerationStats},
    runtime::{Placement, Runtime},
    text,
};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Duration};
use talechime_core::backend::AudioChunk;

async fn run(directory: PathBuf, output: PathBuf, profile: bool) -> anyhow::Result<Vec<Vec<u32>>> {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    let thread = std::thread::spawn(move || -> anyhow::Result<()> {
        let mut models = Vec::new();
        let mut runtime = Runtime::load_sessions(&directory, Placement::Cuda, |path| {
            let mut builder = ort::session::Session::builder()?
                .with_intra_threads(4)
                .map_err(|error| anyhow::anyhow!("{error}"))?
                .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                .map_err(|error| anyhow::anyhow!("{error}"))?;
            if profile {
                builder = builder
                    .with_profiling(output.join(path.file_stem().unwrap()))
                    .map_err(|error| anyhow::anyhow!("{error}"))?;
            }
            models.push(path.to_string_lossy().into_owned());
            builder.commit_from_file(path).map_err(Into::into)
        })?;
        let tokenizer =
            sentencepiece_rs::SentencePieceProcessor::open(directory.join("tts/tokenizer.model"))?;
        let tokens = tokenizer
            .encode_to_ids(&text::normalize("你好，欢迎使用叙铃。"))?
            .into_iter()
            .map(|value| value as i32)
            .collect::<Vec<_>>();
        let mut reports = Vec::new();
        for _ in 0..2 {
            let mut stats = GenerationStats {
                codes: Some(Vec::new()),
                ..Default::default()
            };
            let end = runtime.generate(
                tokens.clone(),
                "Weiguo",
                Some(3560097011),
                None,
                &sender,
                &mut stats,
            )?;
            anyhow::ensure!(end == GenerationEnd::Eos && stats.frames == 35);
            reports.push(
                serde_json::json!({"frames":stats.frames,"audio_seconds":stats.audio_seconds,"codes":stats.codes}),
            );
        }
        if profile {
            let paths = runtime.finish_profiles()?;
            anyhow::ensure!(paths.len() == models.len());
            let index = paths
                .into_iter()
                .zip(models)
                .map(|(path, model)| {
                    let bytes = std::fs::read(&path)?;
                    Ok(serde_json::json!({"model":model,
                    "path":PathBuf::from(&path).file_name().unwrap().to_string_lossy(),
                    "bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            std::fs::write(
                output.join("profile-index.json"),
                serde_json::to_vec_pretty(&index)?,
            )?;
        }
        std::fs::write(
            output.join("generation.json"),
            serde_json::to_vec_pretty(&reports)?,
        )?;
        Ok(())
    });
    let collected = async {
        let mut rounds = Vec::new();
        let mut bits = Vec::new();
        while let Some(chunk) =
            tokio::time::timeout(Duration::from_secs(90), receiver.recv()).await?
        {
            match chunk? {
                AudioChunk::Pcm(pcm) => {
                    anyhow::ensure!((pcm.sample_rate, pcm.channels) == (48000, 2));
                    anyhow::ensure!(pcm.samples.iter().all(|sample| sample.is_finite()));
                    bits.extend(pcm.samples.into_iter().map(f32::to_bits));
                }
                AudioChunk::End => rounds.push(std::mem::take(&mut bits)),
            }
        }
        anyhow::ensure!(rounds.len() == 2 && bits.is_empty());
        Ok::<_, anyhow::Error>(rounds)
    }
    .await;
    drop(receiver);
    thread
        .join()
        .map_err(|_| anyhow::anyhow!("profiling owner panicked"))??;
    collected
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "explicit local CUDA ONNX models, dynamic runtime and fresh profile output; no downloads"]
async fn real_cuda_runtime_profile_preserves_raw_pcm() -> anyhow::Result<()> {
    let directory = PathBuf::from(
        std::env::var_os("TALECHIME_NANO_ONNX_ROOT")
            .ok_or_else(|| anyhow::anyhow!("TALECHIME_NANO_ONNX_ROOT required"))?,
    );
    let output = PathBuf::from(
        std::env::var_os("TALECHIME_NANO_ONNX_PROFILE_OUTPUT")
            .ok_or_else(|| anyhow::anyhow!("TALECHIME_NANO_ONNX_PROFILE_OUTPUT required"))?,
    );
    std::fs::create_dir(&output)?;
    let mut reference = None;
    let mut reference_codes = None;
    for (name, profile) in [("control", false), ("profile", true)] {
        let path = output.join(name);
        std::fs::create_dir(&path)?;
        let rounds = run(directory.clone(), path.clone(), profile).await?;
        let generation: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("generation.json"))?)?;
        if let Some(expected) = &reference_codes {
            anyhow::ensure!(
                &generation == expected,
                "profiling changed codes/completion"
            );
        } else {
            reference_codes = Some(generation);
        }
        for (index, bits) in rounds.iter().enumerate() {
            anyhow::ensure!(bits.len() == 268800);
            if let Some(expected) = &reference {
                assert_eq!(bits, expected);
            } else {
                reference = Some(bits.clone());
            }
            let mut hash = Sha256::new();
            let mut wav = hound::WavWriter::create(
                path.join(format!("round-{index}.wav")),
                hound::WavSpec {
                    channels: 2,
                    sample_rate: 48000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )?;
            for &value in bits {
                hash.update(value.to_le_bytes());
                wav.write_sample(f32::from_bits(value))?;
            }
            wav.finalize()?;
            eprintln!(
                "native {name} round {index}: {} raw PCM values SHA256 {:x}",
                bits.len(),
                hash.finalize()
            );
        }
    }
    Ok(())
}
