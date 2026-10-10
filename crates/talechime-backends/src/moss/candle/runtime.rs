use super::{Mode, resources, voice_store};
use candle_core::{DType, Device};
use moss_tts::{Generation, Model, codec::AudioCodec};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::sync::{mpsc, oneshot};
use tts_core::backend::{AudioChunk, BackendError, Pcm};
pub(super) struct Request {
    pub context: Option<tts_core::SpeechContext>,
    pub text: String,
    pub voice: String,
    pub seed: u64,
    pub resolved: super::params::Resolved,
    pub audio: mpsc::Sender<Result<AudioChunk, BackendError>>,
}
pub(super) fn device(selected: tts_protocol::Device) -> anyhow::Result<Device> {
    match selected {
        #[cfg(all(
            feature = "moss-candle-cuda",
            any(target_os = "windows", target_os = "linux")
        ))]
        tts_protocol::Device::Cuda => Ok(Device::new_cuda(0)?),
        #[cfg(all(feature = "moss-candle-metal", target_os = "macos"))]
        tts_protocol::Device::Metal => Ok(Device::new_metal(0)?),
        _ => {
            anyhow::bail!("MOSS Candle production adapter currently requires a compiled GPU device")
        }
    }
}
pub(super) fn model_dtype(device: &Device) -> DType {
    if device.is_cuda() {
        DType::BF16
    } else {
        DType::F16
    }
}
pub(super) fn run(
    root: PathBuf,
    mode: Mode,
    selected: tts_protocol::Device,
    mut jobs: mpsc::Receiver<Request>,
    ready: oneshot::Sender<Result<(), BackendError>>,
) {
    let load = || -> anyhow::Result<_> {
        let dev = device(selected)?;
        let model = Model::load(mode.id(), &mode.directory(&root), &dev, model_dtype(&dev))?;
        // Finish Metal weight uploads/casts before loading the next component.
        if dev.is_metal() {
            dev.synchronize()?;
        }
        anyhow::ensure!(!ready.is_closed(), "initialization cancelled");
        let codec = AudioCodec::load(&resources::codec_directory(&root), &dev, DType::F16)?;
        if dev.is_metal() {
            dev.synchronize()?;
        }
        Ok((model, codec))
    };
    let (mut model, mut codec) = match load() {
        Ok(value) => value,
        Err(e) => {
            let _ = ready.send(Err(BackendError::Initialize(e.to_string())));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    while let Some(request) = jobs.blocking_recv() {
        if request.audio.is_closed() {
            continue;
        }
        let result = generate(&mut model, &mut codec, &root, mode, &request);
        if let Err(error) = result
            && !request.audio.is_closed()
        {
            let _ = request
                .audio
                .blocking_send(Err(BackendError::Synthesis(error.to_string())));
        }
    }
}
fn generate(
    model: &mut Model,
    codec: &mut AudioCodec,
    root: &Path,
    mode: Mode,
    request: &Request,
) -> anyhow::Result<()> {
    let continuation = request
        .context
        .as_ref()
        .map(|context| {
            let samples = crate::reference::mono(context.pcm(), 24000)?;
            codec.encode(
                &samples,
                match mode {
                    Mode::Local => 32,
                    Mode::Realtime => 16,
                },
                &|| request.audio.is_closed(),
            )
        })
        .transpose()?;
    let reference = if continuation.is_none() && request.voice.starts_with("custom:") {
        Some(prompt(
            codec,
            &mode.directory(root),
            mode,
            &request.voice,
            &|| request.audio.is_closed(),
        )?)
    } else {
        None
    };
    codec.reset_decoder();
    let mut generation = Generation::new(&request.text);
    generation.seed = request.seed;
    generation.max_frames = request.resolved.max_frames;
    generation.instruction = request.resolved.instruction.as_deref();
    generation.language = request.resolved.language.as_deref();
    generation.reference = reference.as_deref().or_else(|| {
        if mode == Mode::Realtime {
            continuation.as_deref()
        } else {
            None
        }
    });
    generation.continuation =
        request
            .context
            .as_ref()
            .zip(continuation.as_deref())
            .map(|(context, codes)| moss_tts::Continuation {
                text: context.text(),
                codes,
            });
    if mode == Mode::Local
        && let Some(codes) = &continuation
    {
        // Prime the causal decoder, but never deliver the already-played prefix.
        for frames in codes.chunks(5) {
            codec.decode(frames, &|| request.audio.is_closed())?;
        }
    }
    let mut pending = Vec::new();
    let mut delivered = false;
    model.generate(&generation, &|| request.audio.is_closed(), |frame| {
        pending.push(frame.to_vec());
        if pending.len() == 5 {
            deliver(codec, &pending, &request.audio)?;
            pending.clear();
            delivered = true;
        }
        Ok(!request.audio.is_closed())
    })?;
    if !pending.is_empty() {
        deliver(codec, &pending, &request.audio)?;
        delivered = true;
    }
    anyhow::ensure!(
        delivered && !request.audio.is_closed(),
        "empty or cancelled generation"
    );
    request
        .audio
        .blocking_send(Ok(AudioChunk::End))
        .map_err(|_| anyhow::anyhow!("generation cancelled"))?;
    Ok(())
}

fn deliver(
    codec: &mut AudioCodec,
    frames: &[Vec<u32>],
    audio: &mpsc::Sender<Result<AudioChunk, BackendError>>,
) -> anyhow::Result<()> {
    let pcm = Pcm {
        samples: codec.decode(frames, &|| audio.is_closed())?,
        sample_rate: codec.sample_rate,
        channels: 1,
    };
    pcm.duration_ms()?;
    anyhow::ensure!(!audio.is_closed(), "generation cancelled");
    audio
        .blocking_send(Ok(AudioChunk::Pcm(pcm)))
        .map_err(|_| anyhow::anyhow!("generation cancelled"))
}
#[derive(serde::Serialize, serde::Deserialize)]
struct CachedCodes {
    codec_revision: String,
    wav_sha256: String,
    codebooks: usize,
    frames: Vec<Vec<u32>>,
}
fn prompt(
    codec: &mut AudioCodec,
    directory: &Path,
    mode: Mode,
    voice: &str,
    cancelled: &impl Fn() -> bool,
) -> anyhow::Result<Vec<Vec<u32>>> {
    let store = voice_store(directory, mode)?;
    store.load(voice)?;
    let path = store.path(voice)?;
    let wav = path.join("reference.wav");
    let digest = format!("{:x}", Sha256::digest(std::fs::read(&wav)?));
    let codebooks = match mode {
        Mode::Local => 32,
        Mode::Realtime => 16,
    };
    let cache = path.join("moss-prompt.json");
    if cache.exists() {
        let cached: CachedCodes = serde_json::from_reader(std::fs::File::open(&cache)?)?;
        anyhow::ensure!(
            cached.codec_revision == resources::CODEC_REVISION
                && cached.wav_sha256 == digest
                && cached.codebooks == codebooks
                && (1..=125).contains(&cached.frames.len())
                && cached
                    .frames
                    .iter()
                    .all(|f| f.len() == codebooks && f.iter().all(|id| *id < 1024)),
            "invalid MOSS reference cache; re-import {voice}"
        );
        return Ok(cached.frames);
    }
    let samples = crate::reference::load(&wav, 24000)?;
    anyhow::ensure!(
        samples.len() <= 240000,
        "MOSS trial reference must be at most 10 seconds"
    );
    let frames = codec.encode(&samples, codebooks, cancelled)?;
    anyhow::ensure!(!cancelled(), "reference encoding cancelled");
    let temporary = tempfile::NamedTempFile::new_in(&path)?;
    serde_json::to_writer(
        temporary.as_file(),
        &CachedCodes {
            codec_revision: resources::CODEC_REVISION.into(),
            wav_sha256: digest,
            codebooks,
            frames: frames.clone(),
        },
    )?;
    temporary.persist(cache)?;
    Ok(frames)
}
