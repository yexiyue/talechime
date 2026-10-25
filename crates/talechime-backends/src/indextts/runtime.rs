use std::path::{Path, PathBuf};
use talechime_core::backend::{AudioChunk, BackendError, Pcm};
use talechime_protocol::Device;
use tokio::sync::{mpsc, oneshot};

pub(super) struct Request {
    #[cfg(feature = "performance-tools")]
    pub(crate) perf: crate::performance::OwnerContext,
    pub text: String,
    pub voice: String,
    pub resolved: super::params::Resolved,
    pub audio: mpsc::Sender<Result<AudioChunk, BackendError>>,
}

pub(super) fn candle_device(device: Device) -> candle_core::Result<candle_core::Device> {
    match device {
        Device::Cpu => Ok(candle_core::Device::Cpu),
        #[cfg(all(feature = "indextts-metal", target_os = "macos"))]
        Device::Metal => candle_core::Device::new_metal(0),
        #[cfg(all(
            feature = "indextts-cuda-cudnn",
            any(target_os = "linux", target_os = "windows")
        ))]
        Device::Cuda => candle_core::Device::new_cuda(0),
        _ => candle_core::bail!("unsupported IndexTTS device {device:?}"),
    }
}

pub(super) fn run(
    directory: PathBuf,
    device: Device,
    mut jobs: mpsc::Receiver<Request>,
    ready: oneshot::Sender<Result<(), BackendError>>,
) {
    let loaded = candle_device(device)
        .and_then(|device| talechime_indextts::Model::load(&directory, &device));
    let mut model = match loaded {
        Ok(model) => model,
        Err(error) => {
            let _ = ready.send(Err(BackendError::Initialize(error.to_string())));
            return;
        }
    };
    if ready.send(Ok(())).is_err() {
        return;
    }
    // Scope reuse to the platform/device being verified. Other paths
    // retain unconditional preparation until separately measured and accepted.
    let mut reference_cache = super::reference_cache::ReferenceCache::new(
        cfg!(target_os = "windows") && device == Device::Cuda,
    );
    while let Some(request) = jobs.blocking_recv() {
        #[cfg(feature = "performance-tools")]
        let _owner_context = request.perf.enter();
        if request.audio.is_closed() {
            continue;
        }
        if let Err(error) = generate(&mut model, &mut reference_cache, &directory, &request)
            && !request.audio.is_closed()
        {
            let _ = crate::observation::send_audio(&request.audio, Err(error));
        }
    }
}

fn load_reference(
    store: &talechime_core::voices::VoiceStore,
    voice: &str,
) -> Result<talechime_indextts::Reference, BackendError> {
    store
        .load(voice)
        .map_err(|e| BackendError::Unsupported(e.to_string()))?;
    let path = store
        .path(voice)
        .map_err(|e| BackendError::Unsupported(e.to_string()))?;
    let samples =
        crate::reference::load(&path.join("reference.wav"), talechime_indextts::SAMPLE_RATE)
            .map_err(|e| BackendError::Synthesis(e.to_string()))?;
    talechime_indextts::Reference::from_pcm(&samples, talechime_indextts::SAMPLE_RATE)
        .map_err(|e| BackendError::Synthesis(e.to_string()))
}

fn generate(
    model: &mut talechime_indextts::Model,
    reference_cache: &mut super::reference_cache::ReferenceCache,
    directory: &Path,
    request: &Request,
) -> Result<(), BackendError> {
    model.clear_emo_reference();
    let store = super::voice_store(directory)
        .map_err(|error| BackendError::Synthesis(error.to_string()))?;
    store
        .load(&request.voice)
        .map_err(|error| BackendError::Unsupported(error.to_string()))?;
    let wav = store
        .path(&request.voice)
        .map_err(|error| BackendError::Unsupported(error.to_string()))?
        .join("reference.wav");
    let samples = crate::reference::load(&wav, talechime_indextts::SAMPLE_RATE)
        .map_err(|error| BackendError::Synthesis(error.to_string()))?;
    if request.audio.is_closed() {
        return Ok(());
    }
    reference_cache.prepare(&samples, || {
        let reference =
            talechime_indextts::Reference::from_pcm(&samples, talechime_indextts::SAMPLE_RATE)
                .map_err(|error| BackendError::Synthesis(error.to_string()))?;
        model
            .prepare_reference(&reference)
            .map_err(|error| BackendError::Synthesis(error.to_string()))
    })?;
    if request.audio.is_closed() {
        return Ok(());
    }
    if let Some(voice) = &request.resolved.emotion_voice {
        let emotion = load_reference(&store, voice)?;
        if request.audio.is_closed() {
            return Ok(());
        }
        model
            .prepare_emo_reference(&emotion)
            .map_err(|e| BackendError::Synthesis(e.to_string()))?;
    }
    let outcome = model
        .generate_with_cancel(
            &request.text,
            &request.resolved.language,
            &request.resolved.options,
            &|| request.audio.is_closed(),
        )
        .map_err(|error| BackendError::Synthesis(error.to_string()))?;
    if request.audio.is_closed() {
        return Ok(());
    }
    let pcm = Pcm {
        samples: outcome
            .samples
            .into_iter()
            .map(|sample| sample / talechime_indextts::PCM16_MAX)
            .collect(),
        sample_rate: outcome.sample_rate,
        channels: 1,
    };
    pcm.duration_ms()?;
    crate::observation::send_audio(&request.audio, Ok(AudioChunk::Pcm(pcm)))
        .map_err(|_| BackendError::Synthesis("IndexTTS receiver closed".into()))?;
    let _ = crate::observation::send_audio(&request.audio, Ok(AudioChunk::End));
    Ok(())
}

#[cfg(all(test, target_os = "windows", feature = "indextts-cuda-cudnn"))]
mod tests {
    use super::super::reference_cache::ReferenceCache;
    use super::*;

    #[test]
    #[ignore = "requires converted IndexTTS CUDA weights and reference WAV; resets only test-owned RNG"]
    fn reference_cache_matches_uncached_pcm() -> anyhow::Result<()> {
        let source = PathBuf::from(
            std::env::var_os("INDEXTTS_REFERENCE_WAV").expect("INDEXTTS_REFERENCE_WAV"),
        );
        let weights = PathBuf::from(
            std::env::var_os("INDEXTTS_CONVERTED_DIR").expect("INDEXTTS_CONVERTED_DIR"),
        );
        let directory = tempfile::tempdir()?;
        let store = super::super::voice_store(directory.path())?;
        store.import(
            "custom:check",
            "reference check",
            &source,
            "test fixture transcript; not used by model",
            None,
        )?;
        let wav = store.path("custom:check")?.join("reference.wav");
        let original = std::fs::read(&wav)?;
        let device = candle_device(Device::Cuda)?;
        let mut model = talechime_indextts::Model::load(&weights, &device)?;
        let synth_with = |model: &mut talechime_indextts::Model,
                          cache: &mut ReferenceCache,
                          label: &str,
                          params: &talechime_core::params::GenerationParams,
                          style: Option<&str>|
         -> anyhow::Result<Vec<u32>> {
            // Independent test control, never a production seed-policy change.
            device.set_seed(20261024)?;
            let (tx, mut rx) = mpsc::channel(2);
            let request = Request {
                #[cfg(feature = "performance-tools")]
                perf: crate::performance::OwnerContext::capture(),
                text: "你好，欢迎收听今天的故事。".into(),
                voice: "custom:check".into(),
                resolved: super::super::params::resolve(params, style, 7330414270409262323)?,
                audio: tx,
            };
            let started = std::time::Instant::now();
            generate(model, cache, directory.path(), &request)?;
            let Some(Ok(AudioChunk::Pcm(pcm))) = rx.try_recv().ok() else {
                anyhow::bail!("missing PCM")
            };
            assert!(matches!(rx.try_recv(), Ok(Ok(AudioChunk::End))));
            println!(
                "CACHE_PCM {label} samples={} seconds={}",
                pcm.samples.len(),
                started.elapsed().as_secs_f64()
            );
            Ok(pcm.samples.iter().map(|v| v.to_bits()).collect())
        };
        let synth =
            |model: &mut talechime_indextts::Model, cache: &mut ReferenceCache, label: &str| {
                synth_with(
                    model,
                    cache,
                    label,
                    &talechime_core::params::GenerationParams::new(),
                    None,
                )
            };
        let mut cache = ReferenceCache::new(true);
        let check = |expected: &[u32], actual: Vec<u32>| {
            assert_eq!(expected.len(), actual.len(), "PCM length changed");
            let differences = expected.iter().zip(&actual).filter(|(a, b)| a != b).count();
            assert_eq!(differences, 0, "PCM bits changed");
        };
        let expected = synth(&mut model, &mut ReferenceCache::new(false), "uncached-a")?;
        check(&expected, synth(&mut model, &mut cache, "first-a")?);
        check(&expected, synth(&mut model, &mut cache, "hit-a")?);
        // Same pathname and PCM duration, changed content invalidates the identity.
        let mut samples = crate::reference::load(&wav, talechime_indextts::SAMPLE_RATE)?;
        samples.reverse();
        let mut writer = hound::WavWriter::create(
            &wav,
            hound::WavSpec {
                channels: 1,
                sample_rate: talechime_indextts::SAMPLE_RATE,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            },
        )?;
        for sample in samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        let actual_b = synth(&mut model, &mut cache, "replace-b")?;
        check(
            &actual_b,
            synth(&mut model, &mut ReferenceCache::new(false), "uncached-b")?,
        );
        // Only test-owned audio: retain the changed snapshot as an independent
        // emotion reference before restoring the speaker fixture.
        store.import(
            "custom:emotion",
            "emotion check",
            &wav,
            "test fixture",
            None,
        )?;
        std::fs::write(&wav, &original)?;
        check(&expected, synth(&mut model, &mut cache, "return-a")?);
        check(&expected, synth(&mut model, &mut cache, "hit-return-a")?);
        let request = |audio| Request {
            #[cfg(feature = "performance-tools")]
            perf: crate::performance::OwnerContext::capture(),
            text: "你好，欢迎收听今天的故事。".into(),
            voice: "custom:check".into(),
            resolved: super::super::params::resolve(
                &talechime_core::params::GenerationParams::new(),
                None,
                7330414270409262323,
            )
            .expect("valid default test request"),
            audio,
        };
        let (tx, rx) = mpsc::channel(2);
        drop(rx);
        generate(&mut model, &mut cache, directory.path(), &request(tx))?;
        check(&expected, synth(&mut model, &mut cache, "after-preclosed")?);
        let (tx, mut rx) = mpsc::channel(2);
        std::fs::write(&wav, b"invalid WAV")?;
        assert!(generate(&mut model, &mut cache, directory.path(), &request(tx)).is_err());
        assert!(rx.try_recv().is_err());
        std::fs::write(&wav, &original)?;
        check(
            &expected,
            synth(&mut model, &mut cache, "after-invalid-wav")?,
        );
        let (tx, rx) = mpsc::channel(2);
        let closer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            drop(rx);
        });
        let outcome = generate(&mut model, &mut cache, directory.path(), &request(tx));
        closer.join().unwrap();
        assert!(
            matches!(outcome, Err(BackendError::Synthesis(ref message)) if message.contains("cancelled"))
        );
        check(
            &expected,
            synth(&mut model, &mut cache, "after-generation-cancel")?,
        );
        use super::super::params::IndexTtsParams;
        for (label, params, style) in [
            ("named-happy", IndexTtsParams::default(), Some("happy")),
            (
                "mixed-vector",
                IndexTtsParams::default()
                    .emotion_vector([0.5, 0., 0., 0., 0., 0., 0., 0.3])
                    .emotion_strength(0.6),
                None,
            ),
            (
                "seeded-random-vector",
                IndexTtsParams::default()
                    .emotion_vector([0.5, 0., 0., 0., 0., 0., 0., 0.3])
                    .emotion_random(true),
                None,
            ),
            (
                "independent-reference",
                IndexTtsParams::default()
                    .emotion_voice("custom:emotion")
                    .emotion_strength(0.6),
                None,
            ),
        ] {
            let params = params.to_generation_params();
            let control = synth_with(
                &mut model,
                &mut ReferenceCache::new(false),
                label,
                &params,
                style,
            )?;
            check(
                &control,
                synth_with(&mut model, &mut cache, label, &params, style)?,
            );
            check(
                &control,
                synth_with(&mut model, &mut cache, label, &params, style)?,
            );
            check(&expected, synth(&mut model, &mut cache, "after-emotion")?);
        }
        Ok(())
    }
}
