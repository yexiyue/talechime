use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
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

/// Logical model resource names mapped to independently verified files.
pub(super) struct Files(BTreeMap<String, PathBuf>);
impl Files {
    const REQUIRED: &'static [&'static str] = &[
        "config.json",
        "multilingual_zh_ja_yue_char_del.tiktoken",
        "gpt.safetensors",
        "codec.safetensors",
        "s2mel.safetensors",
        "campplus.safetensors",
        "bigvgan.safetensors",
        "bigvgan.config.json",
        "feats.safetensors",
        "w2v-bert-2.0/config.json",
        "w2v-bert-2.0/model.safetensors",
        "japanese.dic",
        "tn/wetext/zh/tn/tagger.fst",
        "tn/wetext/zh/tn/verbalizer.fst",
        "tn/wetext/en/tn/tagger.fst",
        "tn/wetext/en/tn/verbalizer.fst",
        "tn/linux/zh/tagger.fst",
        "tn/linux/zh/verbalizer.fst",
        "tn/linux/en/tagger.fst",
        "tn/linux/en/verbalizer.fst",
        "tn/es/tagger.fst",
        "tn/es/verbalizer.fst",
        "qwen0.6bemo4-merge/chat_template.jinja",
        "qwen0.6bemo4-merge/config.json",
        "qwen0.6bemo4-merge/generation_config.json",
        "qwen0.6bemo4-merge/model.safetensors",
        "qwen0.6bemo4-merge/tokenizer.json",
        "qwen0.6bemo4-merge/tokenizer_config.json",
    ];
    pub(super) fn from_resolved(paths: BTreeMap<String, PathBuf>) -> anyhow::Result<Self> {
        for name in Self::REQUIRED {
            anyhow::ensure!(
                paths.contains_key(*name),
                "IndexTTS resource manifest lacks {name}"
            );
        }
        Ok(Self(paths))
    }
    #[cfg(test)]
    pub(super) fn from_directory(directory: &Path) -> anyhow::Result<Self> {
        Self::from_resolved(
            Self::REQUIRED
                .iter()
                .map(|name| ((*name).to_owned(), directory.join(name)))
                .collect(),
        )
    }
    fn load(&self, device: &candle_core::Device) -> candle_core::Result<talechime_indextts::Model> {
        let file = |name: &str| {
            self.0.get(name).map(PathBuf::as_path).ok_or_else(|| {
                candle_core::Error::Msg(format!("IndexTTS resource manifest lacks {name}"))
            })
        };
        let linux = cfg!(target_os = "linux");
        let names = if linux {
            [
                "tn/linux/zh/tagger.fst",
                "tn/linux/zh/verbalizer.fst",
                "tn/linux/en/tagger.fst",
                "tn/linux/en/verbalizer.fst",
            ]
        } else {
            [
                "tn/wetext/zh/tn/tagger.fst",
                "tn/wetext/zh/tn/verbalizer.fst",
                "tn/wetext/en/tn/tagger.fst",
                "tn/wetext/en/tn/verbalizer.fst",
            ]
        };
        let emotion = [
            file("qwen0.6bemo4-merge/chat_template.jinja")?.to_owned(),
            file("qwen0.6bemo4-merge/config.json")?.to_owned(),
            file("qwen0.6bemo4-merge/generation_config.json")?.to_owned(),
            file("qwen0.6bemo4-merge/model.safetensors")?.to_owned(),
            file("qwen0.6bemo4-merge/tokenizer.json")?.to_owned(),
            file("qwen0.6bemo4-merge/tokenizer_config.json")?.to_owned(),
        ];
        talechime_indextts::Model::load_files(
            talechime_indextts::model::ModelFiles {
                config: file("config.json")?,
                tokenizer: file("multilingual_zh_ja_yue_char_del.tiktoken")?,
                weights: talechime_indextts::weights::WeightFiles {
                    gpt: file("gpt.safetensors")?,
                    codec: file("codec.safetensors")?,
                    s2mel: file("s2mel.safetensors")?,
                    campplus: file("campplus.safetensors")?,
                    bigvgan: file("bigvgan.safetensors")?,
                    bigvgan_config: file("bigvgan.config.json")?,
                    feats: file("feats.safetensors")?,
                },
                w2v_config: file("w2v-bert-2.0/config.json")?,
                w2v_weights: file("w2v-bert-2.0/model.safetensors")?,
                text: talechime_indextts::front::TextResources {
                    normalizer: [
                        file(names[0])?.to_owned(),
                        file(names[1])?.to_owned(),
                        file(names[2])?.to_owned(),
                        file(names[3])?.to_owned(),
                    ],
                    spanish: [
                        file("tn/es/tagger.fst")?.to_owned(),
                        file("tn/es/verbalizer.fst")?.to_owned(),
                    ],
                    linux,
                },
                japanese: file("japanese.dic")?,
                emotion,
            },
            device,
        )
    }
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
    root: PathBuf,
    files: Files,
    device: Device,
    mut jobs: mpsc::Receiver<Request>,
    ready: oneshot::Sender<Result<(), BackendError>>,
) {
    let loaded = candle_device(device).and_then(|device| files.load(&device));
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
        if let Err(error) = generate(&mut model, &mut reference_cache, &root, &request)
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

#[cfg(test)]
mod resource_tests {
    use super::*;

    #[test]
    fn resolved_file_map_preserves_independent_paths_and_rejects_missing_entries() {
        let outputs: serde_json::Value =
            serde_json::from_str(include_str!("native-assets.json")).unwrap();
        let known: std::collections::BTreeSet<_> = outputs["outputs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["path"].as_str().unwrap())
            .collect();
        let mut paths: BTreeMap<_, _> = Files::REQUIRED
            .iter()
            .enumerate()
            .map(|(i, name)| {
                assert!(known.contains(name));
                (
                    (*name).to_owned(),
                    PathBuf::from(format!("/independent/blob-{i}")),
                )
            })
            .collect();
        let expected = paths["config.json"].clone();
        let files = Files::from_resolved(paths.clone()).unwrap();
        assert_eq!(files.0["config.json"], expected);
        paths.remove("japanese.dic");
        assert!(Files::from_resolved(paths).is_err());
    }

    #[test]
    fn invalid_internal_mapping_reports_error_without_panicking() {
        let files = Files(BTreeMap::new());
        let error = match files.load(&candle_core::Device::Cpu) {
            Ok(_) => panic!("missing model paths were accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("resource manifest lacks"));
    }

    #[tokio::test]
    #[ignore = "requires fixed IndexTTS model and separately staged frontend resources"]
    async fn owner_loads_files_from_independent_locations() {
        let source =
            PathBuf::from(std::env::var_os("INDEXTTS_CONVERTED_DIR").expect("fixed model"));
        let frontend = PathBuf::from(
            std::env::var_os("TALECHIME_INDEX_FRONTEND_FILES").expect("fixed frontend"),
        );
        let wetext =
            PathBuf::from(std::env::var_os("TALECHIME_WETEXT_FILES").expect("fixed WeText"));
        let mut files = Files::from_directory(&source).unwrap();
        for (name, path) in &mut files.0 {
            if let Some(relative) = name.strip_prefix("tn/wetext/") {
                *path = wetext.join(relative);
            } else if name.starts_with("tn/linux/")
                || name.starts_with("tn/es/")
                || name == "japanese.dic"
            {
                *path = frontend.join(name);
            }
        }
        let data = tempfile::tempdir().unwrap();
        let root = data.path().join("application-data");
        let (send, jobs) = mpsc::channel(1);
        let (ready, received) = oneshot::channel();
        let owner = std::thread::spawn(move || run(root, files, Device::Cpu, jobs, ready));
        received
            .await
            .expect("owner exited")
            .expect("model initialization failed");
        drop(send);
        owner.join().unwrap();
        assert!(!data.path().join("application-data").exists());
    }
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
