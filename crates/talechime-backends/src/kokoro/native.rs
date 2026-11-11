//! Pinned community tensors, verified against official Kokoro F32 values.
use crate::native_preparation::{
    PreparationControl, SafeArchive, TensorSpec, fold_weight_norm, write_tensors,
};
use std::{collections::BTreeMap, path::Path};

fn convert_weights(
    input: &Path,
    output: &Path,
    control: &PreparationControl,
) -> anyhow::Result<()> {
    let archive = SafeArchive::open(
        input,
        327115152,
        "c3e1448dbd823dc54bb950a4f0650975c2385e789e2684fea37c93185bc104a0",
        control,
    )?;
    anyhow::ensure!(
        archive.tensors().len() == 548,
        "unexpected Kokoro tensor coverage"
    );
    let mut names = BTreeMap::new();
    let mut specs = BTreeMap::new();
    // Pinned upstream AdaIN1d explicitly initializes InstanceNorm(affine=True).
    // The checkpoint omits these non-trained defaults. Their exact 70 modules
    // and dimensions are fixed by the official model, not inferred for unknown inputs.
    let affine: BTreeMap<String, usize> = serde_json::from_str(include_str!("native-affine.json"))?;
    anyhow::ensure!(affine.len() == 70, "unexpected official affine coverage");
    for (name, info) in archive.tensors() {
        if name.ends_with(".weight_v") {
            continue;
        }
        let fused = name.ends_with(".weight_g");
        let target = if fused {
            name.strip_suffix("_g")
                .ok_or_else(|| anyhow::anyhow!("invalid normalization name"))?
                .to_owned()
        } else {
            name.clone()
        };
        let shape = if fused {
            archive
                .tensors()
                .get(&name.replace("weight_g", "weight_v"))
                .ok_or_else(|| anyhow::anyhow!("missing Kokoro normalization pair"))?
                .shape
                .clone()
        } else {
            info.shape.clone()
        };
        anyhow::ensure!(
            info.dtype == candle_core::DType::F32,
            "Kokoro native preparation requires F32"
        );
        specs.insert(
            target.clone(),
            TensorSpec {
                shape,
                dtype: candle_core::DType::F32,
            },
        );
        names.insert(target, name.clone());
    }
    for (name, width) in &affine {
        for suffix in ["weight", "bias"] {
            anyhow::ensure!(
                specs
                    .insert(
                        format!("{name}.{suffix}"),
                        TensorSpec {
                            shape: vec![*width],
                            dtype: candle_core::DType::F32,
                        }
                    )
                    .is_none(),
                "unexpected checkpoint override of official AdaIN defaults"
            );
        }
    }
    anyhow::ensure!(specs.len() == 599, "incomplete official Kokoro model");
    write_tensors(output, &specs, control, |target| {
        let (base, suffix) = target
            .rsplit_once('.')
            .ok_or_else(|| anyhow::anyhow!("invalid tensor name"))?;
        if let Some(width) = affine.get(base) {
            return Ok(if suffix == "weight" {
                candle_core::Tensor::ones(
                    *width,
                    candle_core::DType::F32,
                    &candle_core::Device::Cpu,
                )?
            } else {
                candle_core::Tensor::zeros(
                    *width,
                    candle_core::DType::F32,
                    &candle_core::Device::Cpu,
                )?
            });
        }
        let name = &names[target];
        if name.ends_with(".weight_g") {
            fold_weight_norm(
                &archive.get(&name.replace("weight_g", "weight_v"), control)?,
                &archive.get(name, control)?,
                control,
            )
        } else {
            archive.get(name, control)
        }
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "explicit fixed community Kokoro input; writes one new conversion"]
    fn convert_fixed_kokoro_weights() {
        let output =
            std::env::var_os("TALECHIME_KOKORO_NATIVE_OUTPUT").expect("set new output path");
        let output = std::path::PathBuf::from(output);
        assert!(output.is_absolute() && !output.exists());
        let input = std::env::var_os("TALECHIME_KOKORO_COMMUNITY_INPUT")
            .expect("set fixed community weight path");
        let input = Path::new(&input).canonicalize().unwrap();
        convert_weights(&input, &output, &PreparationControl::default()).unwrap();
    }
}

#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "'de: 'static"))]
struct Assets {
    inputs: Vec<crate::native_preparation::Input>,
    outputs: Vec<crate::native_preparation::Output>,
}
fn assets(contextual: bool) -> &'static Assets {
    static BASE: std::sync::OnceLock<Assets> = std::sync::OnceLock::new();
    static CONTEXT: std::sync::OnceLock<Assets> = std::sync::OnceLock::new();
    let (slot, data) = if contextual {
        (&CONTEXT, include_str!("native-context-assets.json"))
    } else {
        (&BASE, include_str!("native-assets.json"))
    };
    slot.get_or_init(|| {
        serde_json::from_str(data).expect("compiled Kokoro asset metadata must be valid")
    })
}
pub(super) fn manifest(contextual: bool) -> &'static [u8] {
    if contextual {
        include_bytes!("native-context-manifest.json")
    } else {
        include_bytes!("native-manifest.json")
    }
}
pub(super) fn publication_marker(contextual: bool) -> anyhow::Result<Vec<u8>> {
    let model = if contextual {
        "1.1-zh-g2pw"
    } else {
        super::MODEL
    };
    let recipe = recipe(model).ok_or_else(|| anyhow::anyhow!("Kokoro variant not compiled"))?;
    Ok(serde_json::to_vec(&(
        1u32,
        recipe.id,
        recipe.inputs,
        recipe.outputs,
    ))?)
}
pub(crate) fn recipe(model: &str) -> Option<crate::native_preparation::Recipe> {
    static ORDINARY: std::sync::OnceLock<Vec<crate::native_preparation::Input>> =
        std::sync::OnceLock::new();
    static CONTEXT: std::sync::OnceLock<Vec<crate::native_preparation::Input>> =
        std::sync::OnceLock::new();
    let (inputs, id) = if model == super::MODEL {
        (
            ORDINARY.get_or_init(|| {
                serde_json::from_str(include_str!("distribution-inputs.json"))
                    .expect("compiled distribution pins")
            }),
            "kokoro-1.1-zh-runtime-f32-v3",
        )
    } else {
        (
            CONTEXT.get_or_init(|| {
                serde_json::from_str(include_str!("distribution-context-inputs.json"))
                    .expect("compiled distribution pins")
            }),
            "kokoro-1.1-zh-g2pw-runtime-f32-v3",
        )
    };
    Some(source_recipe(model)?.distributed(id, inputs, super::directory_for(Path::new(""), model)))
}

fn source_recipe(model: &str) -> Option<crate::native_preparation::Recipe> {
    let (contextual, id, transform): (bool, &'static str, crate::native_preparation::Transform) =
        match model {
            super::MODEL => (false, "kokoro-1.1-zh-native-f32-v2", prepare_mandarin),
            #[cfg(feature = "kokoro-g2pw")]
            super::CONTEXT_MODEL => (true, "kokoro-1.1-zh-g2pw-native-f32-v2", prepare_context),
            _ => return None,
        };
    let catalog = assets(contextual);
    Some(crate::native_preparation::Recipe {
        id,
        target: format!("kokoro/models/{model}/01e7505-native-f32-v2").into(),
        inputs: &catalog.inputs,
        outputs: &catalog.outputs,
        transform_outputs: if contextual {
            &["model.safetensors", "context/g2pw.onnx"]
        } else {
            &["model.safetensors"]
        },
        workspace_bytes: 256 * 1024 * 1024,
        transform,
    })
}
fn prepare_mandarin(work: &crate::native_preparation::Workspace<'_>) -> anyhow::Result<()> {
    prepare(work, false)
}
#[cfg(feature = "kokoro-g2pw")]
fn prepare_context(work: &crate::native_preparation::Workspace<'_>) -> anyhow::Result<()> {
    prepare(work, true)
}
fn prepare(
    work: &crate::native_preparation::Workspace<'_>,
    contextual: bool,
) -> anyhow::Result<()> {
    work.component("model.safetensors", |input, output, control| {
        convert_weights(&input.join("model-input.safetensors"), output, control)
    })?;
    for file in assets(contextual).inputs.iter().skip(1) {
        if file.path != "context/archive.zip" {
            work.copy(file.path)?;
        }
    }
    if contextual {
        work.component("context/g2pw.onnx", |input, output, control| {
            use std::io::{Read, Write};
            let mut archive =
                zip::ZipArchive::new(std::fs::File::open(input.join("context/archive.zip"))?)?;
            let mut source = archive.by_name("G2PWModel/g2pw.onnx")?;
            let pin = assets(true)
                .outputs
                .iter()
                .find(|pin| pin.path == "context/g2pw.onnx")
                .ok_or_else(|| anyhow::anyhow!("missing g2pW pin"))?;
            anyhow::ensure!(
                source.size() == pin.size,
                "unexpected g2pW archive member size"
            );
            let mut file = std::fs::File::create_new(output)?;
            let mut buffer = [0u8; 64 * 1024];
            let mut copied = 0u64;
            loop {
                control.check()?;
                let n = source.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                copied += n as u64;
                anyhow::ensure!(copied <= pin.size, "g2pW extraction exceeds pinned size");
                file.write_all(&buffer[..n])?;
            }
            anyhow::ensure!(copied == pin.size, "incomplete g2pW archive member");
            file.sync_all()?;
            Ok(())
        })?;
    }
    for (name, bytes) in [
        (
            "config.json",
            include_bytes!("../../../kokoro/assets/config.json").as_slice(),
        ),
        (
            "KOKORO-LICENSE",
            include_bytes!("../../../kokoro/LICENSE").as_slice(),
        ),
        (
            "official-export.json",
            include_bytes!("native-export.json").as_slice(),
        ),
        ("manifest.json", manifest(contextual)),
    ] {
        work.write(name, bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod recipe_tests {
    use super::*;
    #[test]
    fn recipe_covers_all_official_mandarin_voices_and_public_frontend() {
        let official: Vec<serde_json::Value> =
            serde_json::from_str(talechime_kokoro::BUILTIN_VOICES_JSON).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("native-manifest.json")).unwrap();
        let voices = manifest["voices"].as_object().unwrap();
        assert_eq!(voices.len(), 100);
        for voice in official
            .iter()
            .filter(|v| v["id"].as_str().unwrap().starts_with('z'))
        {
            let id = voice["id"].as_str().unwrap();
            assert_eq!(voices[id]["source_sha256"], voice["sha256"]);
        }
        let data = assets(false);
        for input in &data.inputs {
            assert_eq!(input.revision.len(), 40);
            assert_eq!(input.sha256.len(), 64);
            if input.path.starts_with("frontend/") {
                assert_eq!(input.repo, "yexiyue/talechime-resources");
                assert_eq!(input.revision, "8eb001f5498e099e6ac929454e7fb28e457bf035");
            }
        }
        assert_eq!(recipe(super::super::MODEL).unwrap().outputs.len(), 114);
    }
    #[tokio::test]
    #[ignore = "explicit verified isolated Kokoro inputs; no new weight downloads expected"]
    async fn prepare_kokoro_from_verified_inputs() {
        let root = std::path::PathBuf::from(
            std::env::var_os("TALECHIME_NATIVE_ENGINE_ROOT").expect("set isolated engine root"),
        );
        let downloads =
            crate::downloads::Downloads::new(&crate::downloads::DownloadConfig::default()).unwrap();
        let model =
            std::env::var("TALECHIME_KOKORO_MODEL").unwrap_or_else(|_| super::super::MODEL.into());
        let source_build = std::env::var_os("TALECHIME_BUILD_NATIVE_SOURCE").is_some();
        let recipe = if source_build {
            source_recipe(&model)
        } else {
            recipe(&model)
        }
        .unwrap();
        let plan = crate::native_preparation::plan_recipe(
            recipe.clone(),
            root.clone(),
            downloads.clone(),
            PreparationControl::default(),
        )
        .await
        .unwrap();
        assert_eq!(plan.download_bytes, 0);
        let (stages, _) = tokio::sync::mpsc::channel(32);
        let (files, _) = tokio::sync::mpsc::channel(32);
        recipe
            .clone()
            .execute(
                root.clone(),
                downloads.clone(),
                plan.fingerprint,
                PreparationControl::default(),
                stages,
                files,
            )
            .await
            .unwrap();
        let plan = crate::native_preparation::plan_recipe(
            recipe.clone(),
            root.clone(),
            downloads,
            PreparationControl::default(),
        )
        .await
        .unwrap();
        assert!(plan.ready);
        assert_eq!(plan.peak_additional_bytes, 0);
        assert_eq!(recipe.status(&root), crate::ResourceStatus::Ready);
        if source_build {
            recipe
                .verify_files(&root.join(&recipe.target), &PreparationControl::default())
                .unwrap();
        } else {
            let manifest =
                super::super::resources::read_for(&root.join(&recipe.target), &model).unwrap();
            assert_eq!(manifest.voices.len(), 100);
        }
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    #[test]
    fn manifest_and_publication_identity_cannot_be_replaced_or_mixed() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("manifest.json"), manifest(false)).unwrap();
        assert!(super::super::resources::read_for(root.path(), super::super::MODEL).is_err());
        std::fs::write(
            root.path().join("preparation.json"),
            publication_marker(false).unwrap(),
        )
        .unwrap();
        assert!(super::super::resources::read_for(root.path(), super::super::MODEL).is_ok());
        assert!(super::super::resources::read_for(root.path(), "1.1-zh-g2pw").is_err());
        let bytes = String::from_utf8(manifest(false).to_vec())
            .unwrap()
            .replace("model.safetensors", "other.safetensors");
        std::fs::write(root.path().join("manifest.json"), bytes).unwrap();
        assert!(super::super::resources::read_for(root.path(), super::super::MODEL).is_err());
    }
    #[tokio::test]
    #[ignore = "explicit published Kokoro resources, device and new WAV output path"]
    async fn load_native_kokoro_and_synthesize() {
        use talechime_core::{
            backend::{Backend, SegmentRequest},
            params::GenerationParams,
        };
        use talechime_protocol::Device;
        let root = std::path::PathBuf::from(
            std::env::var_os("TALECHIME_NATIVE_ENGINE_ROOT").expect("set isolated engine root"),
        );
        let model =
            std::env::var("TALECHIME_KOKORO_MODEL").unwrap_or_else(|_| super::super::MODEL.into());
        let device = match std::env::var("TALECHIME_NATIVE_DEVICE")
            .as_deref()
            .unwrap_or("cpu")
        {
            "cpu" => Device::Cpu,
            "metal" => Device::Metal,
            _ => panic!("unsupported verification device"),
        };
        let output = std::path::PathBuf::from(
            std::env::var_os("TALECHIME_NATIVE_AUDIO_OUTPUT").expect("set new absolute WAV path"),
        );
        assert!(output.is_absolute());
        let started = std::time::Instant::now();
        let backend = super::super::KokoroBackend::load_model_on(
            super::super::directory_for(&root, &model),
            &model,
            device,
        )
        .await
        .unwrap();
        let loaded = started.elapsed();
        assert_eq!(backend.capabilities().voices.len(), 100);
        let params = GenerationParams::new();
        let text = std::env::var("TALECHIME_NATIVE_AUDIO_TEXT")
            .unwrap_or_else(|_| "春天来了，院子里的花开了。".into());
        let pcm = backend
            .synthesize(SegmentRequest {
                text: &text,
                voice: "zf_001",
                style: None,
                seed: 7,
                params: &params,
            })
            .await
            .unwrap();
        pcm.duration_ms().unwrap();
        assert_eq!((pcm.sample_rate, pcm.channels), (24000, 1));
        let rms = (pcm
            .samples
            .iter()
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>()
            / pcm.samples.len() as f64)
            .sqrt();
        assert!(rms > 0.001, "silent output");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 24000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut wav =
            hound::WavWriter::new(std::fs::File::create_new(&output).unwrap(), spec).unwrap();
        for sample in &pcm.samples {
            wav.write_sample(*sample).unwrap();
        }
        wav.finalize().unwrap();
        let wav = hound::WavReader::open(&output).unwrap();
        assert_eq!(wav.duration() as usize, pcm.samples.len());
        println!(
            "model={model} device={device:?} loaded={loaded:?} total={:?} samples={} rms={rms:.6}; listening not evaluated",
            started.elapsed(),
            pcm.samples.len()
        );
    }
}
