//! Fixed official F32 checkpoint adaptation; no MLX or external script runtime.
use crate::native_preparation::{Archive, PreparationControl, TensorSpec, write_tensors};
use candle_core::{DType, Device, Tensor};
use std::{collections::BTreeMap, path::Path};

#[derive(serde::Deserialize)]
#[serde(bound(deserialize = "'de: 'static"))]
struct Assets {
    inputs: Vec<crate::native_preparation::Input>,
    outputs: Vec<crate::native_preparation::Output>,
}
fn assets() -> &'static Assets {
    static ASSETS: std::sync::OnceLock<Assets> = std::sync::OnceLock::new();
    ASSETS.get_or_init(|| {
        serde_json::from_str(include_str!("native-assets.json"))
            .expect("compiled IndexTTS assets must be valid")
    })
}
fn open_source(
    input: &Path,
    name: &str,
    selection: &[&str],
    control: &PreparationControl,
) -> anyhow::Result<Archive> {
    let pin = assets()
        .inputs
        .iter()
        .find(|pin| pin.path == name)
        .ok_or_else(|| anyhow::anyhow!("unknown IndexTTS checkpoint input"))?;
    Archive::open(
        &input.join(pin.path),
        pin.size,
        pin.sha256,
        selection,
        control,
    )
}
fn convert(
    cache: &Path,
    name: &str,
    output: &Path,
    control: &PreparationControl,
) -> anyhow::Result<()> {
    let (file, selection, prefix, count): (&str, &[&str], &str, usize) = match name {
        "gpt" => ("gpt.pth", &[], "", 456),
        "codec" => ("codec.pth", &["model"], "", 243),
        "s2mel" => ("s2mel.pth", &["net"], "models.", 284),
        "campplus" => ("campplus_cn_common.bin", &[], "", 937),
        "bigvgan" => ("bigvgan_generator.pt", &["generator"], "", 783),
        _ => anyhow::bail!("unknown fixed IndexTTS component"),
    };
    let archive = open_source(cache, file, selection, control)?;
    anyhow::ensure!(
        archive.tensors().len() == count,
        "unexpected IndexTTS tensor coverage"
    );
    let mut names = BTreeMap::new();
    let specs: BTreeMap<_, _> = archive
        .tensors()
        .iter()
        .map(|(name, info)| {
            let target = format!("{prefix}{name}");
            names.insert(target.clone(), name);
            (
                target,
                TensorSpec {
                    shape: info.layout.dims().to_vec(),
                    dtype: DType::F32,
                },
            )
        })
        .collect();
    write_tensors(output, &specs, control, |name| {
        Ok(archive.get(names[name], control)?.to_dtype(DType::F32)?)
    })
}
fn convert_features(
    cache: &Path,
    output: &Path,
    control: &PreparationControl,
) -> anyhow::Result<()> {
    let mut values = BTreeMap::new();
    for (name, source, selection) in [
        ("spk_matrix", "feat1.pt", None),
        ("emo_matrix", "feat2.pt", None),
        ("w2v_mean", "wav2vec2bert_stats.pt", Some("mean")),
        ("w2v_std", "wav2vec2bert_stats.pt", Some("var")),
    ] {
        let selected: Vec<_> = selection.into_iter().collect();
        let archive = open_source(cache, source, &selected, control)?;
        anyhow::ensure!(
            archive.tensors().len() == 1,
            "unexpected IndexTTS feature input"
        );
        let mut value = archive.get("value", control)?.to_dtype(DType::F32)?;
        if name == "w2v_std" {
            let variance = value.to_vec1::<f32>()?;
            anyhow::ensure!(
                variance
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0),
                "invalid IndexTTS variance"
            );
            value = Tensor::from_vec(
                variance.into_iter().map(f32::sqrt).collect::<Vec<_>>(),
                value.dims(),
                &Device::Cpu,
            )?;
        }
        let expected: &[usize] = match name {
            "spk_matrix" => &[73, 192],
            "emo_matrix" => &[73, 1280],
            _ => &[1024],
        };
        anyhow::ensure!(
            value.dims() == expected,
            "unexpected IndexTTS feature shape"
        );
        values.insert(name.to_owned(), value);
    }
    let specs = values
        .iter()
        .map(|(name, value)| {
            (
                name.clone(),
                TensorSpec {
                    shape: value.dims().to_vec(),
                    dtype: DType::F32,
                },
            )
        })
        .collect();
    write_tensors(output, &specs, control, |name| Ok(values[name].clone()))
}
pub(crate) fn recipe() -> crate::native_preparation::Recipe {
    static INPUTS: std::sync::OnceLock<Vec<crate::native_preparation::Input>> =
        std::sync::OnceLock::new();
    let inputs = INPUTS.get_or_init(|| {
        serde_json::from_str(include_str!("distribution-inputs.json"))
            .expect("compiled distribution pins")
    });
    source_recipe().distributed(
        "indextts-2.5-runtime-f32-v3",
        inputs,
        super::directory(Path::new("")),
    )
}

fn source_recipe() -> crate::native_preparation::Recipe {
    let assets = assets();
    crate::native_preparation::Recipe {
        id: "indextts-2.5-native-f32-v2",
        target: "indextts/models/2.5/c39ce5ba-native-f32-v2".into(),
        inputs: &assets.inputs,
        outputs: &assets.outputs,
        transform_outputs: &[
            "gpt.safetensors",
            "codec.safetensors",
            "s2mel.safetensors",
            "campplus.safetensors",
            "bigvgan.safetensors",
            "feats.safetensors",
        ],
        workspace_bytes: 512 * 1024 * 1024,
        transform: |work| {
            for (name, file) in [
                ("gpt", "gpt.safetensors"),
                ("codec", "codec.safetensors"),
                ("s2mel", "s2mel.safetensors"),
                ("campplus", "campplus.safetensors"),
                ("bigvgan", "bigvgan.safetensors"),
            ] {
                work.component(file, |input, output, control| {
                    convert(input, name, output, control)
                })?;
            }
            work.component("feats.safetensors", convert_features)?;
            for name in [
                "bigvgan.config.json",
                "multilingual_zh_ja_yue_char_del.tiktoken",
                "w2v-bert-2.0/model.safetensors",
                "w2v-bert-2.0/config.json",
                "LICENSE-INDEXTTS",
            ] {
                work.copy(name)?;
            }
            work.write("config.json", include_bytes!("native-config.json"))?;
            work.write("SOURCES.json", include_bytes!("native-sources.json"))?;
            Ok(())
        },
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    #[test]
    #[ignore = "explicit fixed official IndexTTS checkpoints and new output directory"]
    fn convert_fixed_indextts_components() {
        let cache = PathBuf::from(
            std::env::var_os("TALECHIME_INDEXTTS_INPUTS")
                .expect("set explicit verified IndexTTS input directory"),
        );
        let output = PathBuf::from(
            std::env::var_os("TALECHIME_INDEXTTS_NATIVE_OUTPUT").expect("set new absolute output"),
        );
        assert!(cache.is_absolute() && output.is_absolute() && !output.exists());
        std::fs::create_dir(&output).unwrap();
        let control = PreparationControl::default();
        for name in ["gpt", "codec", "s2mel", "campplus", "bigvgan"] {
            convert(
                &cache,
                name,
                &output.join(format!("{name}.safetensors")),
                &control,
            )
            .unwrap();
            println!("converted {name}");
        }
        convert_features(&cache, &output.join("feats.safetensors"), &control).unwrap();
    }
}

#[cfg(test)]
mod recipe_tests {
    use super::*;
    #[test]
    fn config_and_recipe_keep_the_fixed_f32_contract() {
        let config =
            talechime_indextts::config::Config::from_json_str(include_str!("native-config.json"))
                .unwrap();
        config.validate().unwrap();
        assert_eq!(config.gpt.number_text_tokens, 60509);
        assert_eq!(assets().inputs.len(), 14);
        assert_eq!(assets().outputs.len(), 13);
        assert_eq!(source_recipe().transform_outputs.len(), 6);
        assert!(recipe().transform_outputs.is_empty());
    }
    #[tokio::test]
    #[ignore = "explicit verified IndexTTS inputs in isolated engine root; no new downloads expected"]
    async fn prepare_native_indextts_from_verified_inputs() {
        let root = std::path::PathBuf::from(
            std::env::var_os("TALECHIME_NATIVE_ENGINE_ROOT").expect("set isolated engine root"),
        );
        let downloads =
            crate::downloads::Downloads::new(&crate::downloads::DownloadConfig::default()).unwrap();
        let recipe = if std::env::var_os("TALECHIME_BUILD_NATIVE_SOURCE").is_some() {
            source_recipe()
        } else {
            recipe()
        };
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
        assert!(plan.ready && plan.conversions.is_empty());
        assert_eq!(plan.peak_additional_bytes, 0);
        assert_eq!(recipe.status(&root), crate::ResourceStatus::Ready);
        recipe
            .verify_existing(root.join(&recipe.target), PreparationControl::default())
            .await
            .unwrap();
    }
}
