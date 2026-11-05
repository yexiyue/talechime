//! Native preparation against the fixed official conversion baseline.
// Special token data: Alibaba Inc, official CosyVoice 074ca6dc, Apache-2.0.
use crate::native_preparation::{
    Archive, PreparationControl, TensorSpec, fold_weight_norm, write_tensors,
};
use candle_core::DType;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::{collections::BTreeMap, path::Path};

fn convert_hift(input: &Path, output: &Path, control: &PreparationControl) -> anyhow::Result<()> {
    let archive = Archive::open(
        input,
        83202622,
        "b279d7641eb97ae55b3b540cfba4f953c26492a2df758328a89a4d007ab87a65",
        &[],
        control,
    )?;
    let mut names = BTreeMap::new();
    let mut specs = BTreeMap::new();
    for (name, info) in archive.tensors() {
        if name.ends_with(".parametrizations.weight.original1") {
            continue;
        }
        let fused = name.ends_with(".parametrizations.weight.original0");
        let target = if fused {
            name.replace(".parametrizations.weight.original0", ".weight")
        } else {
            name.clone()
        };
        let shape = if fused {
            archive
                .tensors()
                .get(&name.replace("original0", "original1"))
                .ok_or_else(|| anyhow::anyhow!("missing normalization pair"))?
                .layout
                .dims()
                .to_vec()
        } else {
            info.layout.dims().to_vec()
        };
        let dtype = if fused && name.starts_with("f0_predictor.") {
            DType::F64
        } else {
            info.dtype
        };
        anyhow::ensure!(
            names.insert(target.clone(), name.clone()).is_none(),
            "duplicate HiFT tensor"
        );
        specs.insert(target, TensorSpec { shape, dtype });
    }
    anyhow::ensure!(specs.len() == 246, "unexpected HiFT tensor coverage");
    write_tensors(output, &specs, control, |target| {
        let name = &names[target];
        if name.ends_with(".parametrizations.weight.original0") {
            let dtype = specs[target].dtype;
            let g = archive.get(name, control)?.to_dtype(dtype)?;
            let v = archive
                .get(&name.replace("original0", "original1"), control)?
                .to_dtype(dtype)?;
            fold_weight_norm(&v, &g, control)
        } else {
            archive.get(name, control)
        }
    })
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AddedToken {
    id: u32,
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizerGraph {
    version: Box<RawValue>,
    truncation: Box<RawValue>,
    padding: Box<RawValue>,
    added_tokens: Vec<AddedToken>,
    normalizer: Box<RawValue>,
    pre_tokenizer: Box<RawValue>,
    post_processor: Box<RawValue>,
    decoder: Box<RawValue>,
    model: Box<RawValue>,
}
fn complete_tokenizer(input: &Path, output: &Path) -> anyhow::Result<()> {
    let mut graph: TokenizerGraph = serde_json::from_reader(std::fs::File::open(input)?)?;
    let tokens: Vec<String> = serde_json::from_str(include_str!("native-special-tokens.json"))?;
    anyhow::ensure!(
        tokens.len() == 280 && graph.added_tokens.len() == 21,
        "unexpected tokenizer input contract"
    );
    for token in &graph.added_tokens {
        let expected = if token.content == "<|endoftext|>" {
            151643
        } else {
            151644
                + tokens
                    .iter()
                    .position(|t| *t == token.content)
                    .ok_or_else(|| anyhow::anyhow!("unexpected existing special token"))?
                    as u32
        };
        anyhow::ensure!(
            token.id == expected
                && token.special
                && !token.normalized
                && !token.single_word
                && !token.lstrip
                && !token.rstrip,
            "tokenizer identity or flags differ from baseline"
        );
    }
    for (index, content) in tokens.into_iter().enumerate() {
        let id = 151644 + index as u32;
        if graph.added_tokens.iter().any(|t| t.content == content) {
            continue;
        }
        graph.added_tokens.push(AddedToken {
            id,
            content,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: true,
        });
    }
    graph.added_tokens.sort_by_key(|t| t.id);
    anyhow::ensure!(
        graph.added_tokens.len() == 281,
        "tokenizer token coverage mismatch"
    );
    let mut file = std::fs::File::create_new(output)?;
    serde_json::to_writer_pretty(&mut file, &graph)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_special_token_contract_is_fixed() {
        let tokens: Vec<String> =
            serde_json::from_str(include_str!("native-special-tokens.json")).unwrap();
        assert_eq!(tokens.len(), 280);
        assert_eq!(tokens[2], "<|endofprompt|>");
        assert_eq!(tokens[20], "[AA]");
        assert_eq!(tokens[279], "[ǜ]");
        let _: super::TokenizerGraph = serde_json::from_str(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{}}"#).unwrap();
    }
    #[test]
    #[ignore = "explicit pinned CosyVoice3 inputs; writes only a new output directory"]
    fn convert_fixed_cosyvoice_inputs() {
        let root =
            std::env::var_os("TALECHIME_NATIVE_AUDIT_OUTPUT").expect("set new output directory");
        let root = std::path::PathBuf::from(root);
        assert!(root.is_absolute() && !root.exists());
        std::fs::create_dir(&root).unwrap();
        let hub = crate::downloads::default_cache_dir();
        let raw = hub.join("models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512/snapshots/29e01c4e8d000f4bcd70751be16fa94bf3d85a18/hift.pt").canonicalize().unwrap();
        let tokenizer = hub.join("models--spensercai--CosyVoice3-0.5B-Candle/snapshots/ccdf65f11d4352c13692e700560076bd61a531e3/tokenizer/tokenizer.json").canonicalize().unwrap();
        convert_hift(
            &raw,
            &root.join("hift.safetensors"),
            &PreparationControl::default(),
        )
        .unwrap();
        complete_tokenizer(&tokenizer, &root.join("tokenizer.json")).unwrap();
        std::fs::write(
            root.join("config.json"),
            include_bytes!("native-config.json"),
        )
        .unwrap();
    }
}

pub(crate) fn recipe() -> crate::native_preparation::Recipe {
    static INPUTS: std::sync::OnceLock<Vec<crate::native_preparation::Input>> =
        std::sync::OnceLock::new();
    let inputs = INPUTS.get_or_init(|| {
        serde_json::from_str(include_str!("distribution-inputs.json"))
            .expect("compiled distribution pins")
    });
    source_recipe().distributed(
        "cosyvoice3-2512-runtime-f64-v3",
        inputs,
        super::directory(Path::new("")),
    )
}

fn source_recipe() -> crate::native_preparation::Recipe {
    use crate::native_preparation::{Input, Output, Recipe};
    const COMMUNITY: &str = "spensercai/CosyVoice3-0.5B-Candle";
    const REVISION: &str = "ccdf65f11d4352c13692e700560076bd61a531e3";
    Recipe {
        id: "cosyvoice3-2512-native-f64-v2",
        target: "cosyvoice3/models/0.5b/2512-native-f64-v2".into(),
        inputs: &[
            Input {
                path: "llm.safetensors",
                repo: COMMUNITY,
                revision: REVISION,
                filename: "llm.safetensors",
                size: 2569164632,
                sha256: "06661f03d1ae2adf67abd03efb758cdd59df5495617b0066ead5ae8d17f80983",
            },
            Input {
                path: "flow.safetensors",
                repo: COMMUNITY,
                revision: REVISION,
                filename: "flow.safetensors",
                size: 1329066360,
                sha256: "fa408e2c1476a298271f7610578228819a97d30ade959d228a9459a0a135d95b",
            },
            Input {
                path: "campplus.onnx",
                repo: COMMUNITY,
                revision: REVISION,
                filename: "campplus.onnx",
                size: 28303423,
                sha256: "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73",
            },
            Input {
                path: "speech_tokenizer_v3.onnx",
                repo: COMMUNITY,
                revision: REVISION,
                filename: "speech_tokenizer_v3.onnx",
                size: 969451503,
                sha256: "23236a74175dbdda47afc66dbadd5bcb41303c467a57c261cb8539ad9db9208d",
            },
            Input {
                path: "tokenizer-base.json",
                repo: COMMUNITY,
                revision: REVISION,
                filename: "tokenizer/tokenizer.json",
                size: 10526718,
                sha256: "485c75a9c2af8848f63829abf170cae7365cee7a2d16a4a9b0217a5a21f197c4",
            },
            Input {
                path: "hift.pt",
                repo: "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                revision: "29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
                filename: "hift.pt",
                size: 83202622,
                sha256: "b279d7641eb97ae55b3b540cfba4f953c26492a2df758328a89a4d007ab87a65",
            },
        ],
        outputs: &[
            Output {
                path: "llm.safetensors",
                size: 2569164632,
                sha256: "06661f03d1ae2adf67abd03efb758cdd59df5495617b0066ead5ae8d17f80983",
            },
            Output {
                path: "flow.safetensors",
                size: 1329066360,
                sha256: "fa408e2c1476a298271f7610578228819a97d30ade959d228a9459a0a135d95b",
            },
            Output {
                path: "campplus.onnx",
                size: 28303423,
                sha256: "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73",
            },
            Output {
                path: "speech_tokenizer_v3.onnx",
                size: 969451503,
                sha256: "23236a74175dbdda47afc66dbadd5bcb41303c467a57c261cb8539ad9db9208d",
            },
            Output {
                path: "hift.safetensors",
                size: 96324828,
                sha256: "caeedf222ed35ea151a3478eea0d63b72e458d295d8b991a7a7c362221517e8b",
            },
            Output {
                path: "tokenizer/tokenizer.json",
                size: 10573953,
                sha256: "caa0dadcb8c40c79a3244aaf290492e298a49c5c25e0db16cad12f2f8ff639c5",
            },
            Output {
                path: "config.json",
                size: 2074,
                sha256: "455c574d7ccefeed2419d3ad67c87d17545eed2644f65ca46733d126d5790b26",
            },
        ],
        transform_outputs: &["hift.safetensors", "tokenizer/tokenizer.json"],
        workspace_bytes: 256 * 1024 * 1024,
        transform: |work| {
            for name in [
                "llm.safetensors",
                "flow.safetensors",
                "campplus.onnx",
                "speech_tokenizer_v3.onnx",
            ] {
                work.copy(name)?;
            }
            work.component("hift.safetensors", |input, output, control| {
                convert_hift(&input.join("hift.pt"), output, control)
            })?;
            work.component("tokenizer/tokenizer.json", |input, output, _| {
                complete_tokenizer(&input.join("tokenizer-base.json"), output)
            })?;
            work.write("config.json", include_bytes!("native-config.json"))?;
            Ok(())
        },
    }
}
#[cfg(test)]
mod recipe_tests {
    use super::*;
    #[tokio::test]
    #[ignore = "explicit verified existing cache; no new weight downloads expected"]
    async fn prepare_cosyvoice_from_verified_cache() {
        let root =
            std::env::var_os("TALECHIME_NATIVE_ENGINE_ROOT").expect("set isolated engine root");
        let root = std::path::PathBuf::from(root);
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
        let ready = crate::native_preparation::plan_recipe(
            recipe.clone(),
            root.clone(),
            downloads,
            PreparationControl::default(),
        )
        .await
        .unwrap();
        assert!(ready.ready);
        assert_eq!(recipe.status(&root), crate::ResourceStatus::Ready);
        assert_eq!(ready.download_bytes, 0);
        assert_eq!(ready.peak_additional_bytes, 0);
    }
}
