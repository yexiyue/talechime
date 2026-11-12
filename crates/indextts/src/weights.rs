//! Converted-checkpoint loading.
//!
//! The retired development exporter (Git revision `e591c94`)
//! converted the official torch checkpoints into safetensors with the Python
//! state-dict names preserved verbatim:
//!
//! ```text
//! <dir>/config.json                      (config.yaml -> JSON)
//! <dir>/gpt.safetensors                   (gpt.pth['model'])
//! <dir>/codec.safetensors                 (codec.pth['model'])
//! <dir>/s2mel.safetensors                 (s2mel.pth['model'])
//! <dir>/campplus.safetensors              (campplus_cn_common.bin)
//! <dir>/bigvgan.safetensors               (bigvgan_generator.pt, weight-norm folded at load)
//! <dir>/bigvgan.config.json               (nvidia bigvgan config.json)
//! <dir>/feats.safetensors                 (spk_matrix, emo_matrix, w2v_mean, w2v_std)
//! <dir>/multilingual_zh_ja_yue_char_del.tiktoken
//! <dir>/w2v-bert-2.0/                     (original HF repo: model.safetensors + config.json)
//! ```
//!
//! The mmapped safetensors builders are kept alive for the lifetime of
//! [`Weights`] (candle keeps the mapping inside the `VarBuilder`), matching
//! the pattern used by the voxcpm crate.

use candle_core::{DType, Device, Result, Tensor, bail};
use candle_nn::VarBuilder;

pub struct FeatTensors {
    /// feat1.pt — speaker anchor rows, per emotion group (`spk_matrix`).
    pub spk_matrix: Tensor,
    /// feat2.pt — emotion anchor rows (`emo_matrix`), split by `emo_num`.
    pub emo_matrix: Tensor,
    /// wav2vec2bert_stats.pt `mean` / sqrt(var) for hidden-state
    /// normalization.
    pub w2v_mean: Tensor,
    pub w2v_std: Tensor,
}

pub struct Weights {
    pub gpt: VarBuilder<'static>,
    pub codec: VarBuilder<'static>,
    pub s2mel: VarBuilder<'static>,
    pub campplus: VarBuilder<'static>,
    pub bigvgan: VarBuilder<'static>,
    pub bigvgan_config_json: String,
    pub feats: FeatTensors,
}

fn mmap(path: &std::path::Path, device: &Device) -> Result<VarBuilder<'static>> {
    if !path.is_file() {
        bail!(
            "missing converted IndexTTS weight {} (provide offline converted weights from the pinned upstream checkpoint; see docs/indextts.md)",
            path.display()
        );
    }
    // SAFETY: the mapping is owned by the returned VarBuilder and the
    // checkpoint files are not modified while the model is loaded.
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
}

/// Pinned weight and configuration files resolved independently by the caller.
pub struct WeightFiles<'a> {
    pub gpt: &'a std::path::Path,
    pub codec: &'a std::path::Path,
    pub s2mel: &'a std::path::Path,
    pub campplus: &'a std::path::Path,
    pub bigvgan: &'a std::path::Path,
    pub bigvgan_config: &'a std::path::Path,
    pub feats: &'a std::path::Path,
}

fn take_feat(all: &mut std::collections::HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    all.remove(name).ok_or_else(|| {
        candle_core::Error::Msg(format!("feats.safetensors is missing tensor {name}"))
    })
}

impl Weights {
    pub fn load_dir(dir: &std::path::Path, device: &Device) -> Result<Self> {
        Self::load_files(
            &WeightFiles {
                gpt: &dir.join("gpt.safetensors"),
                codec: &dir.join("codec.safetensors"),
                s2mel: &dir.join("s2mel.safetensors"),
                campplus: &dir.join("campplus.safetensors"),
                bigvgan: &dir.join("bigvgan.safetensors"),
                bigvgan_config: &dir.join("bigvgan.config.json"),
                feats: &dir.join("feats.safetensors"),
            },
            device,
        )
    }
    pub fn load_files(files: &WeightFiles<'_>, device: &Device) -> Result<Self> {
        let bigvgan_config_path = files.bigvgan_config;
        if !bigvgan_config_path.is_file() {
            bail!("missing {}", bigvgan_config_path.display());
        }
        let bigvgan_config_json = std::fs::read_to_string(bigvgan_config_path)
            .map_err(|e| candle_core::Error::Msg(format!("read bigvgan config: {e}")))?;
        let mut all_feats = candle_core::safetensors::load(files.feats, device)?;
        let feats = FeatTensors {
            spk_matrix: take_feat(&mut all_feats, "spk_matrix")?,
            emo_matrix: take_feat(&mut all_feats, "emo_matrix")?,
            w2v_mean: take_feat(&mut all_feats, "w2v_mean")?,
            w2v_std: take_feat(&mut all_feats, "w2v_std")?,
        };
        Ok(Self {
            gpt: mmap(files.gpt, device)?,
            codec: mmap(files.codec, device)?,
            s2mel: mmap(files.s2mel, device)?,
            campplus: mmap(files.campplus, device)?,
            bigvgan: mmap(files.bigvgan, device)?,
            bigvgan_config_json,
            feats,
        })
    }
}
