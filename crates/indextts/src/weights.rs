//! Converted-checkpoint loading.
//!
//! The development export script (`tools/tts/indextts_export_safetensors.py`)
//! converts the official torch checkpoints into safetensors with the Python
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
    pub w2v_bert: VarBuilder<'static>,
    pub bigvgan_config_json: String,
    pub feats: FeatTensors,
}

fn mmap(dir: &std::path::Path, name: &str, device: &Device) -> Result<VarBuilder<'static>> {
    let path = dir.join(name);
    if !path.is_file() {
        bail!(
            "missing converted IndexTTS weight {} (run tools/tts/indextts_export_safetensors.py)",
            path.display()
        );
    }
    // SAFETY: the mapping is owned by the returned VarBuilder and the
    // checkpoint files are not modified while the model is loaded.
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
}

fn feat(dir: &std::path::Path, name: &str, device: &Device) -> Result<Tensor> {
    let all = candle_core::safetensors::load(dir.join("feats.safetensors"), device)?;
    all.get(name).cloned().ok_or_else(|| {
        candle_core::Error::Msg(format!("feats.safetensors is missing tensor {name}"))
    })
}

impl Weights {
    pub fn load_dir(dir: &std::path::Path, device: &Device) -> Result<Self> {
        let bigvgan_config_path = dir.join("bigvgan.config.json");
        if !bigvgan_config_path.is_file() {
            bail!("missing {}", bigvgan_config_path.display());
        }
        let bigvgan_config_json = std::fs::read_to_string(&bigvgan_config_path)
            .map_err(|e| candle_core::Error::Msg(format!("read bigvgan config: {e}")))?;
        let feats = FeatTensors {
            spk_matrix: feat(dir, "spk_matrix", device)?,
            emo_matrix: feat(dir, "emo_matrix", device)?,
            w2v_mean: feat(dir, "w2v_mean", device)?,
            w2v_std: feat(dir, "w2v_std", device)?,
        };
        Ok(Self {
            gpt: mmap(dir, "gpt.safetensors", device)?,
            codec: mmap(dir, "codec.safetensors", device)?,
            s2mel: mmap(dir, "s2mel.safetensors", device)?,
            campplus: mmap(dir, "campplus.safetensors", device)?,
            bigvgan: mmap(dir, "bigvgan.safetensors", device)?,
            w2v_bert: mmap(&dir.join("w2v-bert-2.0"), "model.safetensors", device)?,
            bigvgan_config_json,
            feats,
        })
    }
}
