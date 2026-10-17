//! Typed view of the official IndexTTS 2.5 `config.yaml`.
//!
//! Upstream reads the YAML with OmegaConf; the development conversion script
//! (`tools/tts/indextts_export_safetensors.py`) emits the same structure as
//! JSON next to the converted weights, and this crate only consumes that JSON
//! so no YAML parser is needed at runtime. Field names and defaults mirror
//! the Python constructors (`UnifiedVoice`, `EnhancedCodec`, `MyModel`).

use candle_core::{Result, bail};
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub gpt: GptConfig,
    pub semantic_codec: SemanticCodecConfig,
    pub s2mel: S2MelConfig,
    pub gpt_checkpoint: String,
    pub w2v_stat: String,
    pub s2mel_checkpoint: String,
    pub emo_matrix: String,
    pub spk_matrix: String,
    pub emo_num: Vec<usize>,
    pub qwen_emo_path: Option<String>,
    pub vocoder: VocoderConfig,
    #[serde(deserialize_with = "deserialize_version", default)]
    pub version: Option<String>,
}

fn deserialize_version<'de, D>(deserializer: D) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde_json::Value;
    match Value::deserialize(deserializer)? {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s)),
        Value::Number(n) => Ok(Some(n.to_string())),
        other => Err(serde::de::Error::custom(format!(
            "unsupported version value: {other}"
        ))),
    }
}

impl Config {
    pub fn from_json_str(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| candle_core::Error::Msg(format!("invalid IndexTTS config: {e}")))
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| candle_core::Error::Msg(format!("read config {}: {e}", path.display())))?;
        Self::from_json_str(&json)
    }

    /// Validate cross-field invariants the Python code assumes implicitly.
    pub fn validate(&self) -> Result<()> {
        if !self.gpt.model_dim.is_multiple_of(self.gpt.heads) {
            bail!("gpt.model_dim must be divisible by gpt.heads");
        }
        if self.gpt.number_mel_codes <= self.gpt.stop_mel_token as usize {
            bail!("gpt.number_mel_codes must exceed stop_mel_token");
        }
        if self.semantic_codec.downsample_scale != 2 {
            bail!(
                "only downsample_scale=2 is implemented (got {})",
                self.semantic_codec.downsample_scale
            );
        }
        if self.semantic_codec.num_quantizers != 1 {
            bail!(
                "only num_quantizers=1 is implemented (got {})",
                self.semantic_codec.num_quantizers
            );
        }
        if self.s2mel.length_regulator.is_discrete {
            bail!("the shipped 2.5 checkpoint uses a continuous length regulator");
        }
        if self.s2mel.dit.is_causal {
            bail!("the shipped 2.5 checkpoint uses a non-causal DiT");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct GptConfig {
    pub model_dim: usize,
    pub max_mel_tokens: usize,
    pub max_text_tokens: usize,
    pub heads: usize,
    pub layers: usize,
    pub number_text_tokens: usize,
    pub number_mel_codes: usize,
    pub start_mel_token: u32,
    pub stop_mel_token: u32,
    pub start_text_token: u32,
    pub stop_text_token: u32,
    pub train_solo_embeddings: bool,
    pub use_mel_codes_as_input: bool,
    pub mel_length_compression: usize,
    pub max_conditioning_inputs: usize,
    pub condition_type: String,
    pub condition_module: Option<ConformerModuleConfig>,
    pub emo_condition_module: ConformerModuleConfig,
    /// Number of perceiver latents (`condition_num_latent` in Python).
    pub condition_num_latent: usize,
}

impl Default for GptConfig {
    fn default() -> Self {
        // Defaults follow `UnifiedVoice.__init__`; values present in the
        // shipped 2.5 config.yaml are kept aligned with that file.
        Self {
            model_dim: 1280,
            max_mel_tokens: 1815,
            max_text_tokens: 600,
            heads: 20,
            layers: 24,
            number_text_tokens: 60509,
            number_mel_codes: 8194,
            start_mel_token: 8192,
            stop_mel_token: 8193,
            start_text_token: 0,
            stop_text_token: 1,
            train_solo_embeddings: false,
            use_mel_codes_as_input: true,
            mel_length_compression: 1024,
            max_conditioning_inputs: 1,
            condition_type: "conformer_perceiver".to_string(),
            condition_module: None,
            emo_condition_module: ConformerModuleConfig::default(),
            condition_num_latent: 32,
        }
    }
}

impl GptConfig {
    /// The 2.5 inference entry always uses `spk_cond_mode="campplus"`; the
    /// speaker conformer/perceiver modules of the other modes are absent from
    /// the checkpoint and are intentionally not ported.
    pub fn requires_speaker_conformer(&self) -> bool {
        self.condition_type != "campplus"
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ConformerModuleConfig {
    pub output_size: usize,
    pub linear_units: usize,
    pub attention_heads: usize,
    pub num_blocks: usize,
    pub input_layer: String,
    pub perceiver_mult: usize,
}

impl Default for ConformerModuleConfig {
    fn default() -> Self {
        Self {
            output_size: 512,
            linear_units: 1024,
            attention_heads: 4,
            num_blocks: 4,
            input_layer: "conv2d2".to_string(),
            perceiver_mult: 2,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SemanticCodecConfig {
    pub codebook_size: usize,
    pub hidden_size: usize,
    pub codebook_dim: usize,
    pub vocos_dim: usize,
    pub vocos_intermediate_dim: usize,
    pub vocos_num_layers: usize,
    pub num_quantizers: usize,
    pub downsample_scale: usize,
}

impl Default for SemanticCodecConfig {
    fn default() -> Self {
        Self {
            codebook_size: 8192,
            hidden_size: 1024,
            codebook_dim: 8,
            vocos_dim: 384,
            vocos_intermediate_dim: 2048,
            vocos_num_layers: 12,
            num_quantizers: 1,
            downsample_scale: 2,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct S2MelConfig {
    pub preprocess_params: PreprocessParams,
    pub dit_type: Option<String>,
    pub reg_loss_type: Option<String>,
    pub style_encoder: Option<StyleEncoderConfig>,
    pub length_regulator: LengthRegulatorConfig,
    /// YAML key is `DiT`.
    #[serde(rename = "DiT")]
    pub dit: DiTConfig,
    pub wavenet: WavenetConfig,
}

impl Default for S2MelConfig {
    fn default() -> Self {
        Self {
            preprocess_params: PreprocessParams::default(),
            dit_type: Some("DiT".to_string()),
            reg_loss_type: Some("l1".to_string()),
            style_encoder: None,
            length_regulator: LengthRegulatorConfig::default(),
            dit: DiTConfig::default(),
            wavenet: WavenetConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PreprocessParams {
    pub sr: u32,
    pub spect_params: SpectParams,
}

impl Default for PreprocessParams {
    fn default() -> Self {
        Self {
            sr: 22_050,
            spect_params: SpectParams::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SpectParams {
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub fmin: f64,
    /// Upstream stores `fmax: None` as the YAML string "None".
    #[serde(deserialize_with = "deserialize_fmax")]
    pub fmax: Option<f64>,
}

impl Default for SpectParams {
    fn default() -> Self {
        Self {
            n_fft: 1024,
            win_length: 1024,
            hop_length: 256,
            n_mels: 80,
            fmin: 0.0,
            fmax: None,
        }
    }
}

fn deserialize_fmax<'de, D>(deserializer: D) -> std::result::Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde_json::Value;
    let v = Value::deserialize(deserializer)?;
    match &v {
        Value::Null => Ok(None),
        Value::String(s) if s.eq_ignore_ascii_case("none") => Ok(None),
        Value::Number(n) => Ok(n.as_f64()),
        other => Err(serde::de::Error::custom(format!(
            "unsupported fmax value: {other}"
        ))),
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct StyleEncoderConfig {
    pub dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LengthRegulatorConfig {
    pub channels: usize,
    pub is_discrete: bool,
    pub in_channels: usize,
    pub content_codebook_size: usize,
    pub sampling_ratios: Vec<usize>,
    pub vector_quantize: bool,
    pub n_codebooks: usize,
    pub quantizer_dropout: f64,
    pub f0_condition: bool,
    pub n_f0_bins: usize,
}

impl Default for LengthRegulatorConfig {
    fn default() -> Self {
        Self {
            channels: 512,
            is_discrete: false,
            in_channels: 1024,
            content_codebook_size: 2048,
            sampling_ratios: vec![1, 1, 1, 1],
            vector_quantize: false,
            n_codebooks: 1,
            quantizer_dropout: 0.0,
            f0_condition: false,
            n_f0_bins: 512,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DiTConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub class_dropout_prob: f64,
    pub block_size: usize,
    pub in_channels: usize,
    pub style_condition: bool,
    pub final_layer_type: String,
    pub target: String,
    pub content_dim: usize,
    pub content_codebook_size: usize,
    pub content_type: String,
    pub f0_condition: bool,
    pub n_f0_bins: usize,
    pub content_codebooks: usize,
    pub is_causal: bool,
    pub long_skip_connection: bool,
    pub zero_prompt_speech_token: bool,
    pub time_as_token: bool,
    pub style_as_token: bool,
    pub uvit_skip_connection: bool,
    pub add_resblock_in_transformer: bool,
}

impl Default for DiTConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 512,
            num_heads: 8,
            depth: 13,
            class_dropout_prob: 0.1,
            block_size: 8192,
            in_channels: 80,
            style_condition: true,
            final_layer_type: "wavenet".to_string(),
            target: "mel".to_string(),
            content_dim: 512,
            content_codebook_size: 1024,
            content_type: "discrete".to_string(),
            f0_condition: false,
            n_f0_bins: 512,
            content_codebooks: 1,
            is_causal: false,
            long_skip_connection: true,
            zero_prompt_speech_token: false,
            time_as_token: false,
            style_as_token: false,
            uvit_skip_connection: true,
            add_resblock_in_transformer: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct WavenetConfig {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub kernel_size: usize,
    pub dilation_rate: usize,
    pub p_dropout: f64,
    pub style_condition: bool,
}

impl Default for WavenetConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 512,
            num_layers: 8,
            kernel_size: 5,
            dilation_rate: 1,
            p_dropout: 0.2,
            style_condition: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VocoderConfig {
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            kind: "bigvgan".to_string(),
            name: "bigvgan_generator.pt".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped_config_json() -> String {
        // Structure of IndexTeam/IndexTTS-2.5 config.yaml (yaml -> json).
        r#"{
            "gpt": {
                "model_dim": 1280, "max_mel_tokens": 1815, "max_text_tokens": 600,
                "heads": 20, "use_mel_codes_as_input": true, "mel_length_compression": 1024,
                "layers": 24, "number_text_tokens": 60509, "number_mel_codes": 8194,
                "start_mel_token": 8192, "stop_mel_token": 8193,
                "start_text_token": 0, "stop_text_token": 1, "train_solo_embeddings": false,
                "condition_type": "conformer_perceiver",
                "condition_module": {"output_size": 512, "linear_units": 2048, "attention_heads": 8,
                                      "num_blocks": 6, "input_layer": "conv2d2", "perceiver_mult": 2},
                "emo_condition_module": {"output_size": 512, "linear_units": 1024, "attention_heads": 4,
                                          "num_blocks": 4, "input_layer": "conv2d2", "perceiver_mult": 2}
            },
            "semantic_codec": {"codebook_size": 8192, "hidden_size": 1024, "codebook_dim": 8,
                                "vocos_dim": 384, "vocos_intermediate_dim": 2048, "vocos_num_layers": 12},
            "s2mel": {
                "preprocess_params": {"sr": 22050, "spect_params": {"n_fft": 1024, "win_length": 1024,
                    "hop_length": 256, "n_mels": 80, "fmin": 0, "fmax": "None"}},
                "dit_type": "DiT", "reg_loss_type": "l1",
                "style_encoder": {"dim": 192},
                "length_regulator": {"channels": 512, "is_discrete": false, "in_channels": 1024,
                    "content_codebook_size": 2048, "sampling_ratios": [1, 1, 1, 1], "vector_quantize": false,
                    "n_codebooks": 1, "quantizer_dropout": 0.0, "f0_condition": false, "n_f0_bins": 512},
                "DiT": {"hidden_dim": 512, "num_heads": 8, "depth": 13, "class_dropout_prob": 0.1,
                    "block_size": 8192, "in_channels": 80, "style_condition": true,
                    "final_layer_type": "wavenet", "target": "mel", "content_dim": 512,
                    "content_codebook_size": 1024, "content_type": "discrete", "f0_condition": false,
                    "n_f0_bins": 512, "content_codebooks": 1, "is_causal": false,
                    "long_skip_connection": true, "zero_prompt_speech_token": false,
                    "time_as_token": false, "style_as_token": false, "uvit_skip_connection": true,
                    "add_resblock_in_transformer": false},
                "wavenet": {"hidden_dim": 512, "num_layers": 8, "kernel_size": 5, "dilation_rate": 1,
                    "p_dropout": 0.2, "style_condition": true}
            },
            "gpt_checkpoint": "gpt.pth", "w2v_stat": "wav2vec2bert_stats.pt",
            "s2mel_checkpoint": "s2mel.pth", "emo_matrix": "feat2.pt", "spk_matrix": "feat1.pt",
            "emo_num": [3, 17, 2, 8, 4, 5, 10, 24],
            "qwen_emo_path": "qwen0.6bemo4-merge/",
            "vocoder": {"type": "bigvgan", "name": "bigvgan_generator.pt"},
            "version": 2.5
        }"#
        .to_string()
    }

    #[test]
    fn parses_shipped_config() {
        let cfg = Config::from_json_str(&shipped_config_json()).expect("parse");
        assert_eq!(cfg.gpt.model_dim, 1280);
        assert_eq!(cfg.gpt.layers, 24);
        assert_eq!(cfg.gpt.number_mel_codes, 8194);
        assert_eq!(cfg.semantic_codec.vocos_num_layers, 12);
        assert_eq!(cfg.s2mel.dit.depth, 13);
        assert_eq!(cfg.s2mel.preprocess_params.spect_params.n_mels, 80);
        assert_eq!(cfg.s2mel.preprocess_params.spect_params.fmax, None);
        assert_eq!(cfg.emo_num, vec![3, 17, 2, 8, 4, 5, 10, 24]);
        assert_eq!(cfg.version.as_deref(), Some("2.5"));
        cfg.validate().expect("valid");
    }
}
