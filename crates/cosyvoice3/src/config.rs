//! CosyVoice3 configuration.
//!
//! Defaults mirror `cosyvoice3.yaml` shipped with
//! FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (fetched 2026-10-17); see
//! docs/records/cosyvoice3-candle-audit-2026-10-17.md for provenance.

use serde::Deserialize;

pub const SAMPLE_RATE: usize = 24_000;
pub const MEL_N_MELS: usize = 80;
pub const MEL_N_FFT: usize = 1920;
pub const MEL_HOP_SIZE: usize = 480;
pub const SPEECH_TOKEN_SIZE: usize = 6561;
pub const TOKEN_FRAME_RATE: usize = 25;
pub const TOKEN_MEL_RATIO: usize = 2;
pub const SPK_EMBED_DIM: usize = 192;
pub const LLM_DIM: usize = 896;
/// speech vocab = 6561 tokens + 200 reserved (sos/eos/task/fill live here).
pub const SPEECH_VOCAB_SIZE: usize = SPEECH_TOKEN_SIZE + 200;
pub const CHUNK_SIZE: usize = 25;
pub const PRE_LOOKAHEAD_LEN: usize = 3;

/// Special speech-token ids (`CosyVoice3LM.__init__`).
pub const SOS_TOKEN: u32 = SPEECH_TOKEN_SIZE as u32;
pub const EOS_TOKEN: u32 = SPEECH_TOKEN_SIZE as u32 + 1;
pub const TASK_ID_TOKEN: u32 = SPEECH_TOKEN_SIZE as u32 + 2;
pub const FILL_TOKEN: u32 = SPEECH_TOKEN_SIZE as u32 + 3;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosyVoice3Config {
    pub sample_rate: usize,
    pub llm: LlmConfig,
    pub flow: FlowConfig,
    pub hift: HiftConfig,
}

impl Default for CosyVoice3Config {
    fn default() -> Self {
        Self {
            sample_rate: SAMPLE_RATE,
            llm: LlmConfig::default(),
            flow: FlowConfig::default(),
            hift: HiftConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmConfig {
    pub llm_input_size: usize,
    pub llm_output_size: usize,
    pub speech_token_size: usize,
    pub qwen2: Qwen2Config,
    #[serde(default = "default_sampling")]
    pub sampling: SamplingConfig,
}

fn default_sampling() -> SamplingConfig {
    SamplingConfig::default()
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            llm_input_size: LLM_DIM,
            llm_output_size: LLM_DIM,
            speech_token_size: SPEECH_TOKEN_SIZE,
            qwen2: Qwen2Config::default(),
            sampling: SamplingConfig::default(),
        }
    }
}

/// Qwen2-0.5B backbone as shipped in `CosyVoice-BlankEN`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qwen2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
}

impl Default for Qwen2Config {
    fn default() -> Self {
        Self {
            hidden_size: 896,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            intermediate_size: 4864,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            vocab_size: 151_936,
            tie_word_embeddings: true,
        }
    }
}

/// `ras_sampling` parameters from the yaml (`utils.common.ras_sampling`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingConfig {
    pub top_p: f32,
    pub top_k: usize,
    pub win_size: usize,
    pub tau_r: f32,
    /// Optional seed; `None` draws a fresh seed per request. Reproduces
    /// Rust-side trajectories only — PyTorch samples via `multinomial`, so
    /// cross-framework bit parity does not exist.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            top_p: 0.8,
            top_k: 25,
            win_size: 10,
            tau_r: 0.1,
            seed: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowConfig {
    pub input_size: usize,
    pub output_size: usize,
    pub vocab_size: usize,
    pub token_mel_ratio: usize,
    pub pre_lookahead_len: usize,
    pub dit: DiTConfig,
    pub cfm: CfmConfig,
}

impl Default for FlowConfig {
    fn default() -> Self {
        Self {
            input_size: MEL_N_MELS,
            output_size: MEL_N_MELS,
            vocab_size: SPEECH_TOKEN_SIZE,
            token_mel_ratio: TOKEN_MEL_RATIO,
            pre_lookahead_len: PRE_LOOKAHEAD_LEN,
            dit: DiTConfig::default(),
            cfm: CfmConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiTConfig {
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub dim_head: usize,
    pub ff_mult: usize,
    pub mel_dim: usize,
    pub mu_dim: usize,
    pub spk_dim: usize,
    pub out_channels: usize,
    pub static_chunk_size: usize,
}

impl Default for DiTConfig {
    fn default() -> Self {
        Self {
            dim: 1024,
            depth: 22,
            heads: 16,
            dim_head: 64,
            ff_mult: 2,
            mel_dim: MEL_N_MELS,
            mu_dim: MEL_N_MELS,
            spk_dim: MEL_N_MELS,
            out_channels: MEL_N_MELS,
            // yaml: chunk_size * token_mel_ratio
            static_chunk_size: CHUNK_SIZE * TOKEN_MEL_RATIO,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CfmConfig {
    pub sigma_min: f64,
    pub t_scheduler: String,
    pub training_cfg_rate: f64,
    /// Official yaml value is 0.7; do not regress to 0.0.
    pub inference_cfg_rate: f64,
    pub n_timesteps: usize,
}

impl Default for CfmConfig {
    fn default() -> Self {
        Self {
            sigma_min: 1e-6,
            t_scheduler: "cosine".to_string(),
            training_cfg_rate: 0.2,
            inference_cfg_rate: 0.7,
            n_timesteps: 10,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HiftConfig {
    pub in_channels: usize,
    pub base_channels: usize,
    pub nb_harmonics: usize,
    pub sampling_rate: usize,
    pub nsf_alpha: f64,
    pub nsf_sigma: f64,
    #[serde(default = "default_nsf_voiced_threshold")]
    pub nsf_voiced_threshold: f64,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub istft_n_fft: usize,
    pub istft_hop_len: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub source_resblock_kernel_sizes: Vec<usize>,
    pub source_resblock_dilation_sizes: Vec<Vec<usize>>,
    #[serde(default = "default_lrelu_slope")]
    pub lrelu_slope: f64,
    #[serde(default = "default_audio_limit")]
    pub audio_limit: f64,
    pub conv_pre_look_right: usize,
}

fn default_nsf_voiced_threshold() -> f64 {
    10.0
}

fn default_lrelu_slope() -> f64 {
    0.1
}

fn default_audio_limit() -> f64 {
    0.99
}

impl Default for HiftConfig {
    fn default() -> Self {
        Self {
            in_channels: MEL_N_MELS,
            base_channels: 512,
            nb_harmonics: 8,
            sampling_rate: SAMPLE_RATE,
            nsf_alpha: 0.1,
            nsf_sigma: 0.003,
            nsf_voiced_threshold: 10.0,
            upsample_rates: vec![8, 5, 3],
            upsample_kernel_sizes: vec![16, 11, 7],
            istft_n_fft: 16,
            istft_hop_len: 4,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
            source_resblock_kernel_sizes: vec![7, 7, 11],
            source_resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
            lrelu_slope: 0.1,
            audio_limit: 0.99,
            conv_pre_look_right: 4,
        }
    }
}

impl CosyVoice3Config {
    /// Load `config.json` from a converted model directory.
    pub fn from_model_dir(dir: &std::path::Path) -> std::io::Result<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_official_yaml() {
        let cfg = CosyVoice3Config::default();
        assert_eq!(cfg.flow.cfm.inference_cfg_rate, 0.7);
        assert_eq!(cfg.flow.pre_lookahead_len, 3);
        assert_eq!(cfg.flow.dit.static_chunk_size, 50);
        assert_eq!(cfg.hift.source_resblock_kernel_sizes, vec![7, 7, 11]);
        assert_eq!(cfg.hift.upsample_rates, vec![8, 5, 3]);
        assert_eq!(SPEECH_VOCAB_SIZE, 6761);
        assert_eq!(SOS_TOKEN, 6561);
    }
}
