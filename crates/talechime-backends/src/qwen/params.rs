//! Declared Qwen generation parameters and their typed builder.
use qwen3_tts::{Language, SynthesisOptions};
use tts_core::params::GenerationParams;
use tts_protocol::{ParamKind, ParamValue, ParameterSpec};

/// Codec frames advance at 12.5 Hz; MAX_FRAMES (375) equals 30 seconds.
const FRAME_HZ: f64 = 12.5;
const MAX_DURATION_SECONDS: f64 = 30.0;
const DEFAULT_MAX_FRAMES: usize = super::MAX_FRAMES;

pub(super) fn catalog() -> Vec<ParameterSpec> {
    vec![
        ParameterSpec {
            name: "temperature".into(),
            kind: ParamKind::Float { min: 0.0, max: 2.0 },
            default: ParamValue::Float(0.9),
            description: "采样温度".into(),
        },
        ParameterSpec {
            name: "top_k".into(),
            kind: ParamKind::Int { min: 1, max: 1000 },
            default: ParamValue::Int(50),
            description: "采样候选数".into(),
        },
        ParameterSpec {
            name: "top_p".into(),
            kind: ParamKind::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(0.9),
            description: "核采样概率阈值".into(),
        },
        ParameterSpec {
            name: "repetition_penalty".into(),
            kind: ParamKind::Float { min: 1.0, max: 2.0 },
            default: ParamValue::Float(1.05),
            description: "重复惩罚；音频参考/克隆路径内部下限 1.5".into(),
        },
        ParameterSpec {
            name: "language".into(),
            kind: ParamKind::Text { max_len: 32 },
            default: ParamValue::Text("auto".into()),
            description: "auto 或 en/zh/ja/ko/de/fr/ru/pt/es/it；auto 按文本启发式".into(),
        },
        ParameterSpec {
            name: "max_duration".into(),
            kind: ParamKind::Float {
                min: 1.0,
                max: MAX_DURATION_SECONDS,
            },
            default: ParamValue::Float(DEFAULT_MAX_FRAMES as f64 / FRAME_HZ),
            description: "单段最长生成秒数，受 30 秒音频块预算钳制".into(),
        },
        ParameterSpec {
            name: "chunk_frames".into(),
            kind: ParamKind::Int { min: 1, max: 100 },
            default: ParamValue::Int(10),
            description: "流式 PCM 粒度（帧）；未设置时 CUDA 用 20、其他设备 10".into(),
        },
    ]
}

/// Typed builder for the declared Qwen parameters; unset fields use catalog defaults.
#[derive(Debug, Clone, Default)]
pub struct QwenParams {
    pub temperature: Option<f64>,
    pub top_k: Option<u32>,
    pub top_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub language: Option<String>,
    pub max_duration_seconds: Option<f64>,
    pub chunk_frames: Option<u32>,
}

impl QwenParams {
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }
    pub fn repetition_penalty(mut self, penalty: f64) -> Self {
        self.repetition_penalty = Some(penalty);
        self
    }
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
    pub fn max_duration_seconds(mut self, seconds: f64) -> Self {
        self.max_duration_seconds = Some(seconds);
        self
    }
    pub fn chunk_frames(mut self, frames: u32) -> Self {
        self.chunk_frames = Some(frames);
        self
    }
    /// Convert into the backend-agnostic parameter map.
    pub fn to_generation_params(&self) -> GenerationParams {
        let mut params = GenerationParams::new();
        if let Some(temperature) = self.temperature {
            params.insert("temperature", ParamValue::Float(temperature));
        }
        if let Some(top_k) = self.top_k {
            params.insert("top_k", ParamValue::Int(i64::from(top_k)));
        }
        if let Some(top_p) = self.top_p {
            params.insert("top_p", ParamValue::Float(top_p));
        }
        if let Some(penalty) = self.repetition_penalty {
            params.insert("repetition_penalty", ParamValue::Float(penalty));
        }
        if let Some(language) = &self.language {
            params.insert("language", ParamValue::Text(language.clone()));
        }
        if let Some(seconds) = self.max_duration_seconds {
            params.insert("max_duration", ParamValue::Float(seconds));
        }
        if let Some(frames) = self.chunk_frames {
            params.insert("chunk_frames", ParamValue::Int(i64::from(frames)));
        }
        params
    }
}

/// Resolved per-request settings; chunk_frames stays optional for device defaults.
pub(super) struct Resolved {
    pub options: SynthesisOptions,
    /// None keeps the CJK text heuristic on the inference thread.
    pub language: Option<Language>,
    pub chunk_frames: Option<usize>,
}

/// Resolve validated parameters plus the attempt seed; unknown languages fail explicitly.
pub(super) fn resolve(
    params: &GenerationParams,
    seed: u64,
) -> Result<Resolved, tts_core::backend::BackendError> {
    let float = |name: &str, fallback: f64| {
        params
            .get(name)
            .and_then(ParamValue::as_f64)
            .unwrap_or(fallback)
    };
    let language = match params.get("language") {
        Some(ParamValue::Text(name)) if !name.eq_ignore_ascii_case("auto") => {
            Some(name.parse::<Language>().map_err(|error| {
                tts_core::backend::BackendError::Unsupported(format!(
                    "unknown Qwen language parameter: {error}"
                ))
            })?)
        }
        _ => None,
    };
    let max_duration = float("max_duration", DEFAULT_MAX_FRAMES as f64 / FRAME_HZ);
    Ok(Resolved {
        options: SynthesisOptions {
            max_length: (max_duration * FRAME_HZ).round().max(1.0) as usize,
            temperature: float("temperature", 0.9),
            top_k: match params.get("top_k") {
                Some(ParamValue::Int(v)) => *v as usize,
                _ => 50,
            },
            top_p: float("top_p", 0.9),
            repetition_penalty: float("repetition_penalty", 1.05),
            seed: Some(seed),
            ..Default::default()
        },
        language,
        chunk_frames: match params.get("chunk_frames") {
            Some(ParamValue::Int(v)) => Some((*v).max(1) as usize),
            _ => None,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_defaults_satisfy_their_own_kinds() {
        for spec in catalog() {
            assert!(spec.has_valid_default(), "{}", spec.name);
        }
    }

    #[test]
    fn typed_builder_round_trips_and_resolves() {
        let typed = QwenParams::default()
            .temperature(0.7)
            .top_k(20)
            .top_p(0.8)
            .repetition_penalty(1.2)
            .language("ja")
            .max_duration_seconds(20.0)
            .chunk_frames(16);
        let params = typed.to_generation_params();
        assert!(params.validate(&catalog()).is_ok());
        let resolved = resolve(&params, 11).unwrap();
        assert_eq!(resolved.options.seed, Some(11));
        assert_eq!(resolved.options.temperature, 0.7);
        assert_eq!(resolved.options.top_k, 20);
        assert_eq!(resolved.options.top_p, 0.8);
        assert_eq!(resolved.options.repetition_penalty, 1.2);
        assert_eq!(resolved.options.max_length, 250);
        assert_eq!(resolved.language, Some(Language::Japanese));
        assert_eq!(resolved.chunk_frames, Some(16));
    }

    #[test]
    fn auto_language_and_unknown_values() {
        let mut params = GenerationParams::new();
        params.insert("language", ParamValue::Text("auto".into()));
        assert!(resolve(&params, 1).unwrap().language.is_none());
        params.insert("language", ParamValue::Text("klingon".into()));
        assert!(resolve(&params, 1).is_err());
    }

    #[test]
    fn empty_params_keep_previous_defaults_except_seed() {
        let resolved = resolve(&GenerationParams::new(), 5).unwrap();
        assert_eq!(resolved.options.seed, Some(5));
        assert_eq!(resolved.options.max_length, DEFAULT_MAX_FRAMES);
        assert_eq!(resolved.options.temperature, 0.9);
        assert_eq!(resolved.options.repetition_penalty, 1.05);
        assert!(resolved.language.is_none());
        assert!(resolved.chunk_frames.is_none());
    }
}
