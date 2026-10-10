//! Declared MOSS Local/Realtime generation parameters and their typed builder.
use tts_core::params::GenerationParams;
use tts_protocol::{ParamKind, ParamValue, ParameterSpec};

/// MOSS audio codes advance at 25 Hz; the 750-frame default equals 30 seconds.
const FRAME_HZ: f64 = 25.0;
const MAX_DURATION_SECONDS: f64 = 30.0;
const DEFAULT_MAX_FRAMES: usize = 750;

pub(super) fn catalog(local: bool) -> Vec<ParameterSpec> {
    let mut specs = vec![
        ParameterSpec {
            name: "instruction".into(),
            kind: ParamKind::Text { max_len: 200 },
            default: ParamValue::Text(String::new()),
            description: "user_inst 模板 Instruction 槽位（如情感/质量指示）；空为无".into(),
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
    ];
    if local {
        // Official v1.5 guidance: set the language tag when it is known.
        // Realtime's message format has no language slot and rejects it.
        specs.push(ParameterSpec {
            name: "language".into(),
            kind: ParamKind::Text { max_len: 48 },
            default: ParamValue::Text(String::new()),
            description: "user_inst Language 标签（如 Chinese/French）；空为无".into(),
        });
    }
    specs
}

/// Typed builder for the declared MOSS parameters; unset fields use catalog defaults.
#[derive(Debug, Clone, Default)]
pub struct MossParams {
    pub instruction: Option<String>,
    pub max_duration_seconds: Option<f64>,
}

impl MossParams {
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = Some(instruction.into());
        self
    }
    pub fn max_duration_seconds(mut self, seconds: f64) -> Self {
        self.max_duration_seconds = Some(seconds);
        self
    }
    /// Convert into the backend-agnostic parameter map.
    pub fn to_generation_params(&self) -> GenerationParams {
        let mut params = GenerationParams::new();
        if let Some(instruction) = &self.instruction {
            params.insert("instruction", ParamValue::Text(instruction.clone()));
        }
        if let Some(seconds) = self.max_duration_seconds {
            params.insert("max_duration", ParamValue::Float(seconds));
        }
        params
    }
}

/// Resolved per-request settings for the inference thread.
#[derive(Debug)]
pub(super) struct Resolved {
    pub instruction: Option<String>,
    pub language: Option<String>,
    pub max_frames: usize,
}

pub(super) fn resolve(params: &GenerationParams) -> Resolved {
    let instruction = match params.get("instruction") {
        Some(ParamValue::Text(value)) if !value.trim().is_empty() => Some(value.clone()),
        _ => None,
    };
    let language = match params.get("language") {
        Some(ParamValue::Text(value)) if !value.trim().is_empty() => Some(value.clone()),
        _ => None,
    };
    let fallback = DEFAULT_MAX_FRAMES as f64 / FRAME_HZ;
    let max_duration = match params.get("max_duration") {
        Some(ParamValue::Float(v)) => *v,
        Some(ParamValue::Int(v)) => *v as f64,
        _ => fallback,
    };
    Resolved {
        instruction,
        language,
        max_frames: (max_duration * FRAME_HZ)
            .round()
            .clamp(25.0, DEFAULT_MAX_FRAMES as f64) as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_defaults_satisfy_their_own_kinds() {
        for spec in catalog(true) {
            assert!(spec.has_valid_default(), "{}", spec.name);
        }
    }

    #[test]
    fn typed_builder_round_trips_and_resolves() {
        let typed = MossParams::default()
            .instruction("用平静的语气朗读")
            .max_duration_seconds(12.0);
        let params = typed.to_generation_params();
        assert!(params.validate(&catalog(true)).is_ok());
        let resolved = resolve(&params);
        assert_eq!(resolved.instruction.as_deref(), Some("用平静的语气朗读"));
        assert_eq!(resolved.max_frames, 300);
    }

    #[test]
    fn realtime_catalog_omits_language_and_local_accepts_it() {
        assert!(catalog(true).iter().any(|spec| spec.name == "language"));
        assert!(!catalog(false).iter().any(|spec| spec.name == "language"));
        let mut params = GenerationParams::new();
        params.insert("language", ParamValue::Text("French".into()));
        let resolved = resolve(&params);
        assert_eq!(resolved.language.as_deref(), Some("French"));
        assert!(params.validate(&catalog(false)).is_err());
        assert!(params.validate(&catalog(true)).is_ok());
    }

    #[test]
    fn blank_instruction_and_empty_params_keep_defaults() {
        let mut params = GenerationParams::new();
        params.insert("instruction", ParamValue::Text("  ".into()));
        assert!(resolve(&params).instruction.is_none());
        let resolved = resolve(&GenerationParams::new());
        assert_eq!(resolved.max_frames, DEFAULT_MAX_FRAMES);
        assert!(resolved.instruction.is_none());
    }
}
