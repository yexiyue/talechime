//! Declared OmniVoice generation parameters and their typed builder.
use ::omnivoice::contracts::GenerationConfig;
use tts_core::params::GenerationParams;
use tts_protocol::{ParamKind, ParamValue, ParameterSpec};

pub(super) fn catalog() -> Vec<ParameterSpec> {
    vec![
        ParameterSpec {
            name: "num_step".into(),
            kind: ParamKind::Int { min: 1, max: 4096 },
            default: ParamValue::Int(32),
            description: "扩散解码步数；更高更精细也更慢".into(),
        },
        ParameterSpec {
            name: "guidance_scale".into(),
            kind: ParamKind::Float {
                min: 0.0,
                max: 10.0,
            },
            default: ParamValue::Float(2.0),
            description: "生成引导强度".into(),
        },
        ParameterSpec {
            name: "language".into(),
            kind: ParamKind::Text { max_len: 48 },
            default: ParamValue::Text("zh".into()),
            description: "BCP-47 代码或语言名（模型内建约 700 种）".into(),
        },
        ParameterSpec {
            name: "speed".into(),
            kind: ParamKind::Float { min: 0.5, max: 2.0 },
            default: ParamValue::Float(1.0),
            description: "生成期原生语速（无损变速）；会话 speed 在支持时自动路由到此处".into(),
        },
    ]
}

/// Typed builder for the declared OmniVoice parameters; unset fields use catalog defaults.
#[derive(Debug, Clone, Default)]
pub struct OmniVoiceParams {
    pub num_step: Option<u32>,
    pub guidance_scale: Option<f32>,
    pub language: Option<String>,
    pub speed: Option<f32>,
}

impl OmniVoiceParams {
    pub fn num_step(mut self, steps: u32) -> Self {
        self.num_step = Some(steps);
        self
    }
    pub fn guidance_scale(mut self, scale: f32) -> Self {
        self.guidance_scale = Some(scale);
        self
    }
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }
    /// Convert into the backend-agnostic parameter map.
    pub fn to_generation_params(&self) -> GenerationParams {
        let mut params = GenerationParams::new();
        if let Some(steps) = self.num_step {
            params.insert("num_step", ParamValue::Int(i64::from(steps)));
        }
        if let Some(scale) = self.guidance_scale {
            params.insert("guidance_scale", ParamValue::Float(f64::from(scale)));
        }
        if let Some(language) = &self.language {
            params.insert("language", ParamValue::Text(language.clone()));
        }
        if let Some(speed) = self.speed {
            params.insert("speed", ParamValue::Float(f64::from(speed)));
        }
        params
    }
}

/// Resolved per-request settings for the inference thread.
#[derive(Debug)]
pub(super) struct Resolved {
    pub config: GenerationConfig,
    /// Normalized language id; the adapter rejects unknown names explicitly.
    pub language: String,
    pub speed: Option<f32>,
}

/// Resolve validated parameters; unknown languages fail explicitly instead of
/// degrading to the prompt's "None" slot.
pub(super) fn resolve(
    params: &GenerationParams,
) -> Result<Resolved, tts_core::backend::BackendError> {
    let language = match params.get("language") {
        Some(ParamValue::Text(name)) => ::omnivoice::frontend::resolve_language(Some(name))
            .ok_or_else(|| {
                tts_core::backend::BackendError::Unsupported(format!(
                    "unknown OmniVoice language parameter '{name}'"
                ))
            })?,
        _ => "zh".to_string(),
    };
    let mut config = GenerationConfig::default();
    if let Some(ParamValue::Int(steps)) = params.get("num_step") {
        config.num_step = *steps as usize;
    }
    if let Some(value) = params.get("guidance_scale") {
        config.guidance_scale = match value {
            ParamValue::Float(v) => *v as f32,
            ParamValue::Int(v) => *v as f32,
            _ => config.guidance_scale,
        };
    }
    let speed = params.get("speed").map(|value| match value {
        ParamValue::Float(v) => *v as f32,
        ParamValue::Int(v) => *v as f32,
        _ => 1.0,
    });
    Ok(Resolved {
        config,
        language,
        speed,
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
        let typed = OmniVoiceParams::default()
            .num_step(64)
            .guidance_scale(1.5)
            .language("Chinese")
            .speed(1.25);
        let params = typed.to_generation_params();
        assert!(params.validate(&catalog()).is_ok());
        let resolved = resolve(&params).unwrap();
        assert_eq!(resolved.config.num_step, 64);
        assert_eq!(resolved.config.guidance_scale, 1.5);
        assert_eq!(resolved.language, "zh");
        assert_eq!(resolved.speed, Some(1.25));
    }

    #[test]
    fn unknown_language_fails_explicitly() {
        let mut params = GenerationParams::new();
        params.insert("language", ParamValue::Text("klingon".into()));
        let error = resolve(&params).unwrap_err().to_string();
        assert!(error.contains("klingon"), "{error}");
    }

    #[test]
    fn empty_params_keep_previous_defaults() {
        let resolved = resolve(&GenerationParams::new()).unwrap();
        assert_eq!(resolved.config.num_step, 32);
        assert_eq!(resolved.config.guidance_scale, 2.0);
        assert_eq!(resolved.language, "zh");
        assert_eq!(resolved.speed, None);
    }
}
