//! IndexTTS emotion controls. Names mirror the model's eight emotion groups.
use talechime_core::{backend::BackendError, params::GenerationParams};
use talechime_indextts::Options;
use talechime_protocol::{ParamKind, ParamValue, ParameterSpec};

const EMOTIONS: [(&str, &str, &str); 8] = [
    ("emo_happy", "happy", "高兴"),
    ("emo_angry", "angry", "愤怒"),
    ("emo_sad", "sad", "悲伤"),
    ("emo_afraid", "afraid", "恐惧"),
    ("emo_disgusted", "disgusted", "厌恶"),
    ("emo_melancholic", "melancholic", "低落"),
    ("emo_surprised", "surprised", "惊讶"),
    ("emo_calm", "calm", "平静"),
];

pub(super) fn catalog() -> Vec<ParameterSpec> {
    let mut specs = vec![
        ParameterSpec {
            name: "language".into(),
            kind: ParamKind::Text { max_len: 5 },
            default: ParamValue::Text("zh".into()),
            description: "文本语言：zh 或 en".into(),
        },
        ParameterSpec {
            name: "emo_alpha".into(),
            kind: ParamKind::Float { min: 0.0, max: 1.0 },
            default: ParamValue::Float(1.0),
            description: "情绪强度；仅作用于情绪向量或独立情绪参考".into(),
        },
        ParameterSpec {
            name: "emo_use_random".into(),
            kind: ParamKind::Bool,
            default: ParamValue::Bool(false),
            description: "情绪向量使用随机锚点；受请求 seed 控制".into(),
        },
        ParameterSpec {
            name: "emo_voice".into(),
            kind: ParamKind::Text { max_len: 71 },
            default: ParamValue::Text(String::new()),
            description: "独立情绪参考：已导入的 custom:音色 ID；空为无".into(),
        },
    ];
    specs.extend(EMOTIONS.map(|(name, emotion, label)| ParameterSpec {
        name: name.into(),
        kind: ParamKind::Float { min: 0.0, max: 1.0 },
        default: ParamValue::Float(0.0),
        description: format!(
            "{label}（{emotion}）权重；总和大于 0.8 时按模型规则归一化；未传任一分量则不启用向量"
        ),
    }));
    specs
}

/// Per-model builder; use `SpeechSpan.style` for one named emotion, or an
/// eight-component vector here for mixtures. Neither changes the voice ID.
#[derive(Debug, Clone, Default)]
pub struct IndexTtsParams {
    pub language: Option<String>,
    /// happy, angry, sad, afraid, disgusted, melancholic, surprised, calm.
    pub emotion_vector: Option<[f32; 8]>,
    pub emotion_strength: Option<f64>,
    pub emotion_voice: Option<String>,
    pub emotion_random: Option<bool>,
}

impl IndexTtsParams {
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
    pub fn emotion_vector(mut self, vector: [f32; 8]) -> Self {
        self.emotion_vector = Some(vector);
        self
    }
    pub fn emotion_strength(mut self, strength: f64) -> Self {
        self.emotion_strength = Some(strength);
        self
    }
    pub fn emotion_voice(mut self, voice: impl Into<String>) -> Self {
        self.emotion_voice = Some(voice.into());
        self
    }
    pub fn emotion_random(mut self, random: bool) -> Self {
        self.emotion_random = Some(random);
        self
    }
    pub fn to_generation_params(&self) -> GenerationParams {
        let mut params = GenerationParams::new();
        if let Some(language) = &self.language {
            params.insert("language", ParamValue::Text(language.clone()));
        }
        if let Some(vector) = self.emotion_vector {
            for ((name, _, _), value) in EMOTIONS.iter().zip(vector) {
                params.insert(*name, ParamValue::from(value));
            }
        }
        if let Some(strength) = self.emotion_strength {
            params.insert("emo_alpha", ParamValue::Float(strength));
        }
        if let Some(voice) = &self.emotion_voice {
            params.insert("emo_voice", ParamValue::Text(voice.clone()));
        }
        if let Some(random) = self.emotion_random {
            params.insert("emo_use_random", ParamValue::Bool(random));
        }
        params
    }
}

pub(super) struct Resolved {
    pub language: String,
    pub options: Options,
    pub emotion_voice: Option<String>,
}

/// Reject ambiguous controls rather than silently replacing another source.
pub(super) fn resolve(
    params: &GenerationParams,
    style: Option<&str>,
    seed: u64,
) -> Result<Resolved, BackendError> {
    params
        .validate(&catalog())
        .map_err(BackendError::Unsupported)?;
    let language = match params.get("language") {
        Some(ParamValue::Text(value)) if value == "zh" || value == "en" => value.as_str(),
        None => "zh",
        _ => {
            return Err(BackendError::Unsupported(
                "IndexTTS language must be zh or en".into(),
            ));
        }
    };
    let style = style.map(str::trim).filter(|s| !s.is_empty());
    let explicit_vector = EMOTIONS
        .iter()
        .any(|(name, _, _)| params.get(name).is_some());
    let emotion_voice = match params.get("emo_voice") {
        Some(ParamValue::Text(value)) if !value.trim().is_empty() => Some(value.clone()),
        _ => None,
    };
    if usize::from(style.is_some())
        + usize::from(explicit_vector)
        + usize::from(emotion_voice.is_some())
        > 1
    {
        return Err(BackendError::Unsupported(
            "IndexTTS style, emo_* vector and emo_voice are mutually exclusive".into(),
        ));
    }
    let vector = if let Some(style) = style {
        let i = EMOTIONS.iter().position(|(_, en, zh)| style.eq_ignore_ascii_case(en) || style == *zh)
            .ok_or_else(|| BackendError::Unsupported("IndexTTS style expects happy/angry/sad/afraid/disgusted/melancholic/surprised/calm or 高兴/愤怒/悲伤/恐惧/厌恶/低落/惊讶/平静".into()))?;
        Some(std::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }))
    } else if explicit_vector {
        Some(std::array::from_fn(|i| {
            params
                .get(EMOTIONS[i].0)
                .and_then(ParamValue::as_f64)
                .unwrap_or(0.0) as f32
        }))
    } else {
        None
    };
    if params.get("emo_alpha").is_some() && vector.is_none() && emotion_voice.is_none() {
        return Err(BackendError::Unsupported(
            "emo_alpha requires style, an emotion vector or emo_voice".into(),
        ));
    }
    let random = matches!(params.get("emo_use_random"), Some(ParamValue::Bool(true)));
    if random && vector.is_none() {
        return Err(BackendError::Unsupported(
            "emo_use_random requires style or an emotion vector".into(),
        ));
    }
    Ok(Resolved {
        language: language.into(),
        emotion_voice,
        options: Options {
            seed: Some(seed),
            emo_vector: vector.map(|v| talechime_indextts::normalize_emo_vec(v, false)),
            emo_alpha: params
                .get("emo_alpha")
                .and_then(ParamValue::as_f64)
                .unwrap_or(1.0) as f32,
            emo_use_random: random,
            ..Default::default()
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_keep_original_model_options() -> Result<(), BackendError> {
        let resolved = resolve(&GenerationParams::new(), None, 7)?;
        let original = Options::default();
        assert_eq!(resolved.language, "zh");
        assert!(resolved.options.emo_vector.is_none() && resolved.emotion_voice.is_none());
        assert_eq!(resolved.options.emo_alpha, original.emo_alpha);
        assert_eq!(resolved.options.emo_use_random, original.emo_use_random);
        assert_eq!(resolved.options.seed, Some(7));
        Ok(())
    }

    #[test]
    fn named_style_and_vector_reach_the_model() -> Result<(), BackendError> {
        let named = resolve(&GenerationParams::new(), Some("悲伤"), 42)?;
        assert_eq!(
            named.options.emo_vector,
            Some([0., 0., 0.8, 0., 0., 0., 0., 0.])
        );
        let params = IndexTtsParams::default()
            .language("en")
            .emotion_vector([0.8, 0., 0.8, 0., 0., 0., 0., 0.])
            .emotion_strength(0.6)
            .emotion_random(true)
            .to_generation_params();
        let resolved = resolve(&params, None, 9)?;
        assert_eq!(resolved.language, "en");
        assert_eq!(
            resolved.options.emo_vector,
            Some([0.4, 0., 0.4, 0., 0., 0., 0., 0.])
        );
        assert_eq!(resolved.options.emo_alpha, 0.6);
        assert!(resolved.options.emo_use_random);
        assert_eq!(resolved.options.seed, Some(9));
        Ok(())
    }

    #[test]
    fn conflicting_unknown_and_invalid_controls_are_rejected() {
        let vector = IndexTtsParams::default()
            .emotion_vector([0.; 8])
            .to_generation_params();
        assert!(resolve(&vector, Some("sad"), 0).is_err());
        let reference = IndexTtsParams::default()
            .emotion_voice("custom:sad")
            .to_generation_params();
        assert!(resolve(&reference, Some("sad"), 0).is_err());
        assert!(resolve(&GenerationParams::new(), Some("anything invented"), 0).is_err());
        for value in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            let params = IndexTtsParams::default()
                .emotion_strength(value)
                .to_generation_params();
            assert!(resolve(&params, Some("sad"), 0).is_err());
        }
        let random = IndexTtsParams::default()
            .emotion_random(true)
            .to_generation_params();
        assert!(resolve(&random, None, 0).is_err());
        let mut params = GenerationParams::new();
        params.insert("emo_sad", ParamValue::Text("1".into()));
        assert!(resolve(&params, None, 0).is_err());
    }
}
