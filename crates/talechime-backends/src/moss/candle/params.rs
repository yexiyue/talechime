//! Realtime generation duration parameter and typed builder.
use talechime_core::params::GenerationParams;
use talechime_protocol::{ParamKind, ParamValue, ParameterSpec};

/// The pinned codec emits 1920 samples per frame at 24 kHz (12.5 Hz).
const FRAME_HZ: f64 = 24_000.0 / 1920.0;
const MAX_DURATION_SECONDS: f64 = 30.0;
const DEFAULT_MAX_FRAMES: usize = 375;

pub(super) fn catalog() -> Vec<ParameterSpec> {
    vec![ParameterSpec {
        name: "max_duration".into(),
        kind: ParamKind::Float {
            min: 1.0,
            max: MAX_DURATION_SECONDS,
        },
        default: ParamValue::Float(DEFAULT_MAX_FRAMES as f64 / FRAME_HZ),
        description: "单段最长生成秒数，受 30 秒音频块预算钳制".into(),
    }]
}

/// Typed builder for the declared MOSS parameters; unset fields use catalog defaults.
#[derive(Debug, Clone, Default)]
pub struct MossParams {
    pub max_duration_seconds: Option<f64>,
}

impl MossParams {
    pub fn max_duration_seconds(mut self, seconds: f64) -> Self {
        self.max_duration_seconds = Some(seconds);
        self
    }
    /// Convert into the backend-agnostic parameter map.
    pub fn to_generation_params(&self) -> GenerationParams {
        let mut params = GenerationParams::new();
        if let Some(seconds) = self.max_duration_seconds {
            params.insert("max_duration", ParamValue::Float(seconds));
        }
        params
    }
}

/// Resolved per-request settings for the inference thread.
#[derive(Debug)]
pub(super) struct Resolved {
    pub max_frames: usize,
}

pub(super) fn resolve(params: &GenerationParams) -> Resolved {
    let fallback = DEFAULT_MAX_FRAMES as f64 / FRAME_HZ;
    let max_duration = params
        .get("max_duration")
        .and_then(ParamValue::as_f64)
        .unwrap_or(fallback);
    Resolved {
        max_frames: (max_duration * FRAME_HZ)
            .floor()
            .clamp(1.0, DEFAULT_MAX_FRAMES as f64) as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duration_limit_matches_decoded_sample_count() {
        for seconds in [1.0, 12.0, 30.0] {
            let params = MossParams::default()
                .max_duration_seconds(seconds)
                .to_generation_params();
            let frames = resolve(&params).max_frames;
            let duration = (frames * 1920) as f64 / 24_000.0;
            assert!(duration <= seconds);
            assert!(seconds - duration < 1920.0 / 24_000.0);
        }
    }

    #[test]
    fn realtime_rejects_instruction_and_language() {
        for name in ["instruction", "language"] {
            let mut params = GenerationParams::new();
            params.insert(name, ParamValue::Text("value".into()));
            assert!(params.validate(&catalog()).is_err());
        }
        assert_eq!(
            resolve(&GenerationParams::new()).max_frames,
            DEFAULT_MAX_FRAMES
        );
    }
}
