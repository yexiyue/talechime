//! Declared VoxCPM generation parameters and their typed builder.
use tts_core::params::GenerationParams;
use tts_protocol::{ParamKind, ParamValue, ParameterSpec};
use voxcpm::Options;

/// One latent frame covers 2560 samples at 48 kHz (18.75 Hz).
const FRAME_HZ: f64 = 48_000.0 / 2560.0;
/// Playback-block budget caps any single segment at 30 seconds.
const MAX_DURATION_SECONDS: f64 = 30.0;
const DEFAULT_MAX_FRAMES: usize = 200;

pub(super) fn catalog() -> Vec<ParameterSpec> {
    vec![
        ParameterSpec {
            name: "steps".into(),
            kind: ParamKind::Int { min: 2, max: 128 },
            default: ParamValue::Int(10),
            description: "Flow-matching Euler 步数；更高更精细也更慢".into(),
        },
        ParameterSpec {
            name: "cfg".into(),
            kind: ParamKind::Float {
                min: 0.0,
                max: 10.0,
            },
            default: ParamValue::Float(2.0),
            description: "分类器自由引导强度".into(),
        },
        ParameterSpec {
            name: "temperature".into(),
            kind: ParamKind::Float { min: 0.0, max: 2.0 },
            default: ParamValue::Float(1.0),
            description: "初始 latent 噪声幅度".into(),
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
    ]
}

/// Typed builder for the declared VoxCPM parameters; unset fields use catalog defaults.
#[derive(Debug, Clone, Default)]
pub struct VoxCpmParams {
    pub steps: Option<u32>,
    pub cfg: Option<f64>,
    pub temperature: Option<f64>,
    pub max_duration_seconds: Option<f64>,
}

impl VoxCpmParams {
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = Some(steps);
        self
    }
    pub fn cfg(mut self, cfg: f64) -> Self {
        self.cfg = Some(cfg);
        self
    }
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }
    pub fn max_duration_seconds(mut self, seconds: f64) -> Self {
        self.max_duration_seconds = Some(seconds);
        self
    }
    /// Convert into the backend-agnostic parameter map.
    pub fn to_generation_params(&self) -> GenerationParams {
        let mut params = GenerationParams::new();
        if let Some(steps) = self.steps {
            params.insert("steps", ParamValue::Int(i64::from(steps)));
        }
        if let Some(cfg) = self.cfg {
            params.insert("cfg", ParamValue::Float(cfg));
        }
        if let Some(temperature) = self.temperature {
            params.insert("temperature", ParamValue::Float(temperature));
        }
        if let Some(seconds) = self.max_duration_seconds {
            params.insert("max_duration", ParamValue::Float(seconds));
        }
        params
    }
}

/// Resolve validated parameters plus the attempt seed into native options.
pub(super) fn resolve(params: &GenerationParams, seed: u64) -> Options {
    let value = |name: &str| params.get(name).cloned();
    let float = |name: &str, fallback: f64| {
        match value(name) {
            Some(ParamValue::Float(v)) => v,
            // Session validation already widened integer literals to floats.
            Some(ParamValue::Int(v)) => v as f64,
            _ => fallback,
        }
    };
    let max_duration = float("max_duration", DEFAULT_MAX_FRAMES as f64 / FRAME_HZ);
    Options {
        seed,
        steps: match value("steps") {
            Some(ParamValue::Int(v)) => v as usize,
            _ => 10,
        },
        cfg: float("cfg", 2.0),
        temperature: float("temperature", 1.0),
        max_frames: (max_duration * FRAME_HZ).round().max(1.0) as usize,
        measure_modules: false,
    }
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
    fn typed_builder_round_trips_through_the_map() {
        let typed = VoxCpmParams::default()
            .steps(64)
            .cfg(1.5)
            .temperature(0.8)
            .max_duration_seconds(12.0);
        let params = typed.to_generation_params();
        assert!(params.validate(&catalog()).is_ok());
        let options = resolve(&params, 7);
        assert_eq!(options.seed, 7);
        assert_eq!(options.steps, 64);
        assert_eq!(options.cfg, 1.5);
        assert_eq!(options.temperature, 0.8);
        assert_eq!(options.max_frames, (12.0 * FRAME_HZ).round() as usize);
    }

    #[test]
    fn empty_params_keep_previous_defaults_except_seed() {
        let options = resolve(&GenerationParams::new(), 99);
        assert_eq!(options.seed, 99);
        assert_eq!(options.steps, 10);
        assert_eq!(options.cfg, 2.0);
        assert_eq!(options.temperature, 1.0);
        assert_eq!(options.max_frames, DEFAULT_MAX_FRAMES);
    }
}
