//! Backend-declared generation parameters and their wire values.
//!
//! Capabilities carry one [`ParameterSpec`] per supported parameter; start
//! requests carry one [`ParamValue`] per parameter the host actually sets.
//! Values outside the declared kind or range are rejected explicitly by the
//! session layer, never silently ignored.

use serde::{Deserialize, Serialize};

/// A generation parameter value supplied by the host.
///
/// The untagged shape keeps hand-written JSON natural: `{"steps": 32}`,
/// `{"language": "english"}`. Whether a value fits a parameter is decided by
/// the backend's [`ParameterSpec`], not by the JSON shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

impl ParamValue {
    /// Stable type name for diagnostics.
    pub fn type_name(&self) -> &'static str {
        match self {
            ParamValue::Bool(_) => "bool",
            ParamValue::Int(_) => "int",
            ParamValue::Float(_) => "float",
            ParamValue::Text(_) => "text",
        }
    }

    /// Short value preview for diagnostics; long text is truncated.
    pub fn preview(&self) -> String {
        match self {
            ParamValue::Bool(v) => v.to_string(),
            ParamValue::Int(v) => v.to_string(),
            ParamValue::Float(v) => v.to_string(),
            ParamValue::Text(v) => {
                let mut short = v.clone();
                short.truncate(30);
                if short.len() < v.len() {
                    short.push('…');
                }
                short
            }
        }
    }
}

/// The accepted type and range of one declared parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParamKind {
    Bool,
    Int { min: i64, max: i64 },
    Float { min: f64, max: f64 },
    Text { max_len: usize },
}

impl ParamKind {
    /// Accepted range description, e.g. `int in 2..=128`.
    fn describe(&self) -> String {
        match self {
            ParamKind::Bool => "bool".into(),
            ParamKind::Int { min, max } => format!("int in {min}..={max}"),
            ParamKind::Float { min, max } => format!("float in {min}..={max}"),
            ParamKind::Text { max_len } => format!("text of at most {max_len} bytes"),
        }
    }

    /// Whether the value fits this kind and range. Integers widen to floats;
    /// every other mismatch is rejected so hosts see explicit errors.
    pub fn accepts(&self, value: &ParamValue) -> bool {
        match (self, value) {
            (ParamKind::Bool, ParamValue::Bool(_)) => true,
            (ParamKind::Int { min, max }, ParamValue::Int(v)) => v >= min && v <= max,
            (ParamKind::Float { min, max }, ParamValue::Float(v)) => {
                v.is_finite() && *v >= *min && *v <= *max
            }
            // Numeric widening only: `1` is an acceptable `1.0`.
            (ParamKind::Float { min, max }, ParamValue::Int(v)) => {
                let widened = *v as f64;
                widened.is_finite() && widened >= *min && widened <= *max
            }
            (ParamKind::Text { max_len }, ParamValue::Text(v)) => v.len() <= *max_len,
            _ => false,
        }
    }
}

/// One declared generation parameter: accepted values, default and purpose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterSpec {
    pub name: String,
    pub kind: ParamKind,
    pub default: ParamValue,
    /// Short human-readable purpose; hosts may display it directly.
    pub description: String,
}

impl ParameterSpec {
    /// Validate one value against this declaration.
    pub fn validate(&self, value: &ParamValue) -> Result<(), String> {
        if self.kind.accepts(value) {
            Ok(())
        } else {
            Err(format!(
                "parameter '{}' expects {}, got {} ({})",
                self.name,
                self.kind.describe(),
                value.type_name(),
                value.preview()
            ))
        }
    }

    /// The declared default must satisfy its own kind.
    pub fn has_valid_default(&self) -> bool {
        self.kind.accepts(&self.default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(kind: ParamKind) -> ParameterSpec {
        ParameterSpec {
            name: "demo".into(),
            kind,
            default: ParamValue::Int(10),
            description: "demo".into(),
        }
    }

    #[test]
    fn int_kind_accepts_range_only() {
        let spec = spec(ParamKind::Int { min: 2, max: 128 });
        assert!(spec.validate(&ParamValue::Int(10)).is_ok());
        assert!(spec.validate(&ParamValue::Int(1)).is_err());
        assert!(spec.validate(&ParamValue::Int(129)).is_err());
        assert!(spec.validate(&ParamValue::Float(10.0)).is_err());
    }

    #[test]
    fn float_kind_widens_ints_but_not_text() {
        let spec = spec(ParamKind::Float { min: 0.5, max: 2.0 });
        assert!(spec.validate(&ParamValue::Float(1.5)).is_ok());
        assert!(spec.validate(&ParamValue::Int(1)).is_ok());
        assert!(spec.validate(&ParamValue::Int(3)).is_err());
        assert!(spec.validate(&ParamValue::Text("1.5".into())).is_err());
    }

    #[test]
    fn untagged_wire_shape_round_trips() {
        let value: ParamValue =
            serde_json::from_value(serde_json::json!({"steps": 32})["steps"].clone()).unwrap();
        assert_eq!(value, ParamValue::Int(32));
        let value: ParamValue =
            serde_json::from_value(serde_json::json!({"speed": 1.5})["speed"].clone()).unwrap();
        assert_eq!(value, ParamValue::Float(1.5));
        let kind: ParamKind = serde_json::from_value(serde_json::json!({
            "type": "int", "min": 2, "max": 128
        }))
        .unwrap();
        assert_eq!(
            kind,
            ParamKind::Int { min: 2, max: 128 },
            "tagged kind shape must stay stable for hosts"
        );
    }
}
