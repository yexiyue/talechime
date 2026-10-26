//! Voice design engine contract: a text description in, a reference recording
//! out. Engines are registered like backends; hosts and the facade never
//! depend on concrete implementations.

use std::{future::Future, path::Path, pin::Pin};
use talechime_protocol::{Device, Event};
use tokio::sync::mpsc;

pub type Preparing<'a> = Pin<Box<dyn Future<Output = anyhow::Result<()>> + 'a>>;
pub type Designing<'a> = Pin<Box<dyn Future<Output = anyhow::Result<DesignOutput>> + 'a>>;

/// Input limits an engine enforces before any weights load.
#[derive(Clone, Copy, Debug)]
pub struct InputConstraints {
    pub text_max_chars: usize,
    /// `None` limits bytes only.
    pub description_max_chars: Option<usize>,
    pub description_max_bytes: usize,
    pub forbid_parentheses: bool,
}
impl InputConstraints {
    pub fn validate(&self, text: &str, description: &str) -> anyhow::Result<()> {
        anyhow::ensure!(
            !text.trim().is_empty() && text.chars().count() <= self.text_max_chars,
            "design reference text must contain 1..{} characters",
            self.text_max_chars
        );
        anyhow::ensure!(
            !description.trim().is_empty() && !description.contains('\0'),
            "provide a voice description without NUL characters"
        );
        if let Some(max) = self.description_max_chars {
            anyhow::ensure!(
                description.chars().count() <= max,
                "description requires at most {max} characters"
            );
        }
        anyhow::ensure!(
            description.len() <= self.description_max_bytes,
            "provide a description up to {} bytes",
            self.description_max_bytes
        );
        if self.forbid_parentheses {
            anyhow::ensure!(
                !description.contains(['(', ')', '（', '）']),
                "description must not contain parentheses"
            );
        }
        Ok(())
    }
}

/// What a host can discover about one design engine without preparing it.
#[derive(Clone, Debug)]
pub struct EngineInfo {
    pub id: &'static str,
    /// Display name; safe for automatic-tier error messages.
    pub name: &'static str,
    /// Devices the engine can run on right now (compiled and available).
    pub devices: Vec<Device>,
    pub inputs: InputConstraints,
    /// Bytes this engine still needs to download before designing; `0` when ready.
    pub extra_download_bytes: u64,
    pub ready: bool,
    /// The design capability rides on the narration model itself.
    pub reuses_synthesis_model: bool,
}

#[derive(Clone, Debug)]
pub struct DesignRequest {
    /// Spoken trial text recorded into the reference.
    pub text: String,
    pub description: String,
    pub device: Device,
    pub seed: Option<u64>,
}

/// A complete, validated design product; WAV bytes are stored verbatim.
#[derive(Clone, Debug)]
pub struct DesignOutput {
    pub wav: Vec<u8>,
}

/// One design engine. Implementations must honor:
/// - input constraints before preparing or loading weights;
/// - cancellation by dropping the returned future: compute stops and inference
///   resources are released without producing a partial asset.
pub trait VoiceDesigner {
    fn id(&self) -> &'static str;
    fn info(&self, root: &Path) -> EngineInfo;
    fn prepare<'a>(&'a self, root: &'a Path, progress: mpsc::Sender<Event>) -> Preparing<'a>;
    fn design<'a>(&'a self, root: &'a Path, request: DesignRequest) -> Designing<'a>;
}

#[cfg(test)]
mod tests {
    use super::*;
    fn constraints() -> InputConstraints {
        InputConstraints {
            text_max_chars: 100,
            description_max_chars: Some(200),
            description_max_bytes: 4096,
            forbid_parentheses: true,
        }
    }
    #[test]
    fn input_constraints_reject_bad_requests() {
        let limits = constraints();
        assert!(limits.validate("你好", "低沉温和的男声").is_ok());
        assert!(limits.validate("", "描述").is_err());
        assert!(limits.validate("文本", "  ").is_err());
        assert!(limits.validate("文本", "（描述）").is_err());
        let long: String = "字".repeat(201);
        assert!(limits.validate("文本", &long).is_err());
        let bytes = "a".repeat(5000);
        assert!(limits.validate("文本", &bytes).is_err());
    }
    #[test]
    fn byte_only_limits_accept_long_char_counts() {
        let limits = InputConstraints {
            description_max_chars: None,
            ..constraints()
        };
        let long: String = "字".repeat(1000);
        assert_eq!(long.len(), 3000);
        assert!(limits.validate("文本", &long).is_ok());
    }
}
