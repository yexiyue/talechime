use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A byte range in the exact UTF-8 text submitted by the client.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextRange {
    pub start: usize,
    pub end: usize,
}

impl TextRange {
    /// Whether both endpoints form a valid half-open range in this snapshot.
    pub fn is_valid(self, text: &str) -> bool {
        self.start <= self.end
            && self.end <= text.len()
            && text.is_char_boundary(self.start)
            && text.is_char_boundary(self.end)
    }
}

/// Stable source identity; namespaces keep CLI files separate from reader books.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceId {
    pub namespace: String,
    pub book: String,
    pub chapter: String,
}

/// Requested synthesis execution policy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    #[default]
    Auto,
    Cpu,
    Coreml,
    Metal,
    Cuda,
}

/// Model-independent user preferences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub volume: f32,
    pub speed: f32,
    pub voice: String,
    pub auto_play: bool,
    pub backend: String,
    /// None selects the backend's default model.
    pub model: Option<String>,
    pub style: Option<String>,
    pub revision: u64,
    pub tts_device: Device,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            volume: 1.0,
            speed: 1.0,
            voice: "Weiguo".into(),
            auto_play: false,
            backend: "moss".into(),
            model: None,
            style: None,
            revision: 0,
            tts_device: Device::Auto,
            extra: BTreeMap::new(),
        }
    }
}

/// Only changed fields are sent, so unrelated preferences can be preserved.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigPatch {
    pub backend: Option<String>,
    pub model: Option<String>,
    /// An empty string clears the current style.
    pub style: Option<String>,
    pub tts_device: Option<Device>,
    pub expected_revision: u64,
    pub volume: Option<f32>,
    pub speed: Option<f32>,
    pub voice: Option<String>,
    pub auto_play: Option<bool>,
}

impl ConfigPatch {
    /// A backend change resets model selection unless a new model is provided.
    pub fn target_model<'a>(&'a self, current: &'a Config) -> Option<&'a str> {
        self.model.as_deref().or_else(|| {
            if self
                .backend
                .as_ref()
                .is_some_and(|id| id != &current.backend)
            {
                None
            } else {
                current.model.as_deref()
            }
        })
    }
}

/// A capability description; callers must not assume all backends are alike.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    /// Devices included in this build; runtime availability is reported separately.
    #[serde(default)]
    pub compiled_devices: Vec<Device>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub model_name: String,
    #[serde(default)]
    pub default_voice: String,
    #[serde(default)]
    pub voice_names: BTreeMap<String, String>,
    pub backend: String,
    pub voices: Vec<String>,
    pub native_streaming: bool,
    pub style: bool,
    pub cloning: bool,
    pub pronunciation: bool,
}

impl Capabilities {
    /// Omitted model IDs select the first (legacy) entry for that backend.
    pub fn matches(&self, backend: &str, model: Option<&str>) -> bool {
        self.backend == backend && model.is_none_or(|id| self.model.as_deref() == Some(id))
    }
}

/// Commands sent by the parent. Session controls require an envelope session ID.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum Command {
    Hello,
    GetStatus,
    GetConfig,
    UpdateConfig(ConfigPatch),
    PrepareModel,
    CancelPrepare,
    /// Start one validated single- or multi-voice plan.
    Start(Box<PlanRequest>),
    Append {
        spans: Vec<VoiceSpan>,
    },
    Seal,
    FailInput {
        message: String,
    },
    GetProgress,
    Pause,
    Resume,
    Stop,
    Seek {
        byte: usize,
        new_session_id: String,
    },
    Shutdown,
}

/// Command envelope with an opaque id for response correlation and deduplication.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Request {
    pub protocol_version: u32,
    pub request_id: String,
    pub session_id: Option<String>,
    #[serde(flatten)]
    pub command: Command,
}

/// User-observable state; generation and actual playback are distinct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Idle,
    Preparing,
    Generating,
    Buffering,
    Playing,
    Paused,
    Stopped,
    Failed,
}

/// A terminal reason. Only Completed is eligible for automatic next chapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndReason {
    Completed,
    Cancelled,
    Failed,
}

/// Structured errors stay actionable without requiring stderr parsing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub code: String,
    pub stage: String,
    pub message: String,
    pub retryable: bool,
}

/// Messages sent by the worker. Accepted does not mean synthesis has completed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum Event {
    Ready(Vec<Capabilities>),
    Accepted,
    Verification(crate::VerificationReport),
    /// Requested snapshot, independent of generation/playback completion.
    Progress(PlanProgressSnapshot),
    Config(Config),
    ConfigChanged(Config),
    ResourceState {
        stage: String,
        resource: String,
    },
    ModelProgress {
        resource: String,
        downloaded: u64,
        total: u64,
    },
    ModelReady,
    DeviceStatus {
        component: String,
        compiled: Vec<Device>,
        available: Vec<Device>,
        selected: Device,
        reason: Option<String>,
    },
    SessionState {
        state: SessionState,
    },
    BufferStatus {
        buffered_ms: u64,
        target_ms: u64,
        underruns: u32,
    },
    SegmentStarted {
        range: TextRange,
        text_hash: String,
    },
    SegmentFinished {
        range: TextRange,
        text_hash: String,
    },
    SessionEnded {
        reason: EndReason,
        text_hash: String,
    },
    Error(ErrorInfo),
}

/// Response or async event; only responses have request_id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub protocol_version: u32,
    pub instance_id: String,
    pub session_id: Option<String>,
    pub sequence: u64,
    pub request_id: Option<String>,
    #[serde(flatten)]
    pub event: Event,
}

/// Explicit plan playback strategy, independent of incremental input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanPlayback {
    Streaming,
    AfterChapterReady,
}

/// Proposed transport assignment. Domain validation happens before acceptance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VoiceSpan {
    pub range: TextRange,
    pub voice: String,
    pub style: Option<String>,
}

/// Transport plan, not a validated executable plan. Identity is exact, including None model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanRequest {
    pub source: SourceId,
    pub text: String,
    pub text_hash: String,
    pub backend: String,
    pub model: Option<String>,
    pub voices: Vec<String>,
    pub spans: Vec<VoiceSpan>,
    pub sealed: bool,
    pub playback: PlanPlayback,
    pub resume_byte: Option<usize>,
    pub restore_checkpoint: bool,
    #[serde(default)]
    pub verification: crate::VerificationOptions,
}

/// Input completion is distinct from generation and playback completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanInputState {
    Open,
    Sealed,
    Failed,
}

/// Contiguous source boundaries queried from the execution owner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanProgressSnapshot {
    pub input_state: PlanInputState,
    pub accepted_end: usize,
    pub generated_end: usize,
    pub played_end: usize,
    pub waiting_for_input: bool,
    pub chapter_ready: bool,
}

/// Maximum assignments per append command.
pub const MAX_PLAN_BATCH_SPANS: usize = 4096;
/// Maximum accepted assignments per execution at the transport boundary.
pub const MAX_PLAN_SPANS: usize = 65536;
