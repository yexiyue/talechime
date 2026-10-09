//! One local, append-only input owner shared by controls and the serial producer.
use super::*;
use std::cell::RefCell;
use tokio::sync::Notify;

/// Playback and recovery settings independent of a plan's voice assignments.
#[derive(Debug, Clone)]
pub struct PlanSessionOptions {
    /// Output volume, 0..=10.
    pub volume: f32,
    /// Playback rate, 0.5..=2.
    pub speed: f32,
    /// Explicit safe source byte to resume from, within the accepted prefix.
    pub resume_byte: Option<usize>,
    /// Restore actual-playback progress if no explicit resume byte is supplied.
    pub restore_checkpoint: bool,
    /// Private execution storage limits for AfterChapterReady.
    pub staging: StagingOptions,
}

impl Default for PlanSessionOptions {
    fn default() -> Self {
        Self {
            volume: 1.0,
            speed: 1.0,
            resume_byte: None,
            restore_checkpoint: false,
            staging: StagingOptions::default(),
        }
    }
}

/// Read-only local progress; generated input is never a playback checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanProgress {
    /// Input may remain open even after its current prefix has been generated.
    pub input_state: PlanState,
    /// Accepted contiguous source prefix.
    pub accepted_end: usize,
    /// Fully generated contiguous prefix, including explicitly skipped whitespace.
    pub generated_end: usize,
    /// Safe actual-playback/resume boundary, not a per-character audio estimate.
    pub played_end: usize,
    /// Producer is waiting for another assignment or for input to be sealed.
    pub waiting_for_input: bool,
    /// Entire chapter successfully generated and verified on disk.
    pub chapter_ready: bool,
}

pub(super) struct PlanInput {
    pub(super) plan: RefCell<SpeechPlan>,
    pub(super) changed: Notify,
    pub(super) generated: Cell<usize>,
    pub(super) waiting: Cell<bool>,
    pub(super) ready: Cell<bool>,
}

impl PlanInput {
    pub(super) fn new(plan: SpeechPlan, byte: usize) -> Self {
        Self {
            plan: RefCell::new(plan),
            changed: Notify::new(),
            generated: Cell::new(byte),
            waiting: Cell::new(false),
            ready: Cell::new(false),
        }
    }

    pub(super) async fn next(&self, index: usize) -> Option<SpeechSpan> {
        loop {
            // notify_one retains a permit if append/seal occurs before awaiting.
            let changed = self.changed.notified();
            {
                let plan = self.plan.borrow();
                if let Some(span) = plan.spans().get(index) {
                    self.waiting.set(false);
                    return Some(span.clone());
                }
                if plan.state() != PlanState::Open {
                    self.waiting.set(false);
                    return None;
                }
            }
            self.waiting.set(true);
            changed.await;
        }
    }
}
