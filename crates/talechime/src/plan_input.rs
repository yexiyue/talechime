//! Transport conversion delegates all source/voice/range invariants to core.
use talechime::{
    Capabilities, PlaybackPolicy, SourceSnapshot, SpeechPlan, SpeechSpan, VoiceSnapshot,
};
use tts_protocol::{MAX_PLAN_BATCH_SPANS, MAX_PLAN_SPANS, PlanRequest, VoiceSpan};

pub fn spans(input: &[VoiceSpan]) -> Vec<SpeechSpan> {
    input
        .iter()
        .map(|span| SpeechSpan::new(span.range, span.voice.clone(), span.style.clone()))
        .collect()
}

pub fn build(input: &PlanRequest, caps: &Capabilities) -> anyhow::Result<SpeechPlan> {
    if input.voices.len() > 256
        || input.spans.len() > MAX_PLAN_SPANS
        || input.text.len() > tts_protocol::MAX_MESSAGE_BYTES
    {
        anyhow::bail!("plan transport limits exceeded");
    }
    let source = SourceSnapshot::new(input.source.clone(), input.text.clone(), &input.text_hash)?;
    let voices = VoiceSnapshot::new(
        &input.backend,
        input.model.as_deref(),
        caps,
        input.voices.clone(),
    )?;
    let policy = match input.playback {
        tts_protocol::PlanPlayback::Streaming => PlaybackPolicy::Streaming,
        tts_protocol::PlanPlayback::AfterChapterReady => PlaybackPolicy::AfterChapterReady,
    };
    let mut plan = SpeechPlan::new(source, voices, policy);
    if !input.spans.is_empty() {
        plan.append(spans(&input.spans))?;
    }
    if input.sealed {
        plan.seal()?;
    }
    Ok(plan)
}

pub fn check_batch(count: usize, accepted: usize) -> anyhow::Result<()> {
    if count > MAX_PLAN_BATCH_SPANS || accepted.saturating_add(count) > MAX_PLAN_SPANS {
        anyhow::bail!("plan assignment limit exceeded");
    }
    Ok(())
}

/// Transport generation settings; validation against the catalog stays in core.
pub fn generation(input: &PlanRequest) -> (talechime::GenerationParams, talechime::SeedPolicy) {
    let mut params = talechime::GenerationParams::new();
    for (name, value) in &input.params {
        params.insert(name.clone(), value.clone());
    }
    (params, talechime::SeedPolicy::from(input.seed))
}

pub fn progress(input: talechime::PlanProgress) -> tts_protocol::PlanProgressSnapshot {
    tts_protocol::PlanProgressSnapshot {
        input_state: match input.input_state {
            talechime::PlanState::Open => tts_protocol::PlanInputState::Open,
            talechime::PlanState::Sealed => tts_protocol::PlanInputState::Sealed,
            talechime::PlanState::Failed => tts_protocol::PlanInputState::Failed,
        },
        accepted_end: input.accepted_end,
        generated_end: input.generated_end,
        played_end: input.played_end,
        waiting_for_input: input.waiting_for_input,
        chapter_ready: input.chapter_ready,
    }
}
