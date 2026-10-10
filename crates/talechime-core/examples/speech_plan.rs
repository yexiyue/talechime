//! Build a plan without a model, runtime, audio device or user configuration.
use talechime_core::{PlaybackPolicy, SourceSnapshot, SpeechPlan, SpeechSpan, VoiceSnapshot};
use tts_protocol::{Capabilities, SourceId, TextRange, text_hash};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A real caller obtains these capabilities from its prepared backend.
    let caps = Capabilities {
        backend: "fixture".into(),
        model: Some("offline".into()),
        model_name: "Offline example".into(),
        compiled_devices: vec![],
        default_voice: "narrator".into(),
        voice_names: Default::default(),
        voices: vec!["narrator".into(), "actor".into()],
        native_streaming: false,
        style: false,
        cloning: false,
        pronunciation: false,
        continuation: false,
    };
    let text = "他说：走吧。";
    let source = SourceSnapshot::new(
        SourceId {
            namespace: "example".into(),
            book: "book".into(),
            chapter: "one".into(),
        },
        text,
        &text_hash(text),
    )?;
    let voices = VoiceSnapshot::new("fixture", Some("offline"), &caps, caps.voices.clone())?;
    let mut plan = SpeechPlan::new(source, voices, PlaybackPolicy::AfterChapterReady);
    plan.append(vec![SpeechSpan::new(
        TextRange { start: 0, end: 9 },
        "narrator",
        None,
    )])?;
    // A later accepted analysis result extends the same immutable source.
    plan.append(vec![SpeechSpan::new(
        TextRange {
            start: 9,
            end: text.len(),
        },
        "actor",
        None,
    )])?;
    plan.seal()?;
    for span in plan.spans() {
        let range = span.range();
        println!(
            "{}: {}",
            span.voice(),
            &plan.source().text()[range.start..range.end]
        );
    }
    println!(
        "{:?}: {} bytes accepted (no audio generated)",
        plan.state(),
        plan.accepted_end()
    );
    Ok(())
}
