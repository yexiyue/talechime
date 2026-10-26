use talechime_core::{
    backend::{Backend, SegmentationRequest},
    text::{TextSegment, duration::DurationEstimator},
};
pub async fn collect(
    backend: &dyn Backend,
    text: &str,
    voice: &str,
) -> anyhow::Result<Vec<TextSegment>> {
    let duration = DurationEstimator::default();
    let mut offset = 0;
    let mut segments = Vec::new();
    while offset < text.len() {
        let remaining = &text[offset..];
        let Some(mut segment) = backend
            .next_segment(SegmentationRequest {
                text: remaining,
                voice,
                duration: &duration,
            })
            .await?
        else {
            break;
        };
        anyhow::ensure!(
            segment.start < segment.end && remaining.get(segment.start..segment.end).is_some(),
            "invalid source segment"
        );
        segment.start += offset;
        segment.end += offset;
        offset = segment.end;
        segments.push(segment);
    }
    Ok(segments)
}
