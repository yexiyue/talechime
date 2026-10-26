//! Preserve the concrete stream and backpressure while propagating request context.
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use talechime_core::backend::{
    Backend, BackendError, Segmentation, SegmentationRequest, Streaming,
};
use tracing::Instrument;

pub const MAX_SEGMENT_RANGES: usize = 1024;

pub struct Observed {
    pub inner: Rc<dyn Backend>,
    pub parent: tracing::Span,
    pub calls: Cell<u64>,
    /// This probe executes one full-text voice assignment; requests are source suffixes.
    pub source_len: usize,
    pub production_ranges: Rc<RefCell<Vec<talechime_protocol::TextRange>>>,
}

impl Backend for Observed {
    fn capabilities(&self) -> talechime_protocol::Capabilities {
        self.inner.capabilities()
    }
    fn cache_identity(&self) -> Option<String> {
        self.inner.cache_identity()
    }
    fn stream<'a>(&'a self, request: talechime_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        let id = self.calls.get();
        self.calls.set(id + 1);
        let span = tracing::info_span!(target:"talechime_perf",parent:&self.parent,"segment_request",call=id);
        Box::pin(self.inner.stream(request).instrument(span))
    }
    fn next_segment<'a>(&'a self, request: SegmentationRequest<'a>) -> Segmentation<'a> {
        Box::pin(async move {
            let segment = self.inner.next_segment(request).await?;
            if let Some(segment) = &segment {
                let offset = self
                    .source_len
                    .checked_sub(request.text.len())
                    .ok_or_else(|| {
                        BackendError::Synthesis("segmentation request exceeds source".into())
                    })?;
                if segment.start >= segment.end
                    || request.text.get(segment.start..segment.end).is_none()
                {
                    return Err(BackendError::Synthesis(
                        "invalid production source range".into(),
                    ));
                }
                let mut ranges = self.production_ranges.borrow_mut();
                if ranges.len() >= MAX_SEGMENT_RANGES {
                    return Err(BackendError::Synthesis(
                        "segment evidence budget exceeded (1024 per request)".into(),
                    ));
                }
                ranges.push(talechime_protocol::TextRange {
                    start: offset + segment.start,
                    end: offset + segment.end,
                });
            }
            Ok(segment)
        })
    }

    fn paragraph_end(&self, text: &str, remaining: &str) -> bool {
        self.inner.paragraph_end(text, remaining)
    }
}
