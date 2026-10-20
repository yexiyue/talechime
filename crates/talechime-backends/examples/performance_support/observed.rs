//! Preserve the concrete stream and backpressure while propagating request context.
use std::{cell::Cell, rc::Rc};
use tracing::Instrument;
use tts_core::backend::{Backend, Segmentation, Streaming};

pub struct Observed {
    pub inner: Rc<dyn Backend>,
    pub parent: tracing::Span,
    pub calls: Cell<u64>,
}

impl Backend for Observed {
    fn capabilities(&self) -> tts_protocol::Capabilities {
        self.inner.capabilities()
    }
    fn cache_identity(&self) -> Option<String> {
        self.inner.cache_identity()
    }
    fn stream<'a>(&'a self, request: tts_core::backend::SegmentRequest<'a>) -> Streaming<'a> {
        let id = self.calls.get();
        self.calls.set(id + 1);
        let span = tracing::info_span!(target:"talechime_perf",parent:&self.parent,"segment_request",call=id);
        Box::pin(self.inner.stream(request).instrument(span))
    }
    fn next_segment<'a>(
        &'a self,
        text: &'a str,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<
                        Option<tts_core::text::TextSegment>,
                        tts_core::backend::BackendError,
                    >,
                > + 'a,
        >,
    > {
        self.inner.next_segment(text)
    }
    fn segments<'a>(&'a self, text: &'a str) -> Segmentation<'a> {
        self.inner.segments(text)
    }
    fn select_voice(&self, voice: &str) {
        self.inner.select_voice(voice);
    }
    fn paragraph_end(&self, text: &str, remaining: &str) -> bool {
        self.inner.paragraph_end(text, remaining)
    }
    fn observe_duration(&self, text: &str, voice: &str, seconds: f64) {
        self.inner.observe_duration(text, voice, seconds);
    }
}
