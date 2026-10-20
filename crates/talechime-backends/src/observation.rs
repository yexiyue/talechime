//! Developer-only observations; normal builds retain the original channel behavior.
use tokio::sync::mpsc;
use tts_core::backend::{AudioChunk, BackendError};

#[inline]
pub(crate) fn send_audio(
    sender: &mpsc::Sender<Result<AudioChunk, BackendError>>,
    chunk: Result<AudioChunk, BackendError>,
) -> Result<(), mpsc::error::SendError<Result<AudioChunk, BackendError>>> {
    #[cfg(feature = "performance-tools")]
    let observation = SendObservation::new(&chunk);
    let result = sender.blocking_send(chunk);
    #[cfg(feature = "performance-tools")]
    observation.complete(result.is_ok());
    result
}

#[cfg(feature = "performance-tools")]
pub(crate) struct SendObservation {
    started: std::time::Instant,
    pcm: bool,
}

#[cfg(feature = "performance-tools")]
impl SendObservation {
    pub(crate) fn new(chunk: &Result<AudioChunk, BackendError>) -> Self {
        let pcm = matches!(chunk, Ok(AudioChunk::Pcm(_)));
        if pcm {
            tracing::info!(target:"talechime_perf",event="owner_pcm_ready");
            metrics::counter!("talechime.chunks","stage"=>"owner").increment(1);
        }
        Self {
            started: std::time::Instant::now(),
            pcm,
        }
    }
    pub(crate) fn complete(self, sent: bool) {
        let seconds = self.started.elapsed().as_secs_f64();
        metrics::histogram!("talechime.channel_wait_seconds","stage"=>"owner").record(seconds);
        tracing::info!(target:"talechime_perf",event="channel_wait",seconds,sent);
        if self.pcm && sent {
            tracing::info!(target:"talechime_perf",event="adapter_channel_enqueue");
        }
    }
}
