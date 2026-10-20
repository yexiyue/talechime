use super::*;
const AUDIO_MILLISECONDS: usize = 30_000;
const AUDIO_BYTES: usize = 16 * 1024 * 1024;
/// Pending plus active audio is limited to 30 seconds and 16 MiB.
#[derive(Clone)]
pub(super) struct Budget {
    milliseconds: Arc<Semaphore>,
    bytes: Arc<Semaphore>,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            milliseconds: Arc::new(Semaphore::new(AUDIO_MILLISECONDS)),
            bytes: Arc::new(Semaphore::new(AUDIO_BYTES)),
        }
    }
}
#[derive(Debug)]
pub(super) struct Packet {
    pub(super) audio: Pcm,
    pub(super) lease: Lease,
}

#[derive(Debug)]
pub(super) struct Lease {
    _time: Option<OwnedSemaphorePermit>,
    _bytes: Option<OwnedSemaphorePermit>,
    observation: Option<tracing::Span>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let budget = self
            .observation
            .as_ref()
            .and(self._time.as_ref())
            .map(|permit| permit.semaphore().clone());
        // Preserve the original permit release order before observing the new state.
        drop(self._time.take());
        drop(self._bytes.take());
        if let (Some(budget), Some(span)) = (budget, self.observation.take()) {
            span.in_scope(|| {
                let seconds = AUDIO_MILLISECONDS.saturating_sub(budget.available_permits()) as f64 / 1000.0;
                tracing::info!(target:"talechime_perf",event="audio_budget_reserved_seconds",seconds);
            });
        }
    }
}

impl Budget {
    pub(super) fn validate(audio: &Pcm) -> Result<(u32, u32), SessionError> {
        let duration = audio.duration_ms()?;
        let bytes = audio
            .samples
            .len()
            .checked_mul(size_of::<f32>())
            .and_then(|bytes| u32::try_from(bytes).ok())
            .ok_or_else(|| SessionError::Invalid("PCM memory overflow".into()))?;
        if duration as usize > AUDIO_MILLISECONDS || bytes as usize > AUDIO_BYTES {
            return Err(SessionError::Invalid(
                "single block exceeds audio prefetch budget".into(),
            ));
        }
        Ok((duration, bytes))
    }
    pub(super) async fn acquire(&self, audio: Pcm) -> Result<Packet, SessionError> {
        let (duration, bytes) = Self::validate(&audio)?;
        let time = self
            .milliseconds
            .clone()
            .acquire_many_owned(duration)
            .await
            .map_err(|_| SessionError::Disconnected)?;
        let observation = tracing::enabled!(target:"talechime_perf",tracing::Level::INFO)
            .then(tracing::Span::current);
        let mut lease = Lease {
            _time: Some(time),
            _bytes: None,
            observation,
        };
        if lease.observation.is_some() {
            let seconds = AUDIO_MILLISECONDS.saturating_sub(self.milliseconds.available_permits())
                as f64
                / 1000.0;
            tracing::info!(target:"talechime_perf",event="audio_budget_reserved_seconds",seconds);
        }
        lease._bytes = Some(
            self.bytes
                .clone()
                .acquire_many_owned(bytes)
                .await
                .map_err(|_| SessionError::Disconnected)?,
        );
        Ok(Packet { audio, lease })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio() -> Pcm {
        Pcm {
            samples: vec![0.0; 1000],
            sample_rate: 1000,
            channels: 1,
        }
    }

    #[tokio::test]
    async fn dropping_packet_releases_original_time_and_byte_budgets() -> anyhow::Result<()> {
        let budget = Budget::default();
        let packet = budget.acquire(audio()).await?;
        assert_eq!(
            budget.milliseconds.available_permits(),
            AUDIO_MILLISECONDS - 1000
        );
        assert_eq!(budget.bytes.available_permits(), AUDIO_BYTES - 4000);
        drop(packet);
        assert_eq!(budget.milliseconds.available_permits(), AUDIO_MILLISECONDS);
        assert_eq!(budget.bytes.available_permits(), AUDIO_BYTES);
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_during_partial_reservation_returns_time_permits() -> anyhow::Result<()> {
        let budget = Budget::default();
        let held = budget
            .bytes
            .clone()
            .acquire_many_owned(AUDIO_BYTES as u32)
            .await?;
        let mut acquire = Box::pin(budget.acquire(audio()));
        tokio::select! {
            _ = &mut acquire => panic!("byte budget must block"),
            _ = tokio::task::yield_now() => {}
        }
        assert_eq!(
            budget.milliseconds.available_permits(),
            AUDIO_MILLISECONDS - 1000
        );
        drop(acquire);
        assert_eq!(budget.milliseconds.available_permits(), AUDIO_MILLISECONDS);
        drop(held);
        assert_eq!(budget.bytes.available_permits(), AUDIO_BYTES);
        Ok(())
    }
}
