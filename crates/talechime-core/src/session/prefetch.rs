use super::*;
/// Pending plus active audio is limited to 30 seconds and 16 MiB.
#[derive(Clone)]
pub(super) struct Budget {
    milliseconds: Arc<Semaphore>,
    bytes: Arc<Semaphore>,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            milliseconds: Arc::new(Semaphore::new(30_000)),
            bytes: Arc::new(Semaphore::new(16 * 1024 * 1024)),
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
    _time: OwnedSemaphorePermit,
    _bytes: OwnedSemaphorePermit,
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
        if duration > 30_000 || bytes > 16 * 1024 * 1024 {
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
        let bytes = self
            .bytes
            .clone()
            .acquire_many_owned(bytes)
            .await
            .map_err(|_| SessionError::Disconnected)?;
        Ok(Packet {
            audio,
            lease: Lease {
                _time: time,
                _bytes: bytes,
            },
        })
    }
}
