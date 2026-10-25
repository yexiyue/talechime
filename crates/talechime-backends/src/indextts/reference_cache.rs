//! One conditioning identity owned by one model thread, never a global voice cache.
use sha2::{Digest, Sha256};

pub(super) struct ReferenceCache {
    enabled: bool,
    identity: Option<[u8; 32]>,
}

impl ReferenceCache {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            identity: None,
        }
    }

    pub(super) fn prepare<E>(
        &mut self,
        samples: &[f32],
        prepare: impl FnOnce() -> Result<(), E>,
    ) -> Result<(), E> {
        if !self.enabled {
            return prepare();
        }
        // Hash the actual validated/resampled snapshot used below, not a path,
        // timestamp, or a separate file read that could race with replacement.
        let mut hash = Sha256::new();
        hash.update(b"indextts-reference-mono-22050-f32/1");
        hash.update((samples.len() as u64).to_le_bytes());
        for sample in samples {
            hash.update(sample.to_bits().to_le_bytes());
        }
        let identity: [u8; 32] = hash.finalize().into();
        if self.identity == Some(identity) {
            return Ok(());
        }
        // A failed preparation must not leave an identity claiming that the
        // model's possibly partially changed state is still the previous one.
        self.identity = None;
        prepare()?;
        self.identity = Some(identity);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reuse_replace_return_and_failed_prepare() {
        let mut cache = ReferenceCache::new(true);
        let mut prepared = 0;
        for samples in [&[0.1, 0.2][..], &[0.1, 0.2], &[0.2, 0.1], &[0.1, 0.2]] {
            cache
                .prepare(samples, || {
                    prepared += 1;
                    Ok::<_, ()>(())
                })
                .unwrap();
        }
        assert_eq!(prepared, 3);
        assert!(cache.prepare(&[0.3], || Err::<(), _>("failed")).is_err());
        cache
            .prepare(&[0.1, 0.2], || {
                prepared += 1;
                Ok::<_, ()>(())
            })
            .unwrap();
        assert_eq!(prepared, 4);
        cache
            .prepare(&[0.1, 0.2], || -> Result<(), ()> {
                panic!("hit must not prepare")
            })
            .unwrap();
    }

    #[test]
    fn disabled_and_bit_changes_never_hit() {
        let mut cache = ReferenceCache::new(false);
        let mut prepared = 0;
        for _ in 0..2 {
            cache
                .prepare(&[0.1], || {
                    prepared += 1;
                    Ok::<_, ()>(())
                })
                .unwrap();
        }
        assert_eq!(prepared, 2);
        let mut cache = ReferenceCache::new(true);
        cache.prepare(&[0.0], || Ok::<_, ()>(())).unwrap();
        assert!(
            cache
                .prepare(&[-0.0], || Err::<(), _>("different bits"))
                .is_err()
        );
        assert!(
            ReferenceCache::new(true)
                .prepare(&[0.0], || Err::<(), _>("different owner"))
                .is_err()
        );
    }
}
