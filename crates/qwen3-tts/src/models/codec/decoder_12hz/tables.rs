//! Bounded immutable decoder constants; graphs pin the entries they read.
use anyhow::Result;
use candle_core::{DType, Tensor};
use std::{collections::VecDeque, sync::Arc};

pub(super) struct DecoderTables {
    pub(super) cos: Tensor,
    pub(super) sin: Tensor,
    pub(super) mask: Tensor,
}

#[derive(Default)]
pub(super) struct TableCache {
    entries: VecDeque<(usize, usize, Arc<DecoderTables>)>,
    bytes: usize,
}

const MAX_ENTRIES: usize = 4;
const MAX_BYTES: usize = 8 * 1024 * 1024;

impl TableCache {
    pub(super) fn get(
        &mut self,
        frames: usize,
        head_dim: usize,
        window: usize,
        inv_freq: &Tensor,
    ) -> Result<Arc<DecoderTables>> {
        if let Some(index) = self.entries.iter().position(|entry| entry.0 == frames) {
            let entry = self.entries.remove(index).expect("located entry");
            let tables = entry.2.clone();
            self.entries.push_back(entry);
            return Ok(tables);
        }
        let positions = Tensor::arange(0u32, u32::try_from(frames)?, inv_freq.device())?;
        let freqs = positions
            .to_dtype(DType::F32)?
            .unsqueeze(1)?
            .matmul(&inv_freq.unsqueeze(0)?)?;
        let tables = Arc::new(DecoderTables {
            cos: freqs.cos()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?,
            sin: freqs.sin()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?,
            mask: decoder_attention_mask(frames, window, inv_freq.device())?,
        });
        let bytes = frames
            .saturating_mul(frames.saturating_add(2 * head_dim))
            .saturating_mul(4);
        if bytes <= MAX_BYTES {
            while self.entries.len() >= MAX_ENTRIES || self.bytes + bytes > MAX_BYTES {
                let (_, removed, _) = self.entries.pop_front().expect("nonempty cache");
                self.bytes -= removed;
            }
            self.bytes += bytes;
            self.entries.push_back((frames, bytes, tables.clone()));
        }
        Ok(tables)
    }
}

pub(super) fn decoder_attention_mask(
    frames: usize,
    window: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    anyhow::ensure!(window > 0, "decoder attention window must be positive");
    let mut values = vec![0.0f32; frames * frames];
    for query in 0..frames {
        for key in 0..frames {
            if key > query || query - key >= window {
                values[query * frames + key] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(values, (frames, frames), device)?
        .unsqueeze(0)?
        .unsqueeze(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn eviction_is_bounded_and_does_not_invalidate_retained_tables() -> Result<()> {
        let inv = Tensor::new(&[1.0f32, 0.1, 0.01, 0.001], &Device::Cpu)?;
        let mut cache = TableCache::default();
        let pinned = cache.get(3, 8, 2, &inv)?;
        let before = pinned.mask.flatten_all()?.to_vec1::<f32>()?;
        for frames in 4..10 {
            cache.get(frames, 8, 2, &inv)?;
        }
        assert_eq!(cache.entries.len(), MAX_ENTRIES);
        assert_eq!(pinned.mask.flatten_all()?.to_vec1::<f32>()?, before);
        assert!(cache.bytes <= MAX_BYTES);
        let previous_bytes = cache.bytes;
        cache.get(1500, 8, 2, &inv)?;
        assert_eq!(
            cache.bytes, previous_bytes,
            "oversized tables must not be retained"
        );
        Ok(())
    }
}
