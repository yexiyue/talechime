//! Bounded codec history and transactional PCM chunk decoding.
//! Each chunk still decodes its full retained context; this is not a KV/conv cache.

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::AudioBuffer;

const DECODER_LEFT_CONTEXT_FRAMES: usize = 25;

pub(super) struct StreamingDecoder {
    pub(super) frames: Vec<Tensor>,
    context: Option<Tensor>,
}

impl StreamingDecoder {
    pub(super) fn new(context: Option<Tensor>) -> Self {
        Self {
            frames: Vec::new(),
            context,
        }
    }

    pub(super) fn decode(
        &mut self,
        decode: impl FnOnce(&Tensor) -> Result<AudioBuffer>,
    ) -> Result<AudioBuffer> {
        let context_len = self.context.as_ref().map_or(0, |codes| codes.dims()[0]);
        let codes = decoder_codes(&self.frames, self.context.as_ref())?;
        let total_frames = codes.dim(2)?;
        let mut audio = decode(&codes)?;
        if context_len > 0 {
            let cut = context_len * audio.samples.len() / total_frames;
            audio.samples.drain(..cut);
        }
        // Keep pending frames and history intact if packing or decoding fails.
        self.context = Some(decoder_context_tail(&codes)?);
        self.frames.clear();
        Ok(audio)
    }
}

/// Retain recent reference/generated codes in frame-major layout for the next block.
fn decoder_context_tail(codes: &Tensor) -> Result<Tensor> {
    let frames = codes.dim(2)?;
    let keep = frames.min(DECODER_LEFT_CONTEXT_FRAMES);
    Ok(codes
        .narrow(2, frames - keep, keep)?
        .squeeze(0)?
        .transpose(0, 1)?
        .contiguous()?)
}

/// Pack device-resident frames into the decoder's `[1, 16, T]` layout.
fn decoder_codes(frames: &[Tensor], reference: Option<&Tensor>) -> Result<Tensor> {
    let generated = Tensor::stack(frames, 0)?.to_dtype(DType::I64)?;
    anyhow::ensure!(
        generated.dims2()?.1 == 16,
        "codec frames must contain 16 codes"
    );
    let combined = match reference {
        Some(reference) => Tensor::cat(&[&reference.to_dtype(DType::I64)?, &generated], 0)?,
        None => generated,
    };
    Ok(combined.transpose(0, 1)?.unsqueeze(0)?.contiguous()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthesis::codes_to_tensor;
    use candle_core::Device;

    #[test]
    fn failed_decode_preserves_pending_frames_and_history_for_retry() -> Result<()> {
        let reference = Tensor::new(&[[1u32; 16], [2; 16]], &Device::Cpu)?;
        let mut decoder = StreamingDecoder::new(Some(reference));
        decoder.frames.push(Tensor::new(&[3u32; 16], &Device::Cpu)?);
        assert!(
            decoder
                .decode(|_| anyhow::bail!("decoder failure"))
                .is_err()
        );
        assert_eq!(decoder.frames.len(), 1);
        assert_eq!(decoder.context.as_ref().unwrap().dim(0)?, 2);
        let audio = decoder.decode(|codes| {
            assert_eq!(codes.dims(), &[1, 16, 3]);
            Ok(AudioBuffer::new(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 24000))
        })?;
        assert_eq!(audio.samples, vec![3.0, 3.0]);
        assert!(decoder.frames.is_empty());
        assert_eq!(decoder.context.as_ref().unwrap().dim(0)?, 3);
        Ok(())
    }

    #[test]
    fn short_final_chunk_outputs_only_new_samples_and_advances_history() -> Result<()> {
        let mut decoder = StreamingDecoder::new(None);
        let mut samples = Vec::new();
        for range in [0u32..10, 10..12] {
            for frame in range {
                decoder
                    .frames
                    .push(Tensor::new(&[frame; 16], &Device::Cpu)?);
            }
            let audio = decoder.decode(|codes| {
                let values = codes.to_vec3::<i64>()?[0][0]
                    .iter()
                    .flat_map(|&frame| [frame as f32; 2])
                    .collect();
                Ok(AudioBuffer::new(values, 24000))
            })?;
            samples.extend(audio.samples);
        }
        let expected: Vec<f32> = (0..12).flat_map(|frame| [frame as f32; 2]).collect();
        assert_eq!(samples, expected);
        assert!(decoder.frames.is_empty());
        assert_eq!(decoder.context.as_ref().unwrap().dims(), &[12, 16]);
        Ok(())
    }

    #[test]
    fn decoder_history_advances_across_chunk_boundaries() -> Result<()> {
        let device = Device::Cpu;
        let reference = Tensor::new((0..44u32).map(|v| vec![v; 16]).collect::<Vec<_>>(), &device)?;
        let mut context = reference;
        for start in [44, 54, 64, 74] {
            let frames = (start..start + 10)
                .map(|v| Tensor::new(&[v; 16], &device))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let packed = decoder_codes(&frames, Some(&context))?;
            assert_eq!(packed.dim(2)?, context.dim(0)? + 10);
            context = decoder_context_tail(&packed)?;
            let expected: Vec<Vec<i64>> = (start + 10 - 25..start + 10)
                .map(|v| vec![i64::from(v); 16])
                .collect();
            assert_eq!(context.to_vec2::<i64>()?, expected);
        }
        let first_frames = [Tensor::new(&[0u32; 16], &device)?];
        let packed = decoder_codes(&first_frames, None)?;
        assert_eq!(decoder_context_tail(&packed)?.dims(), &[1, 16]);
        Ok(())
    }

    #[test]
    fn device_frame_packing_preserves_reference_and_codebook_order() -> Result<()> {
        let device = Device::Cpu;
        let reference = Tensor::new(&[[0u32; 16], [1u32; 16]], &device)?;
        let frames = [Tensor::from_vec(
            (2u32..18).collect::<Vec<_>>(),
            16,
            &device,
        )?];
        let actual = decoder_codes(&frames, Some(&reference))?;
        let expected = codes_to_tensor(&[vec![0; 16], vec![1; 16], (2..18).collect()], &device)?;
        assert_eq!(actual.dims(), &[1, 16, 3]);
        assert_eq!(
            actual.flatten_all()?.to_vec1::<i64>()?,
            expected.flatten_all()?.to_vec1::<i64>()?
        );
        assert_eq!(decoder_codes(&frames, None)?.dims(), &[1, 16, 1]);
        let malformed = [Tensor::new(&[0u32; 15], &device)?];
        assert!(decoder_codes(&malformed, None).is_err());
        Ok(())
    }
}
