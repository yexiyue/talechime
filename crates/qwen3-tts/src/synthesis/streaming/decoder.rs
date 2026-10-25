//! Pending codes and transactional, session-owned causal decoder state.

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::{AudioBuffer, models::codec::decoder_12hz::DecoderState};

pub(super) struct StreamingDecoder {
    pub(super) frames: Vec<Tensor>,
    context: Option<Tensor>,
    state: DecoderState,
}

impl StreamingDecoder {
    pub(super) fn new(context: Option<Tensor>) -> Self {
        Self {
            frames: Vec::new(),
            context,
            state: DecoderState::default(),
        }
    }

    pub(super) fn decode(
        &mut self,
        decode: impl FnOnce(&Tensor, &mut DecoderState) -> Result<AudioBuffer>,
    ) -> Result<AudioBuffer> {
        let context_len = self.context.as_ref().map_or(0, |codes| codes.dims()[0]);
        let codes = decoder_codes(&self.frames, self.context.as_ref())?;
        let total_frames = codes.dim(2)?;
        let mut state = self.state.clone();
        let mut audio = decode(&codes, &mut state)?;
        if context_len > 0 {
            let cut = context_len * audio.samples.len() / total_frames;
            audio.samples.drain(..cut);
        }
        // Keep pending frames and history intact if packing or decoding fails.
        self.state = state;
        self.context = None;
        self.frames.clear();
        Ok(audio)
    }
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
                .decode(|_, _| anyhow::bail!("decoder failure"))
                .is_err()
        );
        assert_eq!(decoder.frames.len(), 1);
        assert_eq!(decoder.context.as_ref().unwrap().dim(0)?, 2);
        let audio = decoder.decode(|codes, _state| {
            assert_eq!(codes.dims(), &[1, 16, 3]);
            Ok(AudioBuffer::new(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0], 24000))
        })?;
        assert_eq!(audio.samples, vec![3.0, 3.0]);
        assert!(decoder.frames.is_empty());
        assert!(decoder.context.is_none());
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
            let audio = decoder.decode(|codes, _state| {
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
        assert!(decoder.context.is_none());
        Ok(())
    }

    #[test]
    #[ignore = "requires local model and saved codes; never downloads"]
    fn real_model_wrapper_preserves_reference_and_rolls_back_partial_decode() -> Result<()> {
        use crate::models::codec::Decoder12Hz;
        let directory = std::env::var("QWEN_CODEC_TEST_MODEL")?;
        let input = std::env::var("QWEN_CODEC_TEST_CODES")?;
        let reference: usize = std::env::var("QWEN_CODEC_TEST_REFERENCE_FRAMES")?.parse()?;
        let device = crate::parse_device(&std::env::var("QWEN_CODEC_TEST_DEVICE")?)?;
        let weights = candle_core::safetensors::load(
            std::path::Path::new(&directory).join("speech_tokenizer/model.safetensors"),
            &device,
        )?;
        let codec = Decoder12Hz::from_weights(&weights, Default::default())?;
        let frames: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(input)?)?;
        anyhow::ensure!(
            reference + 10 < frames.len(),
            "reference must leave multiple generated chunks"
        );
        let full = codec
            .decode(&codes_to_tensor(&frames, &device)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let context = Tensor::new(frames[..reference].to_vec(), &device)?;
        let mut decoder = StreamingDecoder::new(Some(context));
        for frame in &frames[reference..reference + 10] {
            decoder.frames.push(Tensor::new(frame.as_slice(), &device)?);
        }
        // Fail after all layer caches and convolution history have advanced.
        assert!(
            decoder
                .decode(|codes, state| {
                    codec.decode_incremental(codes, state)?;
                    anyhow::bail!("injected PCM conversion failure")
                })
                .is_err()
        );
        assert_eq!(decoder.frames.len(), 10);
        assert_eq!(decoder.context.as_ref().unwrap().dim(0)?, reference);
        let mut output = decoder
            .decode(|codes, state| {
                AudioBuffer::from_tensor(codec.decode_incremental(codes, state)?, 24000)
            })?
            .samples;
        for block in frames[reference + 10..].chunks(10) {
            for frame in block {
                decoder.frames.push(Tensor::new(frame.as_slice(), &device)?);
            }
            output.extend(
                decoder
                    .decode(|codes, state| {
                        AudioBuffer::from_tensor(codec.decode_incremental(codes, state)?, 24000)
                    })?
                    .samples,
            );
        }
        let expected = &full[reference * 1920..];
        assert_eq!(output.len(), expected.len());
        let rmse = (output
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / output.len() as f64)
            .sqrt();
        let maximum = output
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "wrapper reference={reference} samples={} rmse={rmse} max={maximum}",
            output.len()
        );
        anyhow::ensure!(
            rmse < 1e-5 && maximum < 1e-3,
            "wrapper/full mismatch after retry"
        );
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
