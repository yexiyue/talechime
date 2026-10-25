//! Session-owned causal decoder state. Transformer keys/values retain their
//! original history; the stateless vocoder replays only its finite receptive field.
use super::Decoder12Hz;
use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};

#[derive(Clone, Default)]
pub(crate) struct DecoderState {
    offset: usize,
    quantized: Option<Tensor>,
    layers: Vec<LayerCache>,
    features: Option<Tensor>,
}

#[derive(Clone, Default)]
pub(super) struct LayerCache {
    kv: Option<(Tensor, Tensor)>,
}

impl LayerCache {
    pub(super) fn append(
        &mut self,
        key: Tensor,
        value: Tensor,
        window: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (key, value) = match &self.kv {
            Some((previous_key, previous_value)) => (
                Tensor::cat(&[previous_key, &key], 2)?,
                Tensor::cat(&[previous_value, &value], 2)?,
            ),
            None => (key, value),
        };
        let length = key.dim(2)?;
        let keep = length.min(window.saturating_sub(1));
        self.kv = Some((
            key.narrow(2, length - keep, keep)?.contiguous()?,
            value.narrow(2, length - keep, keep)?.contiguous()?,
        ));
        Ok((key, value))
    }
}

impl Decoder12Hz {
    /// The caller commits a cloned state only after PCM conversion succeeds.
    pub(crate) fn decode_incremental(
        &self,
        codes: &Tensor,
        state: &mut DecoderState,
    ) -> Result<Tensor> {
        let (batch, quantizers, count) = codes.dims3()?;
        ensure!(
            batch == 1 && quantizers == 16 && count > 0,
            "invalid streaming codec shape"
        );
        let quantized = self.quantize(codes)?;
        let (quantized, previous) = with_history(&state.quantized, &quantized)?;
        let hidden = self
            .pre_conv
            .forward(&quantized)?
            .narrow(2, previous, count)?;
        state.quantized = Some(tail(&quantized, self.pre_conv.input_context(0))?);
        let hidden = self.linear_3d(
            &hidden.transpose(1, 2)?,
            &self.input_proj_weight,
            Some(&self.input_proj_bias),
        )?;
        let mut hidden = self.transform_incremental(hidden, state)?;
        hidden = self.project_features(&hidden)?;
        let (features, context) = with_history(&state.features, &hidden)?;
        let waveform = self.vocode_incremental(&features)?;
        let upsample = self.total_upsample();
        ensure!(
            waveform.dim(2)? == (context + count) * upsample,
            "codec output length changed"
        );
        state.features = Some(tail(&features, self.vocoder_left_context())?);
        state.offset = state
            .offset
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("codec position overflow"))?;
        Ok(waveform
            .narrow(2, context * upsample, count * upsample)?
            .contiguous()?)
    }

    fn transform_incremental(
        &self,
        mut hidden: Tensor,
        state: &mut DecoderState,
    ) -> Result<Tensor> {
        let (batch, count, _) = hidden.dims3()?;
        let window = self.config.sliding_window;
        ensure!(window > 0, "decoder attention window must be positive");
        let end = state
            .offset
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("codec position overflow"))?;
        let positions = Tensor::arange(
            u32::try_from(state.offset)?,
            u32::try_from(end)?,
            hidden.device(),
        )?;
        let frequencies = positions
            .to_dtype(DType::F32)?
            .unsqueeze(1)?
            .matmul(&self.inv_freq.unsqueeze(0)?)?;
        let cos = frequencies
            .cos()?
            .repeat((1, 2))?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = frequencies
            .sin()?
            .repeat((1, 2))?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let mask = incremental_mask(count, state.offset.min(window - 1), window, hidden.device())?;
        if state.layers.is_empty() {
            state
                .layers
                .resize_with(self.transformer_weights.layers.len(), LayerCache::default);
        }
        ensure!(
            state.layers.len() == self.transformer_weights.layers.len(),
            "decoder state layer mismatch"
        );
        for (index, (layer, cache)) in self
            .transformer_weights
            .layers
            .iter()
            .zip(&mut state.layers)
            .enumerate()
        {
            hidden = self.run_layer(
                &hidden,
                layer,
                &cos,
                &sin,
                &mask,
                batch,
                count,
                index,
                Some(cache),
            )?;
        }
        Ok(hidden)
    }

    fn vocode_incremental(&self, features: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
        if features.device().is_cuda() {
            return self
                .graphs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .vocode(self, features);
        }
        self.vocode(features)
    }

    /// Reverse the loaded causal convolutions and transposed convolutions.
    /// For the bundled model this is 10 projected transformer frames.
    fn vocoder_left_context(&self) -> usize {
        let mut context = self.final_conv.input_context(0);
        for block in self.decoder_blocks.iter().rev() {
            context = block.input_context(context);
        }
        context = self.decoder_init_conv.input_context(context);
        for stage in self.upsample_stages.iter().rev() {
            context = stage
                .trans_conv
                .input_context(stage.convnext.input_context(context));
        }
        context
    }
}

fn with_history(history: &Option<Tensor>, current: &Tensor) -> Result<(Tensor, usize)> {
    match history {
        Some(history) => Ok((Tensor::cat(&[history, current], 2)?, history.dim(2)?)),
        None => Ok((current.clone(), 0)),
    }
}

fn tail(tensor: &Tensor, keep: usize) -> Result<Tensor> {
    let length = tensor.dim(2)?;
    let keep = keep.min(length);
    Ok(tensor.narrow(2, length - keep, keep)?.contiguous()?)
}

fn incremental_mask(
    count: usize,
    history: usize,
    window: usize,
    device: &Device,
) -> Result<Tensor> {
    ensure!(window > 0, "decoder attention window must be positive");
    let keys = history + count;
    let mut values = vec![0.0f32; count * keys];
    for query in 0..count {
        let position = history + query;
        for key in 0..keys {
            if key > position || position - key >= window {
                values[query * keys + key] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(Tensor::from_vec(values, (1, 1, count, keys), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_attention_mask_matches_full_window_after_eviction() -> Result<()> {
        for (offset, count) in [(0, 3), (3, 5), (71, 10), (72, 7), (137, 1)] {
            let history = offset.min(71);
            let full =
                super::super::tables::decoder_attention_mask(offset + count, 72, &Device::Cpu)?;
            let expected =
                full.narrow(2, offset, count)?
                    .narrow(3, offset - history, history + count)?;
            let actual = incremental_mask(count, history, 72, &Device::Cpu)?;
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    fn layer_cache_keeps_original_values_and_only_window_history() -> Result<()> {
        let mut cache = LayerCache::default();
        for start in (0..120).step_by(10) {
            let key = Tensor::from_vec(
                (start..start + 10).map(|v| v as f32).collect(),
                (1, 1, 10, 1),
                &Device::Cpu,
            )?;
            let (_, value) = cache.append(key.clone(), key, 72)?;
            let history = start.min(71);
            let expected: Vec<f32> = (start - history..start + 10).map(|v| v as f32).collect();
            assert_eq!(value.flatten_all()?.to_vec1::<f32>()?, expected);
            assert_eq!(cache.kv.as_ref().unwrap().0.dim(2)?, (start + 10).min(71));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires local model and saved codes; never downloads"]
    fn real_model_incremental_matches_full_across_window_and_vocoder_boundaries() -> Result<()> {
        let directory = std::env::var("QWEN_CODEC_TEST_MODEL")?;
        let input = std::env::var("QWEN_CODEC_TEST_CODES")?;
        let device = crate::parse_device(
            &std::env::var("QWEN_CODEC_TEST_DEVICE").unwrap_or_else(|_| "cpu".into()),
        )?;
        let weights = candle_core::safetensors::load(
            std::path::Path::new(&directory).join("speech_tokenizer/model.safetensors"),
            &device,
        )?;
        let decoder = Decoder12Hz::from_weights(&weights, Default::default())?;
        let frames: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(input)?)?;
        let codes = crate::synthesis::codes_to_tensor(&frames, &device)?;
        assert_eq!(decoder.vocoder_left_context(), 10);
        let full = decoder.decode(&codes)?.flatten_all()?.to_vec1::<f32>()?;
        for chunk in [1, 7, 10, 20, 73] {
            let mut state = DecoderState::default();
            let mut samples = Vec::new();
            for start in (0..frames.len()).step_by(chunk) {
                let count = chunk.min(frames.len() - start);
                let audio =
                    decoder.decode_incremental(&codes.narrow(2, start, count)?, &mut state)?;
                samples.extend(audio.flatten_all()?.to_vec1::<f32>()?);
                assert!(state.features.as_ref().unwrap().dim(2)? <= 10);
                assert!(state.quantized.as_ref().unwrap().dim(2)? <= 2);
                assert!(
                    state
                        .layers
                        .iter()
                        .all(|layer| layer.kv.as_ref().unwrap().0.dims()[2] <= 71)
                );
            }
            assert_eq!(samples.len(), full.len());
            let rmse = (samples
                .iter()
                .zip(&full)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / full.len() as f64)
                .sqrt();
            let maximum = samples
                .iter()
                .zip(&full)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("chunk={chunk} rmse={rmse} max={maximum}");
            ensure!(rmse < 1e-5 && maximum < 1e-3, "stream/full mismatch");
        }
        Ok(())
    }
}
