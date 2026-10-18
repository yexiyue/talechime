//! Per-generation acoustic code decoding state.
use super::CodePredictor;
use crate::models::kv_cache::AnyKVCache;
use anyhow::Result;
use candle_core::Tensor;

/// Decodes the acoustic codes of every frame for exactly one generation.
///
/// Owns the code predictor KV caches and, on CUDA, the optional frame graph.
/// Full synthesis creates one per call and streaming keeps one per stream, so
/// caches and captured device addresses are never shared between requests.
/// The predictor and its weights must outlive the session.
pub struct CodePredictorSession<'a> {
    predictor: &'a CodePredictor,
    // Declared before `caches`: a graph must be destroyed before the buffers
    // whose addresses it captured are released.
    #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
    graph: Option<super::graph::AcousticGraph>,
    caches: Vec<AnyKVCache>,
}

impl<'a> CodePredictorSession<'a> {
    pub fn new(predictor: &'a CodePredictor) -> Self {
        Self {
            predictor,
            #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
            graph: super::graph::AcousticGraph::new(&predictor.device),
            caches: predictor.new_kv_caches(),
        }
    }

    /// Acoustic codes for one frame: `[num_code_groups - 1]` `U32` on device.
    pub fn generate(&mut self, talker_hidden: &Tensor, semantic_embed: &Tensor) -> Result<Tensor> {
        #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
        if let Some(graph) = &mut self.graph
            && let Some(codes) = graph.try_generate(
                self.predictor,
                &mut self.caches,
                talker_hidden,
                semantic_embed,
            )?
        {
            return Ok(codes);
        }
        self.predictor
            .generate_acoustic_codes(talker_hidden, semantic_embed, &mut self.caches)
    }

    #[cfg(test)]
    pub fn replaying(&self) -> bool {
        #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
        {
            self.graph
                .as_ref()
                .is_some_and(super::graph::AcousticGraph::active)
        }
        #[cfg(not(all(feature = "cuda", any(target_os = "windows", target_os = "linux"))))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::code_predictor::CodePredictorConfig;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    fn tiny_predictor(device: &Device) -> CodePredictor {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let config = CodePredictorConfig {
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            vocab_size: 64,
            num_code_groups: 4,
            ..Default::default()
        };
        CodePredictor::new(config, vb).expect("tiny code predictor")
    }

    fn frame(predictor: &CodePredictor, seed: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let dim = predictor.codec_embed_dim();
        let hidden = Tensor::from_vec(
            (0..dim)
                .map(|i| ((i + seed * 7) as f32 * 0.31).sin())
                .collect::<Vec<_>>(),
            (1, 1, dim),
            device,
        )?;
        let semantic = Tensor::from_vec(
            (0..dim)
                .map(|i| ((i + seed * 13 + 1) as f32 * 0.17).cos())
                .collect::<Vec<_>>(),
            (1, 1, dim),
            device,
        )?;
        Ok((hidden, semantic))
    }

    #[test]
    fn session_matches_direct_eager_across_frames() -> Result<()> {
        let device = Device::Cpu;
        let predictor = tiny_predictor(&device);
        let mut session = CodePredictorSession::new(&predictor);
        let mut direct = predictor.new_kv_caches();
        assert!(!session.replaying());
        for seed in 0..5 {
            let (hidden, semantic) = frame(&predictor, seed, &device)?;
            let expected = predictor.generate_acoustic_codes(&hidden, &semantic, &mut direct)?;
            let actual = session.generate(&hidden, &semantic)?;
            assert_eq!(actual.dtype(), DType::U32);
            assert_eq!(actual.dims(), &[predictor.config.num_code_groups - 1]);
            // The session must reset its caches exactly like a direct caller.
            assert_eq!(actual.to_vec1::<u32>()?, expected.to_vec1::<u32>()?);
        }
        Ok(())
    }
}
