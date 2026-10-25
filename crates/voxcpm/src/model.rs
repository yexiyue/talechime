//! Request-local caches and cancellation boundaries around native Candle inference.
use crate::timing::{Timings, measure};
use crate::{
    acoustic::Acoustic, codec::Codec, tokenizer::Tokenizer, transformer::Transformer,
    weights::Weights,
};
use candle_core::{DType, Device, Result, Tensor};
use std::path::Path;

#[cfg(all(test, feature = "cuda", target_os = "windows"))]
mod cuda_decoder_tests;
#[cfg(all(test, feature = "cuda", target_os = "windows"))]
mod cuda_profile_tests;

#[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
use crate::cuda_graph::CfmGraphs;

/// Production output sample rate.
pub const SAMPLE_RATE: u32 = 48_000;
/// Generation termination; only EOS completes a playback checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Eos,
    Cancelled,
    Truncated,
}
/// Cached CUDA specializations available for subsequent unprofiled frames.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CudaGraphStatus {
    pub cfm: bool,
    pub decoder: bool,
}
/// Flow matching and autoregressive sampling parameters.
#[derive(Debug, Clone)]
pub struct Options {
    pub seed: u64,
    pub steps: usize,
    pub cfg: f64,
    pub temperature: f64,
    pub max_frames: usize,
    /// Synchronize at module boundaries to measure costs; affects throughput.
    pub measure_modules: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            seed: 42,
            steps: 10,
            cfg: 2.0,
            temperature: 1.0,
            max_frames: 200,
            measure_modules: false,
        }
    }
}
/// Latent reference patches, with optional transcript for continuation.
pub struct Reference {
    pub features: Tensor,
    pub transcript: Option<String>,
}
/// Experimental isolated identity prefix plus an optional spoken continuation.
#[cfg(feature = "experimental-dual-reference")]
pub struct DualReference<'a> {
    pub identity: &'a Reference,
    pub continuation: Option<&'a Reference>,
}
/// All weights use one Candle device. No runtime subprocess or native TTS engine.
pub struct Model {
    pub tokenizer: Tokenizer,
    /// Graphs must be destroyed before the weights and state they reference.
    #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
    graphs: Option<CfmGraphs>,
    #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
    decoder_graph: Option<crate::cuda_graph::decoder::DecoderGraph>,
    embedding: Tensor,
    base: Transformer,
    residual: Transformer,
    acoustic: Acoustic,
    codec: Codec,
    device: Device,
    dtype: DType,
    backbone_dtype: DType,
    codec_dtype: DType,
    timings: Timings,
}

struct GenerateRequest<'a> {
    text: &'a str,
    reference: Option<&'a Reference>,
    identity: Option<&'a Reference>,
    options: &'a Options,
}
impl Model {
    /// Load the existing two GGUF files from their current model directory.
    pub fn load(directory: &Path, device: &Device) -> Result<Self> {
        Self::load_with_cancel(directory, device, &|| false)
    }
    /// Cancellation is checked before every weight transfer during initialization.
    pub fn load_with_cancel(
        directory: &Path,
        device: &Device,
        cancel: &impl Fn() -> bool,
    ) -> Result<Self> {
        if cancel() {
            candle_core::bail!("model loading cancelled");
        }
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            DType::F16
        };
        let mut base = Weights::open(
            &directory.join("VoxCPM2-BaseLM-Q8_0.gguf"),
            device,
            DType::F32,
        )?
        .with_cancel(cancel);
        base.expect_string("general.architecture", "minicpm4")?;
        for (name, expected) in [
            ("embedding_length", 2048),
            ("block_count", 28),
            ("feed_forward_length", 6144),
            ("vocab_size", 73448),
            ("attention.head_count", 16),
            ("attention.head_count_kv", 2),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
        ] {
            base.expect_u32(&format!("minicpm4.{name}"), expected)?;
        }
        base.expect_string("tokenizer.ggml.model", "llama")?;
        let tokenizer = Tokenizer::load(&base)?;
        let embedding = base.tensor("token_embd.weight")?;
        let rope = base.tensor("rope_factors_short.weight")?;
        let transformer = Transformer::load(
            &mut base,
            "",
            28,
            "output_norm.weight",
            Some(rope.clone()),
            true,
        )?;
        let mut acoustic =
            Weights::open(&directory.join("VoxCPM2-Acoustic-F16.gguf"), device, dtype)?
                .with_cancel(cancel);
        acoustic.expect_string("general.architecture", "voxcpm-acoustic")?;
        if acoustic.value("voxcpm.model_version")?.to_f32()? != 2.0 {
            candle_core::bail!("unsupported VoxCPM acoustic model version");
        }
        for (name, expected) in [
            ("patch_size", 4),
            ("feat_dim", 64),
            ("residual_lm.n_layer", 8),
            ("residual_lm.n_embd", 2048),
            ("locenc.n_layer", 12),
            ("locenc.n_embd", 1024),
            ("locdit.n_layer", 12),
            ("locdit.n_embd", 1024),
            ("audiovae.sample_rate", 16000),
            ("audiovae.out_sample_rate", 48000),
        ] {
            acoustic.expect_u32(&format!("voxcpm.{name}"), expected)?;
        }
        let residual = Transformer::load(
            &mut acoustic,
            "residual_lm.",
            8,
            "residual_lm.output_norm.weight",
            None,
            true,
        )?;
        let network = Acoustic::load(&mut acoustic, &rope)?;
        let codec = Codec::load(&mut acoustic)?;
        Ok(Self {
            tokenizer,
            embedding,
            base: transformer,
            residual,
            acoustic: network,
            codec,
            device: device.clone(),
            dtype,
            backbone_dtype: DType::F32,
            codec_dtype: dtype,
            timings: Timings::default(),
            #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
            graphs: CfmGraphs::new(device),
            #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
            decoder_graph: crate::cuda_graph::decoder::DecoderGraph::for_device(device, true),
        })
    }
    /// Load original OpenBMB weights without quantization. CPU requires F32;
    /// GPU accepts BF16 or F16. AudioVAE follows the official F32 path.
    pub fn load_original(
        directory: &Path,
        device: &Device,
        dtype: DType,
        cancel: &impl Fn() -> bool,
    ) -> Result<Self> {
        if cancel() {
            candle_core::bail!("model loading cancelled");
        }
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
            || (device.is_cpu() && dtype != DType::F32)
        {
            candle_core::bail!("original weights require CPU F32 or GPU F32/F16/BF16");
        }
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        validate_original_config(&config)?;
        let tokenizer = Tokenizer::from_file(&directory.join("tokenizer.json"))?;
        let mut weights = Weights::native(&directory.join("model.safetensors"), device, dtype)?
            .with_cancel(cancel);
        let embedding = weights.tensor("token_embd.weight")?;
        let factors: Vec<f32> =
            serde_json::from_value(config["lm_config"]["rope_scaling"]["short_factor"].clone())
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        if factors.len() != 64 || factors.iter().any(|f| !f.is_finite() || *f <= 0.0) {
            candle_core::bail!("original RoPE requires 64 finite positive factors");
        }
        let rope = Tensor::from_vec(factors, 64, device)?;
        let base = Transformer::load(
            &mut weights,
            "",
            28,
            "output_norm.weight",
            Some(rope.clone()),
            true,
        )?;
        let residual = Transformer::load(
            &mut weights,
            "residual_lm.",
            8,
            "residual_lm.output_norm.weight",
            None,
            true,
        )?;
        let acoustic = Acoustic::load(&mut weights, &rope)?;
        let vae = if directory.join("audiovae.safetensors").exists() {
            directory.join("audiovae.safetensors")
        } else {
            directory.join("audiovae.pth")
        };
        let mut vae = Weights::native(&vae, device, DType::F32)?.with_cancel(cancel);
        let codec = Codec::load(&mut vae)?;
        Ok(Self {
            tokenizer,
            embedding,
            base,
            residual,
            acoustic,
            codec,
            device: device.clone(),
            dtype,
            backbone_dtype: dtype,
            codec_dtype: DType::F32,
            timings: Timings::default(),
            #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
            graphs: CfmGraphs::new(device),
            #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
            decoder_graph: crate::cuda_graph::decoder::DecoderGraph::for_device(
                device,
                cfg!(target_os = "windows") && dtype == DType::BF16,
            ),
        })
    }
    /// Measurements for the latest request when module timing is enabled.
    pub fn timings(&self) -> &Timings {
        &self.timings
    }
    /// Whether the per-frame CFM step is running as a replayed CUDA graph.
    ///
    /// False on CPU and Metal, while a CUDA model is still warming up, and
    /// after a capture or replay failure has forced the model back to the
    /// eager path for the rest of its lifetime.
    pub fn cuda_graph_replay(&self) -> bool {
        self.cuda_graph_status().cfm
    }

    /// Graph availability by module; all fields are false on CPU and Metal.
    pub fn cuda_graph_status(&self) -> CudaGraphStatus {
        #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
        return CudaGraphStatus {
            cfm: self.graphs.as_ref().is_some_and(CfmGraphs::active),
            decoder: self
                .decoder_graph
                .as_ref()
                .is_some_and(crate::cuda_graph::decoder::DecoderGraph::active),
        };
        #[cfg(not(all(feature = "cuda", any(target_os = "windows", target_os = "linux"))))]
        CudaGraphStatus::default()
    }
    /// Encode mono 16kHz samples. Padding follows the official reference mode.
    pub fn reference(
        &mut self,
        samples: &[f32],
        transcript: Option<String>,
        cancel: &impl Fn() -> bool,
    ) -> Result<Reference> {
        if transcript
            .as_ref()
            .is_some_and(|text| text.trim().is_empty())
        {
            candle_core::bail!("reference transcript must not be empty");
        }
        if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
            candle_core::bail!("reference audio must contain finite samples");
        }
        if cancel() {
            candle_core::bail!("generation cancelled");
        }
        let padding = (2560 - samples.len() % 2560) % 2560;
        let mut padded = Vec::with_capacity(samples.len() + padding);
        if transcript.is_some() {
            padded.resize(padding, 0.0);
            padded.extend_from_slice(samples);
        } else {
            padded.extend_from_slice(samples);
            padded.resize(samples.len() + padding, 0.0);
        }
        let x = Tensor::from_vec(padded, (1, 1, samples.len() + padding), &self.device)?
            .to_dtype(self.codec_dtype)?;
        let encoded = self.codec.encode(&x, cancel)?;
        let frames = encoded.dim(2)?;
        let features = encoded
            .reshape((64, frames / 4, 4))?
            .transpose(0, 1)?
            .transpose(1, 2)?
            .contiguous()?;
        let features = features.to_dtype(self.dtype)?;
        Ok(Reference {
            features,
            transcript,
        })
    }
    /// Generate bounded patches and immediately deliver finite 48kHz PCM.
    pub fn generate(
        &mut self,
        text: &str,
        reference: Option<&Reference>,
        options: &Options,
        cancel: &impl Fn() -> bool,
        mut pcm: impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<Outcome> {
        self.generate_conditioned(text, reference, None, options, cancel, &mut pcm)
    }
    /// Experimental upstream ref_continuation mode; identity has right padding,
    /// continuation has left padding and a matching spoken transcript.
    #[cfg(feature = "experimental-dual-reference")]
    pub fn generate_dual(
        &mut self,
        text: &str,
        references: DualReference<'_>,
        options: &Options,
        cancel: &impl Fn() -> bool,
        mut pcm: impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<Outcome> {
        if references.identity.transcript.is_some()
            || references
                .continuation
                .is_some_and(|r| r.transcript.is_none())
        {
            candle_core::bail!(
                "dual reference requires isolated identity and transcribed continuation"
            );
        }
        self.generate_conditioned(
            text,
            references.continuation,
            Some(references.identity),
            options,
            cancel,
            &mut pcm,
        )
    }
    fn generate_conditioned(
        &mut self,
        text: &str,
        reference: Option<&Reference>,
        identity: Option<&Reference>,
        options: &Options,
        cancel: &impl Fn() -> bool,
        pcm: &mut impl FnMut(&[f32]) -> Result<()>,
    ) -> Result<Outcome> {
        self.timings = Timings::default();
        if options.steps < 2
            || options.steps > 128
            || options.max_frames == 0
            || options.max_frames > 4096
            || !options.cfg.is_finite()
            || options.cfg < 0.0
            || !options.temperature.is_finite()
            || options.temperature < 0.0
        {
            candle_core::bail!("invalid generation options");
        }
        if cancel() {
            return Ok(Outcome::Cancelled);
        }
        let result = self.generate_inner(
            GenerateRequest {
                text,
                reference,
                identity,
                options,
            },
            cancel,
            pcm,
            &mut |_| Ok(()),
        );
        self.codec.reset();
        if cancel() {
            Ok(Outcome::Cancelled)
        } else {
            result
        }
    }
    fn generate_inner(
        &mut self,
        request: GenerateRequest<'_>,
        cancel: &impl Fn() -> bool,
        pcm: &mut impl FnMut(&[f32]) -> Result<()>,
        patch_capture: &mut impl FnMut(&Tensor) -> Result<()>,
    ) -> Result<Outcome> {
        let GenerateRequest {
            text,
            reference,
            identity,
            options: o,
        } = request;
        if text.trim().is_empty() {
            candle_core::bail!("generation text is empty");
        }
        if text.len() > 16_384 {
            candle_core::bail!("text exceeds bounded context");
        }
        let prefill_start = if o.measure_modules {
            self.device.synchronize()?;
            Some(std::time::Instant::now())
        } else {
            None
        };
        self.codec.reset();
        let joined = match reference.and_then(|r| r.transcript.as_ref()) {
            Some(t) => format!("{t}{text}"),
            None => text.to_owned(),
        };
        let mut ids = self.tokenizer.encode_with_cancel(&joined, cancel)?;
        ids.push(101);
        let text_embedding = self
            .embedding
            .index_select(&Tensor::from_vec(ids.clone(), ids.len(), &self.device)?, 0)?
            .unsqueeze(0)?;
        let zeros = Tensor::zeros((1, ids.len(), 2048), self.dtype, &self.device)?;
        let mut audio_embedding = zeros.clone();
        let mut input = text_embedding;
        let mut prefix = Tensor::zeros((1, 64, 4), self.dtype, &self.device)?;
        let mut audio_range = None;
        if let Some(r) = reference {
            let (n, patch, width) = r.features.dims3()?;
            if n == 0
                || patch != 4
                || width != 64
                || !r.features.device().same_device(&self.device)
                || r.features.dtype() != self.dtype
            {
                candle_core::bail!("reference features do not match this model/device");
            }
            let n = r.features.dim(0)?;
            let encoded = self.acoustic.encode(&r.features, cancel)?.unsqueeze(0)?;
            if r.transcript.is_some() {
                input = Tensor::cat(&[input, encoded.to_dtype(self.backbone_dtype)?], 1)?;
                audio_embedding = Tensor::cat(&[zeros, encoded], 1)?;
                audio_range = Some((ids.len(), n));
                prefix = r.features.narrow(0, n - 1, 1)?.transpose(1, 2)?;
            } else {
                let marks = self
                    .embedding
                    .index_select(&Tensor::new(&[103u32, 104], &self.device)?, 0)?
                    .unsqueeze(0)?;
                input = Tensor::cat(
                    &[
                        marks.narrow(1, 0, 1)?,
                        encoded.to_dtype(self.backbone_dtype)?,
                        marks.narrow(1, 1, 1)?,
                        input,
                    ],
                    1,
                )?;
                audio_embedding = Tensor::cat(
                    &[
                        Tensor::zeros((1, 1, 2048), self.dtype, &self.device)?,
                        encoded,
                        Tensor::zeros((1, ids.len() + 1, 2048), self.dtype, &self.device)?,
                    ],
                    1,
                )?;
                audio_range = Some((1, n));
            }
        }
        let mut audio_ranges: Vec<_> = audio_range.into_iter().collect();
        if let Some(identity) = identity {
            let (n, patch, width) = identity.features.dims3()?;
            if n == 0
                || patch != 4
                || width != 64
                || !identity.features.device().same_device(&self.device)
                || identity.features.dtype() != self.dtype
            {
                candle_core::bail!("identity features do not match this model/device");
            }
            let encoded = self
                .acoustic
                .encode(&identity.features, cancel)?
                .unsqueeze(0)?;
            let marks = self
                .embedding
                .index_select(&Tensor::new(&[103u32, 104], &self.device)?, 0)?
                .unsqueeze(0)?;
            input = Tensor::cat(
                &[
                    marks.narrow(1, 0, 1)?,
                    encoded.to_dtype(self.backbone_dtype)?,
                    marks.narrow(1, 1, 1)?,
                    input,
                ],
                1,
            )?;
            audio_embedding = Tensor::cat(
                &[
                    Tensor::zeros((1, 1, 2048), self.dtype, &self.device)?,
                    encoded,
                    Tensor::zeros((1, 1, 2048), self.dtype, &self.device)?,
                    audio_embedding,
                ],
                1,
            )?;
            for (start, _) in &mut audio_ranges {
                *start += n + 2;
            }
            audio_ranges.insert(0, (1, n));
        }
        let context = input.dim(1)?;
        if context + o.max_frames > 4096 {
            candle_core::bail!("reference/text exceeds bounded context");
        }
        let mut base_cache = self.base.cache(context + o.max_frames);
        let mut res_cache = self.residual.cache(context + o.max_frames);
        let full = self
            .base
            .forward(&input, Some(&mut base_cache), cancel)?
            .to_dtype(self.dtype)?;
        let full = quantize_audio_ranges(&full, &audio_ranges, |audio| self.acoustic.fsq(audio))?;
        let residual_input = self
            .acoustic
            .fusion
            .forward(&Tensor::cat(&[&full, &audio_embedding], 2)?)?;
        let residual = self
            .residual
            .forward(&residual_input, Some(&mut res_cache), cancel)?;
        let mut lm = full.narrow(1, context - 1, 1)?.squeeze(1)?;
        let mut res = residual.narrow(1, context - 1, 1)?.squeeze(1)?;
        if let Some(start) = prefill_start {
            self.device.synchronize()?;
            self.timings.prefill = start.elapsed();
        }
        #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
        if let Some(graph) = &mut self.decoder_graph {
            graph.reset();
        }
        let mut noise = Noise(o.seed);
        for i in 0..o.max_frames {
            if cancel() {
                return Ok(Outcome::Cancelled);
            }
            let mu = Tensor::cat(
                &[
                    self.acoustic.lm_to_dit.forward(&lm)?,
                    self.acoustic.res_to_dit.forward(&res)?,
                ],
                1,
            )?;
            let values: Vec<f32> = (0..256)
                .map(|_| noise.normal() * o.temperature as f32)
                .collect();
            let initial = Tensor::from_vec(values, (1, 64, 4), &self.device)?;
            let (patch, elapsed) = measure(o.measure_modules, &self.device, || {
                // Module timing synchronizes inside the measured region, which
                // a captured graph cannot contain, so profiling stays eager.
                #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
                if let Some(graphs) = self.graphs.as_mut()
                    && !o.measure_modules
                {
                    return graphs.sample(
                        &mut self.acoustic,
                        &mu,
                        &prefix,
                        &initial,
                        o.steps,
                        o.cfg,
                        cancel,
                    );
                }
                self.acoustic
                    .sample(&mu, &prefix, &initial, o.steps, o.cfg, cancel)
            })?;
            self.timings.cfm += elapsed;
            patch_capture(&patch)?;
            #[cfg(all(
                test,
                feature = "cuda",
                any(target_os = "windows", target_os = "linux")
            ))]
            crate::attention::cuda_tests::capture_patch(&patch)?;
            let (decoded, elapsed) = measure(o.measure_modules, &self.device, || {
                #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
                if let Some(graph) = &mut self.decoder_graph
                    && !o.measure_modules
                {
                    return graph.decode(
                        &mut self.codec,
                        &patch.to_dtype(self.codec_dtype)?,
                        cancel,
                    );
                }
                self.codec
                    .decode(&patch.to_dtype(self.codec_dtype)?, cancel)
            })?;
            self.timings.decoder += elapsed;
            let decoded = decoded
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if decoded.is_empty() || decoded.iter().any(|v| !v.is_finite()) {
                candle_core::bail!("decoder produced invalid PCM");
            }
            if cancel() {
                return Ok(Outcome::Cancelled);
            }
            pcm(&decoded)?;
            if cancel() {
                return Ok(Outcome::Cancelled);
            }
            if i > 2 && self.acoustic.stop(&lm)? {
                return Ok(Outcome::Eos);
            }
            let (current, elapsed) = measure(o.measure_modules, &self.device, || {
                self.acoustic.encode(&patch.transpose(1, 2)?, cancel)
            })?;
            self.timings.local_encoder += elapsed;
            let (next_lm, elapsed) = measure(o.measure_modules, &self.device, || {
                self.acoustic.fsq(
                    &self
                        .base
                        .forward(
                            &current.to_dtype(self.backbone_dtype)?.unsqueeze(1)?,
                            Some(&mut base_cache),
                            cancel,
                        )?
                        .squeeze(1)?
                        .to_dtype(self.dtype)?,
                )
            })?;
            self.timings.backbone += elapsed;
            lm = next_lm;
            let fused = self
                .acoustic
                .fusion
                .forward(&Tensor::cat(&[&lm, &current], 1)?)?;
            let (next_res, elapsed) = measure(o.measure_modules, &self.device, || {
                self.residual
                    .forward(&fused.unsqueeze(1)?, Some(&mut res_cache), cancel)?
                    .squeeze(1)
            })?;
            self.timings.residual += elapsed;
            res = next_res;
            prefix = patch;
        }
        Ok(Outcome::Truncated)
    }
}

#[cfg(all(test, feature = "codec-experiment", target_os = "macos"))]
mod accelerate_experiment;
fn quantize_audio_ranges(
    full: &Tensor,
    ranges: &[(usize, usize)],
    mut quantize: impl FnMut(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    if ranges.is_empty() {
        return Ok(full.clone());
    }
    let length = full.dim(1)?;
    let mut cursor = 0;
    let mut parts = Vec::new();
    for &(start, count) in ranges {
        let end = start
            .checked_add(count)
            .filter(|&end| end <= length)
            .ok_or_else(|| candle_core::Error::Msg("invalid audio conditioning range".into()))?;
        if start < cursor || count == 0 {
            candle_core::bail!("overlapping or empty audio conditioning range");
        }
        if start > cursor {
            parts.push(full.narrow(1, cursor, start - cursor)?);
        }
        parts.push(quantize(&full.narrow(1, start, count)?)?);
        cursor = end;
    }
    if cursor < length {
        parts.push(full.narrow(1, cursor, length - cursor)?);
    }
    Tensor::cat(&parts, 1)
}

#[cfg(test)]
mod conditioning_tests {
    use super::*;
    #[test]
    fn disjoint_identity_and_continuation_masks_preserve_text_and_markers() -> Result<()> {
        let full = Tensor::from_vec((0..8).map(|i| i as f32).collect(), (1, 8, 1), &Device::Cpu)?;
        let result = quantize_audio_ranges(&full, &[(1, 2), (5, 2)], |x| x.affine(1.0, 100.0))?;
        assert_eq!(
            result.flatten_all()?.to_vec1::<f32>()?,
            vec![0.0, 101.0, 102.0, 3.0, 4.0, 105.0, 106.0, 7.0]
        );
        assert!(quantize_audio_ranges(&full, &[(1, 3), (2, 2)], |x| Ok(x.clone())).is_err());
        assert!(quantize_audio_ranges(&full, &[(7, 2)], |x| Ok(x.clone())).is_err());
        Ok(())
    }
    #[test]
    fn single_reference_and_no_reference_keep_existing_masks() -> Result<()> {
        let full = Tensor::from_vec(vec![0f32, 1.0, 2.0, 3.0], (1, 4, 1), &Device::Cpu)?;
        let result = quantize_audio_ranges(&full, &[(2, 2)], |x| x.affine(1.0, 10.0))?;
        assert_eq!(
            result.flatten_all()?.to_vec1::<f32>()?,
            vec![0.0, 1.0, 12.0, 13.0]
        );
        assert_eq!(
            quantize_audio_ranges(&full, &[], |_| panic!("no audio"))?
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![0.0, 1.0, 2.0, 3.0]
        );
        Ok(())
    }
}

fn validate_original_config(config: &serde_json::Value) -> Result<()> {
    if config["architecture"] != "voxcpm2"
        || config["lm_config"]["use_mup"] != false
        || config["residual_lm_no_rope"] != true
    {
        candle_core::bail!("unsupported original VoxCPM2 architecture");
    }
    for (path, expected) in [
        ("/lm_config/hidden_size", 2048),
        ("/lm_config/intermediate_size", 6144),
        ("/lm_config/num_hidden_layers", 28),
        ("/lm_config/num_attention_heads", 16),
        ("/lm_config/num_key_value_heads", 2),
        ("/lm_config/kv_channels", 128),
        ("/lm_config/vocab_size", 73448),
        ("/lm_config/max_position_embeddings", 32768),
        (
            "/lm_config/rope_scaling/original_max_position_embeddings",
            32768,
        ),
        ("/patch_size", 4),
        ("/feat_dim", 64),
        ("/scalar_quantization_latent_dim", 512),
        ("/scalar_quantization_scale", 9),
        ("/residual_lm_num_layers", 8),
        ("/encoder_config/hidden_dim", 1024),
        ("/encoder_config/ffn_dim", 4096),
        ("/encoder_config/num_heads", 16),
        ("/encoder_config/num_layers", 12),
        ("/encoder_config/kv_channels", 128),
        ("/dit_config/hidden_dim", 1024),
        ("/dit_config/ffn_dim", 4096),
        ("/dit_config/num_heads", 16),
        ("/dit_config/num_layers", 12),
        ("/dit_config/kv_channels", 128),
        ("/audio_vae_config/encoder_dim", 128),
        ("/audio_vae_config/latent_dim", 64),
        ("/audio_vae_config/decoder_dim", 2048),
        ("/audio_vae_config/sample_rate", 16000),
        ("/audio_vae_config/out_sample_rate", 48000),
    ] {
        if config.pointer(path).and_then(serde_json::Value::as_u64) != Some(expected) {
            candle_core::bail!("unsupported original config {path}; expected {expected}");
        }
    }
    for (path, expected) in [
        (
            "/audio_vae_config/encoder_rates",
            serde_json::json!([2, 5, 8, 8]),
        ),
        (
            "/audio_vae_config/decoder_rates",
            serde_json::json!([8, 6, 5, 2, 2, 2]),
        ),
        (
            "/audio_vae_config/sr_bin_boundaries",
            serde_json::json!([20000, 30000, 40000]),
        ),
    ] {
        if config.pointer(path) != Some(&expected) {
            candle_core::bail!("unsupported original config {path}");
        }
    }
    if config["lm_config"]["rms_norm_eps"] != 1e-5
        || config["lm_config"]["rope_theta"] != 10000
        || config["dit_config"]["mean_mode"] != false
        || config["dit_config"]["cfm_config"]["solver"] != "euler"
        || config["lm_config"]["rope_scaling"]["type"] != "longrope"
    {
        candle_core::bail!("unsupported original normalization/position/DiT config");
    }
    Ok(())
}
struct Noise(u64);
impl Noise {
    fn uniform(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) as u32 as f64 + 1.0) as f32 / (u32::MAX as f32 + 2.0)
    }
    fn normal(&mut self) -> f32 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f32::consts::TAU * self.uniform()).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_precision_and_initialization_cancel_fail_before_io() {
        let absent = Path::new("missing-original-model-fixture");
        assert!(
            Model::load_original(absent, &Device::Cpu, DType::F16, &|| false)
                .err()
                .unwrap()
                .to_string()
                .contains("require CPU F32")
        );
        assert!(
            Model::load_original(absent, &Device::Cpu, DType::F32, &|| true)
                .err()
                .unwrap()
                .to_string()
                .contains("cancelled")
        );
    }
}
