//! Wav2Vec2-BERT 2.0 semantic feature extractor (facebook/w2v-bert-2.0).
//!
//! Port of the HF transformers v4.52.1 implementation used by upstream
//! IndexTTS 2.5 (`Wav2Vec2BertModel` + `SeamlessM4TFeatureExtractor`,
//! reference copies in `D:\workspace\index-tts\reference\`). Upstream
//! (`indextts/infer_v2_5.py:get_emb`) runs the model with
//! `output_hidden_states=True`, takes `hidden_states[17]` and normalizes it
//! with the statistics stored in `wav2vec2bert_stats.pt`; the caller does
//! that normalization, [`Wav2Vec2Bert::hidden_17`] returns the raw tensor.
//!
//! Two properties of the reference implementation that differ from the
//! common wav2vec2 intuition (verified against the pinned sources and the
//! published checkpoint keys):
//!
//! * The processor is *not* a raw-waveform normalizer.
//!   `SeamlessM4TFeatureExtractor.__call__` computes 80-bin Kaldi-style
//!   log-mel filter banks (25 ms Povey window, 10 ms shift, 512-point FFT,
//!   pre-emphasis 0.97, per-frame DC removal, 16-bit scaling, mel floor
//!   2^-23), then zero-mean/unit-variance normalizes **per mel bin over
//!   time** (`ddof=1` unbiased variance, floor `sqrt(var + 1e-7)`), pads the
//!   frame axis to a multiple of 2 (HF `pad_to_multiple_of=2`) and reshapes
//!   `(1, N, 80) -> (1, N/2, 160)` by interleaving frame pairs.
//! * The model has **no convolutional feature extractor and no adapter**
//!   (`add_adapter=false`): `feature_projection` consumes the `(B, T, 160)`
//!   fbank features directly (`feature_projection_input_dim = 160 = 80 x 2`).
//!   There is also no encoder-level final layer norm and no
//!   `pos_conv_embed`; every conformer block ends in its own
//!   `final_layer_norm`.
//!
//! Positional attention: `facebook/w2v-bert-2.0` uses
//! `position_embeddings_type = "relative_key"` (Shaw et al., arXiv:1803.02155):
//! a learned `self_attn.distance_embedding` of shape
//! `(left_max + right_max + 1, head_size) = (73, 64)` indexed by the clamped
//! key-minus-query distance `clamp(r - l, -64, 8) + 64`, added to the scores
//! scaled by `1/sqrt(head_size)`. The Transformer-XL "rel_shift" trick and
//! `pos_bias_u/pos_bias_v`/`linear_pos` weights belong to the unused
//! `"relative"` branch (and rotary to `"rotary"`); the published checkpoint
//! indeed only contains `distance_embedding` in the attention blocks.
//!
//! Hidden-state indexing: HF appends the pre-layer hidden state before each
//! encoder layer, so `hidden_states = [feature_projection out, layer-0 out,
//! ..., layer-23 out]` and `hidden_states[17]` is the output of encoder
//! layer 16 (0-based; the 17th of 24), after that layer's
//! `final_layer_norm`. Only the first 17 layers are loaded and executed.
//!
//! Attention mask: upstream processes single clips, so the mask is normally
//! all ones and no masking enters the math. The one exception produced by
//! the processor itself is an odd fbank frame count: the padded tail frame
//! yields a single trailing zero in the mask. Both cases are handled
//! faithfully (encoder-entry zero fill of padded frames, additive
//! `-f32::MIN` key mask in attention, zero fill before the depthwise
//! convolution), matching `Wav2Vec2BertEncoder.forward`.
//!
//! Weight source: the original `model.safetensors` + `config.json` of
//! facebook/w2v-bert-2.0, loaded directly via
//! `VarBuilder::from_mmaped_safetensors` (HF names verbatim, F32).

use candle_core::{DType, Device, Module, Result, Tensor, bail};
use candle_nn::ops;
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------------------
// SeamlessM4TFeatureExtractor (16 kHz mono path)
// ---------------------------------------------------------------------------

/// Number of mel bins of the SeamlessM4T feature extractor.
const NUM_MEL_BINS: usize = 80;
/// Analysis frame length in samples (25 ms at 16 kHz).
const FRAME_LENGTH: usize = 400;
/// Hop length in samples (10 ms at 16 kHz).
const HOP_LENGTH: usize = 160;
/// FFT buffer size (next power of two above 400).
const FFT_LENGTH: usize = 512;
/// One-sided frequency bins of the 512-point real FFT.
const NUM_FREQ_BINS: usize = FFT_LENGTH / 2 + 1; // 257
/// Frame-pair stride of the `(N, 80) -> (N/2, 160)` reshape.
const FEATURE_STRIDE: usize = 2;
/// `mel_floor` of the reference `spectrogram` call: `2^-23`.
const MEL_FLOOR: f64 = 1.192092955078125e-07;
/// Pre-emphasis coefficient (Kaldi default).
const PREEMPHASIS: f64 = 0.97;
/// Variance floor of the per-mel-bin normalization: `sqrt(var + 1e-7)`.
const NORM_VARIANCE_EPS: f64 = 1e-7;

/// Kaldi mel scale (`mel_scale="kaldi"`): `1127 * ln(1 + f / 700)`.
fn hertz_to_mel_kaldi(freq: f64) -> f64 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// Povey window of `FRAME_LENGTH` samples:
/// `np.hanning(M)^0.85` with the symmetric (non-periodic) Hann
/// `w[n] = 0.5 - 0.5 * cos(2*pi*n / (M-1))` — `window_function(400,
/// "povey", periodic=False)` in transformers `audio_utils`.
fn povey_window() -> &'static [f64] {
    static WINDOW: OnceLock<Vec<f64>> = OnceLock::new();
    WINDOW.get_or_init(|| {
        let m = FRAME_LENGTH;
        (0..m)
            .map(|n| {
                let hann =
                    0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (m - 1) as f64).cos();
                hann.powf(0.85)
            })
            .collect()
    })
}

/// 80 x 257 Kaldi mel filter bank, matching `mel_filter_bank(
/// num_frequency_bins=257, num_mel_filters=80, min_frequency=20,
/// max_frequency=8000, sampling_rate=16000, norm=None, mel_scale="kaldi",
/// triangularize_in_mel_space=True)`. Returns row-major `(mel bin, freq bin)`
/// weights.
fn kaldi_mel_filters() -> &'static [f64] {
    static FILTERS: OnceLock<Vec<f64>> = OnceLock::new();
    FILTERS.get_or_init(|| {
        // Center points of the triangular filters, linearly spaced in mel
        // space; with `triangularize_in_mel_space=True` the filter corners
        // stay in mel space while the FFT bin centers are mapped to mel.
        let mel_min = hertz_to_mel_kaldi(20.0);
        let mel_max = hertz_to_mel_kaldi(8000.0);
        let num_corners = NUM_MEL_BINS + 2;
        let mel_freqs: Vec<f64> = (0..num_corners)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (num_corners - 1) as f64)
            .collect();
        let filter_diff: Vec<f64> = (0..num_corners - 1)
            .map(|i| mel_freqs[i + 1] - mel_freqs[i])
            .collect();
        let fft_bin_width = 16000.0 / ((NUM_FREQ_BINS - 1) * 2) as f64; // 31.25 Hz
        let fft_mel: Vec<f64> = (0..NUM_FREQ_BINS)
            .map(|i| hertz_to_mel_kaldi(fft_bin_width * i as f64))
            .collect();

        let mut filters = vec![0.0f64; NUM_MEL_BINS * NUM_FREQ_BINS];
        for (i, &f) in fft_mel.iter().enumerate() {
            // _create_triangular_filter_bank: down slope towards corner j,
            // up slope from corner j+2, clipped at zero.
            for j in 0..NUM_MEL_BINS {
                let down = -(mel_freqs[j] - f) / filter_diff[j];
                let up = (mel_freqs[j + 2] - f) / filter_diff[j + 1];
                filters[j * NUM_FREQ_BINS + i] = down.min(up).max(0.0);
            }
        }
        filters
    })
}

fn forward_fft() -> Arc<dyn Fft<f64>> {
    static FFT: OnceLock<Arc<dyn Fft<f64>>> = OnceLock::new();
    FFT.get_or_init(|| FftPlanner::new().plan_fft_forward(FFT_LENGTH))
        .clone()
}

/// Log-mel filter bank of one 16 kHz mono clip, matching
/// `SeamlessM4TFeatureExtractor._extract_fbank_features` (transformers
/// `audio_utils.spectgram` with `frame_length=400, hop_length=160,
/// fft_length=512, power=2.0, center=False, preemphasis=0.97,
/// log_mel="log", mel_floor=2^-23, remove_dc_offset=True` on the 16-bit
/// scaled waveform). Returns `(num_frames, frames row-major (N, 80))`.
fn log_mel_fbank(waveform: &[f32]) -> Result<(usize, Vec<f32>)> {
    if waveform.len() < FRAME_LENGTH {
        bail!(
            "waveform of {} samples is shorter than one {}-sample analysis frame",
            waveform.len(),
            FRAME_LENGTH
        );
    }
    let num_frames = 1 + (waveform.len() - FRAME_LENGTH) / HOP_LENGTH;
    let window = povey_window();
    let filters = kaldi_mel_filters();
    let fft = forward_fft();
    let mut buffer = vec![Complex::new(0.0f64, 0.0f64); FFT_LENGTH];
    let mut frame = [0.0f64; FRAME_LENGTH];
    let mut out = vec![0.0f32; num_frames * NUM_MEL_BINS];

    for t in 0..num_frames {
        let start = t * HOP_LENGTH;
        // Kaldi compliance: 16-bit signed integer scaling, then per-frame
        // DC offset removal (both exact in f64; the f32 -> f64 widening and
        // the 2^15 factor are exact).
        let mut mean = 0.0f64;
        for v in &waveform[start..start + FRAME_LENGTH] {
            mean += *v as f64;
        }
        mean = mean * 32768.0 / FRAME_LENGTH as f64;
        for i in 0..FRAME_LENGTH {
            frame[i] = waveform[start + i] as f64 * 32768.0 - mean;
        }
        // Pre-emphasis; descending order keeps the un-shifted samples,
        // reproducing the (alias-free) numpy slice update.
        for i in (1..FRAME_LENGTH).rev() {
            frame[i] -= PREEMPHASIS * frame[i - 1];
        }
        frame[0] *= 1.0 - PREEMPHASIS;
        for i in 0..FRAME_LENGTH {
            buffer[i] = Complex::new(frame[i] * window[i], 0.0);
        }
        for slot in &mut buffer[FRAME_LENGTH..] {
            *slot = Complex::new(0.0, 0.0);
        }
        fft.process(&mut buffer);

        for bin in 0..NUM_MEL_BINS {
            let filter = &filters[bin * NUM_FREQ_BINS..(bin + 1) * NUM_FREQ_BINS];
            let mut energy = 0.0f64;
            for (k, w) in filter.iter().enumerate() {
                let re = buffer[k].re;
                let im = buffer[k].im;
                energy += w * (re * re + im * im);
            }
            out[t * NUM_MEL_BINS + bin] = energy.max(MEL_FLOOR).ln() as f32;
        }
    }
    Ok((num_frames, out))
}

/// Input featurization performed by `SeamlessM4TFeatureExtractor.__call__`
/// for 16 kHz mono input (upstream: `extract_features(audio_16k,
/// sampling_rate=16000, return_tensors="pt")`, i.e. defaults
/// `padding="longest"`, `pad_to_multiple_of=2`, `truncation=False`,
/// `do_normalize_per_mel_bins=True`).
///
/// For a single clip the batch contains one sequence, so no cross-sample
/// padding happens; only the `pad_to_multiple_of=2` rule can add at most
/// one all-zero 80-bin frame when the raw frame count `N` (25 ms hop-10 ms
/// framing of the raw length) is odd. With `stride=2` the output length is
/// `T = N / 2` rounded up: the raw length for even `N`, one extra frame for
/// odd `N`.
///
/// Returns `(input_features (1, T, 160), attention_mask (1, T) u32)` on the
/// CPU device. The mask is all ones except for a single trailing zero when
/// `N` is odd; see the module docs for how [`Wav2Vec2Bert::hidden_17`]
/// consumes it.
///
/// Numeric parity (verified against transformers 4.52.1
/// `SeamlessM4TFeatureExtractor` on identical input): raw log-mel frames
/// match within float32 rounding (~2e-6). The per-bin normalization
/// amplifies last-bit differences by `1/std` at *degenerate* bins whose
/// variance is near zero (mel-floor-bound bins of narrow-band synthetic
/// tones) — the reference formula shows the same 1e-2-scale self-noise
/// there when computed with float32 vs float64 statistics. Broadband
/// speech keeps every bin's std O(1) and differences stay at rounding
/// level.
pub fn seamless_m4t_features(waveform: &[f32]) -> Result<(Tensor, Tensor)> {
    let (num_frames, mel) = log_mel_fbank(waveform)?;
    if num_frames < 2 {
        // The unbiased (ddof=1) per-bin variance needs at least two frames.
        bail!(
            "waveform of {} samples yields {} fbank frame(s); at least 2 frames (560 samples) are required",
            waveform.len(),
            num_frames
        );
    }

    // Zero-mean unit-variance normalization per mel bin over time. HF
    // computes the statistics on the float32 frames in float32; we
    // accumulate in f64, which only differs from numpy's float32 pairwise
    // summation by float32 rounding noise.
    let mut normed = vec![0.0f32; num_frames * NUM_MEL_BINS];
    for bin in 0..NUM_MEL_BINS {
        let mut mean = 0.0f64;
        for t in 0..num_frames {
            mean += mel[t * NUM_MEL_BINS + bin] as f64;
        }
        mean /= num_frames as f64;
        let mut var = 0.0f64;
        for t in 0..num_frames {
            let d = mel[t * NUM_MEL_BINS + bin] as f64 - mean;
            var += d * d;
        }
        var /= (num_frames - 1) as f64;
        let inv_std = 1.0 / (var + NORM_VARIANCE_EPS).sqrt();
        for t in 0..num_frames {
            normed[t * NUM_MEL_BINS + bin] =
                ((mel[t * NUM_MEL_BINS + bin] as f64 - mean) * inv_std) as f32;
        }
    }

    // Pad to a multiple of the stride, then interleave frame pairs into
    // (1, T, 80 * 2) rows (np.reshape of (1, N, 80)).
    let padded_frames = num_frames + (num_frames % FEATURE_STRIDE);
    let out_len = padded_frames / FEATURE_STRIDE;
    let row = NUM_MEL_BINS * FEATURE_STRIDE; // 160
    let mut features = vec![0.0f32; out_len * row];
    let mut mask = vec![1u32; out_len];
    for t in 0..out_len {
        let even = 2 * t;
        let odd = even + 1;
        features[t * row..t * row + NUM_MEL_BINS]
            .copy_from_slice(&normed[even * NUM_MEL_BINS..odd * NUM_MEL_BINS]);
        if odd < num_frames {
            features[t * row + NUM_MEL_BINS..(t + 1) * row]
                .copy_from_slice(&normed[odd * NUM_MEL_BINS..(odd + 1) * NUM_MEL_BINS]);
        } else {
            // The single padded frame selects attention-mask position
            // indices % stride == 1, i.e. this row's second half.
            mask[t] = 0;
        }
    }
    let device = Device::Cpu;
    let input_features = Tensor::from_vec(
        features,
        (1, out_len, NUM_MEL_BINS * FEATURE_STRIDE),
        &device,
    )?;
    let attention_mask = Tensor::from_vec(mask, (1, out_len), &device)?;
    Ok((input_features, attention_mask))
}

// ---------------------------------------------------------------------------
// Wav2Vec2BertModel
// ---------------------------------------------------------------------------

/// `hidden_states[17]` index consumed by upstream `get_emb`.
const HIDDEN_STATE_INDEX: usize = 17;

struct Config {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    feature_projection_input_dim: usize,
    layer_norm_eps: f64,
    left_max_position_embeddings: usize,
    right_max_position_embeddings: usize,
    conv_depthwise_kernel_size: usize,
}

impl Config {
    /// Parse `facebook/w2v-bert-2.0`'s `config.json`, falling back to the
    /// `Wav2Vec2BertConfig` defaults for absent fields and rejecting
    /// configurations outside the ported inference path.
    fn from_json(text: &str) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(candle_core::Error::wrap)?;
        let get_usize = |key: &str, default: usize| {
            value
                .get(key)
                .and_then(serde_json::Value::as_u64)
                .map_or(default, |v| v as usize)
        };
        let position_embeddings_type = value
            .get("position_embeddings_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("relative_key");
        if position_embeddings_type != "relative_key" {
            bail!(
                "unsupported position_embeddings_type {position_embeddings_type:?}; only \
                 \"relative_key\" (the facebook/w2v-bert-2.0 setting) is ported"
            );
        }
        if value
            .get("add_adapter")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            bail!("add_adapter=true is not part of the facebook/w2v-bert-2.0 inference path");
        }
        let hidden_act = value
            .get("hidden_act")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("swish");
        if hidden_act != "swish" {
            bail!("unsupported hidden_act {hidden_act:?}; only \"swish\" is ported");
        }
        let config = Self {
            hidden_size: get_usize("hidden_size", 1024),
            num_hidden_layers: get_usize("num_hidden_layers", 24),
            num_attention_heads: get_usize("num_attention_heads", 16),
            intermediate_size: get_usize("intermediate_size", 4096),
            feature_projection_input_dim: get_usize("feature_projection_input_dim", 160),
            layer_norm_eps: value
                .get("layer_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1e-5),
            left_max_position_embeddings: get_usize("left_max_position_embeddings", 64),
            right_max_position_embeddings: get_usize("right_max_position_embeddings", 8),
            conv_depthwise_kernel_size: get_usize("conv_depthwise_kernel_size", 31),
        };
        if !config
            .hidden_size
            .is_multiple_of(config.num_attention_heads)
        {
            bail!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                config.hidden_size,
                config.num_attention_heads
            );
        }
        if config.conv_depthwise_kernel_size.is_multiple_of(2) {
            bail!(
                "conv_depthwise_kernel_size {} must be odd for 'SAME' padding",
                config.conv_depthwise_kernel_size
            );
        }
        Ok(config)
    }
}

/// `Wav2Vec2BertFeatureProjection`: layer norm over the 160 fbank input
/// dims followed by the projection to `hidden_size` (dropouts are zero /
/// inference-identity).
struct FeatureProjection {
    layer_norm: LayerNorm,
    projection: Linear,
    input_dim: usize,
}

impl FeatureProjection {
    fn load(config: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            layer_norm: candle_nn::layer_norm(
                config.feature_projection_input_dim,
                config.layer_norm_eps,
                vb.pp("layer_norm"),
            )?,
            projection: candle_nn::linear(
                config.feature_projection_input_dim,
                config.hidden_size,
                vb.pp("projection"),
            )?,
            input_dim: config.feature_projection_input_dim,
        })
    }

    fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        let normed = self.layer_norm.forward(input_features)?;
        self.projection.forward(&normed)
    }
}

/// `Wav2Vec2BertFeedForward`: intermediate dense, `swish`, output dense.
struct FeedForward {
    intermediate_dense: Linear,
    output_dense: Linear,
}

impl FeedForward {
    fn load(config: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            intermediate_dense: candle_nn::linear(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("intermediate_dense"),
            )?,
            output_dense: candle_nn::linear(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("output_dense"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.intermediate_dense.forward(hidden_states)?;
        let hidden_states = ops::silu(&hidden_states)?;
        self.output_dense.forward(&hidden_states)
    }
}

/// `nn.GLU(dim=1)` for `(b, 2c, t)` conv stacks: `a * sigmoid(b)`.
fn glu_dim1(hidden_states: &Tensor) -> Result<Tensor> {
    let (_, channels2, _) = hidden_states.dims3()?;
    let channels = channels2 / 2;
    let a = hidden_states.narrow(1, 0, channels)?;
    let gate = hidden_states.narrow(1, channels, channels)?;
    a.broadcast_mul(&ops::sigmoid(&gate)?)
}

/// `Wav2Vec2BertConvolutionModule`: layernorm, pointwise conv + GLU,
/// causal (left-)padded depthwise conv, depthwise layernorm, swish,
/// pointwise conv. `attention_mask` zero-fills padded frames before the
/// depthwise convolution so padding does not leak into real frames.
struct ConvolutionModule {
    layer_norm: LayerNorm,
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    depthwise_layer_norm: LayerNorm,
    pointwise_conv2: Conv1d,
    kernel_size: usize,
}

impl ConvolutionModule {
    fn load(config: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden = config.hidden_size;
        let kernel = config.conv_depthwise_kernel_size;
        Ok(Self {
            layer_norm: candle_nn::layer_norm(hidden, config.layer_norm_eps, vb.pp("layer_norm"))?,
            pointwise_conv1: candle_nn::conv1d_no_bias(
                hidden,
                2 * hidden,
                1,
                Conv1dConfig::default(),
                vb.pp("pointwise_conv1"),
            )?,
            depthwise_conv: candle_nn::conv1d_no_bias(
                hidden,
                hidden,
                kernel,
                Conv1dConfig {
                    groups: hidden,
                    ..Conv1dConfig::default()
                },
                vb.pp("depthwise_conv"),
            )?,
            depthwise_layer_norm: candle_nn::layer_norm(
                hidden,
                config.layer_norm_eps,
                vb.pp("depthwise_layer_norm"),
            )?,
            pointwise_conv2: candle_nn::conv1d_no_bias(
                hidden,
                hidden,
                1,
                Conv1dConfig::default(),
                vb.pp("pointwise_conv2"),
            )?,
            kernel_size: kernel,
        })
    }

    fn forward(&self, hidden_states: &Tensor, frame_mask: Option<&Tensor>) -> Result<Tensor> {
        // frame_mask: (1, T) f32 in {0, 1}; masked_fill(~mask, 0.0) is a
        // multiply by the mask for finite values.
        let hidden_states = self.layer_norm.forward(hidden_states)?;
        let hidden_states = match frame_mask {
            Some(mask) => hidden_states.broadcast_mul(&mask.unsqueeze(2)?)?,
            None => hidden_states,
        };
        let hidden_states = hidden_states.transpose(1, 2)?.contiguous()?;
        let hidden_states = self.pointwise_conv1.forward(&hidden_states)?;
        let hidden_states = glu_dim1(&hidden_states)?;
        // Pad entirely on the left: causal depthwise convolution.
        let hidden_states = hidden_states.pad_with_zeros(2, self.kernel_size - 1, 0)?;
        let hidden_states = self.depthwise_conv.forward(&hidden_states)?;
        let hidden_states = self
            .depthwise_layer_norm
            .forward(&hidden_states.transpose(1, 2)?)?
            .transpose(1, 2)?
            .contiguous()?;
        let hidden_states = ops::silu(&hidden_states)?;
        let hidden_states = self.pointwise_conv2.forward(&hidden_states)?;
        hidden_states.transpose(1, 2)
    }
}

/// Per-forward attention context shared by all encoder layers: the Shaw
/// relative-distance lookup table (clamped `r - l + left_max`) and, when the
/// processor produced a padded clip, the frame/key masks.
struct AttentionContext {
    /// `(1, T, T)` u32: clamped key-minus-query distance + left max.
    distance_index: Tensor,
    /// `(1, T)` f32 {0, 1} frame mask; `None` when the mask is all ones.
    frame_mask: Option<Tensor>,
    /// `(1, 1, 1, T)` additive attention bias (`0` / `f32::MIN`); `None`
    /// when the mask is all ones.
    additive_mask: Option<Tensor>,
}

impl AttentionContext {
    fn new(attention_mask_f32: &Tensor, left_max: usize, right_max: usize) -> Result<Self> {
        let (b, t) = attention_mask_f32.dims2()?;
        if b != 1 {
            bail!("attention_mask must be (1, T); got ({b}, {t})");
        }
        let mask_sum = attention_mask_f32.sum_all()?.to_scalar::<f32>()? as f64;
        let all_ones = mask_sum == t as f64;
        let frame_mask = if all_ones {
            None
        } else {
            Some(attention_mask_f32.to_owned())
        };
        let additive_mask = if all_ones {
            None
        } else {
            // (1.0 - mask) * torch.finfo(float32).min; the (1, T) mask only
            // needs two unsqueezes to reach (1, 1, 1, T).
            Some(
                attention_mask_f32
                    .affine(-1.0, 1.0)?
                    .affine(f32::MIN as f64, 0.0)?
                    .unsqueeze(0)?
                    .unsqueeze(0)?,
            )
        };
        let left = left_max as i64;
        let right = right_max as i64;
        let mut index = Vec::with_capacity(t * t);
        for l in 0..t as i64 {
            for r in 0..t as i64 {
                // Add the offset while still signed; an integer `as u32`
                // truncates instead of saturating.
                index.push(((r - l).clamp(-left, right) + left) as u32);
            }
        }
        let distance_index = Tensor::from_vec(index, (1, t, t), attention_mask_f32.device())?;
        Ok(Self {
            distance_index,
            frame_mask,
            additive_mask,
        })
    }
}

/// `Wav2Vec2BertSelfAttention` with `position_embeddings_type =
/// "relative_key"` (Shaw et al.): scores `q k^T / sqrt(d)` plus
/// `q E[clamp(r - l)]^T / sqrt(d)` where `E` is the learned
/// `distance_embedding`.
struct SelfAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    /// `(left_max + right_max + 1, head_size)`
    distance_embedding: Tensor,
    num_heads: usize,
    head_size: usize,
}

impl SelfAttention {
    fn load(config: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_size = hidden / num_heads;
        let num_positions =
            config.left_max_position_embeddings + config.right_max_position_embeddings + 1;
        Ok(Self {
            linear_q: candle_nn::linear(hidden, hidden, vb.pp("linear_q"))?,
            linear_k: candle_nn::linear(hidden, hidden, vb.pp("linear_k"))?,
            linear_v: candle_nn::linear(hidden, hidden, vb.pp("linear_v"))?,
            linear_out: candle_nn::linear(hidden, hidden, vb.pp("linear_out"))?,
            distance_embedding: vb.get((num_positions, head_size), "distance_embedding.weight")?,
            num_heads,
            head_size,
        })
    }

    fn forward(&self, hidden_states: &Tensor, ctx: &AttentionContext) -> Result<Tensor> {
        let (b, t, hidden) = hidden_states.dims3()?;
        let (h, d) = (self.num_heads, self.head_size);
        let project = |linear: &Linear| {
            linear
                .forward(hidden_states)?
                .reshape((b, t, h, d))?
                .transpose(1, 2)?
                .contiguous()
        };
        let query = project(&self.linear_q)?;
        let key = project(&self.linear_k)?;
        let value = project(&self.linear_v)?;

        let scale = 1.0 / (d as f64).sqrt();
        let scores = query.matmul(&key.transpose(2, 3)?)?.affine(scale, 0.0)?;

        // Shaw relative keys: first the (b, h, t, num_positions) inner
        // products with every distance bucket, then a gather with the
        // clamped distance matrix — equal to the reference
        // einsum("bhld,lrd->bhlr", q, positional_embedding) / sqrt(d).
        let num_positions = self.distance_embedding.dims()[0];
        let distance_t = self.distance_embedding.t()?.contiguous()?;
        // Flatten to a plain 2D matmul: candle requires equal ranks.
        let rel_scores = query
            .reshape((b * h * t, d))?
            .matmul(&distance_t)?
            .reshape((b, h, t, num_positions))?
            .affine(scale, 0.0)?;
        let mut gathered = Vec::with_capacity(b * h);
        for bi in 0..b {
            for hi in 0..h {
                // (1, t, num_positions); gather the distance bucket on the
                // last axis (matching ranks are required by candle).
                let per_head = rel_scores.narrow(0, bi, 1)?.narrow(1, hi, 1)?.squeeze(1)?;
                gathered.push(per_head.gather(&ctx.distance_index, 2)?);
            }
        }
        let rel_scores = Tensor::cat(&gathered, 0)?.reshape((b, h, t, t))?;
        let mut scores = scores.broadcast_add(&rel_scores)?;
        if let Some(additive) = &ctx.additive_mask {
            scores = scores.broadcast_add(additive)?;
        }

        let probs = ops::softmax(&scores, 3)?;
        let attn = probs.matmul(&value)?;
        let attn = attn.transpose(1, 2)?.reshape((b, t, hidden))?;
        self.linear_out.forward(&attn)
    }
}

/// `Wav2Vec2BertEncoderLayer`: macaron FFN x 0.5, self attention,
/// conformer conv module, FFN x 0.5, final layer norm.
struct EncoderLayer {
    ffn1_layer_norm: LayerNorm,
    ffn1: FeedForward,
    self_attn_layer_norm: LayerNorm,
    self_attn: SelfAttention,
    conv_module: ConvolutionModule,
    ffn2_layer_norm: LayerNorm,
    ffn2: FeedForward,
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    fn load(config: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let eps = config.layer_norm_eps;
        Ok(Self {
            ffn1_layer_norm: candle_nn::layer_norm(
                config.hidden_size,
                eps,
                vb.pp("ffn1_layer_norm"),
            )?,
            ffn1: FeedForward::load(config, vb.pp("ffn1"))?,
            self_attn_layer_norm: candle_nn::layer_norm(
                config.hidden_size,
                eps,
                vb.pp("self_attn_layer_norm"),
            )?,
            self_attn: SelfAttention::load(config, vb.pp("self_attn"))?,
            conv_module: ConvolutionModule::load(config, vb.pp("conv_module"))?,
            ffn2_layer_norm: candle_nn::layer_norm(
                config.hidden_size,
                eps,
                vb.pp("ffn2_layer_norm"),
            )?,
            ffn2: FeedForward::load(config, vb.pp("ffn2"))?,
            final_layer_norm: candle_nn::layer_norm(
                config.hidden_size,
                eps,
                vb.pp("final_layer_norm"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, ctx: &AttentionContext) -> Result<Tensor> {
        // 1. Feed-forward 1 (all dropouts are inference identities).
        let residual = hidden_states.clone();
        let hidden_states = self
            .ffn1
            .forward(&self.ffn1_layer_norm.forward(hidden_states)?)?;
        let hidden_states = (hidden_states.affine(0.5, 0.0)? + residual)?;

        // 2. Self-attention.
        let residual = hidden_states.clone();
        let normed = self.self_attn_layer_norm.forward(&hidden_states)?;
        let hidden_states = (self.self_attn.forward(&normed, ctx)? + residual)?;

        // 3. Conformer convolution.
        let residual = hidden_states.clone();
        let conv = self
            .conv_module
            .forward(&hidden_states, ctx.frame_mask.as_ref())?;
        let hidden_states = (residual + conv)?;

        // 4. Feed-forward 2 + final layer norm.
        let residual = hidden_states.clone();
        let hidden_states = self
            .ffn2
            .forward(&self.ffn2_layer_norm.forward(&hidden_states)?)?;
        let hidden_states = (hidden_states.affine(0.5, 0.0)? + residual)?;
        self.final_layer_norm.forward(&hidden_states)
    }
}

pub struct Wav2Vec2Bert {
    feature_projection: FeatureProjection,
    /// The `min(num_hidden_layers, 17)` layers covering `hidden_states[17]`.
    layers: Vec<EncoderLayer>,
    left_max_position_embeddings: usize,
    right_max_position_embeddings: usize,
}

impl Wav2Vec2Bert {
    /// `dir` points at the extracted facebook/w2v-bert-2.0 repository
    /// (model.safetensors + config.json).
    pub fn load(dir: &std::path::Path, device: &Device) -> Result<Self> {
        let config_text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| candle_core::Error::msg(format!("reading {}: {e}", dir.display())))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                device,
            )?
        };
        Self::from_builder(&Config::from_json(&config_text)?, vb)
    }

    fn from_builder(config: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let num_layers = config.num_hidden_layers.min(HIDDEN_STATE_INDEX);
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(EncoderLayer::load(
                config,
                vb.pp(format!("encoder.layers.{i}")),
            )?);
        }
        Ok(Self {
            feature_projection: FeatureProjection::load(config, vb.pp("feature_projection"))?,
            layers,
            left_max_position_embeddings: config.left_max_position_embeddings,
            right_max_position_embeddings: config.right_max_position_embeddings,
        })
    }

    /// Returns `hidden_states[17]` — shape `(B, T, 1024)`, f32, on the model
    /// device. Inputs come from [`seamless_m4t_features`], moved to the
    /// device by the caller (mask dtypes are converted internally).
    ///
    /// `hidden_states[17]` is the output of encoder layer 16 (0-based, the
    /// 17th of 24) after its `final_layer_norm`, so only the first 17
    /// layers are loaded and executed. `attention_mask` must be `(1, T)`
    /// matching `input_features` with values in `{0, 1}`; the all-ones case
    /// (single unpadded clip) skips all masking, while a padded tail frame
    /// (odd fbank frame count) is handled like the reference encoder.
    pub fn hidden_17(&self, input_features: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (b, t, input_dim) = input_features.dims3()?;
        if b != 1 {
            bail!("only batch=1 inference is ported; got batch {b}");
        }
        if input_dim != self.feature_projection.input_dim {
            bail!(
                "input_features last dim {input_dim} != feature_projection_input_dim {}",
                self.feature_projection.input_dim
            );
        }
        let attention_mask = attention_mask.to_dtype(DType::F32)?;
        if attention_mask.dims() != [b, t] {
            bail!(
                "attention_mask shape {:?} != ({b}, {t})",
                attention_mask.dims()
            );
        }
        let ctx = AttentionContext::new(
            &attention_mask,
            self.left_max_position_embeddings,
            self.right_max_position_embeddings,
        )?;

        let mut hidden_states = self.feature_projection.forward(input_features)?;
        // Encoder entry: make sure padded positions output 0.
        if let Some(frame_mask) = &ctx.frame_mask {
            hidden_states = hidden_states.broadcast_mul(&frame_mask.unsqueeze(2)?)?;
        }
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &ctx)?;
        }
        Ok(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// Tests (no model downloads; the real-weights check is #[ignore] + env-gated)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f32, seconds: f32, amplitude: f32) -> Vec<f32> {
        let n = (16000.0 * seconds) as usize;
        (0..n)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / 16000.0).sin())
            .collect()
    }

    #[test]
    fn povey_window_is_symmetric_and_bounded() {
        let window = povey_window();
        assert_eq!(window.len(), FRAME_LENGTH);
        assert_eq!(window[0], 0.0);
        assert_eq!(window[FRAME_LENGTH - 1], 0.0);
        // Symmetric Hann => symmetric Povey (mirror around the center).
        let mid = FRAME_LENGTH / 2;
        for i in 0..mid {
            let a = window[i];
            let b = window[FRAME_LENGTH - 1 - i];
            assert!((a - b).abs() < 1e-12, "{i}: {a} vs {b}");
            assert!((0.0..=1.0).contains(&a));
        }
        assert!(window[mid] > 0.99);
    }

    #[test]
    fn mel_filters_cover_every_bin() {
        let filters = kaldi_mel_filters();
        assert_eq!(filters.len(), NUM_MEL_BINS * NUM_FREQ_BINS);
        for bin in 0..NUM_MEL_BINS {
            let row = &filters[bin * NUM_FREQ_BINS..(bin + 1) * NUM_FREQ_BINS];
            assert!(row.iter().all(|w| *w >= 0.0));
            assert!(row.iter().all(|w| *w <= 1.0));
            assert!(
                row.iter().any(|w| *w > 1e-3),
                "mel filter {bin} is all zero"
            );
        }
    }

    #[test]
    fn featurizer_shapes_and_single_clip_length() {
        // 1 s: N = 1 + (16000 - 400) / 160 = 98 frames (even) => T = 49.
        let wave = sine_wave(440.0, 1.0, 0.5);
        let (features, mask) = seamless_m4t_features(&wave).unwrap();
        assert_eq!(features.dims(), [1, 49, 160]);
        assert_eq!(mask.dims(), [1, 49]);
        assert_eq!(
            mask.to_dtype(DType::F32)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap(),
            49.0
        );

        // 16080 samples: N = 1 + 15680/160 = 99 frames (odd) => padded to
        // 100, T = 50 with a single trailing zero in the mask.
        let wave = sine_wave(440.0, 16080.0 / 16000.0, 0.5);
        let (features, mask) = seamless_m4t_features(&wave).unwrap();
        assert_eq!(features.dims(), [1, 50, 160]);
        let mask: Vec<u32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(mask.len(), 50);
        assert_eq!(&mask[..49], &vec![1u32; 49][..]);
        assert_eq!(mask[49], 0);
        // The padded row's second half (frame 99) is all zeros.
        let features = features.squeeze(0).unwrap();
        let last = features.narrow(0, 49, 1).unwrap();
        let second_half = last.narrow(1, 80, 80).unwrap().flatten_all().unwrap();
        let second_half: Vec<f32> = second_half.to_vec1().unwrap();
        assert!(second_half.iter().all(|v| *v == 0.0));

        // Too short to frame.
        assert!(seamless_m4t_features(&[0.0; 100]).is_err());
    }

    #[test]
    fn featurizer_zero_mean_unit_variance_per_mel_bin() {
        let wave = sine_wave(440.0, 1.0, 0.5);
        let (features, _mask) = seamless_m4t_features(&wave).unwrap();
        let features = features.squeeze(0).unwrap(); // (49, 160)
        // Undo the stride-2 interleave: columns 0..80 are the even frames,
        // columns 80..160 the odd frames.
        let features: Vec<f32> = features.flatten_all().unwrap().to_vec1().unwrap();
        let t_len = 49;
        let mut frames = vec![0.0f32; 2 * t_len * NUM_MEL_BINS];
        for t in 0..t_len {
            for j in 0..80 {
                frames[(2 * t) * 80 + j] = features[t * 160 + j];
                frames[(2 * t + 1) * 80 + j] = features[t * 160 + 80 + j];
            }
        }
        let num_frames = 2 * t_len;
        let mut energetic_bins = 0;
        for j in 0..NUM_MEL_BINS {
            let mut mean = 0.0f64;
            for t in 0..num_frames {
                mean += frames[t * 80 + j] as f64;
            }
            mean /= num_frames as f64;
            assert!(mean.abs() < 1e-3, "bin {j} mean {mean}");
            let mut var = 0.0f64;
            for t in 0..num_frames {
                let d = frames[t * 80 + j] as f64 - mean;
                var += d * d;
            }
            var /= (num_frames - 1) as f64;
            // var / (var + 1e-7) <= 1 always; bins dominated by the mel
            // floor (constant ln(2^-23)) normalize far below 1, while
            // energy-bearing bins are ~1.
            assert!(var <= 1.0 + 1e-3, "bin {j} var {var}");
            if var >= 0.99 {
                energetic_bins += 1;
            }
        }
        assert!(
            energetic_bins >= 3,
            "expected several energy-bearing mel bins, got {energetic_bins}"
        );
    }

    #[test]
    fn tiny_model_forward_full_and_padded_masks() {
        let device = Device::Cpu;
        let config = Config {
            hidden_size: 32,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            intermediate_size: 64,
            feature_projection_input_dim: 160,
            layer_norm_eps: 1e-5,
            left_max_position_embeddings: 64,
            right_max_position_embeddings: 8,
            conv_depthwise_kernel_size: 31,
        };
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = Wav2Vec2Bert::from_builder(&config, vb).unwrap();
        // All 4 (< 17) layers cover the hidden-state path end to end.
        assert_eq!(model.layers.len(), 4);

        let t_len = 12usize;
        let input: Vec<f32> = (0..t_len * 160)
            .map(|i| (i as f32 * 0.037).sin() * 0.5)
            .collect();
        let input = Tensor::from_vec(input, (1, t_len, 160), &device).unwrap();
        let ones = Tensor::from_vec(vec![1u32; t_len], (1, t_len), &device).unwrap();

        let out = model.hidden_17(&input, &ones).unwrap();
        assert_eq!(out.dims(), [1, t_len, 32]);
        assert!(
            out.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );

        // Trailing padded frame: same shapes, finite values, and the padded
        // key is excluded so real positions change.
        let mut mask = vec![1u32; t_len];
        mask[t_len - 1] = 0;
        let padded_mask = Tensor::from_vec(mask, (1, t_len), &device).unwrap();
        let out_padded = model.hidden_17(&input, &padded_mask).unwrap();
        assert_eq!(out_padded.dims(), [1, t_len, 32]);
        assert!(
            out_padded
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .all(|v| v.is_finite())
        );
        let a = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = out_padded.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6));

        // F32 masks and batch checks.
        let f32_mask = ones.to_dtype(DType::F32).unwrap();
        assert_eq!(
            model.hidden_17(&input, &f32_mask).unwrap().dims(),
            [1, t_len, 32]
        );
        let big_batch = input.broadcast_as((2, t_len, 160)).unwrap();
        assert!(
            model
                .hidden_17(&big_batch, &ones.broadcast_as((2, t_len)).unwrap())
                .is_err()
        );
    }

    #[test]
    fn distance_index_clamps_to_window() {
        // Long enough to reach the left clamp bound of -64.
        let t_len = 70usize;
        let mask = Tensor::from_vec(vec![1u32; t_len], (1, t_len), &Device::Cpu).unwrap();
        let ctx = AttentionContext::new(&mask.to_dtype(DType::F32).unwrap(), 64, 8).unwrap();
        assert_eq!(ctx.distance_index.dims(), [1, t_len, t_len]);
        let idx: Vec<u32> = ctx.distance_index.flatten_all().unwrap().to_vec1().unwrap();
        let at = |l: usize, r: usize| idx[l * t_len + r];
        // delta = clamp(r - l, -64, 8) + 64.
        assert_eq!(at(0, 0), 64);
        assert_eq!(at(0, 8), 72); // r - l = 8 (right_max) stays.
        assert_eq!(at(0, 68), 72); // clamped down from 68.
        assert_eq!(at(69, 5), 0); // r - l = -64 exactly.
        assert_eq!(at(69, 0), 0); // clamped up from -69.
        assert!(ctx.frame_mask.is_none() && ctx.additive_mask.is_none());

        let mut partial = vec![1u32; t_len];
        partial[t_len - 1] = 0;
        let mask = Tensor::from_vec(partial, (1, t_len), &Device::Cpu).unwrap();
        let ctx = AttentionContext::new(&mask.to_dtype(DType::F32).unwrap(), 64, 8).unwrap();
        let additive = ctx.additive_mask.as_ref().unwrap();
        assert_eq!(additive.dims(), [1, 1, 1, t_len]);
        let additive: Vec<f32> = additive.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(additive[0], 0.0);
        assert_eq!(additive[t_len - 1], f32::MIN);
    }

    /// Real-weights smoke test: `INDEXTTS_CONVERTED_DIR=<dir>` must contain
    /// `w2v-bert-2.0/{model.safetensors,config.json}`. 1 s of 16 kHz zeros
    /// has 98 fbank frames => hidden_states[17] of shape (1, 49, 1024).
    #[test]
    #[ignore = "requires local facebook/w2v-bert-2.0 weights via INDEXTTS_CONVERTED_DIR"]
    fn real_weights_hidden_17_shape() {
        let dir = std::env::var("INDEXTTS_CONVERTED_DIR").expect("INDEXTTS_CONVERTED_DIR unset");
        let dir = std::path::Path::new(&dir).join("w2v-bert-2.0");
        let model = Wav2Vec2Bert::load(&dir, &Device::Cpu).unwrap();

        let zeros = vec![0.0f32; 16_000];
        let (features, mask) = seamless_m4t_features(&zeros).unwrap();
        assert_eq!(features.dims(), [1, 49, 160]);
        let out = model.hidden_17(&features, &mask).unwrap();
        assert_eq!(out.dims(), [1, 49, 1024]);
        let values: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));

        let wave = sine_wave(440.0, 1.0, 0.1);
        let (features, mask) = seamless_m4t_features(&wave).unwrap();
        let out = model.hidden_17(&features, &mask).unwrap();
        assert_eq!(out.dims(), [1, 49, 1024]);
        let values: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }
}
