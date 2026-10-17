//! Official ONNX artifacts through `ort`.
//!
//! * `speech_tokenizer_v3.onnx` — FSQ speech tokens from whisper log-mel.
//! * `campplus.onnx` — 192-d speaker embedding from mean-centered kaldi
//!   fbank.
//! * optional `flow.decoder.estimator.fp32.onnx` (from the official
//!   `export_onnx.py`) — used to cross-check the Candle DiT on identical
//!   inputs.

use candle_core::{Device, Result, Tensor};
use ort::{
    session::{RunOptions, Session, builder::GraphOptimizationLevel},
    value::Tensor as OrtTensor,
};

use super::audio::{KaldiFbank, WhisperMel};

pub struct FrontendOnnx {
    speech_tokenizer: Session,
    campplus: Session,
    whisper_mel: WhisperMel,
    fbank: KaldiFbank,
    run: RunOptions,
}

fn open(path: &std::path::Path, threads: usize) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        .with_intra_threads(threads)?
        .commit_from_file(path)
}

impl FrontendOnnx {
    /// `model_dir` must contain `speech_tokenizer_v3.onnx` and
    /// `campplus.onnx`.
    pub fn load(model_dir: &std::path::Path, threads: usize) -> std::io::Result<Self> {
        let speech_tokenizer = open(&model_dir.join("speech_tokenizer_v3.onnx"), threads)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let campplus = open(&model_dir.join("campplus.onnx"), threads)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let run = RunOptions::new().map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(Self {
            speech_tokenizer,
            campplus,
            whisper_mel: WhisperMel::new(),
            fbank: KaldiFbank::new(),
            run,
        })
    }

    /// `audio_16k`: `[T]` f32. Returns `[1, Tt]` u32 speech tokens.
    pub fn speech_tokens(&mut self, audio_16k: &Tensor) -> Result<Tensor> {
        if audio_16k.dim(0)? > 30 * 16_000 {
            candle_core::bail!("speech tokenizer supports reference audio up to 30 seconds");
        }
        let mel = self.whisper_mel.forward(audio_16k)?; // [1, 128, T]
        let (_, _, t) = mel.dims3()?;
        let flat: Vec<f32> = mel.flatten_all()?.to_vec1()?;
        let outputs = self
            .speech_tokenizer
            .run_with_options(
                ort::inputs![
                    "feats" => OrtTensor::from_array(([1, 128, t], flat))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?,
                    "feats_length" => OrtTensor::from_array(([1], vec![t as i32]))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                ],
                &self.run,
            )
            .map_err(|e| candle_core::Error::Msg(format!("speech_tokenizer run: {e}")))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<i32>()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        if data
            .iter()
            .any(|&v| !(0..crate::config::SPEECH_TOKEN_SIZE as i32).contains(&v))
        {
            candle_core::bail!("speech tokenizer returned an out-of-vocabulary token");
        }
        let tokens: Vec<u32> = data.iter().map(|&v| v as u32).collect();
        let n: usize = shape.iter().map(|&v| v as usize).product();
        Tensor::from_vec(tokens, (1, n), &Device::Cpu)
    }

    /// `audio_16k`: `[T]` f32. Returns `[1, 192]` speaker embedding.
    pub fn speaker_embedding(&mut self, audio_16k: &Tensor) -> Result<Tensor> {
        let feat = self.fbank.forward(audio_16k)?; // [T, 80]
        let (t, d) = feat.dims2()?;
        if t == 0 {
            candle_core::bail!("reference audio is too short for speaker features");
        }
        let mut v: Vec<f32> = feat.flatten_all()?.to_vec1()?;
        // feat = feat - feat.mean(dim=0)
        for c in 0..d {
            let mean: f32 = (0..t).map(|r| v[r * d + c]).sum::<f32>() / t as f32;
            for r in 0..t {
                v[r * d + c] -= mean;
            }
        }
        let outputs = self
            .campplus
            .run_with_options(
                ort::inputs![
                    OrtTensor::from_array(([1, t, d], v))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                ],
                &self.run,
            )
            .map_err(|e| candle_core::Error::Msg(format!("campplus run: {e}")))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let n: usize = shape.iter().map(|&v| v as usize).product();
        Tensor::from_vec(data.to_vec(), (1, n), &Device::Cpu)
    }
}

/// Runs the officially exported CFM estimator ONNX against the same inputs
/// as [`crate::flow::dit::DiT`] for numeric comparison.
pub struct EstimatorOnnx {
    session: Session,
    run: RunOptions,
}

impl EstimatorOnnx {
    pub fn load(path: &std::path::Path, threads: usize) -> std::io::Result<Self> {
        let session = open(path, threads).map_err(|e| std::io::Error::other(e.to_string()))?;
        let run = RunOptions::new().map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(Self { session, run })
    }

    /// All tensors on CPU f32, shapes as the DiT forward.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        x: &Tensor,
        mask: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        spks: &Tensor,
        cond: &Tensor,
    ) -> Result<Tensor> {
        let to_vec = |tt: &Tensor| -> Result<Vec<f32>> { tt.flatten_all()?.to_vec1() };
        let t_vec: Vec<f32> = t.flatten_all()?.to_vec1()?;
        let outputs = self
            .session
            .run_with_options(
                ort::inputs![
                    "x" => OrtTensor::from_array((x.dims(), to_vec(x)?))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?,
                    "mask" => OrtTensor::from_array((mask.dims(), to_vec(mask)?))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?,
                    "mu" => OrtTensor::from_array((mu.dims(), to_vec(mu)?))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?,
                    "t" => OrtTensor::from_array(([t_vec.len()], t_vec))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?,
                    "spks" => OrtTensor::from_array((spks.dims(), to_vec(spks)?))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?,
                    "cond" => OrtTensor::from_array((cond.dims(), to_vec(cond)?))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?
                ],
                &self.run,
            )
            .map_err(|e| candle_core::Error::Msg(format!("estimator run: {e}")))?;
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let dims: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
        Tensor::from_vec(data.to_vec(), dims, &Device::Cpu)
    }
}
