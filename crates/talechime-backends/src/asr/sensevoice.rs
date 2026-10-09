use kaldi_fbank::{FbankOptions, OnlineFbank, WindowType};
use ort::{
    session::{RunOptions, Session},
    value::Tensor,
};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};
pub(super) struct SenseVoice {
    session: Session,
    tokens: Vec<String>,
    mean: Vec<f32>,
    stddev: Vec<f32>,
    width: usize,
    shift: usize,
    language: i32,
    norm: i32,
}
impl SenseVoice {
    pub(super) fn load(directory: &Path, threads: usize) -> anyhow::Result<Self> {
        anyhow::ensure!((1..=32).contains(&threads), "invalid ASR threads");
        let session = Session::builder()?
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .commit_from_file(directory.join("model.int8.onnx"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let metadata = session.metadata()?;
        let get = |key| {
            metadata
                .custom(key)
                .ok_or_else(|| anyhow::anyhow!("missing SenseVoice metadata: {key}"))
        };
        let mean = get("neg_mean")?
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<f32>, _>>()?;
        let stddev = get("inv_stddev")?
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<f32>, _>>()?;
        let width = get("lfr_window_size")?.parse()?;
        let shift = get("lfr_window_shift")?.parse()?;
        let language = get("lang_zh")?.parse()?;
        let norm = get("without_itn")?.parse()?;
        anyhow::ensure!(
            width == 7 && shift == 6 && mean.len() == 560 && stddev.len() == 560,
            "unsupported SenseVoice frontend"
        );
        let mut tokens = Vec::new();
        for line in std::fs::read_to_string(directory.join("tokens.txt"))?.lines() {
            let (text, id) = line
                .rsplit_once(' ')
                .ok_or_else(|| anyhow::anyhow!("invalid token"))?;
            let id: usize = id.parse()?;
            anyhow::ensure!(id == tokens.len(), "nonsequential ASR tokens");
            tokens.push(text.into());
        }
        drop(metadata);
        Ok(Self {
            session,
            tokens,
            mean,
            stddev,
            width,
            shift,
            language,
            norm,
        })
    }
    pub(super) fn transcribe(
        &mut self,
        samples: &[f32],
        cancelled: &AtomicBool,
        run_options: &RunOptions,
    ) -> anyhow::Result<String> {
        let fbank = frontend(samples);
        let frames = fbank.num_frames_ready() as usize;
        if frames == 0 {
            return Ok(String::new());
        }
        let length = frames.div_ceil(self.shift);
        let dim = self.width * 80;
        let mut features = Vec::with_capacity(length * dim);
        for i in 0..length {
            for context in 0..self.width {
                let at = (i * self.shift + context)
                    .saturating_sub((self.width - 1) / 2)
                    .min(frames - 1);
                for (bin, x) in fbank.get_frame(at as i32).iter().enumerate() {
                    let col = context * 80 + bin;
                    features.push((x + self.mean[col]) * self.stddev[col]);
                }
            }
        }
        anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "ASR cancelled");
        // ORT's single bounded Run completes on its owner; no overlapping use of session.
        let output = self.session.run_with_options(
            ort::inputs![
                "x"=>Tensor::from_array(([1,length,dim],features))?,
                "x_length"=>Tensor::from_array(([1],vec![length as i32]))?,
                "language"=>Tensor::from_array(([1],vec![self.language]))?,
                "text_norm"=>Tensor::from_array(([1],vec![self.norm]))?
            ],
            run_options,
        )?;
        anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "ASR cancelled");
        let (shape, logits) = output[0].try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            shape.len() == 3 && shape[0] == 1 && shape[2] as usize == self.tokens.len(),
            "unexpected ASR logits"
        );
        let mut last = usize::MAX;
        let mut text = String::new();
        for row in logits.chunks_exact(self.tokens.len()) {
            anyhow::ensure!(row.iter().all(|x| x.is_finite()), "nonfinite ASR logits");
            let token = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|x| x.0)
                .ok_or_else(|| anyhow::anyhow!("empty ASR logits"))?;
            if token != last && token != 0 {
                let piece = &self.tokens[token];
                if !piece.starts_with("<|") {
                    text.push_str(piece);
                }
            }
            last = token;
        }
        Ok(text.replace('▁', " ").trim().into())
    }
}

fn frontend(samples: &[f32]) -> OnlineFbank {
    let mut options = FbankOptions::default();
    options.frame_opts.dither = 0.0;
    options.frame_opts.snip_edges = true;
    options.frame_opts.window_type = WindowType::Hamming;
    options.mel_opts.num_bins = 80;
    options.mel_opts.high_freq = 0.0;
    let mut fbank = OnlineFbank::new(options);
    let scaled: Vec<f32> = samples.iter().map(|x| x * 32768.0).collect();
    fbank.accept_waveform(16000.0, &scaled);
    fbank.input_finished();
    fbank
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn kaldi_frontend_matches_independent_fixed_numeric_golden() {
        // Generated with kaldi-native-fbank 1.22.3, not this Rust implementation.
        let golden: Vec<Vec<f32>> =
            serde_json::from_str(include_str!("fbank-golden.json")).unwrap();
        let samples: Vec<f32> = (0..640)
            .map(|i| ((i * 37) % 32768 - 16384) as f32 / 32768.0)
            .collect();
        let fbank = frontend(&samples);
        assert_eq!(fbank.num_frames_ready() as usize, golden.len());
        for (i, frame) in golden.iter().enumerate() {
            for (actual, expected) in fbank.get_frame(i as i32).iter().zip(frame) {
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "{i}: {actual} vs {expected}"
                );
            }
        }
    }
}
