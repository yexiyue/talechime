#![cfg(test)]
//! Real-model numeric oracles against the Python reference dump.
//!
//! All tests are `#[ignore]`d: they need `INDEXTTS_CONVERTED_DIR` (converted
//! checkpoints) and `INDEXTTS_ORACLE_JSON` (produced by
//! the historical IndexTTS oracle exporter (Git revision `e591c94`) from the pinned Python checkout).
//! Inputs are taken verbatim from the oracle JSON so the comparison
//! isolates model math from DSP front-ends (resampling, feature
//! extraction); those are covered by module unit tests. Tolerances follow
//! the voxcpm acceptance record: failures are recorded, never loosened.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;
use serde_json::Value;

use crate::campplus::CampPlus;
use crate::codec::SemanticCodec;
use crate::config::Config;
use crate::gpt::UnifiedVoice;
use crate::mel::mel_spectrogram;
use crate::s2mel::S2Mel;
use crate::tokenizer::Tokenizer;
use crate::w2v_bert::Wav2Vec2Bert;

struct Oracle {
    root: Value,
}

impl Oracle {
    fn load() -> Option<Self> {
        let path = std::env::var_os("INDEXTTS_ORACLE_JSON")?;
        let text = std::fs::read_to_string(path).ok()?;
        Some(Self {
            root: serde_json::from_str(&text).ok()?,
        })
    }

    fn data(&self, path: &[&str]) -> Vec<f32> {
        let mut node = &self.root;
        for key in path {
            node = &node[*key];
        }
        node["data"]
            .as_array()
            .expect("oracle data array")
            .iter()
            .map(|v| v.as_f64().expect("f64") as f32)
            .collect()
    }

    fn shape(&self, path: &[&str]) -> Vec<usize> {
        let mut node = &self.root;
        for key in path {
            node = &node[*key];
        }
        node["shape"]
            .as_array()
            .expect("oracle shape")
            .iter()
            .map(|v| v.as_u64().expect("usize") as usize)
            .collect()
    }

    fn tensor(&self, path: &[&str], device: &Device) -> Result<Tensor> {
        let data = self.data(path);
        let shape = self.shape(path);
        Tensor::from_vec(data, shape, device)
    }

    fn ids(&self, path: &[&str]) -> Vec<u32> {
        self.data(path).iter().map(|v| *v as u32).collect()
    }
}

fn converted_dir() -> std::path::PathBuf {
    std::env::var_os("INDEXTTS_CONVERTED_DIR")
        .expect("INDEXTTS_CONVERTED_DIR")
        .into()
}

/// Reports max abs diff and relative diff; returns Ok when inside bounds.
fn assert_close(name: &str, got: &Tensor, want_data: &[f32], atol: f64, rtol: f64) {
    let got = got.to_dtype(DType::F32).unwrap();
    let got_vec = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(
        got_vec.len(),
        want_data.len(),
        "{name}: length mismatch (got {}, want {})",
        got_vec.len(),
        want_data.len()
    );
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    for (g, w) in got_vec.iter().zip(want_data) {
        let d = (*g - *w) as f64;
        let a = d.abs();
        let r = a / w.abs().max(1e-6) as f64;
        max_abs = max_abs.max(a);
        max_rel = max_rel.max(r);
        assert!(
            a <= atol || r <= rtol,
            "{name}: diff {a} (rel {r}) exceeds atol {atol}/rtol {rtol} near value {w}"
        );
    }
    println!(
        "{name}: max_abs {max_abs:.3e}, max_rel {max_rel:.3e} over {} values",
        want_data.len()
    );
}

fn require_oracle() -> Oracle {
    match Oracle::load() {
        Some(o) => o,
        None => panic!("set INDEXTTS_ORACLE_JSON and INDEXTTS_CONVERTED_DIR"),
    }
}

#[test]
#[ignore = "requires oracle JSON + converted checkpoints"]
fn tokenizer_matches_python() {
    let oracle = require_oracle();
    let dir = converted_dir();
    let tok = Tokenizer::from_tiktoken_file(&dir.join("multilingual_zh_ja_yue_char_del.tiktoken"))
        .unwrap();
    let texts = oracle.root["tokenizer"]["texts"].as_array().unwrap();
    let ids = oracle.root["tokenizer"]["ids"].as_array().unwrap();
    for (i, text) in texts.iter().enumerate() {
        let text = text.as_str().unwrap();
        let got = tok.encode(text);
        let want: Vec<u32> = ids[i]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as u32)
            .collect();
        assert_eq!(got, want, "corpus entry {i}: {text}");
    }
}

#[test]
#[ignore = "requires oracle JSON + converted checkpoints"]
fn w2v_campplus_mel_codec_regulator_match_python() {
    let oracle = require_oracle();
    let dir = converted_dir();
    let device = Device::Cpu;
    let weights = crate::weights::Weights::load_dir(&dir, &device).unwrap();
    let config = Config::load(&dir.join("config.json")).unwrap();

    // --- w2v hidden-17 on the Python-side extracted features ---
    let w2v = Wav2Vec2Bert::load(&dir.join("w2v-bert-2.0"), &device).unwrap();
    let input = oracle.tensor(&["w2v", "input_features"], &device).unwrap();
    let mask = oracle.tensor(&["w2v", "attention_mask"], &device).unwrap();
    let h17 = w2v.hidden_17(&input, &mask).unwrap();
    assert_close(
        "w2v.hidden_17",
        &h17,
        &oracle.data(&["w2v", "hidden_17"]),
        5e-4,
        5e-3,
    );
    let normalized = h17
        .broadcast_sub(&weights.feats.w2v_mean)
        .unwrap()
        .broadcast_div(&weights.feats.w2v_std)
        .unwrap();
    assert_close(
        "w2v.normalized",
        &normalized,
        &oracle.data(&["w2v", "normalized"]),
        5e-4,
        5e-3,
    );

    // --- campplus style on the Python-side fbank ---
    let campplus = CampPlus::load(&weights.campplus).unwrap();
    let fbank = oracle
        .tensor(&["campplus", "fbank_mean_sub"], &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let style = campplus.forward(&fbank).unwrap();
    assert_close(
        "campplus.style",
        &style,
        &oracle.data(&["campplus", "style"]),
        1e-4,
        1e-3,
    );

    // --- reference mel on the Python-side 22 kHz PCM ---
    let pcm = oracle.data(&["reference", "pcm_22k"]);
    let wav = Tensor::from_vec(pcm.clone(), (1, pcm.len()), &device).unwrap();
    let mel = mel_spectrogram(
        &wav,
        &config.s2mel.preprocess_params.spect_params,
        config.s2mel.preprocess_params.sr,
        false,
    )
    .unwrap();
    assert_close(
        "mel.ref_mel",
        &mel,
        &oracle.data(&["mel", "ref_mel"]),
        1e-4,
        1e-3,
    );

    // --- codec quantize/decode on the Python-side conditioning ---
    let codec = SemanticCodec::load(&weights.codec, &config.semantic_codec).unwrap();
    let emb = oracle.tensor(&["codec", "input"], &device).unwrap();
    let (codes, _quantized) = codec.quantize(&emb).unwrap();
    let got_codes = codes.flatten_all().unwrap().to_vec1::<u32>().unwrap();
    let want_codes = oracle.ids(&["codec", "codes"]);
    let mismatch = got_codes
        .iter()
        .zip(&want_codes)
        .filter(|(a, b)| a != b)
        .count();
    // Nearest-neighbour ties may flip; allow a tiny fraction but report it.
    let frac = mismatch as f64 / want_codes.len().max(1) as f64;
    println!(
        "codec codes: {mismatch}/{} mismatched ({frac:.4})",
        want_codes.len()
    );
    assert!(frac < 1e-3, "codec code mismatch fraction {frac}");
    let decoded = codec.decode(&codes).unwrap();
    assert_close(
        "codec.decoded",
        &decoded,
        &oracle.data(&["codec", "decoded"]),
        5e-4,
        5e-3,
    );

    // --- length regulator prompt condition ---
    let s2mel = S2Mel::load(&weights.s2mel, &config.s2mel).unwrap();
    let ylens = [oracle.shape(&["mel", "ref_mel"])[2]];
    let cond = s2mel.length_regulator.forward(&emb, &ylens).unwrap();
    assert_close(
        "regulator.prompt_condition",
        &cond,
        &oracle.data(&["regulator", "prompt_condition"]),
        5e-4,
        5e-3,
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA + oracle/checkpoints + NVIDIA_TF32_OVERRIDE=0"]
fn codec_decode_matches_python_cuda_f32() {
    // The oracle is CPU F32. Set this before starting the process so
    // cuDNN/cuBLAS use full F32 throughout, not only in the depthwise op.
    assert_eq!(std::env::var("NVIDIA_TF32_OVERRIDE").as_deref(), Ok("0"));
    let oracle = require_oracle();
    let dir = converted_dir();
    let device = Device::new_cuda(0).unwrap();
    let weights = crate::weights::Weights::load_dir(&dir, &device).unwrap();
    let config = Config::load(&dir.join("config.json")).unwrap();
    let codec = SemanticCodec::load(&weights.codec, &config.semantic_codec).unwrap();
    // Use Python's codes directly to isolate decoding from quantization.
    let codes = Tensor::from_vec(
        oracle.ids(&["codec", "codes"]),
        oracle.shape(&["codec", "codes"]),
        &device,
    )
    .unwrap();
    let decoded = codec.decode(&codes).unwrap();
    assert_eq!(decoded.dims(), oracle.shape(&["codec", "decoded"]));
    assert_close(
        "codec.decoded.cuda",
        &decoded,
        &oracle.data(&["codec", "decoded"]),
        5e-4,
        5e-3,
    );
}

#[test]
#[ignore = "requires oracle JSON + converted checkpoints"]
fn gpt_conditioning_and_first_step_logits_match_python() {
    let oracle = require_oracle();
    let dir = converted_dir();
    let device = Device::Cpu;
    let weights = crate::weights::Weights::load_dir(&dir, &device).unwrap();
    let config = Config::load(&dir.join("config.json")).unwrap();
    let gpt = UnifiedVoice::load(&weights.gpt, &config.gpt, &device).unwrap();

    // merge_emovec with the oracle conditioning inputs (spk == emo here).
    let spk = oracle.tensor(&["codec", "input"], &device).unwrap();
    let emovec = gpt.merge_emovec(&spk, &spk, 1.0).unwrap();
    assert_close(
        "gpt.emovec",
        &emovec,
        &oracle.data(&["gpt", "emovec"]),
        5e-4,
        5e-3,
    );

    // First-step tensors via the conditioning path on the oracle inputs.
    let style = oracle.tensor(&["campplus", "style"], &device).unwrap();
    let text_tokens = oracle.tensor(&["gpt", "text_tokens"], &device).unwrap();
    let lang = oracle.root["gpt"]["lang"].as_u64().unwrap() as u32;
    let conds = gpt.campplus_conditioning(&style, &emovec).unwrap();
    let (fake_ids, embeds, attn_mask) = gpt.prepare_gpt_inputs(&conds, &text_tokens, lang).unwrap();
    assert_close(
        "gpt.embeds",
        &embeds,
        &oracle.data(&["gpt", "embeds"]),
        5e-4,
        5e-3,
    );
    let parts = gpt.inference_parts();
    let start = fake_ids
        .narrow(1, embeds.dim(1).unwrap(), 1)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap();
    let start_embed = parts
        .embeddings
        .forward(&start)
        .unwrap()
        .broadcast_add(
            &parts
                .mel_pos_embedding
                .embeddings()
                .narrow(0, 0, 1)
                .unwrap()
                .unsqueeze(0)
                .unwrap(),
        )
        .unwrap();
    let input = Tensor::cat(&[&embeds, &start_embed], 1).unwrap();
    let mut cache = crate::gpt2::Gpt2Cache::new(parts.gpt.layer_count());
    let hidden = parts
        .gpt
        .forward(&input, Some(&attn_mask), &mut cache)
        .unwrap();
    let last = hidden.narrow(1, hidden.dim(1).unwrap() - 1, 1).unwrap();
    let logits = parts
        .mel_head
        .forward(&parts.final_norm.forward(&last).unwrap())
        .unwrap()
        .flatten_all()
        .unwrap();
    assert_close(
        "gpt.first_step_logits",
        &logits,
        &oracle.data(&["gpt", "first_step_logits"]),
        5e-4,
        5e-3,
    );
}
