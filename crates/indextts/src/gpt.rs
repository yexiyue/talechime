//! UnifiedVoice GPT (`indextts/gpt/model_v2.py`) in the 2.5 configuration
//! (`spk_cond_mode="campplus"`).
//!
//! Speaker conditioning = CAMPPlus 192-d embedding projected by
//! `spk_emb_proj` to the model dim; emotion conditioning = conformer
//! encoder + single-latent perceiver, then `emovec_layer` + `emo_layer`.
//! The conditioning prefix fed to the GPT is
//! `[spk + emo_vec, 0, 0]` followed by `[pad*][text+lang emb][start_mel]`.
//!
//! Not ported (absent from the 2.5 checkpoint in campplus mode):
//! speaker ConditioningEncoder / speaker PerceiverResampler / speed_emb /
//! train-time heads other than mel (text_head weights exist in the
//! checkpoint but are unused during inference).
//!
//! Wire conventions handed to [`crate::generation`]: `fake_inputs` is i64
//! `(1, target_len + 1)` ending in `start_mel_token`, the attention mask is
//! u8 `(1, target_len + 1)` with 1 = valid, and padding sits LEFT of the
//! `[cond][text]` region exactly like upstream `prepare_gpt_inputs`.

use crate::config::GptConfig;
use crate::conformer::ConformerEncoder;
use crate::generation::{GenerateConfig, InferenceParts};
use crate::gpt2::{Gpt2Config, Gpt2Model};
use crate::perceiver::PerceiverResampler;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};
use rand::rngs::StdRng;

/// Hidden size of the w2v-bert 2.0 features feeding the emotion encoder
/// (`input_size=1024` hardcoded in `UnifiedVoice.__init__`).
const EMO_INPUT_SIZE: usize = 1024;

/// Output dim of the emotion perceiver (its `dim` argument), which feeds
/// `emovec_layer: Linear(1024, model_dim)`.
const EMO_PERCEIVER_DIM: usize = 1024;

/// `nn.LayerNorm` default epsilon (matches `final_norm`).
const LN_EPS: f64 = 1e-5;

/// `lang_embedding` rows: `len(LANGUAGE_DICT) + 1` — the tokenizer's
/// `LANGUAGES` dict holds 106 distinct keys (its literal repeats
/// "common").
const LANG_EMBED_ROWS: usize = 107;

pub struct UnifiedVoice {
    spk_emb_proj: Linear,
    emo_conditioning_encoder: ConformerEncoder,
    emo_perceiver_encoder: PerceiverResampler,
    emovec_layer: Linear,
    emo_layer: Linear,
    text_embedding: Embedding,
    lang_embedding: Embedding,
    /// `text_pos_embedding.emb`
    text_pos_embedding: Embedding,
    mel_embedding: Embedding,
    /// `mel_pos_embedding.emb`
    mel_pos_embedding: Embedding,
    gpt: Gpt2Model,
    final_norm: LayerNorm,
    mel_head: Linear,
    model_dim: usize,
    start_mel_token: u32,
    stop_mel_token: u32,
    start_text_token: u32,
    stop_text_token: u32,
}

impl UnifiedVoice {
    /// `vb` rooted at the gpt.pth state dict (keys like
    /// `emo_conditioning_encoder.*`, `text_embedding.weight`, `gpt.h.*`,
    /// `mel_pos_embedding.emb.weight`, ...).
    pub fn load(vb: &VarBuilder, cfg: &GptConfig, device: &Device) -> Result<Self> {
        let m = &cfg.emo_condition_module;
        let emo_conditioning_encoder = ConformerEncoder::load(
            &vb.pp("emo_conditioning_encoder"),
            EMO_INPUT_SIZE,
            m.output_size,
            m.linear_units,
            m.attention_heads,
            m.num_blocks,
            &m.input_layer,
        )?;
        let emo_perceiver_encoder = PerceiverResampler::load(
            &vb.pp("emo_perceiver_encoder"),
            EMO_PERCEIVER_DIM,
            m.output_size,
            m.perceiver_mult,
            m.attention_heads,
            1,
        )?;

        let spk_emb_proj = candle_nn::linear(192, cfg.model_dim, vb.pp("spk_emb_proj"))?;
        let emovec_layer =
            candle_nn::linear(EMO_PERCEIVER_DIM, cfg.model_dim, vb.pp("emovec_layer"))?;
        let emo_layer = candle_nn::linear(cfg.model_dim, cfg.model_dim, vb.pp("emo_layer"))?;

        let text_embedding = Embedding::new(
            vb.get(
                (cfg.number_text_tokens + 1, cfg.model_dim),
                "text_embedding.weight",
            )?,
            cfg.model_dim,
        );
        let lang_embedding = Embedding::new(
            vb.get((LANG_EMBED_ROWS, cfg.model_dim), "lang_embedding.weight")?,
            cfg.model_dim,
        );
        let text_pos_embedding = Embedding::new(
            vb.get(
                (cfg.max_text_tokens + 2, cfg.model_dim),
                "text_pos_embedding.emb.weight",
            )?,
            cfg.model_dim,
        );
        let mel_embedding = Embedding::new(
            vb.get(
                (cfg.number_mel_codes, cfg.model_dim),
                "mel_embedding.weight",
            )?,
            cfg.model_dim,
        );
        let mel_pos_embedding = Embedding::new(
            vb.get(
                (
                    cfg.max_mel_tokens + 2 + cfg.max_conditioning_inputs,
                    cfg.model_dim,
                ),
                "mel_pos_embedding.emb.weight",
            )?,
            cfg.model_dim,
        );

        // `build_hf_gpt_transformer`: n_positions = (max_mel_tokens + 2 +
        // max_conditioning_inputs) + (max_text_tokens + 2).
        let max_mel_seq = cfg.max_mel_tokens + 2 + cfg.max_conditioning_inputs;
        let max_text_seq = cfg.max_text_tokens + 2;
        let gpt_cfg = Gpt2Config::for_indextts(
            cfg.model_dim,
            cfg.layers,
            cfg.heads,
            max_mel_seq + max_text_seq,
        );
        let gpt = Gpt2Model::load(&vb.pp("gpt"), &gpt_cfg, device)?;

        let final_norm = candle_nn::layer_norm(cfg.model_dim, LN_EPS, vb.pp("final_norm"))?;
        let mel_head = candle_nn::linear(cfg.model_dim, cfg.number_mel_codes, vb.pp("mel_head"))?;

        Ok(Self {
            spk_emb_proj,
            emo_conditioning_encoder,
            emo_perceiver_encoder,
            emovec_layer,
            emo_layer,
            text_embedding,
            lang_embedding,
            text_pos_embedding,
            mel_embedding,
            mel_pos_embedding,
            gpt,
            final_norm,
            mel_head,
            model_dim: cfg.model_dim,
            start_mel_token: cfg.start_mel_token,
            stop_mel_token: cfg.stop_mel_token,
            start_text_token: cfg.start_text_token,
            stop_text_token: cfg.stop_text_token,
        })
    }

    /// `get_emovec`: emo conformer → emo perceiver → `emovec_layer` →
    /// `emo_layer`. `cond`: `(1, 1024, T)` w2v-bert features (the Python
    /// `.transpose(1, 2)` happens inside). Returns `(1, model_dim)`.
    fn get_emovec(&self, cond: &Tensor) -> Result<Tensor> {
        // `cond` is already (B, T, 1024) — upstream's two `.transpose(1, 2)`
        // calls (caller + `get_emo_conditioning`) cancel out.
        let x = cond.contiguous()?;
        let (b, t, _) = x.dims3()?;
        let ilens = Tensor::from_vec(vec![t as i64; b], (b,), x.device())?;
        let (enc_out, mask) = self.emo_conditioning_encoder.forward(&x, &ilens)?;
        let mask = mask.squeeze(1)?; // (1, T')
        let conds = self.emo_perceiver_encoder.forward(&enc_out, &mask)?; // (1, 1, 1024)
        let conds = conds.squeeze(1)?; // (1, 1024)
        let v = self.emovec_layer.forward(&conds)?;
        self.emo_layer.forward(&v)
    }

    /// `base + alpha * (emo - base)` where each vector comes from
    /// `get_emovec` (emo conformer+perceiver → emovec_layer → emo_layer).
    /// Returns `(1, model_dim)`.
    pub fn merge_emovec(
        &self,
        spk_cond_emb: &Tensor,
        emo_cond_emb: &Tensor,
        alpha: f64,
    ) -> Result<Tensor> {
        let base = self.get_emovec(spk_cond_emb)?;
        let emo = self.get_emovec(emo_cond_emb)?;
        let diff = (&emo - &base)? * alpha;
        (&base + diff)?.contiguous()
    }

    /// `merge_emovec(x, x, _)` collapses to `get_emovec(x)` bit-exactly
    /// (`emo - base == 0` exactly), so callers with a single reference clip
    /// can run the emotion conformer once instead of twice.
    pub fn emovec_from_reference(&self, cond: &Tensor) -> Result<Tensor> {
        self.get_emovec(cond)
    }

    /// One inference call for a text segment (num_return_sequences = 1,
    /// no `input_tokens`). `text_tokens`: `(1, L)` i64 already padded with
    /// the trailing pad token (id 1) as in `infer_generator`; `emo_vec`:
    /// `(1, model_dim)` from [`Self::merge_emovec`]; `campplus_style`:
    /// `(1, 192)`. Returns generated mel-code ids `(1, n)` INCLUDING any
    /// trailing stop token; the caller slices at `stop_mel_token`.
    #[allow(clippy::too_many_arguments)]
    pub fn inference_speech(
        &self,
        text_tokens: &Tensor,
        lang_id: u32,
        emo_vec: &Tensor,
        campplus_style: &Tensor,
        cfg: &GenerateConfig,
        rng: &mut StdRng,
    ) -> Result<Tensor> {
        self.inference_speech_with_cancel(
            text_tokens,
            lang_id,
            emo_vec,
            campplus_style,
            cfg,
            rng,
            &|| false,
        )
    }

    /// Synthesize semantic codes while checking for cancellation between tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn inference_speech_with_cancel(
        &self,
        text_tokens: &Tensor,
        lang_id: u32,
        emo_vec: &Tensor,
        campplus_style: &Tensor,
        cfg: &GenerateConfig,
        rng: &mut StdRng,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Tensor> {
        // conds = [spk + emo_vec, 0, 0]  (3 conditioning positions)
        let conds_latent = self.campplus_conditioning(campplus_style, emo_vec)?;

        let (fake_inputs, mel_emb, attention_mask) =
            self.prepare_gpt_inputs(&conds_latent, text_tokens, lang_id)?;

        let parts = self.inference_parts();
        crate::generation::generate_mel_tokens_with_cancel(
            &parts,
            &fake_inputs,
            &mel_emb,
            &attention_mask,
            self.stop_mel_token,
            self.start_mel_token,
            cfg,
            rng,
            cancelled,
        )
    }

    /// Parts handed to [`crate::generation::generate_mel_tokens`].
    pub fn inference_parts(&self) -> InferenceParts<'_> {
        InferenceParts {
            gpt: &self.gpt,
            embeddings: &self.mel_embedding,
            mel_pos_embedding: &self.mel_pos_embedding,
            final_norm: &self.final_norm,
            mel_head: &self.mel_head,
        }
    }

    /// `[spk + emo_vec, 0, 0]` conditioning from a CAMPPlus style row and an
    /// emotion vector (`(1, 192)` / `(1, model_dim)`).
    pub(crate) fn campplus_conditioning(
        &self,
        campplus_style: &Tensor,
        emo_vec: &Tensor,
    ) -> Result<Tensor> {
        let spk = self
            .spk_emb_proj
            .forward(&campplus_style.to_dtype(DType::F32)?)?
            .unsqueeze(1)?;
        let emo = emo_vec.to_dtype(DType::F32)?.unsqueeze(1)?;
        let cond0 = spk.broadcast_add(&emo)?;
        let zeros = Tensor::zeros((1, 2, self.model_dim), DType::F32, spk.device())?;
        Tensor::cat(&[&cond0, &zeros], 1)
    }

    /// Port of `prepare_gpt_inputs` (batch size 1, `single_cond` case).
    /// Returns `(fake_inputs (1, target_len + 1) i64, prompt embeds
    /// (1, target_len, dim), attention_mask (1, target_len + 1) u8)`.
    pub(crate) fn prepare_gpt_inputs(
        &self,
        conditional_latents: &Tensor,
        text_inputs: &Tensor,
        lang_id: u32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        prepare_gpt_input_embeddings(
            &self.text_embedding,
            &self.text_pos_embedding,
            &self.lang_embedding,
            conditional_latents,
            text_inputs,
            lang_id,
            self.start_text_token,
            self.stop_text_token,
            self.start_mel_token,
            self.model_dim,
        )
    }
}

/// Free (testable without the GPT backbone) port of
/// `UnifiedVoice.prepare_gpt_inputs` for the batch-1 campplus path.
///
/// Mirrors model_v2.py exactly: strip start/stop text tokens (which also
/// removes the right-side pad-1 padding), re-add `[start] ... [stop]`,
/// `text_emb = text_embedding + text_pos_embedding.emb(positions) +
/// lang_embedding(lang)` (broadcast over positions), and prepend zero
/// embeds plus a zero attention-mask prefix when `L + 2` exceeds the
/// rebuilt text length (padding sits LEFT of the conditioning block).
#[allow(clippy::too_many_arguments)]
fn prepare_gpt_input_embeddings(
    text_embedding: &Embedding,
    text_pos_embedding: &Embedding,
    lang_embedding: &Embedding,
    conditional_latents: &Tensor,
    text_inputs: &Tensor,
    lang_id: u32,
    start_text_token: u32,
    stop_text_token: u32,
    start_mel_token: u32,
    model_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let dims = text_inputs.dims();
    if dims.len() != 2 || dims[0] != 1 {
        candle_core::bail!("text_inputs must be (1, L), got {dims:?}");
    }
    let l = dims[1];
    if conditional_latents.dims()[0] != 1 || conditional_latents.dims().len() != 3 {
        candle_core::bail!(
            "conditional_latents must be (1, cond, dim), got {:?}",
            conditional_latents.dims()
        );
    }
    let dev = text_inputs.device();

    // valid = row != stop/start text tokens (stop == the pad id 1 used by
    // `F.pad(..., value=1)` in the caller).
    let row = text_inputs
        .to_dtype(DType::I64)?
        .flatten_all()?
        .to_vec1::<i64>()?;
    let valid: Vec<i64> = row
        .into_iter()
        .filter(|&t| t != stop_text_token as i64 && t != start_text_token as i64)
        .collect();
    let mut text_input: Vec<i64> = Vec::with_capacity(valid.len() + 2);
    text_input.push(start_text_token as i64);
    text_input.extend_from_slice(&valid);
    text_input.push(stop_text_token as i64);
    let text_len = text_input.len();

    let text_input = Tensor::from_vec(text_input, (text_len,), dev)?;
    let positions = Tensor::arange(0i64, text_len as i64, dev)?;
    let text_emb = text_embedding.forward(&text_input)?;
    let pos_emb = text_pos_embedding.forward(&positions)?;
    let lang_row = lang_embedding.embeddings().narrow(0, lang_id as usize, 1)?;
    let text_emb = text_emb.broadcast_add(&pos_emb)?.broadcast_add(&lang_row)?;

    let conds = conditional_latents.squeeze(0)?; // (cond, dim)
    let target_len = conditional_latents.dim(1)? + l + 2;
    let padding = (l + 2).saturating_sub(text_len);

    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if padding > 0 {
        parts.push(Tensor::zeros((padding, model_dim), DType::F32, dev)?);
    }
    parts.push(conds);
    parts.push(text_emb);
    let mel_emb = Tensor::cat(&parts, 0)?.unsqueeze(0)?;
    if mel_emb.dims()[1] != target_len {
        candle_core::bail!(
            "mel_emb length {} != target_len {target_len}",
            mel_emb.dims()[1]
        );
    }

    // fake_inputs: all ones with the start mel token appended; the ids
    // themselves are placeholders for HF `generate` bookkeeping.
    let mut fake = vec![1i64; target_len + 1];
    *fake.last_mut().expect("non-empty") = start_mel_token as i64;
    let fake_inputs = Tensor::from_vec(fake, (1, target_len + 1), dev)?;

    let mut mask_vals = vec![0u8; padding];
    mask_vals.resize(target_len + 1, 1u8);
    let attention_mask = Tensor::from_vec(mask_vals, (1, target_len + 1), dev)?;

    Ok((fake_inputs, mel_emb, attention_mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{Init, VarMap};
    use rand::SeedableRng;

    fn embedding_vb(vm: &VarMap, names: &[(&str, usize, usize)]) -> Result<Vec<Embedding>> {
        let dev = Device::Cpu;
        let vb = VarBuilder::from_varmap(vm, DType::F32, &dev);
        let mut out = Vec::new();
        for (name, rows, cols) in names {
            // Embedding weights are (num_embeddings, hidden).
            vm.get((*rows, *cols), name, Init::Const(0.1), DType::F32, &dev)?;
            out.push(Embedding::new(vb.get((*rows, *cols), name)?, *cols));
        }
        Ok(out)
    }

    #[test]
    fn prepare_gpt_inputs_pads_left_and_masks() {
        // L = 5 with tokens [2, 3, 1, 1, 1]: three pad/stop positions are
        // stripped, two real tokens survive -> text becomes [0, 2, 3, 1],
        // padding = 5 + 2 - 4 = 3, target_len = 3 + 5 + 2 = 10.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let emb = embedding_vb(
            &vm,
            &[
                ("text_embedding.weight", 60510, 16),
                ("text_pos_embedding.emb.weight", 602, 16),
                ("lang_embedding.weight", 10, 16),
            ],
        )
        .unwrap();
        let [text, text_pos, lang] = [&emb[0], &emb[1], &emb[2]];

        let conds = Tensor::randn(0f32, 1f32, (1, 3, 16), &dev).unwrap();
        let text_tokens = Tensor::from_vec(vec![2i64, 3, 1, 1, 1], (1, 5), &dev).unwrap();
        let (fake, mel_emb, mask) = prepare_gpt_input_embeddings(
            text,
            text_pos,
            lang,
            &conds,
            &text_tokens,
            2,
            0,
            1,
            8192,
            16,
        )
        .expect("prepare");

        assert_eq!(fake.dims(), vec![1, 11]);
        let f = fake.to_vec2::<i64>().unwrap();
        assert_eq!(f[0][..10], [1i64; 10]);
        assert_eq!(f[0][10], 8192);

        assert_eq!(mel_emb.dims(), vec![1, 10, 16]);
        assert_eq!(mask.dims(), vec![1, 11]);
        let m = mask.to_vec2::<u8>().unwrap()[0].to_vec();
        assert_eq!(&m[..3], &[0, 0, 0]);
        assert!(m[3..].iter().all(|&v| v == 1), "tail must be valid: {m:?}");

        // [pad(3)][cond(3)][text(4)] — the pad region is exactly zero.
        let v = mel_emb.to_vec3::<f32>().unwrap();
        assert!(
            v[0][..3].iter().all(|row| row.iter().all(|&x| x == 0.0)),
            "left padding must be zero"
        );
        // text occupies rows 6..10: row 6 = embedding(start)+pos0+lang,
        // row 9 = embedding(stop)+pos3+lang.
        let te = text
            .embeddings()
            .narrow(0, 0, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(
            (v[0][6][0] - (te[0][0] + 0.1 + 0.1)).abs() < 1e-6,
            "start row mismatch"
        );
        let te_stop = text
            .embeddings()
            .narrow(0, 1, 1)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert!(
            (v[0][9][0] - (te_stop[0][0] + 0.1 + 0.1)).abs() < 1e-6,
            "stop row mismatch"
        );
    }

    #[test]
    fn prepare_gpt_inputs_no_padding() {
        // No pad tokens -> padding = 0 -> all-ones mask, no zero prefix.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let emb = embedding_vb(
            &vm,
            &[
                ("text_embedding.weight", 60510, 8),
                ("text_pos_embedding.emb.weight", 602, 8),
                ("lang_embedding.weight", 10, 8),
            ],
        )
        .unwrap();
        let [text, text_pos, lang] = [&emb[0], &emb[1], &emb[2]];
        let conds = Tensor::randn(0f32, 1f32, (1, 3, 8), &dev).unwrap();
        let text_tokens = Tensor::from_vec(vec![5i64, 6, 7], (1, 3), &dev).unwrap();
        let (fake, mel_emb, mask) = prepare_gpt_input_embeddings(
            text,
            text_pos,
            lang,
            &conds,
            &text_tokens,
            0,
            0,
            1,
            8192,
            8,
        )
        .expect("prepare");
        assert_eq!(mel_emb.dims(), vec![1, 8, 8]); // 3 + 3 + 2
        assert_eq!(fake.dims(), vec![1, 9]);
        assert!(mask.to_vec2::<u8>().unwrap()[0].iter().all(|&v| v == 1));
    }

    #[test]
    fn prepare_gpt_inputs_strips_interior_start_token_too() {
        // A start token inside the row is filtered like upstream.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let emb = embedding_vb(
            &vm,
            &[
                ("text_embedding.weight", 60510, 8),
                ("text_pos_embedding.emb.weight", 602, 8),
                ("lang_embedding.weight", 10, 8),
            ],
        )
        .unwrap();
        let [text, text_pos, lang] = [&emb[0], &emb[1], &emb[2]];
        let conds = Tensor::randn(0f32, 1f32, (1, 3, 8), &dev).unwrap();
        let text_tokens = Tensor::from_vec(vec![0i64, 9, 1], (1, 3), &dev).unwrap();
        let (_, mel_emb, mask) = prepare_gpt_input_embeddings(
            text,
            text_pos,
            lang,
            &conds,
            &text_tokens,
            4,
            0,
            1,
            8192,
            8,
        )
        .expect("prepare");
        // valid = [9] -> text [0, 9, 1], padding = 3 + 2 - 3 = 2
        assert_eq!(mel_emb.dims(), vec![1, 8, 8]);
        assert_eq!(&mask.to_vec2::<u8>().unwrap()[0][..2], &[0, 0]);
    }

    /// Real-weights smoke test. Run with e.g.
    /// `INDEXTTS_CONVERTED_DIR=<dir> cargo test -p talechime-indextts --lib -- --ignored gpt_real`
    /// where `<dir>` contains `gpt.safetensors` (see `weights.rs`). The
    /// `inference_speech` half additionally requires the `gpt2` backbone
    /// and the generation loop to be implemented.
    #[test]
    #[ignore = "requires converted real weights (INDEXTTS_CONVERTED_DIR)"]
    fn gpt_real_weights_smoke() {
        let dir = match std::env::var("INDEXTTS_CONVERTED_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => {
                eprintln!("INDEXTTS_CONVERTED_DIR not set; skipping");
                return;
            }
        };
        let dev = Device::Cpu;
        let tensors = candle_core::safetensors::load(dir.join("gpt.safetensors"), &dev)
            .expect("load gpt.safetensors");
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &dev);
        let cfg = crate::config::GptConfig::default();
        let model = UnifiedVoice::load(&vb, &cfg, &dev).expect("load UnifiedVoice");

        let spk = Tensor::randn(0f32, 1f32, (1, 50, EMO_INPUT_SIZE), &dev).unwrap();
        let emo = Tensor::randn(0f32, 1f32, (1, 37, EMO_INPUT_SIZE), &dev).unwrap();
        let merged = model.merge_emovec(&spk, &emo, 0.7).expect("merge_emovec");
        assert_eq!(merged.dims(), vec![1, cfg.model_dim]);
        assert!(
            merged.to_vec2::<f32>().unwrap()[0]
                .iter()
                .all(|v| v.is_finite())
        );

        let text = Tensor::from_vec(
            {
                let mut v = vec![7i64, 8, 9, 11, 12];
                v.resize(9, 1);
                v
            },
            (1, 9),
            &dev,
        )
        .unwrap();
        let style = Tensor::randn(0f32, 1f32, (1, 192), &dev).unwrap();
        let gen_cfg = GenerateConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let codes = model
            .inference_speech(&text, 0, &merged, &style, &gen_cfg, &mut rng)
            .expect("inference_speech");
        assert_eq!(codes.dims()[0], 1);
        assert!(codes.dims()[1] > 0);
    }

    /// Layer-level oracle for the emotion conditioning path. Run with
    /// `INDEXTTS_CONVERTED_DIR`, `INDEXTTS_ORACLE_JSON` and
    /// `INDEXTTS_EMO_LAYERS` (stage dump produced by the Python hook
    /// script in `target/indextts-oracle`). Prints every stage diff, then
    /// fails at the first divergent stage.
    #[test]
    #[ignore = "requires oracle JSON, converted checkpoints, INDEXTTS_EMO_LAYERS"]
    fn emo_layers_stage_dump_matches_python() {
        let Some(layers_path) = std::env::var_os("INDEXTTS_EMO_LAYERS") else {
            eprintln!("INDEXTTS_EMO_LAYERS not set; skipping");
            return;
        };
        let oracle_path = std::env::var_os("INDEXTTS_ORACLE_JSON").expect("INDEXTTS_ORACLE_JSON");
        let dir = std::env::var_os("INDEXTTS_CONVERTED_DIR").expect("INDEXTTS_CONVERTED_DIR");

        let layers: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&layers_path).unwrap()).unwrap();
        let oracle: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&oracle_path).unwrap()).unwrap();

        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(
            std::path::Path::new(&dir).join("gpt.safetensors"),
            &device,
        )
        .expect("load gpt.safetensors");
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let config = crate::config::Config::load(&std::path::Path::new(&dir).join("config.json"))
            .expect("config");
        let model = UnifiedVoice::load(&vb, &config.gpt, &device).expect("load UnifiedVoice");

        let node = &oracle["codec"]["input"];
        let data: Vec<f32> = node["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let shape: Vec<usize> = node["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let cond = Tensor::from_vec(data, shape, &device).unwrap();
        let (b, t, _) = cond.dims3().unwrap();
        let ilens = Tensor::from_vec(vec![t as i64; b], (b,), &device).unwrap();

        fn entry(v: &serde_json::Value) -> (Vec<f32>, Vec<usize>) {
            let data = v["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let shape = v["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            (data, shape)
        }

        let mut report: Vec<(String, f64, f64, usize)> = Vec::new();
        let mut record = |name: &str, got: &Tensor, want: &serde_json::Value| {
            let (want_data, want_shape) = entry(want);
            assert_eq!(got.dims(), want_shape, "{name}: shape mismatch");
            let got_vec = got
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut max_abs = 0f64;
            let mut max_rel = 0f64;
            for (g, w) in got_vec.iter().zip(&want_data) {
                let a = (*g - *w) as f64;
                max_abs = max_abs.max(a.abs());
                max_rel = max_rel.max(a.abs() / w.abs().max(1e-6) as f64);
            }
            report.push((name.to_string(), max_abs, max_rel, want_data.len()));
        };

        // Conformer stages (embed, block internals, after_norm).
        let (stages, masks) = model
            .emo_conditioning_encoder
            .forward_stage_dump(&cond, &ilens)
            .unwrap();
        for (name, tensor) in &stages {
            record(name, tensor, &layers[name.as_str()]);
        }
        record(
            "enc_mask",
            &masks.to_dtype(DType::F32).unwrap(),
            &layers["enc_mask"],
        );

        let enc_out = stages
            .last()
            .and_then(|(n, t)| (n == "after_norm").then_some(t))
            .expect("after_norm stage")
            .clone();
        let conds_mask = {
            let m = masks.squeeze(1).unwrap();
            let ones = Tensor::ones((b, 1), DType::U8, &device).unwrap();
            Tensor::cat(&[&ones, &m], 1).unwrap()
        };
        record(
            "conds_mask",
            &conds_mask.to_dtype(DType::F32).unwrap(),
            &layers["conds_mask"],
        );

        // Perceiver stages + emovec tail.
        let per_stages = model
            .emo_perceiver_encoder
            .forward_stage_dump(&enc_out, &masks.squeeze(1).unwrap())
            .unwrap();
        for (name, tensor) in &per_stages {
            record(name, tensor, &layers[name.as_str()]);
        }
        let per_out = per_stages
            .last()
            .and_then(|(n, t)| (n == "perceiver").then_some(t))
            .expect("perceiver stage")
            .clone();
        let emo_ori = model
            .emovec_layer
            .forward(&per_out.squeeze(1).unwrap())
            .unwrap();
        record("emo_vec_syn_ori", &emo_ori, &layers["emo_vec_syn_ori"]);
        let emovec = model.emo_layer.forward(&emo_ori).unwrap();
        record("emovec", &emovec, &layers["emovec"]);

        // Full public path vs the oracle emovec.
        let merged = model.merge_emovec(&cond, &cond, 1.0).unwrap();
        record("merge_emovec", &merged, &oracle["gpt"]["emovec"]);

        // prepare_gpt_inputs tail (same flow as
        // `validation::gpt_conditioning_and_first_step_logits_match_python`).
        let (style_data, style_shape) = entry(&oracle["campplus"]["style"]);
        let style = Tensor::from_vec(style_data, style_shape, &device).unwrap();
        let (tok_data, tok_shape) = entry(&oracle["gpt"]["text_tokens"]);
        let text_tokens = Tensor::from_vec(
            tok_data.into_iter().map(|v| v as i64).collect::<Vec<_>>(),
            tok_shape,
            &device,
        )
        .unwrap();
        let lang = oracle["gpt"]["lang"].as_u64().unwrap() as u32;
        let conds = model.campplus_conditioning(&style, &merged).unwrap();
        let (_fake, embeds, _mask) = model
            .prepare_gpt_inputs(&conds, &text_tokens, lang)
            .unwrap();
        record("gpt.embeds", &embeds, &oracle["gpt"]["embeds"]);

        for (name, max_abs, max_rel, n) in &report {
            println!("{name}: max_abs {max_abs:.3e}, max_rel {max_rel:.3e} over {n} values");
        }
        for (name, max_abs, max_rel, _) in &report {
            assert!(
                *max_abs <= 5e-4 || *max_rel <= 5e-3,
                "{name}: first divergent stage (max_abs {max_abs:.3e}, max_rel {max_rel:.3e})"
            );
        }
    }
}
