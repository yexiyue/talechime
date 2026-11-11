# indextts — third-party sources

Native Candle port of IndexTTS 2.5 inference. The Rust implementation in
this crate is Apache-2.0; the adapted Candle depthwise workaround is
identified below. The *semantics* it reproduces and the model weights it consumes come from the
sources below. Model weights are never committed or redistributed by this
repository; converted safetensors produced by the development script stay
under the upstream weight licenses.

## Pinned source revisions

| Component | Source | Pinned revision | License |
| --- | --- | --- | --- |
| Codec CUDA depthwise workaround | https://github.com/huggingface/candle/pull/3531 | commit `70bf01e7a56c8cc8d42649e78564d83930ab2cf7` (adapted locally; runtime remains Candle 0.11.0) | Apache-2.0 (upstream MIT OR Apache-2.0; see `third-party/candle/LICENSE-APACHE`) |
| Inference semantics (GPT v2, codec, s2mel, BigVGAN wrapper, CAMPPlus wrapper, frontend, tokenizer) | https://github.com/index-tts/index-tts | commit `d9e41aac89fd00b3d71497fddb287b7f24613712` (2026-09-30, main) | bilibili Model Use License Agreement in the pinned repo LICENSE (its definition of Model includes published code and weights); not MIT |
| Wav2Vec2-BERT model semantics | HF transformers `src/transformers/models/wav2vec2_bert` | tag `v4.52.1` (matches upstream `transformers==4.52.1`) | Apache-2.0 |
| Kaldi fbank semantics | torchaudio `compliance/kaldi.py` | tag `v2.5.1` | BSD-2-Clause |
| w2v-bert-2.0 weights | https://huggingface.co/facebook/w2v-bert-2.0 | commit `da985ba0987f70aaeb84a80f2851cfac8c697a7b` | MIT |
| CAMPPlus weights (`campplus_cn_common.bin`) | https://huggingface.co/funasr/campplus | commit `e4b6ede7ce16997aff4ae69fbca1f0175e2afede` | Apache-2.0 |
| BigVGAN v2 weights + config | https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x | commit `633ff708ed5b74903e86ff1298cf4a98e921c513` | MIT |
| Main checkpoints (gpt.pth / codec.pth / s2mel.pth / feat1.pt / feat2.pt / wav2vec2bert_stats.pt / tiktoken vocab / config.yaml) | https://huggingface.co/IndexTeam/IndexTTS-2.5 | commit `c39ce5ba981572cb187443877ff559dfb246ce63` | bilibili Model Use License Agreement |

Reference copies of the exact upstream files this port was written against
(dev only, outside this repository): local checkout at
`D:\workspace\index-tts` (git clone at the pinned commit) including
`reference/modeling_wav2vec2_bert.py`, `reference/feature_extraction_seamless_m4t.py`,
`reference/configuration_wav2vec2_bert.py` (transformers v4.52.1),
`reference/torchaudio_kaldi.py` (torchaudio v2.5.1),
`reference/w2v-bert-2.0-config.json`, `reference/bigvgan-config.json`.

## Weight layout

Consumed via the development conversion script
`tools/tts/indextts_export_safetensors.py` (state-dict names preserved
verbatim; f32). See `crates/indextts/src/weights.rs` for the directory
layout. The crate performs no downloads.

## Intentionally not ported

- Speaker ConditioningEncoder / speaker PerceiverResampler / `speed_emb`
  (absent from the 2.5 checkpoint in `spk_cond_mode="campplus"`).
- Arabic TN follows the fixed inference passthrough route.
- DeepSpeed / flash-attn / vLLM / TensorRT acceleration paths
  (`backends/trt`, `indextts/accel`): plain eager math only.
- Training code, `archive/`, streaming `quick_streaming_tokens` mode.

## Resource-backed frontend (2026-11-11, unreleased)

- macOS/Windows CN/EN baseline: `wetext==0.1.0` (Apache-2.0),
  fixed PyPI wheel SHA-256
  `cfc595f85d9b0218df6746a9e6bee9240b6e09ee6d803a937ef90cbc03936b3e`.
  `talechime-text` checks each FST digest and uses native rustfst
  composition, strict F32 path selection and fixed token field ordering. The 20
  independent official `TextNormalizer` fixtures pass on macOS CPU.
  The distinct Linux WeTextProcessing baseline is described below.
- Japanese default ratio=0: fixed `ja_g2p.py`, fugashi 1.4.0 and
  unidic-lite 1.0.8 / UniDic 2.1.2 (BSD-3-Clause;
  `licenses/UNIDIC-BSD`). Vibrato 0.5.3 uses the same lexical/cost
  tables, `ignore_space(true)` and `max_grouping_len(24)`.
  Converted dictionary SHA-256:
  `74f2526bcb8637f387bb688ecd15aced3001b349275e7ec921c68324ebec2ddd`.
  All 14 fixed official segmentation and full-frontend fixtures pass.
  Japanese TN is passthrough; digits and kanji remain unchanged.
- Resource preparation, archive identities, output sizes/digests and
  frontend cache revision are maintained by the backend adapter in
  `crates/talechime-backends/src/indextts/native*`. No runtime Python.
  Missing or mismatched resources fail; no approximate numeric fallback.

Official frontend fixtures, successful synthesis and human listening are
separate gates. Four Japanese adapter WAVs have been generated with front-v5;
listening is pending and the complete front-v6 matrix is in progress.

### Linux and Spanish normalization

The adapter bundles a 5 MB compressed finite-state grammar pack (not neural
weights), exported by `tools/tts/indextts_fst_export.py` from
WeTextProcessing 1.3.0, nemo_text_processing 1.2.0 and pynini 2.1.6.post1.
Both source trees, the archive, all output FSTs and Apache-2.0 licenses have
fixed hashes in the adapter manifest. The native executor preserves strict
F32 cost distinctions: rustfst 1.3.1's TropicalWeight equality tolerance
(1/1024) is too broad for these graphs' 1e-5..1e-4 preferences.
CN/EN fixtures: 20 per platform baseline; Spanish fixtures: 16. All pass
on macOS CPU against independent official Python outputs.
The Linux baseline was generated in a Linux x86_64 development container.
This does not validate the full application on other operating systems.

## QwenEmotion (2026-11-11, unreleased)

Six classifier resources under `qwen0.6bemo4-merge/` are pinned to the main
model revision above, with size and SHA-256 checks at preparation and direct
classifier loading. Their upstream model license still applies. The actual
template is System/Human/Assistant, not generic Qwen ChatML. Native Candle
Qwen3 inference uses greedy decoding, EOS 151643 and the fixed eight-score
conversion. Text classification takes priority over a vector, which takes
priority over a separate emotion reference. The classifier result goes
directly to inference without the optional `normalize_emo_vec` scaling.

Twelve independent conversion fixtures and four actual model cases match
the pinned official implementation on macOS CPU: prompt token IDs and
ordered vectors agree. This does not assert identical logits, generated
token IDs, downstream speech or Metal/CUDA numeric equivalence. The oracle
receipt records reference device/dtype. Reproduce with
`tools/tts/indextts_emotion_model_oracle.py` and the explicitly ignored
`fixed_classifier_matches_official_model_outputs` test. No runtime Python.
