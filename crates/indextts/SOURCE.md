# indextts — third-party sources

Native Candle port of IndexTTS 2.5 inference. The Rust implementation in
this crate is original work (Apache-2.0, same as the crate license); the
*semantics* it reproduces and the model weights it consumes come from the
sources below. Model weights are never committed or redistributed by this
repository; converted safetensors produced by the development script stay
under the upstream weight licenses.

## Pinned source revisions

| Component | Source | Pinned revision | License |
| --- | --- | --- | --- |
| Inference semantics (GPT v2, codec, s2mel, BigVGAN wrapper, CAMPPlus wrapper, frontend, tokenizer) | https://github.com/index-tts/index-tts | commit `d9e41aac89fd00b3d71497fddb287b7f24613712` (2026-09-30, main) | code MIT-style per repo LICENSE; **weights: bilibili Model Use License Agreement (IndexTeam/IndexTTS-2.5)** — commercial use requires contacting Bilibili |
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
- QwenEmotion text-to-emotion LLM (optional upstream feature; callers can
  supply `emo_vector` directly).
- wetext / TensorFlow text-normalization stack — an approximate Rust
  subset lives in `src/front.rs` and is documented as not byte-compatible.
- NeMo text normalization (ja/es), JapaneseG2P: zh/en only.
- DeepSpeed / flash-attn / vLLM / TensorRT acceleration paths
  (`backends/trt`, `indextts/accel`): plain eager math only.
- Training code, `archive/`, streaming `quick_streaming_tokens` mode.
