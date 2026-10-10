# Upstream source

Model equations, message formats and generation schedules are ported from:

- https://github.com/OpenMOSS/MOSS-TTS at revision
  `934d6826b084c46a0d033402174d5f8ac4ed2519`.
- https://github.com/OpenMOSS/MOSS-Audio-Tokenizer at revision
  `8c50ac4c5d7287d2ed6ea20a08c90ca439887d23`.

Both upstream codebases use Apache-2.0; LICENSE and NOTICE are retained.
This crate contains native Rust inference, not upstream CLI/server/training
programs. Model resources are pinned in assets and downloaded separately.
Local adaptations originated in TRNovel; see the root SOURCE.md.

Nano GPT2 and stereo codec support were carried from TRNovel experiment
`8be7d953fd91433569767404e5a7a37534956d5b`. Equations and prompt formats
follow OpenMOSS/MOSS-TTS-Nano checkpoint `44502f80dbf9743528fa921cc544d662c685ebec`;
attention conventions were cross-checked against Apache-2.0 community source
https://github.com/ramishi/moss-tts-nano-rust-candle at
`f4d3fcf4de9b4118ee664087f88495f4174836e1`. No gated weights are included.

Realtime previous-turn rows are tested against the pinned upstream
`make_user_prompt` formatter using synthetic text/audio token fixtures.
`tools/tts/realtime_prompt_reference.py` regenerates these fixtures without weights.
