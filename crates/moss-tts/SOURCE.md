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
