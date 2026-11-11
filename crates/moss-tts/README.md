# talechime-moss-tts

The Cargo package is `talechime-moss-tts`; the Rust import is `talechime_moss_tts`.
Select GPU features for the target platform: CUDA on Windows/Linux, Metal on
macOS. Do not enable both through `--all-features`.

Candle implementation of Nano and Realtime synthesis with the shared audio codec.
VoiceDesign remains an auxiliary reference generator, not a product synthesis tier.
This crate reads local weights; it does not manage downloads, playback or configuration.
No Python is invoked by inference. Product adapters expose no style/instruction/language.

The current single-turn Realtime system/reference prompt and twelve-token
lookahead match six independent fixed official processor/prefill fixtures
using the real tokenizer. Inputs are tokenized as the same complete system
and reference string before audio placeholder replacement. Premature EOS
cannot accept unconsumed text. Nano segmentation preserves Latin words and
numeric expressions; an indivisible expression over budget fails explicitly.
These gates do not fix or accept the recorded Nano mixed-English omission,
year continuity or timbre failures, and do not replace real-model listening.

## Explicit adapter lifecycle gates

The ignored Realtime owner tests use `TALECHIME_MOSS17_LIFECYCLE_ROOT` and an
explicit `TALECHIME_MOSS17_LIFECYCLE_DEVICE=metal|cuda`. Compile the matching
`talechime-backends` feature and run `real_realtime_ -- --ignored --nocapture
--test-threads=1`. They do not download models. On 2026-11-11 both tests passed
on Metal: fixed-reference cache reuse, immediate/full-queue cancellation, actual
frame-limit failure without End, and bit-identical PCM after owner reuse.
The owner-state sample contained 51840 PCM samples. CUDA remains unexecuted in
this round; these tests do not accept Nano listening failures.

## Provenance

- TTS equations and scheduling: OpenMOSS/MOSS-TTS commit
  `934d6826b084c46a0d033402174d5f8ac4ed2519`, Apache-2.0.
- Audio codec equations: OpenMOSS/MOSS-Audio-Tokenizer commit
  `8c50ac4c5d7287d2ed6ea20a08c90ca439887d23`, Apache-2.0.
- `assets/*.json` records immutable official weight URLs, sizes and SHA-256.
  Official model cards identify the MOSS weights as Apache-2.0.
- The retired `moss_reference_fixture.py` (Git revision `e591c94`) created numerical fixtures from official
  Python modules for development only. Generated fixtures require no Python to test.

## Experimental probe

```text
cargo run -p talechime-moss-tts --release --features cuda --example realtime_probe -- MODEL_DIR CODEC_DIR OUTPUT.wav "你好，欢迎收听今天的故事。" cuda
```

The probe writes a WAV and RVQ token JSON without opening an audio device.
Model and codec directories must contain verified official safetensors and configs.
The codec preserves decoder state between PCM chunks and resets before an utterance.

## Decoder boundary audit — 2026-10-25

The shared Candle codec (Nano and Realtime) now attends to the previous
K/V plus the entire current chunk before trimming the returned cache to its
configured context. Trimming first removed history still needed by early queries;
chunks longer than the context could also leave queries without any valid keys.
Offsets remain absolute and the causal mask applies the context per query.
The encoder's offline behavior and adapter chunk sizes are unchanged.

This intentionally departs from upstream's pre-attention ring overwrite. The
checked-in `codec_wrap_output` fixture records that historical behavior and remains
unchanged for provenance; it is no longer the correctness oracle after wrapping.
The new untrimmed causal-window reference reproduced a maximum error of
`0.0021459907` before the fix. CPU fixture regressions require error below `1e-5`
for single-position, irregular, normal and oversized chunks. A synthetic quantizer
also checks identical fixed-code PCM across many wraps, short final chunks and
reset after cancellation with partially advanced layer state.

Real-model tests are not part of ordinary validation. For a separately scheduled
codec-only F32 replay, set `TALECHIME_MOSS_CODEC_DIR` to the verified Nano codec
directory or the shared Realtime codec directory. Optionally select a compiled
device with `TALECHIME_MOSS_CODEC_DEVICE=metal` or `cuda` (default `cpu`), then run:

```sh
cargo test --locked -p talechime-moss-tts real_codec_chunk_windows_match_full_decode -- --ignored --nocapture
```

Compile the appropriate platform feature when selecting a GPU. This ignored test
repeats the fixed Nano fixture codes to 137 frames (10.96 seconds, crossing all
pinned codec context windows) and compares full PCM with chunks of three and five,
including final short chunks. It prints maximum error and RMSE, using the F32 Nano reference maximum tolerance of `3e-4`. Set
`TALECHIME_MOSS_CODEC_DTYPE=f16` to check the production shared-codec precision
with explicit F16 tolerances (maximum `1e-2`, RMSE `1e-3`). It loads only the codec,
never downloads resources and does not generate TTS tokens.

Historical codec evidence and its acceptance limits remain in
[the boundary audit](../../docs/records/tts-core/streaming-boundary-audit-2026-10-25.md).
New perceptual quality, CUDA execution and GPU performance require separate validation.

Set `MOSS_MODE` to `realtime-1.7b` or the auxiliary `voice-design-1.7b`.
`MOSS_REFERENCE` accepts a mono 24kHz WAV; `MOSS_INSTRUCTION` supplies a VoiceDesign
description. VoiceGenerator uses BF16 on Metal and CUDA; Realtime uses F16 on
Metal and BF16 on CUDA. Codec weights use F16. `MOSS_DTYPE` overrides standalone
example precision for development comparisons. VoiceGenerator F16 produced invalid
logits on the real Metal checkpoint; F32 codec loading exceeded this 24 GiB Mac's practical memory budget. The BF16 Metal public matrix completed on 2026-11-12:
17 generated clips and two declared reference-design length rejections.
See the [validation record](../../docs/records/tts-core/official-backend-parity-2026-11-11.md).
These results do not establish CUDA execution or perceptual quality.
