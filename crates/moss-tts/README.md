# talechime-moss-tts

The Cargo package is `talechime-moss-tts`; the Rust import is `talechime_moss_tts`.
Select GPU features for the target platform: CUDA on Windows/Linux, Metal on
macOS. Do not enable both through `--all-features`.

> Legacy `tools/tts` scripts mentioned here were removed on 2026-10-20. Retrieve historical scripts from Git revision `e591c94`; their paths below describe provenance or historical reproduction, not current executable entry points.

Experimental Rust implementation of the official MOSS Local, Realtime and Delay
model schedulers and MOSS Audio Tokenizer. This crate uses the workspace's pinned
Candle version and platform features; it does not manage downloads, playback,
configuration or a server. No Python is invoked by inference.

The worker exposes Local and Realtime as optional GPU trial models after real
generation/cancellation checks. VoiceGenerator creates reusable references. The
existing Nano ONNX adapter remains intact. Successful termination does not prove
spoken coverage. See [验收记录](../../docs/records/moss/moss-candle-acceptance.md) for remaining acceptance.

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
cargo run -p talechime-moss-tts --release --features cuda --example local_probe -- MODEL_DIR CODEC_DIR OUTPUT.wav "你好，欢迎收听今天的故事。" cuda
```

The probe writes a WAV and RVQ token JSON without opening an audio device.
Model and codec directories must contain verified official safetensors and configs.
The codec preserves decoder state between PCM chunks and resets before an utterance.

## Decoder boundary audit — 2026-10-25

The shared Candle codec (Nano, Local and Realtime) now attends to the previous
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

Nano ONNX uses a different exported implementation. Static inspection of the
installed decode-step graph, whose SHA-256 matches the pinned manifest
(`9527c86a29e1837edec1f74db57d5eeaadb3a715af3382703566460afed25855`),
verified all 12 layers: all 36 K/V/position paths concatenate the raw previous
cache (only casts intervene) and current chunk for attention/masking, with separate
suffix slices for the returned caches. It does not overwrite the ring before
attention. All 52 state input/output mappings in adapter metadata exist in that
graph. Production creates fresh state per generation, carries every mapped output
into the next call and warms continuation prefixes through that same state, using
three-frame chunks and a short final chunk. No analogous adapter defect was
confirmed; no graph, weights, manifest or ONNX adapter change was made. This is a
static audit; separately scheduled CPU ONNX numerical replay is recorded below.

Real-model tests are not part of ordinary validation. For a separately scheduled
codec-only F32 replay, set `TALECHIME_MOSS_CODEC_DIR` to the verified Nano codec
directory or the shared Local/Realtime codec directory. Optionally select a compiled
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

The coordinated M4 Pro Metal runs passed: Nano F32 maximum error below `9.4e-7`;
Local/Realtime shared codec F32 below `6.6e-6`. Shared-codec F16 RMSE was
`2.02e-4`–`2.15e-4`, maximum `0.00878`, so that precision is not waveform-equivalent
across batching. ONNX CPU replay using the installed pinned full/step graphs
and Python ONNX Runtime 1.24.4 also passed the same 137-frame input with chunks
of three/five (maximum below `7.9e-7`, all 52 state mappings). This is graph-level
CPU evidence, not acceptance of every Rust ORT provider/version. Raw logs are in
`runs/streaming-audit-2026-10-25/`. New perceptual quality, CUDA execution and GPU
performance remain unverified.

Set `MOSS_MODE` to `local-1.7b`, `realtime-1.7b` or `voice-design-1.7b`.
`MOSS_REFERENCE` accepts a mono 24kHz WAV; `MOSS_INSTRUCTION` supplies a design
description. CUDA model weights use BF16, codec weights F16; `MOSS_DTYPE` overrides
model precision for development comparisons. VoiceGenerator F16 overflowed on the
real checkpoint and must not be the CUDA default.

Development reference scripts require the pinned upstream checkouts in
`target/tts-integration/moss-upstream` and `moss-codec-upstream`, PyTorch and
Transformers 5.0.0. Their SDPA implementation supplies the causal behavior required
by the upstream position-free Local attention. Python is not a runtime dependency.
