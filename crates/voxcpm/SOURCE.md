# VoxCPM2 native Candle port

> Legacy `tools/tts` scripts mentioned here were removed on 2026-10-20. Retrieve historical scripts from Git revision `e591c94`; their paths below describe provenance or historical reproduction, not current executable entry points.

Model semantics: [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM),
revision `f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629`, Apache-2.0.
GGUF tensor mapping reference: [llama.cpp-omni](https://github.com/tc-mb/llama.cpp-omni),
revision `873056743b74e1a4ce5dcf7290e2298428e214db`, MIT.
This crate independently implements those model operations using workspace Candle
0.11.0 (crates.io source revision `31f35b147389700ed2a178ee66a91c3cc25cc80d`).
It does not link or execute the mapping reference's inference engine.

Candle is consumed through the workspace `[patch.crates-io]` entry, which points
at the Talechime fork [yexiyue/candle](https://github.com/yexiyue/candle), branch
`talechime-perf`. The fork is based on the same upstream revision
`31f35b147389700ed2a178ee66a91c3cc25cc80d` (upstream tag `0.11.0`) and keeps the
upstream dual Apache-2.0 / MIT license; see `LICENSE.candle-APACHE`,
`LICENSE.candle-MIT` and `docs/legal/candle/SOURCE.md`. Patches in that fork must
be behaviour-only and upstreamable, because published crates.io builds of this
library resolve the registry release instead of the fork.

The `cuda` feature additionally depends on `cudarc` (optional, CUDA builds only)
pinned to the exact version candle-core 0.11.0 resolves, so the CUDA graph
capture API used by `cuda_graph.rs` operates on the same `CudaContext` instance
as candle. No Metal or CPU build pulls it in.

Weights remain the existing DennisHuang648/VoxCPM2-GGUF resource revision
`169f64d8b98bbaab1761e4ca3a83e6af653456cc`; see the production resource manifest
in `novel-tts-backends` for sizes and SHA-256. The production adapter uses this
Candle library after explicit user authorization to integrate before complete
qualification. Strict numerical failures and outstanding acceptance remain
reported; the old C++ development benchmark has been retired. Its fixed source
and licenses remain in `docs/legal/retired-voxcpm-native/` with recovery instructions.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| VoxCPM2-BaseLM-Q8_0.gguf | 1727309920 | 0113177abd11303503bf0b705e1613ec5f0a8508cc74a7dfd0f99312b962a962 |
| VoxCPM2-Acoustic-F16.gguf | 1825096352 | 5bde898488ad635ff55d24da53543768fa33d5e5cdc538ce190e5ef831038e85 |

The weights are derived from Apache-2.0 OpenBMB/VoxCPM2. Their existing default
directory is `~/.novel-tts/voxcpm/models/2b-q8_0/169f64d8b98bbaab1761e4ca3a83e6af653456cc/`.
The caller must verify the resource manifest before opening model files.

Original comparison weights: [openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2/tree/32279effe8c19989596f05d353d1447f51d9e915),
revision `32279effe8c19989596f05d353d1447f51d9e915`, Apache-2.0.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| model.safetensors | 4580080592 | f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d |
| audiovae.pth | 376951122 | 94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1 |
| tokenizer.json | 3676772 | f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20 |
| config.json | 4336 | 405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c |

`tools/tts/download_voxcpm_official.py` resumes into the revision-scoped
`~/.novel-tts/voxcpm/models/2b-bf16/` directory and verifies LFS SHA-256 or
Git blob identities, recording SHA-256 for every downloaded file. It is a
development tool; no Python download or inference runs in the product.
Original BF16 and F16 comparison runs reuse these same assets. F16 is a
compute precision choice, not a separate upstream model or dequantized Q8.
# Generation-bound comparison (2026-11-11, unreleased)

The fixed official `_generate` caps each request at
`min(int(target_token_count * 6 + 10), max_len)`. Target token count excludes
reference transcripts and audio markers. Native generation now applies this
limit within the caller's explicit playback-duration ceiling. The default
product ceiling remains 30 seconds; it is distinct from upstream's default
2000 frames. Stop is allowed only after index 2 and the final decoded patch
is delivered before EOS. Twelve bound and twelve stop cases come from the
fixed source expressions, with source SHA-256 and a reproducible exporter in
`tools/tts/voxcpm_bounds_oracle.py`.

The product intentionally exposes fixed single-reference or unconditioned
generation, as documented in `docs/reference-synthesis.md`; upstream's
combined isolated-reference plus continuation input is outside that public
contract. This comparison does not claim that mode or close Q8/BF16 numeric,
real-model cancellation or listening gates.

## Single-reference conditioning comparison (2026-11-11, unreleased)

`tools/tts/voxcpm_conditioning_oracle.py` extracts the fixed official
`_generate_with_prompt_cache` and `_make_ref_prefix` methods after checking the
source SHA-256 `31ad554bc15a4da18cf820c6fda97a5b2c56ac8e9a5e66031a264d363661575c`.
The offline `conditioning.json` fixtures compare the native model's actual
conditioning assembly and audio quantization ranges for natural, styled,
design, transcript-clone, styled transcript-clone and isolated single-reference
inputs. A transcript clone places the reference transcript before the target
(including any `(description)` style prefix), then the audio-start marker and
reference audio patches. An isolated reference has reference start/end markers
and audio before the target. No combined isolated-reference plus continuation
mode was added.

The fixture preserves its original six cases and adds each of the 19 public
corpus texts in all six conditions (120 cases total), with source text and
SHA-256 on each added case. Ordinary tests run offline without model weights.

These structural fixtures use synthetic token IDs and audio features; they do
not validate a tokenizer against official text, model outputs, generated audio,
Q8/BF16 numerical parity, or listening quality. Real-weight and adapter-level
acceptance are separate gates.
