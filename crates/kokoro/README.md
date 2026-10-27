# talechime-kokoro

Native F32 Kokoro v1.1-zh inference, directly compared with pinned official
hexgrad Kokoro and Misaki Python sources. No community Rust implementation is
used as a reference. Runtime inference does not invoke Python or download models.

See [SOURCE.md](SOURCE.md) for revisions and licenses, and
[the backend guide](../../docs/kokoro.md) for explicit export, preparation,
features and independent comparison commands. Optional `g2pw` enables the
official author's CPU ONNX contextual polyphone predictor.

CPU owns text/prosody encoding and spectral operations. `metal` / `cuda` select
decoder offload; these flags do not imply platform acceptance. Real-model
fixtures, weights and generated audio are deliberately absent from this crate.
