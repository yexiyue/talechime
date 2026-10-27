## Why

Restore Kokoro as a small native synthesis option with correctness established against the official Chinese PyTorch model, rather than the retired Rust wrapper. The official reference, source revisions and initial fixtures are recorded in `docs/records/kokoro/kokoro-official-feasibility-2026-10-27.md`.

## What Changes

- Add an independent Candle Kokoro model library and reproducible official-weight/fixture export tooling.
- Port the official Mandarin 1.1 frontend and preserve complete text when chunking.
- Register an optional Kokoro backend with built-in voices, CPU and explicit GPU features, bounded output and cancellation.
- Record independent official numeric comparisons and real audio/performance validation; ordinary tests never download models.

## Capabilities

### New Capabilities
- `kokoro-synthesis`: Native official-model inference, Mandarin frontend, built-in voices and model lifecycle.

### Modified Capabilities

None.

## Impact

New `crates/kokoro`, optional backend/facade features and registration, documentation and source notices. No default backend switch, wire-version change, historical-directory migration, training or revival of ZipVoice.
