# Changelog

## Unreleased

- Unify backend registration and model preparation around resolved model identities, with adapter-owned resources, devices and recovery. CLI and library use the same preparation service.
- Move audio output behind the facade playback feature and downloads into backend resources; remove audio-output and HTTP dependencies from core.
- Make segmentation requests explicit and duration learning execution-local. Remove the selected-voice state and the separate whole-text segmentation API.
- Remove legacy directory/data migration and configuration-driven session APIs. Current plans, JSON Lines v7 and current data formats remain; historical files are left untouched.

## 0.5.0 — 2026-10-25

- Publish the facade/core 0.5.0, backends 0.3.0, protocol 0.3.1, Qwen3-TTS 0.4.0 and Qwen3-ASR 0.1.1 libraries, plus the first branded 0.1.0 MOSS, VoxCPM, OmniVoice, CosyVoice3 and IndexTTS packages. Protocol wire version remains v7.
- Add experimental CosyVoice3 reference cloning and per-span styles; add IndexTTS 2.5 source-build support with named emotions, vectors and emotion references. Converted local resources are required for both backends; IndexTTS is not included in standard archives.
- Optimize MOSS, Qwen, VoxCPM2, OmniVoice and IndexTTS CPU, Metal and CUDA paths, with model-specific ownership, cancellation and output checks. Performance results remain tied to their recorded model, device and revision.
- Fix Qwen streaming chunk noise by retaining session-owned causal decoder history. Preserve the eager input layout during CUDA vocoder graph replay.
- Correct VoxCPM2 duration limits to count output patches, bounding each segment to the requested duration and the 30-second playback budget.
- Keep standard Linux/Windows archives on CPU and enable Metal only on macOS; CUDA and dynamic ORT remain explicit source-build options. Preserve JSON Lines v7 and the existing data paths.
- Gate application releases on CPU checks, cross-platform performance contracts, and model-free archive/installer smoke tests. Use fresh CI evidence directories and check Windows commit budgets before GPU inspection.

## 0.4.1 — 2026-10-14

- Automatically migrate default legacy settings, checkpoints and resources without overwriting current data or removing originals. Custom paths and library path resolution remain explicit.
- Keep protocol handshakes lightweight while copying large model resources during preparation; validate old voice bundles before making them visible.
- Preserve completed migration stages and wait for cancelled copy work before preparing another model.

## 0.4.0 — 2026-10-14

- Add whole-chapter audio caching for sealed plans with pinned seeds, preset voices and stable backend resource identities. Cache hits replay without synthesis; cancelled, failed and abandoned candidates are cleaned safely.
- Add scene/paragraph continuation boundaries and bounded chain re-anchoring, with smoother output loudness across segments.
- Retry MOSS Nano first-step empty EOS with a fixed attempt limit, without repeating delivered audio.
- Keep cached playback progress tied to actual consumption and preserve cancellation during cache I/O.
- Publish JSON Lines v7 additive continuation fields in talechime-protocol 0.3.0, core 0.4.0 and backends 0.2.0.

Ordinary tests use deterministic fixtures and do not download models. Real-model listening and cross-platform audio quality remain separate acceptance work.
