# Changelog

## 0.4.0 — 2026-10-14

- Add whole-chapter audio caching for sealed plans with pinned seeds, preset voices and stable backend resource identities. Cache hits replay without synthesis; cancelled, failed and abandoned candidates are cleaned safely.
- Add scene/paragraph continuation boundaries and bounded chain re-anchoring, with smoother output loudness across segments.
- Retry MOSS Nano first-step empty EOS with a fixed attempt limit, without repeating delivered audio.
- Keep cached playback progress tied to actual consumption and preserve cancellation during cache I/O.
- Publish JSON Lines v7 additive continuation fields in talechime-protocol 0.3.0, core 0.4.0 and backends 0.2.0.

Ordinary tests use deterministic fixtures and do not download models. Real-model listening and cross-platform audio quality remain separate acceptance work.
