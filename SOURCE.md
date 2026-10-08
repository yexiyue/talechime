# Extraction provenance

Talechime was extracted from https://github.com/yexiyue/TRNovel at commit
`cd8d71051840d063cdfb6f57bbaae4f387c6a3fa` on 2026-10-08.

This repository starts with an extraction commit rather than rewriting the
original Git history. Earlier history remains in TRNovel. Existing copyright,
model manifests, fixtures and component-specific license/source records are retained.

The old `novel-tts` executable remains a compatibility entry point. JSON Lines
protocol v5, `~/.novel/tts_config.json`, `~/.novel/tts/checkpoints/` and
`~/.novel-tts/` model/voice paths retain their existing meaning.

Historical `dev-notes/` describe experiments at the time they were recorded.
Use current source and the new README for current capabilities.

## Upstream fixes

- 2026-10-08: TRNovel [c5d7daa](https://github.com/yexiyue/TRNovel/commit/c5d7daa60b435293d77f1ab886e22fdd56a214c0) — synchronize MOSS Metal model and codec loading before ready; retain upstream macOS verification records with renamed crate paths.
