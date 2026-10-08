# Talechime repository guidance

- Read README.md, docs/architecture.md and docs/development.md before changing behavior. Use current source over historical dev-notes.
- Keep Talechime independent of TRNovel and CastGlean business types. Hosts own chapter acquisition, role analysis and voice bindings.
- Preserve protocol v5, novel-tts compatibility entry point, user configuration, model/voice paths and checkpoints unless migration is explicitly scoped.
- Use UTF-8 byte ranges bound to immutable source hashes. Generation completion must never substitute for actual playback completion.
- Keep sessions on Tokio LocalSet; inference and audio objects stay with their owners. Cancellation and bounded stream ownership are part of the public behavior.
- Keep Candle versions unified, ORT pinned and Windows CRT dynamic. Read relevant dev-notes before changing native dependencies.
- Ordinary tests must not download models. Record real-model numeric, audio and performance verification separately.
- Crates use named module files, explicit feature gates and workspace dependencies. Avoid unrelated inference refactors.
- Use Mermaid for diagrams. Distinguish current capabilities, experiments and roadmap in docs.
- Preserve third-party licenses, pinned source revisions and model manifests. Never commit models, credentials, user audio or novels.
- Run formatting, relevant tests, Clippy and rustdoc. CI separates CPU checks from native CUDA/Metal compilation.
