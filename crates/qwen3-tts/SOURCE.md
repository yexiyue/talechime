# Upstream source

Imported from https://github.com/TrevorS/qwen3-tts-rs at fixed revision
`711ceee07cad92673f86de8997bdf54c30caa49f`.

The upstream Rust implementation is MIT licensed; LICENSE retains its notice.
Talechime inherits the local device, cancellation and streaming adaptations
from TRNovel, whose extraction commit is recorded in the root SOURCE.md.
Model weights are downloaded separately and retain their own licenses.

## Official input-condition comparison (2026-11-11, unreleased)

The independent Python oracle uses QwenLM/Qwen3-TTS revision
`022e286b98fbec7e1e916cb940cdf532cd9f488e`, with model source SHA-256
`25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0`.
The offline prefill fixture preserves the original 11 cases and adds all 19
public corpus texts in each of the 11 CustomVoice, Design and Base ICL
conditions: 220 cases total. Each added case records its source text and hash.
The corpus cases use synthetic UTF-8 byte token IDs and synthetic embeddings;
they compare input conditions, not checkpoint tokenization, weights or audio.
`tools/tts/qwen_prefill_oracle.py` also supports a separate local Base-weight
comparison using the original fixed-token cases; it is not a 19-text real-weight
gate. Ordinary fixture tests need no model or download.
