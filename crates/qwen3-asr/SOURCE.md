# Qwen3-ASR native implementation

Encoder, decoder, configuration and mel frontend originate from
https://github.com/alan890104/qwen3-asr-rs at
`c5ef09646af6278d2ba8b8ceaf543ffb32d1a5dc` (MIT; LICENSE preserved).

Talechime uses Candle 0.11.0, official Transformers-format 0.6B weights at
`Qwen/Qwen3-ASR-0.6B-hf`, revision `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`.
Local changes remove hub/streaming/aligner/device fallback surfaces, remove the
unsafe Send assertion (models stay on their inference owner), map official
configuration/weight names, use periodic Hann, and require EOS and cooperative
cancellation. Models and their Apache-2.0 license are separate from source.
