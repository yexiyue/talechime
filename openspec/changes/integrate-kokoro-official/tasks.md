## 1. Official reference and model
- [x] 1.1 Add reproducible pinned official export tooling, source notices and independent fixtures.
- [x] 1.2 Implement native ALBERT, bidirectional LSTM, duration and prosody/text encoders with fixed-input official comparisons.
- [x] 1.3 Implement AdaIN decoder, excitation and TorchSTFT-compatible vocoder; validate complete F32 output against official random inputs.
- [x] 1.4 Cover invalid inputs, duration/output budgets and numerical edge cases without model downloads.

## 2. Mandarin and integration
- [x] 2.1 Port pinned Mandarin 1.1 frontend data/rules and independently compare normalization, segmentation, pronunciation, sandhi and erhua.
- [x] 2.2 Preserve complete long-input source ranges and handle unsupported symbols/languages explicitly.
- [x] 2.3 Add optional backend/facade features, registry resources, built-in voices and accurate parameter/device capabilities.
- [x] 2.4 Validate cancellation, loading ownership, bounded output and reuse through public synthesis/protocol APIs.

## 3. Acceptance
- [x] 3.1 Expand real official CPU comparisons across voices, speeds and lengths; record audio, numeric and performance results.
- [x] 3.2 Implement and verify available Metal path; separately record CUDA build/hardware verification boundaries.
- [x] 3.3 Update current docs and complete formatting, relevant tests, Clippy and strict rustdoc.

## 4. Separate pronunciation improvement
- [x] 4.1 Establish an independent official contextual-disambiguation baseline and a pronunciation corpus, keeping the faithful Kokoro/Misaki baseline intact.
- [x] 4.2 Implement an explicit optional contextual pronunciation path and verify improvements and regressions separately from acoustic equivalence.
