## Context

Official Kokoro `dfb907a02bba8152ca444717ca5d78747ccb4bec`, Misaki `fba1236595f2d2bf21d414ba6e57d25256afada3`, Chinese model `01e7505bd6a7a2ac4975463114c3a7650a9f7218` are the sole model/frontend references. Local development artifacts already contain F32 folded weights, intermediate tensors and random draws. Existing unrelated dirty files must be preserved.

## Decisions

1. Implement model arithmetic in a separate `talechime-kokoro` crate using the unified Candle dependency. Start with batch-one F32 fixed IDs/style/noise. ALBERT shares its single attention/FFN block across 12 applications; LSTM is bidirectional; duration uses ties-to-even rounding. Preserve default TorchSTFT conventions.
2. Export official weights with development-only Python, fold weight normalization and retain revisions, hashes, actual imported-source provenance and version metadata. Runtime never invokes Python. Model files remain outside Git.
3. Make excitation randomness injectable for independent tests and seeded for production. Never substitute zero noise or assume equal cross-framework seeds.
4. Port Mandarin normalization, segmentation/POS, dictionary pronunciation, tone sandhi, erhua and v1.1 token mapping separately from acoustics. Preserve official behavior before experimenting with disambiguation. Unsupported input must not silently disappear; chunk by valid phoneme budget while retaining original UTF-8 ranges.
5. Reuse the registry preparation contract and dedicated inference-thread ownership. Declare segment output, no reference cloning and only implemented parameters. Preserve LocalSet/session/cancellation and current resources layout. Do not migrate or inspect historical data.

## Validation

Independent Python fixtures establish stage errors, exact integer durations/output length and PCM SNR. Expand across voices, text lengths, speeds and edge cases. Ordinary tests cover math, validation/frontend and lifecycle without models. Run formatting, relevant tests, Clippy and strict rustdoc. Real CPU and available Metal validation are separate from CUDA compilation/hardware claims. Performance measurements include warm/cold boundaries, threads, actual audio duration and source/weight identity. No unverified device is reported as accepted.

## Risks

Small pre-rounding duration errors can change shapes; harmonic phase accumulation amplifies numerical errors; grouped transpose convolution and normalization must retain PyTorch semantics. Official frontend still misreads names and has unknown-character/truncation paths; model-equivalent acoustics do not certify correct pronunciation. Any necessary correction is recorded independently from the official-equivalence fixture.
