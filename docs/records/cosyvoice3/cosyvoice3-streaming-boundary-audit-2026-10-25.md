# Streaming boundary and history audit — 2026-10-25

No analogous truncated-history defect was confirmed in CosyVoice3. Runtime
behavior is unchanged. This audit covers `crates/cosyvoice3` and the
CosyVoice3 backend only; it makes no claim about other models.

## Source comparison

The local official checkout `runs/reference/CosyVoice` resolves to
`074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc`. The relevant APIs are
`cosyvoice/cli/model.py` (`CosyVoice3Model.token2wav` and inherited
`CosyVoice2Model.tts`), `flow/flow.py` (`CausalMaskedDiffWithDiT.inference`),
`transformer/upsample_encoder.py` (`PreLookaheadLayer`),
`hifigan/f0_predictor.py` and `hifigan/generator.py` (`CausalHiFTGenerator`).

| Suspected failure | Current evidence |
| --- | --- |
| Dropped flow history | `streaming.rs::emit` sends `tokens[..plan.take]` plus the full prompt on every solve. It slices off previously committed mel only after the solve. No sliding token suffix approximates attention or convolution history. |
| Lost or duplicated lookahead | The schedule includes three future tokens but advances only by the hop. Flow splits those trailing tokens into context before upsampling. The final solve consumes all remaining tokens. |
| Vocoder history reset at a seam | Previously committed mel is concatenated with only the new mel. HiFT recomputes F0, source, convolutions and iSTFT on that entire accumulated mel; only PCM beyond `speech_offset` is delivered. There is no CV2 source cache or crossfade to port into CV3. |
| Wrong non-final trim or missing tail | Both implementations withhold F0's three right-context frames, conv_pre's four right-context frames and one final mel frame of PCM (480 samples at the default rates). Finalize removes those withholds and slices after the prior sample offset. |
| Final mel differs from an earlier prefix | Official final `token2wav` defaults `stream=False`; earlier chunks pass `stream=True`. Rust uses `!plan.finalize`. Final non-chunked attention can revise the re-solved prefix, but only the new mel is appended, just as official code does. Streaming/offline equality is not a correctness requirement. |
| Partial state committed after decode failure | Mel, last-new-mel, PCM offset and schedule commit only after synthesis and CPU copy succeed. Errors fail the session permanently. Existing terminal-state tests cover rejected continuation. |

The 25→50→100 hop schedule and prompt padding are session-owned in Rust.
Official code mutates the model's hop across requests; Rust deliberately
matches a fresh official model for every session. This is an existing,
documented isolation choice, not a newly fixed defect.

## Deterministic coverage added

- Fixed token arrivals in batches of 1, 17 or the whole input yield the
  same prefixes and offsets, including hop/lookahead thresholds, final
  short tails and every prompt residue modulo 25.
- A nonzero one-channel pre-lookahead network matches independent scalar
  right/left convolution, leaky ReLU, residual and mel-repeat calculations
  at multiple prefixes and final zero padding.
- A nonzero one-channel F0 network matches independent scalar calculations
  for its right-looking first convolution, four left-history convolutions,
  ELU and classifier. Non-final output withholds exactly three frames;
  the full final output includes them.
- The existing official PyTorch harmonic-source oracle now also checks
  every shorter whole-frame prefix against the corresponding full source
  samples, guarding the sample-zero phase anchor used during replay.

These tests need neither model weights nor downloads. Scalar oracles are
independent arithmetic references, not new official PyTorch executions.

## Cancellation and reset ownership

`TokenStreamDecoder` owns tokens, hop, mel, diagnostic mel and sample offset.
`StreamingTts` owns sampling progress and silent-token state and exclusively
borrows model components for backend requests. Its Drop clears LLM KV;
new LLM prefill clears KV before starting another run. Flow and HiFT retain
weights and fixed noise buffers, not incremental request history.

The backend's `deliver` reserves bounded channel capacity before calling
`next_pcm`. Dropping the receiver stops subsequent inference at the next
block boundary; it does not interrupt an executing LLM/flow/HiFT call.
Returning from delivery drops the session on the model owner thread. Only
a final non-budget-limited chunk permits End; End is synthesis completion,
not playback completion. Existing backend tests exercise backpressure,
receiver drop, missing final blocks, errors and token-budget exhaustion.

## Limits and remaining gaps

No models were downloaded. The initial source/scalar audit ran no GPU models;
the separately scheduled real-model checks below subsequently ran on M4 Pro Metal.
This audit does not establish perceptual seam quality, full-checkpoint
cross-framework numeric parity, CUDA/Metal behavior or performance.
The ordinary tests do not load a full multi-chunk Flow→HiFT oracle with
shared fixed tokens, prompt, flow noise and source buffers, including the
first chunk, growing hops and final flush. The separately scheduled same-mel test below validates emitted-prefix slicing
against the current full vocoder. It does not establish official full-vocoder
waveform parity. Session Drop/reset ownership was inspected;
the new numeric tests do not instantiate a full backend model session.

Prior real-model records under `docs/records/cosyvoice3` remain historical
evidence with their own stated numeric and audio limits.

## Validation

- `cargo test --locked -p talechime-cosyvoice3 --lib --tests --examples`:
  40 library tests and 5 example tests passed; 2 GPU tests ignored.
- `cargo test --locked -p talechime-backends --no-default-features --features cosyvoice3 --lib cosyvoice3`:
  7 passed.
- Crate-scoped all-target Clippy with `-D warnings`, rustdoc including
  examples/private items with `-D warnings`, changed-file rustfmt checks
  and scoped `git diff --check` passed.

## Separately scheduled real-model checks

The primary ran `stream_probe` once with pinned converted weights and an isolated
copy of `runs/cv3-case-long` (fixed official tokens, source buffers and flow noise).
All three chunks had the expected PCM/mel lengths and final flags. Mel RMSE was
`1.73e-4`–`2.31e-4`; chunk PCM RMSE was `0.00245`, `0.00814`, and `0.06371`,
within the probe's existing gates. This is not strict cross-framework waveform
parity. The fixture itself records official streaming/nonstreaming RMSE `0.1316`;
those two modes cannot serve as a direct seam-equivalence oracle.

To isolate the vocoder from CFM mel variation, the ignored
`real_hift_same_mel_prefixes_preserve_emitted_samples` test loaded the same local
HiFT weights and the fixed 368-frame official mel/phase/noise fixture. On Metal
F32, prefixes of 25/50/75/100/367 mel frames and the final flush all produced
emitted-prefix PCM **identical** to full-mel decoding (max error and RMSE zero).
The final output covered exactly 176640 samples, including the short final tail.
This establishes prefix stability for that same-mel fixture, not general perceptual
or full CFM-to-vocoder cross-framework equivalence.

Run explicitly with `CV3_HIFT_TEST_MODEL`, `CV3_HIFT_TEST_FIXTURE` (the
`hift.safetensors` fixture file), and `CV3_HIFT_TEST_DEVICE=metal`:

```sh
cargo test --release --locked -p talechime-cosyvoice3 --features metal real_hift_same_mel_prefixes -- --ignored --nocapture
```

Raw logs and copied fixtures are ignored under
`runs/streaming-audit-2026-10-25/`. CUDA was not executed.
