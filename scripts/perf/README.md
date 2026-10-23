# OmniVoice CPU attribution

These tools use installed local weights and the pinned official Python environment.
They do not download models or change production inference. Tensor archives and WAV
files belong under ignored `target/perf/`, never in Git.

## Build and freeze

Use the current worktree and a fresh directory. Set `MODEL` to the installed
OmniVoice model directory, `OFFICIAL` to the pinned official checkout, and `RUN` to
an absolute output directory. These names do not override system environment variables.

```sh
cargo build --release --locked -p talechime-backends --no-default-features \
  --features omnivoice,performance-tools --example performance_probe
cargo test --release --locked -p talechime-omnivoice --lib --no-run \
  --message-format=json > "$RUN/build-tests.jsonl"
cp target/release/examples/performance_probe "$RUN/probe-baseline"
```

Copy the `executable` from the `talechime_omnivoice` test artifact in
`build-tests.jsonl` to `$RUN/cpu-tests`. Finish compilation before inference.
Record revisions, source/weight/input/binary hashes, `otool -L`, `nm -u`, and
thread settings before running. All commands below run serially.

## Capture and replay

The capture fixture is a single unchunked short Chinese request, 32 steps, default
sampling and narrator. It derives the request seed using the current host rule
`SeedPolicy::Pinned(42).resolve(0, 0)` and reports that value. Before accepting the
fixture, compare its complete code record with the matching host baseline: changes
to the host seed policy or text preparation must not silently change the fixture.
The library waveform precedes the host's boundary-silence and leveling processing;
compare final WAVs through the same host harness separately.

```sh
export OMNI_CPU_TRACE_ROOT="$RUN"
export OMNI_CPU_TRACE_MODEL="$MODEL"
export OMNI_CPU_TRACE_TEXT="/absolute/path/to/corpus-short.txt"
export PYTHONPATH="$OFFICIAL"
python3 scripts/perf/omnivoice_cpu_run.py "$RUN/capture" -- \
  "$RUN/cpu-tests" --ignored --exact \
  stage0_qwen3::cpu_attribution::capture_real_cpu_inputs --nocapture --test-threads=1
OMNI_CPU_TRACE_EXPORT=1 python3 scripts/perf/omnivoice_cpu_run.py "$RUN/export-stage0" -- \
  "$RUN/cpu-tests" --ignored --exact \
  stage0_qwen3::cpu_attribution::replay_real_cpu_inputs --nocapture --test-threads=1
```

Capture stores layer inputs/outputs and masks at steps 0/15/31, layers 0/13/27,
plus post-final-norm head inputs. Export replays the actual private layer code,
checks complete bits, and stores Q/K/V, projection outputs and MLP inputs.
Safetensors store canonical values; adjacent layout JSON records original strides
and storage offsets. Python restores those layouts before timing.

Run three independent processes per framework, interleaving `RP`, `PR`, `RP`.
Reverse the operation order in the second pair using `OMNI_CPU_TRACE_REVERSE=1`
for Rust and `--reverse` for Python. Omit `OMNI_CPU_TRACE_EXPORT` during measurement.
Each operation has one warmup, then five groups of five calls. Example Rust and
Python commands, each with a distinct output prefix:

```sh
python3 scripts/perf/omnivoice_cpu_run.py "$RUN/stage0-1-R" -- \
  "$RUN/cpu-tests" --ignored --exact \
  stage0_qwen3::cpu_attribution::replay_real_cpu_inputs --nocapture --test-threads=1
python3 scripts/perf/omnivoice_cpu_run.py "$RUN/stage0-1-P" -- \
  "$OFFICIAL/.venv/bin/python" scripts/perf/omnivoice_cpu_replay.py stage0 \
  --model "$MODEL" --root "$RUN"
```

## Decoder and product baseline

Place JSON fixtures in `$RUN/codes/`, each with `codebooks`, `frames`, and row-major
`values`. Use the baseline's real codes: take the first 1/7/130 frames separately
for every codebook, plus all long generation segments. The host code dump is a
sequence of little-endian `u32 count` + `count * i64`; exclude the separately
identified reference warmup record, not an assumed arbitrary first segment.
The 130-frame fixture must be named `frames-130.json` to enable operator replay.

Run `stage1_model::cpu_attribution::replay_real_cpu_decoder` using the frozen Rust
test binary and Python `omnivoice_cpu_replay.py stage1 --model ... --root ...`.
First run Rust with `OMNI_CPU_TRACE_EXPORT=1` to export real decoder inputs;
measurement runs omit it. Full decoder measurements run before primitive probes.
Use `RPPR`, then one additional local process per framework: three local processes
per framework, while whole-decoder comparison uses the first four slots only.
Each case has one first decode followed by five hot calls. Weight loading is
reported separately; first decode means first occurrence of that shape in the
process, not an independently cold model for every case.
Rust whole decode includes I64 code tensor construction inside `decode_tokens`;
Python constructs that small input tensor before calling `decode`. Disclose this
boundary difference rather than attributing the entire elapsed gap to convolutions.

For the short product baseline run native `performance_probe REQUEST OUTPUT` and
`omnivoice_cpu_official.py` in `RPPR`, one warmup and three hot generations per
process. The official wrapper accepts `--model`, `--text-file`, `--output`, `--wav`,
`--warmup 1`, `--rounds 3`; set `OMNIVOICE_OFFICIAL_REVISION` explicitly. Report both
output duration and RTF. Different durations/seeds/frontend processing make this a
product comparison, not an operator comparison. Check short and long complete
codes and all segment/full F32/PCM16 WAVs using the native host harness.

## Evidence boundaries

`omnivoice_cpu_run.py` retains stdout, stderr, requested thread settings, process
peak RSS samples and compiler observations. It requests eight Rayon/VecLib/OMP
threads; this does not prove internal Accelerate/AMX thread count or dispatch.
No detected compiler is weaker than an idle-system guarantee. Keep every slot;
if a formal run is contaminated, allow at most one complete supplemental group.

Whole layers, attention consumers, MLP consumers, decoder blocks, residual units
and their child operations overlap: never sum them into a production estimate.
Rust decoder `layout_materialize` is a synthetic NCL/NLC roundtrip control; it is
not evidence of that exact copy sequence in production. Python attention includes
the official repeat-KV and sequence-first output materialization. Use the complete
attention consumer as the primary matched boundary.

`omnivoice_cpu_summarize.py RUN OUTPUT_JSON` collects receipts, per-process raw
measurement groups, tensor/layout hashes, code fixture hashes, `manifest.json`,
`e2e-summary.json` and `parity-summary.json`. The JSON can be committed as numeric
evidence; the tensor payloads and audio cannot. Keep rejected fixture screens in a
separate subdirectory so the summarizer cannot mix them into accepted results.

## CPU decoder pointwise validation

The retained CPU path is limited to Accelerate F32, dense contiguous square
weights with 32/64/128/256/512 channels, contiguous NCL input, kernel/stride/
dilation/groups all one, zero padding, and time length at least eight and
multiple of eight. Non-aligned lengths retain the original path: unrestricted
NCL GEMM changed bits in a 13-sample test. Encoder calls use their original entry.

With `OMNI_CPU_TRACE_MODEL` and `OMNI_CPU_TRACE_ROOT` pointing to the immutable
model and prior real-input capture, the frozen CPU test executable provides:

- `codec_layers::cpu_pointwise::compare_real_pointwise`: compare the original
  Conv1d and actual retained decoder path, including bias and a residual/next-Snake
  consumer. Each case runs ABBA slots, each one warmup plus five groups of five
  timed calls; run three independent processes, all serially.
- `codec_layers::cpu_pointwise::compare_fixed_decoder_bits`: compare all fixed
  code fixtures against their archived original raw decoder outputs, twice each.
  This is numeric verification, not a throughput benchmark.

Invoke each using `--ignored --exact TEST_NAME --nocapture --test-threads=1`
through `omnivoice_cpu_run.py`. Ordinary tests exercise shape/layout/fallback
coverage without models. Freeze pre-change and candidate `stage1_probe` binaries
with identical CPU features, then run ABBA per fixed-code case (one first decode
and five hot calls). Compare raw WAV bytes, complete short/long native codes and
all native F32/PCM16 WAVs. Keep formal short product timing separate from dumps
and correctness checks. The run record includes rejected materialized/transposed
layout screens; those alternatives are not runtime options.

## CPU real MLP layout screen

After building the CPU release library tests, run the frozen executable's
`stage0_qwen3::cpu_attribution::compare_real_mlp_layouts` using `--ignored --exact
TEST_NAME --nocapture --test-threads=1`, through `omnivoice_cpu_run.py` in three
serial independent processes. Set `OMNI_CPU_TRACE_MODEL` to the immutable model
and `OMNI_CPU_TRACE_ROOT` to the prior real-input capture directory. It requires
the nine `step-{0,15,31}-layer-{0,13,27}-replay.safetensors` archives.

The probe recomputes merged projections from actual `mlp_input` and weights,
instead of loading canonicalized gate/up as if their layout were unchanged.
Variants 0/1/2/3 retain split views / materialize gate / materialize up /
materialize both. All use the original SiLU and multiplication. Before timing,
compare every activation, MLP and residual output bit, twice per variant.
Each scope warms all variants once, then runs five groups of candidate-specific
ABBA slots, five calls per slot, rotating candidate order between groups.
`CPU_MLP_LAYOUT` JSON lines retain all samples; the first can follow the test
runner's name prefix, so parse the marker wherever it occurs in a line.
Activation and complete MLP+residual timings overlap and must not be added.
Explicit continuity was slower on all three real-input processes in the dated
2026-10-24 record; it remains an offline experiment, not a production option.

## CPU real decoder 7-tap layout screen

The private ignored test `codec_layers::cpu_conv7::compare_real_conv7_layouts`
uses the existing 130-frame `decoder-frames-130-block-{0..4}-res_unit1` and
`-conv1` tensor archives, plus immutable codec weights. Use the same CPU runner
and `OMNI_CPU_TRACE_ROOT` / `OMNI_CPU_TRACE_MODEL` settings as above. Run three
serial independent processes with the frozen release test executable.

Variants 0/1/2 compare original Conv1d / explicit NLC windows / the actual
restricted NCL production helper. Each real case is checked twice for complete
conv+bias and Snake+pointwise+residual output bits. The latter includes the
retained CPU pointwise path. Variants warm once per boundary, then run three
groups of candidate-specific ABBA slots, three calls per slot, alternating
candidate order between groups. Parse `CPU_CONV7` wherever it occurs in a line.

Component controls precompute operands: window materialization, GEMM and output
layout+bias each warm once and run five separate hot calls. These overlapping
controls are not production phase timings. In particular, explicit NLC windows
are a surrogate for the original private im2col implementation, not a direct
measurement of that implementation. Do not add components or consumers.

This probe covers dilation one; production restricts CPU Accelerate F32 to dense,
contiguous, square 32/64/128/256/512-channel weights and contiguous input, kernel
seven, stride one, padding three, groups one, and time length at least eight and
divisible by eight. Other configurations use original Conv1d. Verify complete
fixed-code decoder outputs and short/long native codes and F32/PCM16 WAVs before
retention; local component bits alone are insufficient.
