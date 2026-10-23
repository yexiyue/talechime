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
