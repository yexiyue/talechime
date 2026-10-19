#!/usr/bin/env python3
"""Interleaved VoxCPM Metal A/B runner for macOS.

Why this does not reuse the "1 warmup + 5 hot rounds" protocol: on an M4 Pro,
sustained Metal load inflates single rounds by up to 24% partway through a long
process and then recovers on its own. See
docs/records/voxcpm/voxcpm-metal-performance-2026-10-18.md. Each arm therefore
runs in its own short-lived process and the arms alternate, so drift that lands
on one arm cannot be reported as an improvement.

The two arms are two prebuilt probe binaries, mirroring the Windows stage-1
runner. Never build, profile, or run a second inference process during a sweep.

    tools/tts/voxcpm_perf_macos.py -m MODEL -r REF.wav -t TRANSCRIPT \
        --baseline target/release/examples/voxcpm_candle_probe \
        --candidate target/release/examples/voxcpm_candle_probe_after

Bit-exactness is a separate gate; compare the WAV trees with compare_wav.py.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from voxcpm_perf_common import cooldown_seconds, hot_rounds, run_abba


def run_arm(arm, index, args, output):
    probe = getattr(args, arm)
    directory = output / f"{arm}-{index}"
    env = dict(os.environ)
    env["VOXCPM_BENCH_ROUNDS"] = str(args.rounds)
    env["VOXCPM_BENCH_WARMUP"] = "1"
    env["VOXCPM_BENCH_PRECISION"] = getattr(args, f"{arm}_precision")
    for stale in ("VOXCPM_BENCH_PROFILE", "VOXCPM_CUDA_GRAPH", "VOXCPM_DECODER_GRAPH"):
        env.pop(stale, None)
    started = time.monotonic()
    model = args.candidate_model if arm == "candidate" and args.candidate_model else args.model
    command = [probe, model, "metal", str(directory), args.corpus,
               args.reference, args.transcript]
    with (output / f"run-{index}.log").open("w") as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if code != 0:
        raise SystemExit(f"probe failed ({code}): {directory}; see run-{index}.log")
    summary = directory / "summary.json"
    if not summary.is_file():
        raise SystemExit(f"probe produced no summary: {directory}")
    data = json.loads(summary.read_text())
    if data.get("regressions") != "passed":
        raise SystemExit(f"probe regressions failed: {directory}")
    record = {
        "index": index,
        "arm": arm,
        "probe": probe,
        "model": model,
        "precision": env["VOXCPM_BENCH_PRECISION"],
        "rtf_mean": data["measured"]["rtf_mean"],
        "rtf_min": data["measured"]["rtf_min"],
        "rtf_max": data["measured"]["rtf_max"],
        "first_pcm_ms": next(
            r["first_pcm_ms"] for r in data["rounds"] if not r.get("warmup")
        ),
        "wall_seconds": round(time.monotonic() - started, 1),
    }
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--candidate-model",
                        help="model directory for the candidate; defaults to --model")
    parser.add_argument("-r", "--reference", required=True)
    parser.add_argument("-t", "--transcript", required=True)
    parser.add_argument("-o", "--output")
    parser.add_argument("-c", "--corpus", default="tools/tts/acceptance-corpus.txt")
    parser.add_argument("-n", "--rounds", type=hot_rounds, default=1,
                        help="hot rounds per process; 1..3, short is safer")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-precision", choices=("q8", "f16", "bf16", "f32"), default="q8")
    parser.add_argument("--candidate-precision", choices=("q8", "f16", "bf16", "f32"), default="q8")
    parser.add_argument("--screen", action="store_true",
                        help="four symmetric processes only; never an acceptance result")
    parser.add_argument("--cooldown", type=cooldown_seconds, default=20.0,
                        help="seconds of idle between processes")
    args = parser.parse_args()

    for path in (args.model, args.candidate_model, args.reference, args.transcript, args.corpus):
        if path is None:
            continue
        if not Path(path).is_file():
            if path in (args.model, args.candidate_model):
                if not Path(path).is_dir():
                    raise SystemExit(f"missing model directory: {path}")
            else:
                raise SystemExit(f"missing input: {path}")
    for probe in (args.baseline, args.candidate):
        if not os.access(probe, os.X_OK):
            raise SystemExit(f"probe is not executable: {probe}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output = Path(args.output or f"target/tmp/voxcpm-metal/abba-{stamp}")
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    stats = run_abba(output, args.screen, args.cooldown,
                     lambda arm, index: run_arm(arm, index, args, output))
    print(json.dumps(stats, indent=2))
    if stats["performance_gate"]:
        print("\nPerformance gate passed. Still required: identical WAVs via compare_wav.py.")
    else:
        print("\nNo confirmed gain; inspect paired results and drift before drawing a conclusion.")
    print(f"Artifacts: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
