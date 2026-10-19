#!/usr/bin/env python3
"""Run official VoxCPM MPS modes through the shared short-process ABBA protocol."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from voxcpm_perf_common import cooldown_seconds, hot_rounds, run_abba


def run_arm(arm, index, args, output):
    directory = output / f"{arm}-{index}"
    mode = getattr(args, f"{arm}_mode")
    command = [args.python, args.probe, "--source", args.source, "--models", args.models,
               "--segments", args.segments, "--reference", args.reference, "--transcript", args.transcript,
               "--output", str(directory), "--mode", mode, "--rounds", str(args.rounds), "--warmup-rounds", "1"]
    started = time.monotonic()
    with (output / f"run-{index}.log").open("w") as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
    if code:
        raise SystemExit(f"official probe failed ({code}): {directory}")
    data = json.loads((directory / "summary.json").read_text())
    if not data.get("available"):
        raise SystemExit(f"official {mode} path unavailable: {data.get('reason')}")
    if data.get("output_checks") != "passed":
        raise SystemExit(f"official PCM checks failed: {directory}")
    if data.get("device_audit"):
        raise SystemExit(f"instrumented device audit cannot measure throughput: {directory}")
    return {"index": index, "arm": arm, "mode": mode, "rtf_mean": data["measured"]["rtf_mean"],
            "rtf_min": data["measured"]["rtf_min"], "rtf_max": data["measured"]["rtf_max"],
            "first_pcm_ms": None, "wall_seconds": round(time.monotonic() - started, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--baseline-mode", choices=("eager", "optimize"), required=True)
    parser.add_argument("--candidate-mode", choices=("eager", "optimize"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=hot_rounds, default=1)
    parser.add_argument("--cooldown", type=cooldown_seconds, default=20.0)
    parser.add_argument("--screen", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    statistics = run_abba(args.output, args.screen, args.cooldown,
                          lambda arm, index: run_arm(arm, index, args, args.output))
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
