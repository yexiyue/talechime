"""Run one offline CPU probe serially, retaining logs and compiler observations.

Usage: python3 .../omnivoice_cpu_run.py OUTPUT_PREFIX -- COMMAND [ARG ...]
Inference and build scheduling remains the caller's responsibility.
"""
import argparse
import json
import os
import pathlib
import subprocess
import time


def activity():
    rows = subprocess.check_output(["ps", "-axo", "pid=,pcpu=,comm="], text=True).splitlines()
    compilers = {"cargo", "cargo-clippy", "rustc", "clang", "clang++", "ld", "clippy-driver"}
    return [row.strip() for row in rows if len(row.split(None, 2)) == 3
            and pathlib.Path(row.split(None, 2)[2]).name in compilers]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=pathlib.Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing probe command")
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RAYON_NUM_THREADS="8", VECLIB_MAXIMUM_THREADS="8",
               OMP_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key in ("OMNIVOICE_SYNC_PROBE", "OMNIVOICE_STAGE_TIME", "OMNIVOICE_LAYER_PROBE"):
        env.pop(key, None)
    start = time.monotonic()
    seen, peak = [], 0
    with open(str(args.prefix) + ".out", "x") as out, open(str(args.prefix) + ".err", "x") as err:
        process = subprocess.Popen(command, env=env, stdout=out, stderr=err)
        while process.poll() is None:
            rows = activity()
            if rows:
                seen.append(dict(seconds=time.monotonic()-start, processes=rows))
            rss = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], text=True,
                                 capture_output=True).stdout.strip()
            if rss:
                peak = max(peak, int(rss))
            if time.monotonic()-start > 1200:
                process.terminate()
                process.wait()
                raise TimeoutError("own probe exceeded 20 minutes")
            time.sleep(0.5)
    receipt = dict(command=command, returncode=process.returncode,
                   elapsed_seconds=time.monotonic()-start, peak_rss_kib=peak,
                   threads={k: env[k] for k in ("RAYON_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS")},
                   activity=seen)
    with open(str(args.prefix) + ".receipt.json", "x") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(dict(prefix=str(args.prefix), returncode=process.returncode,
                         activity_observations=len(seen))), flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
