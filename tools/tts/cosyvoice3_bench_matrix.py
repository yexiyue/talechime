#!/usr/bin/env python3
"""Orchestrate the CosyVoice3 Rust-vs-Python benchmark matrix.

Alternates implementations per case to reduce thermal drift, wraps every
process with /usr/bin/time -l (peak RSS) and samples `ps -o rss=` every
0.5 s, aligning the timeline with the tools' BENCH_READY / BENCH_ROUND_*
stderr markers. Merges everything into one JSON per run under OUT_DIR.

Run groups (see --list); paths assume the repository root as cwd:

  python3 tools/tts/cosyvoice3_bench_matrix.py --list
  python3 tools/tts/cosyvoice3_bench_matrix.py --group warm-cpu
  python3 tools/tts/cosyvoice3_bench_matrix.py --group all

Ordinary tests never download models; every model and artifact lives in
git-ignored runs/. The first process after boot pays disk page-in on the
external volume; run `--group warmup` once before cold-start groups and
keep the filesystem-cache state notes in the report in mind.
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RUST_BENCH = ['target/release/examples/bench']
RUST_ENGINE = ['target/release/examples/cosyvoice3_engine_bench']
PY = ['runs/cv3-venv/bin/python', 'tools/tts/cosyvoice3_bench.py']
PY_ENV = {
    'PYTHONPATH': 'runs/reference/CosyVoice:runs/reference/matcha',
}
RUST_ENV = {'RAYON_NUM_THREADS': '8'}

CASES = {
    'short': 'runs/cv3-case-short',
    'long': 'runs/cv3-case-long',
    'xlong': 'runs/cv3-case-xlong',
    'longprompt': 'runs/cv3-case-longprompt',
}
ROUNDS = {'short': 5, 'long': 5, 'xlong': 5, 'longprompt': 5, 'default': 5}
THREADS = 8


def rust_job(mode, case, device, out, extra=(), rounds=None):
    rounds = rounds or ROUNDS.get(case, ROUNDS['default'])
    cmd = RUST_BENCH + [
        'runs/cv3-converted', CASES[case], device, mode,
        '--rounds', str(rounds), '--threads', str(THREADS),
        '--label', f'{case}-{mode}-{device}',
        '--out', str(out), *[str(e) for e in extra],
    ]
    return cmd, dict(RUST_ENV)


def py_job(mode, case, out, extra=(), rounds=None, threads=THREADS):
    rounds = rounds or ROUNDS.get(case, ROUNDS['default'])
    cmd = PY + [
        '--model_dir', 'runs/cv3-model', '--case_dir', CASES[case],
        '--mode', mode, '--rounds', str(rounds), '--threads', str(threads),
        '--label', f'{case}-{mode}', '--out', str(out), *[str(e) for e in extra],
    ]
    return cmd, PY_ENV


def engine_job(device, out, rounds=5):
    cmd = RUST_ENGINE + [
        'runs/cv3-backend-resources', 'custom:bench', device,
        '--rounds', str(rounds), '--label', f'engine-{device}',
        '--out', str(out),
    ]
    return cmd, dict(RUST_ENV)


def build_jobs(group, out_dir):
    """Interleave rust/python per case; groups are independent."""
    jobs = []

    def add(name, cmd, env, note=''):
        jobs.append({'name': name, 'cmd': cmd, 'env': env, 'note': note})

    if group in ('warmup', 'all'):
        add('warmup-page-cache',
            RUST_BENCH + ['runs/cv3-converted', CASES['short'], 'metal',
                          'components', '--rounds', '1', '--threads', '8',
                          '--label', 'warmup', '--out', str(out_dir / 'warmup.json')],
            {}, 'warms the file cache so later cold runs measure process cold start, not disk')

    if group in ('components', 'all'):
        for case in ('short', 'long'):
            for device in ('cpu', 'metal'):
                cmd, env = rust_job('components', case, device,
                                    out_dir / f'components-rust-{device}-{case}.json')
                add(f'components-rust-{device}-{case}', cmd, env)
            cmd, env = py_job('components', case,
                              out_dir / f'components-python-cpu-{case}.json')
            add(f'components-python-cpu-{case}', cmd, env)

    if group in ('warm', 'all'):
        for case in ('short', 'long', 'xlong', 'longprompt'):
            # rust cpu / python cpu alternated per case
            cmd, env = rust_job('warm', case, 'cpu',
                                out_dir / f'warm-rust-cpu-{case}.json')
            add(f'warm-rust-cpu-{case}', cmd, env)
            cmd, env = py_job('warm', case,
                              out_dir / f'warm-python-cpu-{case}.json')
            add(f'warm-python-cpu-{case}', cmd, env)
            cmd, env = rust_job('warm', case, 'metal',
                                out_dir / f'warm-rust-metal-{case}.json')
            add(f'warm-rust-metal-{case}', cmd, env)

    if group in ('reextract', 'all'):
        for case in ('short', 'longprompt'):
            cmd, env = rust_job('warm', case, 'cpu',
                                out_dir / f'reextract-rust-cpu-{case}.json',
                                extra=('--reextract',), rounds=3)
            add(f'reextract-rust-cpu-{case}', cmd, env)
            cmd, env = py_job('warm', case,
                              out_dir / f'reextract-python-cpu-{case}.json',
                              extra=('--reextract',), rounds=3)
            add(f'reextract-python-cpu-{case}', cmd, env)
            cmd, env = rust_job('warm', case, 'metal',
                                out_dir / f'reextract-rust-metal-{case}.json',
                                extra=('--reextract',), rounds=3)
            add(f'reextract-rust-metal-{case}', cmd, env)

    if group in ('stream', 'all'):
        for case in ('short', 'long', 'xlong'):
            cmd, env = rust_job('stream', case, 'cpu',
                                out_dir / f'stream-rust-cpu-{case}.json')
            add(f'stream-rust-cpu-{case}', cmd, env)
            cmd, env = py_job('stream', case,
                              out_dir / f'stream-python-cpu-{case}.json')
            add(f'stream-python-cpu-{case}', cmd, env)
            cmd, env = rust_job('stream', case, 'metal',
                                out_dir / f'stream-rust-metal-{case}.json')
            add(f'stream-rust-metal-{case}', cmd, env)

    if group in ('cold', 'all'):
        for impl in ('rust-cpu', 'python-cpu', 'rust-metal'):
            for rep in range(3):
                device = impl.split('-')[1]
                out = out_dir / f'cold-{impl}-{rep}.json'
                if impl.startswith('rust'):
                    cmd, env = rust_job('cold', 'short', device, out,
                                        extra=('--reextract',), rounds=1)
                else:
                    cmd, env = py_job('cold', 'short', out,
                                      extra=('--reextract',), rounds=1)
                add(f'cold-{impl}-{rep}', cmd, env,
                    'fresh process; fs cache warm (see report methodology)')

    if group in ('sensitivity', 'all'):
        cmd = RUST_BENCH + ['runs/cv3-converted', CASES['long'], 'cpu', 'warm',
                            '--rounds', '3', '--threads', str(THREADS),
                            '--label', 'rust-default-threads',
                            '--out', str(out_dir / 'sens-rust-cpu-default-threads.json')]
        add('sens-rust-cpu-default-threads', cmd, {}, 'RAYON_NUM_THREADS unset (default pool)')
        cmd, env = py_job('warm', 'long',
                          out_dir / 'sens-python-cpu-default-threads.json',
                          rounds=3, threads=0)
        add('sens-python-cpu-default-threads', cmd, env,
            'torch default threads (no set_num_threads call; --threads 0)')

    if group in ('engine', 'all'):
        for device in ('metal', 'cpu'):
            cmd, env = engine_job(device, out_dir / f'engine-{device}.json')
            add(f'engine-{device}', cmd, env,
                'full Talechime Engine backend path (per-request reference re-extraction)')

    if group in ('mps', 'all'):
        cmd, env = py_job('mps_probe', 'short', out_dir / 'mps-probe.json', rounds=1)
        add('mps-probe', cmd, env,
            'unsupported official path; evidence for the report, never a comparison row')

    return jobs


def sample_rss(pid, stop, series):
    while not stop.is_set():
        try:
            out = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)],
                                 capture_output=True, text=True, timeout=5)
            kb = int(out.stdout.strip() or 0)
            if kb:
                series.append({'t': time.time(), 'kb': kb})
        except Exception:  # noqa: BLE001 — sampling is best effort
            pass
        stop.wait(0.5)


def run_job(job, out_dir):
    out_path = Path(job['cmd'][job['cmd'].index('--out') + 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_log = out_path.with_suffix('.time.txt')
    started = time.time()
    env = {**dict(__import__('os').environ), **job['env']}
    with time_log.open('w') as tf:
        proc = subprocess.Popen(
            ['/usr/bin/time', '-l', '--'] + job['cmd'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=REPO)
        stop = threading.Event()
        series = []
        # /usr/bin/time forks the real command; sample the child, not the
        # wrapper (the wrapper's own RSS is ~1 MiB).
        child_pid = None
        for _ in range(40):
            out = subprocess.run(['ps', '-o', 'pid=', '--ppid', str(proc.pid)],
                                 capture_output=True, text=True)
            pids = [line.strip() for line in out.stdout.splitlines() if line.strip()]
            if pids:
                child_pid = int(pids[0])
                break
            if stop.wait(0.25):
                break
        sampler = threading.Thread(
            target=sample_rss, args=(child_pid or proc.pid, stop, series), daemon=True)
        sampler.start()
        stdout, stderr = proc.communicate()
        stop.set()
        sampler.join(timeout=2)
        tf.write(f'$ {shlex.join(str(c) for c in job["cmd"])}\nenv={job["env"]}\n')
        tf.write(stderr)
    elapsed = time.time() - started
    peak = 0
    # macOS prints the value before the label, Linux after it.
    m = (re.search(r'(\d+)\s+peak memory footprint', stderr)
         or re.search(r'maximum resident set size.*?(\d+)', stderr))
    if m:
        peak = int(m.group(1))  # bytes on macOS
    markers = [line for line in stderr.splitlines() if line.startswith('BENCH_')]
    payload = {}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            payload = {'parse_error': True}
    payload.setdefault('meta', {}).update({
        'job': job['name'],
        'note': job['note'],
        'wall_seconds': elapsed,
        'exit_code': proc.returncode,
        'peak_rss_bytes_time_l': peak,
        'rss_series': series,
        'markers': markers,
        'stdout_tail': stdout[-400:],
    })
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    status = 'ok' if proc.returncode == 0 else f'EXIT {proc.returncode}'
    print(f'[{status}] {job["name"]} wall={elapsed:.1f}s peakRSS={peak / 2**20:.0f}MiB',
          file=sys.stderr, flush=True)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', default='all')
    parser.add_argument('--out-dir', default='runs/cv3-bench')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--only', default=None,
                        help='substring filter on job names')
    parser.add_argument('--force', action='store_true',
                        help='rerun jobs whose output JSON already exists')
    args = parser.parse_args()

    out_dir = REPO / args.out_dir
    jobs = build_jobs(args.group, out_dir)
    if args.only:
        jobs = [j for j in jobs if args.only in j['name']]
    if args.list:
        for j in jobs:
            print(f"{j['name']:44s} {j['note']}")
        return
    failures = 0
    for job in jobs:
        out = Path(job['cmd'][job['cmd'].index('--out') + 1])
        if not args.force and out.exists():
            try:
                if json.loads(out.read_text()).get('meta', {}).get('job') == job['name']:
                    print(f'[skip] {job["name"]} (exists)', file=sys.stderr, flush=True)
                    continue
            except json.JSONDecodeError:
                pass
        if run_job(job, out_dir) != 0:
            failures += 1
    print(f'done: {len(jobs)} jobs, {failures} failed', file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
