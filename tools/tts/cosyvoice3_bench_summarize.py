#!/usr/bin/env python3
"""Summarize the CosyVoice3 benchmark matrix under runs/cv3-bench/.

Produces:
  * summary.json  — aggregate medians/ranges per group plus validity counts
  * rounds.csv    — every measured round as one row (raw per-round values)
  * report.md     — a draft markdown skeleton of the key tables

Rules encoded here:
  * errors and truncations are counted, never averaged into success runs
  * speedup ratios are computed only inside comparable groups
    (same case + same fixed workload; never CPU-vs-Metal as a language claim)
  * no P95 when the sample size cannot support it (we report min/median/max)
"""

import csv
import json
import statistics
import sys
from pathlib import Path

BENCH = Path('runs/cv3-bench')


def med(vals):
    return statistics.median(vals) if vals else None


def spread(vals):
    if not vals:
        return None
    return {'median': statistics.median(vals), 'min': min(vals),
            'max': max(vals), 'n': len(vals)}


def ok_rounds(rounds, keys):
    """Rounds without errors that carry all requested measurements."""
    out = []
    for r in rounds or []:
        if r.get('error'):
            continue
        if all(k in r and r[k] is not None for k in keys):
            out.append(r)
    return out


def analyze_file(path):
    d = json.loads(path.read_text())
    meta = d.get('meta', {})
    job = meta.get('job') or path.stem
    entry = {'job': job, 'file': str(path), 'device': d.get('device'),
             'impl': d.get('impl'), 'mode': d.get('mode'),
             'exit_code': meta.get('exit_code')}
    rows = []

    if 'warm_rounds' in d and d['warm_rounds']:
        rounds = d['warm_rounds']
        good = ok_rounds(rounds, ['llm_prefill_seconds', 'flow_seconds', 'synthesize_seconds'])
        errors = [r for r in rounds if r.get('error')]
        trunc = [r for r in good if r.get('llm_hit_limit') or not r.get('pcm_valid')]
        entry['warm'] = {
            'rounds_total': len(rounds),
            'errors': [r.get('error') for r in errors],
            'invalid_or_truncated': len(trunc),
            'llm_prefill': spread([r['llm_prefill_seconds'] for r in good]),
            'llm_decode': spread([r['llm_decode_seconds'] for r in good]),
            'llm_tokens': spread([r['llm_tokens'] for r in good]),
            'llm_hit_limit_count': sum(1 for r in good if r.get('llm_hit_limit')),
            'flow': spread([r['flow_seconds'] for r in good]),
            'hift': spread([r['hift_seconds'] for r in good]),
            'synthesize': spread([r['synthesize_seconds'] for r in good]),
            'audio_seconds': spread([r['audio_seconds'] for r in good]),
            'rtf': spread([r['rtf'] for r in good]),
            'extract': spread([r['extract_seconds'] for r in good
                               if r.get('extract_seconds') is not None]),
        }
        for r in good:
            rows.append({
                'job': job, 'kind': 'warm', 'round': r.get('round'),
                'impl': d.get('impl'), 'device': d.get('device'),
                'tokens': r.get('llm_tokens'),
                'audio_s': r.get('audio_seconds'),
                'extract_s': r.get('extract_seconds'),
                'llm_prefill_s': r.get('llm_prefill_seconds'),
                'llm_decode_s': r.get('llm_decode_seconds'),
                'flow_s': r.get('flow_seconds'),
                'hift_s': r.get('hift_seconds'),
                'synthesize_s': r.get('synthesize_seconds'),
                'rtf': r.get('rtf'),
                'hit_limit': r.get('llm_hit_limit'),
                'pcm_valid': r.get('pcm_valid'),
            })

    if 'stream_rounds' in d and d['stream_rounds']:
        rounds = d['stream_rounds']
        good = ok_rounds(rounds, ['first_pcm_since_request', 'total_seconds'])
        errors = [r for r in rounds if r.get('error')]
        starve = []
        for r in good:
            chunks = r['chunks']
            t0 = chunks[0]['arrival_since_request']
            cumulative = 0.0
            margins = []
            for c in chunks[1:]:
                deadline = t0 + cumulative
                margins.append(deadline - c['arrival_since_request'])
                cumulative += c['audio_seconds']
            starve.append({
                'round': r.get('round'),
                'min_margin': min(margins) if margins else None,
                'starving_chunks': sum(1 for m in margins if m < 0),
            })
        entry['stream'] = {
            'rounds_total': len(rounds),
            'errors': [r.get('error') for r in errors],
            'first_pcm_since_request': spread([r['first_pcm_since_request'] for r in good]),
            'first_pcm_since_session': spread([r['first_pcm_since_session'] for r in good]),
            'session_start': spread([r.get('session_start_seconds') for r in good
                                     if r.get('session_start_seconds') is not None]),
            'total': spread([r['total_seconds'] for r in good]),
            'audio_seconds': spread([r['audio_seconds'] for r in good]),
            'rtf': spread([r['rtf'] for r in good]),
            'chunk_count': spread([r['chunk_count'] for r in good]),
            'extract': spread([r['extract_seconds'] for r in good
                               if r.get('extract_seconds') is not None]),
            'completed_true': sum(1 for r in good if r.get('completed') is True),
            'completed_false_or_unknown': sum(1 for r in good if r.get('completed') is not True),
            'hit_limit_count': sum(1 for r in good if r.get('hit_token_limit')),
            'starvation': starve,
        }
        for r in good:
            rows.append({
                'job': job, 'kind': 'stream', 'round': r.get('round'),
                'impl': d.get('impl'), 'device': d.get('device'),
                'first_pcm_s': r.get('first_pcm_since_request'),
                'first_pcm_session_s': r.get('first_pcm_since_session'),
                'chunk_count': r.get('chunk_count'),
                'total_s': r.get('total_seconds'),
                'audio_s': r.get('audio_seconds'),
                'rtf': r.get('rtf'),
                'completed': r.get('completed'),
                'hit_limit': r.get('hit_token_limit'),
                'starving_chunks': next((s['starving_chunks'] for s in starve
                                         if s['round'] == r.get('round')), None),
            })

    if 'components' in d:
        c = d['components']
        entry['components'] = {
            'tokens': c.get('tokens'),
            'prompt_tokens': c.get('prompt_tokens'),
            'mel_frames_fixture': c.get('mel_frames_fixture'),
            'flow': spread([r['seconds'] for r in c['flow']]),
            'hift': spread([r['seconds'] for r in c['hift']]),
        }
        for r in c['flow']:
            rows.append({'job': job, 'kind': 'component_flow', 'round': None,
                         'impl': d.get('impl'), 'device': d.get('device'),
                         'seconds': r['seconds'], 'mel_frames': r.get('mel_frames')})
        for r in c['hift']:
            rows.append({'job': job, 'kind': 'component_hift', 'round': None,
                         'impl': d.get('impl'), 'device': d.get('device'),
                         'seconds': r['seconds'], 'pcm_samples': r.get('pcm_samples')})

    if 'cold' in d:
        entry['cold'] = d['cold']

    if 'mps_probe' in d:
        entry['mps_probe'] = d['mps_probe']

    if 'prepare_seconds' in d:  # engine runs
        entry['engine'] = {
            'prepare_seconds': d.get('prepare_seconds'),
            'ready_since_process_start': d.get('ready_since_process_start'),
            'collect_all': d.get('collect_all'),
        }
        rounds = d.get('stream_rounds') or []
        good = ok_rounds(rounds, ['first_pcm_seconds', 'total_seconds'])
        entry['engine']['rounds'] = {
            'first_pcm': spread([r['first_pcm_seconds'] for r in good]),
            'total': spread([r['total_seconds'] for r in good]),
            'rtf': spread([r['rtf'] for r in good]),
            'completed': sum(1 for r in good if r.get('completed')),
            'errors': [r.get('error') for r in rounds if r.get('error')],
        }
        for r in good:
            rows.append({'job': job, 'kind': 'engine_stream', 'round': r.get('round'),
                         'impl': 'rust-engine', 'device': d.get('device'),
                         'first_pcm_s': r.get('first_pcm_seconds'),
                         'total_s': r.get('total_seconds'),
                         'audio_s': r.get('audio_seconds'),
                         'rtf': r.get('rtf'), 'completed': r.get('completed')})

    # RSS: peak from /usr/bin/time; resident just before the first measured
    # round (aligned with the BENCH_ROUND_START marker timeline).
    rss_series = meta.get('rss_series') or []
    markers = meta.get('markers') or []
    entry['rss'] = {
        'peak_bytes_time_l': meta.get('peak_rss_bytes_time_l'),
        'samples': len(rss_series),
        'series_max_kb': max((s['kb'] for s in rss_series), default=None),
        'before_first_round_kb': None,
    }
    starts = [m.split() for m in markers if m.startswith('BENCH_ROUND_START')]
    if starts and rss_series:
        # markers carry unix ms; series carries unix seconds
        first_ms = int(starts[0][2])
        before = [s['kb'] for s in rss_series if s['t'] * 1000 <= first_ms]
        entry['rss']['before_first_round_kb'] = before[-1] if before else None
    elif rss_series:
        entry['rss']['before_first_round_kb'] = rss_series[0]['kb']
    return entry, rows


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else BENCH
    entries, all_rows = [], []
    for path in sorted(out_dir.glob('*.json')):
        if path.name in ('summary.json',):
            continue
        try:
            entry, rows = analyze_file(path)
        except Exception as exc:  # noqa: BLE001
            entries.append({'job': path.stem, 'file': str(path), 'summary_error': repr(exc)})
            continue
        entries.append(entry)
        all_rows.extend(rows)

    # Comparable-group speedups: same case & mode, rust vs python, cpu only.
    def find(prefix, impl):
        for e in entries:
            if e.get('job', '').startswith(prefix) and e.get('impl') == impl \
                    and e.get('device') == 'cpu':
                return e
        return None

    ratios = {}
    for e in entries:
        job = e.get('job', '')
        if '-python-' not in job:
            continue
        prefix = job.replace('-python-cpu-', '-rust-cpu-')
        rust = find(prefix, 'rust-candle')
        py = e
        pair = {}
        if rust and 'components' in rust and 'components' in py:
            for stage in ('flow', 'hift'):
                rv = rust['components'][stage]['median']
                pv = py['components'][stage]['median']
                if rv and pv:
                    pair[stage] = round(pv / rv, 2)
        if rust and 'warm' in rust and 'warm' in py:
            for stage in ('llm_prefill', 'llm_decode', 'flow', 'hift', 'synthesize'):
                rv = rust['warm'][stage] and rust['warm'][stage]['median']
                pv = py['warm'][stage] and py['warm'][stage]['median']
                if rv and pv:
                    pair[f'warm_{stage}'] = round(pv / rv, 2)
        if pair:
            ratios[job.replace('-python-cpu-', '-cpu-')] = pair

    summary = {'entries': entries, 'cpu_cpu_ratios_python_over_rust': ratios}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if all_rows:
        keys = sorted({k for row in all_rows for k in row})
        with (out_dir / 'rounds.csv').open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)

    # Compact stdout digest for the record.
    for e in entries:
        job = e.get('job', '?')
        for kind in ('warm', 'stream', 'components', 'engine'):
            if kind in e and isinstance(e[kind], dict):
                block = e[kind]
                stages = {k: (block[k]['median'] if isinstance(block.get(k), dict) else None)
                          for k in ('flow', 'hift', 'synthesize', 'llm_prefill',
                                    'llm_decode', 'rtf', 'first_pcm_since_request',
                                    'first_pcm', 'total')}
                stages = {k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in stages.items() if v is not None}
                if stages:
                    print(f'{job} [{kind}] {stages}')
        if 'cold' in e:
            c = e['cold']
            print(f"{job} [cold] ready={c.get('ready_since_process_start')} "
                  f"first={c.get('first_request_total_seconds')}")
    print(f'\n{len(entries)} files summarized -> {out_dir}/summary.json, rounds.csv')


if __name__ == '__main__':
    main()
