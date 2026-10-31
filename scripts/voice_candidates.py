#!/usr/bin/env python3
"""Bounded, resumable voice-design review batch. Audio stays outside Git."""
import argparse
import csv
import fcntl
import hashlib
import html
import json
import math
import os
import re
from pathlib import Path
import signal
import struct
import subprocess
import time
import zipfile

REFERENCE = '清晨，窗外的风渐渐停了。她合上书，起身走向门口。'
PREVIEW = '清晨的街道还很安静，阳光透过树叶，落在门前的石阶上。他停下脚步，认真听了片刻。“别着急，我们还有时间。”她轻声说道。远处传来钟声，两个人相视一笑，沿着熟悉的小路继续向前走去。昨天没有说完的故事，今天终于可以慢慢讲了。'
PROFILES = {
    'm-young-clear': '青年男声，清晰自然，中音区',
    'm-warm': '成年男声，温暖沉稳，轻微磁性',
    'm-mature': '中年男声，低沉克制，口齿清晰',
    'm-elder': '年长男声，温和有阅历，避免过度沙哑',
    'f-young-clear': '青年女声，清澈自然，避免尖细',
    'f-warm': '成年女声，温暖柔和，避免气声过重',
    'f-mature': '成熟女声，沉静知性，中低音区',
    'f-bright': '成年女声，明亮利落，保持叙述克制',
}
ENGINES = ('voxcpm', 'qwen-voicedesign', 'moss-voicedesign')
SEEDS = (42, 20261031)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('w') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.flush()
        os.fsync(out.fileno())
    temp.replace(path)


def wav_info(data):
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError('not RIFF WAV')
    offset, fmt, pcm = 12, None, None
    while offset + 8 <= len(data):
        kind, size = struct.unpack_from('<4sI', data, offset)
        chunk = data[offset + 8:offset + 8 + size]
        if len(chunk) != size:
            raise ValueError('truncated WAV')
        if kind == b'fmt ':
            fmt = list(struct.unpack_from('<HHIIHH', chunk))
            if fmt[0] == 65534:
                if len(chunk) < 40 or chunk[28:40] != bytes.fromhex('00001000800000aa00389b71'):
                    raise ValueError('invalid extensible WAV subtype')
                fmt[0] = struct.unpack_from('<I', chunk, 24)[0]
        if kind == b'data':
            pcm = chunk
        offset += 8 + size + size % 2
    if fmt is None or not pcm:
        raise ValueError('missing WAV format/audio')
    encoding, channels, rate, _, align, bits = fmt
    if channels != 1 or not rate or align != bits // 8 or len(pcm) % align:
        raise ValueError('invalid mono WAV format')
    if encoding == 3 and bits == 32:
        samples = [x[0] for x in struct.iter_unpack('<f', pcm)]
    elif encoding == 1 and bits == 16:
        samples = [x[0] / 32768 for x in struct.iter_unpack('<h', pcm)]
    else:
        raise ValueError(f'unsupported WAV {encoding}/{bits}')
    if not all(math.isfinite(x) for x in samples) or max(map(abs, samples)) < .001:
        raise ValueError('nonfinite/silent WAV')
    return {'seconds': len(samples) / rate, 'sample_rate': rate, 'sha256': digest(data)}


def matrix(batch):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', batch):
        raise ValueError('batch must be 1..32 ASCII letters, digits, hyphens or underscores')
    rows = []
    for engine in ENGINES:
        for profile, description in PROFILES.items():
            for seed in SEEDS:
                config = dict(engine=engine, profile=profile, seed=seed,
                              description=description + '，普通话，自然语速，中性克制地叙述。',
                              text=REFERENCE, device='metal', preview_text=PREVIEW, preview_seed=7)
                rows.append(dict(id=f'{batch}-{profile}-{engine.split("-")[0]}-{seed}',
                                 config=config, fingerprint=digest(json.dumps(config, sort_keys=True).encode()),
                                 design='pending', preview='pending', review='unreviewed'))
    assert len(rows) == 48 and len({r['id'] for r in rows}) == 48
    assert all(len(r['id']) <= 64 for r in rows)
    return rows


def run(command, log, timeout):
    start = time.monotonic()
    with log.open('wb') as out:
        process = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f'exit {code}: {log.read_text(errors="replace")[-1200:]}')
    return time.monotonic() - start


def asset(library, row):
    folder = library / row['id']
    record = json.loads((folder / 'voice.json').read_text())
    wav = (folder / 'reference.wav').read_bytes()
    conf = row['config']
    expected = digest(conf['text'].encode() + b'\0' + conf['description'].encode() + b'\0' + wav)
    if (record['id'] != row['id'] or record['transcript'] != conf['text'] or
            record.get('description') != conf['description'] or record['fingerprint'] != expected or
            record['source'] != {'kind': 'designed', 'engine': conf['engine']}):
        raise ValueError('existing asset identity/fingerprint conflict')
    return record, wav, wav_info(wav)


def index(root, state):
    cards = []
    for row in state['rows']:
        links = []
        for filename, label in [('reference.wav', '设计参考'), ('preview.wav', '固定参考克隆试听')]:
            path = root / row['id'] / filename
            stage = 'design' if filename == 'reference.wav' else 'preview'
            if row[stage] == 'succeeded' and path.is_file():
                links.append(f'<p>{label}</p><audio controls preload="none" src="{row["id"]}/{filename}"></audio>')
        cards.append(f'<article><h2>{html.escape(row["id"])}</h2><p>{html.escape(row["config"]["description"])}</p>'
                     f'<p>设计：{row["design"]}；试听：{row["preview"]}；人工：待确认</p>'
                     + ''.join(links) + f'<pre>{html.escape(row.get("error", ""))}</pre></article>')
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>旁白候选试听</title>'
        '<style>body{max-width:960px;margin:32px auto;font:16px system-ui;background:#f5f3ef;color:#252525}article{padding:20px;border-bottom:1px solid #ccc}audio{width:100%}pre{white-space:pre-wrap}h2{font-size:16px}</style>'
        '<h1>旁白候选 · 全部待人工确认</h1><p>先听设计参考，再听相同文本的 VoxCPM 克隆。自动检查不判断听感优劣。</p>'
        + f'<p>设计成功 {sum(r["design"] == "succeeded" for r in state["rows"])}/48；试听成功 {sum(r["preview"] == "succeeded" for r in state["rows"])}/48</p>'
        + ''.join(cards))
    feedback = root / 'feedback.csv'
    if not feedback.exists():
        with feedback.open('w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['id', '喜欢程度', '自然度', '音色稳定', '清晰度', '是否愿意长听', '备注'])
            writer.writerows([[r['id']] + [''] * 6 for r in state['rows']])


def execute(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'manifest.json'
    planned = matrix(args.batch)
    if path.exists():
        state = json.loads(path.read_text())
        if [(r['id'], r['fingerprint']) for r in planned] != [(r['id'], r['fingerprint']) for r in state['rows']]:
            raise ValueError('batch configuration changed; use a new batch')
    else:
        state = dict(version=1, rows=planned, resources=str(args.resources.resolve()),
                     binary_sha256=digest(args.binary.read_bytes()), preview_binary_sha256=digest(args.preview.read_bytes()),
                     created_unix=time.time(), run_seconds=0, script_sha256=digest(Path(__file__).read_bytes()),
                     resource_manifests={str(p): json.loads(p.read_text()) for p in [
                         Path('crates/moss-tts/assets/voice-design-1.7b.json'),
                         Path('crates/talechime-backends/src/qwen/models/1.7b-voicedesign.json'),
                         Path('crates/talechime-backends/src/voxcpm/resources.json')] if p.is_file()})
    if state['binary_sha256'] != digest(args.binary.read_bytes()) or state['preview_binary_sha256'] != digest(args.preview.read_bytes()):
        raise ValueError('build changed; use a new batch')
    if state.get('script_sha256') != digest(Path(__file__).read_bytes()):
        raise ValueError('batch tool changed; use a new output directory')
    if state['resources'] != str(args.resources.resolve()):
        raise ValueError('resource root changed; use a new batch')
    (root / 'preview.txt').write_text(PREVIEW)
    started = time.monotonic()
    previous_seconds = state.get('run_seconds', 0)
    deadline = started + min(args.hours * 3600, max(0, 21600 - previous_seconds))
    library = args.config.parent / 'voices'
    prefix = [str(args.binary.resolve()), '--config', str(args.config.resolve()), '--model-dir', str(args.resources.resolve())]
    for row in state['rows']:
        if time.monotonic() >= deadline:
            break
        folder = root / row['id']
        folder.mkdir(exist_ok=True)
        if row['config']['engine'] == 'moss-voicedesign' and not args.moss_verified:
            row.update(design='blocked', preview='blocked', error='MOSS Metal real-model gate has not passed')
            atomic_json(path, state)
            index(root, state)
            continue
        conf = row['config']
        if row['design'] in ('failed', 'blocked') and not args.retry_failed:
            continue
        try:
            if (library / row['id']).exists():
                record, wav, info = asset(library, row)
                if row.get('reference') and row['reference'] != info:
                    raise ValueError('previously accepted reference changed')
                row.update(design='succeeded', asset=record, reference=info)
            elif row['design'] == 'succeeded':
                raise ValueError('previously accepted library asset missing')
            else:
                row['design'] = 'running'
                atomic_json(path, state)
                command = prefix + ['voices', 'design', '--engine', conf['engine'], '--device', 'metal',
                    '--id', row['id'], '--name', row['id'], '--text', conf['text'],
                    '--description', conf['description'], '--seed', str(conf['seed'])]
                row['design_elapsed'] = run(command, folder / 'design.log', min(600, max(1, deadline-time.monotonic())))
                record, wav, info = asset(library, row)
                row.update(design='succeeded', asset=record, reference=info)
            atomic_json(path, state)
            reference_path = folder / 'reference.wav'
            if reference_path.exists() and reference_path.read_bytes() != wav:
                raise ValueError('exported reference changed')
            reference_path.write_bytes(wav)
            (folder / 'reference.txt').write_text(record['transcript'])
            bundle = folder / 'voice.voice'
            if not bundle.exists():
                run(prefix + ['voices', 'export', row['id'], '--output', str(bundle)], folder / 'export.log', 30)
            with zipfile.ZipFile(bundle) as archive:
                if not any(archive.read(name) == wav for name in archive.namelist() if name.endswith('.wav')):
                    raise ValueError('export bundle reference mismatch')
            if row['preview'] == 'succeeded':
                if wav_info((folder / 'preview.wav').read_bytes()) != row['preview_info']:
                    raise ValueError('preview integrity mismatch')
            elif row['preview'] not in ('failed',) or args.retry_failed:
                row['preview'] = 'running'
                atomic_json(path, state)
                row['preview_elapsed'] = run([str(args.preview.resolve()), str(args.resources.resolve()),
                    str(library.resolve()), row['id'], str(root / 'preview.txt'), str(folder / 'preview.wav')],
                    folder / 'preview.log', min(300, max(1, deadline-time.monotonic())))
                row.update(preview='succeeded', preview_info=wav_info((folder / 'preview.wav').read_bytes()))
            if row['preview'] == 'succeeded':
                row.pop('error', None)
        except Exception as error:
            row['error'] = str(error)
            row['preview' if row['design'] == 'succeeded' else 'design'] = 'failed'
        row['updated_unix'] = time.time()
        state['run_seconds'] = previous_seconds + time.monotonic() - started
        atomic_json(path, state)
        index(root, state)
        print(f'{row["id"]}: {row["design"]}/{row["preview"]}', flush=True)
    state['run_seconds'] = previous_seconds + time.monotonic() - started
    atomic_json(path, state)
    index(root, state)
    print(root / 'index.html', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('target/release/talechime'))
    parser.add_argument('--preview', type=Path, default=Path('target/release/examples/voice_candidates'))
    parser.add_argument('--resources', type=Path, default=Path.home() / '.talechime/resources')
    parser.add_argument('--config', type=Path, default=Path.home() / '.talechime/config.json')
    parser.add_argument('--batch', default='vc1031')
    parser.add_argument('--output', type=Path, default=Path('target/voice-candidates/20261031/review'))
    parser.add_argument('--hours', type=float, default=6)
    parser.add_argument('--moss-verified', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if not 0 < args.hours <= 6:
        parser.error('hours must be in (0, 6]')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.batch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)

if __name__ == '__main__':
    main()
