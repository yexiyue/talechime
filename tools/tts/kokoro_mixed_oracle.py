#!/usr/bin/env python3
"""Export independent pinned Misaki 1.1 + US English/eSpeak phoneme evidence.

Requires the pinned Misaki checkout and isolated dependencies documented in
crates/kokoro/SOURCE.md. Output must be a new file and contains public text only.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

REV = 'fba1236595f2d2bf21d414ba6e57d25256afada3'
SPACY_MODEL_SHA = '86cc141f63942d4b2c5fcee06630fd6f904788d2f0ab005cce45aadb8fb73889'
VERSIONS = {'misaki': '0.9.4', 'spacy': '3.7.5', 'en-core-web-sm': '3.7.1',
            'espeakng-loader': '0.2.4', 'phonemizer-fork': '3.3.2',
            'jieba': '0.42.1', 'pypinyin': '0.55.0', 'pypinyin-dict': '0.9.0',
            'cn2an': '0.5.24', 'num2words': '0.5.14'}
CASES = [
    ('dictionary', '你好hello世界。'),
    ('brand', '她说Apple很好。'),
    ('abbreviation', '这是S600。'),
    ('contraction', "他说don't走。"),
    ('hyphen', '你好hello-world朋友。'),
    ('english_words', '这是a book和an apple。'),
    ('context_polyphone', '银行旁边有Apple。'),
    ('oov', '你好zorbulate世界。'),
    ('oov_name', '这是Xylophrenzia。'),
    ('punctuation', 'Hello,世界!'),
    ('year_official', '今天是2026年。'),
    ('year_spaced_product_policy', '今天是2026 年。'),
    ('unsupported', '你好🙂'),
]


def verify_spacy_model(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != SPACY_MODEL_SHA:
        raise ValueError('spaCy model wheel digest mismatch')
    model_root = Path(metadata.distribution('en-core-web-sm').locate_file('en_core_web_sm'))
    import zipfile
    with zipfile.ZipFile(path) as wheel:
        for name in wheel.namelist():
            if name.startswith('en_core_web_sm/') and not name.endswith('/') and '__pycache__' not in name:
                if (model_root.parent / name).read_bytes() != wheel.read(name):
                    raise ValueError(f'installed spaCy model differs from pinned wheel: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--misaki-source', type=Path, required=True)
    parser.add_argument('--spacy-wheel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matrix', type=Path)
    parser.add_argument('--fullwidth-percent-only', action='store_true',
                        help='export the separate fullwidth percent boundary case')
    args = parser.parse_args()
    source = args.misaki_source.resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REV or subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', 'misaki'], text=True).strip():
        raise ValueError('Misaki checkout must match the pinned revision and be clean')
    actual = {name: metadata.version(name) for name in VERSIONS}
    if actual != VERSIONS:
        raise ValueError(f'unexpected dependency versions: {actual}')
    verify_spacy_model(args.spacy_wheel)
    sys.path.insert(0, str(source))
    from misaki import en, espeak
    from misaki.zh import ZHG2P
    english = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk='')
    def english_callable(text):
        return english(text)[0]
    frontend = ZHG2P(version='1.1', en_callable=english_callable)
    cases = []
    cases_to_export = ([(case['id'], case['text']) for case in json.loads(args.matrix.read_text())] if args.matrix else
                       [('fullwidth_percent', '你好％')] if args.fullwidth_percent_only else CASES)
    for name, text in cases_to_export:
        case = {'id': name, 'text': text}
        if args.matrix:
            case.update(source_text=text, source_sha256=hashlib.sha256(text.encode()).hexdigest())
        try:
            case['phonemes'] = frontend(text)[0]
        except Exception as error:
            if not args.matrix:
                raise
            case['official_error'] = f'{type(error).__name__}: {error}'
        cases.append(case)
    result = {'misaki_revision': REV, 'kokoro_revision': 'dfb907a02bba8152ca444717ca5d78747ccb4bec',
              'versions': actual, 'spacy_model_wheel_sha256': SPACY_MODEL_SHA,
              'english': 'US en.G2P(trf=False, fallback=EspeakFallback, unk="")',
              'source_sha256': {name: hashlib.sha256((source / 'misaki' / name).read_bytes()).hexdigest()
                                for name in ('zh.py', 'zh_frontend.py', 'en.py', 'espeak.py')},
              'cases': cases}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
