#!/usr/bin/env python3
"""Export pinned Misaki US eSpeak fallback outputs for isolated OOV tokens."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REV = 'fba1236595f2d2bf21d414ba6e57d25256afada3'
VERSIONS = {'misaki': '0.9.4', 'espeakng-loader': '0.2.4', 'phonemizer-fork': '3.3.2'}
WORDS = ('zorbulate', 'blorfington', 'quendorium')
FUNCTION_WORDS = ('de', 'le', 'the', 'a', 'to', 'of', 'in', 'la', 'du', 'zorbulate',
                  'xylophrenzia', 'quux', 'can', 'and', 'for', 'is', 'at', 'hello', 'i', 'an')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--misaki-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--function-words', action='store_true')
    args = parser.parse_args()
    source = args.misaki_source.resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REV or subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', 'misaki'], text=True).strip():
        raise ValueError('Misaki checkout must match the pinned revision and be clean')
    versions = {name: metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise ValueError(f'unexpected dependency versions: {versions}')
    sys.path.insert(0, str(source))
    from misaki.espeak import EspeakFallback
    fallback = EspeakFallback(british=False)
    cases = []
    for word in FUNCTION_WORDS if args.function_words else WORDS:
        phonemes, rating = fallback(SimpleNamespace(text=word))
        if not phonemes:
            raise ValueError(f'official fallback failed: {word}')
        cases.append({'id': word, 'text': word, 'phonemes': phonemes, 'rating': rating})
    data = {'misaki_revision': REV, 'versions': versions, 'mode': 'US eSpeak fallback in isolation',
            'source_sha256': hashlib.sha256((source / 'misaki/espeak.py').read_bytes()).hexdigest(),
            'cases': cases}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
