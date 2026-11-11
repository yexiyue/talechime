#!/usr/bin/env python3
"""Generate mixed phonemes with pinned Misaki plus independent g2pW predictions."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import subprocess
import sys

from kokoro_mixed_oracle import CASES, REV, VERSIONS, SPACY_MODEL_SHA, verify_spacy_model

G2PW_REV = 'b9328ce8129c4717a70f16a2688542a0197541dd'
MODEL_SHA = 'bb40c8c7b5baa755b2acd317c6bc5a65e4af7b80c40a569247fbd76989299999'
VOCAB_SHA = '45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c'
CONTEXT_SHA = '5bedd0e4460625bc57a92d3edd9ff573f68167d8f62fa259ac90a123fc27b381'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('misaki-source', 'g2pw-source', 'model-dir', 'tokenizer-dir', 'context-dir', 'spacy-wheel', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--matrix', type=Path)
    a = p.parse_args()
    for source, revision in ((a.misaki_source, REV), (a.g2pw_source, G2PW_REV)):
        if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != revision:
            raise ValueError(f'unexpected source revision: {source}')
        if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
            raise ValueError(f'dirty source: {source}')
    if digest(a.model_dir / 'g2pw.onnx') != MODEL_SHA or digest(a.tokenizer_dir / 'vocab.txt') != VOCAB_SHA:
        raise ValueError('model or tokenizer identity mismatch')
    versions = {name: metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise ValueError(f'unexpected dependency versions: {versions}')
    verify_spacy_model(a.spacy_wheel)
    sys.path[:0] = [str(a.misaki_source.resolve()), str(a.g2pw_source.resolve())]
    from g2pw import G2PWConverter
    from misaki import en, espeak
    from misaki.zh import ZHG2P
    from pypinyin import Style
    from pypinyin.converter import UltimateConverter
    if digest(a.context_dir / 'context.json') != CONTEXT_SHA:
        raise ValueError('context tables identity mismatch')
    context = json.loads((a.context_dir / 'context.json').read_text())
    if context['model_sha256'] != MODEL_SHA or context['source_revision'] != G2PW_REV:
        raise ValueError('context metadata does not match pinned g2pW')
    converter = G2PWConverter(model_dir=str(a.model_dir), model_source=str(a.tokenizer_dir),
                              style='pinyin', enable_non_tradional_chinese=True, num_workers=0, batch_size=32)
    english = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk='')
    frontend = ZHG2P(version='1.1', en_callable=lambda text: english(text)[0])
    base_frontend = frontend.frontend
    original = base_frontend._get_initials_finals
    conversion = UltimateConverter(neutral_tone_with_five=True)
    s2t = context['s2t']
    polyphonic = context['char2phonemes']

    class ContextFrontend:
        def __call__(self, text):
            predictions = converter([text])[0]
            cursor = 0
            def readings(word):
                nonlocal cursor
                position = text.find(word, cursor)
                if position < 0:
                    raise ValueError(f'word mapping failed: {word!r} in {text!r}')
                cursor = position + len(word)
                initials, finals = original(word)
                if len(initials) != len(word):
                    raise ValueError('pinyin word alignment failed')
                for i, char in enumerate(word):
                    if s2t.get(char, char) not in polyphonic:
                        continue
                    value = predictions[position + i]
                    if not value:
                        raise ValueError(f'no prediction for {char}')
                    initials[i] = conversion.convert_style(char, value, Style.INITIALS, strict=True)
                    finals[i] = conversion.convert_style(char, value, Style.FINALS_TONE3, strict=True)
                    if char == '嗯':
                        finals[i] = 'n2'
                    if re.fullmatch(r'i\d', finals[i]):
                        if initials[i] in ('z', 'c', 's'):
                            finals[i] = 'i' + finals[i]
                        elif initials[i] in ('zh', 'ch', 'sh', 'r'):
                            finals[i] = 'ii' + finals[i]
                return initials, finals
            base_frontend._get_initials_finals = readings
            try:
                return base_frontend(text)
            finally:
                base_frontend._get_initials_finals = original
    frontend.frontend = ContextFrontend()
    cases = []
    selected = [(c['id'], c['text']) for c in json.loads(a.matrix.read_text())] if a.matrix else CASES
    for name, text in selected:
        case = {'id': name, 'text': text}
        if a.matrix:
            case.update(source_text=text, source_sha256=hashlib.sha256(text.encode()).hexdigest())
        try:
            case['phonemes'] = frontend(text)[0]
        except Exception as error:
            if not a.matrix:
                raise
            case['official_error'] = f'{type(error).__name__}: {error}'
        cases.append(case)
    data = {'misaki_revision': REV, 'g2pw_revision': G2PW_REV, 'model_sha256': MODEL_SHA,
            'vocab_sha256': VOCAB_SHA, 'context_sha256': CONTEXT_SHA,
            'spacy_model_wheel_sha256': SPACY_MODEL_SHA, 'versions': versions, 'onnxruntime': metadata.version('onnxruntime'),
            'method': 'official Misaki tone rules with official g2pW predictions per Mandarin span', 'cases': cases}
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
