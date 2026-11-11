"""Development-only fixed IndexTTS Japanese oracle. Never called by inference."""
import argparse
import ast
import re
import os
import traceback
import warnings
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

REVISION = 'd9e41aac89fd00b3d71497fddb287b7f24613712'
UPSTREAM_SHA256 = 'ac58698c2b6a9163dde4ba841b7ef52c4c86dd02dd19b8d3c28a09a42e98c886'
CASES = [
    '今日はいい天気です。', '2026年に東京へ行きます。', '2026年11月11日、東京都。',
    '東京  大阪   hello', 'カタカナとひらがな、漢字。', 'Alice と zorbulate。',
    '<東京|とうきょう> と <日本|ニホン>。', '東京\t大阪　京都',
    'ちょうど 探しに行こうかなって 思っていたんだ。', '未知語🙂とABC123。',
    '  東京   ', '', '\t', '２０２６年、スーパーコンピューター。',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path, help='fixed upstream ja_g2p.py')
    parser.add_argument('front', type=Path, help='fixed official front.py')
    parser.add_argument('infer', type=Path, help='fixed official infer_v2_5.py')
    parser.add_argument('output', type=Path, help='new fixture path')
    args = parser.parse_args()
    if sha(args.upstream) != UPSTREAM_SHA256:
        raise ValueError('upstream ja_g2p.py identity mismatch')
    assert importlib.metadata.version('fugashi') == '1.4.0'
    assert importlib.metadata.version('unidic-lite') == '1.0.8'
    import unidic_lite
    spec = importlib.util.spec_from_file_location('ja_g2p', args.upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    processor = module.JapaneseG2PProcessor(g2p_ratio=0)
    front_source=args.front.read_bytes()
    infer_source=args.infer.read_bytes()
    if hashlib.sha256(front_source).hexdigest()!='df62a817c3e1b540ed62645850278df0463f934c562c1f3f50b85c2c05fc543d' or hashlib.sha256(infer_source).hexdigest()!='b69330053ed603530458bb181029b3abb4c83dc57c811e70837c04141293b56a':
        raise ValueError('official frontend source identity mismatch')
    front_class=next(node for node in ast.parse(front_source).body if isinstance(node,ast.ClassDef) and node.name=='TextNormalizer')
    scope=dict(re=re,os=os,traceback=traceback,warnings=warnings,lru_cache=lru_cache)
    exec(compile(ast.Module(body=[front_class],type_ignores=[]),str(args.front),'exec'),scope)
    frontend=scope['TextNormalizer']()
    definitions=[]
    for node in ast.parse(infer_source).body:
        if isinstance(node,ast.FunctionDef) and node.name in {'is_kana','apply_pronunciation_annotations'}:
            definitions.append(node)
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='PRONUNCIATION_ANNOTATION_PATTERN' for target in node.targets):
            definitions.append(node)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(args.infer),'exec'),scope)
    def prepare(text):
        cleaned=frontend.clean_pattern.sub(lambda match:frontend.char_rep_map[match.group()],text)
        annotated=scope['apply_pronunciation_annotations'](cleaned.lower())
        segmented=processor.process_ja_text(annotated)
        return re.sub(r'<\|([^|]+)\|>',lambda match:f'<|{match.group(1).upper()}|>',segmented)

    dictionary = Path(unidic_lite.DICDIR)
    rows = []
    for text in CASES:
        first = processor.process_ja_text(text)
        assert all(processor.process_ja_text(text) == first for _ in range(5))
        rows.append(dict(text=text, tokens=[surface for surface, _ in processor.tokenize(text)], output=first, prepared=prepare(text)))
    result = dict(front_sha256=hashlib.sha256(front_source).hexdigest(), infer_sha256=hashlib.sha256(infer_source).hexdigest(), upstream_revision=REVISION, upstream_sha256=sha(args.upstream),
                  fugashi='1.4.0', unidic_lite='1.0.8', unidic_version=(dictionary/'version').read_text().strip(),
                  dictionary_sha256={name:sha(dictionary/name) for name in ['sys.dic','unk.dic','char.bin','matrix.bin','dicrc','BSD']},
                  ratio=0, cases=rows)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
