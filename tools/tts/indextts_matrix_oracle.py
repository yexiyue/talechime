"""Fixed official IndexTTS frontend evidence for every public matrix input.

Developer-only Darwin WeText / UniDic execution; Spanish reuses the exact
independently exported NeMo case. No model inference or downloads.
"""
import argparse
import ast
from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import traceback
import warnings

PINS = {'front':'df62a817c3e1b540ed62645850278df0463f934c562c1f3f50b85c2c05fc543d',
        'infer':'b69330053ed603530458bb181029b3abb4c83dc57c811e70837c04141293b56a',
        'japanese':'ac58698c2b6a9163dde4ba841b7ef52c4c86dd02dd19b8d3c28a09a42e98c886'}
LANGUAGES = {'english':'en','long-word':'en','japanese':'ja','japanese-date':'ja',
             'japanese-pronunciation':'ja','spanish':'es','arabic':'ar','korean':'ko'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['front','infer','japanese','output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        raise ValueError('this corpus profile fixes the Darwin WeText reference')
    for name, expected in PINS.items():
        if sha(getattr(args,name)) != expected:raise ValueError('fixed source mismatch: '+name)
    for name, version in [('wetext','0.1.0'),('fugashi','1.4.0'),('unidic-lite','1.0.8')]:
        if importlib.metadata.version(name) != version:raise ValueError('fixed package mismatch: '+name)
    cls = next(n for n in ast.parse(args.front.read_bytes()).body if isinstance(n,ast.ClassDef) and n.name=='TextNormalizer')
    scope=dict(re=re,os=os,traceback=traceback,warnings=warnings,lru_cache=lru_cache,__file__=str(args.front))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(args.front),'exec'),scope)
    frontend=scope['TextNormalizer']();frontend.load()
    definitions=[]
    for node in ast.parse(args.infer.read_bytes()).body:
        if isinstance(node,ast.FunctionDef) and node.name in ['is_kana','apply_pronunciation_annotations']:definitions.append(node)
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PRONUNCIATION_ANNOTATION_PATTERN' for t in node.targets):definitions.append(node)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(args.infer),'exec'),scope)
    spec=importlib.util.spec_from_file_location('fixed_matrix_ja',args.japanese);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    japanese=module.JapaneseG2PProcessor(g2p_ratio=0)
    base=Path(__file__).resolve().parents[2]
    spanish_path=base/'crates/indextts/tests/fixtures/spanish-oracle.json'
    spanish=json.loads(spanish_path.read_text());spanish={r['text']:r['output'] for r in spanish['cases']}
    rows=[]
    for case in json.loads(Path(__file__).with_name('text-matrix.json').read_text()):
        text=case['text'];language=LANGUAGES.get(case['id'],'zh')
        row=dict(id=case['id'],text=text,language=language,source_sha256=hashlib.sha256(text.encode()).hexdigest())
        if language=='ko':
            row['expected_rejection']='fixed 2.5 API language set is zh/en/ja/es/ar'
        else:
            output=frontend.clean_pattern.sub(lambda m:frontend.char_rep_map[m.group()],text)
            if language in ['zh','en']:output=frontend.normalize(output)
            elif language=='es':output=spanish[output]
            if language in ['ja','zh','en']:output=output.lower()
            if language=='es':output=output.upper()
            output=scope['apply_pronunciation_annotations'](output)
            if language=='ja':output=japanese.process_ja_text(output)
            row['output']=re.sub(r'<\|([^|]+)\|>',lambda m:f'<|{m.group(1).upper()}|>',output)
        rows.append(row)
    result=dict(upstream_revision='d9e41aac89fd00b3d71497fddb287b7f24613712',source_sha256=PINS,
                platform='Darwin',spanish_fixture_sha256=sha(spanish_path),cases=rows)
    with args.output.open('x',encoding='utf-8') as out:
        json.dump(result,out,ensure_ascii=False,indent=2);out.write('\n')

if __name__=='__main__':main()
