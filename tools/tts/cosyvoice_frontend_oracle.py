"""Development oracle from fixed CosyVoice source and actual model tokenizer."""
import argparse
import ast
from functools import partial
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import re
from typing import Callable, Generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frontend', type=Path)
    parser.add_argument('utils', type=Path)
    parser.add_argument('tokenizer', type=Path)
    parser.add_argument('cases', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    for name, expected in [('frontend','8d3dcb5880d76346f053f19344eac48adcff2aca70687293aadf4e1f33fadac3'),
                           ('utils','0d0ca94e0cf519e2d17f72cd1dcab6cb07aea0de48e7e792826362284ea18d30'),
                           ('tokenizer','caa0dadcb8c40c79a3244aaf290492e298a49c5c25e0db16cad12f2f8ff639c5')]:
        if hashlib.sha256(getattr(args,name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'fixed {name} identity mismatch')
    for name, version in [('wetext','0.1.0'),('inflect','7.5.0'),('regex','2026.1.15'),('tokenizers','0.22.2')]:
        if importlib.metadata.version(name)!=version: raise ValueError(f'{name} must be {version}')
    import inflect
    import regex
    import tokenizers
    import wetext
    scope = dict(re=re, regex=regex, Generator=Generator, Callable=Callable, logging=logging, partial=partial)
    names = {'contains_chinese', 'replace_blank', 'replace_corner_mark', 'remove_bracket',
             'spell_out_number', 'split_paragraph', 'is_only_punctuation'}
    definitions = [node for node in ast.parse(args.utils.read_bytes()).body
                   if isinstance(node, ast.FunctionDef) and node.name in names or
                   isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'chinese_char_pattern' for target in node.targets)]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(args.utils), 'exec'), scope)
    cls = next(node for node in ast.parse(args.frontend.read_bytes()).body if isinstance(node, ast.ClassDef) and node.name == 'CosyVoiceFrontEnd')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(args.frontend), 'exec'), scope)
    frontend = scope['CosyVoiceFrontEnd'].__new__(scope['CosyVoiceFrontEnd'])
    frontend.text_frontend = 'wetext'
    frontend.zh_tn_model = wetext.Normalizer(remove_erhua=False)
    frontend.en_tn_model = wetext.Normalizer()
    frontend.inflect_parser = inflect.engine()
    tokenizer = tokenizers.Tokenizer.from_file(str(args.tokenizer))
    class Tokenizer:
        @staticmethod
        def encode(text, allowed_special):
            return tokenizer.encode(text, add_special_tokens=False).ids
    frontend.tokenizer = Tokenizer()
    frontend.allowed_special = 'all'
    cases = json.loads(args.cases.read_text())
    texts = [row['text'] for row in cases]
    texts += ['甲'*65+'。'+'乙'*65+'。短句。', '甲\n乙  ABC DEF 【测试】² - 丙，',
              '<|laughter|>hello 2026', 'Hello 007 and １００１.', '🙂', '']
    rows = []
    for text in texts:
        try:
            rows.append(dict(text=text, output=frontend.text_normalize(text)))
        except Exception as error:
            rows.append(dict(text=text, error=type(error).__name__))
    result = dict(revision='074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc',
                  source_sha256={name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
                                 for name in ['frontend', 'utils', 'tokenizer']},
                  packages={name: importlib.metadata.version(name) for name in ['wetext', 'inflect', 'regex', 'tokenizers']},
                  cases=rows)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
