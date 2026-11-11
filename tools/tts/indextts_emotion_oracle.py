"""Extract unchanged QwenEmotion conversion from the fixed official source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

SOURCE_SHA256 = 'b69330053ed603530458bb181029b3abb4c83dc57c811e70837c04141293b56a'
CASES = [
    ('{"高兴": 2, "愤怒": -1}', '今天很好'),
    ('{"emotion": "angry"}', 'ordinary text'),
    ('"sad"', 'ordinary text'),
    ('{"悲伤": 0.8, "低落": 0.2}', 'I feel gloomy'),
    ('{"高兴": "sad"}', 'ordinary text'),
    ('高兴: 0.3, 自然: 0.4', 'ordinary text'),
    ('{"自然": 0}', 'ordinary text'),
    ('{}', 'ordinary text'),
    ('[1, 2]', 'ordinary text'),
    ('{"label": "反感"}', 'ordinary text'),
    ('{"情绪": "surprised"}', 'ordinary text'),
    ('{"高兴": true}', 'ordinary text'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('official inference source identity mismatch')
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == 'QwenEmotion')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    # Only remove model/tokenizer initialization. Classification/conversion stays unchanged.
    init.body = [node for node in init.body if not isinstance(node, ast.Assign) or
                 not any(isinstance(target, ast.Attribute) and target.attr in {'model', 'tokenizer'} for target in node.targets)]
    inference = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'inference')
    start = next(index for index, node in enumerate(inference.body) if isinstance(node, ast.Try) and
                 isinstance(node.body[0], ast.Assign) and isinstance(node.body[0].value, ast.Call) and
                 isinstance(node.body[0].value.func, ast.Attribute) and node.body[0].value.func.attr == 'loads')
    inference.body = inference.body[start:]
    inference.args.args.append(ast.arg(arg='content'))
    tree = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = dict(json=json, re=re)
    exec(compile(tree, str(args.source), 'exec'), scope)
    classifier = scope['QwenEmotion']('unused')
    cases = []
    for content, text in CASES:
        values = classifier.inference(text, content)
        cases.append(dict(content=content, text=text, vector=[float(value) for value in values.values()]))
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(dict(source_sha256=SOURCE_SHA256, cases=cases), output, ensure_ascii=False, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
