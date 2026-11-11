"""Explicit real-model oracle using the unchanged fixed QwenEmotion class."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != 'b69330053ed603530458bb181029b3abb4c83dc57c811e70837c04141293b56a':
        raise ValueError('fixed official inference source identity mismatch')
    for package, version in [('transformers', '4.52.1'), ('torch', '2.7.1'), ('accelerate', '1.7.0')]:
        if importlib.metadata.version(package) != version:
            raise ValueError(f'{package} must be {version}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == 'QwenEmotion')
    scope = dict(AutoTokenizer=AutoTokenizer, AutoModelForCausalLM=AutoModelForCausalLM, time=time, json=json, re=re)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(args.source), 'exec'), scope)
    classifier = scope['QwenEmotion'](str(args.model.resolve(strict=True)))
    original_generate = classifier.model.generate
    captured = {}
    def generate(*call_args, **kwargs):
        generated = original_generate(*call_args, **kwargs)
        captured['input_ids'] = kwargs['input_ids'][0].tolist()
        captured['output_ids'] = generated[0][len(captured['input_ids']):].tolist()
        return generated
    classifier.model.generate = generate
    rows = []
    for text in ['太好了，我们终于成功了！', '听到这个消息，我非常愤怒。', '我今天很低落，什么也不想做。', 'Hello, welcome to our story today.']:
        torch.manual_seed(42)
        started = time.monotonic()
        vector = classifier.inference(text)
        rows.append(dict(text=text, vector=[float(value) for value in vector.values()],
                         seconds=time.monotonic()-started, **captured))
    result = dict(source_sha256=hashlib.sha256(source).hexdigest(),
                  dtype=str(next(classifier.model.parameters()).dtype), device=str(classifier.model.device),
                  cases=rows)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
