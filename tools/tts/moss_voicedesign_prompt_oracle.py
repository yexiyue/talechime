"""Generate no-reference VoiceGenerator prompt rows from pinned official code.

Requires local official processor/normalizer, model tokenizer and chat template;
no model weights, network or reference audio. Output must not already exist.
"""
import argparse
import ast
import hashlib
import json
import re
import runpy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch
from transformers import AutoTokenizer

REV = '934d6826b084c46a0d033402174d5f8ac4ed2519'
MODEL_REV = '97521ec2b6f3ec5026ac1f5751f8fc302d82c2d4'
PINS = {
    'processor': 'ac0e1bbb6b20e966bd0f4cae395c8160b6c05c7d23183782b2c0da6acc219c1c',
    'normalizer': 'b7c1fab4fd7bdc8c779df99f6af310fd8cd0746b759db684bf147e0c182b3183',
    'chat_template': '894e3ace884e212ae34e1f46063686e7ebcc8b2fa0153af4579dfa5685f63184',
    'tokenizer': 'cb3c8fa82993d515469c2800cc455bff4aaa3c4fed9da1f2b0c0668c304f335a',
}
CASES = [
    ('zh', '你好，欢迎来到这里。', '年轻女性，声音清晰自然'),
    ('mixed', '早上好，Alice说hello。', '温和、平静的旁白'),
    ('spaces', '  Hello   world!  ', 'Clear adult voice'),
    ('punctuation', '这是第一句……\n接着说。', '沉稳自然'),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('processor', 'normalizer', 'model-dir', 'chat-template', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--matrix', type=Path)
    a = p.parse_args()
    paths = {'processor': a.processor, 'normalizer': a.normalizer,
             'chat_template': a.chat_template, 'tokenizer': a.model_dir / 'tokenizer.json'}
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[name]:
            raise ValueError(f'{name} checksum mismatch')
    config = json.loads((a.model_dir / 'config.json').read_text())
    if (config['n_vq'], config['audio_pad_code'], config['audio_start_token_id']) != (16, 1024, 151652):
        raise ValueError('unexpected fixed model configuration')
    tokenizer = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
    tokenizer.chat_template = a.chat_template.read_text()
    normalizer = runpy.run_path(str(a.normalizer))['normalize_tts_text']
    tree = ast.parse(a.processor.read_text())
    names = {'Message', 'UserMessage'}
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MossTTSDelayProcessor')
    needed = {'build_user_message', '_replace_audio_placeholders', '_get_unified_codes'}
    cls.body = [node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in needed]
    cls.bases = []
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *classes, cls], type_ignores=[]))
    namespace = {'dataclass': dataclass, 'List': List, 'torch': torch, 're': re,
                 'AUDIO_PLACEHOLDER': '<|audio|>', 'normalize_tts_text': normalizer}
    exec(compile(module, '<pinned-official-processor>', 'exec'), namespace)
    processor = namespace['MossTTSDelayProcessor']()
    processor.tokenizer = tokenizer
    processor.model_config = SimpleNamespace(**config)
    for field in ('audio_user_slot_token_id', 'audio_assistant_gen_slot_token_id',
                  'audio_start_token_id', 'audio_end_token_id'):
        setattr(processor, field.removesuffix('_id'), tokenizer.convert_ids_to_tokens(config[field]))
    # The reference-free path never uses the assistant delay token, but the
    # official method resolves this attribute before checking the role.
    processor.audio_assistant_delay_slot_token = tokenizer.convert_ids_to_tokens(config['audio_assistant_delay_slot_token_id'])
    cases = []
    selected = [(c['id'], c['text'], '温和、平静的旁白') for c in json.loads(a.matrix.read_text())] if a.matrix else CASES
    for name, text, instruction in selected:
        message = processor.build_user_message(text=text, instruction=instruction)
        chat = tokenizer.apply_chat_template([{'role':'user', 'content':message['content']}],
                                             add_generation_prompt=True, tokenize=False)
        rows = processor._get_unified_codes('user', chat, [], False).tolist()
        case = {'id':name, 'text':text, 'instruction':instruction,
                'normalized_text': normalizer(text), 'chat':chat, 'rows':rows}
        if a.matrix:
            case.update(source_text=text, source_sha256=hashlib.sha256(text.encode()).hexdigest())
        cases.append(case)
    receipt = {'upstream_revision': REV, 'model_revision': MODEL_REV, 'sources': PINS,
               'model_config': {key:config[key] for key in ['n_vq','audio_pad_code','audio_start_token_id','audio_end_token_id','audio_user_slot_token_id']},
               'transformers': __import__('transformers').__version__, 'torch': torch.__version__, 'cases':cases}
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(receipt, f, ensure_ascii=False, separators=(',', ':'))
        f.write('\n')

if __name__ == '__main__':
    main()
