"""Capture fixed official Nano voice-clone prompt rows for raw public text.

Requires PyTorch and sentencepiece, fixed official source files and tokenizer;
no weights, model generation or product text normalization is used.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import sentencepiece as spm
import torch

REVISION = '44502f80dbf9743528fa921cc544d662c685ebec'
PINS = {'prompting': '24c1675e08071fc4b70fbc8f531c5f073234b029c55f3f6a8590891c2cb5f583',
        'modeling': 'b68ffef1173eb39625dc9321f4ca4bfb7644dff54463e08df348e045b8cb5e04',
        'tokenizer': 'c353ee1479b536bf414c1b247f5542b6607fb8ae91320e5af1781fee200fddff'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prompting', 'modeling', 'tokenizer', 'matrix', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for name, pin in PINS.items():
        if hashlib.sha256(getattr(args, name).read_bytes()).hexdigest() != pin:
            raise ValueError(f'{name} does not match fixed Nano revision')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.parse(args.prompting.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))]
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
                 '<official-nano-prompting>', 'exec'), namespace)
    tree = ast.parse(args.modeling.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MossTTSNanoForCausalLM')
    needed = {'_build_text_rows', '_encode_text', '_resolve_inference_mode',
              '_build_audio_prefix_rows', 'build_inference_input_ids'}
    model.body = [node for node in model.body if isinstance(node, ast.FunctionDef) and node.name in needed]
    if {node.name for node in model.body} != needed:
        raise ValueError('official Nano model methods changed')
    model.bases = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, model], type_ignores=[])),
                 '<official-nano-modeling>', 'exec'), namespace)
    model = namespace['MossTTSNanoForCausalLM']()
    model._resolve_device = lambda device: torch.device('cpu')
    model.config = SimpleNamespace(n_vq=16, audio_pad_token_id=1024, im_start_token_id=4,
                                   im_end_token_id=5, audio_start_token_id=6, audio_end_token_id=7,
                                   audio_user_slot_token_id=8, audio_assistant_slot_token_id=9)
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    codes = [[42] * 16, [42] * 16]
    cases = []
    for item in json.loads(args.matrix.read_text()):
        text = item['text']
        rows, mask = model.build_inference_input_ids(text, tokenizer, mode='voice_clone',
                                                     prompt_audio_codes=torch.tensor(codes), device='cpu')
        cases.append({'id': item['id'], 'source_text': text, 'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'input_kind': 'raw public source; no product normalization or segmentation',
                      'text_ids': model._encode_text(tokenizer, text), 'reference_codes': codes,
                      'rows': rows[0].tolist(), 'attention_mask': mask[0].tolist()})
    receipt = {'revision': REVISION, 'sources_sha256': PINS, 'sentencepiece': spm.__version__,
               'torch': torch.__version__, 'cases': cases}
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(receipt, output, ensure_ascii=False, separators=(',', ':'))
        output.write('\n')


if __name__ == '__main__':
    main()
