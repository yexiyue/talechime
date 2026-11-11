"""Extract supported single-reference conditioning from the pinned official model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

REVISION = 'f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629'
SHA = '31ad554bc15a4da18cf820c6fda97a5b2c56ac8e9a5e66031a264d363661575c'

class Captured(Exception):
    pass

def export(path):
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SHA:
        raise ValueError('requires the fixed official VoxCPM2 model source')
    tree = ast.parse(source)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'VoxCPM2Model')
    methods = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in ('_make_ref_prefix', '_generate_with_prompt_cache')]
    for method in methods:
        method.decorator_list = []
        method.returns = None
        for arg in (*method.args.args, *method.args.kwonlyargs):
            arg.annotation = None
    cls.bases = []
    cls.body = methods
    namespace = {'torch': torch, 'get_dtype': lambda _: torch.float32,
                 'materialize_generation_seed': lambda seed: seed or 0,
                 'apply_generation_seed': lambda _: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), namespace)
    model = namespace['VoxCPM2Model']()
    model.device = torch.device('cpu')
    model.patch_size = 1
    model.audio_vae = SimpleNamespace(latent_dim=1)
    model.config = SimpleNamespace(dtype='float32')
    model.audio_start_token = 101
    model.ref_audio_start_token = 103
    model.ref_audio_end_token = 104
    model.text_tokenizer = lambda text: [ord(c) for c in text]
    def inference(text, text_mask, feat, feat_mask, **kwargs):
        model.captured = {'tokens': text.flatten().tolist(), 'text_mask': text_mask.flatten().tolist(),
                          'audio': feat.flatten().tolist(), 'audio_mask': feat_mask.flatten().tolist(),
                          'max_len': kwargs['max_len']}
        raise Captured()
    model._inference = inference
    cases = []
    specs = [
        ('natural', 'A', None, None, 0),
        ('style', '(soft)A', None, None, 0),
        ('design', '(warm)Hi', None, None, 0),
        ('clone', 'A', 'T', 'continuation', 3),
        ('clone_style', '(soft)A', 'T', 'continuation', 1),
        ('isolated', 'A', None, 'reference', 3),
    ]
    matrix = json.loads(Path(__file__).with_name('text-matrix.json').read_text(encoding='utf-8'))
    inputs = [(spec, None) for spec in specs] + [(spec, row) for row in matrix for spec in specs]
    for (name, text, transcript, mode, n), row in inputs:
        if row is not None:
            text = ('(soft)' if name in ('style', 'clone_style') else '(warm)' if name == 'design' else '') + row['text']
        features = torch.arange(1, n + 1, dtype=torch.float32).reshape(n, 1, 1)
        cache = None if mode is None else ({'mode': mode, 'audio_feat': features, 'prompt_text': transcript}
                                         if mode == 'continuation' else {'mode': mode, 'ref_audio_feat': features})
        try:
            next(model._generate_with_prompt_cache(text, cache, seed=1))
        except Captured:
            pass
        else:
            raise RuntimeError('official model did not reach inference capture')
        metadata = {} if row is None else {'source_id': row['id'], 'source_text': row['text'],
                    'source_sha256': hashlib.sha256(row['text'].encode('utf-8')).hexdigest()}
        cases.append({**metadata, 'name': name, 'target': text, 'transcript': transcript, 'mode': mode,
                      'frames': n, **model.captured})
    return {'source_revision': REVISION, 'source_sha256': SHA,
            'tokenizer_kind': 'synthetic Unicode scalar IDs and synthetic reference features; input structure only', 'public_corpus': matrix, 'cases': cases}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(export(args.source), output, indent=2, ensure_ascii=False)
        output.write('\n')
