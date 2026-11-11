"""Export synthetic prefill from fixed official Python, without models or downloads."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

REV = '022e286b98fbec7e1e916cb940cdf532cd9f488e'
SHA = '25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0'

class Capture(Exception):
    pass

class FakeTalker:
    device = torch.device('cpu')
    dtype = torch.float32
    def __init__(self):
        self.code_predictor = SimpleNamespace(get_input_embeddings=lambda: [self.codec] * 15)
    def text(self, ids):
        return ids.float().unsqueeze(-1) / 100000.0
    def codec(self, ids):
        return ids.float().unsqueeze(-1) / 1000.0
    def get_text_embeddings(self): return self.text
    def get_input_embeddings(self): return self.codec
    def text_projection(self, values): return torch.nn.functional.silu(values)
    def generate(self, **kwargs):
        self.result = {k: kwargs[k].squeeze(0).flatten().tolist() for k in ('inputs_embeds', 'trailing_text_hidden')}
        raise Capture()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', type=Path, help="local fixed Base directory for CPU embedding comparison")
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SHA:
        raise ValueError('official model source SHA mismatch')
    tree = ast.parse(source)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Qwen3TTSForConditionalGeneration')
    methods = [x for x in cls.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name in ('generate', 'generate_icl_prompt')]
    for method in methods:
        method.decorator_list = []
        method.returns = None
        for arg in method.args.args: arg.annotation = None
    cls.body = methods
    cls.bases = []
    scope = {'torch': torch, 'Optional': object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(args.source), 'exec'), scope)
    obj = scope[cls.name]()
    talker_config = SimpleNamespace(vocab_size=3072, codec_eos_token_id=2150, codec_language_id={'chinese':2055, 'english':2050, 'japanese':2058, 'sichuan':2062, 'beijing':2074}, spk_id={'eric':2875, 'dylan':2878, 'vivian':3065}, spk_is_dialect={'eric':'sichuan', 'dylan':'beijing', 'vivian':False}, codec_nothink_id=2155, codec_think_bos_id=2156, codec_think_eos_id=2157, codec_think_id=2154, codec_pad_id=2148, codec_bos_id=2149, num_code_groups=16)
    obj.config = SimpleNamespace(talker_config=talker_config, tts_bos_token_id=151672, tts_eos_token_id=151673, tts_pad_token_id=151671)
    obj.talker = FakeTalker()
    obj.talker.config = talker_config
    if args.model:
        from safetensors import safe_open
        path = args.model / 'model.safetensors'
        with path.open('rb') as source_file:
            model_sha = hashlib.file_digest(source_file, 'sha256').hexdigest()
        if model_sha != '38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c':
            raise ValueError('fixed Base model SHA mismatch')
        with safe_open(path, framework='pt', device='cpu') as weights:
            names = ['talker.model.text_embedding.weight', 'talker.model.codec_embedding.weight',
                     'talker.text_projection.linear_fc1.weight', 'talker.text_projection.linear_fc1.bias',
                     'talker.text_projection.linear_fc2.weight', 'talker.text_projection.linear_fc2.bias']
            names += [f'talker.code_predictor.model.codec_embedding.{i}.weight' for i in range(15)]
            tensors = {name: weights.get_tensor(name).float() for name in names}
        def embed(name):
            return lambda ids: torch.nn.functional.embedding(ids, tensors[name])
        obj.talker.text = embed('talker.model.text_embedding.weight')
        obj.talker.codec = embed('talker.model.codec_embedding.weight')
        obj.talker.code_predictor = SimpleNamespace(get_input_embeddings=lambda: [embed(f'talker.code_predictor.model.codec_embedding.{i}.weight') for i in range(15)])
        def project(x):
            x = torch.nn.functional.linear(x, tensors[names[2]], tensors[names[3]])
            return torch.nn.functional.linear(torch.nn.functional.silu(x), tensors[names[4]], tensors[names[5]])
        obj.talker.text_projection = project
    obj.generate_speaker_prompt = lambda prompt: prompt['ref_spk_embedding']
    cases = []
    specs = [('custom','auto','eric',0),('custom','chinese','dylan',0),('custom','auto','dylan',0),('custom','chinese','eric',0),('custom','japanese','eric',0),('custom','auto','vivian',0),('design','auto',None,0),('design','english',None,0),('base','auto',None,2),('base','japanese',None,8),('base','chinese',None,5)]
    matrix = json.loads(Path(__file__).with_name('text-matrix.json').read_text(encoding='utf-8'))
    inputs = [(spec, None) for spec in specs]
    if not args.model:
        inputs += [(spec, row) for row in matrix for spec in specs]
    for (kind, lang, speaker, ref_len), row in inputs:
        if args.model and kind != 'base':
            continue
        target = [100,101,102] if row is None else [100 + b for b in row['text'].encode('utf-8')]
        ids = torch.tensor([[151644,77091,198,*target,151645,198,151644,77091,198]])
        kwargs = dict(input_ids=[ids], languages=[lang], speakers=[speaker], instruct_ids=[torch.tensor([[22,23]])] if kind == 'design' else None)
        if kind == 'base':
            ref = torch.tensor([[151644,77091,198,31,32,151645,198]])
            kwargs['ref_ids'] = [ref]
            kwargs['voice_clone_prompt'] = dict(ref_spk_embedding=[torch.full((2048 if args.model else 1,), 0.75)], x_vector_only_mode=[False], icl_mode=[True], ref_code=[torch.ones((ref_len,16), dtype=torch.long)])
        try: obj.generate(**kwargs)
        except Capture: pass
        metadata = {} if row is None else dict(source_id=row['id'], source_text=row['text'], source_sha256=hashlib.sha256(row['text'].encode('utf-8')).hexdigest())
        cases.append(dict(kind=kind,language=lang,speaker=speaker,ref_len=ref_len, **metadata, **obj.talker.result))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f: json.dump(dict(official_revision=REV, source_sha256=SHA, tokenizer_kind='synthetic UTF-8 bytes + 100 for public corpus; synthetic embeddings; input structure only' if not args.model else 'fixed token IDs with checkpoint embeddings', model_revision='fd4b254389122332181a7c3db7f27e918eec64e3' if args.model else None, model_sha256=model_sha if args.model else None, public_corpus=matrix if not args.model else [], cases=cases), f, separators=(',', ':'), ensure_ascii=False); f.write('\n')

if __name__ == '__main__': main()
