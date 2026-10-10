"""Produce synthetic turn rows using the pinned official formatter, without weights."""
import ast
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

source_root = Path(sys.argv[1])
assert subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip() == '934d6826b084c46a0d033402174d5f8ac4ed2519'
source = source_root / 'moss_tts_realtime/mossttsrealtime/processing_mossttsrealtime.py'
module = ast.parse(source.read_text())
processor = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'MossTTSRealtimeProcessor')
method = next(n for n in processor.body if isinstance(n, ast.FunctionDef) and n.name == 'make_user_prompt')
method.returns = None
for arg in method.args.args:
    arg.annotation = None
namespace = {'np': np}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)

class Tokenizer:
    def encode(self, text):
        if text == '<|im_end|>\n<|im_start|>user\n':
            return [1, 2]
        if text == '<|im_end|>\n<|im_start|>assistant\n':
            return [3, 4]
        prefix = '<|im_end|>\n<|im_start|>user\n'
        ids = []
        if text.startswith(prefix):
            ids = [1, 2]
            text = text[len(prefix):]
        spoken, _, pads = text.partition('<|text_pad|>')
        ids.extend(range(100, 100 + len(spoken)))
        if pads or '<|text_pad|>' in text:
            ids.extend([9] * (1 + pads.count('<|text_pad|>')))
        return ids

    def __call__(self, text):
        return {'input_ids': self.encode(text)}

class Processor:
    tokenizer = Tokenizer()
    channels = 16
    delay_tokens_len = 12
    audio_channel_pad = 1024
    audio_bos_token = 1025
    audio_eos_token = 1026

    def _normalize_audio_tokens(self, codes):
        return np.array(codes)

cases = []
for length in [3, 12, 16]:
    codes = [[42] * 16] * 20
    rows = namespace['make_user_prompt'](Processor(), 'x' * length, codes)
    cases.append({'text': list(range(100, 100 + length)), 'codes': codes, 'rows': rows.tolist()})
Path(sys.argv[2]).write_text(json.dumps(cases, separators=(',', ':')) + '\n')
