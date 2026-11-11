"""Run fixed official processor/prefill AST with a real tokenizer, no weights."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

REVISION = "934d6826b084c46a0d033402174d5f8ac4ed2519"
SOURCES = {
    "processing_mossttsrealtime.py": "54f5c284b344ef9aa59bacb5737f7947c7924836cbd49d9082906b96ecc95241",
    "streaming_mossttsrealtime.py": "04e894d3588b907a5d2eda16607033cc575e641a81b3d9d771b454b343eec63d",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="directory with fixed processor and streaming files")
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--matrix", type=Path)
    args = parser.parse_args()
    sources = {name: (args.source / name).read_bytes() for name in
               ["processing_mossttsrealtime.py", "streaming_mossttsrealtime.py"]}
    for name, data in sources.items():
        if hashlib.sha256(data).hexdigest() != SOURCES[name]:
            raise ValueError(f"fixed official source identity mismatch: {name}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    class Wrapper:
        unk_token_id = None

        def encode(self, text, **kwargs):
            return tokenizer.encode(text, add_special_tokens=False).ids

        def __call__(self, text):
            return {"input_ids": self.encode(text)}

        def convert_tokens_to_ids(self, text):
            return tokenizer.token_to_id(text)

    class Base:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    namespace = {"np": np, "ProcessorMixin": Base}
    tree = ast.parse(sources["processing_mossttsrealtime.py"])
    processor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MossTTSRealtimeProcessor")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, processor], type_ignores=[]))
    exec(compile(module, "<fixed-official-processor>", "exec"), namespace)
    processor = namespace["MossTTSRealtimeProcessor"](Wrapper())

    tree = ast.parse(sources["streaming_mossttsrealtime.py"])
    prefill = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "prefill")
    prefill.decorator_list = []
    # Capture the exact official model input before padding/inference.
    end = next(i for i, n in enumerate(prefill.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "attention_masks" for t in n.targets))
    prefill.body = prefill.body[:end] + [ast.Return(value=ast.Name(id="concat_inputs_id_list", ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[future, prefill], type_ignores=[]))
    exec(compile(module, "<fixed-official-prefill-capture>", "exec"), namespace)

    class Inference:
        device = "cpu"
        channels = 16
        audio_channel_pad = 1024
        audio_bos_token = 1025

        def _normalize_input_ids(self, value):
            return value

        def _normalize_text_prefix(self, value, batch_size):
            return value

    cases = []
    for reference in [None, [[i % 31] * 16 for i in range(20)]]:
        codes = None if reference is None else np.array(reference)
        prefix = processor.tts_system_prompt
        if reference is not None:
            prefix += processor.make_voice_clone_prompt(len(reference))
        encodings = {value: Wrapper().encode(value) for value in [prefix, "<|im_start|>assistant\n"]}
        ensemble = processor.make_ensemble(codes)
        header = np.full((len(encodings["<|im_start|>assistant\n"]), 17), 1024, dtype=np.int64)
        header[:, 0] = encodings["<|im_start|>assistant\n"]
        initial = np.concatenate([ensemble, header])
        selected = json.loads(args.matrix.read_text()) if args.matrix else [None] * 3
        for index, item in enumerate(selected):
            length = [3, 12, 16][index] if item is None else 0
            text = list(range(100, 100 + length)) if item is None else Wrapper().encode(item["text"])
            length = len(text)
            rows = namespace["prefill"](Inference(), [initial], [text], max_prefill_len=12)[0]
            case = {"reference": reference, "text_ids": text, "consumed": min(length, 12),
                    "encodings": encodings, "rows": rows.tolist()}
            if item is not None:
                case.update(id=item["id"], source_text=item["text"], source_sha256=hashlib.sha256(item["text"].encode()).hexdigest())
            cases.append(case)
    receipt = {"source": "OpenMOSS/MOSS-TTS", "revision": REVISION,
               "sources": {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
               "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(), "cases": cases}
    args.output.write_text(json.dumps(receipt, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
