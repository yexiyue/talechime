"""Capture fixed official condition tensors with a synthetic byte tokenizer."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

REVISION = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
MODEL_SHA256 = "631050ba94775b9c8a72ec3b2d84777c38317e2e502b1998d3128fedf88846ff"
DURATION_SHA256 = "3a0b468e89eb2a109a270fe4386ce1afe57305c08375a524a1af08399dfaf772"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="fixed official OmniVoice repository")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source / "omnivoice/models/omnivoice.py"
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("fixed official OmniVoice source mismatch")
    tree = ast.parse(data)
    names = {"_prepare_inference_inputs", "_combine_text", "_tokenize_with_nonverbal_tags", "_estimate_target_tokens"}
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    for function in functions:
        function.decorator_list = []
    pattern = next(node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_NONVERBAL_PATTERN" for t in node.targets))
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, pattern] + functions, type_ignores=[]))
    namespace = {"torch": torch, "re": re}
    exec(compile(module, "<fixed-official-condition>", "exec"), namespace)
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(models.BPE(vocab={ch: i for i, ch in enumerate(alphabet)}, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    class Wrapper:
        def __call__(self, text, return_tensors=None, **kwargs):
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            return SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors else ids)

    owner = SimpleNamespace(text_tokenizer=Wrapper(), config=SimpleNamespace(num_audio_codebook=8, audio_mask_id=1024), device="cpu")
    cases = []
    specs = [(None, None, False, True), ("zh", None, False, True), ("ja", None, False, True),
             (None, "male, low pitch", False, True), ("zh", "male, low pitch", False, True),
             (None, None, True, True), ("en", None, True, False)]
    matrix = json.loads(Path(__file__).with_name("text-matrix.json").read_text(encoding="utf-8"))
    for (language, instruction, clone, denoise), row in ([(spec, None) for spec in specs] + [(spec, row) for row in matrix for spec in specs]):
        text = "Alice says [laughter] hello. 2026 year." if row is None else row["text"]
        reference = [[row + i for i in range(3)] for row in range(8)] if clone else None
        ref_text = "Hello, welcome.\n" if clone else None
        result = namespace["_prepare_inference_inputs"](owner, text, 5, ref_text,
            torch.tensor(reference) if clone else None, language, instruction, denoise)
        metadata = {} if row is None else {"source_id": row["id"], "source_text": text,
            "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        cases.append({**metadata, "text": text, "language": language, "instruction": instruction,
                      "reference": reference, "ref_text": ref_text, "denoise": denoise,
                      "target_length": 5, "input_ids": result["input_ids"].tolist(),
                      "audio_mask": result["audio_mask"].tolist()})
    duration_source = args.source / "omnivoice/utils/duration.py"
    if hashlib.sha256(duration_source.read_bytes()).hexdigest() != DURATION_SHA256:
        raise ValueError("fixed official duration source mismatch")
    spec = importlib.util.spec_from_file_location("fixed_omni_duration", duration_source)
    duration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(duration)
    owner.duration_estimator = duration.RuleDurationEstimator()
    duration_texts = [row["text"] for row in json.loads(Path(__file__).with_name("text-matrix.json").read_text())]
    durations = []
    for index, text in enumerate(duration_texts):
        for ref_text, ref_tokens in [(None, None), ("Hello, welcome.", 50)]:
            for speed in [0.5, 1.0, 2.0]:
                expected = namespace["_estimate_target_tokens"](owner, text, ref_text, ref_tokens, speed)
                durations.append({"text_index": index, "ref_text": ref_text,
                                  "ref_tokens": ref_tokens, "speed": speed, "expected": expected})
    receipt = {"source": "k2-fsa/OmniVoice", "revision": REVISION, "source_sha256": MODEL_SHA256,
               "tokenizer": json.loads(tokenizer.to_str()), "tokenizer_kind": "synthetic byte BPE; no checkpoint numeric claim",
               "torch": torch.__version__, "public_corpus": matrix, "cases": cases, "duration_source_sha256": DURATION_SHA256,
               "duration_texts": duration_texts, "durations": durations}
    args.output.write_text(json.dumps(receipt, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
