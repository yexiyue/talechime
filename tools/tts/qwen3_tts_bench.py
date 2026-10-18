#!/usr/bin/env python3
"""Explicit-policy benchmark for official Qwen3-TTS Base ICL synthesis."""
import argparse
import hashlib
import inspect
import json
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers

from qwen_tts import Qwen3TTSModel
from transformers import RepetitionPenaltyLogitsProcessor


@contextmanager
def repetition_penalty_policy(policy):
    original = RepetitionPenaltyLogitsProcessor.__call__

    def guarded(processor, input_ids, scores):
        # Empty history must be the identity. Torch 2.7.1 MPS scatter can return
        # recycled buffer contents for this case on the local validation host.
        if input_ids.numel() == 0:
            return scores
        return original(processor, input_ids, scores)

    if policy == "guard":
        RepetitionPenaltyLogitsProcessor.__call__ = guarded
    try:
        yield
    finally:
        RepetitionPenaltyLogitsProcessor.__call__ = original


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def capture_talker_sequences(talker):
    original = talker.generate
    sequences = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        # Retain the device tensor; inspect termination only after timing.
        sequences.append(result.sequences)
        return result

    talker.generate = capture
    try:
        yield sequences
    finally:
        talker.generate = original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reference_text", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--empty_repetition", choices=("guard", "original"), default="guard",
                    help="guard skips the invalid MPS empty-history scatter; original reproduces historical runs")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--export_codes", action="store_true", help="save codec inputs after timing for cross-decoder diagnosis")
    ap.add_argument(
        "--policy", choices=("matched", "official-default", "greedy"), default="matched",
        help="matched uses Rust sampling options; official-default resolves checkpoint defaults; greedy disables semantic/acoustic sampling",
    )
    ap.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="model compute dtype; MPS may reject bfloat16 for some operators",
    )
    args = ap.parse_args()
    if args.rounds < 0 or (args.threads is not None and args.threads <= 0):
        ap.error("--rounds must be non-negative and --threads must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens <= 0:
        ap.error("--max_new_tokens must be positive")

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = args.device
    if selected == "auto":
        selected = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(selected)
    source = Path(inspect.getfile(Qwen3TTSModel)).resolve().parents[2]
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    source_diff = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"])
    (out_dir / "official-source.patch").write_bytes(source_diff)
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        args.model_dir,
        device_map=device,
        dtype=getattr(torch, args.dtype),
        attn_implementation="sdpa",
    )
    sync(device)
    load_s = time.perf_counter() - t0
    dtype_status = "loaded"

    t0 = time.perf_counter()
    prompt = model.create_voice_clone_prompt(
        args.reference,
        ref_text=args.reference_text,
        x_vector_only_mode=False,
    )
    sync(device)
    prompt_s = time.perf_counter() - t0

    common = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=False,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
    )
    if args.policy == "official-default":
        common = model._merge_generate_kwargs()
    elif args.policy == "greedy":
        common["do_sample"] = False
    if args.max_new_tokens is not None:
        common["max_new_tokens"] = args.max_new_tokens

    captured_codes = []
    if args.export_codes:
        original_decode = model.model.speech_tokenizer.decode

        def capture_decode(encoded):
            captured_codes.extend(item["audio_codes"] for item in encoded)
            return original_decode(encoded)

        model.model.speech_tokenizer.decode = capture_decode

    records = []
    rounds = args.rounds + int(args.warmup)
    with (
        repetition_penalty_policy(args.empty_repetition),
        capture_talker_sequences(model.model.talker) as sequences,
    ):
        for i in range(rounds):
            captured_codes.clear()
            sequences.clear()
            torch.manual_seed(args.seed)
            sync(device)
            t0 = time.perf_counter()
            wavs, sr = model.generate_voice_clone(
                args.text,
                language="Chinese",
                voice_clone_prompt=prompt,
                non_streaming_mode=False,
                **common,
            )
            sync(device)
            elapsed = time.perf_counter() - t0
            if len(sequences) != 1 or sequences[0].shape[0] != 1:
                raise ValueError("expected one generated talker sequence")
            tokens = sequences[0][0].detach().cpu().tolist()
            eos_token = common.get("eos_token_id")
            if eos_token is None:
                eos_token = model.model.config.talker_config.codec_eos_token_id
            eos_ids = eos_token if isinstance(eos_token, (list, tuple)) else [eos_token]
            eos = bool(tokens) and tokens[-1] in eos_ids
            wav = wavs[0]
            if len(wav) == 0 or not np.isfinite(wav).all():
                raise ValueError("synthesis produced empty or non-finite audio")
            duration = len(wav) / sr
            rec = {
                "round": i,
                "warmup": args.warmup and i == 0,
                "seconds": elapsed,
                "audio_seconds": duration,
                "rtf": elapsed / duration,
                "samples": len(wav),
                "sample_rate": sr,
                "finite": True,
                "eos": eos,
                "termination": "eos" if eos else "incomplete",
                "generated_tokens": len(tokens),
            }
            records.append(rec)
            pcm = np.asarray(wav, dtype="<f4")
            rec["pcm_sha256"] = hashlib.sha256(pcm.tobytes()).hexdigest()
            pcm.tofile(out_dir / f"round-{i}.f32le")
            sf.write(out_dir / f"round-{i}.wav", wav, sr)
            if args.export_codes:
                payload = {
                    "reference_frames": int(prompt[0].ref_code.shape[0]),
                    "codes": [codes.detach().cpu().tolist() for codes in captured_codes],
                }
                (out_dir / f"decode-codes-{i}.json").write_text(json.dumps(payload) + "\n")
            print(json.dumps(rec), flush=True)
            dtype_status = "generated-finite"

    measured = [r for r in records if not r["warmup"] and r["eos"]]
    summary = {
        "implementation": "official-qwen3-tts",
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "revision": revision,
        "source_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
        "source_dirty": bool(source_diff),
        "model_dir": str(Path(args.model_dir).resolve()),
        "device": str(device),
        "reference_prompt_patch": "see source_diff_sha256; local model preprocessor configuration recorded separately",
        "preprocessor_sha256": hashlib.sha256((Path(args.model_dir) / "speech_tokenizer/preprocessor_config.json").read_bytes()).hexdigest(),
        "dtype": args.dtype,
        "dtype_status": dtype_status,
        "threads": args.threads or torch.get_num_threads(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "generation": common,
        "policy": args.policy,
        "seed": args.seed,
        "reference_text": args.reference_text,
        "export_codes": args.export_codes,
        "empty_repetition": args.empty_repetition,
        "warmup_requested": args.warmup,
        "text": args.text,
        "reference_sha256": hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
        "load_seconds": load_s,
        "prompt_seconds": prompt_s,
        "rounds": records,
        "completed_measured_rounds": len(measured),
        "median_seconds": statistics.median(r["seconds"] for r in measured) if measured else None,
        "median_audio_seconds": statistics.median(r["audio_seconds"] for r in measured) if measured else None,
        "median_rtf": statistics.median(r["rtf"] for r in measured) if measured else None,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
