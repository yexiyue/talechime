"""Offline official OmniVoice CPU product baseline.

Timing boundary mirrors the Talechime production request: model load is
reported separately, WAV export is outside the timed region, and every timed
region is bracketed by CPU completion.
"""

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np
import soundfile as sf
import torch

from omnivoice.models.omnivoice import OmniVoice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--position-temperature", type=float, default=5.0)
    parser.add_argument("--class-temperature", type=float, default=0.0)
    parser.add_argument("--t-shift", type=float, default=0.1)
    parser.add_argument("--layer-penalty-factor", type=float, default=5.0)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    device = "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    torch.set_num_interop_threads(1)

    text = open(args.text_file, encoding="utf-8").read().strip()
    dtype = getattr(torch, args.dtype)

    torch.manual_seed(42)
    if hasattr(torch, "mps"):
        torch.mps.manual_seed(42)

    started = time.perf_counter()
    model = OmniVoice.from_pretrained(args.model, device_map=device, dtype=dtype, local_files_only=True)
    if device == "mps":
        torch.mps.synchronize()
    load_seconds = time.perf_counter() - started

    runtime = {"attention":model.llm.config._attn_implementation,
        "llm_dtype":str(next(model.llm.parameters()).dtype),
        "llm_device":str(next(model.llm.parameters()).device),
        "tokenizer_dtype":str(next(model.audio_tokenizer.parameters()).dtype),
        "tokenizer_device":str(next(model.audio_tokenizer.parameters()).device),
        "cpu_threads":torch.get_num_threads()}
    print(json.dumps({"runtime":runtime}),flush=True)
    parameters = {p.device.type for p in model.parameters()}
    buffers = {b.device.type for b in model.buffers()}

    rounds = []
    for index in range(args.warmup + args.rounds):
        if device == "mps":
            torch.mps.synchronize()
        began = time.perf_counter()
        audios = model.generate(
            text=text,
            language=args.language,
            num_step=args.num_step,
            guidance_scale=args.guidance_scale,
            t_shift=args.t_shift,
            layer_penalty_factor=args.layer_penalty_factor,
            position_temperature=args.position_temperature,
            class_temperature=args.class_temperature,
        )
        if device == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - began
        audio = audios[0]
        audio_seconds = len(audio) / float(model.sampling_rate)
        row = {
            "index": index,
            "warmup": index < args.warmup,
            "generation_seconds": elapsed,
            "audio_seconds": audio_seconds,
            "rtf": elapsed / audio_seconds,
            "peak": float(np.abs(audio).max()),
            "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
        }
        rounds.append(row)
        print(json.dumps(row), flush=True)

    hot = [row for row in rounds if not row["warmup"]]
    summary = {
        "schema": "talechime/official-omnivoice-cpu/1",
        "source": {
            "repo": "https://github.com/k2-fsa/OmniVoice",
            "revision": os.environ.get("OMNIVOICE_OFFICIAL_REVISION", "unknown"),
            "model_path": args.model,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
            "dtype": args.dtype,
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "parameters_on": sorted(parameters),
            "buffers_on": sorted(buffers),
        },
        "conditions": {
            "text_file": args.text_file,
            "text": text,
            "language": args.language,
            "num_step": args.num_step,
            "guidance_scale": args.guidance_scale,
            "position_temperature": args.position_temperature,
            "class_temperature": args.class_temperature,
            "t_shift": args.t_shift,
            "layer_penalty_factor": args.layer_penalty_factor,
            "seed": 42,
            "warmup_rounds": args.warmup,
            "hot_rounds": args.rounds,
        },
        "load_seconds": load_seconds,
        "rounds": rounds,
        "hot_summary": {
            "generation_seconds_mean": statistics.mean(r["generation_seconds"] for r in hot),
            "audio_seconds_mean": statistics.mean(r["audio_seconds"] for r in hot),
            "rtf_mean": statistics.mean(r["rtf"] for r in hot),
            "rtf_min": min(r["rtf"] for r in hot),
            "rtf_max": max(r["rtf"] for r in hot),
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    sf.write(args.wav, audios[0], model.sampling_rate)
    print(json.dumps(summary["hot_summary"], indent=2))


if __name__ == "__main__":
    main()
