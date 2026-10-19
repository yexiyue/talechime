#!/usr/bin/env python3
"""Benchmark the pinned official VoxCPM source on macOS MPS.

This is a development probe. It deliberately uses the official public API and
does not import Talechime model code. Segment text comes from the Candle probe's
production-preprocessed ``segments.json`` so framework comparisons do not also
compare different text splitting rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

from voxcpm_perf_common import hot_rounds


SAMPLE_RATE = 48_000
SOURCE_REVISION = "f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629"
MODEL_REVISION = "32279effe8c19989596f05d353d1447f51d9e915"


def git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def sync(torch) -> None:
    torch.mps.synchronize()


def model_devices(model) -> dict:
    params = {str(parameter.device) for parameter in model.parameters()}
    buffers = {str(buffer.device) for buffer in model.buffers()}
    param_dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    buffer_dtypes = {str(buffer.dtype) for buffer in model.buffers()}
    return {"parameters": sorted(params), "buffers": sorted(buffers),
            "parameter_dtypes": sorted(param_dtypes), "buffer_dtypes": sorted(buffer_dtypes)}


def compiled_components(model) -> list[str]:
    names = []
    for name, function in {
        "base_lm.forward_step": model.base_lm.forward_step,
        "residual_lm.forward_step": model.residual_lm.forward_step,
        "feat_encoder": model.feat_encoder,
        "feat_decoder.estimator": model.feat_decoder.estimator,
    }.items():
        if hasattr(function, "_torchdynamo_orig_callable") or hasattr(function, "_orig_mod"):
            names.append(name)
    return names


def tensor_devices(value, torch) -> set[str]:
    """Collect devices recursively without retaining generated tensors."""
    if isinstance(value, torch.Tensor):
        return {str(value.device)}
    if isinstance(value, (list, tuple)):
        return set().union(*(tensor_devices(item, torch) for item in value))
    if isinstance(value, dict):
        return set().union(*(tensor_devices(item, torch) for item in value.values()))
    return set()


def install_device_audit(model, torch) -> dict[str, set[str]]:
    """Observe real generation inputs at the MPS execution boundaries."""
    observed: dict[str, set[str]] = {}

    def record(name, args, kwargs=None):
        observed.setdefault(name, set()).update(tensor_devices(args, torch))
        if kwargs:
            observed[name].update(tensor_devices(kwargs, torch))

    def hook(name):
        return lambda _module, args, kwargs: record(name, args, kwargs)

    model.base_lm.register_forward_pre_hook(hook("base_lm.prefill"), with_kwargs=True)
    model.residual_lm.register_forward_pre_hook(hook("residual_lm.prefill"), with_kwargs=True)
    model.feat_encoder.register_forward_pre_hook(hook("feat_encoder"), with_kwargs=True)
    model.feat_decoder.estimator.register_forward_pre_hook(hook("feat_decoder.estimator"), with_kwargs=True)

    for name, module in (("base_lm.step", model.base_lm), ("residual_lm.step", model.residual_lm),
                         ("audio_vae.decode", model.audio_vae)):
        original = getattr(module, "forward_step" if name.endswith(".step") else "decode")

        def observed_call(*args, _name=name, _original=original, **kwargs):
            record(_name, args, kwargs)
            return _original(*args, **kwargs)

        setattr(module, "forward_step" if name.endswith(".step") else "decode", observed_call)
    return observed


def load_segments(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or any(
        not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip()
        for item in data
    ):
        raise ValueError(f"invalid production segment manifest: {path}")
    return data


def save_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError(f"invalid PCM for {path.name}")
    sf.write(path, pcm, SAMPLE_RATE, subtype="FLOAT")


def generate_round(torch, model, segments: list[dict], reference: Path, transcript: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    elapsed = 0.0
    count = 0
    index = []
    for position, segment in enumerate(segments, start=1):
        # Match Candle's transcribed continuation, without an isolated identity
        # prefix or retries that could change the requested seed.
        sync(torch)
        started = time.monotonic()
        wav = model.generate(
            text=segment["text"],
            prompt_wav_path=str(reference),
            prompt_text=transcript,
            cfg_value=2.0,
            inference_timesteps=10,
            max_len=200,
            retry_badcase=False,
            seed=42,
        )
        sync(torch)
        elapsed += time.monotonic() - started
        pcm = np.asarray(wav, dtype=np.float32).reshape(-1)
        filename = f"{position:03}.wav"
        save_wav(directory / filename, pcm)
        count += len(pcm)
        index.append({"file": filename, "text": segment["text"], "samples": len(pcm),
                      "termination": "not_exposed", "api_returned": True,
                      "actual_seed": getattr(model.tts_model, "last_successful_seed", None)})
    seconds = count / SAMPLE_RATE
    if seconds <= 0:
        raise ValueError("official API produced no audio")
    (directory / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"generate_ms": round(elapsed * 1000, 3), "audio_seconds": seconds,
            "rtf": elapsed / seconds, "segments": len(segments), "first_pcm_ms": None,
            "termination": "not_exposed", "api_returned": True}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "optimize"), default="eager")
    parser.add_argument("--rounds", type=hot_rounds, default=1)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--audit-devices", action="store_true",
                        help="record real generation tensor devices; do not use for throughput")
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 3 or not 0 <= args.warmup_rounds <= args.rounds:
        parser.error("rounds must be 1..3 and warmup-rounds must not exceed rounds")
    return args


def main() -> int:
    args = parse_args()
    for path in (args.source, args.models, args.segments, args.reference, args.transcript):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    if git_revision(args.source) != SOURCE_REVISION:
        raise SystemExit(f"official source must be pinned to {SOURCE_REVISION}")
    source_dirty = subprocess.check_output(
        ["git", "-C", str(args.source), "status", "--porcelain"], text=True,
    ).strip()
    if source_dirty:
        raise SystemExit("official source checkout must be clean")
    manifest = json.loads((args.models / "source-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("revision") != MODEL_REVISION:
        raise SystemExit(f"official model must be pinned to {MODEL_REVISION}")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise SystemExit("disable PYTORCH_ENABLE_MPS_FALLBACK for MPS validation")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    dependencies = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    (args.output / "requirements-snapshot.txt").write_text("\n".join(dependencies) + "\n", encoding="utf-8")

    sys.path.insert(0, str(args.source / "src"))
    import torch
    from voxcpm import VoxCPM

    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS is unavailable")
    segments = load_segments(args.segments)
    transcript = args.transcript.read_text(encoding="utf-8").strip()
    if not transcript:
        raise SystemExit("reference transcript is empty")

    started = time.monotonic()
    model = VoxCPM.from_pretrained(
        str(args.models), load_denoiser=False, local_files_only=True,
        optimize=args.mode == "optimize", device="mps",
    )
    sync(torch)
    load_ms = round((time.monotonic() - started) * 1000, 3)
    devices = model_devices(model.tts_model)
    if model.tts_model.sample_rate != SAMPLE_RATE:
        raise RuntimeError(f"unexpected official sample rate: {model.tts_model.sample_rate}")
    if any(not device.startswith("mps") for device in devices["parameters"]) or any(
        not (device.startswith("mps") or device == "cpu") for device in devices["buffers"]
    ):
        raise RuntimeError(f"official model is not resident on MPS: {devices}")
    compiled = compiled_components(model.tts_model)
    optimize_effective = args.mode == "optimize" and len(compiled) == 4
    observed_devices = install_device_audit(model.tts_model, torch) if args.audit_devices else None
    metadata = {
        "mode": args.mode,
        "optimize_effective": optimize_effective,
        "compiled_components": compiled,
        "source_revision": git_revision(args.source),
        "source_dirty": bool(source_dirty),
        "model_revision": manifest["revision"],
        "workspace_revision": git_revision(Path(__file__).resolve().parents[2]),
        "workspace_dirty": bool(subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "status", "--porcelain"], text=True,
        ).strip()),
        "torch": torch.__version__,
        "device": "mps",
        "configured_dtype": str(model.tts_model.config.dtype),
        "checkpoint_dtype": json.loads((args.models / "config.json").read_text(encoding="utf-8"))["dtype"],
        "devices": devices,
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "sample_rate": SAMPLE_RATE,
        "generation_options": {"seed": 42, "cfg": 2.0, "cfm_steps": 10, "max_len": 200,
                               "retry_badcase": False, "conditioning": "continuation",
                               "temperature": "official_default_1.0_not_exposed"},
        "reference_prepare_ms": None,
        "timing_boundary": "public_generate_including_reference_excluding_wav_io",
        "segments_sha256": hashlib.sha256(args.segments.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "transcript_sha256": hashlib.sha256(args.transcript.read_bytes()).hexdigest(),
        "load_ms": load_ms,
        "device_audit": args.audit_devices,
    }
    if args.mode == "optimize" and not optimize_effective:
        metadata.update({"available": False, "reason": "official optimize() only compiles CUDA paths"})
        (args.output / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False))
        return 0

    results = []
    for round_index in range(args.warmup_rounds + args.rounds):
        torch.manual_seed(42)
        directory = args.output / f"round-{round_index}"
        result = generate_round(torch, model, segments, args.reference, transcript, directory)
        result.update({"round": round_index, "warmup": round_index < args.warmup_rounds})
        print(json.dumps(result), flush=True)
        results.append(result)
    measured = [row["rtf"] for row in results if not row["warmup"]]
    metadata.update({
        "available": True,
        "output_checks": "passed",
        "eos_checks": "not_exposed_by_public_api",
        "warmup_rounds": args.warmup_rounds,
        "first_pcm": "not_exposed_by_non_streaming_public_api",
        "rounds": results,
        "measured": {"rounds": len(measured), "rtf_mean": sum(measured) / len(measured),
                     "rtf_min": min(measured), "rtf_max": max(measured)},
    })
    required_observations = {
        "base_lm.prefill", "base_lm.step", "residual_lm.prefill", "residual_lm.step",
        "feat_encoder", "feat_decoder.estimator", "audio_vae.decode",
    }
    if observed_devices is not None:
        missing = sorted(required_observations - observed_devices.keys())
        if missing or any(devices != {"mps:0"} for devices in observed_devices.values()):
            raise RuntimeError(f"generation did not stay on MPS: missing={missing}, observed={observed_devices}")
        metadata["generated_tensor_devices"] = {
            name: sorted(devices) for name, devices in sorted(observed_devices.items())
        }
    (args.output / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
