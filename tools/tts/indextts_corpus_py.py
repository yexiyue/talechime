"""Development-only Python corpus runner for IndexTTS 2.5 A/B comparison.

Mirrors `crates/indextts/examples/corpus.rs`: synthesizes every non-empty
line of a UTF-8 corpus with one fixed reference, writing per-segment WAVs
and index.json with timings/audio stats. Seeds are NOT transferable between
torch and the Rust RNG; compare aggregate stats and listen, not waveforms.

Usage:
    D:/workspace/index-tts/.venv/Scripts/python.exe tools/tts/indextts_corpus_py.py \
        --checkpoints D:/workspace/index-tts/checkpoints \
        --reference D:/workspace/index-tts/examples/voice_01.wav \
        --corpus tools/tts/acceptance-corpus.txt \
        --out-dir D:/workspace/index-tts/runs/py-corpus --rounds 2 --seed 7
"""
import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--no-text-normalization", action="store_true")
    args = p.parse_args()
    import sys
    sys.path.insert(0, str(args.checkpoints.parent))

    segments = [l.strip() for l in args.corpus.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    assert segments, "corpus is empty"
    assert args.rounds > 0, "rounds must be positive"

    from indextts.infer_v2_5 import IndexTTS2
    t0 = time.perf_counter()
    tts = IndexTTS2(cfg_path=str(args.checkpoints / "config.yaml"),
                    model_dir=str(args.checkpoints), use_bf16=False,
                    device=args.device, use_cuda_kernel=False)
    load_secs = time.perf_counter() - t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    total_audio = total_gen = failed_gen = 0.0
    failures = 0
    for rnd in range(args.rounds):
        for i, text in enumerate(segments):
            seed = None if args.seed is None else args.seed + rnd * 1000 + i
            if seed is not None:
                torch.manual_seed(seed)
            t = time.perf_counter()
            try:
                tts.infer(spk_audio_prompt=str(args.reference), text=text, lang="ZH",
                          text_normalization=not args.no_text_normalization,
                          output_path=str(args.out_dir / f"round{rnd:02}-seg{i:02}.wav"))
            except Exception as exc:
                gen = time.perf_counter() - t
                failures += 1
                failed_gen += gen
                entries.append({"round": rnd, "segment": i, "text": text,
                                "seed": seed, "error": str(exc), "generate_secs": gen})
                print(f"round {rnd} seg {i}: FAILED: {exc}", flush=True)
                continue
            gen = time.perf_counter() - t
            with wave.open(str(args.out_dir / f"round{rnd:02}-seg{i:02}.wav")) as w:
                n, sr = w.getnframes(), w.getframerate()
                d = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32767
            audio = n / sr
            rms = float(np.sqrt((d ** 2).mean()))
            zc = float((np.diff(np.sign(d)) != 0).mean())
            entries.append({"round": rnd, "segment": i,
                            "wav": f"round{rnd:02}-seg{i:02}.wav",
                            "chars": len(text), "audio_secs": audio,
                            "text": text, "seed": seed,
                            "generate_secs": gen, "rtf": gen / max(audio, 1e-6),
                            "rms": rms, "zero_crossing_rate": zc})
            total_audio += audio
            total_gen += gen
            print(f"round {rnd} seg {i}: {audio:.2f}s audio, rtf {gen / max(audio, 1e-6):.2f}")

    index = {"load_secs": load_secs, "seed": args.seed, "device": args.device,
             "text_normalization": not args.no_text_normalization,
             "rounds": entries,
             "summary": {"total_audio_secs": total_audio,
                         "total_generate_secs": total_gen,
                         "failed_generate_secs": failed_gen,
                         "attempts": len(segments) * args.rounds,
                         "failures": failures,
                         "successes": len(segments) * args.rounds - failures,
                         "overall_rtf": total_gen / max(total_audio, 1e-6)}}
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2),
                                              encoding="utf-8")
    print("wrote", args.out_dir / "index.json")


if __name__ == "__main__":
    main()
