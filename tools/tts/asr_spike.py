"""Development-only ASR comparison; never a production content/quality verdict.

Requires numpy, scipy, soundfile, sherpa-onnx,
torch and transformers. Audio, models and raw reports belong under target/.
Reference text is deliberately never supplied as an ASR prompt/hotword.
"""

import argparse
import difflib
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import time
import unicodedata
from pathlib import Path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def normalize(text):
    # Keep number/script/homophone differences visible; no speculative equivalence.
    return "".join(
        c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum()
    )


def differences(source, transcript):
    left, right = normalize(source), normalize(transcript)
    return [
        {
            "operation": op,
            "source": left[a:b],
            "transcript": right[c:d],
            "normalized_source_range": [a, b],
        }
        for op, a, b, c, d in difflib.SequenceMatcher(
            None, left, right, autojunk=False
        ).get_opcodes()
        if op != "equal"
    ]


def corpus(args):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    records = []
    for index in args.index:
        items = json.loads(index.read_text(encoding="utf-8"))
        for position, item in enumerate(items[: args.per_index]):
            samples, rate = sf.read(
                index.parent / item["file"], dtype="float32", always_2d=True
            )
            samples = samples.mean(axis=1)
            divisor = math.gcd(rate, 16000)
            samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype(
                "float32"
            )
            identity = f"{index.parent.name}-{position + 1:03}"
            variants = {"original": samples}
            if position == 1:
                size = len(samples)
                variants.update(
                    {
                        "tail_cut": samples[: int(size * 0.65)],
                        "middle_removed": np.concatenate(
                            [samples[: int(size * 0.3)], samples[int(size * 0.6) :]]
                        ),
                        "repeated": np.concatenate([samples, samples]),
                        "silence": np.zeros_like(samples),
                        "noise": np.random.default_rng(42)
                        .normal(0, 0.08, size)
                        .astype("float32"),
                    }
                )
            for kind, audio in variants.items():
                path = args.output / f"{identity}-{kind}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, audio, 16000, subtype="PCM_16")
                records.append(
                    {
                        "id": f"{identity}-{kind}",
                        "file": str(path.resolve()),
                        "text": item["text"],
                        "kind": kind,
                        "source_index": str(index.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "duration_s": len(audio) / 16000,
                    }
                )
    save(args.output / "corpus.json", records)
    print(f"Prepared {len(records)} cases", flush=True)


def run(args):
    import numpy as np
    import soundfile as sf

    records = json.loads(args.corpus.read_text(encoding="utf-8"))
    model_path = args.model_path
    began = time.perf_counter()
    if args.backend == "qwen":
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        torch.set_num_threads(args.threads)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = (
            AutoModelForMultimodalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.float32 if args.device == "cpu" else torch.float16,
                attn_implementation="eager",
            )
            .to(args.device)
            .eval()
        )

        def recognize(audio):
            inputs = processor.apply_transcription_request(
                audio=audio, language="Chinese"
            )
            inputs = inputs.to(model.device, model.dtype)
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=256)
            ids = output[:, inputs["input_ids"].shape[1] :]
            transcript = processor.decode(ids, return_format="transcription_only")[0]
            return transcript, {
                "output_tokens": ids.shape[1],
                "token_limit_reached": ids.shape[1] >= 256,
            }
    elif args.backend == "whisper":
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        torch.set_num_threads(args.threads)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = (
            AutoModelForSpeechSeq2Seq.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.float32 if args.device == "cpu" else torch.float16,
                attn_implementation="eager",
            )
            .to(args.device)
            .eval()
        )

        def recognize(audio):
            if len(audio) > 30 * 16000:
                raise ValueError(
                    "Whisper spike uses single segments of at most 30 seconds"
                )
            inputs = processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = inputs.to(model.device, model.dtype)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    language="zh",
                    task="transcribe",
                    do_sample=False,
                    max_new_tokens=256,
                    return_timestamps=False,
                    return_dict_in_generate=False,
                    force_unique_generate_call=True,
                )
            # Whisper's plain-tensor return strips EOS; retain raw sequences for
            # completion checks using one raw generation call. Avoid the MPS
            # return-dict cache reassembly path in Transformers 5.19.0.
            transcript = processor.batch_decode(output, skip_special_tokens=True)[0]
            return transcript, {
                "output_tokens": output.shape[1],
                "token_limit_reached": output[0, -1].item()
                != model.generation_config.eos_token_id,
            }
    elif args.backend.endswith("-ort"):
        from asr_onnx_spike import recognizer

        recognize = recognizer(model_path, args.backend, args.threads)
    else:
        import sherpa_onnx

        factory = (
            sherpa_onnx.OfflineRecognizer.from_sense_voice
            if args.backend == "sensevoice"
            else sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc
        )
        options = {
            "model": str(model_path / "model.int8.onnx"),
            "tokens": str(model_path / "tokens.txt"),
            "num_threads": args.threads,
            "provider": "cpu",
        }
        if args.backend == "sensevoice":
            options.update(language="zh", use_itn=False)
        model = factory(**options)

        def recognize(audio):
            stream = model.create_stream()
            stream.accept_waveform(16000, audio)
            model.decode_stream(stream)
            return stream.result.text, {}

    load_seconds = time.perf_counter() - began
    results = []
    packages = {}
    for name in ["torch", "transformers", "sherpa-onnx", "numpy", "soundfile"]:
        packages[name] = importlib.metadata.version(name)
    report = {
        "backend": args.backend,
        "device": args.device,
        "threads": args.threads,
        "model_path": str(model_path.resolve()),
        "load_seconds": load_seconds,
        "platform": platform.platform(),
        "packages": packages,
        "normalization": "NFKC + casefold + alphanumeric; numbers/scripts/homophones retained",
        "results": results,
    }
    # First case is measured cold, then repeated once for a consistent warmup.
    print(f"Loaded {args.backend} in {load_seconds:.2f}s", flush=True)
    for item in records:
        audio, rate = sf.read(item["file"], dtype="float32")
        if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("corpus must contain finite mono 16kHz audio")
        began = time.perf_counter()
        try:
            transcript, extra = recognize(audio)
            elapsed = time.perf_counter() - began
            result = {
                **item,
                "transcript": transcript,
                "elapsed_s": elapsed,
                "rtf": elapsed / item["duration_s"],
                "differences": differences(item["text"], transcript),
                **extra,
            }
        # Candidate-specific exceptions are evidence, not a successful empty transcript.
        except Exception as error:  # noqa: BLE001
            result = {
                **item,
                "error": f"{type(error).__name__}: {error}",
                "elapsed_s": time.perf_counter() - began,
            }
        results.append(result)
        # ru_maxrss is bytes on macOS, KiB on Linux; includes model load and inference.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_rss_bytes"] = (
            peak if platform.system() == "Darwin" else peak * 1024
        )
        save(args.output, report)
        print(
            json.dumps(
                {k: result.get(k) for k in ["id", "transcript", "elapsed_s", "error"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
        if len(results) == 1 and "error" not in result:
            recognize(audio)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("corpus")
    prepare.add_argument("--index", type=Path, action="append", required=True)
    prepare.add_argument("--per-index", type=int, default=8)
    prepare.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument(
        "--backend",
        choices=[
            "qwen",
            "whisper",
            "sensevoice",
            "firered",
            "sensevoice-ort",
            "firered-ort",
        ],
        required=True,
    )
    execute.add_argument("--model-path", type=Path, required=True)
    execute.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    execute.add_argument("--threads", type=int, default=4)
    execute.add_argument("--corpus", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.command == "run"
        and args.backend not in {"qwen", "whisper"}
        and args.device != "cpu"
    ):
        parser.error("ONNX reference adapters in this spike support CPU only")
    (corpus if args.command == "corpus" else run)(args)


if __name__ == "__main__":
    main()
