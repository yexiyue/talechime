"""Official pinned MOSS Local batch API; xtask owns repetition and containment."""
import contextlib
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import time


def checkpoint_parameter_bytes(directory, bytes_per_element):
    """Header-only lower bound for these pinned, parameter-only checkpoints."""
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())) if index.exists() else ["model.safetensors"]
    names = set()
    elements = 0
    for shard in shards:
        path = (directory / shard).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("checkpoint shard escapes model directory")
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 100_000_000 or length > path.stat().st_size - 8:
                raise ValueError("invalid safetensors header length")
            header = json.loads(stream.read(length))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in names:
                raise ValueError(f"duplicate checkpoint tensor: {name}")
            names.add(name)
            elements += math.prod(tensor["shape"])
    return elements * bytes_per_element


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2]).resolve()
    config = request["request"]
    sys.path.insert(0, config["checkout"])
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import soundfile as sf
        import torch
        import transformers
        import safetensors
        from functools import partial
        from unittest.mock import patch
        from transformers import AutoTokenizer
        from moss_tts_local.configuration_moss_tts import MossTTSDelayConfig
        from moss_tts_local.modeling_moss_tts import MossTTSDelayModel
        from moss_tts_local.processing_moss_tts import MossTTSDelayProcessor
        name = "moss_reference_codec"
        source = Path(config["codec_checkout"])
        spec = importlib.util.spec_from_file_location(name, source / "__init__.py", submodule_search_locations=[str(source)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        spec.loader.exec_module(package)
        codec_module = importlib.import_module(name + ".modeling_moss_audio_tokenizer")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    clock = time.perf_counter()
    sequence = 0

    def event(kind, payload):
        nonlocal sequence
        print(json.dumps(dict(schema="talechime/performance/1", attempt_id=request["attempt_id"], sequence=sequence, offset_seconds=time.perf_counter()-clock, kind=kind, payload=payload)), flush=True)
        sequence += 1

    started = time.perf_counter()
    # The pinned official processor leaves its F32 codec unchanged. Do not
    # silently substitute F16/autocast to make the resident-GPU baseline fit.
    capacity = dict(model_bf16_parameter_bytes=checkpoint_parameter_bytes(config["model"], 2),
                    codec_f32_parameter_bytes=checkpoint_parameter_bytes(config["codec"], 4),
                    dedicated_gpu_bytes=torch.cuda.get_device_properties(0).total_memory)
    capacity["parameter_lower_bound_bytes"] = capacity["model_bf16_parameter_bytes"] + capacity["codec_f32_parameter_bytes"]
    capacity["scope"] = "pinned parameter-only checkpoints; resident dedicated VRAM policy, excluding activations and CUDA context"
    (output / "capacity-preflight.json").write_text(json.dumps(capacity, indent=2), encoding="utf-8")
    if capacity["parameter_lower_bound_bytes"] > capacity["dedicated_gpu_bytes"]:
        raise RuntimeError("official BF16 main + F32 codec parameters exceed dedicated VRAM; resident-GPU baseline refused before weight loading (shared-memory/offload not evaluated)")
    # On Windows, mapping every shard can exhaust commit before inference.
    # Use the library's own positional reader only during checkpoint loading.
    with contextlib.redirect_stdout(sys.stderr), patch(
        "transformers.modeling_utils.safe_open",
        partial(safetensors.safe_open, backend="pread"),
    ):
        model_config = MossTTSDelayConfig.from_pretrained(config["model"], local_files_only=True)
        model = MossTTSDelayModel.from_pretrained(config["model"], config=model_config, dtype=torch.bfloat16,
                                                 device_map="cuda:0", attn_implementation="sdpa", local_files_only=True).eval()
        codec = codec_module.MossAudioTokenizerModel.from_pretrained(config["codec"], dtype=torch.float32,
                                                                     device_map="cuda:0", local_files_only=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
        if config.get("chat_template"):
            template = Path(config["chat_template"]).read_text(encoding="utf-8")
            assert tokenizer.chat_template in (None, template), "conflicting tokenizer template"
            tokenizer.chat_template = template
        processor = MossTTSDelayProcessor(tokenizer=tokenizer, audio_tokenizer=codec, model_config=model.config)
    torch.cuda.synchronize()
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__, safetensors=safetensors.__version__,
                   checkpoint_backend="pread", load_seconds=time.perf_counter()-started,
                   model_source=sys.modules[MossTTSDelayModel.__module__].__file__, codec_source=codec_module.__file__,
                   model_dtype=str(next(model.parameters()).dtype), codec_dtype=str(next(codec.parameters()).dtype),
                   attention=model.config._attn_implementation, load_mode="from_pretrained direct CUDA device_map",
                   delivery="official batch generate followed by processor.decode; no audio streaming API used",
                   recommended_torch="2.9.1+cu128", torch_deviation=torch.__version__ != "2.9.1+cu128")
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    event("prepared", runtime)
    text = Path(config["text_file"]).read_text(encoding="utf-8").strip()
    for index in range(request["warmup_rounds"] + request["hot_rounds"]):
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            batch = processor([[processor.build_user_message(text=text)]], mode="generation")
            generated = model.generate(input_ids=batch["input_ids"].to("cuda"), attention_mask=batch["attention_mask"].to("cuda"),
                                       max_new_tokens=375, n_vq_for_inference=32,
                                       text_temperature=1.5, text_top_k=50, text_top_p=1.0, text_repetition_penalty=1.0,
                                       audio_temperature=1.0, audio_top_k=50, audio_top_p=0.95, audio_repetition_penalty=1.1)
            assert len(generated) == 1 and int(generated[0][1][-1, 0]) == model.config.audio_end_token_id, "frame limit reached or missing EOS"
            messages = processor.decode(generated)
            assert len(messages) == 1 and messages[0] is not None
            assert len(messages[0].audio_codes_list) == 1
            pcm = messages[0].audio_codes_list[0].detach().float().cpu().numpy().reshape(-1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert pcm.size > 0 and np.isfinite(pcm).all()
        name = f"round-{index}-f32.wav"
        sf.write(output / name, pcm, model.config.sampling_rate, subtype="FLOAT")
        path = output / name
        artifact = dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size, kind="audio_f32", derived_from=None)
        event("round", dict(index=index, warmup=index < request["warmup_rounds"], outcome="success", synthesis_seconds=elapsed,
                            audio_seconds=pcm.size/model.config.sampling_rate, first_pcm_seconds=elapsed,
                            delivery_boundary="segment_delivery_first_pcm", output_sha256=hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(), observations=[], artifacts=[artifact]))
    event("terminal", dict(outcome="success"))


if __name__ == "__main__":
    main()
