"""Offline PyTorch replay of Rust-captured OmniVoice CPU tensors.

Run with the pinned official checkout's Python, PYTHONPATH and local weights.
No downloads, compilation, or changes to generation parameters.
"""
import argparse
import hashlib
import inspect
import json
import pathlib
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import HiggsAudioV2TokenizerModel, Qwen3Config
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)


def emit(value):
    print("CPU_TRACE " + json.dumps(value), flush=True)


def bench(case, op, run):
    run()
    for group in range(5):
        start = time.perf_counter()
        for _ in range(5):
            result = run()
        emit(dict(case=case, op=op, group=group,
                  microseconds=(time.perf_counter() - start) * 1e6 / 5))
    return result


def read(root, name):
    """Restore original strides and offset, rather than timing canonicalized inputs."""
    tensors = load_file(str(root / (name + ".safetensors")))
    metadata = json.loads((root / (name + ".layout.json")).read_text())
    restored = {}
    for key, tensor in tensors.items():
        m = metadata[key]
        shape, strides, offset = m["shape"], m["strides"], m["offset"]
        size = offset + 1 + sum((d - 1) * s for d, s in zip(shape, strides))
        value = torch.empty(size, dtype=tensor.dtype).as_strided(shape, strides, offset)
        value.copy_(tensor)
        restored[key] = value
    return restored


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def stage0(args):
    config = Qwen3Config(**json.loads((args.model / "config.json").read_text())["llm_config"])
    config._attn_implementation = "sdpa"
    rotary = Qwen3RotaryEmbedding(config)
    with safe_open(str(args.model / "model.safetensors"), framework="pt") as weights:
        for layer_index in (0, 13, 27):
            layer = Qwen3DecoderLayer(config, layer_index).float().eval()
            prefix = f"llm.layers.{layer_index}."
            layer.load_state_dict({k.removeprefix(prefix): weights.get_tensor(k)
                                   for k in weights.keys() if k.startswith(prefix)})
            attn = layer.self_attn
            for step in (0, 15, 31):
                name = f"step-{step}-layer-{layer_index}"
                original, t = read(args.root, name), read(args.root, name + "-replay")
                x, mask = original["input"], original["mask"]
                b, s, _ = x.shape
                position = torch.arange(s).unsqueeze(0)
                embedding = rotary(x, position)
                def norm_rope():
                    q = attn.q_norm(t["q_raw"].reshape(b, s, 16, 128)).transpose(1, 2)
                    k = attn.k_norm(t["k_raw"].reshape(b, s, 8, 128)).transpose(1, 2)
                    return apply_rotary_pos_emb(q, k, *embedding)
                def attention():
                    return sdpa_attention_forward(attn, t["q"], t["k"], t["v"], mask,
                                                  scaling=attn.scaling, is_causal=False)[0]
                ops = {
                    "layer": lambda: layer(x, attention_mask=mask, position_embeddings=embedding),
                    "input_norm": lambda: layer.input_layernorm(x),
                    "q_proj": lambda: attn.q_proj(t["norm"]),
                    "k_proj": lambda: attn.k_proj(t["norm"]),
                    "v_proj": lambda: attn.v_proj(t["norm"]),
                    "norm_rope": norm_rope,
                    # Official attention includes repeat_kv and its sequence-first output copy.
                    "attention": attention,
                    "attention_consumer": lambda: attn(t["norm"], embedding, mask)[0],
                    "context_layout": lambda: t["context"].transpose(1, 2).contiguous().reshape(b, s, 2048),
                    "mlp": lambda: layer.mlp(t["mlp_input"]),
                    "mlp_gate_up": lambda: (layer.mlp.gate_proj(t["mlp_input"]), layer.mlp.up_proj(t["mlp_input"])),
                    "mlp_activation": lambda: torch.nn.functional.silu(t["gate"]) * t["up"],
                    "down_proj": lambda: layer.mlp.down_proj(t["down_input"]),
                    "mlp_consumer": lambda: layer.mlp(t["mlp_input"]) + t["residual"],
                }
                output = ops["layer"]()
                delta = output - original["output"]
                emit(dict(kind="cross_framework_error", case=name,
                          input_sha256=digest(x), output_sha256=digest(output),
                          max_abs=delta.abs().max().item(), rmse=delta.square().mean().sqrt().item()))
                items = list(ops.items())
                for op, run in reversed(items) if args.reverse else items:
                    bench(name, op, run)
                if layer_index == 0 and step == 0:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                        ops["layer"]()
                    emit(dict(kind="kernel_profile", case=name, operators=[
                        dict(name=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total)
                        for e in profile.key_averages()]))
        head = weights.get_tensor("audio_heads.weight")
        for step in (0, 15, 31):
            name = f"step-{step}-head"
            x = read(args.root, name)["input"]
            bench(name, "output_head", lambda: torch.nn.functional.linear(x, head))


def stage1(args):
    start = time.perf_counter()
    model = HiggsAudioV2TokenizerModel.from_pretrained(
        str(args.model / "audio_tokenizer"), local_files_only=True, dtype=torch.float32).eval()
    emit(dict(kind="load", seconds=time.perf_counter()-start))
    for path in sorted((args.root / "codes").glob("*.json")):
        codes = json.loads(path.read_text())
        t = torch.tensor(codes["values"], dtype=torch.int64).reshape(1, codes["codebooks"], codes["frames"])
        first = time.perf_counter()
        output = model.decode(t).audio_values
        emit(dict(case=path.stem, op="first_decode", microseconds=(time.perf_counter()-first)*1e6,
                  samples=output.numel(), output_sha256=digest(output)))
        for group in range(5):
            start = time.perf_counter()
            output = model.decode(t).audio_values
            emit(dict(case=path.stem, op="decode", group=group,
                      microseconds=(time.perf_counter()-start)*1e6, samples=output.numel()))
    if args.whole_only:
        return
    for path in sorted(args.root.glob("decoder-frames-130-*.safetensors")):
        name = path.stem
        if name.endswith("-raw"):
            continue
        suffix = name.removeprefix("decoder-frames-130-")
        # Rust's checkpoint names map to HF DAC's named modules.
        parts = suffix.split("-")
        if parts[0] == "block":
            module = model.acoustic_decoder.block[int(parts[1])]
            suffix = "-".join(parts[2:])
            if suffix.startswith("res_unit"):
                subparts = suffix.split("-")
                module = getattr(module, subparts[0])
                if len(subparts) == 2:
                    module = getattr(module, subparts[1])
            elif suffix:
                module = getattr(module, suffix)
        else:
            module = getattr(model.acoustic_decoder, suffix)
        x = read(args.root, name)["input"]
        if not suffix:
            op = "block"
        elif suffix.startswith("res_unit") and "-" not in suffix:
            op = "residual_consumer"
        else:
            op = "deconv" if "conv_t1" in name else "snake" if "snake1" in name else "conv7" if name.endswith("conv1") else "conv1"
        bench(name, op, lambda: module(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stage0", "stage1"])
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--whole-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    source = pathlib.Path(inspect.getfile(Qwen3DecoderLayer))
    emit(dict(kind="runtime", torch=torch.__version__, threads=torch.get_num_threads(),
              interop_threads=torch.get_num_interop_threads(), config=torch.__config__.show(),
              qwen3_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    with torch.inference_mode():
        (stage0 if args.mode == "stage0" else stage1)(args)


if __name__ == "__main__":
    main()
