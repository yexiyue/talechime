#!/usr/bin/env python3
"""Convert Fun-CosyVoice3-0.5B PyTorch weights to safetensors for crates/cosyvoice3.

Derived from SpenserCai/cosyvoice3.rs scripts/convert_weights.py (Apache-2.0),
with config values pinned to the official cosyvoice3.yaml
(docs/records/cosyvoice3/cosyvoice3-candle-audit-2026-10-17.md).

Usage:
    python cosyvoice3_convert_weights.py --input <Fun-CosyVoice3-0.5B-2512> --output <dir> --official_repo <CosyVoice>
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def fuse_weight_norm(state_dict: dict) -> dict:
    """weight = g * v / ||v|| for torch.nn.utils.parametrizations.weight_norm.

    Must use torch's fused `aten::weight_norm` kernel: a manual
    `g * v / linalg.norm(v)` rounds differently (~2e-6 on this checkpoint),
    which the f64 F0 predictor then amplifies into audible PCM drift.

    The official inference first casts the f0 predictor to float64, so its
    parametrization recomputes the fusion in f64 — those tensors are fused
    in f64 and stored as f64. The rest of hift runs in f32 with the runtime
    f32 fusion, which is what gets stored for every other module.
    """
    new_dict, processed = {}, set()
    for key, value in state_dict.items():
        if ".parametrizations.weight.original0" in key:
            base = key.replace(".parametrizations.weight.original0", "")
            v_key = key.replace("original0", "original1")
            if v_key not in state_dict:
                raise KeyError(f"missing weight-norm tensor: {v_key}")
            g, v = value, state_dict[v_key]
            if v.ndim < 2:
                raise ValueError(f"unexpected weight-norm shape for {base}")
            if base.startswith("f0_predictor."):
                fused = torch._weight_norm(v.double(), g.double(), 0)
            else:
                fused = torch._weight_norm(v, g, 0).to(torch.float32)
            new_dict[base + ".weight"] = fused.contiguous()
            processed.update({key, v_key})
        elif key not in processed and ".parametrizations." not in key:
            new_dict[key] = value
    return new_dict


def convert_llm(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        new = key
        if new.startswith("llm.model.model."):
            new = "llm." + new[len("llm.model.model."):]
        elif new.startswith("llm.model."):
            new = "llm." + new[len("llm.model."):]
        out[new] = value.to(torch.float32).clone().contiguous()
    return out


def convert_flow(state: dict) -> dict:
    return {
        key.replace("decoder.estimator.", "dit."): value.to(torch.float32).contiguous()
        for key, value in state.items()
    }


def convert_hift(state: dict) -> dict:
    return {key: value for key, value in fuse_weight_norm(state).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--official_repo", required=True, help="Pinned CosyVoice checkout for the exact special-token list")
    args = parser.parse_args()
    src, dst = Path(args.input), Path(args.output)
    import sys
    sys.path.insert(0, str(Path(args.official_repo).resolve()))
    for f in ("llm.pt", "flow.pt", "hift.pt"):
        if not (src / f).exists():
            raise FileNotFoundError(src / f)
    dst.mkdir(parents=True, exist_ok=True)

    for name, convert in (("llm", convert_llm), ("flow", convert_flow), ("hift", convert_hift)):
        state = torch.load(src / f"{name}.pt", map_location="cpu", weights_only=True)
        save_file(convert(state), dst / f"{name}.safetensors")
        print(f"{name}.safetensors written")

    for f in ("campplus.onnx", "speech_tokenizer_v3.onnx"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
            print(f"{f} copied")

    # Export the complete fast tokenizer graph, including byte-level
    # pretokenization, decoder and CosyVoice3 phoneme/special tokens.
    from cosyvoice.tokenizer.tokenizer import CosyVoice3Tokenizer
    tokenizer = CosyVoice3Tokenizer(str(src / "CosyVoice-BlankEN")).tokenizer
    assert tokenizer.convert_tokens_to_ids("<|endofprompt|>") == 151646
    tokenizer.save_pretrained(dst / "tokenizer")
    print("tokenizer graph exported")

    # Reference noise so Rust runs can pin the exact CFM initial z; the
    # Rust side reproduces it via its MT19937 implementation, and this file
    # lets a third party verify that equivalence.
    g = torch.Generator(device="cpu").manual_seed(0)
    save_file({"rand_noise": torch.randn([1, 80, 50 * 300], generator=g)}, dst / "rand_noise.safetensors")
    print("rand_noise.safetensors written")

    config = {
        "sample_rate": 24000,
        "llm": {
            "llm_input_size": 896,
            "llm_output_size": 896,
            "speech_token_size": 6561,
            "qwen2": {
                "hidden_size": 896,
                "num_hidden_layers": 24,
                "num_attention_heads": 14,
                "num_key_value_heads": 2,
                "intermediate_size": 4864,
                "max_position_embeddings": 32768,
                "rope_theta": 1000000.0,
                "rms_norm_eps": 1e-6,
                "vocab_size": 151936,
                "tie_word_embeddings": True,
            },
            "sampling": {"top_p": 0.8, "top_k": 25, "win_size": 10, "tau_r": 0.1},
        },
        "flow": {
            "input_size": 80,
            "output_size": 80,
            "vocab_size": 6561,
            "token_mel_ratio": 2,
            "pre_lookahead_len": 3,
            "dit": {
                "dim": 1024, "depth": 22, "heads": 16, "dim_head": 64, "ff_mult": 2,
                "mel_dim": 80, "mu_dim": 80, "spk_dim": 80, "out_channels": 80,
                "static_chunk_size": 50,
            },
            "cfm": {
                "sigma_min": 1e-6, "t_scheduler": "cosine",
                "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7, "n_timesteps": 10,
            },
        },
        "hift": {
            "in_channels": 80, "base_channels": 512, "nb_harmonics": 8,
            "sampling_rate": 24000, "nsf_alpha": 0.1, "nsf_sigma": 0.003,
            "nsf_voiced_threshold": 10.0, "upsample_rates": [8, 5, 3],
            "upsample_kernel_sizes": [16, 11, 7], "istft_n_fft": 16, "istft_hop_len": 4,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "source_resblock_kernel_sizes": [7, 7, 11],
            "source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "lrelu_slope": 0.1, "audio_limit": 0.99, "conv_pre_look_right": 4,
        },
    }
    (dst / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8", newline="\n"
    )
    print("config.json written")


if __name__ == "__main__":
    main()
