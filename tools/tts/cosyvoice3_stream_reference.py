#!/usr/bin/env python3
"""Dump official CosyVoice3 streaming chunk fixtures for the Rust port.

Replicates the exact `CosyVoice2Model.tts(stream=True)` loop from commit
074ca6dc but feeds fixed speech tokens from a cosyvoice3_reference.py
fixture, so Rust can compare chunk-by-chunk against the same schedule.
Also records how the official stream mode differs from the official
non-stream decode of the same tokens (streaming recompute changes the
result; this is the reference for that gap).

Usage:
    python cosyvoice3_stream_reference.py --model_dir runs/cv3-model \\
        --fixture runs/cv3-real-ort124 --out runs/cv3-real-ort124/stream
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: PLC0415

    cv3 = CosyVoice3(args.model_dir)
    model = cv3.model
    llm_fx = load_file(Path(args.fixture) / "llm.safetensors")
    front_fx = load_file(Path(args.fixture) / "frontend.safetensors")
    hift_fx = load_file(Path(args.fixture) / "hift.safetensors")
    tokens = llm_fx["speech_tokens"].flatten().tolist()
    prompt_token = front_fx["prompt_speech_token"]
    prompt_feat = front_fx["prompt_speech_feat"]
    embedding = front_fx["prompt_embedding"]

    uuid_key = "stream-probe"
    model.hift_cache_dict[uuid_key] = None
    source = model.hift.m_source.l_sin_gen
    rand_ini = source.rand_ini.detach().cpu().contiguous()
    need = (prompt_feat.shape[1] + len(tokens) * 2) * 480
    source_noise = source.sine_waves[:, :need].detach().cpu().contiguous()
    chunk_mels = []
    prompt_frames = prompt_feat.shape[1]

    def flow_hook(module, inputs, output):
        # decoder returns (solved_mel, cache); slice off the prompt region.
        chunk_mels.append(output[0][:, :, prompt_frames:].detach().float().cpu().clone())

    model.flow.decoder.register_forward_hook(flow_hook)

    token_hop_len, token_max_hop_len, stream_scale_factor = 25, 100, 2
    pre_lookahead = model.flow.pre_lookahead_len
    token_offset = 0
    prompt_token_pad = int(np.ceil(prompt_token.shape[1] / token_hop_len) * token_hop_len - prompt_token.shape[1])
    chunks = []
    meta = {"pre_lookahead": pre_lookahead, "prompt_token_pad": prompt_token_pad,
            "tokens": len(tokens), "chunks": []}
    while True:
        this_token_hop_len = token_hop_len + prompt_token_pad if token_offset == 0 else token_hop_len
        if len(tokens) - token_offset >= this_token_hop_len + pre_lookahead:
            this_tts_speech_token = torch.tensor(
                tokens[:token_offset + this_token_hop_len + pre_lookahead]).unsqueeze(dim=0)
            pcm = model.token2wav(token=this_tts_speech_token,
                                  prompt_token=prompt_token,
                                  prompt_feat=prompt_feat,
                                  embedding=embedding,
                                  token_offset=token_offset,
                                  uuid=uuid_key,
                                  stream=True,
                                  finalize=False)
            chunks.append(pcm.flatten().float().cpu())
            meta["chunks"].append({"token_offset": token_offset, "hop": this_token_hop_len,
                                   "samples": int(pcm.shape[-1])})
            token_offset += this_token_hop_len
            token_hop_len = min(token_max_hop_len, token_hop_len * stream_scale_factor)
        # token source is exhausted from the start
        if len(tokens) - token_offset < this_token_hop_len + pre_lookahead:
            break
    final = model.token2wav(token=torch.tensor([tokens]).unsqueeze(dim=0) if False else torch.tensor(tokens).unsqueeze(dim=0),
                            prompt_token=prompt_token,
                            prompt_feat=prompt_feat,
                            embedding=embedding,
                            token_offset=token_offset,
                            uuid=uuid_key,
                            finalize=True)
    final_pcm = final.flatten().float().cpu()
    meta["chunks"].append({"token_offset": token_offset, "hop": "final",
                           "samples": int(final_pcm.shape[-1])})
    model.hift_cache_dict.pop(uuid_key, None)

    stream_all = torch.cat(chunks + [final_pcm])
    ref = hift_fx["pcm"].flatten()
    n = min(stream_all.shape[-1], ref.shape[-1])
    gap = (stream_all[:n] - ref[:n]).abs()
    meta["official_stream_vs_official_nonstream"] = {
        "max_abs": float(gap.max()), "rmse": float(gap.pow(2).mean().sqrt()),
        "samples": int(n),
        "note": "streaming recompute (finalize=False chunks + re-decodes) vs one-shot decode of the same tokens",
    }

    tensors = {f"chunk_{i:02}": c.contiguous() for i, c in enumerate(chunks)}
    tensors["final"] = final_pcm.contiguous()
    tensors["rand_ini"] = rand_ini
    tensors["source_noise"] = source_noise
    for i, m in enumerate(chunk_mels):
        tensors[f"flow_mel_{i:02}"] = m
    save_file(tensors, out / "stream-chunks.safetensors")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"chunks": len(chunks) + 1,
                      "stream_vs_nonstream_max": meta["official_stream_vs_official_nonstream"]["max_abs"]}), flush=True)


if __name__ == "__main__":
    main()
