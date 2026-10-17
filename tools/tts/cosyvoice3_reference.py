#!/usr/bin/env python3
"""Dump official CosyVoice3 intermediates as fixtures for the Rust port.

Run inside the official CosyVoice repo (commit 074ca6dc) with the
Fun-CosyVoice3-0.5B-2512 model directory available. Every stage the Rust
crate reimplements gets a tensor pair (input, output) so
`crates/cosyvoice3` probes can compare layer by layer.

Usage:
    python cosyvoice3_reference.py --model_dir <dir> --prompt_wav a.wav \\
        --prompt_text "<|endofprompt|> transcript" --text "你好世界" --out runs/cv3-fixtures
"""

import argparse
import sys
import json
import time
import torchaudio
from pathlib import Path

import torch
from safetensors.torch import save_file


def save_tensors(tensors, path):
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--prompt_wav", required=True)
    parser.add_argument("--prompt_text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocoder_stages", action="store_true", help="export official HiFT/source intermediates for diagnostic probes")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: PLC0415

    torch.set_num_threads(4)
    started = time.perf_counter()
    cv3 = CosyVoice3(args.model_dir)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    frontend = cv3.frontend
    model = cv3.model

    feats = {}
    model_input = frontend.frontend_zero_shot(args.text, args.prompt_text, args.prompt_wav, 24000, '')
    prompt_speech_token = model_input['flow_prompt_speech_token']
    speech_feat = model_input['prompt_speech_feat']
    embedding = model_input['flow_embedding']
    feats["prompt_speech_token"] = prompt_speech_token
    feats["prompt_speech_feat"] = speech_feat
    feats["prompt_embedding"] = embedding
    save_tensors(feats, out / "frontend.safetensors")

    # LLM stage: capture the first-step logits by wrapping the decoder.
    text_token, _ = frontend._extract_text_token(args.text)
    prompt_text_token, _ = frontend._extract_text_token(args.prompt_text)
    llm = model.llm
    captured = {}
    orig_decoder_forward = llm.llm_decoder.forward

    def decoder_forward(x):
        out = orig_decoder_forward(x)
        if "first_logits" not in captured and x.shape[1] >= 1:
            captured["first_logits"] = (out if out.ndim == 2 else out[:, -1]).detach().float().cpu().clone()
        return out

    llm.llm_decoder.forward = decoder_forward
    torch.manual_seed(42)
    tokens = []
    for t in llm.inference(
        text=text_token,
        text_len=torch.tensor([text_token.shape[1]], dtype=torch.int32),
        prompt_text=prompt_text_token,
        prompt_text_len=torch.tensor([prompt_text_token.shape[1]], dtype=torch.int32),
        prompt_speech_token=prompt_speech_token,
        prompt_speech_token_len=torch.tensor([prompt_speech_token.shape[1]], dtype=torch.int32),
        embedding=torch.zeros(0, 192),
    ):
        tokens.append(int(t))
    llm.llm_decoder.forward = orig_decoder_forward
    save_tensors(
        {
            "text_token": text_token.long().cpu(),
            "prompt_text_token": prompt_text_token.long().cpu(),
            "first_logits": captured["first_logits"],
            "speech_tokens": torch.tensor(tokens, dtype=torch.int64),
        },
        out / "llm.safetensors",
    )

    llm_seconds = time.perf_counter() - started
    started = time.perf_counter()

    # Flow stage: capture the estimator call once (inputs -> output).
    flow = model.flow
    captured.clear()
    orig_estimator = flow.decoder.estimator.forward
    stages = {}
    def capture_stage(name):
        def hook(module, inputs, output):
            if name not in stages:
                stages[name] = output.detach().float().cpu().clone()
        return hook
    handles = [flow.decoder.estimator.time_embed.register_forward_hook(capture_stage("time_embed")),
               flow.decoder.estimator.input_embed.register_forward_hook(capture_stage("input_embed"))]

    def estimator_forward(x, mask, mu, t, spks, cond, streaming=False):
        out_t = orig_estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        if "x" not in captured:
            captured.update(
                x=x.detach().float().cpu().clone(), mask=mask.detach().float().cpu().clone(),
                mu=mu.detach().float().cpu().clone(), t=t.detach().float().cpu().clone(),
                spks=spks.detach().float().cpu().clone(), cond=cond.detach().float().cpu().clone(),
                out=out_t.detach().float().cpu().clone(),
            )
        return out_t

    flow.decoder.estimator.forward = estimator_forward
    mel, _ = flow.inference(
        token=torch.tensor([tokens], dtype=torch.int32),
        token_len=torch.tensor([len(tokens)], dtype=torch.int32),
        prompt_token=prompt_speech_token,
        prompt_token_len=torch.tensor([prompt_speech_token.shape[1]], dtype=torch.int32),
        prompt_feat=speech_feat,
        prompt_feat_len=torch.tensor([speech_feat.shape[1]], dtype=torch.int32),
        embedding=embedding,
        streaming=False,
        finalize=True,
    )
    flow.decoder.estimator.forward = orig_estimator
    for handle in handles:
        handle.remove()
    save_tensors(
        {
            **captured,
            **stages,
            "token": torch.tensor([tokens], dtype=torch.int64),
            "prompt_token": prompt_speech_token.long().cpu(),
            "prompt_feat": speech_feat.float().cpu(),
            "embedding": embedding.float().cpu(),
            "mel": mel.float().cpu(),
            "noise": flow.decoder.rand_noise[:, :, :captured["x"].shape[2]].float().cpu().contiguous(),
        },
        out / "flow.safetensors",
    )

    flow_seconds = time.perf_counter() - started
    started = time.perf_counter()

    # Optional diagnostics hook the actual official modules, without
    # maintaining a second implementation of the vocoder in a script.
    hift_stages, sine_stages, hift_handles = {}, {}, []
    if args.vocoder_stages:
        def tensor_hook(name, select=lambda x: x):
            def hook(module, inputs, output):
                hift_stages[name] = select(output).detach().float().cpu().clone()
            return hook
        hift_handles.append(model.hift.f0_predictor.register_forward_hook(tensor_hook("f0")))
        hift_handles.append(model.hift.conv_pre.register_forward_hook(tensor_hook("x_pre")))
        hift_handles.append(model.hift.m_source.register_forward_hook(tensor_hook("source", lambda x: x[0].squeeze(-1))))
        for i, block in enumerate(model.hift.source_resblocks):
            hift_handles.append(block.register_forward_hook(tensor_hook(f"si_{i}")))
        def sine_hook(module, inputs, output):
            sine_stages["f0_up"] = inputs[0].detach().float().cpu().clone()
            sine_stages["sine_waves"] = output[0].detach().float().cpu().clone()
        def merge_hook(module, inputs, output):
            sine_stages["merged"] = output[0].detach().float().cpu().clone()
        hift_handles.append(model.hift.m_source.l_sin_gen.register_forward_hook(sine_hook))
        hift_handles.append(model.hift.m_source.register_forward_hook(merge_hook))
    try:
        pcm, _ = model.hift.inference(speech_feat=mel, finalize=True)
    finally:
        for handle in hift_handles:
            handle.remove()
    if args.vocoder_stages:
        hift_stages["pcm"] = pcm
        save_tensors(hift_stages, out / "official_stages.safetensors")
        save_tensors(sine_stages, out / "sinegen_stages.safetensors")
    source = model.hift.m_source.l_sin_gen
    save_tensors({"mel": mel.float().cpu(), "pcm": pcm.float().cpu(),
               "rand_ini": source.rand_ini.cpu().contiguous(),
               "source_noise": source.sine_waves[:, :mel.shape[2] * 480].cpu().contiguous()}, out / "hift.safetensors")
    torchaudio.save(str(out / "official.wav"), pcm.float().cpu(), 24000)
    (out / "metadata.json").write_text(json.dumps({"text": args.text, "prompt_text": args.prompt_text,
        "prompt_wav": args.prompt_wav, "torch": torch.__version__, "onnxruntime": __import__("onnxruntime").__version__, "load_seconds": load_seconds,
        "llm_and_frontend_seconds": llm_seconds, "flow_seconds": flow_seconds,
        "hift_seconds": time.perf_counter()-started, "tokens":len(tokens), "audio_seconds": pcm.shape[-1]/24000}, ensure_ascii=False, indent=2))
    print(f"fixtures written to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
