#!/usr/bin/env python3
"""Layer-level DiT / CFM / HiFT probe for the Rust CosyVoice3 port.

Extends the fixtures from cosyvoice3_reference.py with:
  1. per-DiT-block activations for the captured first estimator call,
  2. the full Euler trajectory (per-step estimator input and output),
  3. the official F0 prediction for the official mel.

Every captured tensor is cloned immediately: the official Euler loop
reuses its input buffers, so a detached alias would be overwritten.

Usage:
    python cosyvoice3_dit_probe.py --model_dir runs/cv3-model \\
        --fixture runs/cv3-real-ort124 --out runs/cv3-dit-probe
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def save_tensors(tensors, path):
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--f64_out", action="store_true",
                        help="run the first estimator call in float64 and save it as estimator_f64.safetensors")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: PLC0415

    cv3 = CosyVoice3(args.model_dir)
    flow = cv3.model.flow
    hift = cv3.model.hift
    fx = load_file(Path(args.fixture) / "flow.safetensors")
    hf = load_file(Path(args.fixture) / "hift.safetensors")

    # ---- 1. per-block activations for the first estimator call ----------
    blocks = {}
    handles = []
    for i, block in enumerate(flow.decoder.estimator.transformer_blocks):
        def hook(module, inputs, output, idx=i):
            blocks[f"block_{idx:02d}"] = output.detach().float().cpu().clone()
        handles.append(block.register_forward_hook(hook))
    handles.append(flow.decoder.estimator.norm_out.register_forward_hook(
        lambda m, i, o: blocks.__setitem__("norm_out", o.detach().float().cpu().clone())))
    handles.append(flow.decoder.estimator.proj_out.register_forward_hook(
        lambda m, i, o: blocks.__setitem__("proj_out", o.detach().float().cpu().clone())))
    est_out = flow.decoder.estimator(
        fx["x"], fx["mask"], fx["mu"], fx["t"], fx["spks"], fx["cond"], streaming=False)
    for h in handles:
        h.remove()
    blocks["final"] = est_out.detach().float().cpu().clone()
    # activation magnitude profile for scale context
    profile = {k: {"max_abs": float(v.abs().max()), "rms": float(v.pow(2).mean().sqrt())}
               for k, v in blocks.items()}
    save_tensors(blocks, out / "ditblocks.safetensors")

    # ---- 1b. submodule detail for selected blocks (attn_norm tuple,
    #          attention output, feedforward output) -----------------------
    detail = {}
    wanted = {0, *range(15, 22)}
    handles = []
    for i, block in enumerate(flow.decoder.estimator.transformer_blocks):
        if i not in wanted:
            continue

        def norm_hook(module, inputs, output, idx=i):
            for name, t in zip(("norm", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"), output):
                detail[f"b{idx:02d}_{name}"] = t.detach().float().cpu().clone()

        def attn_hook(module, inputs, output, idx=i):
            detail[f"b{idx:02d}_attn"] = output.detach().float().cpu().clone()

        def ff_hook(module, inputs, output, idx=i):
            detail[f"b{idx:02d}_ff"] = output.detach().float().cpu().clone()

        handles.append(block.attn_norm.register_forward_hook(norm_hook))
        handles.append(block.attn.register_forward_hook(attn_hook))
        def block_hook(module, inputs, output, idx=i):
            prefix = f"b{idx:02d}_"
            detail[prefix + "after_attn"] = (
                inputs[0].detach().float().cpu().clone()
                + detail[prefix + "gate_msa"].unsqueeze(1) * detail[prefix + "attn"]
            )
            detail[prefix + "out"] = output.detach().float().cpu().clone()

        handles.append(block.ff.register_forward_hook(ff_hook))
        handles.append(block.register_forward_hook(block_hook))
    est_out2 = flow.decoder.estimator(
        fx["x"], fx["mask"], fx["mu"], fx["t"], fx["spks"], fx["cond"], streaming=False)
    for h in handles:
        h.remove()
    assert torch.equal(est_out2, est_out)
    save_tensors(detail, out / "ditdetail.safetensors")

    # ---- 1c. exact float64 anchor for the first estimator call -----------
    if args.f64_out:
        est64 = flow.decoder.estimator.double()
        try:
            out64 = est64(
                fx["x"].double(), fx["mask"].double(), fx["mu"].double(), fx["t"].double(),
                fx["spks"].double(), fx["cond"].double(), streaming=False)
            d32 = (out64.float() - fx["out"]).abs()
            print(json.dumps({"official_f64_vs_official_f32": {
                "max_abs": float(d32.max()),
                "rmse": float(d32.pow(2).mean().sqrt())}}), flush=True)
            save_tensors({"out": out64.detach().cpu()}, out / "estimator_f64.safetensors")
        finally:
            # Module.double() mutates the model used by the f32 Euler probe.
            flow.decoder.estimator.float()

    # ---- 2. full Euler trajectory with shared token/prompt/noise ---------
    steps = {}
    orig_estimator = flow.decoder.estimator.forward

    def estimator_forward(x, mask, mu, t, spks, cond, streaming=False):
        idx = len(steps) // 3
        steps[f"step_{idx:02d}_x"] = x.detach().float().cpu().clone()
        steps[f"step_{idx:02d}_t"] = t.detach().float().cpu().clone()
        out_t = orig_estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        steps[f"step_{idx:02d}_out"] = out_t.detach().float().cpu().clone()
        return out_t

    flow.decoder.estimator.forward = estimator_forward
    mel, _ = flow.inference(
        token=fx["token"].to(torch.int32),
        token_len=torch.tensor([fx["token"].shape[1]], dtype=torch.int32),
        prompt_token=fx["prompt_token"].to(torch.int32),
        prompt_token_len=torch.tensor([fx["prompt_token"].shape[1]], dtype=torch.int32),
        prompt_feat=fx["prompt_feat"],
        prompt_feat_len=torch.tensor([fx["prompt_feat"].shape[1]], dtype=torch.int32),
        embedding=fx["embedding"],
        streaming=False,
        finalize=True,
    )
    flow.decoder.estimator.forward = orig_estimator
    steps["final_mel"] = mel.detach().float().cpu().clone()
    save_tensors(steps, out / "euler.safetensors")

    # ---- 3. official F0 for the official mel -----------------------------
    f0_cap = {}

    def f0_hook(module, inputs, output):
        if "f0" not in f0_cap:
            f0_cap["f0"] = output.detach().float().cpu().clone()
    handle = hift.f0_predictor.register_forward_hook(f0_hook)
    pcm, _ = hift.inference(speech_feat=hf["mel"], finalize=True)
    handle.remove()
    f0_cap["pcm"] = pcm.detach().float().cpu().clone()
    save_tensors(f0_cap, out / "f0.safetensors")

    # ---- 4. self-sensitivity: official pipeline under this thread count
    #         versus the reference fixtures (which used threads=4) ---------
    pcm2, _ = hift.inference(speech_feat=steps["final_mel"], finalize=True)
    ref_pcm = hf["pcm"]
    diff = (pcm2 - ref_pcm).abs()
    print(json.dumps({
        "threads": args.threads,
        "estimator_vs_ref": {"max_abs": float((blocks["final"] - fx["out"]).abs().max()),
                             "rmse": float(((blocks["final"] - fx["out"]).pow(2)).mean().sqrt())},
        "mel_vs_ref": {"max_abs": float((steps["final_mel"] - fx["mel"]).abs().max()),
                       "rmse": float(((steps["final_mel"] - fx["mel"]).pow(2)).mean().sqrt())},
        "pcm_vs_ref": {"max_abs": float(diff.max()),
                       "rmse": float(diff.pow(2).mean().sqrt()),
                       "snr_db": float(10 * torch.log10(ref_pcm.pow(2).mean() / diff.pow(2).mean()))},
    }), flush=True)

    (out / "profile.json").write_text(json.dumps(
        {"activation_profile": profile, "threads": args.threads,
         "euler_steps": len(steps) // 3}, indent=1))
    print(f"probe fixtures written to {out}", flush=True)


if __name__ == "__main__":
    main()
