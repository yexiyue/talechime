"""Development-only ONNX export for the IndexTTS 2.5 comparison path.

EXPERIMENTAL: the official project ships no ONNX graphs and community
conversions cover only the v1-era GPT, so this script exports the pinned
2.5 components directly with `torch.onnx.export`. Outputs are for local
A/B comparison against the Candle port; they are not used at runtime and
must not be redistributed (weight licenses apply).

Dynamic shapes are used where the exporter supports them (codec lengths,
mel frames); the GPT is exported as a single decode step with past
key/value tensors wired as inputs/outputs so a Rust/ORT runner can loop
it exactly like the HF KV-cache path.

Usage:
    python tools/tts/indextts_export_onnx.py \
        --source /path/to/index-tts --checkpoints /path/to/checkpoints \
        --output /path/to/onnx [--components campplus,bigvgan,codec,gpt_step]

Requires the index-tts dev environment (torch 2.8) plus onnx and
onnxscript. Verify results with onnxruntime on the exporting machine
before trusting them as oracles.
"""
import argparse
import sys
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--components", default="campplus,bigvgan,codec",
                   help="comma list of: campplus,bigvgan,codec,gpt_step")
    args = p.parse_args()
    sys.path.insert(0, str(args.source))
    from omegaconf import OmegaConf

    from indextts.codec.models import EnhancedCodec
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    from indextts.s2mel.modules.bigvgan import bigvgan as bigvgan_mod

    args.output.mkdir(parents=True, exist_ok=True)
    components = [c.strip() for c in args.components.split(",") if c.strip()]
    cfg = OmegaConf.load(args.checkpoints / "config.yaml")

    if "campplus" in components:
        model = CAMPPlus(feat_dim=80, embedding_size=192)
        model.load_state_dict(torch.load(
            args.checkpoints / "hf_cache" / "campplus_cn_common.bin", map_location="cpu"))
        model.eval()
        feat = torch.randn(1, 200, 80)
        torch.onnx.export(model, feat, args.output / "campplus.onnx",
                          input_names=["feat"], output_names=["style"],
                          dynamic_axes={"feat": {1: "frames"}})
        print("exported campplus.onnx")

    if "bigvgan" in components:
        bigvgan = bigvgan_mod.BigVGAN.from_pretrained(
            str(args.checkpoints / "hf_cache" / "bigvgan"), use_cuda_kernel=False)
        bigvgan.remove_weight_norm()
        bigvgan.eval()
        mel = torch.randn(1, 80, 64)
        torch.onnx.export(bigvgan, mel, args.output / "bigvgan.onnx",
                          input_names=["mel"], output_names=["wav"],
                          dynamic_axes={"mel": {2: "frames"}, "wav": {2: "samples"}},
                          opset_version=17)
        print("exported bigvgan.onnx")

    if "codec" in components:
        codec = EnhancedCodec(**cfg.semantic_codec, cfg=cfg.semantic_codec)
        codec.load_checkpoint(str(args.checkpoints / "codec.pth"))
        codec.eval()

        class QuantizeDecode(torch.nn.Module):
            def __init__(self, codec):
                super().__init__()
                self.codec = codec

            def forward(self, codes):
                return self.codec.decode(codes)

        codes = torch.zeros(1, 32, dtype=torch.long)
        torch.onnx.export(QuantizeDecode(codec), codes, args.output / "codec_decode.onnx",
                          input_names=["codes"], output_names=["features"],
                          dynamic_axes={"codes": {1: "frames"}, "features": {1: "frames"}},
                          opset_version=17)
        print("exported codec_decode.onnx")

    if "gpt_step" in components:
        gpt = UnifiedVoice(**cfg.gpt, use_accel=False, spk_cond_mode="campplus")
        from indextts.utils.checkpoint import load_checkpoint
        load_checkpoint(gpt, str(args.checkpoints / "gpt.pth"))
        gpt.eval()
        gpt.post_init_gpt2_config(kv_cache=True, half=False)
        dim = cfg.gpt.model_dim
        layers = cfg.gpt.layers

        class Step(torch.nn.Module):
            """One GPT2 decode step: embeds the newest mel token, appends to
            the KV cache and returns next-token logits plus new cache."""

            def __init__(self, inference_model):
                super().__init__()
                self.m = inference_model

            def forward(self, input_id, attention_mask, mel_emb, *past):
                # past: flattened (k, v) per layer, each (1, heads, T, hd)
                self.m.cached_mel_emb = mel_emb
                ids = input_id.view(1, 1)
                out = self.m(input_ids=ids, attention_mask=attention_mask,
                             past_key_values=past, use_cache=True)
                return (out.logits, *tuple(t for kv in out.past_key_values for t in kv))

        step = Step(gpt.inference_model).eval()
        heads = cfg.gpt.heads
        hd = dim // heads
        input_id = torch.tensor([[8192]])
        attention_mask = torch.ones(1, 10, dtype=torch.long)
        mel_emb = torch.randn(1, 9, dim)
        past = [torch.zeros(1, heads, 0, hd) for _ in range(2 * layers)]
        torch.onnx.export(step, (input_id, attention_mask, mel_emb, *past),
                          args.output / "gpt_step.onnx",
                          input_names=["input_id", "attention_mask", "mel_emb",
                                       *[f"past_{i}" for i in range(2 * layers)]],
                          output_names=["logits", *[f"present_{i}" for i in range(2 * layers)]],
                          dynamic_axes={"attention_mask": {1: "total_len"},
                                        "mel_emb": {1: "prompt_len"},
                                        **{f"past_{i}": {2: "past_len"} for i in range(2 * layers)},
                                        **{f"present_{i}": {2: "past_len1"} for i in range(2 * layers)}},
                          opset_version=17)
        print("exported gpt_step.onnx (verify past/present wiring with onnxruntime)")

    print("done:", ", ".join(p.name for p in args.output.glob("*.onnx")))


if __name__ == "__main__":
    main()
