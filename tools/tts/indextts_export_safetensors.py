"""Development-only checkpoint converter: official IndexTTS 2.5 torch
checkpoints -> the safetensors layout consumed by `crates/indextts`.

Runtime Rust inference never invokes this script. Install torch and
safetensors in a development environment, point it at the official
`checkpoints/` directory (IndexTeam/IndexTTS-2.5 plus the auxiliary models
downloaded to `checkpoints/hf_cache/`), and it writes a converted directory
next to the originals. Converted weights stay under the bilibili model
license; do not redistribute them.

Usage:
    python tools/tts/indextts_export_safetensors.py \
        --checkpoints /path/to/checkpoints \
        --output /path/to/converted

Layout produced (names preserved verbatim from the Python state dicts):
    config.json, gpt.safetensors, codec.safetensors, s2mel.safetensors,
    campplus.safetensors, bigvgan.safetensors, bigvgan.config.json,
    feats.safetensors, multilingual_zh_ja_yue_char_del.tiktoken,
    w2v-bert-2.0/{model.safetensors,config.json}
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def load_state(path: Path, key: str = "model"):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
        return obj[key]
    return obj


def load_s2mel(path: Path):
    """s2mel.pth stores `state['net']` keyed by the ModuleDict names
    (`cfm` / `length_regulator` / `gpt_layer`), each a flat sub-dict; flatten
    to `models.<name>.*` matching `MyModel.models[key].load_state_dict`."""
    net = torch.load(path, map_location="cpu", weights_only=True)["net"]
    state = {}
    for module, sub in net.items():
        for k, v in sub.items():
            state[f"models.{module}.{k}"] = v
    return state


def to_f32(state):
    return {k: v.to(torch.float32).contiguous() for k, v in state.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ckpt = args.checkpoints
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    manifest = {}

    # config.yaml -> config.json (OmegaConf -> plain containers).
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.load(ckpt / "config.yaml"), resolve=True)
    except ImportError:
        import yaml
        cfg = yaml.safe_load((ckpt / "config.yaml").read_text(encoding="utf-8"))
    (out / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    jobs = {
        "gpt.safetensors": (ckpt / "gpt.pth", "model"),
        "codec.safetensors": (ckpt / "codec.pth", "model"),
        # s2mel.pth wraps sub-dicts in state['net'] (flattened specially)
        "s2mel.safetensors": (ckpt / "s2mel.pth", "net-flat"),
        # campplus_cn_common.bin is a plain state dict; bigvgan_generator.pt
        # wraps the generator state dict under 'generator'
        "campplus.safetensors": (ckpt / "hf_cache" / "campplus_cn_common.bin", "model"),
        "bigvgan.safetensors": (
            ckpt / "hf_cache" / "bigvgan" / "bigvgan_generator.pt",
            "generator",
        ),
    }
    for name, (path, key) in jobs.items():
        if not path.is_file():
            raise SystemExit(f"missing checkpoint: {path}")
        state = to_f32(
            load_s2mel(path) if key == "net-flat" else load_state(path, key)
        )
        save_file(state, str(out / name))
        manifest[name] = {"tensors": len(state)}
        print(f"wrote {name}: {len(state)} tensors")

    bigvgan_cfg = ckpt / "hf_cache" / "bigvgan" / "config.json"
    shutil.copy2(bigvgan_cfg, out / "bigvgan.config.json")

    # Emotion / speaker anchors and w2v stats. feat1=spk_matrix, feat2=emo_matrix
    # (config.yaml: spk_matrix: feat1.pt, emo_matrix: feat2.pt).
    feats = {
        "spk_matrix": load_state(ckpt / "feat1.pt").to(torch.float32),
        "emo_matrix": load_state(ckpt / "feat2.pt").to(torch.float32),
    }
    stats = torch.load(ckpt / "wav2vec2bert_stats.pt", map_location="cpu", weights_only=True)
    feats["w2v_mean"] = stats["mean"].to(torch.float32)
    feats["w2v_std"] = torch.sqrt(stats["var"].to(torch.float32))
    save_file(feats, str(out / "feats.safetensors"))
    print(f"wrote feats.safetensors: spk {tuple(feats['spk_matrix'].shape)}, "
          f"emo {tuple(feats['emo_matrix'].shape)}, w2v mean {tuple(feats['w2v_mean'].shape)}")
    manifest["feats.safetensors"] = {k: list(v.shape) for k, v in feats.items()}

    # Tokenizer vocabulary is plain text; copy verbatim.
    tik = ckpt / "multilingual_zh_ja_yue_char_del.tiktoken"
    if tik.is_file():
        shutil.copy2(tik, out / tik.name)

    # w2v-bert-2.0 is already safetensors; copy the two files we need.
    w2v_src = ckpt / "hf_cache" / "w2v-bert-2.0"
    w2v_dst = out / "w2v-bert-2.0"
    w2v_dst.mkdir(exist_ok=True)
    for fname in ("model.safetensors", "config.json"):
        src = w2v_src / fname
        if not src.is_file():
            raise SystemExit(f"missing auxiliary file: {src} (run the official "
                             "IndexTTS once so ensure_models_available downloads it)")
        shutil.copy2(src, w2v_dst / fname)

    (out / "export-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"conversion complete: {out}")


if __name__ == "__main__":
    main()
