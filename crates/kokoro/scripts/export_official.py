"""Export pinned official Kokoro weights and a fixed-noise acoustic fixture.

Development only. Supply existing official checkouts/assets explicitly; this tool
never downloads models, modifies the source checkouts, or writes user resources.
Install the two checkouts in a Python 3.11 environment with misaki[zh], Torch
2.7.1, Transformers 4.51.3, safetensors and soundfile. See SOURCE.md for revisions.
"""

from argparse import ArgumentParser
from contextlib import contextmanager
from importlib import import_module, metadata
from pathlib import Path
import hashlib
import json
import subprocess
import time
import os

import soundfile as sf
import torch
from kokoro import KModel
from misaki.zh import ZHG2P
from safetensors.torch import save_file, load_file

KOKORO_REV = "dfb907a02bba8152ca444717ca5d78747ccb4bec"
MISAKI_REV = "fba1236595f2d2bf21d414ba6e57d25256afada3"
ASSET_HASHES = {
    "kokoro-v1_1-zh.pth": "b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b",
    "config.json": "bc333efa5ce4ceff433c8c8e5d027a1eca0166001e4e4a62bea2d26ff7a46890",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checkout(checkout, revision, modules):
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise ValueError(f"{checkout}: expected {revision}, got {actual}")
    provenance = {}
    for name in modules:
        loaded = Path(import_module(name).__file__).resolve()
        relative = Path(*name.split(".")).with_suffix(".py")
        committed = subprocess.check_output(
            ["git", "-C", str(checkout), "show", f"{revision}:{relative.as_posix()}"]
        )
        digest = sha(loaded)
        if digest != hashlib.sha256(committed).hexdigest():
            raise ValueError(
                f"actually imported {loaded} differs from pinned {revision}"
            )
        provenance[name] = {"path": str(loaded), "sha256": digest, "revision": revision}
    return provenance


@contextmanager
def capture_random(tensors):
    original_rand, original_normal = torch.rand, torch.randn_like
    count = 0

    def draw(kind, fn, *args, **kwargs):
        nonlocal count
        value = fn(*args, **kwargs)
        tensors[f"random.{count}.{kind}"] = value.detach().cpu().contiguous().clone()
        count += 1
        return value

    torch.rand = lambda *a, **kw: draw("rand", original_rand, *a, **kw)
    torch.randn_like = lambda *a, **kw: draw("randn_like", original_normal, *a, **kw)
    try:
        yield
    finally:
        torch.rand, torch.randn_like = original_rand, original_normal


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--kokoro-source", type=Path, required=True)
    parser.add_argument("--misaki-source", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voice-file", type=Path, required=True)
    parser.add_argument("--text", default="春天来了，院子里的花开了。")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--reuse-weights",
        type=Path,
        help="verify and hard-link an existing identical folded weight file",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(
            "output must be a new directory; existing evidence is never overwritten"
        )
    provenance = verify_checkout(
        args.kokoro_source,
        KOKORO_REV,
        ["kokoro.model", "kokoro.modules", "kokoro.istftnet"],
    )
    provenance.update(
        verify_checkout(
            args.misaki_source,
            MISAKI_REV,
            ["misaki.zh", "misaki.zh_frontend", "misaki.tone_sandhi"],
        )
    )
    for name, expected in ASSET_HASHES.items():
        if sha(args.assets / name) != expected:
            raise ValueError(f"official asset hash mismatch: {name}")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = json.loads((args.assets / "config.json").read_text())
    model = KModel(
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        config=config,
        model=str(args.assets / "kokoro-v1_1-zh.pth"),
    ).eval()
    pack = torch.load(args.voice_file, map_location="cpu", weights_only=True)
    ps, _ = ZHG2P(version="1.1")(args.text)
    unknown = [p for p in ps if p not in config["vocab"]]
    if not ps or len(ps) > 510 or unknown:
        raise ValueError(f"fixture must be one supported phoneme segment: {unknown}")
    ids = torch.tensor([[0, *[config["vocab"][p] for p in ps], 0]], dtype=torch.long)
    style = pack[len(ps) - 1].reshape(1, 256)
    tensors = {"input_ids": ids, "style": style}
    handles = []

    def hook(name):
        def capture(module, inputs, value):
            values = value if isinstance(value, tuple) else (value,)
            for i, tensor in enumerate(values):
                if isinstance(tensor, torch.Tensor):
                    key = f"{name}.{i}" if isinstance(value, tuple) else name
                    tensors[key] = tensor.detach().cpu().contiguous().clone()

        return capture

    for name in [
        "bert",
        "bert_encoder",
        "predictor.text_encoder",
        "predictor.lstm",
        "predictor.duration_proj",
        "predictor.F0_proj",
        "predictor.N_proj",
        "text_encoder",
        "decoder.encode",
        "decoder.generator.m_source",
        "decoder.generator.conv_post",
        "decoder",
    ]:
        handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
    try:
        with capture_random(tensors):
            torch.manual_seed(args.seed)
            audio, durations = model.forward_with_tokens(ids, style, args.speed)
    finally:
        for handle in handles:
            handle.remove()
    tensors.update(audio=audio.contiguous(), durations=durations.contiguous())
    for module in model.modules():
        if torch.nn.utils.parametrize.is_parametrized(module, "weight"):
            torch.nn.utils.parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=True
            )
    torch.manual_seed(args.seed)
    folded, folded_durations = model.forward_with_tokens(ids, style, args.speed)
    if not torch.equal(audio, folded) or not torch.equal(durations, folded_durations):
        raise ValueError("weight normalization folding changed the official result")
    timings = []
    for _ in range(3):
        torch.manual_seed(args.seed)
        start = time.perf_counter()
        model.forward_with_tokens(ids, style, args.speed)
        timings.append(time.perf_counter() - start)
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2)
    )
    results = args.output / "results"
    results.mkdir()
    weights = {
        k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()
    }
    if args.reuse_weights:
        existing = load_file(str(args.reuse_weights))
        if existing.keys() != weights.keys() or any(
            not torch.equal(value, existing[key]) for key, value in weights.items()
        ):
            raise ValueError(
                "reuse weight file differs from the pinned official export"
            )
        os.link(args.reuse_weights, results / "weights-folded.safetensors")
    else:
        save_file(weights, str(results / "weights-folded.safetensors"))
    save_file(tensors, str(results / "official-f32.safetensors"))
    save_file({"voice": pack.contiguous()}, str(results / "voice.safetensors"))
    sf.write(results / "official.wav", audio.numpy(), 24000, subtype="FLOAT")
    report = dict(
        text=args.text,
        phonemes=ps,
        speed=args.speed,
        seed=args.seed,
        sample_rate=24000,
        output_samples=audio.numel(),
        source=provenance,
        voice_sha256=sha(args.voice_file),
        assets=ASSET_HASHES,
        versions={
            p: metadata.version(p)
            for p in [
                "torch",
                "transformers",
                "kokoro",
                "misaki",
                "jieba",
                "pypinyin",
                "pypinyin-dict",
                "cn2an",
            ]
        },
        warm_forward_seconds=timings,
        threads=4,
        device="cpu",
        dtype="float32",
    )
    (results / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "source"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
