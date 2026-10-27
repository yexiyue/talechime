"""Assemble a new verified native runtime bundle from explicit official exports.

The output is a new directory. This tool never overwrites an existing runtime
bundle and never scans or migrates historical Talechime user directories.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import shutil

import torch
from safetensors.torch import save_file
from export_official import ASSET_HASHES, sha


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model-export", type=Path, required=True)
    parser.add_argument("--frontend-export", type=Path, required=True)
    parser.add_argument(
        "--context-export",
        type=Path,
        help="explicit optional official g2pW export; use model 1.1-zh-g2pw",
    )
    parser.add_argument(
        "--voice",
        type=Path,
        action="append",
        required=True,
        help="official voice .pt; repeat to install more voices",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must not exist; existing data is never overwritten")
    if args.context_export:
        context = json.loads((args.context_export / "context.json").read_text())
        if (
            context["format"] != "kokoro-g2pw-v2"
            or context["model_sha256"] != sha(args.context_export / "g2pw.onnx")
            or context["vocab_sha256"] != sha(args.context_export / "vocab.txt")
        ):
            raise ValueError("invalid context export")
    report = json.loads((args.model_export / "results/report.json").read_text())
    if report["assets"] != ASSET_HASHES or report["dtype"] != "float32":
        raise ValueError("not a pinned original-weight export")
    pinned_config = json.loads(
        (Path(__file__).resolve().parent.parent / "assets/config.json").read_text()
    )
    if json.loads((args.model_export / "config.json").read_text()) != pinned_config:
        raise ValueError("official config content mismatch")
    frontend = json.loads((args.frontend_export / "frontend.json").read_text())
    if (
        frontend["format"] != "kokoro-misaki-zh-v1"
        or frontend["source_revision"] != "fba1236595f2d2bf21d414ba6e57d25256afada3"
    ):
        raise ValueError("unsupported frontend source")
    expected = {
        x["id"]: x
        for x in json.loads(
            (Path(__file__).resolve().parent.parent / "assets/voices.json").read_text()
        )
    }
    voices = {}
    for voice in args.voice:
        descriptor = expected.get(voice.stem)
        if descriptor is None or sha(voice) != descriptor["sha256"]:
            raise ValueError(f"voice does not match pinned official asset: {voice}")
        if not voice.stem.startswith(("zf_", "zm_")):
            raise ValueError("the current frontend supports Mandarin voices only")
        pack = torch.load(voice, map_location="cpu", weights_only=True)
        if (
            pack.dtype != torch.float32
            or tuple(pack.shape) != (510, 1, 256)
            or not torch.isfinite(pack).all()
        ):
            raise ValueError(f"invalid official voice tensor: {voice}")
        voices[voice.stem] = (voice, pack)
    args.output.mkdir(parents=True)
    if args.context_export:
        (args.output / "context").mkdir()
        for name in [
            "context.json",
            "g2pw.onnx",
            "vocab.txt",
            "G2PW-LICENSE",
            "BERT-SOURCE.txt",
        ]:
            shutil.copyfile(args.context_export / name, args.output / "context" / name)
    shutil.copyfile(
        args.model_export / "results/weights-folded.safetensors",
        args.output / "model.safetensors",
    )
    shutil.copyfile(args.model_export / "config.json", args.output / "config.json")
    shutil.copytree(
        args.frontend_export,
        args.output / "frontend",
        ignore=shutil.ignore_patterns("fixtures.json"),
    )
    (args.output / "voices").mkdir()
    for name, (_, pack) in voices.items():
        save_file(
            {"voice": pack.contiguous()},
            str(args.output / "voices" / f"{name}.safetensors"),
        )
    shutil.copyfile(
        args.model_export / "results/report.json", args.output / "official-export.json"
    )
    shutil.copyfile(
        Path(__file__).resolve().parent.parent / "LICENSE",
        args.output / "KOKORO-LICENSE",
    )
    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            files[path.relative_to(args.output).as_posix()] = dict(
                bytes=path.stat().st_size, sha256=sha(path)
            )
    manifest = dict(
        format="kokoro-native-v1",
        model_revision="01e7505bd6a7a2ac4975463114c3a7650a9f7218",
        contextual=bool(args.context_export),
        voices={
            name: dict(
                path=f"voices/{name}.safetensors",
                source_sha256=expected[name]["sha256"],
            )
            for name in sorted(voices)
        },
        files=files,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(args.output.resolve()),
                voices=list(manifest["voices"]),
                files=len(files),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
