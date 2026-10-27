"""Export pinned official g2pW data and independent ONNX input/output fixtures.

Development only. Reads an already downloaded and SHA-verified official archive
and explicit local official model/tokenizer checkouts. No runtime Python or
training pipeline is introduced.
"""

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import importlib
import json
import shutil
import subprocess
import zipfile

import cn2an
import numpy as np
import onnxruntime
from g2pw import G2PWConverter
from misaki.zh import ZHG2P
from pypinyin import Style
from pypinyin.converter import UltimateConverter
from safetensors.numpy import save_file

REVISION = "b9328ce8129c4717a70f16a2688542a0197541dd"
ARCHIVE_SHA = "699f3c1fd7fb0e2c2d49ed2486826fd5bff233fee7759350a91c3b49aedc4ed2"
TOKENIZER_REV = "8f23c25b06e129b6c986331a13d8d025a92cf0ea"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output must be a new directory")
    if sha(args.archive) != ARCHIVE_SHA:
        raise ValueError("official g2pW archive digest mismatch")
    if (
        sha(args.model_dir / "g2pw.onnx")
        != "bb40c8c7b5baa755b2acd317c6bc5a65e4af7b80c40a569247fbd76989299999"
    ):
        raise ValueError("official ONNX digest mismatch")
    if (
        sha(args.tokenizer_dir / "vocab.txt")
        != "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c"
    ):
        raise ValueError("official tokenizer digest mismatch")
    with zipfile.ZipFile(args.archive) as archive:
        for name in [
            "config.py",
            "MONOPHONIC_CHARS.txt",
            "POLYPHONIC_CHARS.txt",
            "version",
        ]:
            if (args.model_dir / name).read_bytes() != archive.read(
                "G2PWModel/" + name
            ):
                raise ValueError(f"official archive data mismatch: {name}")
    head = subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != REVISION:
        raise ValueError("unexpected official g2pW revision")
    provenance = {}
    for name in ["g2pw.api", "g2pw.dataset", "g2pw.utils"]:
        loaded = Path(importlib.import_module(name).__file__)
        relative = Path(*name.split(".")).with_suffix(".py")
        expected = subprocess.check_output(
            ["git", "-C", str(args.source), "show", f"{REVISION}:{relative.as_posix()}"]
        )
        if hashlib.sha256(expected).hexdigest() != sha(loaded):
            raise ValueError(f"imported {name} differs from source revision")
        provenance[name] = dict(path=str(loaded), sha256=sha(loaded))
    conv = G2PWConverter(
        model_dir=str(args.model_dir),
        model_source=str(args.tokenizer_dir),
        style="pinyin",
        enable_non_tradional_chinese=True,
        num_workers=0,
        batch_size=32,
    )
    if (
        conv.config.window_size != 32
        or not conv.config.use_mask
        or conv.config.use_char_phoneme
    ):
        raise ValueError("unsupported g2pW configuration")
    convert = UltimateConverter(neutral_tone_with_five=True)
    label_readings = []
    pinyins = []
    for label in conv.labels:
        py = conv._convert_bopomofo_to_pinyin(label)
        pinyins.append(py)
        label_readings.append(
            None
            if py is None
            else [
                convert.convert_style("字", py, Style.INITIALS, strict=True),
                convert.convert_style("字", py, Style.FINALS_TONE3, strict=True),
            ]
        )
    args.output.mkdir(parents=True)
    shutil.copyfile(args.model_dir / "g2pw.onnx", args.output / "g2pw.onnx")
    shutil.copyfile(args.tokenizer_dir / "vocab.txt", args.output / "vocab.txt")
    shutil.copyfile(args.source / "LICENCE", args.output / "G2PW-LICENSE")
    data = dict(
        format="kokoro-g2pw-v2",
        source_revision=REVISION,
        archive_sha256=ARCHIVE_SHA,
        tokenizer_revision=TOKENIZER_REV,
        source_repository="https://github.com/GitYCC/g2pW",
        model_repository="https://huggingface.co/YC-Chen/g2pW",
        tokenizer_repository="https://huggingface.co/google-bert/bert-base-chinese",
        model_sha256=sha(args.model_dir / "g2pw.onnx"),
        vocab_sha256=sha(args.tokenizer_dir / "vocab.txt"),
        labels=conv.labels,
        label_pinyin=pinyins,
        label_readings=label_readings,
        chars=conv.chars,
        char2phonemes=conv.char2phonemes,
        s2t=conv.s2t_dict,
        window_size=32,
        batch_size=32,
        source=provenance,
    )
    (args.output / "BERT-SOURCE.txt").write_text(
        f"Vocabulary from google-bert/bert-base-chinese, revision {TOKENIZER_REV}.\nLicensed under Apache-2.0; see G2PW-LICENSE for the full license.\nhttps://huggingface.co/google-bert/bert-base-chinese\n"
    )
    (args.output / "context.json").write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    )
    cases = [
        ("单先生今天来做客。", "单", "shan4"),
        ("解先生答应把书归还给她。", "解", "xie4"),
        ("曾先生正在门口等候。", "曾", "zeng1"),
        ("仇先生住在村口。", "仇", "qiu2"),
        ("区先生正在看书。", "区", "ou1"),
        ("银行行长正在办公室里。", "行", "hang2"),
        ("她还没有回来。", "还", "hai2"),
        ("他把钱还给了朋友。", "还", "huan2"),
        ("这件东西很重。", "重", "zhong4"),
        ("他决定重新开始。", "重", "chong2"),
        ("他长大以后想当医生。", "长", "zhang3"),
        ("长长的街道通向远方。", "长", "chang2"),
        ("将军率领大军出征。", "率", "shuai4"),
        ("这项工作的效率很高。", "率", "lv4"),
        ("她觉得这个故事很有趣。", "觉", "jue2"),
        ("他正在房间里睡觉。", "觉", "jiao4"),
        ("他把衣服放在椅子上。", "衣", "yi1"),
        ("这个苹果很好吃。", "好", "hao3"),
        ("他爱好读书。", "好", "hao4"),
        ("大家都在等着他。", "着", "zhe5"),
        (
            "清晨的阳光照进安静的小院，大家吃过早饭之后就来到门口，单先生正在那里等候。",
            "单",
            "shan4",
        ),
        (
            "他把桌上的资料整理完毕，走出办公室，沿着长长的走廊来到楼下的银行。",
            "行",
            "hang2",
        ),
        ("单先生  今天来做客，大家都在门口等着他。", "单", "shan4"),
        (
            "大家听完故事以后，觉得很有趣，于是决定重新开始阅读这本长篇小说。",
            "重",
            "chong2",
        ),
    ]
    tensors = {}
    reports = []
    session = conv.session_g2pw
    for number, (text, character, gold) in enumerate(cases):
        normalized = ZHG2P.map_punctuation(cn2an.transform(text, "an2cn"))

        class Recorder:
            def __init__(self):
                self.batch = 0

            def run(self, names, feeds):
                prefix = f"case.{number}.batch.{self.batch}"
                output = session.run(names, feeds)
                for name, value in feeds.items():
                    tensors[f"{prefix}.{name}"] = value.copy()
                tensors[f"{prefix}.probabilities"] = output[0].copy()
                self.batch += 1
                return output

        recorder = Recorder()
        conv.session_g2pw = recorder
        prediction = conv(normalized)[0]
        index = normalized.index(character)
        reports.append(
            dict(
                text=text,
                normalized=normalized,
                index=index,
                character=character,
                gold=gold,
                prediction=prediction[index],
                predictions=prediction,
                batches=recorder.batch,
            )
        )
    conv.session_g2pw = session
    save_file(tensors, str(args.output / "official-context.safetensors"))
    report = dict(
        source_revision=REVISION,
        archive_sha256=ARCHIVE_SHA,
        ort=onnxruntime.__version__,
        providers=session.get_providers(),
        cases=reports,
    )
    (args.output / "reference.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            dict(
                cases=len(reports),
                gold_correct=sum(c["prediction"] == c["gold"] for c in reports),
                source_revision=REVISION,
                labels=len(conv.labels),
                chars=len(conv.chars),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
