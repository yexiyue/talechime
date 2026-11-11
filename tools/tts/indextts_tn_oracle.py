"""Development-only oracle from the fixed official TextNormalizer class AST."""
import argparse
import ast
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import traceback
import warnings

UPSTREAM_SHA256 = 'df62a817c3e1b540ed62645850278df0463f934c562c1f3f50b85c2c05fc543d'
CASES = [
    '价格是12.5元，2026年11月11日。', '奔驰S600的速度为每小时500公里。',
    "It's $12.50 on 11/11/2026.", '今天是 2026 年，明年是2027年。',
    '增长5%，下降-2.5%，温度-3℃。', '上午08:30，下午14:05。',
    '2003年1月2日，2026-11-11。', 'GPT-5-nano和Wi-Fi 6，3.5mm接口。',
    '他走进银<行|HANG2>，读作ju4、que4、xun4。',
    '约瑟夫·高登-莱维特与约瑟夫·高登2人。', '请发邮件到alice@example.com。',
    'Hello, Alice. 2026 is a number.', 'The price is $1,234.56, up 5%.',
    '3rd place, 21st century, 11/11/2026.', 'Phone: +86 13800138000.',
    '00123 1000000000000000000000000000000', '全角２０２６年、１９．９９元。',
    '  hello  world  ', '他\n说hello，今天2026年。', '<going|G OW1 . IH0 NG> home.',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.upstream.read_bytes()
    if hashlib.sha256(source).hexdigest() != UPSTREAM_SHA256:
        raise ValueError('official front.py identity mismatch')
    tree = ast.parse(source)
    # Execute the unchanged official class without unrelated tokenizer imports.
    selected = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TextNormalizer')
    scope = dict(re=re, os=os, traceback=traceback, warnings=warnings, lru_cache=lru_cache, __file__=str(args.upstream))
    exec(compile(ast.Module(body=[selected], type_ignores=[]), str(args.upstream), 'exec'), scope)
    frontend = scope['TextNormalizer']()
    frontend.load()
    rows = []
    for text in CASES:
        cleaned = frontend.clean_pattern.sub(lambda match: frontend.char_rep_map[match.group()], text)
        output = frontend.normalize(cleaned)
        assert frontend.normalize(cleaned) == output
        rows.append(dict(text=text, cleaned=cleaned, chinese=frontend.use_chinese(cleaned), output=output))
    result = dict(upstream_revision='d9e41aac89fd00b3d71497fddb287b7f24613712', upstream_sha256=UPSTREAM_SHA256,
                  platform=platform.system(), cases=rows)
    for package in ('wetext', 'WeTextProcessing', 'pynini', 'kaldifst'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
