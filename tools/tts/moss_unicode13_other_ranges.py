#!/usr/bin/env python3
"""Generate Unicode 13 General_Category C ranges and cross-check UnicodeData.txt.

Requires Python 3.9's Unicode 13 database and the official Unicode 13 file.
Use --check to verify the checked-in output without writing it.
"""
import argparse
import hashlib
from pathlib import Path
import unicodedata

SOURCE_SHA256 = 'bdbffbbfc8ad4d3a6d01b5891510458f3d36f7170422af4ea2bed3211a73e8bb'
LIMIT = 0x110000
OUTPUT = Path(__file__).resolve().parents[2] / 'crates/moss-tts/src/prompt/normalize/unicode13.rs'


def official_other(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('UnicodeData.txt does not match the fixed Unicode 13 digest')
    # All unassigned code points have the default General_Category Cn.
    other = bytearray(b'\1') * LIMIT
    first = None
    for line in data.decode('utf-8').splitlines():
        fields = line.split(';')
        code, name, category = int(fields[0], 16), fields[1], fields[2]
        if name.endswith(', First>'):
            if first is not None:
                raise ValueError('nested UnicodeData range')
            first = (code, category, name[:-8])
        elif name.endswith(', Last>'):
            if first is None or first[1] != category or first[2] != name[:-7]:
                raise ValueError('mismatched UnicodeData range')
            start = first[0]
            other[start:code + 1] = bytes([category.startswith('C')]) * (code - start + 1)
            first = None
        else:
            if first is not None:
                raise ValueError('unterminated UnicodeData range')
            other[code] = category.startswith('C')
    if first is not None:
        raise ValueError('unterminated UnicodeData range')
    return other


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unicode_data', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if unicodedata.unidata_version != '13.0.0':
        raise ValueError('requires a Python runtime with Unicode database 13.0.0')
    official = official_other(args.unicode_data)
    python = bytearray(unicodedata.category(chr(code)).startswith('C') for code in range(LIMIT))
    if python != official:
        code = next(i for i in range(LIMIT) if python[i] != official[i])
        raise ValueError(f'official UnicodeData differs from Python at U+{code:04X}')
    ranges = []
    start = None
    for code in range(LIMIT + 1):
        present = code < LIMIT and python[code]
        if present and start is None:
            start = code
        elif not present and start is not None:
            ranges.append((start, code - 1))
            start = None
    lines = ['// Generated from Unicode 13.0.0; see assets/unicode-13/LICENSE.txt.',
             '// Regenerate with tools/tts/moss_unicode13_other_ranges.py.',
             'pub(super) const OTHER_RANGES: &[(u32, u32)] = &[']
    lines += [f'    (0x{start:06X}, 0x{end:06X}),' for start, end in ranges]
    content = '\n'.join(lines + ['];', ''])
    if args.check:
        if OUTPUT.read_text(encoding='utf-8') != content:
            raise ValueError('checked-in Unicode ranges differ')
    else:
        OUTPUT.write_text(content, encoding='utf-8')
    print(f'verified {LIMIT} code points against official UnicodeData; {len(ranges)} ranges')


if __name__ == '__main__':
    main()
