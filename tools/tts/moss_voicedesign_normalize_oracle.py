"""Execute the fixed official robustness normalizer; no model/runtime dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import runpy
import unicodedata

SOURCE_SHA = 'b7c1fab4fd7bdc8c779df99f6af310fd8cd0746b759db684bf147e0c182b3183'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if unicodedata.unidata_version != '13.0.0':
        raise ValueError('fixture requires the pinned Unicode 13.0.0 database')
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('fixed official normalizer checksum mismatch')
    namespace = runpy.run_path(str(args.source))
    normalize = namespace['normalize_tts_text']
    cases = [{'id':'upstream-'+name, 'text':text} for name,text,_ in namespace['TEST_CASES']]
    cases += [{'id':'matrix-'+case['id'], 'text':case['text']} for case in
              json.loads(Path(__file__).with_name('text-matrix.json').read_text())]
    cases += [{'id':'unicode-controls', 'text':'甲\u200b乙\u0378\ue000\x00\tHello\r\n世界'},
              {'id':'unicode-line-separators', 'text':'甲\u2028乙\u2029Hello\x85world'},
              {'id':'unicode13-unassigned', 'text':'甲\U0001fae8乙'},
              {'id':'protected-adjacency', 'text':'甲https://example.com/path?x=1乙和foo@example.com。'}]
    for case in cases:
        case['normalized'] = normalize(case['text'])
    with args.output.open('x', encoding='utf-8') as output:
        json.dump({'source_sha256':SOURCE_SHA, 'upstream_revision':'934d6826b084c46a0d033402174d5f8ac4ed2519',
                   'python':platform.python_version(), 'unicode':unicodedata.unidata_version,
                   'cases':cases}, output, ensure_ascii=False, indent=2)
        output.write('\n')

if __name__ == '__main__':
    main()
