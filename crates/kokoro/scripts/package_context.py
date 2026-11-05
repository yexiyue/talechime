"""Package verified g2pW runtime tables; the model stays at its official source.

Development only. Does not upload, download or access credentials.
"""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import shutil


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--validation-log', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output must be a new directory')
    data = json.loads((args.export / 'context.json').read_text())
    if data['format'] != 'kokoro-g2pw-v2' or data['source_revision'] != 'b9328ce8129c4717a70f16a2688542a0197541dd' or data['tokenizer_revision'] != '8f23c25b06e129b6c986331a13d8d025a92cf0ea':
        raise ValueError('unrecognized context data source')
    if data['model_sha256'] != sha(args.export / 'g2pw.onnx') or data['vocab_sha256'] != sha(args.export / 'vocab.txt'):
        raise ValueError('model/tokenizer digest mismatch')
    if any(item['path'].startswith('/') for item in data['source'].values()):
        raise ValueError('provenance must use relative module paths')
    if 'official input/output parity: 24/24; manual target gold: 23/24' not in args.validation_log.read_text():
        raise ValueError('context_compare has not passed complete official parity')
    args.output.mkdir(parents=True)
    names = ['context.json', 'G2PW-LICENSE', 'BERT-SOURCE.txt', 'reference.json']
    for name in names:
        shutil.copyfile(args.export / name, args.output / name)
    (args.output / 'NOTICE.md').write_text('''# Kokoro optional g2pW context tables v1

Exported from GitYCC/g2pW revision b9328ce8129c4717a70f16a2688542a0197541dd.
https://github.com/GitYCC/g2pW/tree/b9328ce8129c4717a70f16a2688542a0197541dd
Apache-2.0; see G2PW-LICENSE. Data representation changed by TaleChime.

The acoustic engine remains Candle. The optional g2pW pronunciation model uses
ORT CPU, obtained separately from the fixed official YC-Chen/g2pW archive;
the BERT vocabulary comes from fixed google-bert/bert-base-chinese.
No ONNX model or BERT weights are republished in this table package.

All 24 independent official input/output fixtures match the Rust runtime,
including tokenizer inputs, masks and probabilities. Manual target gold is
23/24 for both official and native execution: this is an upstream model
limitation, not a guarantee of correct pronunciation for every sentence.
''')
    manifest = {'format': 'talechime-kokoro-context-v1', 'source_revision': data['source_revision'],
                'archive_sha256': data['archive_sha256'], 'model_sha256': data['model_sha256'],
                'tokenizer_revision': data['tokenizer_revision'], 'vocab_sha256': data['vocab_sha256'],
                'generation': {name: sha(Path(__file__).parent / name) for name in ['export_context.py', 'package_context.py']},
                'validation': {'official_parity': 24, 'cases': 24, 'manual_gold': 23},
                'files': {name: {'bytes': (args.output / name).stat().st_size, 'sha256': sha(args.output / name)} for name in [*names, 'NOTICE.md']}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'files': len(manifest['files'])+1, 'bytes': sum(f.stat().st_size for f in args.output.iterdir())}))


if __name__ == '__main__':
    main()
