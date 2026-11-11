"""Attach independently executed frontend/condition evidence; preserve audio/listening facts."""
import argparse
import hashlib
import json
from pathlib import Path

import run_text_matrix as matrix
import verify_text_matrix


def fixture_path(relative):
    path = (matrix.ROOT / relative).resolve(strict=True)
    if not path.is_relative_to(matrix.ROOT) or path.suffix != '.json':
        raise ValueError('official evidence must name a repository JSON fixture')
    return path


def evidence_index(receipt, cases):
    texts = {case['text']: case for case in cases}
    evidence = {}
    for check in receipt['checks']:
        if check['exit_code'] != 0:
            raise ValueError('official check did not pass: ' + check['id'])
        log = Path(check['log'])
        if matrix.digest(log) != check['log_sha256']:
            raise ValueError('official check log changed')
        snapshots = []
        for item in check['fixtures']:
            path = fixture_path(item['path'])
            if matrix.digest(path) != item['sha256']:
                raise ValueError('official fixture changed since its test receipt')
            snapshots.append(dict(path=item['path'], sha256=item['sha256']))
        for item in check['fixtures']:
            if not item.get('coverage'):
                continue
            value = matrix.read_json(fixture_path(item['path']))
            for entry in value['cases']:
                if any(entry.get(key) != wanted for key, wanted in item.get('selector', {}).items()):
                    continue
                text = entry.get('source_text', entry.get('text'))
                if not isinstance(text, str) or text not in texts:
                    continue
                case = texts[text]
                digest = hashlib.sha256(text.encode()).hexdigest()
                if entry.get('source_sha256', digest) != digest:
                    raise ValueError('official fixture source hash mismatch')
                case_id = entry.get('source_id', entry.get('id', case['id']))
                if case_id != case['id']:
                    raise ValueError('official fixture case identity mismatch')
                status = check['kind']
                if entry.get('expected_rejection'):
                    status = 'scope_declared'
                if 'classification' in entry:
                    status = {'matched':'phonemes_matched', 'difference':'difference',
                              'unsupported_rejection':'policy_rejection'}[entry['classification']]
                for variant in item.get('variants', check['variants']):
                    key = variant + '/' + case['id']
                    proof = evidence.setdefault(key, dict(status=status, checks=[]))
                    if proof['status'] != status:
                        raise ValueError('conflicting official evidence for ' + key)
                    record = dict(check=check['id'], kind=check['kind'], fixtures=snapshots,
                                  source_sha256=digest, scope=check['scope'])
                    if record not in proof['checks']:
                        proof['checks'].append(record)
    return evidence


def annotate(source, output, receipt_path):
    receipt = matrix.read_json(receipt_path)
    cases = matrix.read_json(Path(__file__).with_name('text-matrix.json'))
    evidence = evidence_index(receipt, cases)
    # Recheck original WAV hashes, actual loaded model and source/PCM coverage.
    verify_text_matrix.verify(source, output)
    plan = matrix.read_json(output / 'plan.json')
    rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
    for row in rows:
        proof = evidence.get(row['id'])
        if proof:
            row.update(official=proof['status'], official_evidence=proof['checks'])
        else:
            row.update(official='pending', official_evidence=[])
    matrix.write_json(output / 'official-receipt.json', dict(
        source=str(receipt_path.resolve()), sha256=matrix.digest(receipt_path), checks=receipt['checks']))
    with (output / 'results.jsonl').open('w', encoding='utf-8') as log:
        for row in rows:
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
    return matrix.report(output, plan, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(annotate(args.source.resolve(), args.output.resolve(), args.receipt), ensure_ascii=False))


if __name__ == '__main__':main()
