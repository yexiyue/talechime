"""Reverify immutable matrix audio into a new report, retaining prior conclusions."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import run_text_matrix as matrix


def verify(source, output):
    source = source.resolve(strict=True)
    plan = matrix.read_json(source / 'plan.json')
    rows = [json.loads(line) for line in (source / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != {row['id'] for row in plan}:
        raise ValueError('source matrix is incomplete or has duplicate rows; wait for execution to finish')
    output.mkdir(parents=True, exist_ok=False)
    matrix.write_json(output / 'plan.json', plan)
    matrix.write_json(output / 'verification-source.json', {
        'results_sha256': matrix.digest(source / 'results.jsonl'),
        'source': str(source), 'verifier_sha256': matrix.digest(Path(matrix.__file__)),
    })
    records = []
    planned_by_id = {row['id']: row for row in plan}
    with (output / 'results.jsonl').open('x', encoding='utf-8') as log:
        for row in rows:
            updated = dict(row)
            if row.get('exit_code') == 0 and row.get('process_status') == 'exited':
                # Validate identifiers before resolving external report paths.
                variant, case = row['id'].split('/')
                if not all(matrix.IDENTIFIER.fullmatch(value) for value in (variant, case)):
                    raise ValueError('unsafe result identifier')
                updated['prior_verification'] = {'generation': row['generation'], 'reason': row.get('reason')}
                try:
                    planned = planned_by_id[row['id']]
                    text_hash = hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
                    if text_hash != planned['source_sha256'] or text_hash != row['source_sha256']:
                        raise ValueError('source text does not match the frozen plan')
                    audio = source / row.get('audio_path', row['id'] + '/audio.wav')
                    if audio.name != 'audio.wav':
                        raise ValueError('unexpected audio evidence filename')
                    directory = audio.parent
                    evidence = matrix.generation_evidence(directory, {'text': row['text']},
                                                         {'model_id': planned.get('model_id')})
                    if row.get('pcm', {}).get('sha256') != evidence['pcm']['sha256']:
                        raise ValueError('audio changed since the source receipt')
                    updated.update(evidence, generation='generated', audio_path=os.path.relpath(audio.resolve(), output.resolve()))
                    updated.pop('reason', None)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    updated.update(generation='failed', reason=str(error))
            records.append(updated)
            log.write(json.dumps(updated, ensure_ascii=False) + '\n')
    return matrix.report(output, plan, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path, help='new verification report directory')
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.output.resolve()), ensure_ascii=False))


if __name__ == '__main__':
    main()
