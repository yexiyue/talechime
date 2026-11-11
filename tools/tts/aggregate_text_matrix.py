"""Combine ordered complete reports without treating generation as acceptance."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import run_text_matrix as matrix


def aggregate(sources, variants, cases, output):
    matrix.validate([], variants, cases)
    plan = matrix.make_plan(variants, cases)
    planned = {row["id"]: row for row in plan}
    texts = {case["id"]: case["text"] for case in cases}
    selected = {}
    receipts = []
    for source in sources:
        source = source.resolve(strict=True)
        source_plan = matrix.read_json(source / "plan.json")
        rows = [json.loads(line) for line in (source / "results.jsonl").read_text(encoding="utf-8").splitlines()]
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != {row["id"] for row in source_plan}:
            raise ValueError("only complete source reports with unique rows can be aggregated")
        receipts.append({"source": str(source), "results_sha256": matrix.digest(source / "results.jsonl")})
        for row in rows:
            if row["id"] not in planned:
                raise ValueError("source contains a case outside the declared aggregate plan")
            text = texts[planned[row["id"]]["case"]]
            expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if row["text"] != text or row["source_sha256"] != expected_hash:
                raise ValueError("source text does not match the aggregate corpus")
            previous = selected.get(row["id"])
            record = dict(row, evidence_source=str(source))
            if previous and previous.get("listening") == "failed":
                record["listening"] = "failed"
                record["known_failure"] = previous.get("known_failure", "prior listening failure")
            audio = source / row.get("audio_path", row["id"] + "/audio.wav")
            record["audio_path"] = os.path.relpath(audio.resolve(), output.resolve())
            if record["generation"] == "generated":
                try:
                    if audio.name != "audio.wav":
                        raise ValueError("unexpected audio evidence filename")
                    evidence = matrix.generation_evidence(audio.parent, {"text": text}, planned[row["id"]])
                    if row.get("pcm", {}).get("sha256") != evidence["pcm"]["sha256"]:
                        raise ValueError("audio changed since the source receipt")
                    record.update(evidence)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    record.update(generation="failed", reason=str(error))
            selected[row["id"]] = record
    if set(selected) != set(planned):
        raise ValueError("aggregate has missing planned rows")
    records = []
    by_variant = {row["id"]: row for row in variants}
    for expected in plan:
        record = selected[expected["id"]]
        expected["params"] = record.get("params", {})
        failure = by_variant[expected["variant"]].get("known_failures", {}).get(expected["case"])
        if failure:
            record.update(listening="failed", known_failure=failure)
        records.append(record)
    output.mkdir(parents=True, exist_ok=False)
    matrix.write_json(output / "plan.json", plan)
    matrix.write_json(output / "sources.json", receipts)
    with (output / "results.jsonl").open("x", encoding="utf-8") as log:
        for row in records:
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
    return matrix.report(output, plan, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new aggregate directory")
    parser.add_argument("sources", type=Path, nargs="+", help="complete reports, later sources take precedence")
    parser.add_argument("--catalog", type=Path, default=matrix.HERE / "backend-variants.json")
    parser.add_argument("--cases", type=Path, default=matrix.HERE / "text-matrix.json")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.sources, matrix.read_json(args.catalog), matrix.read_json(args.cases),
                               args.output.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
