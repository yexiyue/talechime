"""Compare locally generated float32 WAVs without changing acceptance thresholds."""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def read_float_wav(path):
    contents = path.read_bytes()
    if contents[:4] != b"RIFF" or contents[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF WAV: {path}")
    offset = 12
    fmt = None
    data = None
    while offset + 8 <= len(contents):
        kind, size = struct.unpack_from("<4sI", contents, offset)
        start = offset + 8
        payload = contents[start:start + size]
        if len(payload) != size:
            raise ValueError(f"Truncated WAV: {path}")
        if kind == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", payload)
            if fmt[0] == 0xFFFE and len(payload) >= 40:
                # WAVE_FORMAT_EXTENSIBLE stores the encoding in its subformat GUID.
                encoding = struct.unpack_from("<I", payload, 24)[0]
                fmt = (encoding, *fmt[1:])
        elif kind == b"data":
            data = payload
        offset = start + size + size % 2
    if fmt is None or fmt[0] != 3 or fmt[-1] != 32 or data is None:
        raise ValueError(f"Expected IEEE float32 WAV: {path}")
    samples = array.array("f")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    if not all(math.isfinite(sample) for sample in samples):
        raise ValueError(f"Nonfinite PCM: {path}")
    return fmt, samples


def compare(baseline, candidate):
    baseline_files = {p.relative_to(baseline): p for p in baseline.rglob("*.wav")}
    candidate_files = {p.relative_to(candidate): p for p in candidate.rglob("*.wav")}
    if not baseline_files or baseline_files.keys() != candidate_files.keys():
        raise ValueError("WAV file sets are empty or differ")
    rows = []
    for name, first in sorted(baseline_files.items()):
        second = candidate_files[name]
        same = hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
        fmt_a, a = read_float_wav(first)
        fmt_b, b = read_float_wav(second)
        if not a or not b:
            raise ValueError(f"Expected nonempty PCM: {name}")
        row = {"file": name.as_posix(), "identical": same,
               "samples_a": len(a), "samples_b": len(b), "format_equal": fmt_a == fmt_b}
        if not same:
            if len(a) == len(b) and fmt_a == fmt_b:
                error = signal = total_abs = max_abs = 0.0
                for first_sample, second_sample in zip(a, b):
                    difference = first_sample - second_sample
                    absolute = abs(difference)
                    error += difference * difference
                    signal += first_sample * first_sample
                    total_abs += absolute
                    max_abs = max(max_abs, absolute)
                row.update(max_abs=max_abs,
                           mean_abs=total_abs / max(1, len(a)),
                           rms=math.sqrt(error / max(1, len(a))),
                           snr_db=10 * math.log10(signal / error) if signal > 0 and error > 0 else None)
        rows.append(row)
    return {"files": len(rows), "identical": sum(row["identical"] for row in rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("files", "identical")}))


if __name__ == "__main__":
    main()
