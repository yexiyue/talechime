"""Summarize raw offline CPU probe logs without discarding contaminated slots."""
import argparse
import collections
import hashlib
import json
import pathlib
import statistics
import struct


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    processes = []
    for path in sorted(args.root.glob("*.receipt.json")):
        receipt = json.loads(path.read_text())
        log = pathlib.Path(str(path).removesuffix(".receipt.json") + ".out")
        groups = collections.defaultdict(list)
        details = []
        for line in log.read_text().splitlines():
            if "CPU_TRACE " not in line:
                continue
            row = json.loads(line.split("CPU_TRACE ", 1)[1])
            if "microseconds" in row:
                groups[row["case"], row["op"]].append(row["microseconds"])
                if "samples" in row or "output_sha256" in row:
                    details.append(row)
            else:
                details.append(row)
        processes.append(dict(name=path.name.removesuffix(".receipt.json"),
                              returncode=receipt["returncode"],
                              compiler_observations=receipt["activity"],
                              peak_rss_kib=receipt["peak_rss_kib"],
                              timings=[dict(case=case, op=op, microseconds=values,
                                            mean_us=statistics.mean(values))
                                       for (case, op), values in sorted(groups.items())],
                              details=details, stdout_sha256=sha(log.read_bytes())))
    archives = {}
    for path in sorted(args.root.glob("*.safetensors")):
        data = path.read_bytes()
        header_len = struct.unpack_from("<Q", data)[0]
        header = json.loads(data[8:8+header_len])
        payload = memoryview(data)[8+header_len:]
        tensors = {}
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            lo, hi = metadata["data_offsets"]
            tensors[name] = dict(shape=metadata["shape"], dtype=metadata["dtype"],
                                 sha256=sha(payload[lo:hi]))
        archives[path.name] = dict(sha256=sha(data), tensors=tensors,
                                  layout=json.loads(path.with_suffix(".layout.json").read_text()))
    summary = dict(manifest=json.loads((args.root / "manifest.json").read_text()),
                   e2e=json.loads((args.root / "e2e-summary.json").read_text()),
                   processes=processes, archives=archives,
                   code_fixtures={path.name:dict(sha256=sha(path.read_bytes()),
                       frames=json.loads(path.read_text())["frames"])
                       for path in sorted((args.root / "codes").glob("*.json"))},
                   parity=json.loads((args.root / "parity-summary.json").read_text()))
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved {len(processes)} process receipts and {len(archives)} tensor archives")


if __name__ == "__main__":
    main()
