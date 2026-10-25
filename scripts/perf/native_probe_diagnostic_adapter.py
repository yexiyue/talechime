"""Translate an xtask reference request for a frozen native diagnostic probe.

Rust xtask owns repetitions, process containment, deadlines and evidence. This
bridge neither implements inference nor labels its output a production baseline.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    envelope = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output = Path(sys.argv[2])
    config = envelope["request"]
    for name, expected in config["required_environment"].items():
        assert os.environ.get(name) == expected, f"diagnostic environment differs: {name}"
    native = dict(config["native_request"])
    native.update(schema="talechime/performance-request/1", attempt_id=envelope["attempt_id"],
                  warmup_rounds=envelope["warmup_rounds"], hot_rounds=envelope["hot_rounds"],
                  observation_mode="off")
    request_file = output / "native-request.json"
    with request_file.open("x", encoding="utf-8") as stream:
        json.dump(native, stream)
    # Inherit the event stream unchanged, and remain in xtask's owned process job.
    subprocess.run([config["native_probe"], str(request_file), str(output)], check=True)


if __name__ == "__main__":
    main()
