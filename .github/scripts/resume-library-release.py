"""Honor explicit registry rejection deadlines without changing release identity."""
from collections import deque
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import re
import subprocess
import sys
import time


def retry_deadline(output):
    if "status 429 Too Many Requests" not in output or "too many new crates" not in output:
        return None
    match = re.search(r"Please try again after ([^\n]+? GMT)", output)
    return parsedate_to_datetime(match[1]) if match else None


def main():
    limit = time.monotonic() + 3600
    for _ in range(10):
        tail = deque(maxlen=80)
        with subprocess.Popen(
            [sys.executable, ".github/scripts/publish-libraries.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                tail.append(line)
            code = process.wait()
        if code == 0:
            return
        deadline = retry_deadline("".join(tail))
        if deadline is None:
            raise SystemExit(code)
        delay = max(1, math.ceil((deadline - datetime.now(timezone.utc)).total_seconds()) + 2)
        if time.monotonic() + delay >= limit:
            raise SystemExit("Registry deadline exceeds the one-hour recovery budget")
        print(f"Registry explicitly rejected a new crate; resume after {deadline.isoformat()}", flush=True)
        time.sleep(delay)
    raise SystemExit("Registry recovery attempt limit reached")


if __name__ == "__main__":
    main()
