"""Publish the library dependency graph, safely resuming the same release commit."""
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tomllib
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "talechime-protocol",
    "tts-candle-platform",
    "moss-tts",
    "qwen3-tts",
    "qwen3-asr",
    "voxcpm",
    "omnivoice",
    "talechime-core",
    "talechime-backends",
    "talechime",
)


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "talechime-library-release"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for directory in PACKAGES:
        manifest = tomllib.loads((ROOT / "crates" / directory / "Cargo.toml").read_text())
        name, version = manifest["package"]["name"], manifest["package"]["version"]
        try:
            published = json.loads(get(f"https://crates.io/api/v1/crates/{name}/{version}"))
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        else:
            if published["version"]["yanked"]:
                raise RuntimeError(f"{name}@{version} is yanked")
            archive = get(f"https://static.crates.io/crates/{name}/{name}-{version}.crate")
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
                info = package.extractfile(f"{name}-{version}/.cargo_vcs_info.json")
                if info is None or json.load(info)["git"]["sha1"] != commit:
                    raise RuntimeError(f"{name}@{version} belongs to another source commit")
            print(f"Already published from this commit: {name}@{version}", flush=True)
            continue
        print(f"Publishing {name}@{version}", flush=True)
        # Cargo verifies the packaged source and waits for registry availability.
        # An uncertain upload stops this run; rerun verifies the original version.
        subprocess.run(["cargo", "publish", "--locked", "-p", name], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
