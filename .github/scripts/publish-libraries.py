"""Publish changed libraries while preserving verified, unchanged versions."""
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tomllib
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "talechime-protocol",
    "moss-tts",
    "qwen3-tts",
    "qwen3-asr",
    "voxcpm",
    "omnivoice",
    "cosyvoice3",
    "indextts",
    "talechime-core",
    "talechime-backends",
    "talechime",
)


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "talechime-library-release"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def verify_published_source(directory, commit, archive, name, version, root=ROOT):
    """Accept the same commit or an unchanged package from a verified ancestor."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
        info = package.extractfile(f"{name}-{version}/.cargo_vcs_info.json")
        if info is None:
            raise RuntimeError(f"{name}@{version} has no source identity")
        identity = json.load(info)
    published_commit = identity.get("git", {}).get("sha1", "")
    package_path = f"crates/{directory}"
    if (
        not re.fullmatch(r"[0-9a-f]{40}", published_commit)
        or identity.get("git", {}).get("dirty", False)
        or identity.get("path_in_vcs") != package_path
    ):
        raise RuntimeError(f"{name}@{version} has an invalid source identity")
    if published_commit == commit:
        return "this commit"
    # Full checkout history must prove the source belongs to this release lineage.
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", published_commit, commit],
        cwd=root, capture_output=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(f"{name}@{version} source commit is not a verified ancestor")
    changed = subprocess.run(
        ["git", "diff", "--quiet", published_commit, commit, "--", package_path, "Cargo.toml"],
        cwd=root, capture_output=True,
    )
    if changed.returncode != 0:
        raise RuntimeError(f"{name}@{version} source changed; increment the package version")
    return f"unchanged ancestor {published_commit}"


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
            source = verify_published_source(directory, commit, archive, name, version)
            print(f"Already published from {source}: {name}@{version}", flush=True)
            continue
        print(f"Publishing {name}@{version}", flush=True)
        # Cargo verifies the packaged source and waits for registry availability.
        # An uncertain upload stops this run; rerun verifies the original version.
        subprocess.run(["cargo", "publish", "--locked", "-p", name], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
