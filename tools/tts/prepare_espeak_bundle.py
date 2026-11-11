"""Build a separate pinned GPL eSpeak program/data/source bundle; never link Rust."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

COMMIT = "4870adfa25b1a32b4361592f1be8a40337c58d6c"
SOURCE_SHA256 = "bb4338102ff3b49a81423da8a1a158b420124b055b60fa76cfb4b18677130a23"
SOURCE_URL = "https://mirrors.tuna.tsinghua.edu.cn/gentoo/distfiles/ba/espeak-ng-1.52.0.tar.gz"
NAME = "espeak-ng-1.52.0"
PROTOCOL = "talechime-espeak-text-to-phonemes-v1"
WRAPPER = "kokoro-espeak-phonemize.exe" if os.name == "nt" else "kokoro-espeak-phonemize"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def extract(archive, destination):
    links = []
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            path = destination / member.name
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError("unsafe source archive path")
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as data, path.open("xb") as output:
                    shutil.copyfileobj(data, output)
                path.chmod(member.mode & 0o777)
            elif member.issym() and member.name == NAME + "/src/include/espeak/speak_lib.h" and member.linkname == "../espeak-ng/speak_lib.h":
                links.append(path)
            else:
                raise ValueError("unexpected source archive entry")
    # Materialize the sole known header alias so Windows needs no symlink privilege.
    for path in links:
        shutil.copyfile(path.parent / "../espeak-ng/speak_lib.h", path)


def verify_bundle(output):
    if output.is_symlink() or not output.is_dir():
        raise ValueError("invalid bundle root")
    manifest_path = output / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file() or manifest_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("invalid bundle manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "talechime-espeak-bundle-v1"
            or manifest.get("source_commit") != COMMIT or manifest.get("source_sha256") != SOURCE_SHA256
            or manifest.get("platform") != platform.system() or manifest.get("machine") != platform.machine()
            or manifest.get("phonemizer_protocol") != PROTOCOL):
        raise ValueError("bundle source or native platform identity mismatch")
    seen = set()
    for entry in manifest["files"]:
        name = entry["path"]
        relative = PurePosixPath(name)
        if (not name or "\\" in name or relative.is_absolute()
                or any(part in (".", "..") for part in relative.parts) or name in seen
                or len(entry["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in entry["sha256"])):
            raise ValueError("invalid bundle entry")
        seen.add(name)
        path = output
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("invalid bundle entry: symbolic link")
        if path.stat().st_size != entry["size"] or digest(path) != entry["sha256"]:
            raise ValueError("bundle checksum mismatch: " + entry["path"])
    required = {"espeak-ng.exe" if os.name == "nt" else "espeak-ng", WRAPPER,
                "espeak-ng-data/en_dict", "espeak-ng-data/intonations", "espeak-ng-data/lang/gmw/en",
                "espeak-ng-data/lang/gmw/en-US", "espeak-ng-data/phondata", "espeak-ng-data/phonindex",
                "espeak-ng-data/phontab", "source/espeak-ng-1.52.0.tar.gz", "source/COPYING",
                "source/kokoro_espeak_phonemize.c"}
    if not required <= seen:
        raise ValueError("incomplete standalone helper bundle")
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("bundle contains a symbolic link")
        if path.is_file() and path != manifest_path and path.relative_to(output).as_posix() not in seen:
            raise ValueError("bundle contains an unlisted file")
        if not path.is_file() and not path.is_dir():
            raise ValueError("bundle contains an unsupported file")
    if digest(output / "source/espeak-ng-1.52.0.tar.gz") != SOURCE_SHA256:
        raise ValueError("bundled source checksum does not match the fixed upstream")
    return manifest


def prepare(archive, output, work):
    if digest(archive) != SOURCE_SHA256:
        raise ValueError("fixed eSpeak source checksum mismatch")
    if output.exists() or work.exists():
        raise ValueError("bundle output and work directory must be new")
    work.mkdir(parents=True)
    extract(archive, work)
    source = work / NAME
    dependencies = source / "cmake/deps.cmake"
    original = dependencies.read_text(encoding="utf-8")
    marker = "else()\n  FetchContent_Declare(sonic-git"
    if original.count(marker) != 1:
        raise ValueError("fixed eSpeak dependency configuration changed")
    dependencies.write_text(original.replace(marker, "elseif(USE_LIBSONIC)\n  FetchContent_Declare(sonic-git"), encoding="utf-8")
    wrapper_source = Path(__file__).with_name("kokoro_espeak_phonemize.c")
    shutil.copyfile(wrapper_source, source / "src/kokoro_espeak_phonemize.c")
    with (source / "src/CMakeLists.txt").open("a", encoding="utf-8") as cmake:
        cmake.write("\nadd_executable(kokoro-espeak-phonemize kokoro_espeak_phonemize.c)\n"
                    "target_link_libraries(kokoro-espeak-phonemize PRIVATE espeak-ng espeak-ng-config)\n"
                    "if(MINGW)\n  target_link_options(kokoro-espeak-phonemize PRIVATE -static-libstdc++ -static)\nendif()\n")
    build = work / "build"
    options = ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DUSE_LIBSONIC=OFF",
               "-DUSE_LIBPCAUDIO=OFF", "-DUSE_MBROLA=OFF", "-DUSE_SPEECHPLAYER=OFF", "-DBUILD_TESTING=OFF"]
    subprocess.run(["cmake", "-S", str(source), "-B", str(build), *options], check=True, timeout=180)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--parallel", "4"], check=True, timeout=900)
    program = "espeak-ng.exe" if os.name == "nt" else "espeak-ng"
    programs = [p for p in build.rglob(program) if p.is_file()]
    if len(programs) != 1:
        raise ValueError("expected exactly one standalone eSpeak executable")
    wrappers = [p for p in build.rglob(WRAPPER) if p.is_file()]
    if len(wrappers) != 1:
        raise ValueError("expected exactly one standalone phonemizer")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=NAME + "-", dir=output.parent))
    try:
        shutil.copy2(programs[0], staging / program)
        shutil.copy2(wrappers[0], staging / WRAPPER)
        shutil.copytree(build / "espeak-ng-data", staging / "espeak-ng-data")
        corresponding = staging / "source"
        corresponding.mkdir()
        shutil.copyfile(archive, corresponding / (NAME + ".tar.gz"))
        shutil.copyfile(Path(__file__), corresponding / "prepare_espeak_bundle.py")
        shutil.copyfile(wrapper_source, corresponding / "kokoro_espeak_phonemize.c")
        for name in ["COPYING", "COPYING.APACHE", "COPYING.BSD2", "COPYING.UCD"]:
            shutil.copyfile(source / name, corresponding / name)
        instructions = (
            "eSpeak NG 1.52.0, GPL-3.0-or-later; independent program, not linked into TaleChime.\n"
            f"Upstream commit: {COMMIT}\nSource SHA-256: {SOURCE_SHA256}\n"
            "The corresponding source archive and complete build script accompany this bundle.\n"
            "The script disables unused Sonic fetching, materializes one header alias and builds the separately licensed phonemizer; no phonetic rules are changed.\n"
            "Rebuild: python prepare_espeak_bundle.py --archive espeak-ng-1.52.0.tar.gz --output NEW_BUNDLE --work NEW_WORK\n"
            "Requires CMake and a native C/C++ compiler; dependencies are not fetched during configuration.\n"
        )
        (corresponding / "BUILD.txt").write_text(instructions, encoding="utf-8")
        files = [{"path": p.relative_to(staging).as_posix(), "size": p.stat().st_size, "sha256": digest(p)}
                 for p in sorted(staging.rglob("*")) if p.is_file()]
        manifest = {"schema": "talechime-espeak-bundle-v1", "source_commit": COMMIT,
                    "source_sha256": SOURCE_SHA256, "source_url": SOURCE_URL,
                    "platform": platform.system(), "machine": platform.machine(),
                    "phonemizer_protocol": PROTOCOL, "files": files}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        result = subprocess.run([str(staging / program), "-q", "-v", "en-us", "-b", "1", "--ipa", "--tie=^", "--stdin"],
                                input="zorbulate\n", capture_output=True, text=True, encoding="utf-8", timeout=10,
                                env=dict(os.environ, ESPEAK_DATA_PATH=str(staging)))
        if result.returncode != 0 or not result.stdout.strip():
            raise ValueError("standalone helper smoke failed")
        result = subprocess.run([str(staging / WRAPPER), f"--path={staging}"], input="de\n",
                                capture_output=True, text=True, encoding="utf-8", timeout=10)
        if result.returncode != 0 or result.stdout.strip() != "də":
            raise ValueError("standalone phonemizer smoke failed")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("target/talechime-helpers") / NAME)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        manifest = verify_bundle(args.output.resolve())
        print(json.dumps({"bundle": str(args.output.resolve()), "files": len(manifest["files"]), "verified": True}))
        return
    with tempfile.TemporaryDirectory(prefix="talechime-espeak-") as temporary:
        archive = args.archive
        if archive is None:
            archive = Path(temporary) / (NAME + ".tar.gz")
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as source, archive.open("xb") as output:
                shutil.copyfileobj(source, output)
        work = args.work or Path(temporary) / "work"
        manifest = prepare(archive.resolve(), args.output.resolve(), work.resolve())
        print(json.dumps({"bundle": str(args.output.resolve()), "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
