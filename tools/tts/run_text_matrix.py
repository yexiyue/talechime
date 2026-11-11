"""Run a predeclared public-text matrix. No downloads; generated != accepted."""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")
# Only these switches can affect a probe. Never persist arbitrary env/config.
ENV_KEYS = {"TALECHIME_KOKORO_MODEL", "MOSS_MODEL", "TALECHIME_ESPEAK_DIR"}
INHERIT_KEYS = {"PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SystemRoot", "WINDIR"}
CONFIG_KEYS = {"id", "backend", "directory", "resource_directories", "voice", "device", "params", "case_params", "style",
               "env", "reference_source"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def unique_ids(items):
    ids = [item["id"] for item in items]
    if any(not isinstance(value, str) or not IDENTIFIER.fullmatch(value) for value in ids):
        raise ValueError("invalid identifier")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate identifier")


def validate(configs, variants, cases):
    for group in (configs, variants, cases):
        unique_ids(group)
    by_id = {variant["id"]: variant for variant in variants}
    for variant in variants:
        for rejection in variant.get("expected_rejections", {}).values():
            if not isinstance(rejection, dict) or not rejection.get("reason") or not rejection.get("error_contains"):
                raise ValueError("expected rejection needs reason and exact error substring")
    for config in configs:
        if set(config) - CONFIG_KEYS:
            raise ValueError("unknown config keys")
        variant = by_id[config["id"]]
        if config["backend"] != variant["backend"]:
            raise ValueError("backend does not match variant")
        if config.get("device", "cpu") not in {"cpu", "metal", "cuda"}:
            raise ValueError("invalid device")
        if set(config.get("env", {})) - ENV_KEYS:
            raise ValueError("environment key is not allowed")
        if not all(isinstance(v, str) for v in config.get("env", {}).values()):
            raise ValueError("environment values must be strings")
        validate_params(config.get("params", {}))
        directories = config.get("resource_directories", [config["directory"]])
        if not isinstance(directories, list) or not directories or not all(isinstance(p, str) for p in directories):
            raise ValueError("resource_directories must be a nonempty list of model directories")
        root = Path(config["directory"]).resolve()
        for path in directories:
            if not Path(path).resolve().is_relative_to(root):
                raise ValueError("resource directory must belong to the configured model root")
        profiles = config.get("case_params", {})
        if not isinstance(profiles, dict) or set(profiles) - {case["id"] for case in cases}:
            raise ValueError("case_params must name declared cases")
        for params in profiles.values():
            validate_params(params)
    for case in cases:
        if not isinstance(case["text"], str) or not case["text"]:
            raise ValueError("case text must not be empty")
    return by_id


def validate_params(params):
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    for value in params.values():
        if not isinstance(value, (str, int, float, bool)) or isinstance(value, float) and not math.isfinite(value):
            raise ValueError("invalid protocol parameter")


def make_plan(variants, cases, configs=None):
    configs = configs or {}
    plan = []
    for variant in variants:
        config = configs.get(variant["id"], {})
        for case in cases:
            params = {**config.get("params", {}), **config.get("case_params", {}).get(case["id"], {})}
            plan.append({"id": f'{variant["id"]}/{case["id"]}', "variant": variant["id"],
                         "model_id": variant.get("model_id"), "case": case["id"], "params": params,
                         "source_sha256": hashlib.sha256(case["text"].encode()).hexdigest(),
                         "expected": variant.get("expected_rejections", {}).get(case["id"])})
    return plan


def stop_tree(process):
    # Kill the group even if its leader exited: children can still hold log handles.
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def invoke(command, env, directory, timeout):
    started = time.monotonic()
    # Files preserve partial output on timeout, crashes and cancellation; no PIPE deadlock.
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=os.name != "nt",
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
        try:
            code = process.wait(timeout=timeout)
            stop_tree(process)
            status = "exited"
        except subprocess.TimeoutExpired:
            status = "timeout"
            stop_tree(process)
            code = process.returncode
        except BaseException:
            stop_tree(process)
            raise
    return {"process_status": status, "exit_code": code, "elapsed_seconds": time.monotonic() - started}


def inspect_wav(path):
    """Validate the native probe's IEEE float WAV, including signal and finiteness."""
    with path.open("rb") as stream:
        if stream.read(4) != b"RIFF":
            raise ValueError("not a RIFF WAV")
        stream.read(4)
        if stream.read(4) != b"WAVE":
            raise ValueError("not a WAVE file")
        fmt = None
        data = None
        while header := stream.read(8):
            if len(header) != 8:
                raise ValueError("truncated chunk")
            kind, size = struct.unpack("<4sI", header)
            payload = stream.read(size)
            if len(payload) != size:
                raise ValueError("truncated payload")
            if size % 2:
                stream.read(1)
            if kind == b"fmt ":
                fmt = struct.unpack("<HHIIHH", payload[:16])
                # hound writes WAVE_FORMAT_EXTENSIBLE for multichannel float PCM.
                if fmt[0] == 0xFFFE:
                    float_guid = bytes.fromhex("0300000000001000800000aa00389b71")
                    if len(payload) < 40 or payload[24:40] != float_guid or struct.unpack_from("<H", payload, 18)[0] != 32:
                        raise ValueError("unsupported extensible WAV subtype")
                    fmt = (3, *fmt[1:])
            elif kind == b"data":
                data = payload
        if not fmt or data is None:
            raise ValueError("missing WAV chunks")
        encoding, channels, rate, _, block_align, bits = fmt
        if (encoding, bits) != (3, 32) or channels < 1 or rate < 1 or block_align != channels * 4:
            raise ValueError("expected native float32 WAV")
        if not data or len(data) % block_align:
            raise ValueError("empty or partial PCM")
        peak = 0.0
        for (sample,) in struct.iter_unpack("<f", data):
            if not math.isfinite(sample):
                raise ValueError("nonfinite PCM")
            peak = max(peak, abs(sample))
        if peak == 0:
            raise ValueError("silent PCM")
        return {"sha256": digest(path), "sample_rate": rate, "channels": channels,
                "samples": len(data) // 4, "seconds": len(data) / block_align / rate, "peak": peak}


def text_diagnostics(directory, case):
    events = []
    errors = []
    source = case["text"].encode("utf-8")
    prefix = "TALECHIME_TEXT_DIAGNOSTIC "
    for line in (directory / "stderr.log").read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(prefix):
            continue
        try:
            event = json.loads(line[len(prefix):])
            if not isinstance(event, dict) or not isinstance(event.get("phase"), str):
                raise ValueError("invalid text diagnostic shape")
            if event["phase"] == "backend_input":
                start, end = event["source_start"], event["source_end"]
                if not 0 <= start < end <= len(source) or source[start:end].decode("utf-8") != event["source_text"]:
                    raise ValueError("text diagnostic does not match source UTF-8 range")
            events.append(event)
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
    return {"status": "failed" if errors else "captured" if events else "missing", "events": events, "errors": errors}


def generation_evidence(directory, case, variant=None):
    metrics = None
    loaded_model = None
    for line in (directory / "stdout.log").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict):
            if "loaded_model" in candidate:
                loaded_model = candidate["loaded_model"]
            if candidate.get("stream_end") is True or candidate.get("api_completed") is True:
                metrics = candidate
    if not metrics:
        raise ValueError("missing stream completion receipt")
    if variant and variant.get("model_id") != loaded_model:
        raise ValueError("loaded model identity does not match planned variant")
    pcm = inspect_wav(directory / "audio.wav")
    segments = read_json(directory / "audio-segments" / "index.json")
    source = case["text"].encode("utf-8")
    previous_end = 0
    previous_sample = 0
    for segment in segments:
        start, end = segment["source_start"], segment["source_end"]
        if not 0 <= start < end <= len(source) or start < previous_end:
            raise ValueError("invalid source range")
        # Decoding validates UTF-8 boundaries independently of normalized execution text.
        source[start:end].decode("utf-8")
        if source[previous_end:start].decode("utf-8").strip():
            raise ValueError("segment index omits source text")
        previous_end = end
        if not previous_sample == segment["start_sample"] < segment["end_sample"] <= pcm["samples"]:
            raise ValueError("invalid PCM range")
        previous_sample = segment["end_sample"]
    if not segments:
        raise ValueError("missing segments")
    if source[previous_end:].decode("utf-8").strip():
        raise ValueError("segment index omits source suffix")
    if previous_sample != pcm["samples"]:
        raise ValueError("segment index omits PCM suffix")
    return {"metrics": metrics, "loaded_model": loaded_model, "pcm": pcm, "segments": segments}


def voice_identity(config, variant):
    selected = []
    for path in sorted(Path(config["directory"]).rglob("voice.json")):
        value = read_json(path)
        backend = {"moss-nano": "moss", "qwen-design": "qwen", "moss-design": "moss"}.get(config["backend"], config["backend"])
        if (value.get("id") == config["voice"] and value.get("backend") == backend
                and value.get("model") == variant.get("model_id")
                and (not variant.get("runtime_revision") or value.get("revision") == variant["runtime_revision"])):
            reference = path.with_name("reference.wav")
            selected.append({"record_sha256": digest(path), "reference_sha256": digest(reference),
                             "transcript": value["transcript"], "model": value["model"],
                             "revision": value["revision"]})
    return selected


def rejection_evidence(directory, expected):
    """Only a declared model error can establish an expected rejection."""
    if not expected:
        return None
    error = (directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
    needle = expected["error_contains"]
    if needle in error:
        return {"reason": expected["reason"], "matched_error": needle}
    return None


def probe_identity(probe):
    env = {key: value for key, value in os.environ.items() if key in INHERIT_KEYS}
    result = subprocess.run([str(probe), "--identity"], env=env, capture_output=True,
                            text=True, timeout=10, check=True)
    identity = json.loads(result.stdout)
    if not isinstance(identity.get("features"), list) or not isinstance(identity.get("package_version"), str):
        raise ValueError("invalid probe build identity")
    return identity


def resource_identity(config):
    directory = Path(config["directory"]).resolve()
    if not directory.is_dir():
        return None
    # Skip mutable voices/caches. Prepared model files remain individually identifiable.
    files = []
    paths = set()
    for root in config.get("resource_directories", [str(directory)]):
        paths.update(Path(root).resolve().rglob("*"))
    for path in sorted(paths):
        if path.is_file() and not any(part in {"voices", ".cache", "voices-v2", ".native-preparation"} for part in path.relative_to(directory).parts):
            files.append({"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size,
                          "sha256": digest(path)})
    return {"directory": str(directory), "files": files}


def helper_identity(config):
    root = config.get("env", {}).get("TALECHIME_ESPEAK_DIR")
    if root is None:
        return None
    root = Path(root).resolve(strict=True)
    manifest = root / "manifest.json"
    value = read_json(manifest)
    # The frontend verifies the fixed source and complete file manifest before use.
    # Keep a separate snapshot: helper files are not neural model resources.
    snapshot = resource_identity({"directory": str(root)})
    return {"manifest_sha256": digest(manifest), "source_commit": value["source_commit"],
            "source_sha256": value["source_sha256"], "resources": snapshot}


def report(output, plan, records):
    expected = {row["id"] for row in plan}
    actual = [row["id"] for row in records]
    complete = len(actual) == len(set(actual)) and expected == set(actual)
    summary = {"coverage_complete": complete, "planned": len(plan), "recorded": len(records),
               "generation": dict(Counter(row["generation"] for row in records)),
               "official": dict(Counter(row["official"] for row in records)),
               "listening": dict(Counter(row["listening"] for row in records)),
               "missing": sorted(expected - set(actual)), "unexpected": sorted(set(actual) - expected)}
    write_json(output / "summary.json", summary)
    rows = []
    for row in records:
        key = row["id"]
        audio_path = row.get("audio_path", key + "/audio.wav")
        audio = f'<audio controls preload="none" src="{html.escape(audio_path, quote=True)}"></audio>' if row["generation"] == "generated" else ""
        conditions = json.dumps({name: row.get(name) for name in
                                 ("model_id", "device", "voice", "params", "reference_source")},
                                ensure_ascii=False)
        execution = list(dict.fromkeys(event["execution_text"] for event in
                         row.get("text_diagnostics", {}).get("events", [])
                         if isinstance(event.get("execution_text"), str)))
        details = '<details><summary>条件与执行文本</summary><pre>' + html.escape(
            conditions + "\n" + "\n".join(execution)) + '</pre></details>'
        rows.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in
                    (key, row["text"], row["generation"], row["official"], row["listening"], row.get("known_failure", "")))
                    + f'<td>{details}</td><td>{audio}</td></tr>')
    (output / "listen.html").write_text('<!doctype html><meta charset="utf-8"><title>TTS matrix</title>'
        '<h1>TTS matrix</h1><p>生成、官方对照和人工听验独立；历史失败不能由 EOS 覆盖。</p>'
        '<table border="1"><thead><tr><th>变体/用例</th><th>原文</th><th>生成</th><th>官方</th><th>听验</th><th>已有失败</th><th>模型与输入</th><th>音频</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table>', encoding="utf-8")
    return summary


def run(configs, variants, cases, probe, output, timeout):
    validate(configs, variants, cases)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    configs = {config["id"]: config for config in configs}
    by_variant = {variant["id"]: variant for variant in variants}
    by_case = {case["id"]: case for case in cases}
    plan = make_plan(variants, cases, configs)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    receipt = {"probe_sha256": digest(probe), "platform": platform.platform(),
               "machine": platform.machine(), "python": platform.python_version(), "seed": 42}
    try:
        receipt["probe_identity"] = probe_identity(probe)
    except (OSError, ValueError, subprocess.SubprocessError):
        receipt["identity_error"] = "probe does not provide a valid bounded build identity"
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, check=False)
    receipt.update(commit=git.stdout.strip(), tracked_diff_sha256=hashlib.sha256(diff.stdout).hexdigest())
    write_json(output / "build.json", receipt)
    resources = {}
    records = []
    cancelled = False
    with (output / "results.jsonl").open("x", encoding="utf-8") as log:
        for planned in plan:
            variant = by_variant[planned["variant"]]
            case = by_case[planned["case"]]
            config = configs.get(variant["id"])
            record = {**planned, "text": case["text"], "source_sha256": hashlib.sha256(case["text"].encode()).hexdigest(),
                      "source_range": [0, len(case["text"].encode())], "official": "pending",
                      "numeric": "pending", "listening": "pending", "generation": "blocked",
                      "model": variant}
            failure = variant.get("known_failures", {}).get(case["id"])
            if failure:
                record.update(known_failure=failure, listening="failed")
            directory = output / planned["id"]
            directory.mkdir(parents=True)
            (directory / "source.txt").write_bytes(case["text"].encode("utf-8"))
            try:
                if cancelled:
                    record.update(generation="cancelled", reason="matrix interrupted")
                elif "identity_error" in receipt:
                    record["reason"] = receipt["identity_error"]
                elif config is None:
                    record["reason"] = "model configuration missing"
                elif config.get("device", "cpu") not in variant["devices"]:
                    record["reason"] = "selected device outside model's local supported path"
                else:
                    if variant["id"] not in resources:
                        resources[variant["id"]] = resource_identity(config)
                        write_json(output / "resources.json", resources)
                    if not resources[variant["id"]] or not resources[variant["id"]]["files"]:
                        record["reason"] = "model resources missing"
                    else:
                        params = planned["params"]
                        record.update(device=config.get("device", "cpu"), voice=config["voice"], params=params,
                                      style=config.get("style"), reference_source=config.get("reference_source"),
                                      reference_identity=voice_identity(config, variant),
                                      helper_identity=helper_identity(config))
                        env = {key: value for key, value in os.environ.items() if key in INHERIT_KEYS}
                        env.update(config.get("env", {}))
                        env["TALECHIME_PROBE_PARAMS"] = json.dumps(params)
                        if variant.get("model_id"):
                            env["TALECHIME_PROBE_MODEL"] = variant["model_id"]
                        env["TALECHIME_TEXT_DIAGNOSTICS"] = "1"
                        if "style" in config:
                            env["NOVEL_TTS_PROBE_STYLE"] = config["style"]
                        command = [str(probe), config["backend"], str(Path(config["directory"]).resolve()),
                                   str((directory / "audio.wav").resolve()), record["device"],
                                   "@" + str((directory / "source.txt").resolve()), config["voice"]]
                        record.update(invoke(command, env, directory, timeout))
                        record["text_diagnostics"] = text_diagnostics(directory, case)
                        if record["process_status"] == "timeout":
                            record["generation"] = "timeout"
                        elif record["exit_code"] != 0:
                            rejection = rejection_evidence(directory, planned["expected"])
                            record["generation"] = "expected_rejection" if rejection else "failed"
                            if rejection:
                                record["rejection"] = rejection
                        else:
                            record.update(generation_evidence(directory, case, variant))
                            record["generation"] = "generated"
            except KeyboardInterrupt:
                cancelled = True
                record.update(generation="cancelled", reason="matrix interrupted")
            except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
                record.update(generation="failed", reason=str(error))
            records.append(record)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            report(output, plan, records)
    summary = report(output, plan, records)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON list of explicit local model configurations")
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--catalog", type=Path, default=HERE / "backend-variants.json")
    parser.add_argument("--cases", type=Path, default=HERE / "text-matrix.json")
    args = parser.parse_args()
    summary = run(read_json(args.config), read_json(args.catalog), read_json(args.cases),
                  args.probe.resolve(strict=True), args.output.resolve(), args.timeout)
    print(json.dumps(summary, ensure_ascii=False))
    # A complete accounting can still contain failures or blockers.
    return 0 if summary["coverage_complete"] and summary["generation"] == {"generated": summary["planned"]} else 1


if __name__ == "__main__":
    raise SystemExit(main())
