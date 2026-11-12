"""Offline failure-boundary tests for matrix execution and evidence accounting."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest

spec = importlib.util.spec_from_file_location("matrix", Path(__file__).with_name("run_text_matrix.py"))
matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matrix)

FAKE_PROBE = '''#!/usr/bin/env python3
import json, os, pathlib, struct, subprocess, sys, time
if sys.argv[1] == "--identity":
    print(json.dumps(dict(features=["fixture"],package_version="test")))
    sys.exit(0)
out=pathlib.Path(sys.argv[3])
mode=sys.argv[6]
if mode == "timeout":
    child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])
    print(child.pid, flush=True)
    print("partial error",file=sys.stderr,flush=True)
    time.sleep(60)
if mode == "crash":
    print("failed to load",file=sys.stderr,flush=True)
    sys.exit(3)
if mode == "reject":
    print("unsupported fixture language",file=sys.stderr,flush=True)
    sys.exit(2)
assert "TALECHIME_TEST_SECRET" not in os.environ
text=pathlib.Path(sys.argv[5][1:]).read_text()
print("TALECHIME_TEXT_DIAGNOSTIC "+json.dumps(dict(backend="fixture",phase="backend_input",source_start=0,source_end=len(text.encode()),source_text=text,execution_text=text)),file=sys.stderr,flush=True)
print("TALECHIME_TEXT_DIAGNOSTIC "+json.dumps(dict(backend="fixture",phase="model_tokens",execution_text=text,tokens=[1,2])),file=sys.stderr,flush=True)
data=struct.pack("<ff", 0.25, -0.25)
fmt=struct.pack("<HHIIHH",3,1,24000,96000,4,32)
out.write_bytes(b"RIFF"+struct.pack("<I",36+len(data))+b"WAVEfmt "+struct.pack("<I",16)+fmt+b"data"+struct.pack("<I",len(data))+data)
segments=out.with_name("audio-segments")
segments.mkdir()
(segments/"index.json").write_text(json.dumps([dict(text=text, source_start=0,source_end=len(text.encode()),start_sample=0,end_sample=2)]))
print(json.dumps(dict(stream_end=True,model_eos="not_exposed",loaded_model=os.environ.get("TALECHIME_PROBE_MODEL"))))
'''


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.probe = self.root / "probe"
        self.probe.write_text(FAKE_PROBE)
        self.probe.chmod(0o755)
        model = self.root / "model"
        model.mkdir()
        (model / "weights").write_bytes(b"fixture")
        self.config = {"id": "nano", "backend": "fake", "directory": str(model), "voice": "ok"}
        self.variants = [{"id": "nano", "backend": "fake", "devices": ["cpu"],
                          "known_failures": {"mixed": "hello omitted in user listening"}}]
        self.cases = [{"id": "mixed", "text": "你好 hello。"}]
        self.output = self.root / "output"

    def tearDown(self):
        self.temp.cleanup()

    def run_matrix(self, configs=None, timeout=3):
        return matrix.run([self.config] if configs is None else configs, self.variants,
                          self.cases, self.probe, self.output, timeout)

    def test_generated_eos_preserves_user_failure_and_pending_official(self):
        summary = self.run_matrix()
        self.assertEqual(summary["generation"], {"generated": 1})
        self.assertEqual(summary["listening"], {"failed": 1})
        self.assertEqual(summary["official"], {"pending": 1})
        self.assertTrue(summary["coverage_complete"])
        row = json.loads((self.output / "results.jsonl").read_text())
        self.assertEqual(row["text_diagnostics"]["status"], "captured")
        self.assertEqual(row["text_diagnostics"]["events"][0]["source_end"], len(self.cases[0]["text"].encode()))
        self.assertIn("hello omitted", (self.output / "listen.html").read_text())

    def test_missing_config_accounts_for_every_planned_row(self):
        self.cases.append({"id": "other", "text": "こんにちは。"})
        summary = self.run_matrix([])
        self.assertEqual(summary["generation"], {"blocked": 2})
        self.assertTrue(summary["coverage_complete"])

    def test_missing_resources_are_blocked(self):
        self.config["directory"] = str(self.root / "missing")
        self.assertEqual(self.run_matrix()["generation"], {"blocked": 1})

    def test_resource_snapshot_uses_declared_model_and_selected_reference(self):
        model = Path(self.config["directory"])
        selected = model / "selected"
        selected.mkdir()
        (selected / "weights").write_bytes(b"selected model")
        self.config["resource_directories"] = [str(selected)]
        self.assertEqual(self.run_matrix()["generation"], {"generated": 1})
        files = json.loads((self.output / "resources.json").read_text())["nano"]["files"]
        self.assertEqual([f["path"] for f in files], ["selected/weights"])
        self.config["resource_directories"] = [str(self.root)]
        with self.assertRaisesRegex(ValueError, "must belong"):
            matrix.validate([self.config], self.variants, self.cases)

    def test_crash_saves_exit_and_partial_logs(self):
        self.config["voice"] = "crash"
        self.assertEqual(self.run_matrix()["generation"], {"failed": 1})
        row = json.loads((self.output / "results.jsonl").read_text())
        self.assertEqual(row["exit_code"], 3)
        self.assertIn("failed to load", (self.output / "nano/mixed/stderr.log").read_text())

    def test_only_declared_model_error_establishes_expected_rejection(self):
        self.config["voice"] = "reject"
        self.variants[0]["expected_rejections"] = {"mixed": {
            "reason": "Fixed model language scope", "error_contains": "unsupported fixture language"}}
        self.assertEqual(self.run_matrix()["generation"], {"expected_rejection": 1})
        self.output = self.root / "crash-output"
        self.config["voice"] = "crash"
        self.assertEqual(self.run_matrix()["generation"], {"failed": 1})

    @unittest.skipIf(os.name == "nt", "POSIX process-group assertion")
    def test_timeout_preserves_logs_and_kills_descendant(self):
        self.config["voice"] = "timeout"
        self.assertEqual(self.run_matrix(timeout=0.5)["generation"], {"timeout": 1})
        directory = self.output / "nano/mixed"
        self.assertIn("partial error", (directory / "stderr.log").read_text())
        pid = int((directory / "stdout.log").read_text())
        # Reaped descendants may briefly remain zombies; a running child is never allowed.
        for _ in range(10):
            result = matrix.subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
            if not result.stdout.strip() or result.stdout.strip().startswith("Z"):
                break
            time.sleep(0.05)
        else:
            self.fail("timeout left a running descendant")

    def test_extensible_stereo_float_is_validated_without_relaxing_pcm_checks(self):
        data = struct.pack("<ffff", 0.25, -0.25, 0.5, -0.5)
        fmt = struct.pack("<HHIIHHHHI", 0xFFFE, 2, 48000, 384000, 8, 32, 22, 32, 3)
        fmt += bytes.fromhex("0300000000001000800000aa00389b71")
        wav = b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE"
        wav += b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
        path = self.root / "extensible.wav"
        path.write_bytes(wav)
        receipt = matrix.inspect_wav(path)
        self.assertEqual((receipt["channels"], receipt["samples"]), (2, 4))
        path.write_bytes(wav[:-len(data)] + struct.pack("<ffff", float("nan"), 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            matrix.inspect_wav(path)

    def test_existing_directory_is_not_overwritten(self):
        self.output.mkdir()
        (self.output / "existing").write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.run_matrix()
        self.assertEqual((self.output / "existing").read_text(), "preserve")

    def test_unsafe_ids_and_duplicate_configs_rejected_before_writing(self):
        self.config["id"] = "../outside"
        with self.assertRaises(ValueError):
            self.run_matrix()
        self.assertFalse(self.output.exists())
        self.config["id"] = "nano"
        with self.assertRaises(ValueError):
            self.run_matrix([self.config, self.config])

    def test_arbitrary_environment_and_nested_params_are_rejected(self):
        self.config["env"] = {"API_KEY": "secret"}
        with self.assertRaises(ValueError):
            self.run_matrix()
        self.config.pop("env")
        self.config["params"] = {"language": {"key": "secret"}}
        with self.assertRaises(ValueError):
            self.run_matrix()

    def test_case_parameters_are_predeclared_without_rewriting_the_source(self):
        self.config["params"] = {"speed": 1.0, "language": "auto"}
        self.config["case_params"] = {"mixed": {"language": "ja"}}
        self.run_matrix()
        plan = matrix.read_json(self.output / "plan.json")[0]
        self.assertEqual(plan["params"], {"speed": 1.0, "language": "ja"})
        self.assertEqual((self.output / "nano/mixed/source.txt").read_text(), self.cases[0]["text"])

    def test_unknown_case_parameter_profile_is_rejected_before_writing(self):
        self.config["case_params"] = {"not-a-case": {"language": "ja"}}
        with self.assertRaises(ValueError):
            self.run_matrix()
        self.assertFalse(self.output.exists())

    def test_credentials_are_not_inherited_or_recorded(self):
        os.environ["TALECHIME_TEST_SECRET"] = "sensitive-value"
        try:
            self.assertEqual(self.run_matrix()["generation"], {"generated": 1})
        finally:
            os.environ.pop("TALECHIME_TEST_SECRET")
        self.assertNotIn("sensitive-value", (self.output / "results.jsonl").read_text())

    def test_duplicate_result_cannot_pass_coverage_gate(self):
        plan = [{"id": "nano/mixed"}]
        row = {"id": "nano/mixed", "text": "hello", "generation": "generated", "official": "pending", "listening": "pending"}
        self.output.mkdir()
        self.assertFalse(matrix.report(self.output, plan, [row, row])["coverage_complete"])

    def test_loaded_model_must_match_the_planned_variant(self):
        self.run_matrix()
        directory = self.output / "nano/mixed"
        with (directory / "stdout.log").open("a") as log:
            log.write(json.dumps({"loaded_model": "1.7b-base"}) + "\n")
        with self.assertRaisesRegex(ValueError, "loaded model identity"):
            matrix.generation_evidence(directory, self.cases[0], {"model_id": "0.6b-base"})
        self.assertEqual(matrix.generation_evidence(directory, self.cases[0], {"model_id": "1.7b-base"})["loaded_model"], "1.7b-base")

    def test_reverification_cannot_upgrade_a_wrong_model_receipt(self):
        import verify_text_matrix
        self.variants[0]["model_id"] = "0.6b-base"
        self.run_matrix()
        with (self.output / "nano/mixed/stdout.log").open("a") as log:
            log.write(json.dumps({"loaded_model": "1.7b-base"}) + "\n")
        summary = verify_text_matrix.verify(self.output, self.root / "reverified")
        self.assertEqual(summary["generation"], {"failed": 1})

    def test_reverification_rejects_source_text_changed_after_planning(self):
        import verify_text_matrix
        self.run_matrix()
        results = self.output / "results.jsonl"
        row = json.loads(results.read_text())
        row["text"] = "changed"
        results.write_text(json.dumps(row) + "\n")
        summary = verify_text_matrix.verify(self.output, self.root / "changed-source")
        self.assertEqual(summary["generation"], {"failed": 1})

    def test_missing_source_or_pcm_suffix_is_not_success(self):
        self.run_matrix()
        directory = self.output / "nano/mixed"
        index = directory / "audio-segments/index.json"
        segments = matrix.read_json(index)
        segments[0]["source_end"] = len("你好".encode())
        matrix.write_json(index, segments)
        with self.assertRaisesRegex(ValueError, "source suffix"):
            matrix.generation_evidence(directory, self.cases[0])
        segments[0]["source_end"] = len(self.cases[0]["text"].encode())
        segments[0]["end_sample"] = 1
        matrix.write_json(index, segments)
        with self.assertRaisesRegex(ValueError, "PCM suffix"):
            matrix.generation_evidence(directory, self.cases[0])

    def test_aggregate_preserves_prior_listening_failure(self):
        import aggregate_text_matrix
        self.run_matrix()
        first = self.output
        self.output = self.root / "second"
        self.variants[0]["known_failures"] = {}
        self.run_matrix()
        output = self.root / "aggregate"
        summary = aggregate_text_matrix.aggregate([first, self.output], self.variants, self.cases, output)
        self.assertEqual(summary["generation"], {"generated": 1})
        self.assertEqual(summary["listening"], {"failed": 1})
        row = json.loads((output / "results.jsonl").read_text())
        self.assertEqual(row["evidence_source"], str(self.output.resolve()))
        self.assertIn("hello omitted", row["known_failure"])

    def test_changed_audio_cannot_be_adopted_by_verification_or_aggregation(self):
        import aggregate_text_matrix
        import verify_text_matrix
        self.run_matrix()
        wav = self.output / "nano/mixed/audio.wav"
        data = wav.read_bytes()
        wav.write_bytes(data[:-4] + struct.pack("<f", 0.5))
        summary = verify_text_matrix.verify(self.output, self.root / "changed-verification")
        self.assertEqual(summary["generation"], {"failed": 1})
        summary = aggregate_text_matrix.aggregate([self.output], self.variants, self.cases, self.root / "aggregate")
        self.assertEqual(summary["generation"], {"failed": 1})

    def test_aggregate_annotation_rechecks_external_audio_and_model(self):
        import aggregate_text_matrix
        import annotate_official_matrix
        import verify_text_matrix
        self.variants[0]["model_id"] = "fixture-model"
        self.run_matrix()
        log = self.output / "nano/mixed/stdout.log"
        with log.open("a") as out:
            out.write(json.dumps({"loaded_model": "fixture-model"}) + "\n")
        aggregate = self.root / "aggregate"
        aggregate_text_matrix.aggregate([self.output], self.variants, self.cases, aggregate)
        receipt = self.root / "official.json"
        receipt.write_text(json.dumps({"checks": []}))
        annotated = self.root / "annotated"
        summary = annotate_official_matrix.annotate(aggregate, annotated, receipt)
        self.assertEqual(summary["generation"], {"generated": 1})
        row = json.loads((annotated / "results.jsonl").read_text())
        self.assertFalse(Path(row["audio_path"]).is_absolute())
        wav = self.output / "nano/mixed/audio.wav"
        self.assertEqual((annotated / row["audio_path"]).resolve(), wav.resolve())
        with log.open("a") as out:
            out.write(json.dumps({"loaded_model": "other-model"}) + "\n")
        summary = verify_text_matrix.verify(aggregate, self.root / "wrong-model")
        self.assertEqual(summary["generation"], {"failed": 1})
        with log.open("a") as out:
            out.write(json.dumps({"loaded_model": "fixture-model"}) + "\n")
        data = wav.read_bytes()
        wav.write_bytes(data[:-4] + struct.pack("<f", 0.5))
        summary = verify_text_matrix.verify(aggregate, self.root / "changed-external-audio")
        self.assertEqual(summary["generation"], {"failed": 1})

    def test_aggregate_rejects_different_source_text_before_writing(self):
        import aggregate_text_matrix
        self.run_matrix()
        self.cases[0]["text"] = "different public text"
        output = self.root / "aggregate"
        with self.assertRaisesRegex(ValueError, "source text"):
            aggregate_text_matrix.aggregate([self.output], self.variants, self.cases, output)
        self.assertFalse(output.exists())

    def test_aggregate_rejects_audio_path_different_from_validated_wav(self):
        import aggregate_text_matrix
        self.run_matrix()
        results = self.output / "results.jsonl"
        row = json.loads(results.read_text())
        row["audio_path"] = "nano/mixed/other.wav"
        results.write_text(json.dumps(row) + "\n")
        output = self.root / "aggregate"
        summary = aggregate_text_matrix.aggregate([self.output], self.variants, self.cases, output)
        self.assertEqual(summary["generation"], {"failed": 1})
        record = json.loads((output / "results.jsonl").read_text())
        self.assertIn("unexpected audio evidence filename", record["reason"])

    def test_committed_catalog_and_cases_have_unique_complete_plan(self):
        variants = matrix.read_json(matrix.HERE / "backend-variants.json")
        cases = matrix.read_json(matrix.HERE / "text-matrix.json")
        matrix.validate([], variants, cases)
        plan = matrix.make_plan(variants, cases)
        self.assertEqual(len(plan), len(variants) * len(cases))
        self.assertEqual(len(plan), len({row["id"] for row in plan}))
        self.assertTrue({"fullwidth", "japanese-date", "spaces", "long-word", "long-paragraph"} <= {c["id"] for c in cases})


if __name__ == "__main__":
    unittest.main()
