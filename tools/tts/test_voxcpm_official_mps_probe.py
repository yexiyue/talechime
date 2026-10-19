"""Probe contract checks; run in the isolated official Python environment."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from voxcpm_official_mps_probe import generate_round, load_segments, parse_args, save_wav


class OfficialProbeTests(unittest.TestCase):
    def test_generation_times_only_synchronized_api_calls(self):
        torch = SimpleNamespace(mps=SimpleNamespace(synchronize=Mock()))
        model = SimpleNamespace(generate=Mock(return_value=np.ones(48_000, dtype=np.float32)),
                                tts_model=SimpleNamespace(last_successful_seed=42))
        with tempfile.TemporaryDirectory() as temporary, \
                patch("voxcpm_official_mps_probe.save_wav") as save, \
                patch("voxcpm_official_mps_probe.time.monotonic", side_effect=[10, 12, 100, 103]):
            result = generate_round(torch, model, [{"text": "one"}, {"text": "two"}],
                                    Path("reference.wav"), "transcript", Path(temporary) / "round")
            self.assertEqual(result["generate_ms"], 5000)
            self.assertEqual(result["audio_seconds"], 2)
            self.assertEqual(torch.mps.synchronize.call_count, 4)
            self.assertEqual(save.call_count, 2)
            for call in model.generate.call_args_list:
                self.assertNotIn("reference_wav_path", call.kwargs)
                self.assertFalse(call.kwargs["retry_badcase"])
                self.assertEqual(call.kwargs["seed"], 42)
            self.assertEqual(result["termination"], "not_exposed")

    def test_invalid_pcm_is_rejected(self):
        for pcm in (np.array([]), np.array([np.nan]), np.array([np.inf])):
            with self.subTest(pcm=pcm), self.assertRaisesRegex(ValueError, "invalid PCM"):
                save_wav(Path("unused.wav"), pcm)

    def test_segment_manifest_requires_nonempty_text_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "segments.json"
            for manifest in ([], [None], [1], [{"text": 42}], [{"text": " "}]):
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                    load_segments(path)

    def test_invalid_round_parameters_fail_before_model_loading(self):
        required = [value for key in ("source", "models", "segments", "reference", "transcript", "output")
                    for value in (f"--{key}", "unused")]
        for extra in (["--rounds", "0"], ["--rounds", "4"], ["--warmup-rounds", "-1"],
                      ["--warmup-rounds", "2"]):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(required + extra)
        self.assertFalse(parse_args(required).audit_devices)


if __name__ == "__main__":
    unittest.main()
