"""Regression tests for the shared ABBA statistics."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from voxcpm_perf_common import (
    SCREEN_SEQUENCE, SEQUENCE, cooldown_seconds, hot_rounds, run_abba, summarize_runs,
)


def run(arm, rtf):
    return {"arm": arm, "rtf_mean": rtf}


class SummarizeRunsTests(unittest.TestCase):
    def test_sequences_are_symmetric_and_pair_balanced(self):
        self.assertEqual(SCREEN_SEQUENCE, ("baseline", "candidate", "candidate", "baseline"))
        self.assertEqual(SEQUENCE, SCREEN_SEQUENCE + SCREEN_SEQUENCE)

    def test_accepts_balanced_gain(self):
        stats = summarize_runs([
            run("baseline", 1.0), run("candidate", 0.8),
            run("candidate", 0.79), run("baseline", 1.01),
            run("baseline", 1.02), run("candidate", 0.81),
            run("candidate", 0.80), run("baseline", 1.03),
        ], screen=False)
        self.assertTrue(stats["performance_gate"])
        self.assertEqual(len(stats["paired"]), 4)

    def test_screen_never_passes_the_acceptance_gate(self):
        stats = summarize_runs([
            run("baseline", 1.0), run("candidate", 0.7),
            run("candidate", 0.7), run("baseline", 1.0),
        ], screen=True)
        self.assertFalse(stats["performance_gate"])

    def test_rejects_unbalanced_pair(self):
        with self.assertRaisesRegex(ValueError, "balanced"):
            summarize_runs([run("baseline", 1.0), run("baseline", 0.9)], screen=False)

    def test_incomplete_runs_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            summarize_runs([run("baseline", 1.0), run("candidate", 0.7)], screen=False)

    def test_invalid_rtf_is_rejected(self):
        for value in (float("nan"), float("inf"), -1, 0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "RTF"):
                summarize_runs([run(arm, value) for arm in SCREEN_SEQUENCE], screen=True)

    def test_common_slowdown_blocks_otherwise_passing_candidate(self):
        stats = summarize_runs([
            run("baseline", 1.0), run("candidate", 0.7),
            run("candidate", 0.7), run("baseline", 1.0),
            run("baseline", 1.1), run("candidate", 0.77),
            run("candidate", 0.77), run("baseline", 1.1),
        ], screen=False)
        self.assertGreater(stats["improvement"], 0.1)
        self.assertTrue(stats["common_slowdown"])
        self.assertFalse(stats["performance_gate"])

    def test_first_pcm_regression_blocks_throughput_gain(self):
        runs = [dict(run(arm, 1.0 if arm == "baseline" else 0.7),
                     first_pcm_ms=100 if arm == "baseline" else 110) for arm in SEQUENCE]
        stats = summarize_runs(runs, screen=False)
        self.assertAlmostEqual(stats["first_pcm_regression"], 0.1)
        self.assertFalse(stats["performance_gate"])

    def test_missing_first_pcm_is_explicit(self):
        stats = summarize_runs([run(arm, 1.0) for arm in SCREEN_SEQUENCE], screen=True)
        self.assertIsNone(stats["first_pcm_regression"])

    def test_runner_preserves_slots_and_cooldown(self):
        with tempfile.TemporaryDirectory() as temporary, patch("voxcpm_perf_common.time.sleep") as sleep:
            output = Path(temporary) / "screen"
            seen = []

            def arm_result(arm, index):
                seen.append((arm, index))
                return run(arm, 1.0)

            run_abba(output, True, 20, arm_result)
            self.assertEqual(seen, list(zip(SCREEN_SEQUENCE, range(4))))
            self.assertEqual(sleep.call_count, 3)
            sleep.assert_called_with(20)
            self.assertEqual(len(json.loads((output / "runs.json").read_text())), 4)
            self.assertTrue((output / "statistics.json").is_file())


class ParameterTests(unittest.TestCase):
    def test_round_limits(self):
        for value in ("1", "2", "3"):
            self.assertEqual(hot_rounds(value), int(value))
        for value in ("0", "4", "-1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                hot_rounds(value)

    def test_cooldown_rejects_nonfinite_and_negative_values(self):
        self.assertEqual(cooldown_seconds("0"), 0)
        for value in ("nan", "inf", "-1"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                cooldown_seconds(value)


if __name__ == "__main__":
    unittest.main()
