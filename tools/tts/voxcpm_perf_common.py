"""Shared short-process ABBA statistics for VoxCPM macOS experiments."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
import json
import math
from pathlib import Path
import time


SCREEN_SEQUENCE = ("baseline", "candidate", "candidate", "baseline")
SEQUENCE = SCREEN_SEQUENCE * 2


def hot_rounds(value: str) -> int:
    rounds = int(value)
    if not 1 <= rounds <= 3:
        raise argparse.ArgumentTypeError("hot rounds must be 1..3")
    return rounds


def cooldown_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("cooldown must be finite and nonnegative")
    return seconds


def summarize_runs(runs: Iterable[dict], screen: bool) -> dict:
    """Return the fixed VoxCPM acceptance statistics for completed run records."""
    runs = list(runs)
    sequence = SCREEN_SEQUENCE if screen else SEQUENCE
    if tuple(run["arm"] for run in runs) != sequence:
        raise ValueError("runs must match the complete balanced screen or ABBA sequence")
    for run in runs:
        value = run["rtf_mean"]
        if not math.isfinite(value) or value <= 0:
            raise ValueError("RTF must be finite and positive")
    stats = {}
    for arm in ("baseline", "candidate"):
        values = [run["rtf_mean"] for run in runs if run["arm"] == arm]
        stats[arm] = {"n": len(values), "mean": sum(values) / len(values),
                      "min": min(values), "max": max(values)}
    stats["improvement"] = 1 - stats["candidate"]["mean"] / stats["baseline"]["mean"]
    stats["screen_only"] = screen

    paired = []
    for index in range(0, len(runs) - 1, 2):
        first, second = runs[index], runs[index + 1]
        baseline = first if first["arm"] == "baseline" else second
        candidate = second if first["arm"] == "baseline" else first
        paired.append({"index": index, "baseline": baseline["rtf_mean"],
                       "candidate": candidate["rtf_mean"],
                       "improvement": 1 - candidate["rtf_mean"] / baseline["rtf_mean"]})
    stats["paired"] = paired
    stats["paired_improvement_min"] = min(pair["improvement"] for pair in paired)
    stats["paired_direction_consistent"] = all(pair["improvement"] > 0 for pair in paired)
    # A five-percent rise in both arms is a drift warning, not a diagnosis.
    drift = {}
    for arm in ("baseline", "candidate"):
        values = [run["rtf_mean"] for run in runs if run["arm"] == arm]
        midpoint = len(values) // 2
        drift[arm] = sum(values[midpoint:]) / sum(values[:midpoint]) - 1
    stats["later_half_change"] = drift
    stats["common_slowdown"] = all(change > 0.05 for change in drift.values())
    first_pcm = {}
    for arm in ("baseline", "candidate"):
        values = [run.get("first_pcm_ms") for run in runs if run["arm"] == arm]
        first_pcm[arm] = None
        if all(value is not None for value in values):
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("first PCM times must be finite and positive")
            first_pcm[arm] = sum(values) / len(values)
    stats["first_pcm_mean_ms"] = first_pcm
    stats["first_pcm_regression"] = None
    if all(value is not None for value in first_pcm.values()):
        stats["first_pcm_regression"] = first_pcm["candidate"] / first_pcm["baseline"] - 1
    stats["performance_gate"] = (
        not screen
        and stats["improvement"] >= 0.10
        and stats["candidate"]["max"] < stats["baseline"]["min"]
        and stats["paired_direction_consistent"]
        and not stats["common_slowdown"]
        and (stats["first_pcm_regression"] is None or stats["first_pcm_regression"] <= 0.05)
    )
    return stats


def run_abba(output: Path, screen: bool, cooldown: float,
             run_arm: Callable[[str, int], dict]) -> dict:
    """Run serial slots and persist each result before waiting for the next."""
    cooldown_seconds(str(cooldown))
    output.mkdir(parents=True, exist_ok=False)
    runs = []
    for index, arm in enumerate(SCREEN_SEQUENCE if screen else SEQUENCE):
        if index:
            time.sleep(cooldown)
        runs.append(run_arm(arm, index))
        (output / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    statistics = summarize_runs(runs, screen)
    (output / "statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    return statistics
