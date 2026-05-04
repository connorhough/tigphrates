"""Tests for python/play_traced.py — headless trace generator.

These tests run the trace generator end-to-end against a real bridge
process. They are integration tests, not unit tests. They rely on
``npm run bridge`` being available (the existing TigphratesEnv smoke
tests rely on the same).
"""
import json
import pathlib
import subprocess
import sys

import pytest


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "python/play_traced.py", *args],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_one_game_vs_heuristic(tmp_path):
    """Generates exactly one JSONL file and at least one line per game."""
    out_dir = tmp_path / "traces" / "abc1234"
    res = _run(
        "--out-dir", str(out_dir),
        "--games-vs-heuristic", "1",
        "--games-vs-champion", "0",
        "--max-turns", "60",
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    files = sorted(out_dir.glob("game_*.jsonl"))
    assert len(files) == 1, f"expected 1 trace file, got {len(files)}: {files}"
    lines = files[0].read_text().splitlines()
    assert len(lines) >= 1
    first = json.loads(lines[0])
    # Minimal schema for this task (telemetry fields land in Task 5).
    for key in ("turn", "active_player", "model_player", "phase", "chosen_action"):
        assert key in first, f"missing key {key} in {first}"


def test_zero_games_is_noop(tmp_path):
    """Asking for zero games of each type produces no files and exits 0."""
    out_dir = tmp_path / "traces" / "empty"
    res = _run(
        "--out-dir", str(out_dir),
        "--games-vs-heuristic", "0",
        "--games-vs-champion", "0",
    )
    assert res.returncode == 0
    assert not out_dir.exists() or not list(out_dir.glob("game_*.jsonl"))
