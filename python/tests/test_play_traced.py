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


def test_one_game_vs_heuristic_with_telemetry(tmp_path):
    """Move records include policy telemetry on the model player's turns."""
    # Use the existing best checkpoint if present; otherwise the test
    # falls back to using a freshly initialized policy (still exercises
    # the telemetry path because forward() runs regardless of weights).
    out_dir = tmp_path / "traces" / "abc1234"
    res = _run(
        "--out-dir", str(out_dir),
        "--games-vs-heuristic", "1",
        "--games-vs-champion", "0",
        "--max-turns", "60",
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    files = sorted(out_dir.glob("game_*.jsonl"))
    assert len(files) == 1
    records = [json.loads(l) for l in files[0].read_text().splitlines()]
    assert len(records) >= 1
    model_records = [r for r in records if r["model_player"]]
    assert len(model_records) >= 1, "expected at least one model-player turn"

    r = model_records[0]
    # Telemetry contract:
    assert "type_top5" in r and isinstance(r["type_top5"], list)
    assert len(r["type_top5"]) >= 1 and len(r["type_top5"]) <= 5
    for entry in r["type_top5"]:
        assert set(entry.keys()) >= {"type_name", "prob"}
        assert 0.0 <= entry["prob"] <= 1.0 + 1e-6
    assert "param_top5_for_chosen_type" in r
    assert "value_estimate" in r and isinstance(r["value_estimate"], (int, float))
    assert "argmax_action_index" in r["chosen_action"]
    assert "sampled_action_index" in r["chosen_action"]
    # Heuristic records (opponent) should NOT have model telemetry.
    opp_records = [r for r in records if not r["model_player"]]
    if opp_records:
        assert "type_top5" not in opp_records[0]


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


def test_two_games_vs_champion_when_pool_has_champion(tmp_path, monkeypatch):
    """If a pool exists with at least one snapshot, --games-vs-champion=2
    produces 2 additional trace files for the champion match-up.

    Schema note: elo.json is a flat dict keyed by full filename
    (including the .pt extension). The agent's own rating lives under
    the "_agent" key and must be excluded from champion candidates.
    See python/evaluate.py:280-298 for the canonical reader/writer.
    """
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    # Flat schema: keys are filenames including ".pt"; "_agent" is the
    # current agent's rating and must NOT be considered a champion.
    (pool_dir / "elo.json").write_text(json.dumps({
        "_agent": 1500.0,
        "policy_v0.pt": 1600.0,
        "policy_v1.pt": 1550.0,
    }))
    # The script will pick policy_v0.pt (highest non-_agent rating).
    src_best = pathlib.Path("models/policy_final.pt")
    if not src_best.exists():
        pytest.skip("models/policy_final.pt not present — champion path needs a real checkpoint")
    (pool_dir / "policy_v0.pt").write_bytes(src_best.read_bytes())

    out_dir = tmp_path / "traces" / "champ"
    res = _run(
        "--out-dir", str(out_dir),
        "--games-vs-heuristic", "0",
        "--games-vs-champion", "2",
        "--max-turns", "60",
        "--pool-dir", str(pool_dir),
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    files = sorted(out_dir.glob("game_*.jsonl"))
    assert len(files) == 2
    # Champion games should be tagged in the records.
    first_record = json.loads(files[0].read_text().splitlines()[0])
    assert first_record.get("opponent_kind") == "champion"
