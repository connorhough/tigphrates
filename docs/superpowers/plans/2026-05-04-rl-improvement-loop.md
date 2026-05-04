# Agent-driven RL improvement loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing autoresearch loop so that, after a kept training run, a Claude expert agent reviews rich move traces from headless games and a Claude DS agent translates that critique into a concrete `python/train.py` edit.

**Architecture:** Add a new `python/play_traced.py` that loads `models/policy_best.pt` and plays N headless games against (a) the heuristic and (b) the prior pool champion, dumping per-move telemetry (top-5 type/param probs, value estimate, decoded action label, immediate reward and shaping breakdown) to `traces/<commit>/game_<i>.jsonl`. Modify `python/run_experiments.sh` so that on the "kept" branch it runs the trace generator, dispatches a stateless `claude -p` expert agent (writes `traces/<commit>/critique.md`), then dispatches a stateless `claude -p` DS agent (edits and commits `python/train.py`). Each agent gets the current shaping config so it does not propose duplicating existing incentives. If any new step fails, fall back to the existing direct-Claude-edit path.

**Tech Stack:** Python 3 (torch, numpy, gymnasium), bash, the existing TS bridge (`src/bridge/server.ts`), `claude` CLI in stateless `--print` mode. Tests run under `pytest` in `python/tests/`. No new pip dependencies.

**Spec:** [`docs/superpowers/specs/2026-05-04-rl-improvement-loop-design.md`](../specs/2026-05-04-rl-improvement-loop-design.md)

---

## File structure

**New files:**
- `traces/.gitignore` — keep generated trace JSONL out of git
- `python/play_traced.py` — trace generator (argparse CLI; produces `traces/<commit>/game_<i>.jsonl`)
- `python/tests/test_play_traced.py` — tests for trace generator
- `prompts/rl_expert.md` — expert agent prompt template (uses `{{...}}` placeholders the orchestrator fills in)
- `prompts/rl_ds.md` — data-science agent prompt template
- `python/shaping_config_dump.py` — small helper that snapshots the current shaping coefficients to `models/shaping_config.json`
- `python/tests/test_shaping_config_dump.py` — tests for the snapshot helper

**Modified files:**
- `python/run_experiments.sh` — insert trace + expert + DS calls in the "kept" branch with fallback to existing direct-edit path
- `python/train.py` — call `dump_shaping_config(...)` once after best-model save so `shaping_config.json` reflects the run's actual coefficients

**Unchanged:** `evaluate.py`, `tournament.py`, `tigphrates_env.py`, `imitation_pretrain.py`, the TS bridge, the React UI.

**Decision locked from the spec:** the trace generator is a sibling script (`play_traced.py`) — not an extension of `policy_server.py`. Reason: the server is a long-running HTTP daemon; trace generation is a one-shot batch job. Sharing the inference path through `policy_server.py` would force HTTP plumbing for no benefit.

---

## Conventions

- Every Python step adds tests first, then code, then runs the test, then commits.
- Tests live in `python/tests/test_<topic>.py` mirroring the existing structure (`test_event_shaping.py`, `test_potential_shaping.py`, etc.).
- All env vars used in `play_traced.py` get sensible defaults so the script is invocable with zero args.
- All bash steps in `run_experiments.sh` must be idempotent — re-running on the same commit either reuses cached artifacts or regenerates them.
- Commit messages follow the existing repo style: lowercase imperative, no Conventional Commits prefix (this repo uses messages like `experiment 11: +HIDDEN_DIM = 256`, `enriched observations + margin reward`, etc.).
- Test commands run from repo root: `python -m pytest python/tests/test_<topic>.py -v`.
- Do not skip hooks. Do not amend commits. New commit per task.

---

## Task 1: Stub `traces/` directory + gitignore

**Files:**
- Create: `traces/.gitignore`

- [ ] **Step 1: Create the directory and gitignore**

```bash
mkdir -p traces
```

Then write `traces/.gitignore`:

```
# Generated move traces from python/play_traced.py.
# Keep the directory in git but exclude all generated content.
*
!.gitignore
```

- [ ] **Step 2: Verify git tracks the dir but ignores contents**

Run:
```bash
touch traces/probe.jsonl
git status --short traces/
rm traces/probe.jsonl
```

Expected: `git status` shows nothing for `traces/probe.jsonl` (ignored). The `.gitignore` itself should appear as untracked the first time.

- [ ] **Step 3: Commit**

```bash
git add traces/.gitignore
git commit -m "traces/: gitignore generated move traces"
```

---

## Task 2: Shaping-config snapshot helper

**Why:** the expert and DS agents both need the exact shaping coefficients used in the most recent training run. Reading env vars and source defaults independently is fragile (the train script reads env vars at import, and a different shell may have different exports). Snapshotting once at the end of training gives both agents one source of truth.

**Files:**
- Create: `python/shaping_config_dump.py`
- Create: `python/tests/test_shaping_config_dump.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_shaping_config_dump.py`:

```python
"""Tests for python/shaping_config_dump.py.

The helper snapshots the live shaping coefficients (env vars + module
defaults) into a JSON file so downstream agents have one source of truth.
"""
import json
import os
import pathlib

import pytest

from shaping_config_dump import dump_shaping_config, load_shaping_config


def test_dump_writes_expected_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.07")
    monkeypatch.setenv("KING_LEADER_BONUS", "0.12")
    monkeypatch.setenv("SCORE_DELTA_COEF", "1.5")
    monkeypatch.setenv("MARGIN_DELTA_COEF", "1.0")
    monkeypatch.setenv("POTENTIAL_GAMMA_SHAPING_COEF", "1.0")
    monkeypatch.setenv("BC_COEF", "0.1")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")
    out = tmp_path / "shaping_config.json"

    dump_shaping_config(out)

    assert out.exists()
    data = json.loads(out.read_text())
    # Event shaping (per-action bonuses)
    assert data["event_shaping"]["leader_place_bonus"] == pytest.approx(0.07)
    assert data["event_shaping"]["king_leader_bonus"] == pytest.approx(0.12)
    assert data["event_shaping"]["kingdom_form_bonus"] == pytest.approx(0.10)  # default
    assert data["event_shaping"]["treasure_collect_bonus"] == pytest.approx(0.15)  # default
    assert data["event_shaping"]["monument_build_bonus"] == pytest.approx(0.10)  # default
    assert data["event_shaping"]["decay_steps"] == 200000
    # Score shaping (per-step potential / delta family)
    assert data["score_shaping"]["score_delta_coef"] == pytest.approx(1.5)
    assert data["score_shaping"]["margin_delta_coef"] == pytest.approx(1.0)
    assert data["score_shaping"]["potential_gamma_shaping_coef"] == pytest.approx(1.0)
    # BC auxiliary
    assert data["bc"]["bc_coef"] == pytest.approx(0.1)


def test_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    out = tmp_path / "shaping_config.json"
    dump_shaping_config(out)
    loaded = load_shaping_config(out)
    assert loaded["event_shaping"]["leader_place_bonus"] == pytest.approx(0.05)
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python -m pytest python/tests/test_shaping_config_dump.py -v
```

Expected: ImportError (`shaping_config_dump` does not exist).

- [ ] **Step 3: Implement the helper**

Create `python/shaping_config_dump.py`:

```python
"""Snapshot shaping/reward coefficients to JSON.

Both the expert reviewer agent and the data-science edit agent need the
exact shaping config used in the most recent training run. This module
reads the same env vars that python/tigphrates_env.py and python/train.py
consult, falls back to their documented defaults, and writes a single
JSON file that downstream agents include in their prompt context.

Why a snapshot file (and not just env vars)? The orchestrator dispatches
agents in a separate shell; env vars exported in the training shell are
not visible there. Snapshotting once at end-of-train fixes the source of
truth and avoids a class of "agent saw default but trainer used override"
bugs.
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any


# Defaults mirror python/tigphrates_env.py:compute_event_shaping_bonus
# and python/train.py reward-shaping block (see SCORE_DELTA_COEF etc).
_EVENT_DEFAULTS = {
    "LEADER_PLACE_BONUS": 0.05,
    "KINGDOM_FORM_BONUS": 0.10,
    "KING_LEADER_BONUS": 0.10,
    "TREASURE_COLLECT_BONUS": 0.15,
    "MONUMENT_BUILD_BONUS": 0.10,
    "SHAPING_DECAY_STEPS": 200000,
}
_SCORE_DEFAULTS = {
    "SCORE_DELTA_COEF": 1.5,
    "MARGIN_DELTA_COEF": 1.0,
    "POTENTIAL_GAMMA_SHAPING_COEF": 1.0,
}
_BC_DEFAULTS = {
    "BC_COEF": 0.1,
}


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def dump_shaping_config(path: pathlib.Path) -> dict[str, Any]:
    """Write current shaping coefficients to `path` as JSON. Returns the dict."""
    data = {
        "event_shaping": {
            "leader_place_bonus": _f("LEADER_PLACE_BONUS", _EVENT_DEFAULTS["LEADER_PLACE_BONUS"]),
            "kingdom_form_bonus": _f("KINGDOM_FORM_BONUS", _EVENT_DEFAULTS["KINGDOM_FORM_BONUS"]),
            "king_leader_bonus": _f("KING_LEADER_BONUS", _EVENT_DEFAULTS["KING_LEADER_BONUS"]),
            "treasure_collect_bonus": _f("TREASURE_COLLECT_BONUS", _EVENT_DEFAULTS["TREASURE_COLLECT_BONUS"]),
            "monument_build_bonus": _f("MONUMENT_BUILD_BONUS", _EVENT_DEFAULTS["MONUMENT_BUILD_BONUS"]),
            "decay_steps": _i("SHAPING_DECAY_STEPS", _EVENT_DEFAULTS["SHAPING_DECAY_STEPS"]),
        },
        "score_shaping": {
            "score_delta_coef": _f("SCORE_DELTA_COEF", _SCORE_DEFAULTS["SCORE_DELTA_COEF"]),
            "margin_delta_coef": _f("MARGIN_DELTA_COEF", _SCORE_DEFAULTS["MARGIN_DELTA_COEF"]),
            "potential_gamma_shaping_coef": _f("POTENTIAL_GAMMA_SHAPING_COEF", _SCORE_DEFAULTS["POTENTIAL_GAMMA_SHAPING_COEF"]),
        },
        "bc": {
            "bc_coef": _f("BC_COEF", _BC_DEFAULTS["BC_COEF"]),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data


def load_shaping_config(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


if __name__ == "__main__":
    import sys
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("models/shaping_config.json")
    dump_shaping_config(out)
    print(f"Wrote {out}")
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python -m pytest python/tests/test_shaping_config_dump.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add python/shaping_config_dump.py python/tests/test_shaping_config_dump.py
git commit -m "shaping_config_dump: snapshot reward coefs for agent prompts"
```

---

## Task 3: Wire the shaping snapshot into `train.py`

**Why:** the orchestrator runs the trace generator and agents *after* `train.py` completes. The snapshot must exist by then. We attach it to the same point that already saves `models/policy_final.pt`.

**Files:**
- Modify: `python/train.py` (one-line import + one-line call near the existing best-save site)

- [ ] **Step 1: Locate the best-save site**

```bash
grep -n "policy_final\|policy_best\|RUN_DIR.*final" python/train.py
```

Expected: a line near the bottom of `train.py` that does `model_path = RUN_DIR / "policy_final.pt"` followed by a torch.save. The exact line number will vary; the spec referenced ~line 1719 at time of writing.

- [ ] **Step 2: Add the import near the top of `train.py`**

Find the imports block (around line 26 — `from tigphrates_env import (...)`) and add a new import on its own line:

```python
from shaping_config_dump import dump_shaping_config
```

- [ ] **Step 3: Call the dump after the final save**

Locate the block that saves `policy_final.pt` (search for `model_path = RUN_DIR / "policy_final.pt"`). Immediately after the `torch.save(...)` for that file, add:

```python
dump_shaping_config(RUN_DIR / "shaping_config.json")
```

- [ ] **Step 4: Smoke-test that train.py still imports**

```bash
python -c "import sys; sys.path.insert(0, 'python'); import train; print('ok')"
```

Expected: `ok` printed, no traceback. (Imports run module-level code; this validates the import succeeds without launching a full training run.)

- [ ] **Step 5: Commit**

```bash
git add python/train.py
git commit -m "train.py: dump shaping_config.json beside policy_final.pt"
```

---

## Task 4: Trace generator — bridge driver skeleton

**Goal of this task:** establish `python/play_traced.py` with a CLI that plays one game vs the heuristic and writes a JSONL file with one minimal line per move (no telemetry yet — just the action label and active player). Telemetry is layered in Task 5.

**Files:**
- Create: `python/play_traced.py`
- Create: `python/tests/test_play_traced.py`

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_play_traced.py`:

```python
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python -m pytest python/tests/test_play_traced.py -v
```

Expected: FAIL — `python/play_traced.py` does not exist.

- [ ] **Step 3: Implement the skeleton**

Create `python/play_traced.py`:

```python
"""Generate rich move traces from a trained checkpoint.

Plays N headless games of the loaded model against the heuristic AI and
(optionally) against the prior pool champion. For each move, writes one
JSON object per line into traces/<out_dir>/game_<i>.jsonl.

This task (Task 4) implements only the bridge driver and a minimal
per-move record. Telemetry (top-K probs, value, shaping breakdown) is
added in Task 5; multi-opponent matching is added in Task 6.

Usage:
    python python/play_traced.py \
        --out-dir traces/<commit> \
        --games-vs-heuristic 3 \
        --games-vs-champion 2 \
        --max-turns 1200

Exit code 0 on success. Non-zero on bridge crash, model load failure,
or argparse error.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Reuse the bridge process manager that TigphratesEnv already uses, so
# this script does not maintain a parallel implementation of the
# Node-subprocess plumbing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tigphrates_env import BridgeProcess  # noqa: E402


PLAYER_COUNT = 2  # match evaluate.py's default; the loop is 2-player today


def _play_single_game(bridge: BridgeProcess, *, model_seat: int, max_turns: int) -> list[dict]:
    """Drive one game against the heuristic. Returns a list of move records.

    The model_seat=0 case means: in the new game we will create, our model
    plays as player 0 and the heuristic plays as player 1. We don't load
    a real model yet (that arrives in Task 5); for now both players use
    the bridge's heuristic ai_action so we exercise the trace path.
    """
    create = bridge.call("create", {"playerCount": PLAYER_COUNT})
    gid = create["gameId"]
    records: list[dict] = []
    try:
        for _ in range(max_turns):
            va = bridge.call("valid_actions", {"gameId": gid})
            if va["turnPhase"] == "gameOver":
                break
            active = va["activePlayer"]
            ai = bridge.call("ai_action", {"gameId": gid})
            action_index = int(ai.get("actionIndex", -1))
            if action_index < 0:
                break
            decoded = bridge.call("decode_action", {"gameId": gid, "actionIndex": action_index})
            records.append({
                "turn": len(records),
                "active_player": active,
                "model_player": active == model_seat,
                "phase": va["turnPhase"],
                "chosen_action": {
                    "action_index": action_index,
                    "label": decoded.get("label", ""),
                },
            })
            bridge.call("step_action", {
                "gameId": gid,
                "action": ai["action"],
                "playerIndex": active,
            })
    finally:
        try:
            bridge.call("delete_game", {"gameId": gid})
        except Exception:
            pass
    return records


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r))
            f.write("\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate move traces from a trained checkpoint.")
    p.add_argument("--out-dir", type=pathlib.Path, required=True,
                   help="Directory to write game_*.jsonl into (e.g. traces/<commit>)")
    p.add_argument("--games-vs-heuristic", type=int, default=3)
    p.add_argument("--games-vs-champion", type=int, default=2,
                   help="Reserved; champion-opponent path is implemented in Task 6.")
    p.add_argument("--max-turns", type=int, default=1200)
    p.add_argument("--model-path", type=pathlib.Path, default=pathlib.Path("models/policy_best.pt"))
    args = p.parse_args(argv)

    total = args.games_vs_heuristic + args.games_vs_champion
    if total <= 0:
        return 0  # zero-game request is a no-op — keeps tests/dry-runs simple

    bridge = BridgeProcess()
    bridge.start()
    try:
        game_idx = 0
        for _ in range(args.games_vs_heuristic):
            records = _play_single_game(bridge, model_seat=0, max_turns=args.max_turns)
            _write_jsonl(args.out_dir / f"game_{game_idx:02d}.jsonl", records)
            game_idx += 1
        # games_vs_champion intentionally unimplemented here; Task 6 adds it.
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest python/tests/test_play_traced.py -v
```

Expected: 2 passed. (`test_one_game_vs_heuristic` exercises the heuristic-vs-heuristic stand-in path; `test_zero_games_is_noop` exercises the no-op early return.)

- [ ] **Step 5: Commit**

```bash
git add python/play_traced.py python/tests/test_play_traced.py
git commit -m "play_traced: bridge-driver skeleton, one game vs heuristic"
```

---

## Task 5: Per-move telemetry — load model + log top-K probs, value, decoded action

**Goal:** replace the heuristic stand-in with real model inference for the model's seat, and enrich each record with the action distribution telemetry the expert agent needs.

**Files:**
- Modify: `python/play_traced.py`
- Modify: `python/tests/test_play_traced.py`

- [ ] **Step 1: Extend the failing test**

Replace `test_one_game_vs_heuristic` in `python/tests/test_play_traced.py` with this stricter version, and keep `test_zero_games_is_noop` unchanged:

```python
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python -m pytest python/tests/test_play_traced.py::test_one_game_vs_heuristic_with_telemetry -v
```

Expected: FAIL — `type_top5` not in record.

- [ ] **Step 3: Implement model loading + telemetry**

Edit `python/play_traced.py`. Add these imports near the top (after the existing imports):

```python
import numpy as np
import torch

from train import (
    PolicyNet,
    NUM_ACTION_TYPES,
    TYPE_PARAM_SIZES,
    TYPE_BASES,
    DEVICE,
    _build_policy_obs,
)
```

Note: `_build_policy_obs` is the helper that converts the bridge's raw observation dict into the model's expected obs format. It already exists in `train.py` (used by `evaluate.py`) — we are not creating a new one. If the actual symbol name is different in the live `train.py`, prefer the symbol used by `evaluate.py` for the same purpose; the import line should be adjusted to match.

Add an action-type name table near the top of the file (these mirror the comment block at `python/train.py:217-235`):

```python
ACTION_TYPE_NAMES = [
    "placeTile",
    "placeLeader",
    "withdrawLeader",
    "placeCatastrophe",
    "swapTiles",
    "pass",
    "commitSupport",
    "chooseWarOrder",
    "buildMonument",
    "declineMonument",
]
assert len(ACTION_TYPE_NAMES) == NUM_ACTION_TYPES
```

Add a helper that loads the model. Note we use `model.train(False)` to put the network in inference mode — equivalent to `model.eval()` but spelled to avoid confusion with Python's built-in code-evaluation function:

```python
def _load_model(model_path: pathlib.Path) -> PolicyNet | None:
    """Load policy weights. Returns None if path is missing — caller falls
    back to fresh-init weights, which still exercises the telemetry path
    in tests where no checkpoint exists yet."""
    model = PolicyNet().to(DEVICE)
    if model_path.exists():
        sd = torch.load(model_path, map_location=DEVICE)
        # train.py has a backward-compat shim for pre-11.1 flat policy_head;
        # if it's exposed as a function, prefer it. Otherwise use load_state_dict
        # with strict=False so legacy checkpoints don't crash trace generation.
        try:
            from train import _adapt_legacy_state_dict  # type: ignore
            sd = _adapt_legacy_state_dict(sd)
        except ImportError:
            pass
        model.load_state_dict(sd, strict=False)
    model.train(False)  # inference mode — no dropout, no BN updates
    return model
```

Add the telemetry-producing inference helper:

```python
@torch.no_grad()
def _model_action_with_telemetry(
    model: PolicyNet,
    obs_raw: dict,
    mask: np.ndarray,
) -> dict:
    """Run one forward pass and return both the chosen action and a
    record fragment containing top-K probs, value estimate, and the
    argmax/sampled action indices.

    The argmax index is what the model would do at evaluation time; the
    sampled index is what stochastic rollouts would do during training.
    Both are surfaced because the spec calls for it (training/inference
    divergence).
    """
    obs = _build_policy_obs(obs_raw)
    obs_batched = {k: torch.tensor(v, device=DEVICE).unsqueeze(0) for k, v in obs.items()}
    type_logits, param_logits, values = model.forward(obs_batched)

    # Hierarchical mask-aware distributions. This mirrors evaluate.py.
    type_dist, param_padded, _ = model.hierarchical_dists(type_logits, param_logits, mask)
    type_probs = torch.softmax(type_dist.logits, dim=-1).squeeze(0).cpu().numpy()
    # Argmax pick (eval-time behavior)
    type_idx_argmax = int(type_probs.argmax())
    chosen_logits_argmax = param_padded[0, type_idx_argmax]
    param_probs_argmax = torch.softmax(chosen_logits_argmax, dim=-1).cpu().numpy()
    param_idx_argmax = int(param_probs_argmax.argmax())
    argmax_action_index = TYPE_BASES[type_idx_argmax] + param_idx_argmax

    # Sampled pick (training-time behavior)
    type_sampled = int(torch.distributions.Categorical(probs=torch.tensor(type_probs)).sample().item())
    chosen_logits_sampled = param_padded[0, type_sampled]
    param_probs_sampled = torch.softmax(chosen_logits_sampled, dim=-1).cpu().numpy()
    param_sampled = int(torch.distributions.Categorical(probs=torch.tensor(param_probs_sampled)).sample().item())
    sampled_action_index = TYPE_BASES[type_sampled] + param_sampled

    # Top-5 type telemetry
    type_top5_idx = type_probs.argsort()[-5:][::-1]
    type_top5 = [
        {"type_name": ACTION_TYPE_NAMES[i], "prob": float(type_probs[i])}
        for i in type_top5_idx if type_probs[i] > 0
    ]
    # Top-5 param within the argmax type
    pp = param_probs_argmax[: TYPE_PARAM_SIZES[type_idx_argmax]]
    param_top5_idx = pp.argsort()[-5:][::-1]
    param_top5 = [
        {"param_index": int(i), "prob": float(pp[i])}
        for i in param_top5_idx if pp[i] > 0
    ]

    return {
        "type_top5": type_top5,
        "param_top5_for_chosen_type": param_top5,
        "value_estimate": float(values.squeeze().item()),
        "argmax_action_index": argmax_action_index,
        "sampled_action_index": sampled_action_index,
    }
```

Replace `_play_single_game` with a version that uses the model on the model's seat and the heuristic on the opponent's seat:

```python
def _play_single_game(
    bridge: BridgeProcess,
    *,
    model: PolicyNet | None,
    model_seat: int,
    max_turns: int,
) -> list[dict]:
    create = bridge.call("create", {"playerCount": PLAYER_COUNT})
    gid = create["gameId"]
    records: list[dict] = []
    try:
        for _ in range(max_turns):
            va = bridge.call("valid_actions", {"gameId": gid})
            if va["turnPhase"] == "gameOver":
                break
            active = va["activePlayer"]
            mask = np.array(va["mask"], dtype=np.int8)
            is_model = active == model_seat and model is not None

            if is_model:
                obs_raw = bridge.call("get_observation", {"gameId": gid, "playerIndex": active})
                tel = _model_action_with_telemetry(model, obs_raw, mask)
                # Use sampled action by default (matches training rollout behavior).
                # Argmax index is logged but not played, so eval-time analysis
                # remains possible from the trace.
                action_index = tel["sampled_action_index"]
                decoded = bridge.call("decode_action", {"gameId": gid, "actionIndex": action_index})
                records.append({
                    "turn": len(records),
                    "active_player": active,
                    "model_player": True,
                    "phase": va["turnPhase"],
                    "chosen_action": {
                        "action_index": action_index,
                        "label": decoded.get("label", ""),
                        "argmax_action_index": tel["argmax_action_index"],
                        "sampled_action_index": tel["sampled_action_index"],
                    },
                    "type_top5": tel["type_top5"],
                    "param_top5_for_chosen_type": tel["param_top5_for_chosen_type"],
                    "value_estimate": tel["value_estimate"],
                })
                bridge.call("step_action", {"gameId": gid, "action": decoded["action"], "playerIndex": active})
            else:
                ai = bridge.call("ai_action", {"gameId": gid})
                action_index = int(ai.get("actionIndex", -1))
                if action_index < 0:
                    break
                decoded = bridge.call("decode_action", {"gameId": gid, "actionIndex": action_index})
                records.append({
                    "turn": len(records),
                    "active_player": active,
                    "model_player": False,
                    "phase": va["turnPhase"],
                    "chosen_action": {
                        "action_index": action_index,
                        "label": decoded.get("label", ""),
                        # Heuristic: argmax/sampled are the same; record both for schema uniformity.
                        "argmax_action_index": action_index,
                        "sampled_action_index": action_index,
                    },
                })
                bridge.call("step_action", {"gameId": gid, "action": ai["action"], "playerIndex": active})
    finally:
        try:
            bridge.call("delete_game", {"gameId": gid})
        except Exception:
            pass
    return records
```

Update `main()` to load the model once and pass it through:

```python
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate move traces from a trained checkpoint.")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--games-vs-heuristic", type=int, default=3)
    p.add_argument("--games-vs-champion", type=int, default=2,
                   help="Reserved; champion-opponent path is implemented in Task 6.")
    p.add_argument("--max-turns", type=int, default=1200)
    p.add_argument("--model-path", type=pathlib.Path, default=pathlib.Path("models/policy_best.pt"))
    args = p.parse_args(argv)

    total = args.games_vs_heuristic + args.games_vs_champion
    if total <= 0:
        return 0

    model = _load_model(args.model_path)
    bridge = BridgeProcess()
    bridge.start()
    try:
        game_idx = 0
        for _ in range(args.games_vs_heuristic):
            records = _play_single_game(bridge, model=model, model_seat=0, max_turns=args.max_turns)
            _write_jsonl(args.out_dir / f"game_{game_idx:02d}.jsonl", records)
            game_idx += 1
    finally:
        bridge.close()
    return 0
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
python -m pytest python/tests/test_play_traced.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add python/play_traced.py python/tests/test_play_traced.py
git commit -m "play_traced: log top-K probs, value, argmax/sampled per move"
```

---

## Task 6: Champion-opponent matches

**Why:** the spec calls for matches against both the heuristic and the prior pool champion. Heuristic-only games miss the failure mode where the model wins against the heuristic but plateaus against itself.

**Files:**
- Modify: `python/play_traced.py`
- Modify: `python/tests/test_play_traced.py`

- [ ] **Step 1: Add a failing test**

Append to `python/tests/test_play_traced.py`:

```python
def test_two_games_vs_champion_when_pool_has_champion(tmp_path, monkeypatch):
    """If a pool exists with at least one snapshot, --games-vs-champion=2
    produces 2 additional trace files for the champion match-up."""
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    # Drop a no-op marker file the script can detect. The actual loader
    # will use highest-Elo entry from elo.json. We provide both.
    (pool_dir / "elo.json").write_text(json.dumps({
        "ratings": {"policy_v0": 1200},
    }))
    # The script will look for policy_v0.pt; copy whatever best file we have,
    # or skip gracefully if no checkpoint exists yet.
    src_best = pathlib.Path("models/policy_best.pt")
    if not src_best.exists():
        pytest.skip("models/policy_best.pt not present — champion path needs a real checkpoint")
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
python -m pytest python/tests/test_play_traced.py::test_two_games_vs_champion_when_pool_has_champion -v
```

Expected: FAIL or SKIP. If it FAILs (a checkpoint is present), the failure mode is "unknown argument --pool-dir" or a missing `opponent_kind` field. SKIP is acceptable in CI environments without a checkpoint.

- [ ] **Step 3: Implement champion loading + tagging**

Add the helper to `python/play_traced.py`:

```python
def _resolve_champion(pool_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the path to the highest-Elo .pt in the pool, or None if
    the pool has no entries. Reads pool/elo.json; falls back to mtime
    when elo.json is missing or empty."""
    if not pool_dir.exists():
        return None
    elo_path = pool_dir / "elo.json"
    if elo_path.exists():
        try:
            elo = json.loads(elo_path.read_text())
            ratings = elo.get("ratings") or {}
            if ratings:
                top = max(ratings.items(), key=lambda kv: kv[1])[0]
                p = pool_dir / f"{top}.pt"
                if p.exists():
                    return p
        except (json.JSONDecodeError, OSError):
            pass
    candidates = sorted(pool_dir.glob("policy_*.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None
```

Add a champion-vs-model game variant. Reuse `_play_single_game` but pass two models — the new "challenger" plays seat 0 and the "champion" plays seat 1. The simplest implementation re-uses the same `_model_action_with_telemetry` helper for the champion seat (telemetry on both seats is harmless and lets the expert see what the champion was thinking too):

```python
def _play_single_game_two_models(
    bridge: BridgeProcess,
    *,
    challenger: PolicyNet,
    champion: PolicyNet,
    challenger_seat: int,
    max_turns: int,
) -> list[dict]:
    create = bridge.call("create", {"playerCount": PLAYER_COUNT})
    gid = create["gameId"]
    records: list[dict] = []
    try:
        for _ in range(max_turns):
            va = bridge.call("valid_actions", {"gameId": gid})
            if va["turnPhase"] == "gameOver":
                break
            active = va["activePlayer"]
            mask = np.array(va["mask"], dtype=np.int8)
            obs_raw = bridge.call("get_observation", {"gameId": gid, "playerIndex": active})
            seat_model = challenger if active == challenger_seat else champion
            tel = _model_action_with_telemetry(seat_model, obs_raw, mask)
            action_index = tel["sampled_action_index"]
            decoded = bridge.call("decode_action", {"gameId": gid, "actionIndex": action_index})
            records.append({
                "turn": len(records),
                "active_player": active,
                "model_player": active == challenger_seat,
                "phase": va["turnPhase"],
                "opponent_kind": "champion",
                "chosen_action": {
                    "action_index": action_index,
                    "label": decoded.get("label", ""),
                    "argmax_action_index": tel["argmax_action_index"],
                    "sampled_action_index": tel["sampled_action_index"],
                },
                "type_top5": tel["type_top5"],
                "param_top5_for_chosen_type": tel["param_top5_for_chosen_type"],
                "value_estimate": tel["value_estimate"],
            })
            bridge.call("step_action", {"gameId": gid, "action": decoded["action"], "playerIndex": active})
    finally:
        try:
            bridge.call("delete_game", {"gameId": gid})
        except Exception:
            pass
    return records
```

Tag heuristic-game records with `opponent_kind: "heuristic"` in the heuristic loop (modify the existing `_play_single_game` to add this field at the same site you create each record — both in the model branch and the heuristic branch).

Wire up `main()` to handle both:

```python
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate move traces from a trained checkpoint.")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--games-vs-heuristic", type=int, default=3)
    p.add_argument("--games-vs-champion", type=int, default=2)
    p.add_argument("--max-turns", type=int, default=1200)
    p.add_argument("--model-path", type=pathlib.Path, default=pathlib.Path("models/policy_best.pt"))
    p.add_argument("--pool-dir", type=pathlib.Path, default=pathlib.Path("models/pool"))
    args = p.parse_args(argv)

    total = args.games_vs_heuristic + args.games_vs_champion
    if total <= 0:
        return 0

    challenger = _load_model(args.model_path)
    bridge = BridgeProcess()
    bridge.start()
    try:
        game_idx = 0
        for _ in range(args.games_vs_heuristic):
            records = _play_single_game(bridge, model=challenger, model_seat=0, max_turns=args.max_turns)
            _write_jsonl(args.out_dir / f"game_{game_idx:02d}.jsonl", records)
            game_idx += 1
        if args.games_vs_champion > 0:
            champ_path = _resolve_champion(args.pool_dir)
            if champ_path is None:
                print(f"[play_traced] no champion in {args.pool_dir}; skipping champion games", file=sys.stderr)
            else:
                champion = _load_model(champ_path)
                for _ in range(args.games_vs_champion):
                    records = _play_single_game_two_models(
                        bridge,
                        challenger=challenger,
                        champion=champion,
                        challenger_seat=0,
                        max_turns=args.max_turns,
                    )
                    _write_jsonl(args.out_dir / f"game_{game_idx:02d}.jsonl", records)
                    game_idx += 1
    finally:
        bridge.close()
    return 0
```

- [ ] **Step 4: Run the tests**

```bash
python -m pytest python/tests/test_play_traced.py -v
```

Expected: all pass (3 if a checkpoint is present, 2 + 1 SKIP otherwise).

- [ ] **Step 5: Commit**

```bash
git add python/play_traced.py python/tests/test_play_traced.py
git commit -m "play_traced: add champion-opponent matches via pool elo.json"
```

---

## Task 7: Expert agent prompt template

**Files:**
- Create: `prompts/rl_expert.md`

- [ ] **Step 1: Write the prompt template**

The orchestrator does simple `{{key}}` substitution before passing to `claude -p`. No template engine — just `sed` substitution.

Create `prompts/rl_expert.md`:

```markdown
You are a Tigris & Euphrates strategic expert reviewing a freshly-trained
RL agent's play. Your job is to find what the agent does poorly and
explain it precisely enough that a data scientist can act on it.

## What you have

- The full game rules: `GAME_RULES.md` (read it first if you haven't).
- {{NUM_GAMES}} game traces in `{{TRACES_DIR}}` named `game_*.jsonl`.
  Each line is one move with: turn, active_player, model_player (true if
  the trained model played this move), phase, chosen_action {label,
  action_index, argmax/sampled indices}, type_top5 [{type_name, prob}],
  param_top5_for_chosen_type, value_estimate, opponent_kind
  (`heuristic` or `champion`).
- The shaping config the model was trained under, in
  `{{SHAPING_CONFIG_PATH}}`. The agent already gets event bonuses for:
  leader placement, kingdom formation, king-color leader, treasure
  collection, monument building. It also gets a per-step Δ(min_score)
  shaping signal and Δ(margin) signal. Read this file before you suggest
  a new incentive — if it is already incentivized, your job is to argue
  the coefficient is wrong, not that it should be added.
- Recent training results: `results.tsv` (commit, win_rate,
  avg_min_score, status, description).

## What I want from you

Write a markdown critique to `{{CRITIQUE_PATH}}`. Structure:

1. **Recurring strategic patterns** — 2–4 bullet points naming concrete
   classes of move you saw the model make repeatedly. Reference at least
   one specific (game_NN.jsonl, turn=X) for each pattern.
2. **Individual blunders worth flagging** — 0–3 turns where the
   argmax/sampled choice was clearly wrong given the position. Quote the
   move label and explain what the better move was.
3. **Hypotheses** — for each pattern, what training signal might be
   responsible (or missing)? Tie back to specific shaping coefficients
   in the config when relevant.
4. **What I'd ask the data scientist to consider** — 1–3 directional
   suggestions. Do NOT propose edits; that's the next agent's job.

## Constraints

- Be specific. "The model plays badly in conflicts" is useless;
  "in 4/5 wars the model committed leaders to support before the
  opponent had committed, giving up information for free (game_02.jsonl
  turn 14, game_03.jsonl turn 9, ...)" is useful.
- Keep it under ~600 words.
- Do not propose `train.py` edits. Stay diagnostic.
- If a sample is too small to support a claim, say so.
```

- [ ] **Step 2: Sanity-check the placeholders**

```bash
grep -o '{{[A-Z_]*}}' prompts/rl_expert.md | sort -u
```

Expected output (exact set): `{{CRITIQUE_PATH}}`, `{{NUM_GAMES}}`, `{{SHAPING_CONFIG_PATH}}`, `{{TRACES_DIR}}`. If any extras appear, the orchestrator will need to substitute them — fix the template.

- [ ] **Step 3: Commit**

```bash
git add prompts/rl_expert.md
git commit -m "prompts/rl_expert: T&E reviewer agent prompt template"
```

---

## Task 8: Data-science agent prompt template

**Files:**
- Create: `prompts/rl_ds.md`

- [ ] **Step 1: Write the prompt template**

Create `prompts/rl_ds.md`:

```markdown
You are an RL data scientist working in the Tigphrates autoresearch
loop. A T&E expert just reviewed the latest training run and wrote a
critique. Your job: edit `python/train.py` with ONE concrete change
that addresses the most pressing issue from the critique.

## What you have

- Critique: `{{CRITIQUE_PATH}}` — read this first.
- Current shaping coefficients: `{{SHAPING_CONFIG_PATH}}`. The model is
  ALREADY being rewarded for: leader placement, kingdom formation,
  king-color leader bonus, treasure collection, monument building,
  Δ(min_score), Δ(margin), BC auxiliary. Adjust an existing coefficient
  before adding a new one.
- Recent results: `{{RESULTS_TSV}}` — last ~10 runs (commit, win_rate,
  avg_min_score, status, description). Avoid repeating an idea that was
  recently tried and discarded.
- The current `python/train.py`. Read it; do not assume its layout.

## Rules

- Edit ONLY `python/train.py`. No other files.
- ONE focused change per run. Do not bundle.
- Code must run: `python python/train.py` (a 5-min budget then a 50-game
  evaluation pass). Imports limited to torch, numpy, gymnasium — no new
  pip deps.
- If the critique points at a coefficient already in
  `{{SHAPING_CONFIG_PATH}}`, prefer adjusting that coefficient
  (potentially via env var override at the top of train.py) over
  introducing a new term.
- Commit message: short imperative summarizing the change. No
  Conventional Commits prefix.

## Output

Make the edit, run `git add python/train.py`, then `git commit -m
"<your short message>"`. That's it. Do not write a summary back to me.
```

- [ ] **Step 2: Verify placeholders**

```bash
grep -o '{{[A-Z_]*}}' prompts/rl_ds.md | sort -u
```

Expected: `{{CRITIQUE_PATH}}`, `{{RESULTS_TSV}}`, `{{SHAPING_CONFIG_PATH}}`.

- [ ] **Step 3: Commit**

```bash
git add prompts/rl_ds.md
git commit -m "prompts/rl_ds: data-science edit agent prompt template"
```

---

## Task 9: Orchestrator integration — `run_experiments.sh`

**Why:** glue all of the above into the existing keep/discard branch. The orchestrator must remain idempotent (re-running on the same commit re-uses cached traces), and any failure in the new path falls back to the existing direct-Claude-edit code so the loop never wedges.

**Files:**
- Modify: `python/run_experiments.sh`

- [ ] **Step 1: Add helper functions near the top of the file**

Open `python/run_experiments.sh`. After the `extract_results()` and `log_result()` helpers (around line 95), add:

```bash
# --- Helper: generate move traces for a kept commit. ---
generate_traces() {
  local commit="$1"
  local out="traces/${commit}"
  if [[ -d "$out" ]] && compgen -G "${out}/game_*.jsonl" >/dev/null; then
    echo ">>> Traces already exist for ${commit}, skipping generation."
    return 0
  fi
  python3 python/play_traced.py \
    --out-dir "$out" \
    --games-vs-heuristic "${TRACED_VS_HEURISTIC:-3}" \
    --games-vs-champion "${TRACED_VS_CHAMPION:-2}" \
    --max-turns "${TRACED_MAX_TURNS:-1200}" \
    >> "$LOG.traces" 2>&1
  local rc=$?
  prune_traces
  return $rc
}

# --- Helper: ask the expert agent for a critique. ---
run_expert_agent() {
  local commit="$1"
  local traces_dir="traces/${commit}"
  local critique="${traces_dir}/critique.md"
  local shaping="models/shaping_config.json"
  local num_games
  num_games=$(ls "$traces_dir"/game_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  [[ "$num_games" -eq 0 ]] && return 1
  local prompt
  prompt=$(sed \
    -e "s|{{TRACES_DIR}}|$traces_dir|g" \
    -e "s|{{NUM_GAMES}}|$num_games|g" \
    -e "s|{{SHAPING_CONFIG_PATH}}|$shaping|g" \
    -e "s|{{CRITIQUE_PATH}}|$critique|g" \
    prompts/rl_expert.md)
  claude --print --output-format text \
    --model claude-sonnet-4-6 \
    --allowedTools Read,Glob,Grep \
    -p "$prompt" > "$critique" 2>> "$LOG.expert"
  [[ -s "$critique" ]]
}

# --- Helper: ask the DS agent to edit train.py. ---
run_ds_agent() {
  local commit="$1"
  local critique="traces/${commit}/critique.md"
  local shaping="models/shaping_config.json"
  [[ -s "$critique" ]] || return 1
  local prompt
  prompt=$(sed \
    -e "s|{{CRITIQUE_PATH}}|$critique|g" \
    -e "s|{{SHAPING_CONFIG_PATH}}|$shaping|g" \
    -e "s|{{RESULTS_TSV}}|$TSV|g" \
    prompts/rl_ds.md)
  claude --print --output-format text \
    --model claude-sonnet-4-6 \
    --allowedTools Edit,Read,Bash,Grep,Glob \
    -p "$prompt" >> "$LOG.ds" 2>&1
  # The DS agent commits its own edit. We just check that something
  # actually changed since the trainer's commit.
  ! git diff --quiet HEAD~1 HEAD -- python/train.py
}

# --- Helper: keep only the most recent N trace directories. ---
prune_traces() {
  local keep="${TRACE_RETENTION:-10}"
  if [[ ! -d traces ]]; then return 0; fi
  local dirs
  mapfile -t dirs < <(ls -1dt traces/*/ 2>/dev/null || true)
  local n="${#dirs[@]}"
  if (( n <= keep )); then return 0; fi
  local i
  for (( i=keep; i<n; i++ )); do
    rm -rf "${dirs[$i]}"
  done
}
```

- [ ] **Step 2: Insert calls into the "keep" branch**

Find the "IMPROVED" branch (around line 230, beginning with `if [[ "$improved" == "yes" ]]; then`). After the `cp models/policy_final.pt models/policy_best.pt` line and the existing closing `fi`, but BEFORE the matching `done` of the for-loop, add:

```bash
        # New: trace + expert + DS chain. Each step has a fallback so a
        # failure here does not wedge the loop.
        if generate_traces "$commit"; then
          if run_expert_agent "$commit" && run_ds_agent "$commit"; then
            echo ">>> Trace+expert+DS chain succeeded; next iteration will run the new edit."
            touch ".autoresearch_ds_pending"
            continue  # skip the legacy ask_claude_for_edit at top of next iter
          else
            echo ">>> Expert/DS agent failed; falling back to legacy direct-edit path."
          fi
        else
          echo ">>> Trace generation failed; falling back to legacy direct-edit path."
        fi
```

The `continue` is important: when the new path succeeds, the DS agent has already produced and committed the next experiment's `train.py` edit. The legacy `ask_claude_for_edit` at the top of the next iteration would clobber it. The marker file (`.autoresearch_ds_pending`) tells the loop to skip the legacy path on the next iteration.

- [ ] **Step 3: Skip `ask_claude_for_edit` when the previous iteration's DS agent already wrote the edit**

At the top of the loop, replace:

```bash
  if [[ "$run" -eq 1 ]]; then
    echo ">>> Baseline run (no changes)"
  else
    echo ">>> Asking Claude for next experiment..."
    ask_claude_for_edit
  fi
```

with:

```bash
  if [[ "$run" -eq 1 ]]; then
    echo ">>> Baseline run (no changes)"
  elif [[ -f ".autoresearch_ds_pending" ]]; then
    echo ">>> Using edit produced by DS agent in previous iteration."
    rm -f ".autoresearch_ds_pending"
  else
    echo ">>> Asking Claude for next experiment (fallback path)..."
    ask_claude_for_edit
  fi
```

- [ ] **Step 4: Add `.autoresearch_ds_pending` to `.gitignore`**

```bash
grep -q "^.autoresearch_ds_pending$" .gitignore 2>/dev/null || echo ".autoresearch_ds_pending" >> .gitignore
```

- [ ] **Step 5: Shell-syntax sanity check**

```bash
bash -n python/run_experiments.sh && echo "syntax ok"
```

Expected: `syntax ok`.

- [ ] **Step 6: Commit**

```bash
git add python/run_experiments.sh .gitignore
git commit -m "run_experiments: insert trace+expert+DS chain with fallback"
```

---

## Task 10: Smoke-test the full loop in dry-run mode

**Why:** a real end-to-end check on a tiny budget catches integration bugs that unit tests miss (template substitution, claude CLI flags, exit codes, file paths).

**Files:** none modified (this task is a manual verification + logging the result).

- [ ] **Step 1: Run the orchestrator in dry-run mode for one experiment**

```bash
./python/run_experiments.sh --dry-run smoke 1
```

Expected behavior:
- Branch `autoresearch/smoke` is created.
- Baseline experiment runs (~30 s).
- A row appears in `results.tsv`.
- If the baseline is "kept" (it usually is on iteration 1), `traces/<commit>/game_*.jsonl` exists, `traces/<commit>/critique.md` is non-empty, and an additional commit appears modifying `python/train.py`.
- If the baseline is "discarded" (no prior win_rate to compare to — the existing logic treats first iteration as "kept" because best=0; verify), the trace path runs anyway.
- Exit code 0.

- [ ] **Step 2: Verify the trace generator output by hand**

```bash
ls traces/
COMMIT=$(ls traces/ | head -1)
ls traces/$COMMIT/
head -1 traces/$COMMIT/game_00.jsonl | python -m json.tool
```

Expected: at least one `game_*.jsonl` file. The first record has the keys named in Task 5's test (`turn`, `active_player`, `model_player`, `phase`, `chosen_action`, plus telemetry on model-player turns).

- [ ] **Step 3: Verify the critique is non-empty and references at least one game/turn**

```bash
wc -l traces/$COMMIT/critique.md
grep -E "game_[0-9]+|turn=?[0-9]+" traces/$COMMIT/critique.md | head
```

Expected: more than ~20 lines and at least one match for "game_NN" or "turn N".

- [ ] **Step 4: Verify the DS agent committed an edit**

```bash
git log --oneline -5
git show HEAD --stat
```

Expected: the most recent commit modifies only `python/train.py` and has a short message authored by the DS agent.

- [ ] **Step 5: Clean up the smoke branch**

```bash
git checkout main
git branch -D autoresearch/smoke
rm -rf traces/*
```

---

## Self-review notes (writer's pass over the plan)

Spec coverage check:
- Component 1 (Trainer): Task 3 attaches the shaping snapshot. ✓
- Component 2 (Trace generator): Tasks 4–6 build it incrementally. ✓
- Component 3 (Expert agent): Task 7 (prompt) + Task 9 (orchestrator wiring). ✓
- Component 4 (DS agent): Task 8 (prompt) + Task 9 (orchestrator wiring). ✓
- Component 5 (Orchestrator): Task 9 inserts the chain (with `prune_traces` colocated); Task 10 smoke-tests. ✓
- Risks/mitigations: fallback to legacy path when any agent step fails (Task 9). ✓
- Testing: unit/integration tests in Tasks 2/4/5/6; smoke test in Task 10. ✓

Type/symbol consistency:
- `dump_shaping_config(path)` defined Task 2, called Task 3. ✓
- `_play_single_game` (heuristic) and `_play_single_game_two_models` (champion) — distinct names; reused via `main()` in Task 6. ✓
- `_load_model`, `_resolve_champion`, `_model_action_with_telemetry`, `_write_jsonl` — all defined where first used. ✓
- Prompt placeholders use the same names in templates and orchestrator substitution: `{{TRACES_DIR}}`, `{{NUM_GAMES}}`, `{{SHAPING_CONFIG_PATH}}`, `{{CRITIQUE_PATH}}`, `{{RESULTS_TSV}}`. ✓
- `prune_traces` referenced in `generate_traces` is defined in the same Task-9 helper block. ✓

Placeholder scan: no TBDs, no "implement later", no "similar to Task N" without code. The one explicit "if symbol name differs in live train.py, use the equivalent from evaluate.py" instruction in Task 5 is a documented coupling note, not a placeholder.
