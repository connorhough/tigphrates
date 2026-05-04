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
