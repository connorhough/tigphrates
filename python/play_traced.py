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
    p.add_argument("--model-path", type=pathlib.Path, default=pathlib.Path("models/policy_final.pt"),
                   help="Default to policy_final.pt (updated every training run) rather "
                        "than policy_best.pt (which may be from an older architecture). "
                        "Set explicitly via --model-path if you want the all-time best.")
    args = p.parse_args(argv)

    total = args.games_vs_heuristic + args.games_vs_champion
    if total <= 0:
        return 0  # zero-game request is a no-op — keeps tests/dry-runs simple

    # BridgeProcess starts the Node subprocess in __init__ (no separate .start()).
    bridge = BridgeProcess()
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
