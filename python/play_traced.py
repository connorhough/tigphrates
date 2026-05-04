"""Generate rich move traces from a trained checkpoint.

Plays N headless games of the loaded model against the heuristic AI and
(optionally) against the prior pool champion. For each move, writes one
JSON object per line into traces/<out_dir>/game_<i>.jsonl.

Task 5 adds: real model inference on the model's seat, and per-move
telemetry (top-K type probs, value estimate, argmax/sampled indices).
Multi-opponent matching (champion) is added in Task 6.

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

import numpy as np
import torch

# train.py owns the network class, action-layout constants, and the
# legacy-state-dict adapter. policy_server.py owns _build_policy_obs
# (it lives there because the server has the same need as we do —
# convert a raw bridge observation into the network's expected shape).
from train import (
    PolicyValueNetwork,
    NUM_ACTION_TYPES,
    TYPE_PARAM_SIZES,
    TYPE_BASES,
    DEVICE,
    _adapt_state_dict,
)
from policy_server import _build_policy_obs


PLAYER_COUNT = 2  # match evaluate.py's default; the loop is 2-player today

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


def _load_model(model_path: pathlib.Path) -> PolicyValueNetwork:
    """Load policy weights. If the checkpoint is missing, returns a
    freshly-initialized network — telemetry from random weights is
    still useful for exercising the trace path in tests where no
    checkpoint exists yet.

    Loader mirrors policy_server.py:102-107: load to CPU first, run
    _adapt_state_dict (handles pre-11.1 flat policy_head -> hierarchical
    type_head/param_head migration), load with strict=False, then move
    to DEVICE.
    """
    model = PolicyValueNetwork()
    if model_path.exists():
        sd = torch.load(model_path, map_location="cpu")
        sd = _adapt_state_dict(sd)
        result = model.load_state_dict(sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(
                f"[play_traced] WARNING: state_dict mismatch loading {model_path}: "
                f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}. "
                "Check that the checkpoint matches the current architecture.",
                file=sys.stderr,
            )
    model.train(False)
    model.to(DEVICE)
    return model


@torch.no_grad()
def _model_action_with_telemetry(
    model: PolicyValueNetwork,
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


def _play_single_game(
    bridge: BridgeProcess,
    *,
    model: PolicyValueNetwork | None,
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


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r))
            f.write("\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate move traces from a trained checkpoint.")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
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
        return 0

    model = _load_model(args.model_path)
    bridge = BridgeProcess()
    try:
        game_idx = 0
        for _ in range(args.games_vs_heuristic):
            records = _play_single_game(bridge, model=model, model_seat=0, max_turns=args.max_turns)
            _write_jsonl(args.out_dir / f"game_{game_idx:02d}.jsonl", records)
            game_idx += 1
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
