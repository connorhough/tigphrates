"""Build an offline policy/value dataset with rollout-improved labels.

The current PPO loop learns from sparse self-generated outcomes. This script
adds a slower but higher-signal path: sample real game states, shortlist legal
actions, estimate each action by heuristic playouts from a cloned bridge state,
and save soft policy targets plus value targets for supervised training.

Example:
    python python/offline_dataset.py --games 20 --states-per-game 12 \
        --rollouts-per-action 2 --top-k-actions 8 --out data/offline_rollouts.npz
"""

from __future__ import annotations

import argparse
import os
import pathlib
import random
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy_server import _build_policy_obs
from tigphrates_env import ACTION_SPACE_SIZE, BridgeProcess
from train import DEVICE, PolicyValueNetwork, _adapt_state_dict, obs_to_tensors


COLORS = ("red", "blue", "green", "black")


@dataclass
class RolloutResult:
    margin: float
    win: float
    root_score: float
    best_opp_score: float
    terminal: bool


def final_score(player: dict) -> float:
    score = player.get("score", {})
    min_score = min(float(score.get(c, 0.0)) for c in COLORS)
    return min_score + float(player.get("treasures", 0.0))


def state_margin(state: dict, root_player: int) -> tuple[float, float, float]:
    totals = [final_score(p) for p in state["players"]]
    root_score = totals[root_player]
    opp_scores = [s for i, s in enumerate(totals) if i != root_player]
    best_opp = max(opp_scores) if opp_scores else 0.0
    return root_score - best_opp, root_score, best_opp


def heuristic_step(bridge: BridgeProcess, game_id: int, active_player: int) -> dict:
    ai = bridge.call("ai_action", {"gameId": game_id})
    action = ai["action"]
    return bridge.call(
        "step_action",
        {"gameId": game_id, "action": action, "playerIndex": int(active_player)},
    )


def play_heuristic_to_end(
    bridge: BridgeProcess,
    game_id: int,
    root_player: int,
    max_steps: int,
) -> RolloutResult:
    terminal = False
    for _ in range(max_steps):
        va = bridge.call("valid_actions", {"gameId": game_id})
        if va.get("turnPhase") == "gameOver":
            terminal = True
            break
        mask = np.asarray(va["mask"], dtype=np.int8)
        if mask.sum() == 0:
            break
        active = int(va.get("activePlayer", 0))
        step = heuristic_step(bridge, game_id, active)
        if bool(step.get("done")):
            terminal = True
            break

    state = bridge.call("get_state", {"gameId": game_id})["state"]
    margin, root_score, best_opp = state_margin(state, root_player)
    if margin > 0:
        win = 1.0
    elif margin == 0:
        win = 0.5
    else:
        win = 0.0
    return RolloutResult(
        margin=float(margin),
        win=float(win),
        root_score=float(root_score),
        best_opp_score=float(best_opp),
        terminal=terminal,
    )


def evaluate_candidate_action(
    bridge: BridgeProcess,
    root_state: dict,
    action_index: int,
    active_player: int,
    max_rollout_steps: int,
) -> RolloutResult:
    loaded = bridge.call("load_state", {"state": root_state})
    scratch_id = int(loaded["gameId"])
    try:
        step = bridge.call(
            "step",
            {
                "gameId": scratch_id,
                "actionIndex": int(action_index),
                "playerIndex": int(active_player),
            },
        )
        if bool(step.get("done")):
            state = bridge.call("get_state", {"gameId": scratch_id})["state"]
            margin, root_score, best_opp = state_margin(state, active_player)
            win = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
            return RolloutResult(margin, win, root_score, best_opp, True)
        return play_heuristic_to_end(
            bridge=bridge,
            game_id=scratch_id,
            root_player=active_player,
            max_steps=max_rollout_steps,
        )
    finally:
        try:
            bridge.call("delete_game", {"gameId": scratch_id})
        except Exception:
            pass


def load_model(path: str | None) -> PolicyValueNetwork | None:
    if not path:
        return None
    import torch

    model = PolicyValueNetwork()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(_adapt_state_dict(state_dict), strict=False)
    model.to(DEVICE)
    model.train(False)
    return model


def flat_policy_prior(model: PolicyValueNetwork | None, raw_obs: dict, mask: np.ndarray) -> np.ndarray:
    if model is None:
        legal = np.where(mask > 0)[0]
        prior = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if len(legal) > 0:
            prior[legal] = 1.0 / len(legal)
        return prior

    import torch
    from train import TYPE_BASES, TYPE_PARAM_SIZES

    obs = _build_policy_obs(raw_obs)
    obs_t = obs_to_tensors(obs)
    with torch.no_grad():
        type_logits, param_logits, _value = model.forward(obs_t)
        type_dist, param_padded, _type_mask = model.hierarchical_dists(
            type_logits, param_logits, mask
        )
        type_probs = torch.softmax(type_dist.logits, dim=-1)
        param_probs = torch.softmax(param_padded, dim=-1)
        param_probs = torch.nan_to_num(param_probs, nan=0.0)

    prior = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    type_np = type_probs[0].detach().cpu().numpy()
    param_np = param_probs[0].detach().cpu().numpy()
    for t, base in enumerate(TYPE_BASES):
        size = TYPE_PARAM_SIZES[t]
        prior[base : base + size] = type_np[t] * param_np[t, :size]
    prior *= mask.astype(np.float32)
    s = float(prior.sum())
    if s > 0:
        prior /= s
    return prior


def choose_candidates(
    bridge: BridgeProcess,
    game_id: int,
    active_player: int,
    mask: np.ndarray,
    prior: np.ndarray,
    top_k: int,
    rng: random.Random,
) -> list[int]:
    legal = np.where(mask > 0)[0].astype(np.int64)
    if len(legal) == 0:
        return []

    candidates: list[int] = []
    heuristic = bridge.call("ai_action", {"gameId": game_id})
    h_idx = int(heuristic.get("actionIndex", -1))
    if 0 <= h_idx < ACTION_SPACE_SIZE and mask[h_idx] > 0:
        candidates.append(h_idx)

    ranked = legal[np.argsort(prior[legal])[::-1]]
    for idx in ranked[: max(0, top_k)]:
        i = int(idx)
        if i not in candidates:
            candidates.append(i)

    if len(candidates) < top_k:
        shuffled = list(map(int, legal))
        rng.shuffle(shuffled)
        for idx in shuffled:
            if idx not in candidates:
                candidates.append(idx)
                if len(candidates) >= top_k:
                    break

    return candidates[: max(1, top_k)]


def soft_policy_target(
    candidates: list[int],
    scores: list[float],
    temperature: float,
) -> tuple[np.ndarray, int]:
    target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    if not candidates:
        return target, -1
    score_arr = np.asarray(scores, dtype=np.float32)
    best_i = int(score_arr.argmax())
    temp = max(temperature, 1e-6)
    weights = np.exp((score_arr - score_arr.max()) / temp)
    weights /= max(float(weights.sum()), 1e-8)
    for action, weight in zip(candidates, weights):
        target[int(action)] = float(weight)
    return target, int(candidates[best_i])


def collect_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    bridge = BridgeProcess()
    model = load_model(args.policy)

    obs_rows: dict[str, list[np.ndarray]] = {
        "board": [],
        "hand": [],
        "hand_seq": [],
        "scores": [],
        "meta": [],
        "conflict": [],
        "leaders": [],
        "opp_scores": [],
        "opp_leaders": [],
    }
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_actions: list[int] = []
    target_values: list[float] = []
    target_margins: list[float] = []
    target_wins: list[float] = []
    root_players: list[int] = []

    labelled = 0
    try:
        for game_idx in range(args.games):
            gid = int(bridge.call("create", {"playerCount": args.player_count})["gameId"])
            states_seen = 0
            try:
                for step_idx in range(args.max_game_steps):
                    va = bridge.call("valid_actions", {"gameId": gid})
                    if va.get("turnPhase") == "gameOver":
                        break
                    mask = np.asarray(va["mask"], dtype=np.int8)
                    if mask.sum() == 0:
                        break
                    active = int(va.get("activePlayer", 0))

                    should_label = (
                        states_seen < args.states_per_game
                        and (step_idx % max(1, args.sample_every) == 0)
                    )
                    if should_label:
                        raw_obs = bridge.call(
                            "get_observation",
                            {"gameId": gid, "playerIndex": active},
                        )
                        policy_obs = _build_policy_obs(raw_obs)
                        prior = flat_policy_prior(model, raw_obs, mask)
                        candidates = choose_candidates(
                            bridge=bridge,
                            game_id=gid,
                            active_player=active,
                            mask=mask,
                            prior=prior,
                            top_k=args.top_k_actions,
                            rng=rng,
                        )
                        root_state = bridge.call("get_state", {"gameId": gid})["state"]
                        action_scores: list[float] = []
                        rollouts: list[RolloutResult] = []
                        for action_index in candidates:
                            per_action: list[RolloutResult] = []
                            for _ in range(args.rollouts_per_action):
                                per_action.append(
                                    evaluate_candidate_action(
                                        bridge=bridge,
                                        root_state=root_state,
                                        action_index=action_index,
                                        active_player=active,
                                        max_rollout_steps=args.max_rollout_steps,
                                    )
                                )
                            avg_margin = float(np.mean([r.margin for r in per_action]))
                            avg_win = float(np.mean([r.win for r in per_action]))
                            action_scores.append(avg_margin + args.win_bonus * avg_win)
                            rollouts.extend(per_action)

                        target, best_action = soft_policy_target(
                            candidates, action_scores, args.target_temperature
                        )
                        best_margin = float(max(r.margin for r in rollouts)) if rollouts else 0.0
                        best_win = float(max(r.win for r in rollouts)) if rollouts else 0.0

                        for key, value in policy_obs.items():
                            obs_rows[key].append(np.asarray(value))
                        masks.append(mask.astype(np.int8))
                        targets.append(target)
                        target_actions.append(best_action)
                        target_values.append(float(np.tanh(best_margin / args.value_scale)))
                        target_margins.append(best_margin)
                        target_wins.append(best_win)
                        root_players.append(active)
                        states_seen += 1
                        labelled += 1

                        if labelled % args.progress_every == 0:
                            print(
                                f"labelled={labelled} game={game_idx + 1}/{args.games} "
                                f"state={states_seen}/{args.states_per_game} "
                                f"best_margin={best_margin:.2f} best_action={best_action}",
                                flush=True,
                            )

                    step = heuristic_step(bridge, gid, active)
                    if bool(step.get("done")):
                        break
                    if states_seen >= args.states_per_game and args.stop_after_game_quota:
                        break
            finally:
                try:
                    bridge.call("delete_game", {"gameId": gid})
                except Exception:
                    pass
    finally:
        try:
            bridge.close()
        except Exception:
            pass

    if labelled == 0:
        raise RuntimeError("no states were labelled; check game/sample settings")

    data: dict[str, np.ndarray] = {
        key: np.stack(rows).astype(np.float32 if key != "hand_seq" else np.int64)
        for key, rows in obs_rows.items()
    }
    data.update(
        {
            "mask": np.stack(masks).astype(np.int8),
            "target_policy": np.stack(targets).astype(np.float32),
            "target_action": np.asarray(target_actions, dtype=np.int64),
            "target_value": np.asarray(target_values, dtype=np.float32),
            "target_margin": np.asarray(target_margins, dtype=np.float32),
            "target_win": np.asarray(target_wins, dtype=np.float32),
            "root_player": np.asarray(root_players, dtype=np.int64),
        }
    )
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/offline_rollouts.npz")
    parser.add_argument("--policy", default=None, help="optional checkpoint used for candidate shortlist priors")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--player-count", type=int, default=2)
    parser.add_argument("--states-per-game", type=int, default=12)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--stop-after-game-quota", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-game-steps", type=int, default=1200)
    parser.add_argument("--max-rollout-steps", type=int, default=800)
    parser.add_argument("--top-k-actions", type=int, default=8)
    parser.add_argument("--rollouts-per-action", type=int, default=2)
    parser.add_argument("--target-temperature", type=float, default=2.5)
    parser.add_argument("--win-bonus", type=float, default=4.0)
    parser.add_argument("--value-scale", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = collect_dataset(args)
    np.savez_compressed(out, **data)
    print(
        f"wrote {out} states={data['target_action'].shape[0]} "
        f"target_action_unique={len(np.unique(data['target_action']))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
