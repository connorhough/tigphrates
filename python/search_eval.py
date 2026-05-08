"""Evaluate a checkpoint with shallow rollout search against heuristic AI.

This is a deliberately expensive diagnostic/evaluation path. At each agent
decision it shortlists legal actions with the policy prior, evaluates them by
cloning the bridge state and running heuristic playouts, then takes the best
estimated action. It answers whether the model's priors are useful when paired
with search, rather than only judging one greedy forward pass.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offline_dataset import (
    choose_candidates,
    evaluate_candidate_action,
    flat_policy_prior,
    heuristic_step,
    load_model,
    state_margin,
)
from tigphrates_env import BridgeProcess


def search_pick_action(
    bridge: BridgeProcess,
    game_id: int,
    active_player: int,
    model,
    top_k_actions: int,
    rollouts_per_action: int,
    max_rollout_steps: int,
    win_bonus: float,
    rng: random.Random,
) -> int:
    va = bridge.call("valid_actions", {"gameId": game_id})
    mask = np.asarray(va["mask"], dtype=np.int8)
    raw_obs = bridge.call("get_observation", {"gameId": game_id, "playerIndex": active_player})
    prior = flat_policy_prior(model, raw_obs, mask)
    candidates = choose_candidates(
        bridge=bridge,
        game_id=game_id,
        active_player=active_player,
        mask=mask,
        prior=prior,
        top_k=top_k_actions,
        rng=rng,
    )
    root_state = bridge.call("get_state", {"gameId": game_id})["state"]
    best_action = candidates[0]
    best_score = -float("inf")
    for action in candidates:
        scores = []
        for _ in range(rollouts_per_action):
            result = evaluate_candidate_action(
                bridge=bridge,
                root_state=root_state,
                action_index=action,
                active_player=active_player,
                max_rollout_steps=max_rollout_steps,
            )
            scores.append(result.margin + win_bonus * result.win)
        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_action = int(action)
    return int(best_action)


def evaluate(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    bridge = BridgeProcess()
    model = load_model(args.model)

    wins = 0
    margins = []
    min_scores = []
    results = []
    try:
        for game_idx in range(args.games):
            seat = game_idx % args.player_count if args.rotate_seats else args.agent_player
            gid = int(bridge.call("create", {"playerCount": args.player_count})["gameId"])
            done = False
            try:
                for turn in range(args.max_turns):
                    va = bridge.call("valid_actions", {"gameId": gid})
                    if va.get("turnPhase") == "gameOver":
                        done = True
                        break
                    mask = np.asarray(va["mask"], dtype=np.int8)
                    if mask.sum() == 0:
                        break
                    active = int(va.get("activePlayer", 0))
                    if active == seat:
                        action = search_pick_action(
                            bridge=bridge,
                            game_id=gid,
                            active_player=active,
                            model=model,
                            top_k_actions=args.top_k_actions,
                            rollouts_per_action=args.rollouts_per_action,
                            max_rollout_steps=args.max_rollout_steps,
                            win_bonus=args.win_bonus,
                            rng=rng,
                        )
                        step = bridge.call(
                            "step",
                            {"gameId": gid, "actionIndex": action, "playerIndex": active},
                        )
                    else:
                        step = heuristic_step(bridge, gid, active)
                    if bool(step.get("done")):
                        done = True
                        break

                state = bridge.call("get_state", {"gameId": gid})["state"]
                margin, root_score, best_opp = state_margin(state, seat)
                won = margin > 0.0
                wins += int(won)
                margins.append(margin)
                min_scores.append(root_score)
                result = {
                    "game": game_idx,
                    "seat": seat,
                    "won": bool(won),
                    "margin": float(margin),
                    "agent_min_score": float(root_score),
                    "opp_min_score": float(best_opp),
                    "done": bool(done),
                }
                results.append(result)
                print(
                    f"game={game_idx + 1}/{args.games} seat={seat} "
                    f"won={int(won)} margin={margin:.1f} score={root_score:.1f}",
                    flush=True,
                )
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

    return {
        "win_rate": wins / max(1, args.games),
        "avg_min_score": float(np.mean(min_scores)) if min_scores else 0.0,
        "avg_margin": float(np.mean(margins)) if margins else 0.0,
        "games_completed": args.games,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/policy_best.pt")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--player-count", type=int, default=2)
    parser.add_argument("--agent-player", type=int, default=0)
    parser.add_argument("--rotate-seats", action="store_true")
    parser.add_argument("--top-k-actions", type=int, default=6)
    parser.add_argument("--rollouts-per-action", type=int, default=2)
    parser.add_argument("--max-rollout-steps", type=int, default=600)
    parser.add_argument("--max-turns", type=int, default=1200)
    parser.add_argument("--win-bonus", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate(args)
    print(
        f"win_rate={metrics['win_rate']:.3f} "
        f"avg_min_score={metrics['avg_min_score']:.2f} "
        f"avg_margin={metrics['avg_margin']:.2f} "
        f"games={metrics['games_completed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
