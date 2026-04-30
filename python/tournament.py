"""
Round-robin tournament over the policy pool.

Plays each pair of pool members head-to-head in both seat orders, prints a
leaderboard sorted by overall win rate, and updates persistent Elo ratings
in models/pool/elo.json.

Catches collapse / forgetting that a single agent-vs-pool eval (which always
treats the latest model as the protagonist) cannot — gives a true ladder.

Usage:
    python python/tournament.py [--games-per-pair 4] [--max-turns 1500] \
        [--include-heuristic]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tigphrates_env import TigphratesEnv
from train import PolicyValueNetwork, make_policy_fn
from evaluate import (
    ELO_FILE,
    ELO_AGENT_KEY,
    ELO_BASELINE,
    _load_elo_table,
    _save_elo_table,
    update_elo_pair,
)

POOL_DIR = pathlib.Path("models/pool")
HEURISTIC_KEY = "_heuristic"


def _load_model(path: pathlib.Path) -> torch.nn.Module:
    m = PolicyValueNetwork()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.train(False)
    from train import DEVICE
    m.to(DEVICE)
    return m


def play_match(
    agent_policy,
    opp_policy_fn,
    n_games: int,
    max_turns: int,
    player_count: int = 2,
) -> int:
    """Play n_games as agent (player 0) vs opp_policy_fn (players 1..N-1).
    `opp_policy_fn=None` means use the env's built-in heuristic AI for all
    non-agent seats. With player_count > 2, the same opponent_policy controls
    every non-agent seat — equivalent to "agent vs N-1 copies of opponent."
    Returns number of games where the agent's final score is strictly
    greater than the best opponent's."""
    env = TigphratesEnv(
        player_count=player_count,
        agent_player=0,
        max_turns=max_turns,
        opponent_policy=opp_policy_fn,
    )
    wins = 0
    for _ in range(n_games):
        obs, info = env.reset()
        terminated = truncated = False
        while not terminated and not truncated:
            mask = env.action_mask()
            if mask.sum() == 0:
                break
            action = agent_policy(obs, mask)
            obs, _, terminated, truncated, info = env.step(action)
        scores = info.get("scores")
        if scores:
            agent_score = scores[0]["min"] + scores[0]["treasures"]
            opp_score = max(s["min"] + s["treasures"] for s in scores[1:])
            if agent_score > opp_score:
                wins += 1
    env.close()
    return wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-pair", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=1500)
    parser.add_argument("--include-heuristic", action="store_true",
                        help="Add the built-in simpleAI as an extra contestant")
    parser.add_argument("--player-count", type=int, default=2,
                        help="Players per game (2-4). With N>2, each match is "
                             "agent vs (N-1) copies of opponent.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Run match pairs in parallel via ThreadPoolExecutor. "
                             "Each match opens its own Node bridge subprocess, "
                             "so keep N <= cpu_count // 4 on a Mac mini.")
    args = parser.parse_args()
    if not 2 <= args.player_count <= 4:
        print("--player-count must be 2, 3, or 4", file=sys.stderr)
        sys.exit(2)

    pool_paths = sorted(POOL_DIR.glob("policy_*.pt"))
    if len(pool_paths) < 2 and not args.include_heuristic:
        print("Pool has <2 members and --include-heuristic not set; nothing to do.")
        sys.exit(2)

    # Load all models. Heuristic is represented by a `None` policy_fn slot —
    # the env interprets that as "use the built-in heuristic AI."
    contestants: list[tuple[str, "callable | None"]] = []
    for p in pool_paths:
        contestants.append((p.stem, make_policy_fn(_load_model(p))))
    if args.include_heuristic:
        contestants.append((HEURISTIC_KEY, None))

    n = len(contestants)
    names = [c[0] for c in contestants]
    print(f"Round-robin: {n} contestants × {args.games_per_pair} games per pair × 2 seats "
          f"(concurrency={args.concurrency})")

    wins = {a: {b: 0 for b in names} for a in names}
    games = {a: {b: 0 for b in names} for a in names}

    # Build the list of match jobs. (i, j) plays games_per_pair games with na
    # as agent and nb as opponent; the result also implies (games_per_pair - w)
    # for the symmetric seat. (j, i) is a SEPARATE job that plays its own set.
    jobs: list[tuple[str, callable, str, callable]] = []
    for i, (na, pa) in enumerate(contestants):
        for j, (nb, pb) in enumerate(contestants):
            if i == j:
                continue
            if pa is None:
                continue  # heuristic-as-agent: captured via symmetric (j, i)
            jobs.append((na, pa, nb, pb))

    def _run(job):
        na, pa, nb, pb = job
        w = play_match(pa, pb, args.games_per_pair, args.max_turns, args.player_count)
        return na, nb, w

    t_start = time.time()
    if args.concurrency <= 1:
        for k, job in enumerate(jobs):
            na, nb, w = _run(job)
            wins[na][nb] += w
            games[na][nb] += args.games_per_pair
            wins[nb][na] += args.games_per_pair - w
            games[nb][na] += args.games_per_pair
            print(f"  match {k + 1}/{len(jobs)} done ({time.time() - t_start:.0f}s)", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(_run, j) for j in jobs]
            for k, fut in enumerate(as_completed(futures)):
                na, nb, w = fut.result()
                wins[na][nb] += w
                games[na][nb] += args.games_per_pair
                wins[nb][na] += args.games_per_pair - w
                games[nb][na] += args.games_per_pair
                print(f"  match {k + 1}/{len(jobs)} done ({time.time() - t_start:.0f}s)", flush=True)

    # Leaderboard.
    leaderboard = []
    for nm in names:
        total_w = sum(wins[nm][n2] for n2 in names if n2 != nm)
        total_g = sum(games[nm][n2] for n2 in names if n2 != nm)
        leaderboard.append((nm, total_w / max(total_g, 1), total_w, total_g))
    leaderboard.sort(key=lambda x: -x[1])

    print("\n=== Tournament Leaderboard ===")
    for nm, wr, w, g in leaderboard:
        print(f"  {nm:32s}  {wr:.3f}  ({w}/{g})")

    # Update persistent Elo ratings symmetrically.
    table = _load_elo_table(POOL_DIR)
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if j <= i:
                continue
            wa = wins[na][nb]
            wb = wins[nb][na]
            if wa + wb == 0:
                continue
            update_elo_pair(table, na, nb, wa, wb)

    # Project agent rating from highest-rated pool member (or pool top) for
    # easy reading vs runs of train.py's persistent_elo line.
    if names:
        ratings = {nm: table.get(nm, ELO_BASELINE) for nm in names}
        top = max(ratings.values())
        table[ELO_AGENT_KEY] = top  # latest top of the league
    _save_elo_table(POOL_DIR, table)

    print(f"\nUpdated Elo ratings in {POOL_DIR / ELO_FILE}")
    sorted_ratings = sorted(((nm, table[nm]) for nm in names), key=lambda x: -x[1])
    for nm, r in sorted_ratings:
        print(f"  {nm:32s}  {r:7.1f}")


if __name__ == "__main__":
    main()
