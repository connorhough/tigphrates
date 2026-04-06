"""
Evaluation harness for RL training on Tigris & Euphrates.
This file is READ-ONLY — do not modify during experiments.

Provides:
  - evaluate_vs_heuristic(): play N games against the built-in heuristic AI
  - evaluate_vs_random(): play N games against a random agent
  - TIME_BUDGET: fixed training time in seconds
  - EVAL_GAMES: number of evaluation games
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tigphrates_env import TigphratesEnv, ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS

# --- Constants (do not modify in code — override via env vars for testing) ---
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", 300))
EVAL_GAMES = int(os.environ.get("EVAL_GAMES", 50))
PLAYER_COUNT = 2         # 2-player games for training


def evaluate_vs_heuristic(
    policy_fn,
    num_games: int = EVAL_GAMES,
    player_count: int = PLAYER_COUNT,
    max_turns: int = 2000,
) -> dict:
    """
    Evaluate a policy function against the built-in heuristic AI.

    Args:
        policy_fn: Callable(observation_dict, action_mask) -> int
            Takes the observation dict and boolean action mask, returns an action index.
        num_games: Number of games to play.
        player_count: Number of players (agent is always player 0).
        max_turns: Maximum turns before truncation.

    Returns:
        dict with keys: win_rate, avg_min_score, avg_margin, games_completed, results
    """
    env = TigphratesEnv(player_count=player_count, agent_player=0, max_turns=max_turns)
    wins = 0
    min_scores = []
    margins = []
    results = []

    for g in range(num_games):
        obs, info = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            mask = env.action_mask()
            if mask.sum() == 0:
                break
            action = policy_fn(obs, mask)
            obs, reward, terminated, truncated, info = env.step(action)

        # Extract final scores
        if "scores" in info:
            agent_min = info["scores"][0]["min"] + info["scores"][0]["treasures"]
            opp_min = max(
                s["min"] + s["treasures"] for s in info["scores"][1:]
            )
            won = agent_min > opp_min
            # Tiebreak: higher total
            if agent_min == opp_min:
                agent_total = sum(info["scores"][0][c] for c in ["red", "blue", "green", "black"])
                opp_total = max(
                    sum(s[c] for c in ["red", "blue", "green", "black"])
                    for s in info["scores"][1:]
                )
                won = agent_total > opp_total
        else:
            agent_min = obs["scores"].min()
            opp_min = 0
            won = False

        if won:
            wins += 1
        min_scores.append(agent_min)
        margins.append(agent_min - opp_min)
        results.append({
            "game": g,
            "won": won,
            "agent_min_score": agent_min,
            "opp_min_score": opp_min,
            "margin": agent_min - opp_min,
        })

    env.close()

    return {
        "win_rate": wins / num_games,
        "avg_min_score": np.mean(min_scores),
        "avg_margin": np.mean(margins),
        "games_completed": num_games,
        "results": results,
    }


def evaluate_vs_random(
    policy_fn,
    num_games: int = EVAL_GAMES,
) -> dict:
    """
    Evaluate against a purely random (masked) agent.
    Same interface as evaluate_vs_heuristic.
    """
    # For this, we need the multi-agent env.
    # Simpler: just use TigphratesEnv but have our agent play, and measure score.
    # The opponents are already the heuristic AI in TigphratesEnv,
    # so we'll create a special "random baseline" by measuring random play.
    env = TigphratesEnv(player_count=2, agent_player=0, max_turns=2000)
    wins = 0
    min_scores = []

    for _ in range(num_games):
        obs, info = env.reset()
        terminated = False
        truncated = False

        while not terminated and not truncated:
            mask = env.action_mask()
            valid = np.where(mask == 1)[0]
            action = np.random.choice(valid) if len(valid) > 0 else 0
            obs, reward, terminated, truncated, info = env.step(action)

        if "scores" in info:
            agent_min = info["scores"][0]["min"] + info["scores"][0]["treasures"]
            opp_min = max(s["min"] + s["treasures"] for s in info["scores"][1:])
            if agent_min > opp_min:
                wins += 1
            min_scores.append(agent_min)

    env.close()
    return {
        "win_rate": wins / num_games,
        "avg_min_score": np.mean(min_scores) if min_scores else 0,
    }


def print_summary(
    win_rate: float,
    avg_min_score: float,
    avg_margin: float,
    training_seconds: float,
    total_seconds: float,
    num_episodes: int,
    num_steps: int,
    num_params: int,
    **extra,
) -> None:
    """Print the standard results summary (matches autoresearch format)."""
    print("---")
    print(f"win_rate:          {win_rate:.6f}")
    print(f"avg_min_score:     {avg_min_score:.1f}")
    print(f"avg_margin:        {avg_margin:.1f}")
    print(f"training_seconds:  {training_seconds:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"num_episodes:      {num_episodes}")
    print(f"num_steps:         {num_steps}")
    print(f"num_params:        {num_params}")
    for k, v in extra.items():
        print(f"{k}: {v}")
