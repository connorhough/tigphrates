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

import json
import os
import pathlib
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tigphrates_env import TigphratesEnv, ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS

# --- Constants (do not modify in code — override via env vars for testing) ---
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", 300))
EVAL_GAMES = int(os.environ.get("EVAL_GAMES", 50))
PLAYER_COUNT = int(os.environ.get("PLAYER_COUNT", 2))  # 2/3/4 players


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


def evaluate_vs_pool(
    policy_fn,
    opponent_loader,
    opponent_paths: list[str],
    games_per_opponent: int = 4,
    player_count: int = PLAYER_COUNT,
    max_turns: int = 2000,
) -> dict:
    """
    Evaluate a policy against every member of a checkpoint pool.

    Args:
        policy_fn: Callable(observation_dict, action_mask) -> int (the agent).
        opponent_loader: Callable(path: str) -> torch.nn.Module
            Loads a saved policy network. Caller passes a function that knows
            how to construct the right model class.
        opponent_paths: list of saved checkpoint paths.
        games_per_opponent: how many games to play vs each opponent.

    Returns:
        dict with vs_pool_win_rate, n_opponents, per_opponent map, total_games.
        If the pool is empty, returns vs_pool_win_rate=None.
    """
    import torch  # local import keeps base eval lightweight
    if not opponent_paths:
        return {
            "vs_pool_win_rate": None,
            "n_opponents": 0,
            "per_opponent": {},
            "total_games": 0,
        }

    total_wins = 0
    total_games = 0
    per_opponent: dict[str, float] = {}

    for opp_path in opponent_paths:
        opp_model = opponent_loader(opp_path)

        def opp_policy_fn(obs, action_mask, _m=opp_model):
            # Tensorize obs; rely on the model's forward signature matching
            # the training network. Lazy import to avoid coupling.
            from train import obs_to_tensors  # circular at import-time, fine here
            obs_tensor = obs_to_tensors(obs)
            with torch.no_grad():
                action, _, _, _ = _m.get_action_and_value(obs_tensor, action_mask)
            return action.item()

        env = TigphratesEnv(
            player_count=player_count,
            agent_player=0,
            max_turns=max_turns,
            opponent_policy=opp_policy_fn,
        )
        wins = 0
        for _ in range(games_per_opponent):
            obs, info = env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                mask = env.action_mask()
                if mask.sum() == 0:
                    break
                action = policy_fn(obs, mask)
                obs, reward, terminated, truncated, info = env.step(action)
            scores_info = info.get("scores")
            if scores_info:
                agent_min = scores_info[0]["min"] + scores_info[0]["treasures"]
                opp_min = max(s["min"] + s["treasures"] for s in scores_info[1:])
                if agent_min > opp_min:
                    wins += 1
        env.close()

        per_opponent[opp_path] = wins / games_per_opponent
        total_wins += wins
        total_games += games_per_opponent

    elo = _compute_elo(per_opponent)

    return {
        "vs_pool_win_rate": total_wins / max(total_games, 1),
        "n_opponents": len(opponent_paths),
        "per_opponent": per_opponent,
        "total_games": total_games,
        "elo": elo,
    }


def _compute_elo(per_opponent: dict[str, float], k: float = 32.0, baseline: float = 1500.0) -> float:
    """
    Compute the agent's Elo rating treating each pool opponent as fixed at the
    baseline rating. Iterates Elo updates over the per-opponent win rates so
    later matches use the freshly updated rating — gives a single scalar that
    moves intuitively when win rate goes up or down.

    Returns the resulting rating (baseline = no progress).
    """
    rating = baseline
    for _, win_rate in per_opponent.items():
        expected = 1.0 / (1.0 + 10 ** ((baseline - rating) / 400))
        rating += k * (win_rate - expected)
    return rating


# --- Persistent Elo ladder ---------------------------------------------------

ELO_FILE = "elo.json"
ELO_AGENT_KEY = "_agent"
ELO_BASELINE = 1500.0
ELO_K = 16.0


def _load_elo_table(pool_dir: pathlib.Path) -> dict[str, float]:
    p = pool_dir / ELO_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_elo_table(pool_dir: pathlib.Path, table: dict[str, float]) -> None:
    pool_dir.mkdir(parents=True, exist_ok=True)
    (pool_dir / ELO_FILE).write_text(json.dumps(table, indent=2, sort_keys=True))


def update_elo_pair(
    table: dict[str, float],
    key_a: str,
    key_b: str,
    wins_a: int,
    wins_b: int,
    k: float = ELO_K,
    baseline: float = ELO_BASELINE,
) -> dict[str, float]:
    """Symmetric pairwise Elo update — applies game-by-game so a 10-0 sweep
    moves ratings smoothly rather than as one giant step. Mutates and
    returns the table."""
    ra = table.get(key_a, baseline)
    rb = table.get(key_b, baseline)
    for _ in range(wins_a):
        expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
        ra += k * (1.0 - expected_a)
        rb += k * (0.0 - (1.0 - expected_a))
    for _ in range(wins_b):
        expected_a = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
        ra += k * (0.0 - expected_a)
        rb += k * (1.0 - (1.0 - expected_a))
    table[key_a] = ra
    table[key_b] = rb
    return table


def update_persistent_elo(
    pool_dir: pathlib.Path,
    per_opponent: dict[str, float],
    games_per_opponent: int,
    k: float = ELO_K,
) -> dict[str, float]:
    """
    Update persistent Elo ratings for the agent and each pool member.

    Treats each game played as a separate Elo update so a single 0.0 vs 1.0
    sweep doesn't swing one opponent's rating by hundreds of points.

    Returns the updated table. Agent rating is stored under ELO_AGENT_KEY.
    """
    table = _load_elo_table(pool_dir)
    agent_rating = table.get(ELO_AGENT_KEY, ELO_BASELINE)

    for opp_path, win_rate in per_opponent.items():
        opp_key = pathlib.Path(opp_path).name
        opp_rating = table.get(opp_key, ELO_BASELINE)
        # Distribute the aggregate win rate over discrete games — each game
        # contributes a 1.0 (win) or 0.0 (loss) sample. Approximate by
        # rounding wins from the win rate.
        wins = int(round(win_rate * games_per_opponent))
        losses = games_per_opponent - wins
        for _ in range(wins):
            expected = 1.0 / (1.0 + 10 ** ((opp_rating - agent_rating) / 400))
            agent_rating += k * (1.0 - expected)
            opp_rating += k * (0.0 - (1.0 - expected))
        for _ in range(losses):
            expected = 1.0 / (1.0 + 10 ** ((opp_rating - agent_rating) / 400))
            agent_rating += k * (0.0 - expected)
            opp_rating += k * (1.0 - (1.0 - expected))
        table[opp_key] = opp_rating

    table[ELO_AGENT_KEY] = agent_rating
    _save_elo_table(pool_dir, table)
    return table


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
