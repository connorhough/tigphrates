"""Tests for leader-placement / kingdom-formation shaping rewards.

Shaping is computed by a pure helper in tigphrates_env.py so it can be
unit-tested without spawning the Node bridge.

Action layout (must match src/bridge/encoder.ts):
- placeTile:   indices [0,        704)
- placeLeader: indices [704,     1408)   <- 4 colors × 176 cells
- everything else: indices [1408, 1728)
"""
import os
import numpy as np
import pytest


# --- Fixtures: pure observation builders that mimic the bridge encoding -----

BOARD_CHANNELS = 15
BOARD_ROWS = 11
BOARD_COLS = 16


def _empty_obs():
    """Minimal observation dict shaped like the env's _get_obs output."""
    return {
        "board": np.zeros((BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32),
        "scores": np.zeros(4, dtype=np.int32),
        "opp_scores": np.zeros(4, dtype=np.float32),
    }


def _board_with_leader_at(row: int, col: int, color_idx: int = 0):
    """Build a board observation with a leader of `color_idx` at (row, col).
    Channels 4-7 are leaders by color (red=4, blue=5, green=6, black=7);
    cell value encodes 1 + ownerPlayerIndex (>= 1)."""
    obs = _empty_obs()
    obs["board"][4 + color_idx, row, col] = 1.0  # owner=0 → value 1
    return obs


def _board_with_leader_adjacent_to_tile(row: int, col: int, color_idx: int = 0):
    """Leader at (row,col) with a tile at (row, col+1)."""
    obs = _board_with_leader_at(row, col, color_idx)
    # Channels 0-3 are tiles by color
    obs["board"][color_idx, row, col + 1] = 1.0
    return obs


# --- Action index helpers ---------------------------------------------------

def _placetile_idx(color: int, row: int, col: int) -> int:
    return color * BOARD_ROWS * BOARD_COLS + row * BOARD_COLS + col


def _placeleader_idx(color: int, row: int, col: int) -> int:
    base = 4 * BOARD_ROWS * BOARD_COLS  # 704
    return base + color * BOARD_ROWS * BOARD_COLS + row * BOARD_COLS + col


# --- Tests ------------------------------------------------------------------

def test_leader_placement_returns_bonus(monkeypatch):
    """A leader placement awards the LEADER_PLACE_BONUS shaping reward."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.0")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_leader_shaping_bonus

    prev_obs = _empty_obs()
    next_obs = _board_with_leader_at(5, 5, color_idx=0)
    action = _placeleader_idx(color=0, row=5, col=5)

    bonus = compute_leader_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
    )
    assert bonus == pytest.approx(0.05)


def test_non_leader_placement_returns_zero(monkeypatch):
    """Placing a regular tile gives zero shaping bonus."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.1")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_leader_shaping_bonus

    prev_obs = _empty_obs()
    next_obs = _empty_obs()
    next_obs["board"][0, 5, 5] = 1.0  # red tile placed
    action = _placetile_idx(color=0, row=5, col=5)

    bonus = compute_leader_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
    )
    assert bonus == 0.0


def test_kingdom_formation_bonus(monkeypatch):
    """Placing a leader adjacent to a tile awards both place + kingdom bonuses."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.1")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_leader_shaping_bonus

    prev_obs = _empty_obs()
    # Pre-existing tile at (5,6); leader placed at (5,5)
    prev_obs["board"][0, 5, 6] = 1.0
    next_obs = _board_with_leader_adjacent_to_tile(5, 5, color_idx=0)
    action = _placeleader_idx(color=0, row=5, col=5)

    bonus = compute_leader_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
    )
    # Both place + kingdom bonus apply.
    assert bonus == pytest.approx(0.15)


def test_cap_of_4_bonuses_per_game(monkeypatch):
    """The bonus is capped at LEADER_PLACE_CAP (default 4) per game."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.0")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_leader_shaping_bonus

    prev_obs = _empty_obs()
    next_obs = _board_with_leader_at(5, 5, color_idx=0)
    action = _placeleader_idx(color=0, row=5, col=5)

    # 5th leader placement should return zero.
    bonus_5th = compute_leader_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=4,  # already 4 placements awarded
    )
    assert bonus_5th == 0.0

    # 4th placement (index 3 already counted) still awards bonus.
    bonus_4th = compute_leader_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=3,
    )
    assert bonus_4th == pytest.approx(0.05)


def test_decay_reaches_zero(monkeypatch):
    """At SHAPING_DECAY_STEPS the bonus has linearly decayed to zero."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.1")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "1000")

    from tigphrates_env import compute_leader_shaping_bonus

    prev_obs = _empty_obs()
    next_obs = _board_with_leader_at(5, 5, color_idx=0)
    action = _placeleader_idx(color=0, row=5, col=5)

    bonus_at_decay = compute_leader_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=1000,
        leader_placements_so_far=0,
    )
    assert bonus_at_decay == 0.0

    # Past decay, still zero (no negative shaping).
    bonus_past = compute_leader_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=2000,
        leader_placements_so_far=0,
    )
    assert bonus_past == 0.0

    # Halfway through decay, bonus should be ~half.
    bonus_half = compute_leader_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=500,
        leader_placements_so_far=0,
    )
    assert bonus_half == pytest.approx(0.025, abs=1e-6)


def test_disabled_bonuses_match_baseline(monkeypatch):
    """With both bonuses set to 0, shaping is a no-op (regression guard)."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_leader_shaping_bonus

    # Try a variety of action types and step counts.
    for action in [
        _placetile_idx(0, 5, 5),
        _placeleader_idx(0, 5, 5),
        _placeleader_idx(2, 3, 3),
        1500,  # some arbitrary other action
    ]:
        for step in [0, 1, 100, 100000, 1000000]:
            bonus = compute_leader_shaping_bonus(
                action_index=action,
                prev_obs=_empty_obs(),
                next_obs=_board_with_leader_at(5, 5, color_idx=0),
                global_step=step,
                leader_placements_so_far=0,
            )
            assert bonus == 0.0, (
                f"Bonus must be zero when both env vars are 0; "
                f"got {bonus} at action={action}, step={step}"
            )
