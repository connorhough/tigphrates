"""Tests for the extended per-event reward-shaping helper.

`compute_event_shaping_bonus` is the superset of `compute_leader_shaping_bonus`
that additionally awards:
  - Treasure-collect bonus when the active player's treasure count goes up.
  - Monument-build bonus when the action falls in the buildMonument slot range.
  - A king-specific (black leader) bumped place bonus.

All bonuses share the same linear decay over SHAPING_DECAY_STEPS env steps.

Action layout (must match src/bridge/encoder.ts):
  placeTile:   [0, 704)
  placeLeader: [704, 1408)        — 4 colors × 176 cells, in red/blue/green/black order
  withdrawLeader (4) + placeCatastrophe (176) + swapTiles (64) + pass (1)
    + commitSupport (64) + chooseWarOrder (4)         [1408, 1721)
  buildMonument (6):  [1721, 1727)
  declineMonument (1): [1727, 1728)
"""
import os
import numpy as np
import pytest


BOARD_CHANNELS = 15
BOARD_ROWS = 11
BOARD_COLS = 16


# --- Observation builders ---------------------------------------------------

def _empty_obs(treasures: float = 0.0):
    """Minimal observation dict shaped like the env's _get_obs output.
    `meta[0]` is the active-player treasure count."""
    meta = np.zeros(8, dtype=np.float32)
    meta[0] = float(treasures)
    return {
        "board": np.zeros((BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32),
        "scores": np.zeros(4, dtype=np.int32),
        "opp_scores": np.zeros(4, dtype=np.float32),
        "meta": meta,
    }


def _board_with_leader_at(row, col, color_idx=0, treasures=0.0):
    obs = _empty_obs(treasures=treasures)
    obs["board"][4 + color_idx, row, col] = 1.0
    return obs


# --- Action index helpers ---------------------------------------------------

def _placetile_idx(color: int, row: int, col: int) -> int:
    return color * BOARD_ROWS * BOARD_COLS + row * BOARD_COLS + col


def _placeleader_idx(color: int, row: int, col: int) -> int:
    base = 4 * BOARD_ROWS * BOARD_COLS  # 704
    return base + color * BOARD_ROWS * BOARD_COLS + row * BOARD_COLS + col


def _buildmonument_idx(slot: int) -> int:
    """Type 8 = buildMonument. Base is the cumulative sum of all earlier types:
       704 + 704 + 4 + 176 + 64 + 1 + 64 + 4 = 1721, with 6 slots."""
    assert 0 <= slot < 6
    return 1721 + slot


# --- Tests ------------------------------------------------------------------

def test_treasure_delta_awards_bonus(monkeypatch):
    """Going from treasures=2 to treasures=3 awards exactly one TREASURE_COLLECT_BONUS."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.0")
    monkeypatch.setenv("KING_LEADER_BONUS", "0.10")
    monkeypatch.setenv("TREASURE_COLLECT_BONUS", "0.15")
    monkeypatch.setenv("MONUMENT_BUILD_BONUS", "0.10")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_event_shaping_bonus

    prev_obs = _empty_obs(treasures=2)
    next_obs = _empty_obs(treasures=3)
    # Use a non-leader, non-monument action so only the treasure bonus fires.
    action = _placetile_idx(0, 5, 5)

    bonus = compute_event_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
        monument_builds_so_far=0,
    )
    assert bonus == pytest.approx(0.15)


def test_treasure_no_change_returns_zero(monkeypatch):
    """No treasure delta = no bonus."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.0")
    monkeypatch.setenv("TREASURE_COLLECT_BONUS", "0.15")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_event_shaping_bonus

    prev_obs = _empty_obs(treasures=2)
    next_obs = _empty_obs(treasures=2)
    action = _placetile_idx(0, 5, 5)

    bonus = compute_event_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
        monument_builds_so_far=0,
    )
    assert bonus == 0.0


def test_treasure_uncapped_5_deltas(monkeypatch):
    """5 separate treasure deltas all award (no cap). Sum should be 5 * 0.15 = 0.75."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.0")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.0")
    monkeypatch.setenv("TREASURE_COLLECT_BONUS", "0.15")
    monkeypatch.setenv("MONUMENT_BUILD_BONUS", "0.0")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_event_shaping_bonus

    total = 0.0
    for i in range(5):
        prev_obs = _empty_obs(treasures=i)
        next_obs = _empty_obs(treasures=i + 1)
        action = _placetile_idx(0, 5, 5)
        bonus = compute_event_shaping_bonus(
            action_index=action,
            prev_obs=prev_obs,
            next_obs=next_obs,
            global_step=0,
            leader_placements_so_far=0,
            monument_builds_so_far=0,
        )
        total += bonus
    assert total == pytest.approx(0.75)


def test_monument_build_single_award(monkeypatch):
    """A buildMonument action awards MONUMENT_BUILD_BONUS once."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.0")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.0")
    monkeypatch.setenv("TREASURE_COLLECT_BONUS", "0.0")
    monkeypatch.setenv("MONUMENT_BUILD_BONUS", "0.10")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_event_shaping_bonus

    prev_obs = _empty_obs()
    next_obs = _empty_obs()
    action = _buildmonument_idx(slot=0)

    bonus = compute_event_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
        monument_builds_so_far=0,
    )
    assert bonus == pytest.approx(0.10)


def test_monument_build_capped_at_two(monkeypatch):
    """Third monument build (monument_builds_so_far=2) returns no bonus."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.0")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.0")
    monkeypatch.setenv("TREASURE_COLLECT_BONUS", "0.0")
    monkeypatch.setenv("MONUMENT_BUILD_BONUS", "0.10")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_event_shaping_bonus

    prev_obs = _empty_obs()
    next_obs = _empty_obs()
    action = _buildmonument_idx(slot=2)

    # 1st build (so_far=0): awards.
    b1 = compute_event_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
        monument_builds_so_far=0,
    )
    assert b1 == pytest.approx(0.10)

    # 2nd build (so_far=1): awards.
    b2 = compute_event_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
        monument_builds_so_far=1,
    )
    assert b2 == pytest.approx(0.10)

    # 3rd build (so_far=2): zero, hits the cap.
    b3 = compute_event_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
        monument_builds_so_far=2,
    )
    assert b3 == 0.0


def test_king_specific_bonus_bump(monkeypatch):
    """Black-leader placement uses KING_LEADER_BONUS (0.10);
    red-leader placement uses LEADER_PLACE_BONUS (0.05). Both still cap at 4."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.0")
    monkeypatch.setenv("KING_LEADER_BONUS", "0.10")
    monkeypatch.setenv("TREASURE_COLLECT_BONUS", "0.0")
    monkeypatch.setenv("MONUMENT_BUILD_BONUS", "0.0")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_event_shaping_bonus

    prev_obs = _empty_obs()
    # Red leader at color_idx=0
    red_action = _placeleader_idx(color=0, row=5, col=5)
    next_obs_red = _board_with_leader_at(5, 5, color_idx=0)
    red_bonus = compute_event_shaping_bonus(
        action_index=red_action,
        prev_obs=prev_obs,
        next_obs=next_obs_red,
        global_step=0,
        leader_placements_so_far=0,
        monument_builds_so_far=0,
    )
    assert red_bonus == pytest.approx(0.05)

    # Black leader at color_idx=3 (king)
    black_action = _placeleader_idx(color=3, row=5, col=5)
    next_obs_black = _board_with_leader_at(5, 5, color_idx=3)
    black_bonus = compute_event_shaping_bonus(
        action_index=black_action,
        prev_obs=prev_obs,
        next_obs=next_obs_black,
        global_step=0,
        leader_placements_so_far=0,
        monument_builds_so_far=0,
    )
    assert black_bonus == pytest.approx(0.10)

    # Cap still bites for black: at so_far=4 the bonus is gone.
    black_capped = compute_event_shaping_bonus(
        action_index=black_action,
        prev_obs=prev_obs,
        next_obs=next_obs_black,
        global_step=0,
        leader_placements_so_far=4,
        monument_builds_so_far=0,
    )
    assert black_capped == 0.0


def test_decay_zeros_all_three_bonuses(monkeypatch):
    """At global_step == SHAPING_DECAY_STEPS every event bonus is 0."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.10")
    monkeypatch.setenv("KING_LEADER_BONUS", "0.10")
    monkeypatch.setenv("TREASURE_COLLECT_BONUS", "0.15")
    monkeypatch.setenv("MONUMENT_BUILD_BONUS", "0.10")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "1000")

    from tigphrates_env import compute_event_shaping_bonus

    # Leader: black king, kingdom-forming, treasure delta, monument-build.
    prev_obs = _empty_obs(treasures=2)
    next_obs = _board_with_leader_at(5, 5, color_idx=3, treasures=3)

    leader_action = _placeleader_idx(color=3, row=5, col=5)
    monument_action = _buildmonument_idx(slot=0)
    treasure_only_action = _placetile_idx(0, 5, 5)

    for action in (leader_action, monument_action, treasure_only_action):
        bonus = compute_event_shaping_bonus(
            action_index=action,
            prev_obs=prev_obs,
            next_obs=next_obs,
            global_step=1000,
            leader_placements_so_far=0,
            monument_builds_so_far=0,
        )
        assert bonus == 0.0, f"bonus must be zero at full decay; got {bonus} for action={action}"


def test_compose_king_kingdom_treasure(monkeypatch):
    """A single action that simultaneously is a king-leader placement that forms
    a kingdom AND collects a treasure should sum all three contributions."""
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    monkeypatch.setenv("KINGDOM_FORM_BONUS", "0.10")
    monkeypatch.setenv("KING_LEADER_BONUS", "0.10")
    monkeypatch.setenv("TREASURE_COLLECT_BONUS", "0.15")
    monkeypatch.setenv("MONUMENT_BUILD_BONUS", "0.10")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")

    from tigphrates_env import compute_event_shaping_bonus

    # Leader (black) placed at (5, 5) with adjacent tile at (5, 6).
    prev_obs = _empty_obs(treasures=2)
    prev_obs["board"][0, 5, 6] = 1.0  # adjacent red tile
    next_obs = _board_with_leader_at(5, 5, color_idx=3, treasures=3)
    next_obs["board"][0, 5, 6] = 1.0  # tile still there

    action = _placeleader_idx(color=3, row=5, col=5)

    bonus = compute_event_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
        monument_builds_so_far=0,
    )
    # KING_LEADER_BONUS (0.10) + KINGDOM_FORM_BONUS (0.10) + TREASURE (0.15) = 0.35
    assert bonus == pytest.approx(0.35)


def test_backward_compat_alias_matches_old_helper(monkeypatch):
    """The old `compute_leader_shaping_bonus` alias must return the same value
    as it did before the refactor for a leader-only event."""
    # Use the documented production defaults (no king bump enabled, since the
    # alias must behave identically to the original 0.05/0.10 helper).
    monkeypatch.delenv("LEADER_PLACE_BONUS", raising=False)
    monkeypatch.delenv("KINGDOM_FORM_BONUS", raising=False)
    monkeypatch.delenv("SHAPING_DECAY_STEPS", raising=False)
    # Disable the new bonuses so the alias is a strict no-op extension.
    monkeypatch.setenv("KING_LEADER_BONUS", "0.05")           # same as LEADER
    monkeypatch.setenv("TREASURE_COLLECT_BONUS", "0.0")
    monkeypatch.setenv("MONUMENT_BUILD_BONUS", "0.0")

    from tigphrates_env import compute_leader_shaping_bonus

    # Red leader, kingdom-forming.
    prev_obs = _empty_obs()
    prev_obs["board"][0, 5, 6] = 1.0
    next_obs = _board_with_leader_at(5, 5, color_idx=0)
    next_obs["board"][0, 5, 6] = 1.0
    action = _placeleader_idx(color=0, row=5, col=5)

    bonus = compute_leader_shaping_bonus(
        action_index=action,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
    )
    # Original behavior: LEADER_PLACE_BONUS (0.05) + KINGDOM_FORM_BONUS (0.10).
    assert bonus == pytest.approx(0.15)

    # And a non-leader is still zero.
    nonleader_bonus = compute_leader_shaping_bonus(
        action_index=_placetile_idx(0, 5, 5),
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=0,
        leader_placements_so_far=0,
    )
    assert nonleader_bonus == 0.0
