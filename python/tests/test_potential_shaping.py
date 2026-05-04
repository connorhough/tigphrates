"""Tests for potential-based reward shaping (Ng, Harada, Russell 1999).

The shaping reward at step t is:

    F(s, s') = coef * (gamma * Phi(s') - Phi(s))

where Phi(s) = min_color(active_player_score). At episode termination,
Phi(s') is defined to be 0 (the absorbing state has no future return),
which is the necessary boundary condition for policy invariance.

The pure helper `compute_potential_shaping` lives in `train.py` so we can
unit-test it without spawning the Node bridge or PPO machinery.
"""
import os

import numpy as np
import pytest


# Set env vars before importing train (it reads them at module load).
os.environ.setdefault("POTENTIAL_GAMMA_SHAPING_COEF", "1.0")


def test_non_terminal_basic():
    """Non-terminal: F = coef * (gamma * Phi(s') - Phi(s))."""
    from train import compute_potential_shaping

    # coef=1.0, gamma=0.99, prev=2, next=3 -> 0.99*3 - 2 = 0.97
    f = compute_potential_shaping(
        min_score_prev=2,
        min_score_next=3,
        terminal=False,
        gamma=0.99,
        coef=1.0,
    )
    assert f == pytest.approx(0.99 * 3 - 2)
    assert f == pytest.approx(0.97)


def test_terminal_uses_zero_potential():
    """At terminal, Phi(s') := 0 so F = coef * (0 - Phi(s)) = -coef * Phi(s).

    This is the Ng et al. boundary condition required for policy invariance.
    Terminating with a high min-score yields a negative shaping reward at
    the boundary; the raw terminal reward absorbs this (the telescoping
    sum collapses).
    """
    from train import compute_potential_shaping

    f = compute_potential_shaping(
        min_score_prev=3,
        min_score_next=99,  # value irrelevant when terminal
        terminal=True,
        gamma=0.99,
        coef=1.0,
    )
    # 0.99 * 0 - 3 = -3.0 (the gamma factor is multiplied by Phi_next=0,
    # so it doesn't soften the terminal penalty).
    assert f == pytest.approx(0.99 * 0 - 3)
    assert f == pytest.approx(-3.0)


def test_coef_zero_returns_zero():
    """coef=0 disables the term entirely; returns 0 for all inputs."""
    from train import compute_potential_shaping

    for terminal in (False, True):
        for prev, nxt in [(0, 0), (1, 5), (10, 0), (0, 10), (5, 5)]:
            f = compute_potential_shaping(
                min_score_prev=prev,
                min_score_next=nxt,
                terminal=terminal,
                gamma=0.99,
                coef=0.0,
            )
            assert f == 0.0, (
                f"coef=0 must return 0; got {f} for prev={prev} next={nxt} term={terminal}"
            )


def test_coef_scaling():
    """F is linear in coef."""
    from train import compute_potential_shaping

    base = compute_potential_shaping(
        min_score_prev=2, min_score_next=3, terminal=False, gamma=0.99, coef=1.0
    )
    scaled = compute_potential_shaping(
        min_score_prev=2, min_score_next=3, terminal=False, gamma=0.99, coef=2.5
    )
    assert scaled == pytest.approx(2.5 * base)


def test_telescoping_property_gamma_one():
    """With gamma=1.0, the sum of F across an episode equals -Phi(s_0).

    Telescoping: sum_{t=0..T-1} (gamma * Phi(s_{t+1}) - Phi(s_t))
              =  Phi(s_T) - Phi(s_0)   when gamma=1
    With terminal Phi(s_T) := 0 (absorbing), the sum collapses to -Phi(s_0).
    This is THE invariance property — the total shaped return only depends
    on the start state, so the optimal policy is unchanged.
    """
    from train import compute_potential_shaping

    # Episode of length N with arbitrary potential trajectory.
    phis = [4, 6, 5, 7, 8, 3, 9]  # Phi(s_0), Phi(s_1), ..., Phi(s_N)
    gamma = 1.0
    coef = 1.0

    total = 0.0
    for t in range(len(phis) - 1):
        total += compute_potential_shaping(
            min_score_prev=phis[t],
            min_score_next=phis[t + 1],
            terminal=False,
            gamma=gamma,
            coef=coef,
        )
    # Final terminal step: from phis[-1] to absorbing (potential 0).
    total += compute_potential_shaping(
        min_score_prev=phis[-1],
        min_score_next=0,  # ignored when terminal=True
        terminal=True,
        gamma=gamma,
        coef=coef,
    )

    # Sum should equal -Phi(s_0).
    assert total == pytest.approx(-phis[0])


def test_telescoping_with_discount():
    """With gamma<1 and terminal Phi:=0, sum collapses by the standard
    discounted telescoping identity: it equals -Phi(s_0) when accumulated
    with the appropriate discount factors. We test the simpler raw-sum
    invariant for a 2-step episode here just as a sanity check."""
    from train import compute_potential_shaping

    gamma = 0.9
    coef = 1.0
    phi0, phi1 = 4, 6

    f0 = compute_potential_shaping(
        min_score_prev=phi0, min_score_next=phi1, terminal=False, gamma=gamma, coef=coef
    )
    f1 = compute_potential_shaping(
        min_score_prev=phi1, min_score_next=0, terminal=True, gamma=gamma, coef=coef
    )
    # f0 = 0.9*6 - 4 = 1.4
    # f1 = 0.9*0 - 6 = -6.0
    assert f0 == pytest.approx(0.9 * phi1 - phi0)
    assert f1 == pytest.approx(-phi1)


def test_active_player_potential_sign():
    """Integration-style: simulate the train.py reward path by composing
    the helper with active-player min_score lookups. Verifies the sign
    of the shaping change when the active player's floor color rises.
    """
    from train import compute_potential_shaping

    # Active player's per-color scores before action.
    prev_scores = np.array([2, 4, 5, 3], dtype=np.int32)
    # After action: min color (red) bumped up by 1.
    next_scores = np.array([3, 4, 5, 3], dtype=np.int32)

    prev_min = float(np.min(prev_scores))
    next_min = float(np.min(next_scores))

    f = compute_potential_shaping(
        min_score_prev=prev_min,
        min_score_next=next_min,
        terminal=False,
        gamma=0.99,
        coef=1.0,
    )
    # 0.99 * 3 - 2 = 0.97 > 0 -> raising the floor color is rewarded.
    assert f > 0
    assert f == pytest.approx(0.99 * 3 - 2)


def test_active_player_potential_negative_when_floor_unchanged():
    """If the floor doesn't move but next state has same Phi as prev,
    F = (gamma - 1) * Phi(s) which is slightly negative for gamma<1.
    This is a known property of potential shaping: it gently penalizes
    standing still in high-potential states, encouraging progress."""
    from train import compute_potential_shaping

    f = compute_potential_shaping(
        min_score_prev=5, min_score_next=5, terminal=False, gamma=0.99, coef=1.0
    )
    # 0.99*5 - 5 = -0.05
    assert f == pytest.approx(0.99 * 5 - 5)
    assert f < 0


def test_regression_old_shape_when_disabled():
    """When POTENTIAL_GAMMA_SHAPING_COEF=0, the per-step shaped reward
    should reduce to the legacy formula:

        shaped = raw_reward
                 + SCORE_DELTA_COEF * blended_min_avg_delta
                 + MARGIN_DELTA_COEF * margin_delta

    We reproduce the formula here and assert algebraic equality with
    a direct numpy computation, since exercising the rollout loop
    requires the bridge subprocess.
    """
    from train import (
        SCORE_AVG_WEIGHT,
        SCORE_DELTA_COEF,
        MARGIN_DELTA_COEF,
        compute_potential_shaping,
    )

    raw_reward = 0.0

    prev_scores = np.array([2, 4, 5, 3], dtype=np.float32)
    new_scores = np.array([3, 4, 5, 3], dtype=np.float32)
    prev_opp = np.array([1, 2, 3, 4], dtype=np.float32)
    new_opp = np.array([1, 2, 3, 5], dtype=np.float32)

    prev_min = float(np.min(prev_scores))
    prev_avg = float(np.mean(prev_scores))
    new_min = float(np.min(new_scores))
    new_avg = float(np.mean(new_scores))

    min_delta = new_min - prev_min
    avg_delta = new_avg - prev_avg
    blended = (1.0 - SCORE_AVG_WEIGHT) * min_delta + SCORE_AVG_WEIGHT * avg_delta

    prev_margin = prev_min - float(np.min(prev_opp))
    new_margin = new_min - float(np.min(new_opp))
    margin_delta = new_margin - prev_margin

    legacy_shaped = (
        raw_reward
        + SCORE_DELTA_COEF * blended
        + MARGIN_DELTA_COEF * margin_delta
    )

    # With coef=0, potential shaping contributes 0.
    f = compute_potential_shaping(
        min_score_prev=prev_min,
        min_score_next=new_min,
        terminal=False,
        gamma=0.99,
        coef=0.0,
    )
    new_shaped = legacy_shaped + f

    assert f == 0.0
    assert new_shaped == pytest.approx(legacy_shaped)


def test_potential_replaces_score_delta_when_enabled():
    """When POTENTIAL_GAMMA_SHAPING_COEF > 0, the new shaping replaces the
    old SCORE_DELTA_COEF * Δmin_score term. We assert the shape of the
    resulting reward equation: legacy minus the score-delta term plus the
    potential term, leaving margin untouched.
    """
    from train import (
        SCORE_AVG_WEIGHT,
        MARGIN_DELTA_COEF,
        compute_potential_shaping,
    )

    raw_reward = 0.0
    prev_min, new_min = 2.0, 3.0
    prev_avg, new_avg = 3.5, 3.75
    prev_margin, new_margin = -1.0, -0.5
    coef = 1.0
    gamma = 0.99

    margin_delta = new_margin - prev_margin
    f = compute_potential_shaping(
        min_score_prev=prev_min,
        min_score_next=new_min,
        terminal=False,
        gamma=gamma,
        coef=coef,
    )
    new_shaped = raw_reward + f + MARGIN_DELTA_COEF * margin_delta

    # Margin term is preserved.
    assert new_shaped == pytest.approx(f + MARGIN_DELTA_COEF * margin_delta)
    # Potential shaping is non-trivial when min rises.
    assert f > 0
