"""Unit tests for the BC training step in imitation_pretrain.py.

These tests intentionally avoid spinning up the Node bridge — they
exercise the loss / step on synthetic batches so they run in a few
seconds and don't depend on npx.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imitation_pretrain import bc_train_step  # noqa: E402
from train import (  # noqa: E402
    ACTION_SPACE_SIZE,
    BOARD_CHANNELS,
    BOARD_COLS,
    BOARD_ROWS,
    PolicyValueNetwork,
    TYPE_BASES,
    TYPE_PARAM_SIZES,
)


def _make_obs_dict(batch_size: int) -> dict:
    """A minimal but shape-correct observation dict on CPU."""
    return {
        "board": np.zeros((batch_size, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32),
        "hand": np.zeros((batch_size, 4), dtype=np.float32),
        "hand_seq": -np.ones((batch_size, 6), dtype=np.int64),
        "scores": np.zeros((batch_size, 4), dtype=np.float32),
        "meta": np.zeros((batch_size, 8), dtype=np.float32),
        "conflict": np.zeros((batch_size, 7), dtype=np.float32),
        "leaders": np.zeros((batch_size, 8), dtype=np.float32),
        "opp_scores": np.zeros((batch_size, 4), dtype=np.float32),
        "opp_leaders": np.zeros((batch_size, 8), dtype=np.float32),
    }


def _make_batch(actions: list[int], allow_full_mask: bool = True) -> dict:
    """Build a batch dict suitable for bc_train_step."""
    B = len(actions)
    obs = _make_obs_dict(B)
    if allow_full_mask:
        mask = np.ones((B, ACTION_SPACE_SIZE), dtype=np.int8)
    else:
        mask = np.zeros((B, ACTION_SPACE_SIZE), dtype=np.int8)
        for i, a in enumerate(actions):
            mask[i, a] = 1
    return {
        "obs": obs,
        "action_mask": mask,
        "target_action": np.array(actions, dtype=np.int64),
    }


def test_bc_step_returns_expected_keys():
    """bc_train_step returns a dict containing loss + top1_acc."""
    torch.manual_seed(0)
    model = PolicyValueNetwork()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _make_batch([0, 1, 2, 3])
    metrics = bc_train_step(model, opt, batch)
    assert "loss" in metrics
    assert "top1_acc" in metrics
    assert isinstance(metrics["loss"], float)
    assert isinstance(metrics["top1_acc"], float)
    assert 0.0 <= metrics["top1_acc"] <= 1.0


def test_bc_step_decreases_loss_over_iterations():
    """Repeated BC steps on a fixed tiny batch drive loss down — the
    canonical 'gradient descent works' sanity check.

    Uses a restricted mask (target plus a small set of distractors) so
    the loss has clear gradient signal — full-mask 4-action-batch makes
    the type head hard to disambiguate when all extras are uniform.
    """
    torch.manual_seed(0)
    model = PolicyValueNetwork()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Use 4 distinct target actions across different action types.
    actions = [
        0,                           # type 0 (placeTile), param 0
        TYPE_BASES[2],               # type 2 (withdrawLeader), param 0
        TYPE_BASES[5],               # type 5 (pass), param 0
        TYPE_BASES[8],               # type 8 (buildMonument), param 0
    ]
    # Restricted mask: per-row, allow target + 3 distractors of the same
    # type. Drives loss steeply since param-head only competes within the
    # local type slot.
    B = len(actions)
    obs = _make_obs_dict(B)
    mask = np.zeros((B, ACTION_SPACE_SIZE), dtype=np.int8)
    rng = np.random.default_rng(0)
    for i, a in enumerate(actions):
        mask[i, a] = 1
        for j in rng.choice(ACTION_SPACE_SIZE, size=3, replace=False):
            mask[i, int(j)] = 1
    batch = {
        "obs": obs,
        "action_mask": mask,
        "target_action": np.array(actions, dtype=np.int64),
    }
    initial_loss = bc_train_step(model, opt, batch)["loss"]
    final_loss = initial_loss
    for _ in range(100):
        final_loss = bc_train_step(model, opt, batch)["loss"]
    # Loss should drop substantially on this tiny batch — 50% reduction
    # is a generous threshold.
    assert final_loss < 0.5 * initial_loss, (
        f"loss did not decrease: {initial_loss=}, {final_loss=}"
    )


def test_bc_step_top1_acc_correct():
    """When the chosen action is the only legal one in the mask,
    hierarchical-argmax must yield top1_acc == 1.0 after enough steps."""
    torch.manual_seed(0)
    model = PolicyValueNetwork()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    actions = [0, TYPE_BASES[5], TYPE_BASES[8]]  # diverse types
    batch = _make_batch(actions, allow_full_mask=False)  # mask only the target

    # With mask = {target_action}, the hierarchical argmax has no ambiguity
    # — even one step should already give top1_acc=1.0.
    metrics = bc_train_step(model, opt, batch)
    assert metrics["top1_acc"] == 1.0


def test_bc_step_respects_action_mask():
    """A masked batch must not produce gradient flow at masked positions —
    we verify via behavior: training with only target unmasked converges
    and top1 stays 1.0."""
    torch.manual_seed(0)
    model = PolicyValueNetwork()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    actions = [3, TYPE_BASES[1] + 5]  # placeTile param=3, placeLeader param=5
    batch = _make_batch(actions, allow_full_mask=False)

    # First-step top1 should already be 1.0 (mask leaves only the target
    # legal). Run a few extra steps to confirm loss is bounded.
    for _ in range(20):
        metrics = bc_train_step(model, opt, batch)

    assert metrics["top1_acc"] == 1.0
    # With only one legal action per sample, the model converges quickly.
    assert metrics["loss"] < 2.0


def test_bc_step_does_not_explode_with_partial_mask():
    """Partial mask: target action is in the mask but so are several other
    actions. Step should still produce finite loss and run cleanly."""
    torch.manual_seed(0)
    model = PolicyValueNetwork()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B = 4
    obs = _make_obs_dict(B)
    actions = [0, TYPE_BASES[2], TYPE_BASES[5], TYPE_BASES[8]]
    mask = np.zeros((B, ACTION_SPACE_SIZE), dtype=np.int8)
    # Allow each target plus one or two random other actions
    rng = np.random.default_rng(0)
    for i, a in enumerate(actions):
        mask[i, a] = 1
        for j in rng.choice(ACTION_SPACE_SIZE, size=3, replace=False):
            mask[i, int(j)] = 1
    batch = {
        "obs": obs,
        "action_mask": mask,
        "target_action": np.array(actions, dtype=np.int64),
    }
    metrics = bc_train_step(model, opt, batch)
    assert np.isfinite(metrics["loss"])
    assert 0.0 <= metrics["top1_acc"] <= 1.0
