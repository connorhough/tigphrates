"""Architecture shape tests — verify the new conv-tower / spatial-head
PolicyValueNetwork preserves the (type_logits, param_logits, value) interface
that PPO and the bridge depend on.

Workstream-C scope (architecture). Do not add tests here that touch reward
shaping, BC, evaluation, or curriculum — those are owned by other agents.
"""
from __future__ import annotations

import numpy as np
import torch


def _make_obs_batch(B: int):
    from tigphrates_env import BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS

    return {
        "board": torch.zeros(B, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS, dtype=torch.float32),
        "hand": torch.zeros(B, 4, dtype=torch.float32),
        "hand_seq": torch.full((B, 6), -1, dtype=torch.int64),
        "scores": torch.zeros(B, 4, dtype=torch.float32),
        "meta": torch.zeros(B, 8, dtype=torch.float32),
        "conflict": torch.zeros(B, 7, dtype=torch.float32),
        "leaders": torch.full((B, 8), -1, dtype=torch.float32),
        "opp_scores": torch.zeros(B, 4, dtype=torch.float32),
        "opp_leaders": torch.full((B, 8), -1, dtype=torch.float32),
    }


def test_forward_shapes_b1():
    from train import PolicyValueNetwork, NUM_ACTION_TYPES
    from tigphrates_env import ACTION_SPACE_SIZE

    model = PolicyValueNetwork()
    model.eval()
    obs = _make_obs_batch(1)
    type_logits, param_logits, value = model.forward(obs)

    assert type_logits.shape == (1, NUM_ACTION_TYPES), type_logits.shape
    assert param_logits.shape == (1, ACTION_SPACE_SIZE), param_logits.shape
    assert value.shape == (1,), value.shape


def test_forward_shapes_b4():
    from train import PolicyValueNetwork, NUM_ACTION_TYPES
    from tigphrates_env import ACTION_SPACE_SIZE

    model = PolicyValueNetwork()
    model.eval()
    obs = _make_obs_batch(4)
    type_logits, param_logits, value = model.forward(obs)

    assert type_logits.shape == (4, NUM_ACTION_TYPES)
    assert param_logits.shape == (4, ACTION_SPACE_SIZE)
    assert value.shape == (4,)


def test_all_type_slices_finite_at_init():
    """Every slot in param_logits across every TYPE_BASES slice must be a
    finite float at fresh init — catches NaN-init and accidentally-uninit
    slice bugs in the scatter."""
    from train import PolicyValueNetwork, TYPE_BASES, TYPE_PARAM_SIZES

    model = PolicyValueNetwork()
    model.eval()
    obs = _make_obs_batch(2)
    _, param_logits, value = model.forward(obs)

    pl = param_logits.detach().cpu().numpy()
    assert np.isfinite(pl).all(), "param_logits has non-finite entries at init"
    assert np.isfinite(value.detach().cpu().numpy()).all()

    for t, (base, size) in enumerate(zip(TYPE_BASES, TYPE_PARAM_SIZES)):
        chunk = pl[:, base : base + size]
        assert np.isfinite(chunk).all(), (
            f"type {t} slice [{base}:{base+size}] has non-finite values"
        )
