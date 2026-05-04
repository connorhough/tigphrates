"""End-to-end sanity for get_action_and_value with the new architecture.

Confirms that the hierarchical sampling interface still works and that
gradients flow through the new spatial heads.
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


def test_sampled_action_lives_in_mask():
    from train import PolicyValueNetwork
    from tigphrates_env import ACTION_SPACE_SIZE

    torch.manual_seed(0)
    np.random.seed(0)
    model = PolicyValueNetwork()
    model.eval()

    obs = _make_obs_batch(1)
    mask = np.zeros((1, ACTION_SPACE_SIZE), dtype=bool)
    legal = [10, 800, 1600, 1652]
    for i in legal:
        mask[0, i] = True

    action, log_prob, entropy, value = model.get_action_and_value(obs, mask)
    a = int(action.item())
    assert a in legal, f"sampled action {a} not in legal set {legal}"
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()


def test_log_prob_gradients_flow_spatial():
    """Forcing an action and calling .backward() on log_prob.sum() should
    populate gradients on the spatial head AND the encoder.

    Use a non-uniform board input so the conv head's logits aren't all equal
    (otherwise the softmax is uniform and weight gradients cancel)."""
    from train import PolicyValueNetwork
    from tigphrates_env import ACTION_SPACE_SIZE

    torch.manual_seed(0)
    model = PolicyValueNetwork()
    model.train()

    obs = _make_obs_batch(2)
    # Inject distinct non-zero patterns so the encoder produces non-uniform
    # spatial features and the conv-head logits actually depend on weights.
    obs["board"][0, 0, 3, 4] = 2.0
    obs["board"][0, 1, 5, 8] = 1.5
    obs["board"][1, 2, 7, 11] = 2.5
    obs["board"][1, 0, 1, 2] = 1.0

    mask = np.zeros((2, ACTION_SPACE_SIZE), dtype=bool)
    mask[0, 0:704] = True
    mask[1, 0:704] = True

    forced_action = torch.tensor([0, 100], dtype=torch.long)
    _, log_prob, _, _ = model.get_action_and_value(obs, mask, action=forced_action)
    log_prob.sum().backward()

    pt_head_grad = model.place_tile_head.weight.grad
    assert pt_head_grad is not None, "place_tile_head got no gradient"
    assert torch.any(pt_head_grad != 0), "place_tile_head gradient is all-zero"

    found_encoder_grad = False
    for p in model.board_encoder.parameters():
        if p.grad is not None and torch.any(p.grad != 0):
            found_encoder_grad = True
            break
    assert found_encoder_grad, "no gradient reached the board encoder"


def test_log_prob_gradients_flow_nonspatial():
    """Forcing a non-spatial action (commitSupport) should populate gradient
    on the nonspatial_head."""
    from train import PolicyValueNetwork, TYPE_BASES
    from tigphrates_env import ACTION_SPACE_SIZE

    torch.manual_seed(0)
    model = PolicyValueNetwork()
    model.train()

    obs = _make_obs_batch(2)
    obs["board"][0, 0, 3, 4] = 2.0
    obs["board"][1, 1, 5, 8] = 1.5
    mask = np.zeros((2, ACTION_SPACE_SIZE), dtype=bool)
    base = TYPE_BASES[6]  # commitSupport
    size = 64
    mask[0, base : base + size] = True
    mask[1, base : base + size] = True

    forced_action = torch.tensor([base, base + 5], dtype=torch.long)
    _, log_prob, _, _ = model.get_action_and_value(obs, mask, action=forced_action)
    log_prob.sum().backward()

    ns_head_grad = model.nonspatial_head.weight.grad
    assert ns_head_grad is not None
    assert torch.any(ns_head_grad != 0), "nonspatial_head gradient is all-zero"
