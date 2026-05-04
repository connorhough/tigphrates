"""Verify the spatial conv heads are wired correctly:
1. The placeTile head produces different per-cell logits when a distinguishable
   feature is placed at one cell vs another (proves spatial info flows through).
2. The flatten ordering of the spatial logits into param_logits matches
   `param = ci * (R*C) + r*C + c`, the contract used by encoder.ts and
   _compute_mirror_index in train.py.
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


def test_spatial_distinguishes_cells():
    """A distinguishable input at cell A should give a different placeTile
    logits slice vs the same setup with the marker moved to cell B."""
    from train import PolicyValueNetwork, TYPE_BASES
    from tigphrates_env import BOARD_ROWS, BOARD_COLS

    torch.manual_seed(0)
    model = PolicyValueNetwork()
    model.eval()

    obs_a = _make_obs_batch(1)
    obs_b = _make_obs_batch(1)

    # Drop a strong marker into channel 0 at two distinct cells.
    obs_a["board"][0, 0, 2, 5] = 5.0
    obs_b["board"][0, 0, 7, 12] = 5.0

    with torch.no_grad():
        _, param_a, _ = model.forward(obs_a)
        _, param_b, _ = model.forward(obs_b)

    base = TYPE_BASES[0]
    size = 4 * BOARD_ROWS * BOARD_COLS
    pa = param_a[0, base : base + size]
    pb = param_b[0, base : base + size]

    assert not torch.allclose(pa, pb), "placeTile head ignores spatial input"


class _FakeHead(torch.nn.Module):
    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, x):
        B = x.shape[0]
        return self.out.expand(B, -1, -1, -1)


def test_spatial_flatten_layout_place_tile():
    """Verify the spatial -> flat layout uses param = ci*(R*C) + r*C + c.

    Strategy: monkey-patch the place_tile_head to a known function of (ci,r,c),
    run forward, and check the resulting param_logits slice matches the
    explicit ordering byte-for-byte.
    """
    from train import PolicyValueNetwork, TYPE_BASES
    from tigphrates_env import BOARD_ROWS, BOARD_COLS

    R, C = BOARD_ROWS, BOARD_COLS
    model = PolicyValueNetwork()
    model.eval()

    target = torch.zeros(1, 4, R, C)
    for ci in range(4):
        for r in range(R):
            for c in range(C):
                target[0, ci, r, c] = ci * 1000 + r * 100 + c

    model.place_tile_head = _FakeHead(target)

    obs = _make_obs_batch(1)
    with torch.no_grad():
        _, param_logits, _ = model.forward(obs)

    base = TYPE_BASES[0]
    flat = param_logits[0, base : base + 4 * R * C].cpu().numpy()

    expected = np.zeros(4 * R * C, dtype=np.float32)
    for ci in range(4):
        for r in range(R):
            for c in range(C):
                p = ci * (R * C) + r * C + c
                expected[p] = ci * 1000 + r * 100 + c

    assert np.allclose(flat, expected), (
        f"placeTile flatten layout mismatch; first 5 diffs: "
        f"{[(i, flat[i], expected[i]) for i in range(len(expected)) if flat[i] != expected[i]][:5]}"
    )


def test_place_leader_flatten_layout():
    """Same layout check for place_leader_head."""
    from train import PolicyValueNetwork, TYPE_BASES
    from tigphrates_env import BOARD_ROWS, BOARD_COLS

    R, C = BOARD_ROWS, BOARD_COLS
    model = PolicyValueNetwork()
    model.eval()

    target = torch.zeros(1, 4, R, C)
    for ci in range(4):
        for r in range(R):
            for c in range(C):
                target[0, ci, r, c] = ci * 1000 + r * 100 + c

    model.place_leader_head = _FakeHead(target)

    obs = _make_obs_batch(1)
    with torch.no_grad():
        _, param_logits, _ = model.forward(obs)

    base = TYPE_BASES[1]
    flat = param_logits[0, base : base + 4 * R * C].cpu().numpy()

    expected = np.zeros(4 * R * C, dtype=np.float32)
    for ci in range(4):
        for r in range(R):
            for c in range(C):
                p = ci * (R * C) + r * C + c
                expected[p] = ci * 1000 + r * 100 + c

    assert np.allclose(flat, expected)


def test_place_catastrophe_flatten_layout():
    """placeCatastrophe is single-channel: param = r*C + c."""
    from train import PolicyValueNetwork, TYPE_BASES
    from tigphrates_env import BOARD_ROWS, BOARD_COLS

    R, C = BOARD_ROWS, BOARD_COLS
    model = PolicyValueNetwork()
    model.eval()

    target = torch.zeros(1, 1, R, C)
    for r in range(R):
        for c in range(C):
            target[0, 0, r, c] = r * 100 + c

    model.place_catastrophe_head = _FakeHead(target)

    obs = _make_obs_batch(1)
    with torch.no_grad():
        _, param_logits, _ = model.forward(obs)

    base = TYPE_BASES[3]
    flat = param_logits[0, base : base + R * C].cpu().numpy()

    expected = np.zeros(R * C, dtype=np.float32)
    for r in range(R):
        for c in range(C):
            expected[r * C + c] = r * 100 + c

    assert np.allclose(flat, expected)
