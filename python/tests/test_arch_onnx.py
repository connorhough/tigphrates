"""ONNX export must continue to work with the new architecture. We export a
freshly-initialized network into a temp path; the test only asserts the
export pipeline runs — value-equivalence with PyTorch is out of scope."""
from __future__ import annotations

import torch


def test_fresh_network_exports_to_onnx(tmp_path):
    from export_onnx import FlatPolicy
    from train import PolicyValueNetwork
    from tigphrates_env import ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS

    base = PolicyValueNetwork()
    base.eval()
    flat = FlatPolicy(base)
    flat.eval()

    sample = (
        torch.zeros(1, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS, dtype=torch.float32),
        torch.zeros(1, 4, dtype=torch.float32),
        torch.zeros(1, 6, dtype=torch.int64),
        torch.zeros(1, 4, dtype=torch.float32),
        torch.zeros(1, 8, dtype=torch.float32),
        torch.zeros(1, 7, dtype=torch.float32),
        torch.zeros(1, 8, dtype=torch.float32),
        torch.zeros(1, 4, dtype=torch.float32),
        torch.zeros(1, 8, dtype=torch.float32),
        torch.ones(1, ACTION_SPACE_SIZE, dtype=torch.bool),
    )

    out_path = tmp_path / "model.onnx"
    export_kwargs = dict(
        opset_version=17,
        input_names=[
            "board", "hand", "hand_seq", "scores", "meta",
            "conflict", "leaders", "opp_scores", "opp_leaders", "mask",
        ],
        output_names=["type_logits", "param_logits", "value"],
    )
    try:
        torch.onnx.export(flat, sample, str(out_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(flat, sample, str(out_path), **export_kwargs)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
