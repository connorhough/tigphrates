"""
Export a trained PolicyValueNetwork to ONNX for inference outside Python
(e.g. onnxruntime-web in the browser, onnxruntime in C++/Go).

The training network takes a dict of named tensors. Browser-friendly ONNX
runtimes prefer a flat input list, so this script wraps the network in a
module whose forward takes positional tensors in a fixed order, then exports.

Output input names (in order):
    board       (BOARD_CHANNELS, 11, 16) float32
    hand        (4,)                       float32
    hand_seq    (6,)                       int64
    scores      (4,)                       float32
    meta        (8,)                       float32
    conflict    (7,)                       float32
    leaders     (8,)                       float32
    opp_scores  (4,)                       float32
    opp_leaders (8,)                       float32
    mask        (ACTION_SPACE_SIZE,)       bool — if any False, those logits get -inf

Output:
    logits      (ACTION_SPACE_SIZE,)       float32  (already masked)
    value       scalar                     float32

Usage:
    python python/export_onnx.py [--model models/policy_best.pt] [--out models/policy.onnx]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import PolicyValueNetwork
from tigphrates_env import ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS


class FlatPolicy(nn.Module):
    """Wraps PolicyValueNetwork with a flat positional forward signature so
    ONNX can export it cleanly. All inputs are batched (B=1 dimension preserved).
    The mask is applied inside so callers don't have to replicate the masking
    logic on the consuming side."""

    def __init__(self, base: PolicyValueNetwork):
        super().__init__()
        self.base = base

    def forward(
        self,
        board: torch.Tensor,
        hand: torch.Tensor,
        hand_seq: torch.Tensor,
        scores: torch.Tensor,
        meta: torch.Tensor,
        conflict: torch.Tensor,
        leaders: torch.Tensor,
        opp_scores: torch.Tensor,
        opp_leaders: torch.Tensor,
        mask: torch.Tensor,
    ):
        obs = {
            "board": board, "hand": hand, "hand_seq": hand_seq,
            "scores": scores, "meta": meta, "conflict": conflict,
            "leaders": leaders, "opp_scores": opp_scores, "opp_leaders": opp_leaders,
        }
        logits, value = self.base.forward(obs)
        # Apply mask in-graph so the consuming runtime gets pre-masked logits.
        # Use a finite penalty rather than -inf so the export is portable
        # across runtimes that may not handle -inf in softmax cleanly.
        logits = logits.masked_fill(~mask.bool(), -1e9)
        return logits, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/policy_best.pt")
    parser.add_argument("--out", default="models/policy.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if not pathlib.Path(args.model).exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(2)

    base = PolicyValueNetwork()
    base.load_state_dict(torch.load(args.model, map_location="cpu"))
    base.train(False)
    flat = FlatPolicy(base)
    flat.train(False)

    # Sample inputs (B=1) for tracing.
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

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {args.model} -> {out_path}")
    # Prefer the legacy TorchScript-based exporter — doesn't require the
    # newer `onnxscript` dep that PyTorch 2.4+ pulls in for the dynamo path.
    export_kwargs = dict(
        opset_version=args.opset,
        input_names=[
            "board", "hand", "hand_seq", "scores", "meta",
            "conflict", "leaders", "opp_scores", "opp_leaders", "mask",
        ],
        output_names=["logits", "value"],
        dynamic_axes={
            "board": {0: "batch"}, "hand": {0: "batch"}, "hand_seq": {0: "batch"},
            "scores": {0: "batch"}, "meta": {0: "batch"}, "conflict": {0: "batch"},
            "leaders": {0: "batch"}, "opp_scores": {0: "batch"},
            "opp_leaders": {0: "batch"}, "mask": {0: "batch"},
            "logits": {0: "batch"}, "value": {0: "batch"},
        },
    )
    try:
        torch.onnx.export(flat, sample, str(out_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch versions don't accept the dynamo kwarg.
        torch.onnx.export(flat, sample, str(out_path), **export_kwargs)
    size = out_path.stat().st_size
    print(f"Wrote {out_path} ({size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
