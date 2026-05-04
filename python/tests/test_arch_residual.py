"""Verify the residual conv tower: RES_BLOCKS controls depth, and the default
is 6 blocks. Reducing it to 2 should shrink the parameter count."""
from __future__ import annotations

import importlib
import os
import sys


def _reload_train():
    """Force-reimport train so a freshly-set RES_BLOCKS env var takes effect."""
    if "train" in sys.modules:
        del sys.modules["train"]
    return importlib.import_module("train")


def test_default_res_blocks_is_six(monkeypatch):
    monkeypatch.delenv("RES_BLOCKS", raising=False)
    train = _reload_train()
    model = train.PolicyValueNetwork()
    # The encoder must expose its block count for the test (and for diagnostics).
    assert hasattr(model.board_encoder, "res_blocks")
    blocks = model.board_encoder.res_blocks
    assert len(blocks) == 6, f"expected 6 default residual blocks, got {len(blocks)}"


def test_res_blocks_env_var_changes_param_count(monkeypatch):
    """Setting RES_BLOCKS=2 produces fewer total parameters than the default
    (6 blocks)."""
    monkeypatch.setenv("RES_BLOCKS", "2")
    train_small = _reload_train()
    small = train_small.PolicyValueNetwork()
    n_small = sum(p.numel() for p in small.parameters())

    monkeypatch.setenv("RES_BLOCKS", "6")
    train_big = _reload_train()
    big = train_big.PolicyValueNetwork()
    n_big = sum(p.numel() for p in big.parameters())

    assert n_small < n_big, (
        f"RES_BLOCKS=2 ({n_small}) should have fewer params than RES_BLOCKS=6 ({n_big})"
    )

    # And the difference should be material (each residual block has 2 conv
    # layers + 2 BN; expect at least a few thousand params per block).
    assert n_big - n_small >= 4 * 1000, (
        f"residual block param delta too small ({n_big - n_small}); blocks may not be wired"
    )


def test_default_board_conv_channels_is_64(monkeypatch):
    """The new default channel count is 64 (up from the old 32)."""
    monkeypatch.delenv("BOARD_CONV_CHANNELS", raising=False)
    monkeypatch.delenv("RES_BLOCKS", raising=False)
    train = _reload_train()
    assert train.BOARD_CONV_CHANNELS == 64
