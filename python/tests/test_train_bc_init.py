"""Tests for train.py's BC-pretrain wiring:
  - BC_INIT_CHECKPOINT loads a saved policy_init.pt at startup
  - BC_AUX_DISABLED forces the PPO BC auxiliary coefficient to 0
  - default behavior unchanged when the env vars are unset

These tests poke at the small helper functions train.py exposes —
they intentionally avoid kicking off the full main() to keep tests
fast and bridge-free.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import PolicyValueNetwork, build_policy  # noqa: E402


def _first_named_param(model):
    """Pick a stable parameter to compare on — avoids coupling tests to
    architecture-specific attribute names like `param_head`."""
    for name, p in model.named_parameters():
        return name, p
    raise RuntimeError("model has no parameters")


def test_build_policy_random_when_path_none():
    """build_policy(None) returns a random PolicyValueNetwork."""
    torch.manual_seed(0)
    m1 = build_policy(None)
    torch.manual_seed(0)
    m2 = PolicyValueNetwork()
    # Same seed → identical params for the first named parameter.
    name, p1 = _first_named_param(m1)
    p2 = dict(m2.named_parameters())[name]
    assert torch.allclose(p1, p2, atol=1e-6)


def test_build_policy_loads_checkpoint(tmp_path):
    """build_policy(path) loads a saved state_dict into the policy."""
    # Create a model, perturb its weights, save to tmp.
    torch.manual_seed(7)
    src = PolicyValueNetwork()
    with torch.no_grad():
        # Touch a stable surface parameter — value_head is part of the API
        # that hasn't changed across architecture refactors.
        src.value_head.weight.fill_(0.123)
        src.type_head.weight.fill_(0.456)
    ckpt = tmp_path / "policy_init.pt"
    torch.save(src.state_dict(), ckpt)

    loaded = build_policy(str(ckpt))
    assert torch.allclose(loaded.value_head.weight,
                          torch.full_like(loaded.value_head.weight, 0.123))
    assert torch.allclose(loaded.type_head.weight,
                          torch.full_like(loaded.type_head.weight, 0.456))


def test_build_policy_missing_checkpoint_warns_and_random_inits(tmp_path, capsys):
    """Missing path → fall back to random init and print a warning."""
    missing = tmp_path / "does_not_exist.pt"
    m = build_policy(str(missing))
    assert isinstance(m, PolicyValueNetwork)
    captured = capsys.readouterr()
    assert "not found" in (captured.out + captured.err).lower()


def test_bc_aux_disabled_zeroes_coef(monkeypatch):
    """When BC_AUX_DISABLED=1, the effective BC_COEF used in the PPO loss is 0."""
    monkeypatch.setenv("BC_AUX_DISABLED", "1")
    monkeypatch.setenv("BC_COEF", "0.5")

    # Reload the train module so it re-reads env vars at import.
    import train as train_mod
    importlib.reload(train_mod)

    assert train_mod.effective_bc_coef() == 0.0


def test_bc_aux_default_uses_bc_coef(monkeypatch):
    """When BC_AUX_DISABLED is unset (or 0), effective_bc_coef returns BC_COEF."""
    monkeypatch.delenv("BC_AUX_DISABLED", raising=False)
    monkeypatch.setenv("BC_COEF", "0.25")

    import train as train_mod
    importlib.reload(train_mod)

    assert train_mod.effective_bc_coef() == 0.25


def test_bc_init_checkpoint_env(monkeypatch, tmp_path):
    """BC_INIT_CHECKPOINT env var resolves to the path build_policy reads.

    We don't actually invoke main(); we verify that a helper exposed by
    train.py reads the env var correctly.
    """
    torch.manual_seed(11)
    src = PolicyValueNetwork()
    with torch.no_grad():
        src.value_head.bias.fill_(0.789)
    ckpt = tmp_path / "init.pt"
    torch.save(src.state_dict(), ckpt)

    monkeypatch.setenv("BC_INIT_CHECKPOINT", str(ckpt))

    import train as train_mod
    importlib.reload(train_mod)

    path = train_mod.bc_init_checkpoint_path()
    assert path is not None
    assert pathlib.Path(path) == ckpt


def test_bc_init_checkpoint_unset_returns_none(monkeypatch):
    """No BC_INIT_CHECKPOINT → bc_init_checkpoint_path() returns None."""
    monkeypatch.delenv("BC_INIT_CHECKPOINT", raising=False)
    import train as train_mod
    importlib.reload(train_mod)
    assert train_mod.bc_init_checkpoint_path() is None
