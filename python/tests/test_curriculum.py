"""Tests for the heuristic-opponent curriculum.

The curriculum linearly decays the share of heuristic opponents from
CURRICULUM_HEURISTIC_START at step 0 to CURRICULUM_HEURISTIC_END at the end
of training. With CURRICULUM_ENABLED=0 the static TRAIN_VS_HEURISTIC_PROB
is preserved.
"""
import importlib
import sys


def _reload_train_with_env(monkeypatch, **env_overrides):
    """Reload python/train.py with the given env vars so module-level
    constants pick them up. Returns the freshly imported module."""
    for k, v in env_overrides.items():
        monkeypatch.setenv(k, str(v))
    if "train" in sys.modules:
        del sys.modules["train"]
    return importlib.import_module("train")


def test_curriculum_enabled_by_default(monkeypatch):
    """CURRICULUM_ENABLED defaults to '1' so a fresh-init training run uses
    the curriculum schedule out of the box."""
    monkeypatch.delenv("CURRICULUM_ENABLED", raising=False)
    monkeypatch.delenv("CURRICULUM_HEURISTIC_START", raising=False)
    monkeypatch.delenv("CURRICULUM_HEURISTIC_END", raising=False)
    train = _reload_train_with_env(monkeypatch)
    assert train.CURRICULUM_ENABLED is True
    # Sane new defaults — less aggressive than 1.0 → 0.1.
    assert train.CURRICULUM_HEURISTIC_START == 0.9
    assert train.CURRICULUM_HEURISTIC_END == 0.3


def test_curriculum_starts_at_start_value(monkeypatch):
    """At step 0, heuristic_prob_now returns the start value."""
    train = _reload_train_with_env(
        monkeypatch,
        CURRICULUM_ENABLED="1",
        CURRICULUM_HEURISTIC_START="0.9",
        CURRICULUM_HEURISTIC_END="0.3",
    )
    assert train.heuristic_prob_now(elapsed=0.0, total=1000.0) == 0.9


def test_curriculum_ends_at_end_value(monkeypatch):
    """At the end of the schedule, heuristic_prob_now returns the end value."""
    train = _reload_train_with_env(
        monkeypatch,
        CURRICULUM_ENABLED="1",
        CURRICULUM_HEURISTIC_START="0.9",
        CURRICULUM_HEURISTIC_END="0.3",
    )
    # At elapsed == total, t == 1, so result is END.
    assert train.heuristic_prob_now(elapsed=1000.0, total=1000.0) == 0.3
    # Past total it stays clamped to end value.
    assert train.heuristic_prob_now(elapsed=2000.0, total=1000.0) == 0.3


def test_curriculum_disabled_uses_static_prob(monkeypatch):
    """With CURRICULUM_ENABLED=0, heuristic_prob_now returns the static
    TRAIN_VS_HEURISTIC_PROB regardless of elapsed/total."""
    train = _reload_train_with_env(
        monkeypatch,
        CURRICULUM_ENABLED="0",
        TRAIN_VS_HEURISTIC_PROB="0.42",
    )
    assert train.CURRICULUM_ENABLED is False
    assert train.heuristic_prob_now(elapsed=0.0, total=1000.0) == 0.42
    assert train.heuristic_prob_now(elapsed=500.0, total=1000.0) == 0.42
    assert train.heuristic_prob_now(elapsed=1000.0, total=1000.0) == 0.42


def test_curriculum_midpoint_is_linear(monkeypatch):
    """Halfway through the schedule the prob is the linear midpoint."""
    train = _reload_train_with_env(
        monkeypatch,
        CURRICULUM_ENABLED="1",
        CURRICULUM_HEURISTIC_START="0.9",
        CURRICULUM_HEURISTIC_END="0.3",
    )
    mid = train.heuristic_prob_now(elapsed=500.0, total=1000.0)
    # Linear interpolation: 0.9 * 0.5 + 0.3 * 0.5 = 0.6
    assert abs(mid - 0.6) < 1e-9
