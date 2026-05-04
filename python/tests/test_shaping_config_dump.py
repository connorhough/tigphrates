"""Tests for python/shaping_config_dump.py.

The helper snapshots the live shaping coefficients (env vars + module
defaults) into a JSON file so downstream agents have one source of truth.
"""
import json
import os
import pathlib

import pytest

from shaping_config_dump import dump_shaping_config, load_shaping_config


def test_dump_writes_expected_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.07")
    monkeypatch.setenv("KING_LEADER_BONUS", "0.12")
    monkeypatch.setenv("SCORE_DELTA_COEF", "1.5")
    monkeypatch.setenv("MARGIN_DELTA_COEF", "1.0")
    monkeypatch.setenv("POTENTIAL_GAMMA_SHAPING_COEF", "1.0")
    monkeypatch.setenv("BC_COEF", "0.1")
    monkeypatch.setenv("SHAPING_DECAY_STEPS", "200000")
    out = tmp_path / "shaping_config.json"

    dump_shaping_config(out)

    assert out.exists()
    data = json.loads(out.read_text())
    # Event shaping (per-action bonuses)
    assert data["event_shaping"]["leader_place_bonus"] == pytest.approx(0.07)
    assert data["event_shaping"]["king_leader_bonus"] == pytest.approx(0.12)
    assert data["event_shaping"]["kingdom_form_bonus"] == pytest.approx(0.10)  # default
    assert data["event_shaping"]["treasure_collect_bonus"] == pytest.approx(0.15)  # default
    assert data["event_shaping"]["monument_build_bonus"] == pytest.approx(0.10)  # default
    assert data["event_shaping"]["decay_steps"] == 200000
    # Score shaping (per-step potential / delta family)
    assert data["score_shaping"]["score_delta_coef"] == pytest.approx(1.5)
    assert data["score_shaping"]["margin_delta_coef"] == pytest.approx(1.0)
    assert data["score_shaping"]["potential_gamma_shaping_coef"] == pytest.approx(1.0)
    # BC auxiliary
    assert data["bc"]["bc_coef"] == pytest.approx(0.1)


def test_load_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LEADER_PLACE_BONUS", "0.05")
    out = tmp_path / "shaping_config.json"
    dump_shaping_config(out)
    loaded = load_shaping_config(out)
    assert loaded["event_shaping"]["leader_place_bonus"] == pytest.approx(0.05)
