"""Unit tests for evaluate_top1_match_vs_heuristic.

Uses monkeypatched bridge calls so the test runs without spinning up
the real Node bridge subprocess (no `npx tsx` invocation).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import (  # noqa: E402
    ACTION_SPACE_SIZE,
    BOARD_CHANNELS,
    BOARD_COLS,
    BOARD_ROWS,
    PolicyValueNetwork,
    TYPE_BASES,
)


class FakeBridge:
    """Stub for tigphrates_env.BridgeProcess that scripts a finite game."""

    def __init__(self, scripted_actions: list[int], num_steps_until_game_over: int = None):
        """
        scripted_actions: action_index that ai_action will return at each
            step. game ends when len(scripted_actions) consumed.
        """
        self.scripted_actions = list(scripted_actions)
        self.calls: list[tuple[str, dict]] = []
        self._step_idx = 0
        self._next_game_id = 1

    def call(self, method: str, params: dict | None = None) -> dict:
        self.calls.append((method, dict(params or {})))
        if method == "create":
            gid = self._next_game_id
            self._next_game_id += 1
            self._step_idx = 0
            return {"gameId": gid}
        if method == "valid_actions":
            if self._step_idx >= len(self.scripted_actions):
                return {"turnPhase": "gameOver", "activePlayer": 0,
                        "mask": [0] * ACTION_SPACE_SIZE}
            mask = [0] * ACTION_SPACE_SIZE
            # Allow the scripted target plus 5 random other actions.
            target = self.scripted_actions[self._step_idx]
            mask[target] = 1
            for j in range(5):
                mask[(target + j + 1) % ACTION_SPACE_SIZE] = 1
            return {"turnPhase": "action", "activePlayer": 0, "mask": mask}
        if method == "get_observation":
            return _fake_obs_raw()
        if method == "ai_action":
            target = self.scripted_actions[self._step_idx]
            return {"actionIndex": int(target), "action": {"type": "stub"}}
        if method == "step_action":
            self._step_idx += 1
            return {"done": self._step_idx >= len(self.scripted_actions)}
        if method == "delete_game":
            return {}
        if method == "decode_action":
            return {"action": {"type": "stub"}}
        raise RuntimeError(f"unexpected bridge call: {method}")

    def close(self):
        pass


def _fake_obs_raw():
    """Mimic the bridge's get_observation output shape."""
    return {
        "board": np.zeros((BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32).tolist(),
        "hand": [0, 0, 0, 0],
        "handSeq": [-1, -1, -1, -1, -1, -1],
        "scores": [0, 0, 0, 0],
        "treasures": 0,
        "catastrophesRemaining": 0,
        "bagSize": 0,
        "actionsRemaining": 2,
        "turnPhase": 0,
        "currentPlayer": 0,
        "playerIndex": 0,
        "numPlayers": 2,
        "leaderPositions": [-1] * 8,
        "opponentScores": [[0, 0, 0, 0]],
        "opponentLeaderPositions": [[-1] * 8],
    }


def test_evaluate_top1_match_returns_expected_keys():
    """Basic structure check — uses a stub bridge with a few scripted moves."""
    from evaluate import evaluate_top1_match_vs_heuristic

    bridge = FakeBridge(scripted_actions=[0, TYPE_BASES[5], TYPE_BASES[8]])
    torch.manual_seed(0)
    model = PolicyValueNetwork()

    res = evaluate_top1_match_vs_heuristic(
        model, num_games=1, bridge=bridge,
    )
    assert "top1_acc" in res
    assert "n_decisions" in res
    assert isinstance(res["top1_acc"], float)
    assert 0.0 <= res["top1_acc"] <= 1.0
    assert res["n_decisions"] > 0


def test_evaluate_top1_match_perfect_when_target_is_only_legal():
    """If only one action is legal at every step (and it equals the
    heuristic's choice), the model's argmax under mask is forced to match
    — top1_acc should be 1.0."""
    from evaluate import evaluate_top1_match_vs_heuristic

    class SingleLegalBridge(FakeBridge):
        def call(self, method, params=None):
            if method == "valid_actions":
                if self._step_idx >= len(self.scripted_actions):
                    return {"turnPhase": "gameOver", "activePlayer": 0,
                            "mask": [0] * ACTION_SPACE_SIZE}
                mask = [0] * ACTION_SPACE_SIZE
                mask[self.scripted_actions[self._step_idx]] = 1
                return {"turnPhase": "action", "activePlayer": 0, "mask": mask}
            return super().call(method, params)

    bridge = SingleLegalBridge(scripted_actions=[0, TYPE_BASES[2], TYPE_BASES[5], TYPE_BASES[8]])
    torch.manual_seed(0)
    model = PolicyValueNetwork()

    res = evaluate_top1_match_vs_heuristic(
        model, num_games=1, bridge=bridge,
    )
    assert res["top1_acc"] == 1.0
    assert res["n_decisions"] == 4


def test_evaluate_top1_match_respects_decision_cap():
    """num_decisions caps the number of decisions sampled across games."""
    from evaluate import evaluate_top1_match_vs_heuristic

    bridge = FakeBridge(scripted_actions=[0] * 100)
    torch.manual_seed(0)
    model = PolicyValueNetwork()

    res = evaluate_top1_match_vs_heuristic(
        model, num_games=10, num_decisions=7, bridge=bridge,
    )
    assert res["n_decisions"] == 7
