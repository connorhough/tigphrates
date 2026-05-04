"""Tests for AlphaZero-style PUCT MCTS at inference (python/mcts.py).

We use stub model + stub bridge objects so the tests don't spin up Node.
The integration smoke test (marked) does spin up a real bridge if available
and runs MCTS with a tiny simulation budget.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import pytest

# Ensure python/ is importable for direct module imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Helpers / stubs --------------------------------------------------------

class StubBridge:
    """Minimal in-memory bridge stub.

    Knows how to: report the active player and turn phase, expose a tiny
    `valid_actions` mask, "step" by deterministically transitioning to a
    fixed terminal state, and serve `get_state`/`load_state`/`delete_game`
    via opaque JSON blobs (just integer tags).

    The "game" lives in self.games[gid] = {"player": int, "phase": str,
    "step": int, "terminal": bool}. It's only rich enough to drive the
    selection / expansion code paths.
    """

    def __init__(
        self,
        action_space_size: int = 4,
        legal_actions: list[int] | None = None,
        max_steps: int = 1,
    ):
        self.action_space_size = action_space_size
        self.legal = legal_actions if legal_actions is not None else [0, 1, 2]
        self.max_steps = max_steps
        self._next_id = 1
        self.games: dict[int, dict] = {}
        self.calls: list[tuple[str, dict]] = []

    def _new_game(self, step: int = 0, player: int = 0) -> int:
        gid = self._next_id
        self._next_id += 1
        self.games[gid] = {"step": step, "player": player, "phase": "action",
                           "terminal": step >= self.max_steps}
        return gid

    def call(self, method: str, params: dict | None = None) -> dict:
        self.calls.append((method, dict(params or {})))
        params = params or {}
        if method == "get_state":
            gid = params["gameId"]
            return {"state": dict(self.games[gid])}
        if method == "load_state":
            blob = params["state"]
            new_gid = self._next_id
            self._next_id += 1
            self.games[new_gid] = dict(blob)
            return {"gameId": new_gid}
        if method == "delete_game":
            gid = params.get("gameId")
            self.games.pop(gid, None)
            return {"deleted": True}
        if method == "valid_actions":
            gid = params["gameId"]
            g = self.games[gid]
            mask = np.zeros(self.action_space_size, dtype=np.int8)
            if not g["terminal"]:
                for a in self.legal:
                    mask[a] = 1
            actions = [{"index": a, "label": f"a{a}"} for a in self.legal] if not g["terminal"] else []
            return {
                "activePlayer": g["player"],
                "turnPhase": "gameOver" if g["terminal"] else g["phase"],
                "actions": actions,
                "mask": mask.tolist(),
            }
        if method == "get_observation":
            return {"raw": True}
        if method == "step":
            gid = params["gameId"]
            g = self.games[gid]
            g["step"] += 1
            g["player"] = 1 - g["player"]
            if g["step"] >= self.max_steps:
                g["terminal"] = True
                g["phase"] = "gameOver"
            return {
                "reward": 0.0,
                "done": g["terminal"],
                "activePlayer": g["player"],
                "turnPhase": "gameOver" if g["terminal"] else g["phase"],
            }
        if method == "decode_action":
            return {"action": {"type": "stub", "i": params["actionIndex"]},
                    "label": f"a{params['actionIndex']}",
                    "activePlayer": self.games[params["gameId"]]["player"]}
        raise NotImplementedError(method)


class StubModelEvaluator:
    """Stub callable: takes (bridge, gameId, playerIndex, mask) -> (prior, value).

    `prior_factory` builds the per-call prior given the call count and mask
    so tests can encode tree-shape expectations.
    """

    def __init__(self, prior_factory, value_factory=None):
        self.prior_factory = prior_factory
        self.value_factory = value_factory or (lambda call_index: 0.0)
        self.calls = 0
        self.history = []

    def __call__(self, bridge, game_id, player_index, mask):
        idx = self.calls
        prior = self.prior_factory(idx, mask)
        value = self.value_factory(idx)
        self.history.append({"gameId": game_id, "playerIndex": player_index,
                             "mask": np.array(mask, dtype=np.int8).copy(),
                             "prior": np.array(prior, dtype=np.float32).copy(),
                             "value": value})
        self.calls += 1
        return np.asarray(prior, dtype=np.float32), float(value)


# --- 1. Pure PUCT formula --------------------------------------------------

def test_puct_score_matches_hand_computation():
    from mcts import puct_score
    expected = 0.5 + 1.5 * 0.6 * math.sqrt(10) / (1 + 2)
    got = puct_score(q=0.5, p=0.6, n_total=10, n_action=2, c_puct=1.5)
    assert math.isclose(got, expected, rel_tol=1e-7)


def test_puct_score_zero_visits():
    from mcts import puct_score
    s = puct_score(q=0.3, p=0.5, n_total=0, n_action=0, c_puct=1.5)
    assert math.isclose(s, 0.3, rel_tol=1e-7)


def test_puct_score_unvisited_child_gets_full_prior_term():
    from mcts import puct_score
    s = puct_score(q=0.0, p=0.25, n_total=4, n_action=0, c_puct=2.0)
    expected = 0.0 + 2.0 * 0.25 * math.sqrt(4) / 1
    assert math.isclose(s, expected, rel_tol=1e-7)


# --- 2. Action mask honored ------------------------------------------------

def test_mcts_respects_action_mask_in_prior():
    from mcts import MCTS

    bridge = StubBridge(action_space_size=3, legal_actions=[0, 2], max_steps=1)
    gid = bridge._new_game()

    model = StubModelEvaluator(
        prior_factory=lambda i, m: np.array([0.5, 0.3, 0.2], dtype=np.float32),
        value_factory=lambda i: 0.0,
    )

    mcts = MCTS(model=model, bridge=bridge, num_simulations=8, c_puct=1.5)
    out = mcts.pick_action(game_id=gid, player_index=0)

    assert out["policy_prior"][1] == 0.0
    assert out["visits"][1] == 0
    assert out["actionIndex"] in (0, 2)


# --- 3. Visit-count argmax at root -----------------------------------------

def test_mcts_picks_action_with_highest_visit_count():
    from mcts import MCTS, _Node

    bridge = StubBridge(action_space_size=3, legal_actions=[0, 1, 2], max_steps=1)
    model = StubModelEvaluator(prior_factory=lambda i, m: np.array([0.4, 0.4, 0.2]))
    mcts = MCTS(model=model, bridge=bridge, num_simulations=0, c_puct=1.5)

    node = _Node(
        to_play=0,
        prior=np.array([0.3, 0.5, 0.2], dtype=np.float32),
        legal_mask=np.array([1, 1, 1], dtype=np.int8),
        value=0.0,
    )
    node.child_visits = np.array([30, 5, 1], dtype=np.int64)
    node.child_value_sum = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    node.is_expanded = True

    chosen = mcts._select_argmax_visit(node)
    assert chosen == 0


# --- 4. Two-player Q-flip --------------------------------------------------

def test_two_player_q_flip_in_selection():
    """At an opponent-to-play node, the selection formula should use -Q
    (because we assume opponent picks the action that minimises root's value)."""
    from mcts import MCTS, _Node

    bridge = StubBridge(action_space_size=2, legal_actions=[0, 1], max_steps=1)
    model = StubModelEvaluator(prior_factory=lambda i, m: np.array([0.5, 0.5]))
    mcts = MCTS(model=model, bridge=bridge, num_simulations=0, c_puct=1.5)

    node = _Node(
        to_play=1,
        prior=np.array([0.5, 0.5], dtype=np.float32),
        legal_mask=np.array([1, 1], dtype=np.int8),
        value=0.0,
    )
    node.is_expanded = True
    node.child_visits = np.array([1, 1], dtype=np.int64)
    node.child_value_sum = np.array([0.8, -0.4], dtype=np.float64)

    chosen = mcts._select_child_puct(node, root_player=0)
    assert chosen == 1


def test_root_player_q_not_flipped():
    from mcts import MCTS, _Node

    bridge = StubBridge(action_space_size=2, legal_actions=[0, 1], max_steps=1)
    model = StubModelEvaluator(prior_factory=lambda i, m: np.array([0.5, 0.5]))
    mcts = MCTS(model=model, bridge=bridge, num_simulations=0, c_puct=1.5)

    node = _Node(
        to_play=0,
        prior=np.array([0.5, 0.5], dtype=np.float32),
        legal_mask=np.array([1, 1], dtype=np.int8),
        value=0.0,
    )
    node.is_expanded = True
    node.child_visits = np.array([1, 1], dtype=np.int64)
    node.child_value_sum = np.array([0.8, -0.4], dtype=np.float64)

    chosen = mcts._select_child_puct(node, root_player=0)
    assert chosen == 0


# --- 5. Simulation budget ---------------------------------------------------

def test_simulation_budget_respected():
    """With num_simulations=N, exactly N model forward passes happen.

    The expansion at the root counts as one of those passes (the first
    simulation expands the root and propagates V).
    """
    from mcts import MCTS

    bridge = StubBridge(action_space_size=3, legal_actions=[0, 1, 2], max_steps=10)
    gid = bridge._new_game()

    model = StubModelEvaluator(
        prior_factory=lambda i, m: np.array([0.5, 0.3, 0.2], dtype=np.float32),
        value_factory=lambda i: 0.1,
    )

    mcts = MCTS(model=model, bridge=bridge, num_simulations=10, c_puct=1.5)
    mcts.pick_action(game_id=gid, player_index=0)

    assert model.calls == 10, f"expected 10 model calls, got {model.calls}"


# --- 6. Returns valid action dict ------------------------------------------

def test_pick_action_returns_expected_shape():
    from mcts import MCTS

    bridge = StubBridge(action_space_size=3, legal_actions=[0, 1, 2], max_steps=1)
    gid = bridge._new_game()

    model = StubModelEvaluator(
        prior_factory=lambda i, m: np.array([0.5, 0.3, 0.2], dtype=np.float32),
        value_factory=lambda i: 0.1,
    )
    mcts = MCTS(model=model, bridge=bridge, num_simulations=4, c_puct=1.5)
    out = mcts.pick_action(game_id=gid, player_index=0)

    assert set(out.keys()) >= {"action", "actionIndex", "visits", "policy_prior", "root_value"}
    assert isinstance(out["actionIndex"], int)
    assert len(out["visits"]) == bridge.action_space_size
    assert len(out["policy_prior"]) == bridge.action_space_size
    assert isinstance(out["root_value"], float)
    assert out["action"]["type"] == "stub"


# --- 7. Integration smoke test ---------------------------------------------

@pytest.mark.integration
def test_mcts_with_real_bridge_terminates():
    """Spin up the real bridge, run MCTS with a tiny budget, verify it
    completes a single move without crashing. Slow - opt-in via -m integration."""
    from tigphrates_env import BridgeProcess
    from mcts import MCTS, build_default_evaluator

    bridge = BridgeProcess()
    try:
        created = bridge.call("create", {"playerCount": 2})
        gid = created["gameId"]

        from train import PolicyValueNetwork
        import torch
        torch.manual_seed(0)
        model = PolicyValueNetwork()
        model.train(False)
        evaluator = build_default_evaluator(model)

        mcts = MCTS(model=evaluator, bridge=bridge, num_simulations=5, c_puct=1.5)
        t0 = time.time()
        out = mcts.pick_action(game_id=gid, player_index=0)
        elapsed = time.time() - t0
        print(f"\n[mcts smoke] 5-sim move took {elapsed:.2f}s -> action {out['actionIndex']}")
        assert "action" in out
        assert out["actionIndex"] >= 0
    finally:
        bridge.close()
