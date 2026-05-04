"""AlphaZero-style PUCT MCTS at inference time.

This module implements Monte Carlo Tree Search using a learned policy as
the prior and a learned value head as the leaf evaluator (no rollouts).

The selection rule is the AlphaZero PUCT variant:

    a* = argmax_a [ Q(s, a) + c_puct * P(s, a) * sqrt(sum_b N(s, b)) / (1 + N(s, a)) ]

Two-player handling
-------------------
Tigris & Euphrates is a 2-4 player game. We treat it as a zero-sum two-player
game from the root player's perspective:

  * At root-player nodes we use +Q.
  * At non-root-player nodes (an opponent decision point) we use -Q because we
    assume the opponent picks the action that minimises root's value.

State save/restore
------------------
We rely on the bridge's `get_state` / `load_state` / `delete_game` RPCs.
At the start of `pick_action` we snapshot the root state once. For each
simulation we allocate a single scratch gameId via `load_state`, walk down
the tree by issuing `step` RPCs against that scratch game, expand the leaf,
and discard the scratch game with `delete_game`. This is correct but RPC-heavy
- the bridge round-trip is the dominant cost at runtime.

Final move selection uses argmax over visit counts (more robust at low
simulation budgets than argmax over Q).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def puct_score(q: float, p: float, n_total: int, n_action: int, c_puct: float) -> float:
    """AlphaZero PUCT score for a single action.

    score = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

    Args:
        q: average value through this edge from the perspective of the
           player to move at the parent node. Caller must already have
           applied the two-player sign flip if needed.
        p: prior probability for the action under the policy network
           (already mask-and-renormalised).
        n_total: sum of visit counts across all sibling actions at the parent.
        n_action: visit count for this specific action.
        c_puct: exploration constant.
    """
    explore = c_puct * p * math.sqrt(max(n_total, 0)) / (1 + n_action)
    return q + explore


def normalise_prior(raw_prior: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Intersect a raw policy prior with the legal-action mask and renormalise.

    Returns a numpy array of the same length as `raw_prior`. If every entry
    is zero or masked out, returns a uniform distribution over the legal
    actions; if there are no legal actions at all, returns all-zeros.
    """
    raw = np.asarray(raw_prior, dtype=np.float64)
    m = np.asarray(mask, dtype=np.float64)
    if raw.shape != m.shape:
        raise ValueError(f"prior shape {raw.shape} != mask shape {m.shape}")
    masked = raw * m
    masked = np.maximum(masked, 0.0)  # guard against negative logits-as-prior
    s = masked.sum()
    if s > 0:
        return (masked / s).astype(np.float32)
    # Fallback: uniform over legal actions.
    legal_count = float(m.sum())
    if legal_count > 0:
        return (m / legal_count).astype(np.float32)
    return np.zeros_like(raw, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    """A single MCTS node. Children are indexed by flat action index.

    Storage uses dense numpy arrays sized to the action space; this is
    memory-cheap because the action space is fixed at 1728 entries and
    most games visit only a handful of nodes per move.
    """

    to_play: int
    prior: np.ndarray              # (A,) masked + normalised prior
    legal_mask: np.ndarray         # (A,) int8 mask of legal actions
    value: float                   # leaf value V(s) returned by the network
    is_expanded: bool = False
    is_terminal: bool = False
    # Per-action statistics.
    child_visits: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    child_value_sum: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    # Lazy children: action_index -> _Node
    children: dict = field(default_factory=dict)
    # Cached scratch state (the JSON state at this node) for replay-on-load_state.
    state_json: dict | None = None

    def __post_init__(self):
        a = self.prior.shape[0]
        if self.child_visits.size == 0:
            self.child_visits = np.zeros(a, dtype=np.int64)
        if self.child_value_sum.size == 0:
            self.child_value_sum = np.zeros(a, dtype=np.float64)

    @property
    def n_total(self) -> int:
        return int(self.child_visits.sum())

    def q_for(self, action_index: int) -> float:
        n = int(self.child_visits[action_index])
        if n == 0:
            return 0.0
        return float(self.child_value_sum[action_index] / n)


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------


ModelEvaluator = Callable[[object, int, int, np.ndarray], tuple[np.ndarray, float]]
"""(bridge, gameId, playerIndex, mask) -> (raw_prior, value)."""


class MCTS:
    """AlphaZero-style PUCT MCTS using a learned policy + value as leaf eval.

    The model evaluator is a callable that, given the bridge and a gameId/
    playerIndex/mask, returns `(raw_prior, value)`. We then mask + renormalise
    the prior internally. Decoupling the network call this way keeps the MCTS
    independent of the specific PolicyValueNetwork shape.
    """

    def __init__(
        self,
        model: ModelEvaluator,
        bridge,
        num_simulations: int,
        c_puct: float = 1.5,
    ):
        self.model = model
        self.bridge = bridge
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)

    # --- selection helpers (testable in isolation) ---

    def _select_child_puct(self, node: _Node, root_player: int) -> int:
        """Return the action index that maximises the PUCT score, applying
        the two-player Q-flip when it's not the root player's turn."""
        flip = -1.0 if node.to_play != root_player else 1.0
        best_score = -float("inf")
        best_action = -1
        n_total = node.n_total
        legal = np.where(node.legal_mask > 0)[0]
        for a in legal:
            q = node.q_for(int(a)) * flip
            score = puct_score(
                q=q,
                p=float(node.prior[a]),
                n_total=n_total,
                n_action=int(node.child_visits[a]),
                c_puct=self.c_puct,
            )
            if score > best_score:
                best_score = score
                best_action = int(a)
        return best_action

    def _select_argmax_visit(self, node: _Node) -> int:
        """Final move selection: action with the highest visit count.

        Ties broken by prior (then by index)."""
        visits = node.child_visits.astype(np.float64)
        # Mask out illegal actions.
        visits = np.where(node.legal_mask > 0, visits, -1.0)
        max_v = visits.max()
        candidates = np.where(visits == max_v)[0]
        if len(candidates) == 1:
            return int(candidates[0])
        # Tiebreak by prior.
        priors = node.prior[candidates]
        best = candidates[int(priors.argmax())]
        return int(best)

    # --- public API ---

    def pick_action(self, game_id: int, player_index: int) -> dict:
        """Run num_simulations rollouts from the bridge's current state.

        Returns:
            {"action": <decoded action dict>,
             "actionIndex": int,
             "visits": list[int],          # length = action space size
             "policy_prior": list[float],  # length = action space size
             "root_value": float}
        """
        # 1. Snapshot the root state JSON.
        snap = self.bridge.call("get_state", {"gameId": game_id})
        root_state = snap["state"]

        # 2. Build the root node by querying valid_actions + the model evaluator.
        root_va = self.bridge.call("valid_actions", {"gameId": game_id})
        root_mask = np.asarray(root_va["mask"], dtype=np.int8)
        root_to_play = int(root_va.get("activePlayer", player_index))
        if root_va.get("turnPhase") == "gameOver" or root_mask.sum() == 0:
            # Nothing to do - return a no-op shaped result. Caller is unlikely
            # to invoke MCTS on a terminal state but be defensive.
            return {
                "action": {"type": "pass"},
                "actionIndex": -1,
                "visits": [0] * len(root_mask),
                "policy_prior": [0.0] * len(root_mask),
                "root_value": 0.0,
            }

        raw_prior, root_value = self.model(self.bridge, game_id, player_index, root_mask)
        prior = normalise_prior(raw_prior, root_mask)
        root = _Node(
            to_play=root_to_play,
            prior=prior,
            legal_mask=root_mask,
            value=float(root_value),
            is_expanded=True,
            state_json=root_state,
        )
        # The root expansion itself is one model evaluation; subsequent
        # simulations descend, expand a leaf, and back up. So the budget
        # of N simulations corresponds to N model evaluations total.
        sims_remaining = max(0, self.num_simulations - 1)

        for _ in range(sims_remaining):
            self._run_one_simulation(root, game_id, player_index)

        # 3. Final move selection: visit-count argmax.
        action_index = self._select_argmax_visit(root)

        # 4. Decode the action dict from the original gameId (which still
        # reflects the root state - we've only ever touched scratch games).
        decoded = self.bridge.call(
            "decode_action", {"gameId": game_id, "actionIndex": action_index}
        )

        return {
            "action": decoded["action"],
            "label": decoded.get("label", ""),
            "actionIndex": int(action_index),
            "visits": root.child_visits.tolist(),
            "policy_prior": root.prior.tolist(),
            "root_value": float(root.value),
        }

    # --- internals ---

    def _run_one_simulation(self, root: _Node, root_game_id: int, root_player: int) -> None:
        """Execute one rollout: load scratch state, walk down the tree, expand
        a leaf, propagate the leaf value back up."""
        # Allocate a scratch gameId from the root state JSON.
        loaded = self.bridge.call("load_state", {"state": root.state_json})
        scratch_gid = loaded["gameId"]

        try:
            path: list[tuple[_Node, int]] = []  # (parent_node, action_taken)
            node = root

            # ---- 1. Selection: descend until we hit an unexpanded or terminal node.
            while node.is_expanded and not node.is_terminal:
                action_index = self._select_child_puct(node, root_player=root_player)
                if action_index < 0:
                    break
                path.append((node, action_index))

                child = node.children.get(action_index)
                if child is None:
                    # Take the action in the scratch game and expand.
                    step_res = self.bridge.call(
                        "step",
                        {
                            "gameId": scratch_gid,
                            "actionIndex": int(action_index),
                            "playerIndex": int(node.to_play),
                        },
                    )
                    child = self._expand_child(scratch_gid, step_res, root_player)
                    node.children[action_index] = child
                    node = child
                    break  # leaf reached; exit selection loop
                else:
                    # Replay the same step in the scratch game so the bridge
                    # state matches the tree node we just descended into.
                    self.bridge.call(
                        "step",
                        {
                            "gameId": scratch_gid,
                            "actionIndex": int(action_index),
                            "playerIndex": int(node.to_play),
                        },
                    )
                    node = child

            # ---- 2. Backup: propagate node.value (from root_player's POV)
            # up the path. The leaf's `.value` is already from root_player's
            # perspective because the model is queried with player_index=root.
            leaf_value = float(node.value)
            for parent, action in path:
                parent.child_visits[action] += 1
                parent.child_value_sum[action] += leaf_value
        finally:
            # Always discard the scratch gameId, even on error.
            try:
                self.bridge.call("delete_game", {"gameId": scratch_gid})
            except Exception:
                pass

    def _expand_child(self, scratch_gid: int, step_res: dict, root_player: int) -> _Node:
        """Build a new child node from a scratch game just stepped into.

        Queries valid_actions + the model evaluator, builds the masked prior,
        records the leaf value (from root's perspective), and snapshots the
        scratch game's state so future simulations can re-load it.
        """
        done = bool(step_res.get("done"))

        if done:
            # Terminal node: legal mask is empty, value is the per-bridge
            # terminal reward but we don't have a clean signed-from-root view
            # without inspecting scores. Fall back to 0 - the bridge's reward
            # is from the actor's perspective, not root's, and reconstructing
            # the sign is fragile. For a proper terminal value the caller
            # could load info["scores"]; the smoke test confirms terminal
            # nodes don't crash search.
            mask = np.zeros(0, dtype=np.int8)
            # Re-fetch length-aware mask if we need it for prior shape.
            va = self.bridge.call("valid_actions", {"gameId": scratch_gid})
            mask_full = np.asarray(va["mask"], dtype=np.int8)
            # Best-effort signed reward from info.scores.
            value = self._terminal_value(step_res, root_player)
            child = _Node(
                to_play=int(va.get("activePlayer", root_player)),
                prior=np.zeros_like(mask_full, dtype=np.float32),
                legal_mask=mask_full,
                value=value,
                is_expanded=True,
                is_terminal=True,
                state_json=None,
            )
            return child

        # Snapshot for future re-loads.
        snap = self.bridge.call("get_state", {"gameId": scratch_gid})
        state_json = snap["state"]

        va = self.bridge.call("valid_actions", {"gameId": scratch_gid})
        mask = np.asarray(va["mask"], dtype=np.int8)
        to_play = int(va.get("activePlayer", root_player))

        if mask.sum() == 0:
            # No legal actions but not flagged as done - treat as terminal.
            return _Node(
                to_play=to_play,
                prior=np.zeros_like(mask, dtype=np.float32),
                legal_mask=mask,
                value=0.0,
                is_expanded=True,
                is_terminal=True,
                state_json=state_json,
            )

        # Always evaluate from root_player's perspective so backed-up values
        # are consistently signed.
        raw_prior, value = self.model(self.bridge, scratch_gid, root_player, mask)
        prior = normalise_prior(raw_prior, mask)

        return _Node(
            to_play=to_play,
            prior=prior,
            legal_mask=mask,
            value=float(value),
            is_expanded=True,
            is_terminal=False,
            state_json=state_json,
        )

    @staticmethod
    def _terminal_value(step_res: dict, root_player: int) -> float:
        """Compute leaf value at a terminal node from the root player's view.

        Uses info["scores"] when available; otherwise falls back to the raw
        reward (which the bridge stamps from the actor's perspective).
        """
        info = step_res.get("info") or {}
        scores = info.get("scores")
        if scores:
            try:
                totals = [s["min"] + s["treasures"] for s in scores]
                root_total = totals[root_player]
                others = [t for i, t in enumerate(totals) if i != root_player]
                best_other = max(others) if others else 0
                if root_total > best_other:
                    return 1.0
                if root_total < best_other:
                    return -1.0
                return 0.0
            except (KeyError, IndexError, TypeError):
                pass
        # Fall back to the bridge's reward; sign may be off-by-perspective for
        # multi-player games but it's at least bounded and rare-path.
        return float(step_res.get("reward", 0.0))


# ---------------------------------------------------------------------------
# Default evaluator using the project's PolicyValueNetwork
# ---------------------------------------------------------------------------


def build_default_evaluator(model) -> ModelEvaluator:
    """Wrap a PolicyValueNetwork instance into a MCTS-compatible evaluator.

    The evaluator queries `get_observation` from the bridge, runs a single
    forward pass through `model.forward`, and returns the masked-prior +
    scalar value pair MCTS expects.

    The returned closure captures `model` by reference; callers can hot-swap
    the underlying weights between calls.
    """
    # Local imports keep mcts.py importable in environments that don't have
    # torch / the project policy module loaded (e.g. for unit-testing the
    # pure-MCTS code paths with stubs).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch  # noqa: F401
    from policy_server import _build_policy_obs
    from train import obs_to_tensors, TYPE_BASES_T

    def evaluator(bridge, game_id: int, player_index: int, mask: np.ndarray) -> tuple[np.ndarray, float]:
        obs_raw = bridge.call(
            "get_observation", {"gameId": game_id, "playerIndex": int(player_index)}
        )
        obs = _build_policy_obs(obs_raw)
        obs_t = obs_to_tensors(obs)
        with torch.no_grad():
            type_logits, param_logits, value = model.forward(obs_t)
            type_dist, param_padded, _type_mask = model.hierarchical_dists(
                type_logits, param_logits, np.asarray(mask, dtype=np.int8)
            )
            # Build a flat distribution over the full action space by
            # combining type prob with conditional param prob.
            B = type_logits.shape[0]
            assert B == 1
            # Type probs (masked).
            type_probs = torch.softmax(type_dist.logits, dim=-1)  # (1, NT)
            # Per-type param probs.
            param_probs = torch.softmax(param_padded, dim=-1)     # (1, NT, MP)
            # Replace NaNs (from -inf rows that softmax made nan).
            param_probs = torch.nan_to_num(param_probs, nan=0.0)
            # Joint = type_prob * param_prob, gather into flat layout via TYPE_BASES.
            type_bases = TYPE_BASES_T.to(param_probs.device)
            flat_size = int(mask.shape[-1])
            flat = np.zeros(flat_size, dtype=np.float32)
            type_probs_np = type_probs[0].detach().cpu().numpy()
            param_probs_np = param_probs[0].detach().cpu().numpy()  # (NT, MP)
            type_bases_np = type_bases.detach().cpu().numpy()
            for t in range(type_bases_np.shape[0]):
                # number of params for this type = next base - this base
                if t + 1 < type_bases_np.shape[0]:
                    sz = int(type_bases_np[t + 1] - type_bases_np[t])
                else:
                    sz = flat_size - int(type_bases_np[t])
                base = int(type_bases_np[t])
                flat[base:base + sz] = type_probs_np[t] * param_probs_np[t, :sz]

            v = float(value.item())
        return flat, v

    return evaluator
