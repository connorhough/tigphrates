"""
Gymnasium + PettingZoo environment wrapping the Tigris & Euphrates TypeScript engine
via a JSON-over-stdin/stdout bridge to a Node.js subprocess.

Usage:
    env = TigphratesEnv(player_count=2)
    obs, info = env.reset()
    while True:
        action = env.action_space.sample(mask=env.action_mask())
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""

from __future__ import annotations

import json
import os
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path

BOARD_ROWS = 11
BOARD_COLS = 16
BOARD_CHANNELS = 15  # see src/bridge/server.ts encodeBoard for channel layout
MAX_PLAYERS = 4

# Must match the TS server's action space layout
_CELLS = BOARD_ROWS * BOARD_COLS  # 176
ACTION_SPACE_SIZE = (
    8 * _CELLS      # placeTile (4 colors) + placeLeader (4 colors) × 176
    + 4              # withdrawLeader
    + _CELLS         # placeCatastrophe
    + 64             # swapTiles (6-bit mask)
    + 1              # pass
    + 64             # commitSupport (6-bit mask)
    + 4              # chooseWarOrder
    + 6              # buildMonument
    + 1              # declineMonument
)  # = 1728


class BridgeProcess:
    """Manages the Node.js subprocess running the TS bridge server."""

    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        self._proc = subprocess.Popen(
            ["npx", "tsx", str(project_root / "src" / "bridge" / "server.ts")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root),
            text=True,
            bufsize=1,  # line-buffered
        )
        self._next_id = 1
        # Read the ready message
        ready = self._read_response()
        result = ready.get("result", {})
        if not result.get("ready"):
            raise RuntimeError(f"Bridge server failed to start: {ready}")

    def call(self, method: str, params: dict | None = None) -> dict:
        """Send a request and return the result (blocking)."""
        req_id = self._next_id
        self._next_id += 1
        msg = json.dumps({"id": req_id, "method": method, "params": params or {}})
        self._proc.stdin.write(msg + "\n")
        self._proc.stdin.flush()
        resp = self._read_response()
        if "error" in resp:
            raise RuntimeError(f"Bridge error ({method}): {resp['error']}")
        return resp.get("result", {})

    def _read_response(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            stderr = self._proc.stderr.read()
            raise RuntimeError(f"Bridge process died. stderr: {stderr}")
        return json.loads(line)

    def close(self):
        if self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)


class TigphratesEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper for Tigris & Euphrates.

    The agent controls one player; all other players use the built-in
    heuristic AI (simpleAI). The environment automatically advances
    through opponent turns and sub-phases handled by the AI.

    Observation: dict with
        - board: (13, 11, 16) float32 tensor
        - hand: (4,) int   — tile counts by color
        - scores: (4,) int — VP by color
        - meta: (8,) float — treasures, catastrophes, bag_size, actions_remaining,
                              turn_phase, current_player, player_index, num_players
        - conflict: (7,) float — conflict info or zeros

    Action: int in [0, ACTION_SPACE_SIZE) with masking
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        player_count: int = 2,
        agent_player: int = 0,
        max_turns: int = 500,
        opponent_policy=None,
    ):
        """
        opponent_policy: optional Callable(observation_dict, action_mask) -> int
            If provided, replaces the built-in heuristic AI for non-agent
            players. Used for self-play training and pool/league evaluation.
        """
        super().__init__()
        self.player_count = player_count
        self.agent_player = agent_player
        self.max_turns = max_turns
        self.opponent_policy = opponent_policy

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, 4, shape=(BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32),
            "hand": spaces.Box(0, 6, shape=(4,), dtype=np.int32),
            "hand_seq": spaces.Box(-1, 3, shape=(6,), dtype=np.int32),
            "scores": spaces.Box(0, 200, shape=(4,), dtype=np.int32),
            "meta": spaces.Box(-1, 500, shape=(8,), dtype=np.float32),
            "conflict": spaces.Box(-1, 200, shape=(7,), dtype=np.float32),
            "leaders": spaces.Box(-1, 16, shape=(8,), dtype=np.float32),
            "opp_scores": spaces.Box(0, 200, shape=(4,), dtype=np.float32),
            "opp_leaders": spaces.Box(-1, 16, shape=(8,), dtype=np.float32),
        })

        self._bridge: BridgeProcess | None = None
        self._game_id: int | None = None
        self._turn_count = 0
        self._last_mask: np.ndarray | None = None

    def _ensure_bridge(self):
        if self._bridge is None:
            self._bridge = BridgeProcess()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_bridge()

        if self._game_id is None:
            result = self._bridge.call("create", {"playerCount": self.player_count})
            self._game_id = result["gameId"]
        else:
            self._bridge.call("reset", {"gameId": self._game_id, "playerCount": self.player_count})

        self._turn_count = 0
        self._last_mask = None  # invalidate any cached mask from prior episode

        # If agent isn't first to act, advance AI turns
        self._advance_ai_turns()

        obs = self._get_obs()
        return obs, {"turn": self._turn_count}

    def step(self, action: int):
        assert self._game_id is not None, "Must call reset() first"

        # Fast path: when the opponent policy is the heuristic, the bridge
        # can run the agent's action AND every opponent's turn server-side
        # in one RPC, returning the post-loop observation + mask. Cuts the
        # 3-6 RPCs per step (step + valid_actions + ai_action + step_action
        # + ...) down to 1.
        if self.opponent_policy is None:
            result = self._bridge.call("agent_step", {
                "gameId": self._game_id,
                "actionIndex": int(action),
                "agentPlayer": self.agent_player,
            })
            reward = result["reward"]
            done = result["done"]
            info = result.get("info") or {}
            self._last_mask = np.array(result["mask"], dtype=np.int8)
            obs = self._raw_obs_to_dict(result["obs"])
            self._turn_count += 1
            truncated = self._turn_count >= self.max_turns and not done
            info["turn"] = self._turn_count
            return obs, reward, done, truncated, info

        # Slow path: trained-policy opponent. Python drives each opponent
        # decision through the model, so we can't collapse on the bridge.
        result = self._bridge.call("step", {
            "gameId": self._game_id,
            "actionIndex": int(action),
            "playerIndex": self.agent_player,
        })

        reward = result["reward"]
        done = result["done"]
        info = result.get("info", {}) or {}

        if not done:
            done_during_opp = self._advance_ai_turns()
            if done_during_opp:
                done = True
                reward = self._terminal_reward()
                info["scores"] = self._final_scores_info()

        self._turn_count += 1
        truncated = self._turn_count >= self.max_turns and not done

        # Invalidate the mask cache: state changed after action_mask() was
        # called at the start of this iteration, so any cached value is stale.
        self._last_mask = None

        obs = self._get_obs()
        info["turn"] = self._turn_count

        return obs, reward, done, truncated, info

    def _raw_obs_to_dict(self, raw: dict) -> dict:
        """Shape a raw bridge observation (already encoded) into the same
        dict the policy network expects. Mirrors `_get_obs` without doing a
        second `get_observation` round-trip."""
        board = np.array(raw["board"], dtype=np.float32)
        hand = np.array(raw["hand"], dtype=np.int32)
        hand_seq = np.array(raw.get("handSeq", [-1] * 6), dtype=np.int32)
        scores = np.array(raw["scores"], dtype=np.int32)
        meta = np.array([
            raw["treasures"], raw["catastrophesRemaining"], raw["bagSize"],
            raw["actionsRemaining"], raw["turnPhase"], raw["currentPlayer"],
            raw["playerIndex"], raw["numPlayers"],
        ], dtype=np.float32)
        conflict_raw = raw.get("conflict")
        if conflict_raw:
            conflict = np.array([
                conflict_raw["type"], conflict_raw["color"],
                conflict_raw["attackerStrength"], conflict_raw["defenderStrength"],
                1.0 if conflict_raw["attackerCommitted"] else 0.0,
                1.0 if conflict_raw["isAttacker"] else 0.0,
                1.0 if conflict_raw["isDefender"] else 0.0,
            ], dtype=np.float32)
        else:
            conflict = np.zeros(7, dtype=np.float32)
        leader_pos = np.array(raw["leaderPositions"], dtype=np.float32)
        opp_scores_raw = raw.get("opponentScores", [])
        opp_scores = (np.array(opp_scores_raw[0], dtype=np.float32)
                      if opp_scores_raw else np.zeros(4, dtype=np.float32))
        opp_leaders_raw = raw.get("opponentLeaderPositions", [])
        opp_leaders = (np.array(opp_leaders_raw[0], dtype=np.float32)
                       if opp_leaders_raw else np.full(8, -1.0, dtype=np.float32))
        return {
            "board": board, "hand": hand, "hand_seq": hand_seq, "scores": scores,
            "meta": meta, "conflict": conflict, "leaders": leader_pos,
            "opp_scores": opp_scores, "opp_leaders": opp_leaders,
        }

    def _terminal_reward(self) -> float:
        agent_obs = self._get_obs_for(self.agent_player)
        agent_score = float(np.min(agent_obs["scores"])) + float(agent_obs.get("treasures", 0) or 0)
        best_opp = -1.0
        for pi in range(self.player_count):
            if pi == self.agent_player:
                continue
            opp_obs = self._get_obs_for(pi)
            opp_score = float(np.min(opp_obs["scores"])) + float(opp_obs.get("treasures", 0) or 0)
            if opp_score > best_opp:
                best_opp = opp_score
        return 1.0 if agent_score > best_opp else (-1.0 if agent_score < best_opp else 0.0)

    def _final_scores_info(self) -> list:
        out = []
        for pi in range(self.player_count):
            obs = self._get_obs_for(pi)
            scores = obs["scores"]
            out.append({
                "min": int(np.min(scores)),
                "red": int(scores[0]),
                "blue": int(scores[1]),
                "green": int(scores[2]),
                "black": int(scores[3]),
                "treasures": int(obs.get("treasures", 0) or 0),
            })
        return out

    def _get_obs_for(self, player_idx: int) -> dict:
        raw = self._bridge.call("get_observation", {
            "gameId": self._game_id,
            "playerIndex": player_idx,
        })
        return {
            "board": np.array(raw["board"], dtype=np.float32),
            "hand": np.array(raw["hand"], dtype=np.int32),
            "hand_seq": np.array(raw.get("handSeq", [-1] * 6), dtype=np.int32),
            "scores": np.array(raw["scores"], dtype=np.int32),
            "treasures": raw["treasures"],
            "meta": np.array([
                raw["treasures"], raw["catastrophesRemaining"], raw["bagSize"],
                raw["actionsRemaining"], raw["turnPhase"], raw["currentPlayer"],
                raw["playerIndex"], raw["numPlayers"],
            ], dtype=np.float32),
        }

    def action_mask(self) -> np.ndarray:
        """Return a boolean mask over the action space. Reuses the mask
        cached by the fast-path agent_step RPC when available; otherwise
        fetches a fresh one from the bridge."""
        assert self._game_id is not None
        if self._last_mask is not None:
            return self._last_mask
        result = self._bridge.call("valid_actions", {"gameId": self._game_id})
        self._last_mask = np.array(result["mask"], dtype=np.int8)
        return self._last_mask

    def expert_action_index(self) -> int:
        """Return the simpleAI's chosen action as a flat action index for the
        current active player, or -1 if it can't be encoded. Used for
        BC-auxiliary loss during PPO so the policy stays anchored to
        heuristic-quality moves while exploring."""
        assert self._game_id is not None
        ai = self._bridge.call("ai_action", {"gameId": self._game_id})
        idx = ai.get("actionIndex", -1)
        return int(idx)

    def _advance_ai_turns(self) -> bool:
        """
        Advance the game while the active player is not the agent.
        Uses the injected opponent_policy if provided, otherwise the built-in
        heuristic AI. Returns True if the game terminated during the advance.
        """
        safety = 0
        while safety < 5000:
            safety += 1
            va = self._bridge.call("valid_actions", {"gameId": self._game_id})

            if va["turnPhase"] == "gameOver":
                return True
            if va["activePlayer"] == self.agent_player:
                return False

            if self.opponent_policy is not None:
                opp_idx = va["activePlayer"]
                opp_obs = self._get_obs_for(opp_idx)
                # Build a policy-compatible obs (fill in conflict / leaders /
                # opp_scores / opp_leaders fields the same way _get_obs does).
                opp_obs_full = self._get_obs_for_policy(opp_idx)
                mask = np.array(va["mask"], dtype=np.int8)
                action = self.opponent_policy(opp_obs_full, mask)
                result = self._bridge.call("step", {
                    "gameId": self._game_id,
                    "actionIndex": int(action),
                    "playerIndex": opp_idx,
                })
                _ = opp_obs  # keep variable used
            else:
                ai = self._bridge.call("ai_action", {"gameId": self._game_id})
                result = self._bridge.call("step_action", {
                    "gameId": self._game_id,
                    "action": ai["action"],
                    "playerIndex": va["activePlayer"],
                })

            if result["done"]:
                return True

        return False

    def _get_obs_for_policy(self, player_idx: int) -> dict:
        """Full observation dict shaped for the policy network, viewed from
        an arbitrary player's perspective (used for opponent_policy)."""
        raw = self._bridge.call("get_observation", {
            "gameId": self._game_id,
            "playerIndex": player_idx,
        })
        board = np.array(raw["board"], dtype=np.float32)
        hand = np.array(raw["hand"], dtype=np.int32)
        hand_seq = np.array(raw.get("handSeq", [-1] * 6), dtype=np.int32)
        scores = np.array(raw["scores"], dtype=np.int32)
        meta = np.array([
            raw["treasures"], raw["catastrophesRemaining"], raw["bagSize"],
            raw["actionsRemaining"], raw["turnPhase"], raw["currentPlayer"],
            raw["playerIndex"], raw["numPlayers"],
        ], dtype=np.float32)
        conflict_raw = raw.get("conflict")
        if conflict_raw:
            conflict = np.array([
                conflict_raw["type"], conflict_raw["color"],
                conflict_raw["attackerStrength"], conflict_raw["defenderStrength"],
                1.0 if conflict_raw["attackerCommitted"] else 0.0,
                1.0 if conflict_raw["isAttacker"] else 0.0,
                1.0 if conflict_raw["isDefender"] else 0.0,
            ], dtype=np.float32)
        else:
            conflict = np.zeros(7, dtype=np.float32)
        leader_pos = np.array(raw["leaderPositions"], dtype=np.float32)
        opp_scores_raw = raw.get("opponentScores", [])
        opp_scores = np.array(opp_scores_raw[0], dtype=np.float32) if opp_scores_raw else np.zeros(4, dtype=np.float32)
        opp_leaders_raw = raw.get("opponentLeaderPositions", [])
        opp_leaders = np.array(opp_leaders_raw[0], dtype=np.float32) if opp_leaders_raw else np.full(8, -1.0, dtype=np.float32)
        return {
            "board": board, "hand": hand, "hand_seq": hand_seq, "scores": scores,
            "meta": meta, "conflict": conflict, "leaders": leader_pos,
            "opp_scores": opp_scores, "opp_leaders": opp_leaders,
        }

    def _get_obs(self) -> dict:
        raw = self._bridge.call("get_observation", {
            "gameId": self._game_id,
            "playerIndex": self.agent_player,
        })

        board = np.array(raw["board"], dtype=np.float32)
        hand = np.array(raw["hand"], dtype=np.int32)
        hand_seq = np.array(raw.get("handSeq", [-1] * 6), dtype=np.int32)
        scores = np.array(raw["scores"], dtype=np.int32)

        meta = np.array([
            raw["treasures"],
            raw["catastrophesRemaining"],
            raw["bagSize"],
            raw["actionsRemaining"],
            raw["turnPhase"],
            raw["currentPlayer"],
            raw["playerIndex"],
            raw["numPlayers"],
        ], dtype=np.float32)

        conflict_raw = raw.get("conflict")
        if conflict_raw:
            conflict = np.array([
                conflict_raw["type"],
                conflict_raw["color"],
                conflict_raw["attackerStrength"],
                conflict_raw["defenderStrength"],
                1.0 if conflict_raw["attackerCommitted"] else 0.0,
                1.0 if conflict_raw["isAttacker"] else 0.0,
                1.0 if conflict_raw["isDefender"] else 0.0,
            ], dtype=np.float32)
        else:
            conflict = np.zeros(7, dtype=np.float32)

        # Own leader positions (4 leaders × row,col)
        leader_pos = np.array(raw["leaderPositions"], dtype=np.float32)

        # Opponent scores and leader positions (first opponent for 2p)
        opp_scores_raw = raw.get("opponentScores", [])
        if opp_scores_raw:
            opp_scores = np.array(opp_scores_raw[0], dtype=np.float32)
        else:
            opp_scores = np.zeros(4, dtype=np.float32)

        opp_leaders_raw = raw.get("opponentLeaderPositions", [])
        if opp_leaders_raw:
            opp_leaders = np.array(opp_leaders_raw[0], dtype=np.float32)
        else:
            opp_leaders = np.full(8, -1.0, dtype=np.float32)

        return {
            "board": board,
            "hand": hand,
            "hand_seq": hand_seq,
            "scores": scores,
            "meta": meta,
            "conflict": conflict,
            "leaders": leader_pos,
            "opp_scores": opp_scores,
            "opp_leaders": opp_leaders,
        }

    def close(self):
        if self._bridge:
            self._bridge.close()
            self._bridge = None


# --- PettingZoo parallel environment (all players are agents) ---

class TigphratesMultiAgentEnv:
    """
    PettingZoo-style AEC (Agent-Environment-Cycle) wrapper where
    all players are RL agents.

    Usage:
        env = TigphratesMultiAgentEnv(player_count=2)
        env.reset()
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                action = None
            else:
                mask = env.action_mask()
                action = ...  # pick action
            env.step(action)
    """

    def __init__(self, player_count: int = 2, max_turns: int = 500):
        self.player_count = player_count
        self.max_turns = max_turns
        self.agents = [f"player_{i}" for i in range(player_count)]
        self.possible_agents = list(self.agents)

        self._bridge: BridgeProcess | None = None
        self._game_id: int | None = None
        self._turn_count = 0
        self._rewards = {a: 0.0 for a in self.agents}
        self._dones = {a: False for a in self.agents}
        self._current_agent_idx = 0

    def _ensure_bridge(self):
        if self._bridge is None:
            self._bridge = BridgeProcess()

    def reset(self, seed=None, options=None):
        self._ensure_bridge()
        if self._game_id is None:
            result = self._bridge.call("create", {"playerCount": self.player_count})
            self._game_id = result["gameId"]
        else:
            self._bridge.call("reset", {"gameId": self._game_id, "playerCount": self.player_count})
        self._turn_count = 0
        self._rewards = {a: 0.0 for a in self.agents}
        self._dones = {a: False for a in self.agents}
        self._update_current_agent()

    def _update_current_agent(self):
        va = self._bridge.call("valid_actions", {"gameId": self._game_id})
        self._current_agent_idx = va["activePlayer"]

    @property
    def agent_selection(self) -> str:
        return self.agents[self._current_agent_idx]

    def observe(self, agent: str) -> dict:
        idx = self.agents.index(agent)
        raw = self._bridge.call("get_observation", {
            "gameId": self._game_id,
            "playerIndex": idx,
        })
        return raw  # Return raw dict; downstream code can tensorize

    def action_mask(self) -> np.ndarray:
        result = self._bridge.call("valid_actions", {"gameId": self._game_id})
        return np.array(result["mask"], dtype=np.int8)

    def step(self, action: int | None):
        if action is None:
            return

        agent_idx = self._current_agent_idx
        result = self._bridge.call("step", {
            "gameId": self._game_id,
            "actionIndex": int(action),
            "playerIndex": agent_idx,
        })

        self._rewards[self.agents[agent_idx]] += result["reward"]

        if result["done"]:
            for a in self.agents:
                self._dones[a] = True
        else:
            self._turn_count += 1
            self._update_current_agent()

    def last(self):
        agent = self.agent_selection
        idx = self.agents.index(agent)
        obs = self.observe(agent)
        reward = self._rewards[agent]
        self._rewards[agent] = 0.0
        terminated = self._dones[agent]
        truncated = self._turn_count >= self.max_turns and not terminated
        return obs, reward, terminated, truncated, {}

    def agent_iter(self):
        """Iterate over agents until game ends."""
        while not all(self._dones.values()) and self._turn_count < self.max_turns:
            yield self.agent_selection

    def close(self):
        if self._bridge:
            self._bridge.close()
            self._bridge = None


if __name__ == "__main__":
    # Quick smoke test
    print("Creating environment...")
    env = TigphratesEnv(player_count=2)
    obs, info = env.reset()
    print(f"Board shape: {obs['board'].shape}")
    print(f"Hand: {obs['hand']}")
    print(f"Scores: {obs['scores']}")
    print(f"Meta: {obs['meta']}")

    mask = env.action_mask()
    valid_count = mask.sum()
    print(f"Valid actions: {valid_count} / {ACTION_SPACE_SIZE}")

    # Play a few random steps
    for step_i in range(20):
        mask = env.action_mask()
        valid_indices = np.where(mask == 1)[0]
        if len(valid_indices) == 0:
            print(f"Step {step_i}: No valid actions!")
            break
        action = np.random.choice(valid_indices)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_i}: action={action}, reward={reward:.4f}, done={terminated}, trunc={truncated}")
        if terminated or truncated:
            break

    env.close()
    print("Done!")
