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

# --- Action-range constants (must match src/bridge/encoder.ts) -------------
# Used by reward shaping to detect leader placements without round-tripping
# through the bridge.
PLACE_TILE_BASE = 0
PLACE_TILE_END = 4 * _CELLS                  # 704 (exclusive)
PLACE_LEADER_BASE = 4 * _CELLS               # 704
PLACE_LEADER_END = 8 * _CELLS                # 1408 (exclusive)

# Black leader is the king (color index 3 in src/bridge/encoder.ts COLOR_INDEX).
# A placeLeader action's color is encoded as `base + color * _CELLS + cell`,
# so any index whose `(rel // _CELLS) == 3` is a black-leader placement.
BLACK_LEADER_COLOR = 3

# buildMonument is action type 8 in the hierarchical layout (src/bridge/encoder.ts):
#   placeTile(704) + placeLeader(704) + withdrawLeader(4) + placeCatastrophe(176)
# + swapTiles(64) + pass(1) + commitSupport(64) + chooseWarOrder(4) = 1721 base,
# 6 slots wide.
BUILD_MONUMENT_BASE = 1721
BUILD_MONUMENT_END = BUILD_MONUMENT_BASE + 6  # 1727 (exclusive)

# --- Reward shaping helper -------------------------------------------------

# Cap on how many leader-placement bonuses can be awarded per game. Prevents
# the agent from farming the bonus by repeatedly withdrawing and re-placing.
LEADER_PLACE_CAP = 4
# Cap on monument-build awards per game (a player can build at most a handful
# of monuments in a real game; the bonus is intentionally one-shot-ish).
MONUMENT_BUILD_CAP = 2


def _is_leader_placement(action_index: int) -> bool:
    return PLACE_LEADER_BASE <= action_index < PLACE_LEADER_END


def _leader_placement_color(action_index: int) -> int:
    """Decode the leader color index (0=red,1=blue,2=green,3=black) from a
    placeLeader action. Caller must already have verified `_is_leader_placement`."""
    rel = action_index - PLACE_LEADER_BASE
    return rel // _CELLS


def _is_monument_build(action_index: int) -> bool:
    return BUILD_MONUMENT_BASE <= action_index < BUILD_MONUMENT_END


def _leader_placement_position(action_index: int) -> tuple[int, int]:
    """Decode (row, col) from a placeLeader action index. Caller must already
    have verified `_is_leader_placement`."""
    rel = action_index - PLACE_LEADER_BASE
    cell = rel % _CELLS
    return cell // BOARD_COLS, cell % BOARD_COLS


def _has_adjacent_tile(board: np.ndarray, row: int, col: int) -> bool:
    """Return True if any of the 4 orthogonal neighbours of (row, col) holds
    a tile of any color (channels 0-3 in the encoded board)."""
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = row + dr, col + dc
        if 0 <= rr < BOARD_ROWS and 0 <= cc < BOARD_COLS:
            if board[0:4, rr, cc].sum() > 0:
                return True
    return False


def _treasures_from_obs(obs: dict) -> float:
    """Read the active player's treasure count from an observation dict.
    Treasures live at meta[0] (see TigphratesEnv._get_obs)."""
    meta = obs.get("meta")
    if meta is None:
        return 0.0
    try:
        return float(meta[0])
    except (IndexError, TypeError):
        return 0.0


def compute_event_shaping_bonus(
    action_index: int,
    prev_obs: dict,
    next_obs: dict,
    global_step: int,
    leader_placements_so_far: int,
    monument_builds_so_far: int,
) -> float:
    """Compute a dense per-event shaping reward.

    Awards (subject to a single linear decay factor over SHAPING_DECAY_STEPS):
      - Leader-placement bonus for any placeLeader action. Black (king) leaders
        get KING_LEADER_BONUS instead of LEADER_PLACE_BONUS. Counts toward
        LEADER_PLACE_CAP (4 awards / game).
      - Kingdom-formation bonus when the placed leader is adjacent to a tile.
      - Treasure-collect bonus when the active player's treasure count went up
        from prev_obs to next_obs. Uncapped, one award per delta.
      - Monument-build bonus when the action falls in the buildMonument slot
        range. Capped at MONUMENT_BUILD_CAP awards / game.

    Reads env vars at call time so test monkeypatching works.
    """
    place_bonus = float(os.environ.get("LEADER_PLACE_BONUS", "0.05"))
    kingdom_bonus = float(os.environ.get("KINGDOM_FORM_BONUS", "0.1"))
    king_bonus = float(os.environ.get("KING_LEADER_BONUS", "0.10"))
    treasure_bonus = float(os.environ.get("TREASURE_COLLECT_BONUS", "0.15"))
    monument_bonus = float(os.environ.get("MONUMENT_BUILD_BONUS", "0.10"))
    decay_steps = int(os.environ.get("SHAPING_DECAY_STEPS", "200000"))

    # Fast no-op short-circuit: when every bonus is zero, this is a no-op
    # regardless of action / step. Guarantees byte-identical reward to the
    # pre-shaping pipeline when production tunes all bonuses to zero.
    if (place_bonus == 0.0 and kingdom_bonus == 0.0 and king_bonus == 0.0
            and treasure_bonus == 0.0 and monument_bonus == 0.0):
        return 0.0

    # Linear decay factor. At global_step=0, factor=1; at decay_steps, factor=0.
    if decay_steps <= 0:
        decay_factor = 0.0
    else:
        decay_factor = max(0.0, 1.0 - global_step / decay_steps)

    if decay_factor == 0.0:
        return 0.0

    bonus = 0.0

    # --- Leader placement (capped) -----------------------------------------
    if _is_leader_placement(action_index) and leader_placements_so_far < LEADER_PLACE_CAP:
        color = _leader_placement_color(action_index)
        # Black king gets the bumped bonus; other colors use the base.
        if color == BLACK_LEADER_COLOR:
            bonus += king_bonus
        else:
            bonus += place_bonus
        # Kingdom-formation bonus: leader is now adjacent to at least one tile.
        if kingdom_bonus > 0.0:
            row, col = _leader_placement_position(action_index)
            if _has_adjacent_tile(next_obs["board"], row, col):
                bonus += kingdom_bonus

    # --- Treasure collect (uncapped) ---------------------------------------
    if treasure_bonus > 0.0:
        prev_t = _treasures_from_obs(prev_obs)
        next_t = _treasures_from_obs(next_obs)
        delta = next_t - prev_t
        if delta > 0.0:
            bonus += treasure_bonus * delta

    # --- Monument build (capped at MONUMENT_BUILD_CAP) ---------------------
    if (monument_bonus > 0.0
            and _is_monument_build(action_index)
            and monument_builds_so_far < MONUMENT_BUILD_CAP):
        bonus += monument_bonus

    return bonus * decay_factor


def compute_leader_shaping_bonus(
    action_index: int,
    prev_obs: dict,
    next_obs: dict,
    global_step: int,
    leader_placements_so_far: int,
) -> float:
    """Backward-compatible alias for the original leader-only shaping helper.

    Forwards to `compute_event_shaping_bonus` with `monument_builds_so_far=0`.
    The treasure/monument/king bonuses default to zero only if the caller has
    explicitly cleared the env vars; the new defaults (KING=0.10, TREASURE=0.15,
    MONUMENT=0.10) are documented in compute_event_shaping_bonus and the
    callers in train.py opt into the richer signature directly.

    Existing test_shaping.py callers monkeypatch only the LEADER/KINGDOM env
    vars and rely on the new bonuses being absent for non-treasure / non-monument
    actions, which is naturally true: a leader placement with prev==next obs
    triggers no treasure delta and is not in the monument range.
    """
    return compute_event_shaping_bonus(
        action_index=action_index,
        prev_obs=prev_obs,
        next_obs=next_obs,
        global_step=global_step,
        leader_placements_so_far=leader_placements_so_far,
        monument_builds_so_far=0,
    )


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
