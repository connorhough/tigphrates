"""
Minimal HTTP server that exposes a trained Tigris & Euphrates policy.

POST /action
    Request:  { "state": <GameState JSON>, "playerIndex": <int> }
    Response: { "action": <GameAction JSON>, "label": <str>, "actionIndex": <int> }

The server owns one Node bridge subprocess; every incoming request is loaded
into the bridge as a fresh game (load_state), the observation is fetched, the
policy network picks an action, and decode_action resolves the index back to
a dispatch-ready GameAction.

Usage:
    pip install -r python/requirements.txt
    python python/policy_server.py [--model models/policy_best.pt] [--port 8765]

Browser integration: see PLAN.md section 4.1 for the RemotePolicyAdapter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tigphrates_env import BridgeProcess
from train import PolicyValueNetwork, obs_to_tensors, DEVICE

DEFAULT_PORT = 8765
DEFAULT_MODEL_PATH = "models/policy_best.pt"


def _build_policy_obs(raw: dict) -> dict:
    """Mirror the field shape produced by TigphratesEnv._get_obs_for_policy
    so the policy network sees identical inputs as during training."""
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


class PolicyServer:
    """Wraps a model + a Node bridge in a thread-safe-enough way for a single
    HTTP server. All bridge calls are serialized through a lock since the
    Node subprocess is a single stdin/stdout stream."""

    def __init__(self, model_path: str):
        self.bridge = BridgeProcess()
        self.model = PolicyValueNetwork()
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.train(False)
        self.model.to(DEVICE)
        self._lock = threading.Lock()

    def pick_action(self, state: dict, player_index: int) -> dict:
        with self._lock:
            loaded = self.bridge.call("load_state", {"state": state})
            gid = loaded["gameId"]
            try:
                obs_raw = self.bridge.call("get_observation", {
                    "gameId": gid, "playerIndex": player_index,
                })
                va = self.bridge.call("valid_actions", {"gameId": gid})
                mask = np.array(va["mask"], dtype=np.int8)
                if mask.sum() == 0:
                    return {"action": {"type": "pass"}, "label": "pass", "actionIndex": -1}
                obs = _build_policy_obs(obs_raw)
                obs_t = obs_to_tensors(obs)
                with torch.no_grad():
                    a, _, _, _ = self.model.get_action_and_value(obs_t, mask)
                action_idx = int(a.item())
                decoded = self.bridge.call("decode_action", {
                    "gameId": gid, "actionIndex": action_idx,
                })
                return {
                    "action": decoded["action"],
                    "label": decoded["label"],
                    "actionIndex": action_idx,
                }
            finally:
                try:
                    self.bridge.call("delete_game", {"gameId": gid})
                except Exception:
                    pass


def make_handler(server: PolicyServer):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # CORS preflight for browser fetch
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.end_headers()

        def do_POST(self):
            if self.path != "/action":
                self._send_json(404, {"error": f"unknown path {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                payload = json.loads(body)
                state = payload["state"]
                player_index = int(payload.get("playerIndex", 0))
                result = server.pick_action(state, player_index)
                self._send_json(200, result)
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        def log_message(self, fmt, *args):
            sys.stderr.write("[policy-server] " + (fmt % args) + "\n")

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not pathlib.Path(args.model).exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading {args.model}...")
    server = PolicyServer(args.model)
    httpd = HTTPServer((args.host, args.port), make_handler(server))
    print(f"Serving on http://{args.host}:{args.port}/action")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.bridge.close()


if __name__ == "__main__":
    main()
