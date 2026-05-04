"""
Policy + Lab HTTP server. One process backs three flows:

POST /action
    Pick an action with the loaded default policy. Used by the
    "Trained (server)" option in the browser. Body = {state, playerIndex}.

GET /leaderboard
    Returns the contents of <pool>/elo.json plus per-model metadata
    (filename, size_bytes, mtime, has_onnx). Used by the Lab leaderboard.

GET /pool/<name>.onnx
    Streams an ONNX export of <pool>/<name>.pt. Lazily exports if the
    .onnx file is missing or older than the .pt source. Used by the Lab's
    "watch model A vs B" flow — the browser fetches per-seat models from
    here.

POST /train
    Body = {time_budget?, env?}. Spawns python/train.py as a subprocess
    with the given env overrides; returns {job_id}. Concurrency is capped
    at 1 active training job (Mac mini constraint). The training run
    writes new pool snapshots that the leaderboard endpoint surfaces on
    its next refresh.

GET /train/<job_id>
    Returns {status, log, exit_code?}. status ∈ {running, done, failed}.
    The log is the tail of train.py's stdout/stderr.

Usage:
    python python/policy_server.py [--model models/policy_best.pt] [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
import pathlib
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tigphrates_env import BridgeProcess
from train import PolicyValueNetwork, obs_to_tensors, DEVICE, _adapt_state_dict

DEFAULT_PORT = 8765
DEFAULT_MODEL_PATH = "models/policy_best.pt"
DEFAULT_POOL_DIR = pathlib.Path("models/pool")
DEFAULT_GAMES_DIR = pathlib.Path("models/games")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG_TAIL = 400  # lines of train.py output retained per job


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

    def __init__(self, model_path: str, pool_dir: pathlib.Path, games_dir: pathlib.Path):
        self.bridge = BridgeProcess()
        self.model = PolicyValueNetwork()
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(_adapt_state_dict(state_dict), strict=False)
        self.model.train(False)
        self.model.to(DEVICE)
        self._lock = threading.Lock()

        self.pool_dir = pool_dir
        self.games_dir = games_dir
        # Single concurrent training job at a time — Mac mini can't handle two
        # 4-env training runs simultaneously without thrashing.
        self._jobs: dict[str, dict] = {}
        self._job_lock = threading.Lock()
        # Per-pool-member ONNX export lock so concurrent /pool/<x>.onnx
        # requests don't double-export.
        self._export_lock = threading.Lock()

    def pick_action(self, state: dict, player_index: int) -> dict:
        with self._lock:
            loaded = self.bridge.call("load_state", {"state": state})
            gid = loaded["gameId"]
            try:
                va = self.bridge.call("valid_actions", {"gameId": gid})
                mask = np.array(va["mask"], dtype=np.int8)
                if mask.sum() == 0:
                    return {"action": {"type": "pass"}, "label": "pass", "actionIndex": -1}

                # Optional AlphaZero-style MCTS at inference. Gated on
                # MCTS_SIMULATIONS so the default behaviour (direct policy)
                # is bit-identical for SIMULATIONS=0.
                num_sims = int(os.environ.get("MCTS_SIMULATIONS", "0"))
                if num_sims > 0:
                    from mcts import MCTS, build_default_evaluator  # local import
                    c_puct = float(os.environ.get("MCTS_C_PUCT", "1.5"))
                    evaluator = build_default_evaluator(self.model)
                    mcts = MCTS(
                        model=evaluator,
                        bridge=self.bridge,
                        num_simulations=num_sims,
                        c_puct=c_puct,
                    )
                    out = mcts.pick_action(game_id=gid, player_index=player_index)
                    return {
                        "action": out["action"],
                        "label": out.get("label", ""),
                        "actionIndex": int(out["actionIndex"]),
                    }

                # Direct-policy path (default, MCTS_SIMULATIONS=0).
                obs_raw = self.bridge.call("get_observation", {
                    "gameId": gid, "playerIndex": player_index,
                })
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

    # --- Lab: leaderboard ---

    def leaderboard(self) -> dict:
        elo_path = self.pool_dir / "elo.json"
        elo: dict[str, float] = {}
        if elo_path.exists():
            try:
                elo = json.loads(elo_path.read_text())
            except Exception:
                elo = {}
        members = []
        if self.pool_dir.exists():
            for p in sorted(self.pool_dir.glob("policy_*.pt")):
                onnx_path = p.with_suffix(".onnx")
                # elo.json uses the full filename ("policy_init.pt") as the
                # key — see evaluate.update_persistent_elo. Fall back to
                # the stem in case an older sidecar used the bare name.
                rating = elo.get(p.name)
                if rating is None:
                    rating = elo.get(p.stem)
                members.append({
                    "name": p.stem,
                    "path": p.name,
                    "size_bytes": p.stat().st_size,
                    "mtime": p.stat().st_mtime,
                    "elo": rating,
                    "has_onnx": onnx_path.exists(),
                })
        members.sort(key=lambda m: -(m.get("elo") or 0.0))
        return {
            "pool_dir": str(self.pool_dir),
            "agent_elo": elo.get("_agent"),
            "members": members,
        }

    # --- Lab: lazy ONNX export ---

    def export_onnx_for(self, name: str) -> pathlib.Path:
        """Ensure <pool>/<name>.onnx exists and is fresher than <name>.pt;
        export if not. Returns the resolved .onnx path."""
        pt_path = self.pool_dir / f"{name}.pt"
        if not pt_path.exists() or not pt_path.is_file():
            raise FileNotFoundError(f"unknown pool member: {name}")
        onnx_path = pt_path.with_suffix(".onnx")
        with self._export_lock:
            if onnx_path.exists() and onnx_path.stat().st_mtime >= pt_path.stat().st_mtime:
                return onnx_path
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "python" / "export_onnx.py"),
                "--model", str(pt_path),
                "--out", str(onnx_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            if res.returncode != 0:
                raise RuntimeError(f"export_onnx failed: {res.stderr.strip() or res.stdout.strip()}")
            return onnx_path

    # --- Lab: training jobs ---

    def start_training(self, time_budget: int, extra_env: dict[str, str] | None = None) -> str:
        with self._job_lock:
            for job in self._jobs.values():
                if job["status"] == "running":
                    raise RuntimeError(
                        "another training job is already running; only one is allowed at a time"
                    )
            job_id = uuid.uuid4().hex[:8]
            env = os.environ.copy()
            env["TIME_BUDGET"] = str(time_budget)
            if extra_env:
                for k, v in extra_env.items():
                    env[str(k)] = str(v)
            log: deque[str] = deque(maxlen=LOG_TAIL)
            job = {
                "status": "running",
                "started_at": time.time(),
                "exit_code": None,
                "log": log,
                "time_budget": time_budget,
            }
            self._jobs[job_id] = job

        def _runner():
            cmd = [sys.executable, str(PROJECT_ROOT / "python" / "train.py")]
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    log.append(line.rstrip())
                proc.wait()
                with self._job_lock:
                    job["status"] = "done" if proc.returncode == 0 else "failed"
                    job["exit_code"] = proc.returncode
                    job["finished_at"] = time.time()
            except Exception as e:
                with self._job_lock:
                    job["status"] = "failed"
                    job["log"].append(f"[server] runner error: {e}")
                    job["finished_at"] = time.time()

        threading.Thread(target=_runner, daemon=True).start()
        return job_id

    def get_job(self, job_id: str) -> dict:
        with self._job_lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {
                "status": job["status"],
                "exit_code": job["exit_code"],
                "started_at": job["started_at"],
                "finished_at": job.get("finished_at"),
                "time_budget": job["time_budget"],
                "log": list(job["log"]),
            }

    # --- Lab: game-log archive ---

    def save_game_log(self, log_lines: list[str], meta: dict | None = None) -> dict:
        """Persist a finished-game log to <games_dir>/<ts>-<id>.log and
        update <games_dir>/latest.log. Returns {path, latest_path}."""
        self.games_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        gid = uuid.uuid4().hex[:6]
        path = self.games_dir / f"{ts}-{gid}.log"
        latest = self.games_dir / "latest.log"
        body_lines: list[str] = []
        if meta:
            body_lines.append(f"# meta: {json.dumps(meta, sort_keys=True)}")
        body_lines.extend(log_lines)
        body = "\n".join(body_lines) + "\n"
        path.write_text(body)
        latest.write_text(body)
        return {"path": str(path), "latest": str(latest)}

    def list_jobs(self) -> list[dict]:
        with self._job_lock:
            return [
                {
                    "job_id": jid,
                    "status": j["status"],
                    "started_at": j["started_at"],
                    "finished_at": j.get("finished_at"),
                    "exit_code": j["exit_code"],
                    "time_budget": j["time_budget"],
                }
                for jid, j in self._jobs.items()
            ]


def make_handler(server: PolicyServer):
    class Handler(BaseHTTPRequestHandler):
        def _set_cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send_json(self, status: int, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self._set_cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, status: int, content_type: str, body: bytes):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self._set_cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self._set_cors()
            self.end_headers()

        def do_GET(self):
            try:
                if self.path == "/leaderboard":
                    return self._send_json(200, server.leaderboard())
                if self.path == "/jobs":
                    return self._send_json(200, {"jobs": server.list_jobs()})
                if self.path.startswith("/train/"):
                    job_id = self.path[len("/train/"):]
                    try:
                        return self._send_json(200, server.get_job(job_id))
                    except KeyError:
                        return self._send_json(404, {"error": f"unknown job {job_id}"})
                if self.path.startswith("/pool/") and self.path.endswith(".onnx"):
                    name = self.path[len("/pool/"):-len(".onnx")]
                    if "/" in name or ".." in name:
                        return self._send_json(400, {"error": "invalid pool name"})
                    try:
                        onnx_path = server.export_onnx_for(name)
                    except FileNotFoundError:
                        return self._send_json(404, {"error": f"unknown pool member {name}"})
                    except Exception as e:
                        return self._send_json(500, {"error": str(e)})
                    body = onnx_path.read_bytes()
                    return self._send_bytes(200, "application/octet-stream", body)
                return self._send_json(404, {"error": f"unknown path {self.path}"})
            except Exception as e:
                return self._send_json(500, {"error": str(e)})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(raw or "{}")
            except Exception as e:
                return self._send_json(400, {"error": f"bad JSON: {e}"})

            if self.path == "/action":
                try:
                    state = payload["state"]
                    player_index = int(payload.get("playerIndex", 0))
                    result = server.pick_action(state, player_index)
                    return self._send_json(200, result)
                except Exception as e:
                    return self._send_json(500, {"error": str(e)})

            if self.path == "/games":
                try:
                    log = payload.get("log") or []
                    if not isinstance(log, list) or not all(isinstance(s, str) for s in log):
                        return self._send_json(400, {"error": "log must be a list[str]"})
                    meta = payload.get("meta") or None
                    res = server.save_game_log(log, meta)
                    return self._send_json(200, res)
                except Exception as e:
                    return self._send_json(500, {"error": str(e)})

            if self.path == "/train":
                try:
                    time_budget = int(payload.get("time_budget", 300))
                    if time_budget < 10 or time_budget > 7200:
                        return self._send_json(
                            400, {"error": "time_budget must be 10..7200 seconds"}
                        )
                    extra_env = payload.get("env") or {}
                    job_id = server.start_training(time_budget, extra_env)
                    return self._send_json(200, {"job_id": job_id})
                except RuntimeError as e:
                    return self._send_json(409, {"error": str(e)})
                except Exception as e:
                    return self._send_json(500, {"error": str(e)})

            return self._send_json(404, {"error": f"unknown path {self.path}"})

        def log_message(self, fmt, *args):
            sys.stderr.write("[policy-server] " + (fmt % args) + "\n")

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--pool-dir", default=str(DEFAULT_POOL_DIR))
    parser.add_argument("--games-dir", default=str(DEFAULT_GAMES_DIR),
                        help="Directory where finished-game logs land (POST /games).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind interface. Default 0.0.0.0 so Tailscale / "
                             "LAN clients can reach the lab. Use 127.0.0.1 to "
                             "restrict to localhost.")
    args = parser.parse_args()

    if not pathlib.Path(args.model).exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(2)

    print(f"Loading {args.model}...")
    server = PolicyServer(
        args.model, pathlib.Path(args.pool_dir), pathlib.Path(args.games_dir),
    )
    httpd = HTTPServer((args.host, args.port), make_handler(server))
    print(f"Serving on http://{args.host}:{args.port}")
    print(f"  POST /action                 — pick action with default model")
    print(f"  GET  /leaderboard            — pool members + persistent Elo")
    print(f"  GET  /pool/<name>.onnx       — lazily-exported per-snapshot ONNX")
    print(f"  POST /train                  — start a training run (time_budget seconds)")
    print(f"  GET  /train/<job_id>         — training status + log tail")
    print(f"  POST /games                  — archive a finished game log to {args.games_dir}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.bridge.close()


if __name__ == "__main__":
    main()
