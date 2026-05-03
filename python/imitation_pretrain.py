"""
Behavior-cloning warm start from the heuristic AI (simpleAI).

Plays N games where every player is the heuristic, records the heuristic's
(obs, action_index, mask) tuples, then trains a PolicyValueNetwork to predict
the heuristic's actions via masked cross-entropy. Saves the result to
models/policy_bc.pt — the autoresearch loop in train.py can optionally pick
this up as a warm start instead of random init.

The heuristic is now competent (Phase 1), so BC over its decisions gives a
much stronger starting policy than random and short-circuits the early
sparse-reward learning phase.

Usage:
    python python/imitation_pretrain.py [--games 50] [--epochs 5] \
        [--batch-size 256] [--out models/policy_bc.pt]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tigphrates_env import BridgeProcess
from train import (
    PolicyValueNetwork, obs_to_tensors, stack_obs, DEVICE,
    FLAT_TO_TYPE, FLAT_TO_PARAM, TYPE_BASES_T, _adapt_state_dict,
)
from policy_server import _build_policy_obs


def collect_traces(
    bridge: BridgeProcess,
    n_games: int,
    expert_model: torch.nn.Module | None = None,
    max_steps_per_game: int = 5000,
    player_count: int = 2,
) -> list[dict]:
    """Run n_games self-play games and record (obs, action, mask) transitions.

    If `expert_model` is None, every player uses the heuristic AI (simpleAI)
    via the bridge's `ai_action` RPC. Otherwise, every player uses the given
    model — set this to `models/policy_best.pt` once the trained policy
    surpasses the heuristic so future BC pretrains start from an even
    stronger demonstrator (self-imitation)."""
    transitions: list[dict] = []
    for g in range(n_games):
        r = bridge.call("create", {"playerCount": player_count})
        gid = r["gameId"]
        safety = 0
        try:
            while safety < max_steps_per_game:
                safety += 1
                va = bridge.call("valid_actions", {"gameId": gid})
                if va["turnPhase"] == "gameOver":
                    break
                active = va["activePlayer"]
                obs_raw = bridge.call("get_observation", {
                    "gameId": gid, "playerIndex": active,
                })
                if expert_model is None:
                    ai = bridge.call("ai_action", {"gameId": gid})
                    if ai["actionIndex"] < 0:
                        break
                    action_index = int(ai["actionIndex"])
                    action_obj = ai["action"]
                else:
                    obs = _build_policy_obs(obs_raw)
                    mask = np.array(va["mask"], dtype=np.int8)
                    obs_t = obs_to_tensors(obs)
                    with torch.no_grad():
                        type_logits, param_logits, _ = expert_model.forward(obs_t)
                        type_dist, param_padded, _ = expert_model.hierarchical_dists(
                            type_logits, param_logits, mask
                        )
                        # Hierarchical argmax: pick best type, then best param
                        # within that type. Same decoding the browser ONNX
                        # path uses so behavior matches across surfaces.
                        type_idx = type_dist.logits.argmax(dim=-1)
                        chosen_logits = param_padded[
                            torch.arange(1, device=type_idx.device), type_idx
                        ]
                        param_idx = chosen_logits.argmax(dim=-1)
                        action_index = int(
                            (TYPE_BASES_T.to(type_idx.device)[type_idx] + param_idx).item()
                        )
                    decoded = bridge.call("decode_action", {
                        "gameId": gid, "actionIndex": action_index,
                    })
                    action_obj = decoded["action"]
                transitions.append({
                    "obs": _build_policy_obs(obs_raw),
                    "action": action_index,
                    "mask": np.array(va["mask"], dtype=np.int8),
                })
                bridge.call("step_action", {
                    "gameId": gid,
                    "action": action_obj,
                    "playerIndex": active,
                })
        finally:
            try:
                bridge.call("delete_game", {"gameId": gid})
            except Exception:
                pass
        if (g + 1) % 5 == 0:
            print(f"  collected {len(transitions)} transitions across {g + 1} games", flush=True)
    return transitions


def train_bc(
    transitions: list[dict],
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 3e-4,
    value_coef: float = 0.5,
) -> PolicyValueNetwork:
    """Train a PolicyValueNetwork to imitate the heuristic.

    The policy head is supervised against the heuristic's chosen action via
    masked cross-entropy. The value head is trained to predict a small
    constant target (zero) so it has reasonable initial scale before PPO
    finetuning takes over.
    """
    model = PolicyValueNetwork()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(transitions)
    if n == 0:
        return model

    print(f"BC training on {n} transitions, {epochs} epochs, batch={batch_size}, device={DEVICE}")
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = idx[start:end]
            batch = [transitions[i] for i in batch_idx]

            obs_batch = stack_obs([t["obs"] for t in batch])
            actions = torch.tensor([t["action"] for t in batch], dtype=torch.long, device=DEVICE)
            masks = np.stack([t["mask"] for t in batch], axis=0)

            type_logits, param_logits, values = model.forward(obs_batch)
            type_dist, param_padded, _type_mask = model.hierarchical_dists(
                type_logits, param_logits, masks
            )
            B = type_logits.shape[0]
            device = type_logits.device

            # Decompose target action into (type, param) and CE each head.
            flat_to_type = FLAT_TO_TYPE.to(device)
            flat_to_param = FLAT_TO_PARAM.to(device)
            target_type = flat_to_type[actions]
            target_param = flat_to_param[actions]

            # type_dist.logits is already log-softmax (Categorical normalizes).
            type_ce = F.nll_loss(type_dist.logits, target_type)

            chosen_param_logits = param_padded[torch.arange(B, device=device), target_type]
            param_log_probs = F.log_softmax(chosen_param_logits, dim=-1)
            param_ce = F.nll_loss(param_log_probs, target_param)
            ce_loss = type_ce + param_ce

            # Predicted accuracy: hierarchical argmax matches the heuristic's
            # exact (type, param). Stricter than the old flat argmax, since a
            # mismatch on either head counts as wrong.
            pred_type = type_dist.logits.argmax(dim=-1)
            pred_chosen = param_padded[torch.arange(B, device=device), pred_type]
            pred_param = pred_chosen.argmax(dim=-1)
            acc = ((pred_type == target_type) & (pred_param == target_param)).float().mean().item()

            value_target = torch.zeros_like(values)
            v_loss = F.mse_loss(values, value_target)

            loss = ce_loss + value_coef * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += ce_loss.item()
            total_acc += acc
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)
        print(f"  epoch {epoch + 1}/{epochs}: ce_loss={avg_loss:.4f} acc={avg_acc:.3f}", flush=True)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="models/policy_bc.pt")
    parser.add_argument("--expert", default=None,
                        help="Path to a trained policy checkpoint to use as "
                             "demonstrator instead of the heuristic. Use this "
                             "for self-imitation once the trained model beats "
                             "simpleAI.")
    parser.add_argument("--player-count", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    expert_model = None
    if args.expert:
        expert_path = pathlib.Path(args.expert)
        if not expert_path.exists():
            print(f"--expert path not found: {expert_path}", file=sys.stderr)
            sys.exit(2)
        expert_model = PolicyValueNetwork()
        raw = torch.load(expert_path, map_location="cpu")
        expert_model.load_state_dict(_adapt_state_dict(raw), strict=False)
        expert_model.train(False)
        expert_model.to(DEVICE)
        print(f"Using {expert_path} as demonstrator (self-imitation)")

    src = "trained policy" if expert_model is not None else "heuristic"
    print(f"Collecting {src} traces from {args.games} games...")
    bridge = BridgeProcess()
    try:
        t0 = time.time()
        transitions = collect_traces(
            bridge,
            n_games=args.games,
            expert_model=expert_model,
            player_count=args.player_count,
        )
        t1 = time.time()
        print(f"Collected {len(transitions)} transitions in {t1 - t0:.1f}s")
    finally:
        bridge.close()

    if not transitions:
        print("No transitions collected; aborting.", file=sys.stderr)
        sys.exit(1)

    t2 = time.time()
    model = train_bc(
        transitions,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    t3 = time.time()
    print(f"BC training took {t3 - t2:.1f}s")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
