"""
Behavior-cloning pretraining from the heuristic AI (simpleAI).

Two-stage philosophy (AlphaGo / AlphaStar playbook):
  1. Pretrain the policy via supervised cross-entropy against the heuristic
     AI on every legal-action decision until top-1 match converges (target
     >= 50% — AlphaGo's SL net got 57% top-1 on KGS data).
  2. *Then* PPO fine-tunes from the BC checkpoint. The BC auxiliary loss
     during PPO is no longer needed (`BC_AUX_DISABLED=1`).

This file is structured for testability:
  - `bc_train_step(model, optimizer, batch) -> dict`
        Pure-ish function. Takes a batch dict {obs, action_mask, target_action},
        runs one optimizer step, returns {"loss": float, "top1_acc": float}.
        No bridge access — exercisable from unit tests on synthetic data.
  - `bc_train(model, data_iterator, num_steps, eval_fn, eval_every, ...)`
        High-level training loop driver. Yields per-step metrics.
  - `collect_traces(...)` — bridge-driven heuristic rollout collection
        (kept compatible with the old API).
  - `train_bc(...)` — backwards-compatible wrapper over the new primitives,
        used by the legacy multi-epoch path.

Usage:
    python python/imitation_pretrain.py --steps 10000 --out models/policy_init.pt \\
        --eval-every 1000 [--games 50 --batch-size 256 --player-count 2]

On exit, prints final top-1-vs-heuristic match and writes a bare-state-dict
checkpoint compatible with `train.py`'s `_adapt_state_dict` loader.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import random
import sys
import time
from typing import Callable, Iterator

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


# ---------------------------------------------------------------------------
# Core BC primitives — testable without the bridge
# ---------------------------------------------------------------------------

def bc_train_step(
    model: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    value_coef: float = 0.0,
    grad_clip: float | None = 0.5,
) -> dict:
    """One BC gradient step on a batch.

    Args:
        model: PolicyValueNetwork.
        optimizer: torch optimizer with model.parameters().
        batch: dict with
            - "obs": dict-of-arrays or dict-of-tensors (B-batched)
            - "action_mask": (B, ACTION_SPACE_SIZE) bool/int mask
            - "target_action": (B,) int64 — heuristic's chosen action index
        value_coef: weight on a zero-target value MSE (default 0 — pure BC).
        grad_clip: clip-grad-norm value, or None to skip.

    Returns:
        {"loss": float (cross-entropy only), "top1_acc": float}.

    Loss is masked-aware via model.hierarchical_dists — masked-out logits
    contribute -inf and don't move during backprop.
    """
    obs = batch["obs"]
    masks = batch["action_mask"]
    target_action = batch["target_action"]

    obs_batched = _ensure_batched_obs(obs)
    actions = torch.as_tensor(target_action, dtype=torch.long, device=DEVICE)
    if isinstance(masks, np.ndarray):
        masks_np = masks
    else:
        masks_np = np.asarray(masks)

    type_logits, param_logits, values = model.forward(obs_batched)
    type_dist, param_padded, _type_mask = model.hierarchical_dists(
        type_logits, param_logits, masks_np
    )
    B = type_logits.shape[0]
    device = type_logits.device

    flat_to_type = FLAT_TO_TYPE.to(device)
    flat_to_param = FLAT_TO_PARAM.to(device)
    target_type = flat_to_type[actions]
    target_param = flat_to_param[actions]

    type_ce = F.nll_loss(type_dist.logits, target_type)
    chosen_param_logits = param_padded[torch.arange(B, device=device), target_type]
    param_log_probs = F.log_softmax(chosen_param_logits, dim=-1)
    param_ce = F.nll_loss(param_log_probs, target_param)
    ce_loss = type_ce + param_ce

    pred_type = type_dist.logits.argmax(dim=-1)
    pred_chosen = param_padded[torch.arange(B, device=device), pred_type]
    pred_param = pred_chosen.argmax(dim=-1)
    acc = ((pred_type == target_type) & (pred_param == target_param)).float().mean().item()

    if value_coef > 0:
        value_target = torch.zeros_like(values)
        v_loss = F.mse_loss(values, value_target)
        loss = ce_loss + value_coef * v_loss
    else:
        loss = ce_loss

    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {"loss": float(ce_loss.item()), "top1_acc": float(acc)}


def _ensure_batched_obs(obs):
    """Accept either a dict-of-numpy-arrays-already-batched, a dict-of-tensors,
    or a single-instance dict. Convert to batched tensors on DEVICE."""
    # Heuristic: if board has 4 dims, it's already batched.
    board = obs["board"]
    if hasattr(board, "shape") and len(board.shape) == 4:
        # Already batched — just convert to tensors on DEVICE.
        out = {}
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                out[key] = val.to(DEVICE) if val.device != DEVICE else val
            else:
                # Pick dtype based on key — board / float fields → float32,
                # hand_seq → int64.
                if key == "hand_seq":
                    out[key] = torch.as_tensor(val, dtype=torch.long, device=DEVICE)
                else:
                    out[key] = torch.as_tensor(val, dtype=torch.float32, device=DEVICE)
        return out
    # Single instance — fall back to obs_to_tensors.
    return obs_to_tensors(obs)


def bc_train(
    model: PolicyValueNetwork,
    data_iterator: Iterator[dict],
    num_steps: int,
    optimizer: torch.optim.Optimizer | None = None,
    lr: float = 3e-4,
    eval_fn: Callable[[PolicyValueNetwork], dict] | None = None,
    eval_every: int = 1000,
    log_every: int = 100,
    value_coef: float = 0.0,
) -> list[dict]:
    """Drive `num_steps` BC training iterations from `data_iterator`.

    Args:
        model: the policy network being trained.
        data_iterator: yields batch dicts (obs, action_mask, target_action).
        num_steps: total number of gradient steps.
        optimizer: optional; defaults to Adam(model.parameters(), lr=lr).
        eval_fn: optional callable(model) -> dict — invoked every eval_every
            steps. Its result is merged into the metrics history.
        eval_every: step interval for eval_fn.
        log_every: print interval.

    Returns:
        list of per-step (or per-log-period) metrics dicts.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict] = []
    running_loss = 0.0
    running_acc = 0.0
    n_logged = 0
    t0 = time.time()

    for step in range(1, num_steps + 1):
        try:
            batch = next(data_iterator)
        except StopIteration:
            print(f"Data iterator exhausted at step {step}")
            break

        metrics = bc_train_step(model, optimizer, batch, value_coef=value_coef)
        running_loss += metrics["loss"]
        running_acc += metrics["top1_acc"]
        n_logged += 1
        history.append({"step": step, **metrics})

        if step % log_every == 0:
            avg_loss = running_loss / max(n_logged, 1)
            avg_acc = running_acc / max(n_logged, 1)
            elapsed = time.time() - t0
            print(
                f"  step {step}/{num_steps}  loss={avg_loss:.4f}  "
                f"top1={avg_acc:.3f}  elapsed={elapsed:.1f}s",
                flush=True,
            )
            running_loss = 0.0
            running_acc = 0.0
            n_logged = 0

        if eval_fn is not None and step % eval_every == 0:
            eval_metrics = eval_fn(model)
            print(f"  [eval @ step {step}] {eval_metrics}", flush=True)
            history[-1].update({f"eval_{k}": v for k, v in eval_metrics.items()})

    return history


# ---------------------------------------------------------------------------
# Bridge-driven trace collection (heuristic-only)
# ---------------------------------------------------------------------------

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


def make_batch_iterator(
    transitions: list[dict],
    batch_size: int = 256,
    shuffle: bool = True,
) -> Iterator[dict]:
    """Cycle over transitions, yielding batches in the bc_train_step format.
    Reshuffles each pass — yields indefinitely so training can run for any
    `num_steps`."""
    n = len(transitions)
    if n == 0:
        return
    while True:
        idx = np.random.permutation(n) if shuffle else np.arange(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = idx[start:end]
            batch = [transitions[i] for i in batch_idx]
            obs_batch = stack_obs([t["obs"] for t in batch])
            actions = np.array([t["action"] for t in batch], dtype=np.int64)
            masks = np.stack([t["mask"] for t in batch], axis=0)
            # obs_batch is already on DEVICE — the bc_train_step's
            # _ensure_batched_obs will short-circuit since tensors keep their
            # device. masks/actions stay numpy until consumption.
            yield {
                "obs": obs_batch,
                "action_mask": masks,
                "target_action": actions,
            }


# ---------------------------------------------------------------------------
# Backwards-compatible epoch-style trainer
# ---------------------------------------------------------------------------

def train_bc(
    transitions: list[dict],
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 3e-4,
    value_coef: float = 0.5,
) -> PolicyValueNetwork:
    """Multi-epoch BC over a fixed transition set. Kept as a thin wrapper
    over the new primitives so existing callers don't break.

    The policy head is supervised against the heuristic's chosen action via
    masked cross-entropy. With `value_coef > 0`, the value head is trained
    to a zero target so it has reasonable initial scale before PPO finetunes.
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
            actions = np.array([t["action"] for t in batch], dtype=np.int64)
            masks = np.stack([t["mask"] for t in batch], axis=0)

            metrics = bc_train_step(
                model, optimizer,
                {"obs": obs_batch, "action_mask": masks, "target_action": actions},
                value_coef=value_coef,
            )
            total_loss += metrics["loss"]
            total_acc += metrics["top1_acc"]
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)
        print(f"  epoch {epoch + 1}/{epochs}: ce_loss={avg_loss:.4f} acc={avg_acc:.3f}", flush=True)

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Legacy "epochs over fixed-size dataset" knobs.
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=0,
                        help="If > 0, run epoch-mode (legacy). Otherwise use --steps.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", default="models/policy_init.pt")
    parser.add_argument("--expert", default=None,
                        help="Path to a trained policy checkpoint to use as "
                             "demonstrator instead of the heuristic. Use this "
                             "for self-imitation once the trained model beats "
                             "simpleAI.")
    parser.add_argument("--player-count", type=int, default=2)
    # New step-based driver.
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of BC gradient steps (used when --epochs=0).")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=5,
                        help="Heuristic-match eval games per checkpoint.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-coef", type=float, default=0.0,
                        help="Weight on zero-target value MSE during BC. "
                             "Default 0 — value head is trained from scratch by PPO.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.epochs > 0:
        # Legacy multi-epoch mode.
        t2 = time.time()
        model = train_bc(
            transitions,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            value_coef=args.value_coef,
        )
        t3 = time.time()
        print(f"BC training took {t3 - t2:.1f}s")
        torch.save(model.state_dict(), out_path)
        print(f"Saved {out_path}")
        return

    # Step-based driver — preferred.
    model = PolicyValueNetwork().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    iterator = make_batch_iterator(transitions, batch_size=args.batch_size)

    # Eval closure: heuristic-match top-1 over fresh games via the bridge.
    eval_bridge: list[BridgeProcess] = []  # lazy holder so we open at most once

    def eval_fn(m):
        # Lazy-open a fresh bridge for eval games (the trace bridge is closed).
        if not eval_bridge:
            eval_bridge.append(BridgeProcess())
        try:
            from evaluate import evaluate_top1_match_vs_heuristic
        except ImportError:
            return {}
        return evaluate_top1_match_vs_heuristic(
            m, num_games=args.eval_games, bridge=eval_bridge[0]
        )

    print(f"Step-based BC: {args.steps} steps, batch={args.batch_size}, lr={args.lr}")
    t2 = time.time()
    history = bc_train(
        model, iterator, num_steps=args.steps,
        optimizer=optimizer,
        eval_fn=eval_fn, eval_every=args.eval_every,
        log_every=max(50, args.steps // 50),
        value_coef=args.value_coef,
    )
    t3 = time.time()
    print(f"BC training took {t3 - t2:.1f}s")

    if eval_bridge:
        eval_bridge[0].close()

    torch.save(model.state_dict(), out_path)
    print(f"Saved {out_path}")
    if history:
        last = history[-1]
        print(f"Final step={last['step']} loss={last['loss']:.4f} top1={last['top1_acc']:.3f}")


if __name__ == "__main__":
    main()
