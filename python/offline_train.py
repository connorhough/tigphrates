"""Supervised policy/value training from rollout-improved offline labels.

This is intentionally separate from PPO. It trains the same PolicyValueNetwork
on dense soft targets produced by offline_dataset.py, giving the model direct
examples of actions that survive short search and heuristic playouts.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tigphrates_env import ACTION_SPACE_SIZE
from train import (
    DEVICE,
    TYPE_BASES,
    TYPE_PARAM_SIZES,
    PolicyValueNetwork,
    _adapt_state_dict,
)


OBS_KEYS = (
    "board",
    "hand",
    "hand_seq",
    "scores",
    "meta",
    "conflict",
    "leaders",
    "opp_scores",
    "opp_leaders",
)


def load_checkpoint(path: str | None) -> PolicyValueNetwork:
    model = PolicyValueNetwork()
    if path:
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(_adapt_state_dict(state_dict), strict=False)
    model.to(DEVICE)
    return model


def batch_to_tensors(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, torch.Tensor]:
    obs = {}
    for key in OBS_KEYS:
        dtype = torch.int64 if key == "hand_seq" else torch.float32
        obs[key] = torch.tensor(data[key][idx], dtype=dtype, device=DEVICE)
    return obs


def flat_log_probs(model: PolicyValueNetwork, obs: dict[str, torch.Tensor], mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    type_logits, param_logits, value = model.forward(obs)
    type_dist, param_padded, _type_mask = model.hierarchical_dists(type_logits, param_logits, mask)
    type_logp = F.log_softmax(type_dist.logits, dim=-1)
    param_logp = F.log_softmax(param_padded, dim=-1)
    param_logp = torch.nan_to_num(param_logp, nan=-1e9, neginf=-1e9, posinf=0.0)

    bsz = type_logits.shape[0]
    flat = torch.full((bsz, ACTION_SPACE_SIZE), -1e9, dtype=torch.float32, device=DEVICE)
    for t, base in enumerate(TYPE_BASES):
        size = TYPE_PARAM_SIZES[t]
        flat[:, base : base + size] = type_logp[:, t : t + 1] + param_logp[:, t, :size]
    flat = torch.nan_to_num(flat, nan=-1e9, neginf=-1e9, posinf=0.0)
    return flat, value.view(-1)


def split_indices(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = max(1, int(round(n * val_frac))) if n > 1 else 0
    return idx[val_n:], idx[:val_n]


def evaluate_loss(
    model: PolicyValueNetwork,
    data: dict[str, np.ndarray],
    idx: np.ndarray,
    batch_size: int,
    value_coef: float,
) -> dict[str, float]:
    if len(idx) == 0:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "top1": 0.0}
    model.train(False)
    losses = []
    policy_losses = []
    value_losses = []
    top1_hits = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            bi = idx[start : start + batch_size]
            obs = batch_to_tensors(data, bi)
            mask = torch.tensor(data["mask"][bi], dtype=torch.bool, device=DEVICE)
            target_policy = torch.tensor(data["target_policy"][bi], dtype=torch.float32, device=DEVICE)
            target_value = torch.tensor(data["target_value"][bi], dtype=torch.float32, device=DEVICE)
            target_action = torch.tensor(data["target_action"][bi], dtype=torch.long, device=DEVICE)
            logp, value = flat_log_probs(model, obs, mask)
            policy_loss = -(target_policy * logp).sum(dim=-1).mean()
            value_loss = F.mse_loss(value, target_value)
            loss = policy_loss + value_coef * value_loss
            pred = logp.argmax(dim=-1)
            top1 = (pred == target_action).float().mean()
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            top1_hits.append(float(top1.item()))
    return {
        "loss": float(np.mean(losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "top1": float(np.mean(top1_hits)),
    }


def train(args: argparse.Namespace) -> dict[str, float]:
    data = dict(np.load(args.data))
    n = int(data["target_action"].shape[0])
    train_idx, val_idx = split_indices(n, args.val_frac, args.seed)

    model = load_checkpoint(args.init)
    model.train(True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    best_val = float("inf")
    best_state = None
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        if len(train_idx) == 0:
            batch_idx = val_idx
        else:
            batch_idx = rng.choice(train_idx, size=min(args.batch_size, len(train_idx)), replace=len(train_idx) < args.batch_size)
        obs = batch_to_tensors(data, batch_idx)
        mask = torch.tensor(data["mask"][batch_idx], dtype=torch.bool, device=DEVICE)
        target_policy = torch.tensor(data["target_policy"][batch_idx], dtype=torch.float32, device=DEVICE)
        target_value = torch.tensor(data["target_value"][batch_idx], dtype=torch.float32, device=DEVICE)
        target_action = torch.tensor(data["target_action"][batch_idx], dtype=torch.long, device=DEVICE)

        logp, value = flat_log_probs(model, obs, mask)
        policy_loss = -(target_policy * logp).sum(dim=-1).mean()
        hard_loss = F.nll_loss(logp, target_action)
        value_loss = F.mse_loss(value, target_value)
        loss = policy_loss + args.hard_coef * hard_loss + args.value_coef * value_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_loss(model, data, val_idx if len(val_idx) else train_idx, args.batch_size, args.value_coef)
            print(
                f"step={step} loss={float(loss.item()):.4f} "
                f"val_policy={metrics['policy_loss']:.4f} val_value={metrics['value_loss']:.4f} "
                f"val_top1={metrics['top1']:.3f}",
                flush=True,
            )
            if metrics["loss"] < best_val:
                best_val = metrics["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, args.out)

    if best_state is None:
        torch.save(model.state_dict(), args.out)
    else:
        model.load_state_dict(best_state)

    final = evaluate_loss(model, data, val_idx if len(val_idx) else train_idx, args.batch_size, args.value_coef)
    final["states"] = float(n)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/offline_rollouts.npz")
    parser.add_argument("--init", default="models/policy_best.pt")
    parser.add_argument("--out", default="models/offline_policy.pt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-coef", type=float, default=0.35)
    parser.add_argument("--hard-coef", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train(args)
    print(
        f"wrote {args.out} states={int(metrics['states'])} "
        f"val_policy={metrics['policy_loss']:.4f} val_value={metrics['value_loss']:.4f} "
        f"val_top1={metrics['top1']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
