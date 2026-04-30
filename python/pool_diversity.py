"""
Diversity audit for the policy pool.

Plays a handful of self-play games to gather a fixed observation sample,
then runs every pool member over that sample and reports pairwise mean KL
divergence between their action distributions. Surfaces near-duplicate
checkpoints that contribute nothing new to league play.

Optionally prunes pool members whose minimum pairwise KL falls below a
threshold (keeping the higher-Elo one when persistent ratings are present).

Usage:
    python python/pool_diversity.py [--sample-size 200] [--prune-threshold 0]
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
from tigphrates_env import BridgeProcess
from train import PolicyValueNetwork, obs_to_tensors, DEVICE
from policy_server import _build_policy_obs
from evaluate import _load_elo_table, ELO_BASELINE

POOL_DIR = pathlib.Path("models/pool")


def gather_sample(bridge: BridgeProcess, n_obs: int) -> list[dict]:
    """Self-play simpleAI games until n_obs (obs, mask) pairs are captured.
    Returns a list of dicts shaped for the policy network."""
    samples: list[dict] = []
    while len(samples) < n_obs:
        r = bridge.call("create", {"playerCount": 2})
        gid = r["gameId"]
        try:
            safety = 0
            while safety < 5000 and len(samples) < n_obs:
                safety += 1
                va = bridge.call("valid_actions", {"gameId": gid})
                if va["turnPhase"] == "gameOver":
                    break
                active = va["activePlayer"]
                obs_raw = bridge.call("get_observation", {
                    "gameId": gid, "playerIndex": active,
                })
                samples.append({
                    "obs": _build_policy_obs(obs_raw),
                    "mask": np.array(va["mask"], dtype=np.int8),
                })
                ai = bridge.call("ai_action", {"gameId": gid})
                if ai["actionIndex"] < 0:
                    break
                bridge.call("step_action", {
                    "gameId": gid,
                    "action": ai["action"],
                    "playerIndex": active,
                })
        finally:
            try:
                bridge.call("delete_game", {"gameId": gid})
            except Exception:
                pass
    return samples


def compute_action_probs(model: torch.nn.Module, samples: list[dict]) -> np.ndarray:
    """Run model over all samples, return shape (n_samples, n_actions) of
    masked softmax probabilities."""
    n = len(samples)
    probs = []
    for s in samples:
        obs_t = obs_to_tensors(s["obs"])
        mask = torch.tensor(s["mask"], dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model.forward(obs_t)
            logits = logits.masked_fill(~mask, float("-inf"))
            p = F.softmax(logits, dim=-1)
        probs.append(p.squeeze(0).cpu().numpy())
    return np.stack(probs, axis=0)


def mean_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Mean KL(p || q) over batch dim. Both inputs are (n, A) probabilities."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q)), axis=-1).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--prune-threshold", type=float, default=None,
                        help="If set, delete pool members whose min pairwise "
                             "KL falls below this threshold. Skips the "
                             "highest-Elo member of each redundant pair.")
    args = parser.parse_args()

    paths = sorted(POOL_DIR.glob("policy_*.pt"))
    if len(paths) < 2:
        print("Pool has <2 members; nothing to compare.")
        sys.exit(2)

    print(f"Pool: {len(paths)} members. Sampling {args.sample_size} obs from self-play...")
    bridge = BridgeProcess()
    try:
        samples = gather_sample(bridge, args.sample_size)
    finally:
        bridge.close()
    print(f"Got {len(samples)} sample obs. Running each model over the sample...")

    probs_per: dict[str, np.ndarray] = {}
    for p in paths:
        m = PolicyValueNetwork()
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.train(False)
        m.to(DEVICE)
        probs_per[p.stem] = compute_action_probs(m, samples)

    names = list(probs_per.keys())
    n = len(names)
    kl = np.zeros((n, n), dtype=np.float32)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                continue
            # Symmetric KL — easier to read pairwise.
            kl[i, j] = 0.5 * (
                mean_kl(probs_per[ni], probs_per[nj])
                + mean_kl(probs_per[nj], probs_per[ni])
            )

    print("\n=== Pairwise mean symmetric KL (lower = more similar) ===")
    print(f"{'':32s}" + " ".join(f"{nm[:10]:>10s}" for nm in names))
    for i, ni in enumerate(names):
        row = " ".join(f"{kl[i, j]:10.3f}" if i != j else f"{'-':>10s}"
                       for j in range(n))
        print(f"{ni:32s} {row}")

    # Per-member minimum KL: lower = more redundant.
    min_kl = np.where(np.eye(n, dtype=bool), np.inf, kl).min(axis=1)
    elo_table = _load_elo_table(POOL_DIR)
    print("\n=== Redundancy ranking (low min KL = duplicate of someone) ===")
    ranked = sorted(zip(names, min_kl), key=lambda x: x[1])
    for nm, mk in ranked:
        rating = elo_table.get(nm, ELO_BASELINE)
        print(f"  {nm:32s}  min_kl={mk:7.3f}  elo={rating:7.1f}")

    if args.prune_threshold is not None:
        print(f"\nPruning members with min_kl < {args.prune_threshold} (keeping higher Elo)...")
        deleted: list[str] = []
        survivors = set(names)
        for i, ni in enumerate(names):
            if ni not in survivors:
                continue
            for j, nj in enumerate(names):
                if i == j or nj not in survivors:
                    continue
                if kl[i, j] >= args.prune_threshold:
                    continue
                # Both close — drop the lower-Elo one.
                rating_i = elo_table.get(ni, ELO_BASELINE)
                rating_j = elo_table.get(nj, ELO_BASELINE)
                drop = nj if rating_i >= rating_j else ni
                survivors.discard(drop)
                drop_path = POOL_DIR / f"{drop}.pt"
                if drop_path.exists():
                    drop_path.unlink()
                    deleted.append(drop)
                if drop == ni:
                    break  # i is gone; advance outer loop
        print(f"Deleted {len(deleted)}: {deleted}")


if __name__ == "__main__":
    main()
