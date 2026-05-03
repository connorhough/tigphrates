"""
Hyperparameter sweep harness for python/train.py.

Designed for a Mac mini: sequential by default, optional --concurrency N
parallelism (each run spawns NUM_ENVS Node bridges, so keep N small —
recommended N <= cpu_count // 6).

Each run gets its own RUN_DIR with an isolated pool, so concurrent runs
don't stomp each other's checkpoints. After all runs complete, prints a
leaderboard sorted by vs_pool_win_rate.

Usage:
    python python/sweep.py [--concurrency 1] [--time-budget 60] [--out sweep.tsv]

Edit DEFAULT_GRID below to define the parameter grid for your sweep.
"""

from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


DEFAULT_GRID: dict[str, list[str]] = {
    # Phase 11.3 — reward sparsity ablation. Probes whether the dense
    # shaping (min-score delta + margin delta) is genuinely helping or just
    # adding noise. 6 cells × time_budget per cell.
    "SCORE_DELTA_COEF": ["0.0", "0.5", "1.5"],
    "MARGIN_DELTA_COEF": ["0.0", "1.0"],
}

# Metrics to extract from train.py stdout. Order matters: first is the
# primary sort key in the final leaderboard.
METRICS = ["vs_pool_win_rate", "win_rate", "avg_min_score", "persistent_elo", "elo"]


def _grid_combinations(grid: dict[str, list[str]]) -> list[dict[str, str]]:
    keys = list(grid.keys())
    combos = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, vals)))
    return combos


def _label_for(combo: dict[str, str]) -> str:
    parts = [f"{k.lower()}={v}" for k, v in sorted(combo.items())]
    return "_".join(parts).replace(".", "p")


def run_one(combo: dict[str, str], time_budget: int, eval_games: int, base_dir: pathlib.Path) -> dict:
    label = _label_for(combo)
    run_dir = base_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    env = os.environ.copy()
    env.update(combo)
    env["TIME_BUDGET"] = str(time_budget)
    env["EVAL_GAMES"] = str(eval_games)
    env["RUN_DIR"] = str(run_dir)
    env["POOL_DIR"] = str(run_dir / "pool")

    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(
            [sys.executable, "python/train.py"],
            stdout=f, stderr=subprocess.STDOUT,
            env=env,
            cwd=pathlib.Path(__file__).resolve().parent.parent,
        )
    elapsed = time.time() - t0

    metrics = {m: None for m in METRICS}
    metrics.update({
        "label": label,
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 1),
    })
    try:
        for line in log_path.read_text().splitlines():
            for m in METRICS:
                prefix = f"{m}:"
                if line.startswith(prefix):
                    raw = line[len(prefix):].strip()
                    try:
                        metrics[m] = float(raw)
                    except ValueError:
                        metrics[m] = raw
                    break
    except OSError:
        pass
    return {**combo, **metrics}


def _print_leaderboard(rows: list[dict]) -> None:
    primary = METRICS[0]
    valid = [r for r in rows if isinstance(r.get(primary), (int, float))]
    invalid = [r for r in rows if r not in valid]
    valid.sort(key=lambda r: -float(r[primary]))
    print("\n=== Sweep leaderboard (sorted by vs_pool_win_rate) ===")
    cols = list(DEFAULT_GRID.keys()) + METRICS + ["elapsed_seconds"]
    header = " | ".join(f"{c:>16s}" if i >= len(DEFAULT_GRID) else f"{c:>14s}"
                        for i, c in enumerate(cols))
    print(header)
    for r in valid + invalid:
        cells = []
        for i, c in enumerate(cols):
            v = r.get(c)
            if v is None:
                cell = "-"
            elif isinstance(v, float):
                cell = f"{v:.3f}"
            else:
                cell = str(v)
            cells.append(f"{cell:>16s}" if i >= len(DEFAULT_GRID) else f"{cell:>14s}")
        print(" | ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--time-budget", type=int, default=60,
                        help="TIME_BUDGET (seconds) per run")
    parser.add_argument("--eval-games", type=int, default=6)
    parser.add_argument("--out", default="sweep_results.tsv")
    parser.add_argument("--base-dir", default="models/sweep",
                        help="Per-run model directories live under this path")
    args = parser.parse_args()

    base_dir = pathlib.Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    combos = _grid_combinations(DEFAULT_GRID)
    print(f"Sweep: {len(combos)} configurations × {args.time_budget}s "
          f"each, concurrency={args.concurrency}")
    for c in combos:
        print(f"  {_label_for(c)}")

    rows: list[dict] = []
    if args.concurrency <= 1:
        for c in combos:
            print(f"\n>>> running {_label_for(c)}")
            r = run_one(c, args.time_budget, args.eval_games, base_dir)
            rows.append(r)
            print(f">>> done: {r.get(METRICS[0])}")
    else:
        with ProcessPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(run_one, c, args.time_budget, args.eval_games, base_dir): c
                for c in combos
            }
            for fut in as_completed(futures):
                r = fut.result()
                rows.append(r)
                print(f">>> finished {r['label']}: {r.get(METRICS[0])}")

    # Persist TSV.
    out_path = pathlib.Path(args.out)
    cols = list(DEFAULT_GRID.keys()) + ["label"] + METRICS + ["returncode", "elapsed_seconds"]
    with out_path.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nWrote {out_path}")

    _print_leaderboard(rows)


if __name__ == "__main__":
    main()
