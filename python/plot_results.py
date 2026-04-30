"""
Plot generator for results.tsv (autoresearch experiment log).

Reads the TSV produced by python/run_experiments.sh and renders a PNG with
per-commit win_rate and avg_min_score. Helps spot trends across many
experiments without scrolling text.

Usage:
    python python/plot_results.py [--in results.tsv] [--out results.png]

Falls back to ASCII spark output (no matplotlib import) when called with
--ascii or when matplotlib is not installed — handy on the Mac mini for
quick read-outs without producing image files.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys


def _load_rows(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)
    return rows


def _to_floats(rows: list[dict], col: str) -> list[float | None]:
    out: list[float | None] = []
    for r in rows:
        raw = r.get(col, "").strip()
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _ascii_sparkline(values: list[float | None], width: int = 60) -> str:
    """Tiny block-character sparkline. None → space."""
    blocks = "▁▂▃▄▅▆▇█"
    nums = [v for v in values if v is not None]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    rng = hi - lo or 1.0
    out = []
    for v in values:
        if v is None:
            out.append(" ")
        else:
            idx = int((v - lo) / rng * (len(blocks) - 1))
            out.append(blocks[max(0, min(len(blocks) - 1, idx))])
    s = "".join(out)
    if len(s) > width:
        s = s[-width:]
    return s


def _render_ascii(rows: list[dict]) -> None:
    cols = ["win_rate", "avg_min_score"]
    for col in cols:
        values = _to_floats(rows, col)
        nums = [v for v in values if v is not None]
        if not nums:
            print(f"{col}: (no data)")
            continue
        spark = _ascii_sparkline(values)
        print(f"{col:18s} {spark}  min={min(nums):.3f} max={max(nums):.3f} latest={nums[-1]:.3f}")


def _render_png(rows: list[dict], out_path: pathlib.Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; falling back to ASCII output.", file=sys.stderr)
        _render_ascii(rows)
        return

    win = _to_floats(rows, "win_rate")
    avg = _to_floats(rows, "avg_min_score")
    x = list(range(1, len(rows) + 1))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("experiment #")
    ax1.set_ylabel("win_rate vs heuristic", color="tab:blue")
    ax1.plot(x, [v if v is not None else float("nan") for v in win],
             "o-", color="tab:blue", label="win_rate")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("avg_min_score", color="tab:orange")
    ax2.plot(x, [v if v is not None else float("nan") for v in avg],
             "s--", color="tab:orange", alpha=0.7, label="avg_min_score")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle(f"autoresearch progress — {len(rows)} experiments")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="results.tsv")
    parser.add_argument("--out", default="results.png")
    parser.add_argument("--ascii", action="store_true",
                        help="Print ASCII sparklines instead of writing PNG.")
    args = parser.parse_args()

    inp = pathlib.Path(args.inp)
    if not inp.exists():
        print(f"Not found: {inp}", file=sys.stderr)
        sys.exit(2)

    rows = _load_rows(inp)
    if not rows:
        print(f"{inp} is empty.", file=sys.stderr)
        sys.exit(2)

    if args.ascii:
        _render_ascii(rows)
    else:
        _render_png(rows, pathlib.Path(args.out))


if __name__ == "__main__":
    main()
