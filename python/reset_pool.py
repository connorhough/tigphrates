"""Reset the self-play pool directory.

Wipes all snapshot weights (`policy_*.pt`), exported ONNX (`policy_*.onnx`),
and the persisted Elo table (`elo.json`) from the pool. Use this when the
pool has collapsed onto a degenerate self-play equilibrium and training is
regressing — start fresh with an init checkpoint and let the curriculum
re-bootstrap from heuristic opponents.

CLI:
    python reset_pool.py --pool-dir models/pool --yes
    python reset_pool.py --pool-dir models/pool --keep-init --yes
    python reset_pool.py --pool-dir models/pool --init-checkpoint path.pt --yes

Refuses to run without --yes (the operation is destructive). The reset
function itself does not gate on --yes — that is purely a CLI safeguard so
test code can call `reset_pool(...)` directly.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable


# Patterns we wipe. Anything else in the pool directory is left alone.
_PT_GLOB = "policy_*.pt"
_ONNX_GLOB = "policy_*.onnx"
_ELO_FILE = "elo.json"


def _iter_pool_files(pool_dir: Path) -> Iterable[Path]:
    yield from pool_dir.glob(_PT_GLOB)
    yield from pool_dir.glob(_ONNX_GLOB)
    elo = pool_dir / _ELO_FILE
    if elo.exists():
        yield elo


def reset_pool(
    pool_dir: Path | str,
    init_checkpoint: Path | str | None = None,
    keep_init: bool = False,
) -> list[Path]:
    """Wipe the pool directory and optionally seed it with an init checkpoint.

    Args:
        pool_dir: directory holding policy_*.pt / policy_*.onnx / elo.json.
        init_checkpoint: if provided, copy this file to <pool_dir>/policy_init.pt
            after the wipe. Overrides `keep_init` (the user explicitly asked
            for a new init).
        keep_init: if True and policy_init.pt exists in the pool, preserve it.
            Useful when re-bootstrapping with the same architecture seed.

    Returns:
        List of paths that were removed.
    """
    pool_dir = Path(pool_dir)
    init_keep_path = pool_dir / "policy_init.pt"

    # Snapshot init bytes BEFORE iterating-and-removing, so a single pass over
    # the glob can wipe everything (we restore the init at the end if asked).
    keep_bytes: bytes | None = None
    if keep_init and init_checkpoint is None and init_keep_path.exists():
        keep_bytes = init_keep_path.read_bytes()

    removed: list[Path] = []
    for path in list(_iter_pool_files(pool_dir)):
        try:
            path.unlink()
            removed.append(path)
        except FileNotFoundError:
            # Race / glob double-yield safety.
            pass

    # Restore preserved init bytes if requested.
    if keep_bytes is not None:
        pool_dir.mkdir(parents=True, exist_ok=True)
        init_keep_path.write_bytes(keep_bytes)

    # Apply explicit init checkpoint last so it always wins over keep_init.
    if init_checkpoint is not None:
        src = Path(init_checkpoint)
        if not src.exists():
            raise FileNotFoundError(f"--init-checkpoint not found: {src}")
        pool_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, init_keep_path)

    return removed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--pool-dir",
        default="models/pool",
        help="Path to the pool directory (default: models/pool).",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional path to a fresh policy_init.pt to seed after wipe.",
    )
    parser.add_argument(
        "--keep-init",
        action="store_true",
        help="Preserve <pool>/policy_init.pt if it exists. Ignored when "
             "--init-checkpoint is provided.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation flag. Without it the script refuses to "
             "modify any files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.yes:
        sys.stderr.write(
            "reset_pool: refusing to run without --yes (this operation is "
            "destructive: it deletes every policy_*.pt, policy_*.onnx and "
            "elo.json in the pool dir).\n"
        )
        return 2

    pool_dir = Path(args.pool_dir)
    init_ckpt = Path(args.init_checkpoint) if args.init_checkpoint else None

    if not pool_dir.exists():
        sys.stderr.write(f"reset_pool: {pool_dir} does not exist; nothing to do.\n")
        # Still create the dir if an init checkpoint needs to land there.
        if init_ckpt is not None:
            pool_dir.mkdir(parents=True, exist_ok=True)

    removed = reset_pool(pool_dir, init_checkpoint=init_ckpt, keep_init=args.keep_init)

    if removed:
        sys.stdout.write(f"reset_pool: removed {len(removed)} file(s) from {pool_dir}:\n")
        for p in removed:
            sys.stdout.write(f"  - {p.name}\n")
    else:
        sys.stdout.write(f"reset_pool: no matching files in {pool_dir}.\n")

    if init_ckpt is not None:
        sys.stdout.write(
            f"reset_pool: seeded {pool_dir / 'policy_init.pt'} from {init_ckpt}\n"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
