"""Manage trained Tigphrates model artifacts.

This script gives the ignored local models/ tree a small lifecycle:

    python manage_models.py inventory
    python manage_models.py promote models/policy_final.pt --run-id ds_may08
    python manage_models.py prune --keep-pool 5
    python manage_models.py prune --keep-pool 5 --yes

`prune` is a dry run unless --yes is passed. It always protects
<models>/current so consumers have one stable location to load from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_PATTERNS = ("*.pt", "*.onnx")
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class PruneItem:
    path: Path
    reason: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _copy_optional(src: Path | None, dst: Path) -> dict[str, Any] | None:
    if src is None:
        return None
    if not src.exists():
        raise FileNotFoundError(f"artifact not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return {
        "path": str(dst),
        "source": str(src),
        "size_bytes": dst.stat().st_size,
        "sha256": sha256_file(dst),
    }


def promote_model(
    models_dir: Path | str,
    policy_path: Path | str,
    run_id: str,
    onnx_path: Path | str | None = None,
    metrics: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Copy a policy into <models>/current and write a manifest.

    The current directory is the stable consumer-facing location. Existing
    files there are overwritten atomically enough for local CLI use.
    """
    models_dir = Path(models_dir)
    policy_src = Path(policy_path)
    onnx_src = Path(onnx_path) if onnx_path is not None else None
    if not policy_src.exists():
        raise FileNotFoundError(f"policy not found: {policy_src}")

    current_dir = models_dir / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    policy_dst = current_dir / "policy.pt"
    shutil.copyfile(policy_src, policy_dst)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": _now_iso(),
        "promoted": True,
        "source": str(policy_src),
        "path": str(policy_dst),
        "size_bytes": policy_dst.stat().st_size,
        "sha256": sha256_file(policy_dst),
        "metrics": metrics or {},
        "notes": notes,
    }

    onnx_meta = _copy_optional(onnx_src, current_dir / "policy.onnx")
    if onnx_meta is not None:
        manifest["onnx"] = onnx_meta

    manifest_path = current_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _model_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in MODEL_PATTERNS:
        files.extend(root.rglob(pattern))
    return sorted(p for p in files if p.is_file())


def _sort_newest(paths: list[Path]) -> list[Path]:
    def key(path: Path) -> tuple[int, float, str]:
        match = re.search(r"_r(\d+)", path.stem)
        rollout = int(match.group(1)) if match else -1
        return rollout, path.stat().st_mtime, path.name

    return sorted(paths, key=key, reverse=True)


def plan_prune(
    models_dir: Path | str,
    keep_pool: int = 5,
    delete_sweeps: bool = True,
    delete_legacy_pool: bool = False,
) -> list[PruneItem]:
    """Return files that should be deleted by a prune operation.

    Rules:
    - never delete <models>/current
    - optionally delete all sweep model artifacts
    - optionally delete the top-level legacy <models>/pool artifacts
    - in every remaining pool directory, keep only the newest `keep_pool`
      policy snapshots and delete older snapshots
    """
    models_dir = Path(models_dir)
    current_dir = models_dir / "current"
    items: list[PruneItem] = []
    planned: set[Path] = set()

    def add(path: Path, reason: str) -> None:
        if _is_under(path, current_dir):
            return
        if path in planned:
            return
        planned.add(path)
        items.append(PruneItem(path=path, reason=reason))

    sweep_dir = models_dir / "sweep"
    if delete_sweeps and sweep_dir.exists():
        for path in _model_files(sweep_dir):
            add(path, "stale sweep artifact")

    legacy_pool = models_dir / "pool"
    if delete_legacy_pool and legacy_pool.exists():
        for path in _model_files(legacy_pool):
            add(path, "legacy top-level pool artifact")

    pool_dirs = [p for p in models_dir.rglob("pool") if p.is_dir()]
    if legacy_pool.exists() and legacy_pool not in pool_dirs:
        pool_dirs.append(legacy_pool)

    for pool_dir in sorted(pool_dirs):
        if _is_under(pool_dir, current_dir):
            continue
        if delete_legacy_pool and pool_dir == legacy_pool:
            continue
        snapshots = sorted(pool_dir.glob("policy_*.pt"))
        if keep_pool < 0:
            keep = set()
        else:
            keep = set(_sort_newest(snapshots)[:keep_pool])
        for path in snapshots:
            if path not in keep:
                add(path, f"older pool snapshot beyond keep_pool={keep_pool}")

    return sorted(items, key=lambda item: str(item.path))


def apply_prune(plan: list[PruneItem], yes: bool = False) -> list[Path]:
    """Delete planned files when confirmed. Without yes this is a no-op."""
    if not yes:
        return []
    removed: list[Path] = []
    for item in plan:
        try:
            item.path.unlink()
            removed.append(item.path)
        except FileNotFoundError:
            pass
    return removed


def inventory(models_dir: Path | str) -> dict[str, Any]:
    models_dir = Path(models_dir)
    files = _model_files(models_dir) if models_dir.exists() else []
    by_hash: dict[str, list[str]] = {}
    artifacts = []
    for path in files:
        digest = sha256_file(path)
        by_hash.setdefault(digest, []).append(str(path))
        artifacts.append({
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "mtime": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).replace(microsecond=0).isoformat(),
            "sha256": digest,
        })
    duplicates = {
        digest: paths for digest, paths in by_hash.items() if len(paths) > 1
    }
    return {
        "models_dir": str(models_dir),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "duplicate_groups": duplicates,
    }


def _json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        raise argparse.ArgumentTypeError("--metrics-json must decode to an object")
    return loaded


def _print_prune_plan(plan: list[PruneItem], dry_run: bool) -> None:
    label = "dry run" if dry_run else "deleting"
    print(f"prune: {label}; {len(plan)} file(s) selected")
    for item in plan:
        print(f"  - {item.path} ({item.reason})")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Root model artifact directory (default: models).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("inventory", help="Print model artifact inventory as JSON.")

    promote = sub.add_parser("promote", help="Promote a policy into models/current.")
    promote.add_argument("policy_path")
    promote.add_argument("--run-id", required=True)
    promote.add_argument("--onnx-path", default=None)
    promote.add_argument("--metrics-json", default=None)
    promote.add_argument("--notes", default="")

    prune = sub.add_parser("prune", help="Prune stale model artifacts.")
    prune.add_argument("--keep-pool", type=int, default=5)
    prune.add_argument(
        "--keep-sweeps",
        action="store_true",
        help="Do not delete models/sweep artifacts.",
    )
    prune.add_argument(
        "--delete-legacy-pool",
        action="store_true",
        help="Delete all artifacts in the top-level models/pool directory.",
    )
    prune.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete selected files. Without this, prune is a dry run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    models_dir = Path(args.models_dir)

    if args.command == "inventory":
        print(json.dumps(inventory(models_dir), indent=2, sort_keys=True))
        return 0

    if args.command == "promote":
        manifest = promote_model(
            models_dir=models_dir,
            policy_path=args.policy_path,
            run_id=args.run_id,
            onnx_path=args.onnx_path,
            metrics=_json_arg(args.metrics_json),
            notes=args.notes,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if args.command == "prune":
        plan = plan_prune(
            models_dir=models_dir,
            keep_pool=args.keep_pool,
            delete_sweeps=not args.keep_sweeps,
            delete_legacy_pool=args.delete_legacy_pool,
        )
        _print_prune_plan(plan, dry_run=not args.yes)
        removed = apply_prune(plan, yes=args.yes)
        if args.yes:
            print(f"prune: removed {len(removed)} file(s)")
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
