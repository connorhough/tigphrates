"""Tests for python/reset_pool.py — destructive pool wipe used to recover
from a collapsed self-play pool. Tests use tmp_path so the real
models/pool/ is never touched.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _populate_fake_pool(pool_dir: Path, with_init: bool = True) -> dict:
    """Create a representative set of pool artifacts. Returns a dict of the
    paths created so tests can assert on individual items."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["init_pt"] = pool_dir / "policy_init.pt"
    paths["snap_pt"] = pool_dir / "policy_r0001.pt"
    paths["snap_onnx"] = pool_dir / "policy_r0001.onnx"
    paths["final_pt"] = pool_dir / "policy_final_r0010.pt"
    paths["elo"] = pool_dir / "elo.json"
    if with_init:
        paths["init_pt"].write_bytes(b"INIT")
    paths["snap_pt"].write_bytes(b"SNAP")
    paths["snap_onnx"].write_bytes(b"SNAP_ONNX")
    paths["final_pt"].write_bytes(b"FINAL")
    paths["elo"].write_text(json.dumps({"_agent": 1468, "policy_init": 1500}))
    return paths


def test_reset_function_wipes_pool(tmp_path):
    from reset_pool import reset_pool

    paths = _populate_fake_pool(tmp_path, with_init=True)
    removed = reset_pool(pool_dir=tmp_path, init_checkpoint=None, keep_init=False)

    # All policy_*.pt, policy_*.onnx and elo.json must be gone.
    assert not paths["init_pt"].exists()
    assert not paths["snap_pt"].exists()
    assert not paths["snap_onnx"].exists()
    assert not paths["final_pt"].exists()
    assert not paths["elo"].exists()
    # And the function reported what it removed (≥ 5 items here).
    assert len(removed) >= 5


def test_keep_init_preserves_existing_init(tmp_path):
    from reset_pool import reset_pool

    paths = _populate_fake_pool(tmp_path, with_init=True)
    reset_pool(pool_dir=tmp_path, init_checkpoint=None, keep_init=True)

    assert paths["init_pt"].exists()
    assert paths["init_pt"].read_bytes() == b"INIT"
    # But everything else is wiped.
    assert not paths["snap_pt"].exists()
    assert not paths["snap_onnx"].exists()
    assert not paths["final_pt"].exists()
    assert not paths["elo"].exists()


def test_init_checkpoint_copies_file(tmp_path):
    from reset_pool import reset_pool

    _populate_fake_pool(tmp_path, with_init=False)
    src = tmp_path / "src_init.pt"
    src.write_bytes(b"FRESH")

    reset_pool(pool_dir=tmp_path, init_checkpoint=src, keep_init=False)

    new_init = tmp_path / "policy_init.pt"
    assert new_init.exists()
    assert new_init.read_bytes() == b"FRESH"


def test_init_checkpoint_overrides_keep_init(tmp_path):
    """If both --init-checkpoint and --keep-init are passed, the explicit
    checkpoint wins (the user explicitly asked for a new init)."""
    from reset_pool import reset_pool

    paths = _populate_fake_pool(tmp_path, with_init=True)
    src = tmp_path / "src_init.pt"
    src.write_bytes(b"FRESH")

    reset_pool(pool_dir=tmp_path, init_checkpoint=src, keep_init=True)

    new_init = paths["init_pt"]
    assert new_init.exists()
    assert new_init.read_bytes() == b"FRESH"


def test_cli_requires_yes(tmp_path):
    """Running the CLI without --yes refuses to do anything."""
    paths = _populate_fake_pool(tmp_path, with_init=True)

    script = Path(__file__).resolve().parent.parent / "reset_pool.py"
    result = subprocess.run(
        [sys.executable, str(script), "--pool-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    # Non-zero exit when --yes is missing.
    assert result.returncode != 0
    # And nothing was deleted.
    assert paths["init_pt"].exists()
    assert paths["snap_pt"].exists()
    assert paths["elo"].exists()


def test_cli_with_yes_wipes(tmp_path):
    """CLI with --yes removes pool files."""
    paths = _populate_fake_pool(tmp_path, with_init=True)

    script = Path(__file__).resolve().parent.parent / "reset_pool.py"
    result = subprocess.run(
        [sys.executable, str(script), "--pool-dir", str(tmp_path), "--yes"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert not paths["init_pt"].exists()
    assert not paths["snap_pt"].exists()
    assert not paths["elo"].exists()


def test_cli_keep_init(tmp_path):
    """CLI --keep-init preserves policy_init.pt."""
    paths = _populate_fake_pool(tmp_path, with_init=True)

    script = Path(__file__).resolve().parent.parent / "reset_pool.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--pool-dir", str(tmp_path),
            "--keep-init", "--yes",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert paths["init_pt"].exists()
    assert paths["init_pt"].read_bytes() == b"INIT"
    assert not paths["snap_pt"].exists()
    assert not paths["elo"].exists()
