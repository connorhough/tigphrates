"""Tests for python/manage_models.py.

The real models/ directory contains large local artifacts and is ignored by
git, so these tests build small fake model trees under tmp_path.
"""
import json
import subprocess
import sys
from pathlib import Path


def _write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_promote_copies_policy_and_writes_manifest(tmp_path):
    from manage_models import promote_model

    src = _write(tmp_path / "runs" / "run-a" / "policy_final.pt", b"MODEL")
    onnx = _write(tmp_path / "runs" / "run-a" / "policy.onnx", b"ONNX")

    manifest = promote_model(
        models_dir=tmp_path,
        policy_path=src,
        run_id="run-a",
        onnx_path=onnx,
        metrics={"win_rate": 0.25},
        notes="smoke",
    )

    current = tmp_path / "current"
    assert (current / "policy.pt").read_bytes() == b"MODEL"
    assert (current / "policy.onnx").read_bytes() == b"ONNX"
    assert manifest["run_id"] == "run-a"
    assert manifest["sha256"] == (
        "c8b6c094bbbda1f69723dcf2333ab9c9086576065bd1554f7466c52f47f2720d"
    )
    assert manifest["metrics"] == {"win_rate": 0.25}
    assert json.loads((current / "manifest.json").read_text())["notes"] == "smoke"


def test_plan_prune_protects_current_and_keeps_newest_pool_members(tmp_path):
    from manage_models import plan_prune

    current = _write(tmp_path / "current" / "policy.pt")
    older = _write(tmp_path / "pool" / "policy_r0004.pt")
    newer = _write(tmp_path / "pool" / "policy_r0008.pt")
    newest = _write(tmp_path / "pool" / "policy_r0012.pt")
    elo = _write(tmp_path / "pool" / "elo.json", b"{}")
    stale_sweep = _write(tmp_path / "sweep" / "old" / "policy_final.pt")

    plan = plan_prune(
        models_dir=tmp_path,
        keep_pool=2,
        delete_sweeps=True,
        delete_legacy_pool=False,
    )

    delete_paths = {item.path for item in plan}
    assert current not in delete_paths
    assert older in delete_paths
    assert newer not in delete_paths
    assert newest not in delete_paths
    assert elo not in delete_paths
    assert stale_sweep in delete_paths


def test_plan_prune_ranks_pool_snapshots_by_rollout_label(tmp_path):
    from manage_models import plan_prune

    newest = _write(tmp_path / "pool" / "policy_r0048.pt")
    older = _write(tmp_path / "pool" / "policy_r0004.pt")
    # Touch the older file last; retention should still prefer r0048.
    older.write_bytes(b"newer mtime")

    plan = plan_prune(tmp_path, keep_pool=1, delete_sweeps=False)

    delete_paths = {item.path for item in plan}
    assert older in delete_paths
    assert newest not in delete_paths


def test_apply_prune_requires_yes(tmp_path):
    from manage_models import apply_prune, plan_prune

    stale = _write(tmp_path / "sweep" / "old" / "policy_final.pt")
    plan = plan_prune(tmp_path, delete_sweeps=True)

    removed = apply_prune(plan, yes=False)

    assert removed == []
    assert stale.exists()


def test_apply_prune_deletes_when_confirmed(tmp_path):
    from manage_models import apply_prune, plan_prune

    stale = _write(tmp_path / "sweep" / "old" / "policy_final.pt")
    plan = plan_prune(tmp_path, delete_sweeps=True)

    removed = apply_prune(plan, yes=True)

    assert stale in removed
    assert not stale.exists()


def test_cli_prune_without_yes_is_dry_run(tmp_path):
    stale = _write(tmp_path / "sweep" / "old" / "policy_final.pt")
    script = Path(__file__).resolve().parent.parent / "manage_models.py"

    result = subprocess.run(
        [sys.executable, str(script), "--models-dir", str(tmp_path), "prune"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "dry run" in result.stdout
    assert stale.exists()
