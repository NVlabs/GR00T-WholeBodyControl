"""Adversarial provenance checks for true23 training and simulation."""

from __future__ import annotations

import copy
from pathlib import Path
import shutil

import pytest

from gear_sonic.utils import g1_23dof_artifact as artifact


def test_runtime_allowlist_covers_xr_bridge_and_input_decoder() -> None:
    assert (
        "external_dependencies/"
        "XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/"
        "bindings/py_bindings.cpp"
    ) in artifact.RUNTIME_SOURCE_RELPATHS
    assert (
        "gear_sonic/utils/teleop/input_readers.py"
        in artifact.RUNTIME_SOURCE_RELPATHS
    )
    assert (
        "gear_sonic/envs/manager_env/robots/g1.py"
        in artifact.RUNTIME_SOURCE_RELPATHS
    )
    assert set(artifact.TRUE23_ROBOT_ASSET_RELPATHS).issubset(
        artifact.RUNTIME_SOURCE_RELPATHS
    )


def test_runtime_completeness_rejects_omitted_local_import(
    tmp_path: Path,
) -> None:
    package = tmp_path / "gear_sonic"
    package.mkdir()
    (package / "a.py").write_text(
        "from gear_sonic import b\n",
        encoding="utf-8",
    )
    (package / "b.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="omits local imports"):
        artifact.validate_runtime_source_completeness(
            repo_root=tmp_path,
            relpaths=("gear_sonic/a.py",),
        )


def test_runtime_completeness_rejects_omitted_hydra_target() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    resolved_config = {
        "callbacks": {
            "wandb": {
                "_target_": (
                    "gear_sonic.trl.callbacks.wandb_callback.WandbCallback"
                )
            }
        }
    }
    with pytest.raises(ValueError, match="omits Hydra target modules"):
        artifact.validate_training_dynamic_target_completeness(
            resolved_config,
            repo_root=repo_root,
            relpaths=(),
        )


def test_true23_robot_asset_manifest_is_complete_and_content_bound(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = repo_root / "gear_sonic/data/robots/g1"
    destination = tmp_path / "gear_sonic/data/robots/g1"
    destination.parent.mkdir(parents=True)
    shutil.copytree(source, destination)

    first = artifact.canonical_true23_robot_asset_manifest(
        repo_root=tmp_path
    )
    assert first["file_count"] == len(artifact.TRUE23_ROBOT_ASSET_RELPATHS)
    mesh = destination / "meshes/head_link.STL"
    mesh.write_bytes(mesh.read_bytes() + b"drift")
    second = artifact.canonical_true23_robot_asset_manifest(
        repo_root=tmp_path
    )
    assert second["manifest_sha256"] != first["manifest_sha256"]

    mesh.unlink()
    with pytest.raises(ValueError, match="runtime source is missing"):
        artifact.canonical_true23_robot_asset_manifest(repo_root=tmp_path)


def _runtime_config() -> dict:
    return {
        "manager_env": {
            "commands": {
                "motion": {
                    "motion_lib_cfg": {
                        "motion_file": (
                            artifact.MOTION_DATASET_PROCESSED_ROOT_RELPATH
                        )
                    }
                }
            }
        }
    }


def _fixture_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict]:
    source_bytes = b"tiny deterministic BONES-SEED archive fixture"
    source = {
        "repository": "bones-studio/seed",
        "revision": "2f59b2077b9da34dd4e43618e705c7cb962c9a66",
        "relpath": "g1.tar.gz",
        "size_bytes": len(source_bytes),
        "sha256": artifact.sha256_bytes(source_bytes),
    }
    monkeypatch.setattr(artifact, "MOTION_DATASET_SOURCE_ARCHIVE", source)
    (tmp_path / source["relpath"]).write_bytes(source_bytes)

    motion_root = (
        tmp_path / artifact.MOTION_DATASET_PROCESSED_ROOT_RELPATH
    )
    (motion_root / "z").mkdir(parents=True)
    (motion_root / "z" / "b.pkl").write_bytes(b"b")
    (motion_root / "a.pkl").write_bytes(b"aa")
    processed = artifact.canonical_processed_motion_manifest(
        repo_root=tmp_path
    )

    source_relpaths = ("runtime/a.py", "runtime/z.py")
    monkeypatch.setattr(
        artifact,
        "RUNTIME_SOURCE_RELPATHS",
        source_relpaths,
    )
    for relpath, payload in zip(
        source_relpaths,
        (b"print('a')\n", b"print('z')\n"),
        strict=True,
    ):
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    runtime_sources = artifact.canonical_runtime_source_manifest(
        repo_root=tmp_path,
        relpaths=source_relpaths,
    )
    config = {
        "runtime_sources": {
            "schema_version": 1,
            "relpaths": list(source_relpaths),
            "file_count": runtime_sources["file_count"],
            "manifest_sha256": runtime_sources["manifest_sha256"],
        },
        "motion_dataset": {
            "schema_version": 1,
            "source_archive": source,
            "processed": processed,
            "promotion_enabled": True,
        },
    }
    evidence = {
        "schema_version": 1,
        "source_archive": source,
        "processed": processed,
    }
    return config, evidence


def test_processed_manifest_is_sorted_deterministic_and_content_bound(
    tmp_path: Path,
) -> None:
    root = tmp_path / artifact.MOTION_DATASET_PROCESSED_ROOT_RELPATH
    (root / "z").mkdir(parents=True)
    (root / "z" / "b.pkl").write_bytes(b"b")
    (root / "a.pkl").write_bytes(b"aa")

    first = artifact.canonical_processed_motion_manifest(
        repo_root=tmp_path
    )
    second = artifact.canonical_processed_motion_manifest(
        repo_root=tmp_path
    )
    assert first == second
    assert first["file_count"] == 2
    assert first["total_bytes"] == 3

    (root / "a.pkl").write_bytes(b"changed")
    changed = artifact.canonical_processed_motion_manifest(
        repo_root=tmp_path
    )
    assert changed["manifest_sha256"] != first["manifest_sha256"]


def test_motion_provenance_rejects_missing_archive_and_processed_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, evidence = _fixture_contract(tmp_path, monkeypatch)
    source_path = tmp_path / "g1.tar.gz"
    source_path.unlink()
    with pytest.raises(ValueError, match="source archive is missing"):
        artifact.approved_motion_dataset_provenance(
            _runtime_config(),
            validation_config=config,
            repo_root=tmp_path,
        )

    source_path.write_bytes(b"tiny deterministic BONES-SEED archive fixture")
    for runtime_config in (
        _runtime_config(),
        _runtime_config()["manager_env"],
    ):
        assert (
            artifact.approved_motion_dataset_provenance(
                runtime_config,
                validation_config=config,
                repo_root=tmp_path,
            )
            == evidence
        )
    divergent = _runtime_config()
    divergent["commands"] = copy.deepcopy(
        divergent["manager_env"]["commands"]
    )
    divergent["commands"]["motion"]["motion_lib_cfg"]["motion_file"] = (
        "data/other"
    )
    with pytest.raises(ValueError, match="divergent motion_file"):
        artifact.approved_motion_dataset_provenance(
            divergent,
            validation_config=config,
            repo_root=tmp_path,
        )
    motion_file = (
        tmp_path
        / artifact.MOTION_DATASET_PROCESSED_ROOT_RELPATH
        / "a.pkl"
    )
    motion_file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="processed motion dataset manifest"):
        artifact.approved_motion_dataset_provenance(
            _runtime_config(),
            validation_config=config,
            repo_root=tmp_path,
        )


def test_runtime_source_manifest_rejects_drift_escape_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _evidence = _fixture_contract(tmp_path, monkeypatch)
    approved = artifact.approved_runtime_source_manifest(
        validation_config=config,
        repo_root=tmp_path,
    )
    assert approved["file_count"] == 2

    (tmp_path / "runtime/a.py").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="runtime source manifest differs"):
        artifact.approved_runtime_source_manifest(
            validation_config=config,
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="repository-relative"):
        artifact.canonical_runtime_source_manifest(
            repo_root=tmp_path,
            relpaths=("../escape.py",),
        )

    link = tmp_path / "runtime/link.py"
    try:
        link.symlink_to(tmp_path / "runtime/z.py")
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="may not traverse symlinks"):
        artifact.canonical_runtime_source_manifest(
            repo_root=tmp_path,
            relpaths=("runtime/link.py",),
        )


def test_simulation_material_provenance_must_equal_trained_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, evidence = _fixture_contract(tmp_path, monkeypatch)
    material = artifact.simulation_material_provenance(
        _runtime_config(),
        checkpoint_motion_dataset=evidence,
        validation_config=config,
        repo_root=tmp_path,
    )
    assert material["motion_dataset"] == evidence
    assert material["runtime_source"]["file_count"] == 2

    wrong = copy.deepcopy(evidence)
    wrong["processed"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from trained checkpoint"):
        artifact.simulation_material_provenance(
            _runtime_config(),
            checkpoint_motion_dataset=wrong,
            validation_config=config,
            repo_root=tmp_path,
        )


def test_processed_manifest_rejects_escape_and_symlink(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        artifact.canonical_processed_motion_manifest(
            repo_root=tmp_path,
            root_relpath="../outside",
        )

    root = tmp_path / artifact.MOTION_DATASET_PROCESSED_ROOT_RELPATH
    root.mkdir(parents=True)
    target = tmp_path / "outside.pkl"
    target.write_bytes(b"outside")
    link = root / "linked.pkl"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symlink files"):
        artifact.canonical_processed_motion_manifest(
            repo_root=tmp_path
        )
