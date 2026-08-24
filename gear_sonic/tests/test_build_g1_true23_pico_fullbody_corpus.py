from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts.build_g1_true23_pico_fullbody_corpus import (
    MANIFEST_KIND,
    RECOVERY_METADATA_SCHEMA,
    SPAN_KIND,
    build_fullbody_corpus,
)
from gear_sonic.utils.g1_23dof_incremental_corpus import sha256_file


def _motion(path: Path, frames: int) -> None:
    arrays = {
        "fps": np.asarray([50.0], dtype=np.float64),
        "joint_pos": np.zeros((frames, 23), dtype=np.float32),
        "joint_vel": np.zeros((frames, 23), dtype=np.float32),
        "body_pos_w": np.zeros((frames, 24, 3), dtype=np.float32),
        "body_quat_w": np.zeros((frames, 24, 4), dtype=np.float32),
        "body_lin_vel_w": np.zeros((frames, 24, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((frames, 24, 3), dtype=np.float32),
    }
    arrays["body_quat_w"][..., 0] = 1.0
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def test_builds_v14_compatible_span_sidecar(tmp_path: Path) -> None:
    model = tmp_path / "external_dependencies/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1_23dof.xml"
    model.parent.mkdir(parents=True)
    model.write_text("<mujoco/>", encoding="utf-8")
    _motion(tmp_path / "short.npz", 40)
    _motion(tmp_path / "long.npz", 530)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": MANIFEST_KIND,
                "motions": [
                    {"name": "short", "path": "short.npz", "weight": 2.0},
                    {"name": "long", "path": "long.npz", "weight": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    spans = tmp_path / "corpus.spans.json"
    recovery = tmp_path / "corpus.recovery.json"
    report = build_fullbody_corpus(
        repository_root=tmp_path,
        manifest_path=manifest,
        output_path=tmp_path / "corpus.npz",
        catalog_path=tmp_path / "catalog.json",
        spans_path=spans,
        recovery_metadata_path=recovery,
        episode_frames=500,
    )
    assert report["kind"] == SPAN_KIND
    assert report["clip_count"] == 2
    assert report["total_frames"] == 1030
    assert [span["start"] for span in report["spans"]] == [0, 500]
    assert [span["length"] for span in report["spans"]] == [500, 530]
    assert [span["original_length"] for span in report["spans"]] == [40, 530]
    with np.load(tmp_path / "corpus.npz", allow_pickle=False) as archive:
        assert archive["joint_pos"].shape == (1030, 23)
    assert json.loads(spans.read_text(encoding="utf-8")) == report
    metadata = json.loads(recovery.read_text(encoding="utf-8"))
    assert metadata["schema"] == RECOVERY_METADATA_SCHEMA
    assert metadata["output"] == {
        "duration_s": 20.58,
        "filename": "corpus.npz",
        "fps": 50.0,
        "frames": 1030,
        "sha256": sha256_file(tmp_path / "corpus.npz"),
    }
    assert metadata["segments_inclusive"] == {"long": [500, 1029], "short": [0, 499]}
    assert metadata["corpus"]["full_body_controlled_joint_count"] == 23
    assert metadata["corpus"]["source_29dof_physics_used"] is False
    assert metadata["authorization"]["hardware_authorized"] is False
