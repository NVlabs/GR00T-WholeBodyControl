from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.scripts import analyze_g1_23dof_omitted_joint_correlates as audit


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "retarget"
    (root / "reports").mkdir(parents=True)
    (root / "experts").mkdir()
    csv = tmp_path / "source.csv"
    count = 8
    frame = {
        "root_translateX": np.zeros(count),
        "root_translateY": np.zeros(count),
        "root_translateZ": np.ones(count),
        "root_rotateX": np.zeros(count),
        "root_rotateY": np.zeros(count),
        "root_rotateZ": np.zeros(count),
    }
    for i, name in enumerate(audit.OMITTED_JOINTS):
        frame[f"{name}_dof"] = np.arange(count) * (i + 1)
    columns = list(frame)
    lines = [",".join(columns)]
    lines.extend(",".join(str(float(frame[column][index])) for column in columns) for index in range(count))
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    expert = root / "experts" / "clip.task_space.npz"
    np.savez(
        expert,
        schema_version=np.asarray([6]),
        expert_valid=np.asarray([True, False, False, True, True, False], dtype=np.bool_),
    )
    report = {
        "schema_version": 6,
        "clip_id": "clip",
        "frame_count": 6,
        "fps_source": 60.0,
        "fps_target": 50.0,
        "source_csv": str(csv),
        "source_csv_sha256": _hash(csv),
        "expert_output_sha256": _hash(expert),
        "retiming": {"interpolation": audit.INTERPOLATION, "base_frame_count": 6, "retimed_frame_count": 6},
    }
    (root / "reports" / "clip.retarget.json").write_text(json.dumps(report), encoding="utf-8")
    return root


def test_build_audit_reconstructs_two_pass_clipped_values_and_windows(tmp_path: Path) -> None:
    payload = audit.build_omitted_joint_audit(_write_fixture(tmp_path))
    assert payload["schema"] == audit.AUDIT_SCHEMA
    assert payload["frame_count"] == 6
    assert payload["invalid_frame_count"] == 3
    assert payload["expert_authorized"] is False
    clip = payload["clips"][0]
    assert [(x["start_frame"], x["end_frame"]) for x in clip["invalid_windows"]] == [(1, 2), (5, 5)]
    assert clip["feature_stats"]["waist_roll_joint_angle_abs_rad"]["auc_valid_higher"] is not None


def test_rejects_lineage_mismatch(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    report_path = root / "reports" / "clip.retarget.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["expert_output_sha256"] = "bad"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="lineage hash"):
        audit.build_omitted_joint_audit(root)


def test_derivatives_groups_and_schema_fail_closed(tmp_path: Path) -> None:
    values = np.arange(30, dtype=np.float64).reshape(5, 6)
    velocity, acceleration = audit._derivatives(values, 2.0)
    np.testing.assert_allclose(velocity[0], velocity[1])
    assert np.all(acceleration[0] == 0.0)
    features = audit._features(values, 2.0)
    assert "waist_l2_angle" in features
    assert "all_six_l2_acceleration" in features

    root = _write_fixture(tmp_path)
    report_path = root / "reports" / "clip.retarget.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema_version"] = 3
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        audit.build_omitted_joint_audit(root)


def test_tie_aware_ranks_and_valid_q90_lift() -> None:
    scores = np.asarray([0.0, 0.0, 1.0, 2.0, 3.0])
    valid = np.asarray([True, True, True, False, False])
    ranks = audit._ranks(scores)
    assert ranks[0] == ranks[1]
    assert ranks[0] < ranks[-1]
    # valid q90=0.8; selected values 1,2,3 are 2/3 invalid; baseline is 2/5.
    assert audit._lift(valid, scores) == pytest.approx((2.0 / 3.0) / (2.0 / 5.0))
