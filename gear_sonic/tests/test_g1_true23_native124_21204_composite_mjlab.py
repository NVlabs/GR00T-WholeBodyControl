from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils import g1_true23_native124_21204_composite_mjlab as diagnostic
from gear_sonic.utils.g1_23dof_native124_21204_adapter import ONNX_SHA256
from gear_sonic.utils.g1_true23_teacher_support import (
    SUPPORT_CONFIG_SHA256,
    compose_checkpoint21204_teacher_action,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DAD_DANCE = REPOSITORY_ROOT / "artifacts/g1_native124_multimotion/scaling_all61/feasible_v1/npz/B_DadDance.npz"


def test_frozen_contract_and_primary_dad_dance_motion_are_hash_locked() -> None:
    contract = diagnostic.load_composite_mjlab_contract(REPOSITORY_ROOT)

    assert contract["artifact_identity"]["onnx_sha256"] == ONNX_SHA256
    assert contract["artifact_identity"]["support_config_sha256"] == SUPPORT_CONFIG_SHA256
    assert contract["causal_window"]["teacher_control_transitions"] == 500
    assert contract["causal_window"]["expected_final_q9_index"] == 511
    assert contract["action_chain"]["safe_target_v11_action_term_required"] is True
    motion = diagnostic.validate_diagnostic_motion(DAD_DANCE, contract)
    assert motion["name"] == "dad_dance_primary"
    assert motion["sha256"] == "a4962f1e4df45ca70ada473a962b52527b17ac667dfb17ef3cd37d4ed21c3bfb"
    assert motion["frame_count"] == 2090


def _write_motion(path: Path, *, frames: int = 20, nonfinite: bool = False) -> str:
    joint_pos = np.zeros((frames, 23), dtype=np.float32)
    if nonfinite:
        joint_pos[0, 0] = np.nan
    body_quat = np.zeros((frames, 24, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez(
        path,
        fps=np.asarray([50.0], dtype=np.float64),
        joint_pos=joint_pos,
        joint_vel=np.zeros((frames, 23), dtype=np.float32),
        body_pos_w=np.zeros((frames, 24, 3), dtype=np.float32),
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((frames, 24, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 24, 3), dtype=np.float32),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _motion_contract(path: Path, sha256: str, frames: int = 20) -> dict[str, object]:
    return {
        "motions": [
            {
                "name": "fixture",
                "basename": path.name,
                "sha256": sha256,
                "fps": 50.0,
                "frame_count": frames,
            }
        ],
        "motion_array_contract": {
            "required_keys": [
                "fps",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ]
        },
    }


def test_motion_validation_fails_closed_on_hash_or_nonfinite_drift(tmp_path: Path) -> None:
    valid = tmp_path / "fixture.npz"
    valid_sha = _write_motion(valid)
    manifest = diagnostic.validate_diagnostic_motion(
        valid,
        _motion_contract(valid, valid_sha),
    )
    assert manifest["frame_count"] == 20

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        diagnostic.validate_diagnostic_motion(
            valid,
            _motion_contract(valid, "0" * 64),
        )

    bad = tmp_path / "bad.npz"
    bad_sha = _write_motion(bad, nonfinite=True)
    with pytest.raises(ValueError, match="NaN or Inf"):
        diagnostic.validate_diagnostic_motion(
            bad,
            _motion_contract(bad, bad_sha),
        )


def test_composite_record_exposes_chain_without_admitting_label() -> None:
    composite = compose_checkpoint21204_teacher_action(
        np.zeros(23, dtype=np.float32),
        repository_root=REPOSITORY_ROOT,
    )
    record = diagnostic.composite_action_diagnostic_record(composite)

    assert len(record["selected_teacher_raw_action_hardware"]) == 23
    assert len(record["teacher_candidate_target_hardware"]) == 23
    assert len(record["plain_sonic_raw_native_diagnostic"]) == 23
    assert len(record["applied_safe_native_action"]) == 23
    assert len(record["teacher_composite_target_hardware"]) == 23
    assert record["projection_linf_rad"] > 0.0
    assert record["teacher_label_admitted"] is False
    assert "teacher_action_native" not in record

    with pytest.raises(ValueError, match="requires plain SONIC raw clipping"):
        compose_checkpoint21204_teacher_action(
            np.full(23, 100.0, dtype=np.float32),
            repository_root=REPOSITORY_ROOT,
        )


def _passing_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for step in range(500):
        records.append(
            {
                "q9_before": 11 + step,
                "q9_after": 12 + step,
                "teacher_join_inference_duration_ms": 0.25,
                "composite_action": {
                    "max_abs_selected_teacher_raw_action": 2.0,
                    "max_abs_plain_sonic_raw_native": 2.5,
                    "projection_linf_rad": 0.1,
                    "projection_l2_rad": 0.2,
                },
                "metrics": {
                    "action_semantics_match": True,
                    "target_soft_limit_violation_count": 0,
                    "actuator_target_soft_limit_violation_count": 0,
                    "measured_soft_limit_violation_count": 0,
                    "joint_velocity_limit_violation_count": 0,
                    "maximum_joint_velocity_ratio": 0.2,
                    "base_height_m": 0.75,
                    "base_tilt_rad": 0.1,
                    "target_tracking_rmse_rad": 0.2,
                },
            }
        )
    return records


def test_exact_500_step_summary_rejects_q9_discontinuity() -> None:
    contract = diagnostic.load_composite_mjlab_contract(REPOSITORY_ROOT)
    records = _passing_records()
    summary = diagnostic._summary(records, contract)  # noqa: SLF001
    assert summary["record_count"] == 500
    assert summary["q9_final_after"] == 511
    assert summary["nominal_gate_pass"] is True

    records[250]["q9_before"] = 999
    with pytest.raises(RuntimeError, match="q9 sequence"):
        diagnostic._summary(records, contract)  # noqa: SLF001


def test_report_writer_uses_exclusive_create(tmp_path: Path) -> None:
    output = tmp_path / "diagnostic.json"
    report = {"kind": "fixture", "computed_pass": False}

    diagnostic.write_report_new(output, report)

    assert output.read_bytes() == diagnostic.canonical_json_bytes(report)
    with pytest.raises(FileExistsError):
        diagnostic.write_report_new(output, report)
