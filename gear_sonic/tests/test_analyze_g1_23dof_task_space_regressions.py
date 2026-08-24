from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from gear_sonic.scripts import analyze_g1_23dof_task_space_regressions as audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_clip(
    root: Path,
    *,
    clip_id: str = "heldout_walk_001",
    mismatch_frame: int | None = None,
    no_contact_transitions: bool = False,
) -> tuple[Path, Path]:
    frame_count = 20
    fps = 50.0
    reports = root / "reports"
    experts = root / "experts"
    reports.mkdir(parents=True)
    experts.mkdir(parents=True)

    before = np.ones(frame_count, dtype=np.float32)
    after = np.full(frame_count, 0.9, dtype=np.float32)
    after[2:4] = 1.2
    after[8:18] = 1.1
    priority_0_before = np.full(frame_count, 0.1, dtype=np.float32)
    priority_0_after = priority_0_before.copy()
    priority_0_after[4] = 0.2
    priority_1_before = np.full(frame_count, 0.2, dtype=np.float32)
    priority_1_after = priority_1_before.copy()
    feasible = before.copy()
    feasible[2] = 1.3
    priority_0_seed = priority_0_before.copy()
    priority_0_seed[4] = 0.3

    valid = ~((after > before + 1.0e-7) | (priority_0_after > priority_0_before + 1.0e-7))
    if mismatch_frame is not None:
        valid[mismatch_frame] = ~valid[mismatch_frame]

    contacts = np.ones((frame_count, 2), dtype=np.bool_)
    if not no_contact_transitions:
        contacts[3:, 0] = False
        contacts[12:, 1] = False
    desired = np.zeros((frame_count, 2, 3), dtype=np.float32)
    desired[:, 0, 0] = np.arange(frame_count, dtype=np.float32) * 0.001
    desired[:, 1, 0] = np.arange(frame_count, dtype=np.float32) * 0.001
    desired[5:, 0, 0] += 0.1

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray([2], dtype=np.int64),
        "expert_valid": valid.astype(np.bool_),
        "weighted_task_error_before": before,
        "weighted_task_error_after": after,
        "weighted_task_error_feasible_seed": feasible,
        "priority_0_error_before": priority_0_before,
        "priority_0_error_after": priority_0_after,
        "priority_0_error_feasible_seed": priority_0_seed,
        "priority_1_error_before": priority_1_before,
        "priority_1_error_after": priority_1_after,
        "priority_1_error_feasible_seed": priority_1_before.copy(),
        "contact_flags": contacts,
        "desired_task_pos_w": desired,
        "trajectory_velocity_abs_max": np.zeros(frame_count, dtype=np.float32),
        "trajectory_acceleration_abs_max": np.zeros(frame_count, dtype=np.float32),
        "solver_iterations": np.ones(frame_count, dtype=np.float32),
        "position_limit_hit_count": np.zeros(frame_count, dtype=np.float32),
        "constraint_relaxation_count": np.zeros(frame_count, dtype=np.float32),
        "task_names": np.asarray(["left_hand", "right_hand"], dtype=np.str_),
    }
    arrays["trajectory_velocity_abs_max"][2] = 8.0
    arrays["trajectory_acceleration_abs_max"][3] = 80.0
    arrays["position_limit_hit_count"][4] = 1.0
    arrays["solver_iterations"][8] = 16.0
    for task_index, task_name in enumerate(("left_hand", "right_hand")):
        position_before = np.full(frame_count, 0.1 + task_index * 0.01, dtype=np.float32)
        position_after = position_before - 0.02
        position_after[2] = position_before[2] + 0.05 + task_index * 0.01
        arrays[f"task_{task_name}_position_error_before_m"] = position_before
        arrays[f"task_{task_name}_position_error_after_m"] = position_after
        arrays[f"task_{task_name}_orientation_error_before_rad"] = np.full(
            frame_count, 0.2, dtype=np.float32
        )
        arrays[f"task_{task_name}_orientation_error_after_rad"] = np.full(
            frame_count, 0.1, dtype=np.float32
        )

    expert_path = experts / f"{clip_id}.task_space.npz"
    np.savez(expert_path, **arrays)
    report_path = reports / f"{clip_id}.retarget.json"
    report = {
        "schema": "g1_29dof_to_true23_task_space_retarget_v3",
        "schema_version": 2,
        "clip_id": clip_id,
        "category": "walk",
        "frame_count": frame_count,
        "fps": fps,
        "expert_output_sha256": _sha256(expert_path),
        "retarget_config": {
            "priority_relative_tolerance": 1.0e-7,
            "protected_priority_tiers": 2,
            "max_velocity_rad_s": 8.0,
            "max_acceleration_rad_s2": 80.0,
            "max_iterations": 16,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, expert_path


def _write_v3_clip(root: Path) -> tuple[Path, Path]:
    clip_id = "heldout_v3_physical"
    frame_count = 12
    fps = 50.0
    reports = root / "reports"
    experts = root / "experts"
    reports.mkdir(parents=True)
    experts.mkdir(parents=True)
    task_names = (
        "left_foot",
        "right_foot",
        "whole_robot_com",
        "left_hand",
        "right_hand",
    )

    before = np.ones(frame_count, dtype=np.float32)
    after = np.full(frame_count, 0.9, dtype=np.float32)
    after[1] = 1.2
    constraint_relaxation = np.zeros(frame_count, dtype=np.float32)
    constraint_relaxation[5] = 1.0
    action = np.zeros((frame_count, 23), dtype=np.float32)
    action[6, 0] = 10.01
    priority_0_before = np.full(frame_count, 0.1, dtype=np.float32)
    priority_1_before = np.full(frame_count, 0.2, dtype=np.float32)
    priority_1_after = priority_1_before.copy()
    priority_1_after[7] = 0.3  # Old rule rejects this; v3 physical rule must not.

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray([3], dtype=np.int64),
        "weighted_task_error_before": before,
        "weighted_task_error_after": after,
        "weighted_task_error_feasible_seed": before.copy(),
        "priority_0_error_before": priority_0_before,
        "priority_0_error_after": priority_0_before.copy(),
        "priority_0_error_feasible_seed": priority_0_before.copy(),
        "priority_1_error_before": priority_1_before,
        "priority_1_error_after": priority_1_after,
        "priority_1_error_feasible_seed": priority_1_before.copy(),
        "contact_flags": np.ones((frame_count, 2), dtype=np.bool_),
        "desired_task_pos_w": np.zeros(
            (frame_count, len(task_names), 3), dtype=np.float32
        ),
        "trajectory_velocity_abs_max": np.zeros(frame_count, dtype=np.float32),
        "trajectory_acceleration_abs_max": np.zeros(frame_count, dtype=np.float32),
        "solver_iterations": np.ones(frame_count, dtype=np.float32),
        "position_limit_hit_count": np.zeros(frame_count, dtype=np.float32),
        "constraint_relaxation_count": constraint_relaxation,
        "action_target_native": action,
        "task_names": np.asarray(task_names, dtype=np.str_),
    }
    for task_name in task_names:
        position_before = np.zeros(frame_count, dtype=np.float32)
        position_after = np.zeros(frame_count, dtype=np.float32)
        orientation_before = np.zeros(frame_count, dtype=np.float32)
        orientation_after = np.zeros(frame_count, dtype=np.float32)
        if task_name == "left_foot":
            position_after[2] = 0.006
        elif task_name == "right_foot":
            orientation_after[3] = 0.006
        elif task_name == "whole_robot_com":
            position_before[:] = 0.02
            position_after[:] = 0.02
            position_after[4] = 0.0211
        arrays[f"task_{task_name}_position_error_before_m"] = position_before
        arrays[f"task_{task_name}_position_error_after_m"] = position_after
        arrays[f"task_{task_name}_orientation_error_before_rad"] = orientation_before
        arrays[f"task_{task_name}_orientation_error_after_rad"] = orientation_after

    valid = np.ones(frame_count, dtype=np.bool_)
    valid[1:7] = False
    arrays["expert_valid"] = valid
    expert_path = experts / f"{clip_id}.task_space.npz"
    np.savez(expert_path, **arrays)
    report_path = reports / f"{clip_id}.retarget.json"
    report = {
        "schema": "g1_29dof_to_true23_task_space_retarget_v3",
        "schema_version": 3,
        "clip_id": clip_id,
        "category": "mixed",
        "frame_count": frame_count,
        "fps": fps,
        "expert_output_sha256": _sha256(expert_path),
        "retarget_config": {
            "priority_relative_tolerance": 1.0e-7,
            "protected_priority_tiers": 2,
            "max_velocity_rad_s": 8.0,
            "max_acceleration_rad_s2": 80.0,
            "max_iterations": 16,
            "valid_max_foot_position_error_m": 0.005,
            "valid_max_foot_orientation_regression_rad": 0.005,
            "valid_max_com_regression_m": 0.001,
            "native_action_clip": 10.0,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path, expert_path


def test_audit_reports_exact_frames_windows_causes_and_task_deltas(tmp_path: Path) -> None:
    _write_clip(tmp_path)

    report = audit.build_regression_audit(tmp_path)

    assert report["schema"] == audit.AUDIT_SCHEMA
    assert report["deployment_ready"] is False
    assert report["expert_authorized"] is False
    assert report["training_expert_authorized"] is False
    assert report["frame_count"] == 20
    assert report["invalid_frame_count"] == 13
    assert report["invalid_frame_fraction"] == 13 / 20
    assert report["long_invalid_window_count"] == 1
    assert report["long_invalid_frame_count"] == 10
    assert report["integrity_passed"] is True

    clip = report["clips"][0]
    assert clip["invalid_frames"] == [2, 3, 4, *range(8, 18)]
    assert [
        (window["start_frame"], window["end_frame"], window["length_frames"])
        for window in clip["invalid_windows"]
    ] == [(2, 4, 3), (8, 17, 10)]
    causes = clip["cause_counts_on_invalid_frames"]
    assert causes["weighted_regression"] == 12
    assert causes["protected_priority_0_regression"] == 1
    assert causes["protected_priority_1_regression"] == 0
    assert causes["feasible_seed_already_worse"] == 2
    assert causes["velocity_ceiling"] == 1
    assert causes["acceleration_ceiling"] == 1
    assert causes["safe_envelope_hit"] == 1
    assert causes["solver_iteration_ceiling"] == 1
    assert clip["contact_transitions"]["transition_frames"] == [3, 12]
    assert clip["contact_transitions"]["changed_sides_by_frame"] == {
        "3": ["left"],
        "12": ["right"],
    }

    frame_two = next(
        item for item in clip["invalid_frame_diagnostics"] if item["frame"] == 2
    )
    assert "weighted_regression" in frame_two["causes"]
    assert "velocity_ceiling" in frame_two["causes"]
    assert frame_two["nearest_contact_transition_distance_frames"] == 1
    left_delta = frame_two["task_error_deltas"]["left_hand"]
    assert left_delta["position_delta_m"] > 0.0

    # Deterministic and strict-JSON-safe: no timestamp, NaN, or Infinity.
    second = audit.build_regression_audit(tmp_path)
    assert json.dumps(report, allow_nan=False, sort_keys=True) == json.dumps(
        second, allow_nan=False, sort_keys=True
    )


def test_mask_mismatch_is_exact_and_main_persists_non_authorizing_report(
    tmp_path: Path,
) -> None:
    _write_clip(tmp_path, mismatch_frame=1)
    output = tmp_path / "audit.json"

    exit_code = audit.main([str(tmp_path), "--output", str(output)])

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["integrity_passed"] is False
    assert payload["expert_valid_mismatch_count"] == 1
    assert payload["clips"][0]["expert_valid_mismatch_frames"] == [1]
    assert payload["expert_authorized"] is False
    assert not list(tmp_path.rglob("*.partial"))


def test_v3_recomputes_physical_validity_rule_and_ignores_legacy_tier_gate(
    tmp_path: Path,
) -> None:
    report_path, expert_path = _write_v3_clip(tmp_path)

    clip, _masks = audit.analyze_clip(report_path, expert_path)

    assert clip["validity_rule"] == {
        "schema_version": 3,
        "name": "physical_task_bounds_v3",
        "priority_relative_tolerance": 1.0e-7,
        "valid_max_foot_position_error_m": 0.005,
        "valid_max_foot_orientation_regression_rad": 0.005,
        "valid_max_com_regression_m": 0.001,
        "native_action_clip": 10.0,
    }
    assert clip["invalid_frames"] == [1, 2, 3, 4, 5, 6]
    assert clip["expert_valid_recomputed_matches_stored"] is True
    assert clip["integrity_passed"] is True
    assert clip["validity_failure_counts"] == {
        "weighted_regression": 1,
        "left_foot_position_threshold": 1,
        "left_foot_orientation_regression": 0,
        "right_foot_position_threshold": 0,
        "right_foot_orientation_regression": 1,
        "com_regression_threshold": 1,
        "constraint_relaxation": 1,
        "native_action_clip_exceeded": 1,
    }
    assert clip["cause_counts_on_invalid_frames"][
        "protected_priority_1_regression"
    ] == 0
    assert 7 not in clip["invalid_frames"]


def test_no_contact_transitions_uses_null_distances_not_nonfinite_values(
    tmp_path: Path,
) -> None:
    report_path, expert_path = _write_clip(tmp_path, no_contact_transitions=True)

    clip, _masks = audit.analyze_clip(report_path, expert_path)

    assert clip["contact_transitions"]["transition_count"] == 0
    assert clip["contact_transitions"]["invalid_nearest_distance_frames"] == {
        "min": None,
        "median": None,
        "p95": None,
        "max": None,
    }
    assert all(
        frame["nearest_contact_transition_distance_frames"] is None
        for frame in clip["invalid_frame_diagnostics"]
    )
    json.dumps(clip, allow_nan=False)


def test_pre_manifest_report_uses_exact_filename_as_clip_id(tmp_path: Path) -> None:
    report_path, expert_path = _write_clip(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["clip_id"]
    report_path.write_text(json.dumps(report), encoding="utf-8")

    clip, _masks = audit.analyze_clip(report_path, expert_path)

    assert clip["clip_id"] == "heldout_walk_001"


def test_legacy_artifact_without_stored_mask_is_recomputed_and_flagged(
    tmp_path: Path,
) -> None:
    report_path, expert_path = _write_clip(tmp_path)
    with np.load(expert_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "expert_valid"}
    np.savez(expert_path, **arrays)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["expert_output_sha256"] = _sha256(expert_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    clip, _masks = audit.analyze_clip(report_path, expert_path)

    assert clip["invalid_frames"] == [2, 3, 4, *range(8, 18)]
    assert clip["expert_valid_stored_present"] is False
    assert clip["expert_valid_recomputed_matches_stored"] is None
    assert clip["expert_valid_mismatch_frames"] == []
    assert clip["integrity_passed"] is False
