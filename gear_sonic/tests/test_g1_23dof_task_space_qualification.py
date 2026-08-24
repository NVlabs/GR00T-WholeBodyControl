from __future__ import annotations

import json

import pytest

from gear_sonic.utils.g1_23dof_task_space_qualification import (
    QUALIFICATION_CATEGORIES,
    build_task_space_qualification_report,
)


def _summary(
    *,
    frame_count: int = 250,
    cost_before: float = 1.0,
    cost_after: float = 0.85,
    hand_before: float = 0.10,
    hand_after: float = 0.07,
) -> dict[str, object]:
    def task(before: float, after: float, *, maximum: float | None = None):
        result = {
            "position_error_before_mean_m": before,
            "position_error_after_mean_m": after,
        }
        if maximum is not None:
            result["position_error_after_max_m"] = maximum
        return result

    return {
        "frame_count": frame_count,
        "kinematic_gate_passed": True,
        "expert_valid_frame_fraction": 0.98,
        "weighted_task_error_before_mean": cost_before,
        "weighted_task_error_after_mean": cost_after,
        "position_limit_hit_count": 0,
        "hard_position_limit_violation_count": 0,
        "trajectory_constraint_relaxation_count": 0,
        "native_action_abs_gt_ten_fraction": 0.0,
        "retiming": {"time_scale": 1.0},
        "constraints": {
            "max_velocity_rad_s": 8.0,
            "max_acceleration_rad_s2": 80.0,
            "measured_velocity_abs_max_rad_s": 7.5,
            "measured_acceleration_abs_max_rad_s2": 79.0,
        },
        "per_task_error": {
            "left_foot": task(0.0, 0.0, maximum=0.004),
            "right_foot": task(0.0, 0.0, maximum=0.003),
            "whole_robot_com": task(0.020, 0.019),
            "left_hand": task(hand_before, hand_after),
            "right_hand": task(hand_before, hand_after),
        },
    }


def _passing_inputs():
    summaries = {}
    metadata = {}
    statuses = {}
    requested = []
    for category in QUALIFICATION_CATEGORIES:
        for index in range(2):
            clip_id = f"{category}_{index}"
            summaries[clip_id] = _summary()
            metadata[clip_id] = {
                "category": category,
                "independence_key": f"source/{clip_id}.csv",
            }
            statuses[clip_id] = {
                "completed": True,
                "fresh": True,
                "ok": True,
                "skipped": False,
            }
            requested.append(clip_id)
    return summaries, metadata, statuses, requested


def test_passing_report_is_frame_weighted_and_only_authorizes_step_1b():
    summaries, metadata, statuses, requested = _passing_inputs()
    # Force an unequal-weight aggregate whose answer differs from clip averaging.
    summaries["idle_0"] = _summary(
        frame_count=750,
        cost_before=2.0,
        cost_after=1.4,
        hand_before=0.12,
        hand_after=0.06,
    )

    report = build_task_space_qualification_report(
        summaries, metadata, statuses, requested
    )

    expected_frames = 750 + 11 * 250
    expected_before = (750 * 2.0 + 11 * 250 * 1.0) / expected_frames
    expected_after = (750 * 1.4 + 11 * 250 * 0.85) / expected_frames
    assert report["metrics"]["frame_count"] == expected_frames
    assert report["metrics"]["weighted_task_error_before_mean"] == pytest.approx(
        expected_before
    )
    assert report["metrics"]["weighted_task_error_after_mean"] == pytest.approx(
        expected_after
    )
    assert report["metrics"]["cost_improvement"] == pytest.approx(
        (expected_before - expected_after) / expected_before
    )
    assert report["qualification_gate_passed"] is True
    assert report["step_1b_authorized"] is True
    assert report["authorization"] == "step_1b_fixed_horizon_expert_collection_only"
    assert report["expert_gate_passed"] is False
    assert report["deployment_ready"] is False
    json.dumps(report)


def test_absent_declared_categories_keeps_all_six_categories() -> None:
    summaries, metadata, statuses, requested = _passing_inputs()

    report = build_task_space_qualification_report(
        summaries, metadata, statuses, requested
    )

    assert report["declared_categories"] == list(QUALIFICATION_CATEGORIES)
    assert list(report["categories"]) == list(QUALIFICATION_CATEGORIES)
    assert report["qualification_gate_passed"] is True


def test_idle_only_declaration_qualifies_idle_clips() -> None:
    summaries, metadata, statuses, requested = _passing_inputs()
    requested = [clip_id for clip_id in requested if clip_id.startswith("idle_")]

    report = build_task_space_qualification_report(
        summaries,
        metadata,
        statuses,
        requested,
        declared_categories=["idle"],
    )

    assert report["declared_categories"] == ["idle"]
    assert list(report["categories"]) == ["idle"]
    assert report["qualification_gate_passed"] is True


def test_requested_undeclared_category_fails_qualification() -> None:
    summaries, metadata, statuses, requested = _passing_inputs()
    requested = ["idle_0", "idle_1", "walk_0"]

    report = build_task_space_qualification_report(
        summaries,
        metadata,
        statuses,
        requested,
        declared_categories=["idle"],
    )

    assert (
        "metadata.category 'walk' is outside declared_categories"
        in report["clips"]["walk_0"]["hard_gate_failures"]
    )
    assert report["qualification_gate_passed"] is False


@pytest.mark.parametrize("declared_categories", [[], ["idle", "idle"], ["nope"]])
def test_invalid_declared_categories_are_rejected(
    declared_categories: list[str],
) -> None:
    summaries, metadata, statuses, requested = _passing_inputs()

    with pytest.raises(ValueError, match="declared_categories"):
        build_task_space_qualification_report(
            summaries,
            metadata,
            statuses,
            requested,
            declared_categories=declared_categories,
        )


def test_missing_metric_and_stale_skipped_status_fail_explicitly():
    summaries, metadata, statuses, requested = _passing_inputs()
    del summaries["walk_0"]["per_task_error"]["left_foot"][
        "position_error_after_max_m"
    ]
    statuses["walk_0"] = {
        "completed": True,
        "fresh": False,
        "ok": True,
        "skipped": True,
    }

    report = build_task_space_qualification_report(
        summaries, metadata, statuses, requested
    )

    clip = report["clips"]["walk_0"]
    assert clip["hard_gate_passed"] is False
    assert (
        "missing summary.per_task_error.left_foot.position_error_after_max_m"
        in clip["hard_gate_failures"]
    )
    assert clip["status_gate_passed"] is False
    assert "status.fresh must be true" in clip["status_gate_failures"]
    assert (
        "status.skipped must be false for first qualification"
        in clip["status_gate_failures"]
    )
    assert report["categories"]["walk"]["gate_passed"] is False
    assert report["requested_state_passed"] is False
    assert report["hard_violation_count"] > 0
    assert report["qualification_gate_passed"] is False
    assert report["step_1b_authorized"] is False
    assert report["authorization"] == "none"
    assert report["expert_gate_passed"] is False
    assert any(
        "not all requested clips completed fresh" in failure
        for failure in report["qualification_gate_failures"]
    )
    json.dumps(report)


def test_large_time_dilation_cannot_qualify_original_speed_expert() -> None:
    summaries, metadata, statuses, requested = _passing_inputs()
    summaries["walk_0"]["retiming"] = {"time_scale": 6.2}

    report = build_task_space_qualification_report(
        summaries, metadata, statuses, requested
    )

    failures = report["clips"]["walk_0"]["hard_gate_failures"]
    assert any("time_scale 6.2" in failure for failure in failures)
    assert report["qualification_gate_passed"] is False
    assert report["step_1b_authorized"] is False
    assert report["authorization"] == "none"
    assert report["expert_gate_passed"] is False
    json.dumps(report)


def test_missing_hand_task_is_not_treated_as_no_task():
    summaries, metadata, statuses, requested = _passing_inputs()
    del summaries["reach_lift_1"]["per_task_error"]["right_hand"]

    report = build_task_space_qualification_report(
        summaries, metadata, statuses, requested
    )

    failures = report["clips"]["reach_lift_1"]["hard_gate_failures"]
    assert "missing or invalid summary.per_task_error.right_hand: expected object" in failures
    assert report["categories"]["reach_lift"]["gate_passed"] is False
    assert report["qualification_gate_passed"] is False
