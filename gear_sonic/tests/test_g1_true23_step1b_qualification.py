from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from gear_sonic.scripts import validate_g1_true23_idle_step1b as cli
from gear_sonic.utils import g1_true23_step1b_qualification as step1b

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_TRUE23_ACTION_LAW = step1b.TRUE23_CONTROLLER_SEMANTICS


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(step1b.canonical_json_bytes(payload))


def _json_ref(root: Path, path: Path, payload: object) -> dict[str, object]:
    _write_json(path, payload)
    return {
        "file": path.relative_to(root).as_posix(),
        "sha256": step1b.sha256_file(path),
        "payload_sha256": step1b.sha256_bytes(step1b.canonical_json_bytes(payload)),
    }


def _binary_ref(root: Path, path: Path, payload: bytes = b"bound artifact\n") -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "file": path.relative_to(root).as_posix(),
        "sha256": step1b.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _repository_binding(relpath: str) -> dict[str, object]:
    path = REPOSITORY_ROOT / relpath
    return {
        "relpath": relpath,
        "sha256": step1b.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _action_law(arm: str, action_law: str, output_space: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": step1b.INPUT_ACTION_LAW_SCHEMA,
        "arm": arm,
        "action_law": action_law,
        "state_dependence": "live_robot_state_feedback",
        "input_fields": ["robot_state", "reference"],
        "output_space": output_space,
    }


def _step(index: int, joint_count: int, *, timeout: bool = False) -> dict[str, object]:
    zeros3 = [0.0, 0.0, 0.0]
    velocity = {"linear_mps": zeros3, "angular_radps": zeros3}
    return {
        "step_index": index,
        "terminated": False,
        "timed_out": timeout,
        "post_termination": False,
        "episode_instance_id": "",
        "reset_generation": 0,
        "sim_advanced": True,
        "reported_nonfinite": False,
        "reference_frame_index": 0,
        "joint_positions_rad": [0.0] * joint_count,
        "disturbance_delta": [0.0] * 6,
        "contacts": {"left_foot": True, "right_foot": True},
        "pelvis": {"position_m": zeros3, "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        "com_position_m": zeros3,
        "semantic_points_m": {
            "left_foot": zeros3,
            "right_foot": zeros3,
            "left_hand": zeros3,
            "right_hand": zeros3,
            "head_proxy": zeros3,
        },
        "semantic_orientations_xyzw": {
            "left_foot": [0.0, 0.0, 0.0, 1.0],
            "right_foot": [0.0, 0.0, 0.0, 1.0],
        },
        "link_velocities": {
            name: velocity
            for name in (
                "pelvis",
                "left_foot",
                "right_foot",
                "left_hand",
                "right_hand",
                "head_proxy",
            )
        },
    }


def _termination_terms(*, index: int, terminated: bool = False) -> dict[str, object]:
    anchor_error = 0.3 if terminated else 0.0
    return {
        "time_out": index == 499,
        "anchor_pos": terminated,
        "anchor_pos_error_z_m": anchor_error,
        "anchor_pos_threshold_m": 0.25,
        "anchor_ori": False,
        "anchor_ori_projected_gravity_z_error": 0.0,
        "anchor_ori_threshold": 0.8,
        "ee_body_pos": False,
        "ee_body_pos_error_z_m": {
            name: 0.0
            for name in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_roll_rubber_hand",
                "right_wrist_roll_rubber_hand",
            )
        },
        "ee_body_pos_threshold_m": 0.25,
        "terminated": terminated,
        "timed_out": index == 499 and not terminated,
    }


def _raw_step(
    index: int,
    joint_count: int,
    frame_index: int,
    source_frame_count: int,
    *,
    terminated: bool = False,
) -> dict[str, object]:
    qpos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, *([0.0] * joint_count)]
    qvel = [0.0] * (6 + joint_count)
    pose = {
        "measured_position_m": [0.0, 0.0, 0.3 if terminated else 0.0],
        "measured_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "reference_position_m": [0.0, 0.0, 0.0],
        "reference_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    ee_pose = {
        "measured_position_m": [0.0, 0.0, 0.0],
        "measured_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        "reference_position_m": [0.0, 0.0, 0.0],
        "reference_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    return {
        "step_index": index,
        "reference_frame_index": frame_index,
        "reference_terminal_held": frame_index == source_frame_count - 1,
        "sim_time_before_s": index / 50.0,
        "sim_time_after_s": (index + 1) / 50.0,
        "reset_generation": 0,
        "sim_advanced": True,
        "qpos_before": qpos.copy(),
        "qvel_before": qvel.copy(),
        "qpos_after": qpos.copy(),
        "qvel_after": qvel.copy(),
        "raw_action_native": [0.0] * joint_count,
        "safe_action_native": [0.0] * joint_count,
        "q_target_hardware_rad": [0.0] * joint_count,
        "disturbance_delta_world": [0.0] * 6,
        "pd_substeps_sha256": "0" * 64,
        "applied_torque_first_hardware_nm": [0.0] * joint_count,
        "applied_torque_last_hardware_nm": [0.0] * joint_count,
        "applied_torque_peak_abs_hardware_nm": [0.0] * joint_count,
        "raw_contact_geom_pairs": [[0, 1], [0, 2]],
        "torso_reference": pose,
        "ee_reference": {
            name: copy.deepcopy(ee_pose)
            for name in (
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_roll_rubber_hand",
                "right_wrist_roll_rubber_hand",
            )
        },
        "termination_terms": _termination_terms(index=index, terminated=terminated),
        "reported_nonfinite": False,
    }


def _early_terminate(steps: list[dict[str, object]], index: int = 100) -> None:
    steps[index]["terminated"] = True
    steps[-1]["timed_out"] = False
    for step in steps[index + 1 :]:
        step["post_termination"] = True


def _build_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    episodes_per_clip: int = 1,
    early_terminations: set[tuple[str, int]] | None = None,
) -> dict[str, object]:
    early_terminations = early_terminations or set()
    contract = json.loads(step1b.DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["episodes_per_clip"] = episodes_per_clip
    contract["controller_semantics"]["true23_expert"] = TEST_TRUE23_ACTION_LAW
    clips = contract["frozen_clips"]

    support = {
        "schema": "g1_23dof_task_space_supported_manifest_v1",
        "qualification_categories": ["idle"],
        "clips": [
            {
                "clip_id": clip["clip_id"],
                "category": "idle",
                "source_csv": clip["source_csv"],
                "fps_source": 120.0,
            }
            for clip in clips
        ],
    }
    qualification_clips = {
        clip_id: {
            "hard_gate_passed": True,
            "status_gate_passed": True,
            "hard_gate_failures": [],
            "status_gate_failures": [],
        }
        for clip_id in step1b.FROZEN_CLIP_IDS
    }
    qualification = {
        "schema": "g1_true23_task_space_heldout_qualification_v1",
        "schema_version": 1,
        "deployment_ready": False,
        "expert_gate_passed": False,
        "authorization": "step_1b_fixed_horizon_expert_collection_only",
        "step_1b_authorized": True,
        "qualification_gate_passed": True,
        "requested_state_passed": True,
        "declared_categories": ["idle"],
        "unique_requested_clip_count": 2,
        "hard_violation_count": 0,
        "categories": {"idle": {"gate_passed": True}},
        "clips": qualification_clips,
    }
    summaries = {clip_id: {"schema_version": 6} for clip_id in step1b.FROZEN_CLIP_IDS}
    batch = {
        "schema": "g1_true23_task_space_retarget_batch_v1",
        "deployment_ready": False,
        "input_count": 2,
        "completed_count": 2,
        "ok_count": 2,
        "skipped_count": 0,
        "pending_count": 0,
        "failed_count": 0,
        "rejected_count": 0,
        "all_complete": True,
        "clip_provenance": {
            clip["clip_id"]: {
                "source_csv": f"/root/source/{clip['source_csv']}",
                "category": "idle",
                "qualification_categories": ["idle"],
            }
            for clip in clips
        },
        "results": {
            clip_id: f"ok: {frame_count} frames"
            for clip_id, frame_count in zip(
                step1b.FROZEN_CLIP_IDS,
                step1b.FROZEN_FRAME_COUNTS,
                strict=True,
            )
        },
        "successful_summaries": summaries,
        "heldout_qualification": qualification,
    }
    support_ref = _json_ref(tmp_path, tmp_path / "step1a" / "support.json", support)
    qualification_ref = _json_ref(tmp_path, tmp_path / "step1a" / "qualification.json", qualification)
    batch_ref = _json_ref(tmp_path, tmp_path / "step1a" / "batch.json", batch)

    teacher_model = _binary_ref(
        tmp_path,
        tmp_path / "assets" / "teacher.xml",
        (REPOSITORY_ROOT / "gear_sonic/data/robots/g1/g1_29dof.xml").read_bytes(),
    )
    teacher_checkpoint = _binary_ref(tmp_path, tmp_path / "assets" / "teacher.pt", b"teacher checkpoint\n")
    teacher_config = _json_ref(
        tmp_path,
        tmp_path / "assets" / "teacher_config.json",
        {"schema_version": 1, "kind": "low_latency_policy_config"},
    )
    teacher_law = _json_ref(
        tmp_path,
        tmp_path / "assets" / "teacher_action_law.json",
        _action_law("teacher_reference", step1b.TEACHER_CONTROLLER_SEMANTICS, "native_il29_actions"),
    )
    semantic_reference = _json_ref(
        tmp_path,
        tmp_path / "assets" / "semantic_reference.json",
        {"schema_version": 1, "kind": "semantic_reference"},
    )
    teacher = {
        "kind": "exact_29dof_teacher_reference",
        "controller_semantics": step1b.TEACHER_CONTROLLER_SEMANTICS,
        "model": teacher_model,
        "checkpoint": teacher_checkpoint,
        "controller_config": teacher_config,
        "input_action_law": teacher_law,
        "semantic_reference": semantic_reference,
    }

    target_model = _binary_ref(
        tmp_path,
        tmp_path / "expert" / "true23.xml",
        (REPOSITORY_ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml").read_bytes(),
    )
    expert_controller = _binary_ref(
        tmp_path, tmp_path / "expert" / "controller.bin", b"state feedback controller\n"
    )
    expert_controller_config = _json_ref(
        tmp_path,
        tmp_path / "expert" / "controller_config.json",
        {
            "schema_version": 1,
            "kind": "state_feedback_controller_config",
            "resolved_config": {
                "controller_descriptor": step1b.TRUE23_CONTROLLER_DESCRIPTOR,
                "physics_gains": {
                    "kp": [1.0] * 23,
                    "kd": [1.0] * 23,
                    "effort_limits_nm": [10.0] * 23,
                },
            },
        },
    )
    expert_law = _json_ref(
        tmp_path,
        tmp_path / "expert" / "input_action_law.json",
        _action_law("true23_expert", TEST_TRUE23_ACTION_LAW, "joint_torque"),
    )
    expert_clips = []
    trusted_clip_bindings = {}
    for clip in clips:
        clip_id = clip["clip_id"]
        motion = _binary_ref(tmp_path, tmp_path / "expert" / f"{clip_id}.motion.npz")
        expert = _binary_ref(tmp_path, tmp_path / "expert" / f"{clip_id}.expert.npz")
        report_payload = {
            "schema_version": 6,
            "clip_id": clip_id,
            "category": "idle",
            "artifact_provenance_schema_version": 6,
            "source_model_sha256": teacher_model["sha256"],
            "target_model_sha256": target_model["sha256"],
            "motion_output_sha256": motion["sha256"],
            "expert_output_sha256": expert["sha256"],
            "serialization_constraint_audit_passed": True,
            "expert_gate_passed": False,
            "constraints": {"certificate_basis": "serialized_float32_position_arrays"},
        }
        report_ref = _json_ref(tmp_path, tmp_path / "expert" / f"{clip_id}.report.json", report_payload)
        expert_clips.append({"clip_id": clip_id, "motion": motion, "expert": expert, "report": report_ref})
        trusted_clip_bindings[clip_id] = {
            "motion_sha256": motion["sha256"],
            "expert_sha256": expert["sha256"],
            "report_payload_sha256": report_ref["payload_sha256"],
        }

    runner = _binary_ref(tmp_path, tmp_path / "runtime" / "runner.py", b"runner\n")
    resolved_config = {
        "fixed_horizon": True,
        "selected_gate": "mjlab_true23",
        "controller_descriptor": step1b.TRUE23_CONTROLLER_DESCRIPTOR,
    }
    runtime_config_payload = {
        "schema_version": 1,
        "kind": step1b.RUNTIME_CONFIG_SCHEMA,
        "control_hz": 50,
        "horizon_steps": 500,
        "post_termination_policy": step1b.POST_TERMINATION_POLICY,
        "shipped_end_effector_termination_threshold_m": 0.25,
        "teacher_controller_semantics": step1b.TEACHER_CONTROLLER_SEMANTICS,
        "true23_controller_semantics": TEST_TRUE23_ACTION_LAW,
        "resolved_config": resolved_config,
        "resolved_config_sha256": step1b.sha256_bytes(step1b.canonical_json_bytes(resolved_config)),
    }
    runtime_config = _json_ref(tmp_path, tmp_path / "runtime" / "resolved_config.json", runtime_config_payload)
    robot_asset = _binary_ref(
        tmp_path,
        tmp_path / "runtime" / "true23.xml",
        (REPOSITORY_ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml").read_bytes(),
    )
    termination_paths = contract["selected_termination_gate"]["config_relpaths"]
    runtime = {
        "simulator_name": "PinnedTestSimulator",
        "simulator_version": "1.0",
        "runner": runner,
        "runtime_config": runtime_config,
        "robot_assets": [{"role": "true23_model", "artifact": robot_asset}],
        "termination_configs": [_repository_binding(path) for path in termination_paths],
    }

    root_state = {
        "root_position_m": [0.0, 0.0, 0.0],
        "root_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "root_linear_velocity_mps": [0.0, 0.0, 0.0],
        "root_angular_velocity_radps": [0.0, 0.0, 0.0],
        "reference_frame_index": 0,
    }
    common_initial = _json_ref(
        tmp_path,
        tmp_path / "schedule" / "common_initial.json",
        {
            "schema_version": 1,
            "kind": step1b.INITIAL_STATE_SCHEMA,
            "scope": step1b.INITIAL_STATE_SCOPES["common"],
            "state": root_state,
        },
    )
    teacher_initial = _json_ref(
        tmp_path,
        tmp_path / "schedule" / "teacher_initial.json",
        {
            "schema_version": 1,
            "kind": step1b.INITIAL_STATE_SCHEMA,
            "scope": step1b.INITIAL_STATE_SCOPES["teacher_reference"],
            "state": {
                **root_state,
                "joint_names": list(step1b.MODEL_29_JOINT_NAMES),
                "joint_positions_rad": [0.0] * 29,
                "joint_velocities_radps": [0.0] * 29,
            },
        },
    )
    true23_initial = _json_ref(
        tmp_path,
        tmp_path / "schedule" / "true23_initial.json",
        {
            "schema_version": 1,
            "kind": step1b.INITIAL_STATE_SCHEMA,
            "scope": step1b.INITIAL_STATE_SCOPES["true23_expert"],
            "state": {
                **root_state,
                "joint_names": list(step1b.MODEL_23_JOINT_NAMES),
                "joint_positions_rad": [0.0] * 23,
                "joint_velocities_radps": [0.0] * 23,
            },
        },
    )
    disturbance = _json_ref(
        tmp_path,
        tmp_path / "schedule" / "disturbance.json",
        {
            "schema_version": 1,
            "kind": step1b.DISTURBANCE_SCHEMA,
            "control_hz": 50,
            "deltas": [[0.0] * 6 for _ in range(500)],
        },
    )
    reference_refs = {}
    reference_indices = {}
    for clip in clips:
        indices = [min(index, clip["source_frame_count"] - 1) for index in range(500)]
        reference_indices[clip["clip_id"]] = indices
        reference_refs[clip["clip_id"]] = _json_ref(
            tmp_path,
            tmp_path / "schedule" / f"{clip['clip_id']}.reference.json",
            {
                "schema_version": 1,
                "kind": step1b.REFERENCE_SCHEDULE_SCHEMA,
                "clip_id": clip["clip_id"],
                "source_frame_count": clip["source_frame_count"],
                "start_frame": 0,
                "continuation": "terminal_hold_last",
                "frame_indices": indices,
            },
        )

    schedule = []
    pair_payloads = []
    raw_payloads = []
    report_pairs = []
    for clip_index, clip_id in enumerate(step1b.FROZEN_CLIP_IDS):
        for episode in range(episodes_per_clip):
            pair_id = f"{clip_index:02d}-{episode:03d}"
            seed = contract["campaign_schedule"]["seeds_by_clip"][clip_id][episode]
            schedule.append(
                {
                    "pair_id": pair_id,
                    "clip_id": clip_id,
                    "seed": seed,
                    "reference_schedule": reference_refs[clip_id],
                    "common_initial_state": common_initial,
                    "teacher_initial_state": teacher_initial,
                    "true23_initial_state": true23_initial,
                    "disturbance_schedule": disturbance,
                }
            )
            refs = {}
            raw_refs = {}
            for arm, joint_count, arm_initial in (
                ("teacher_reference", 29, teacher_initial),
                ("true23_expert", 23, true23_initial),
            ):
                steps = [_step(index, joint_count, timeout=index == 499) for index in range(500)]
                for index, trace_step in enumerate(steps):
                    trace_step["episode_instance_id"] = pair_id
                    trace_step["reference_frame_index"] = reference_indices[clip_id][index]
                if (clip_id, episode) in early_terminations:
                    _early_terminate(steps)
                joint_names, joint_lower, joint_upper = step1b.MODEL_JOINT_SPECS[arm]
                trace = {
                    "schema_version": 1,
                    "kind": step1b.TRACE_SCHEMA,
                    "pair_id": pair_id,
                    "arm": arm,
                    "clip_id": clip_id,
                    "seed": seed,
                    "controller_semantics": (
                        step1b.TEACHER_CONTROLLER_SEMANTICS
                        if arm == "teacher_reference"
                        else TEST_TRUE23_ACTION_LAW
                    ),
                    "post_termination_policy": step1b.POST_TERMINATION_POLICY,
                    "episode_instance_id": pair_id,
                    "reset_generation": 0,
                    "reference_schedule_sha256": reference_refs[clip_id]["sha256"],
                    "reference_schedule_payload_sha256": reference_refs[clip_id]["payload_sha256"],
                    "common_initial_state_sha256": common_initial["sha256"],
                    "common_initial_state_payload_sha256": common_initial["payload_sha256"],
                    "arm_initial_state_sha256": arm_initial["sha256"],
                    "arm_initial_state_payload_sha256": arm_initial["payload_sha256"],
                    "disturbance_schedule_sha256": disturbance["sha256"],
                    "disturbance_schedule_payload_sha256": disturbance["payload_sha256"],
                    "control_hz": 50,
                    "horizon_steps": 500,
                    "joint_names": list(joint_names),
                    "hard_joint_position_lower_rad": list(joint_lower),
                    "hard_joint_position_upper_rad": list(joint_upper),
                    "record_count": 500,
                    "steps": steps,
                }
                path = tmp_path / "traces" / f"{pair_id}-{arm}.json"
                refs[arm] = _json_ref(tmp_path, path, trace)
                pair_payloads.append({"path": path, "payload": trace, "arm": arm})
                model_relpath = (
                    "gear_sonic/data/robots/g1/g1_29dof.xml"
                    if arm == "teacher_reference"
                    else "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml"
                )
                model_binding = _repository_binding(model_relpath)
                raw_steps = [
                    _raw_step(
                        index,
                        joint_count,
                        reference_indices[clip_id][index],
                        clips[clip_index]["source_frame_count"],
                        terminated=bool(steps[index]["terminated"]),
                    )
                    for index in range(500)
                ]
                terminal_seen = False
                for raw_step in raw_steps:
                    if terminal_seen:
                        raw_step["termination_terms"]["time_out"] = False
                        raw_step["termination_terms"]["timed_out"] = False
                    terminal_seen = terminal_seen or bool(raw_step["termination_terms"]["terminated"])
                raw_trace = {
                    "schema_version": 1,
                    "kind": step1b.RAW_AUDIT_SCHEMA,
                    "pair_id": pair_id,
                    "arm": arm,
                    "clip_id": clip_id,
                    "seed": seed,
                    "controller_semantics": trace["controller_semantics"],
                    "trace_sha256": refs[arm]["sha256"],
                    "trace_payload_sha256": refs[arm]["payload_sha256"],
                    "model": {
                        "repository_relpath": model_binding["relpath"],
                        "sha256": model_binding["sha256"],
                        "size_bytes": model_binding["size_bytes"],
                        "nq": 7 + joint_count,
                        "nv": 6 + joint_count,
                        "nu": joint_count,
                    },
                    "control_hz": 50,
                    "physics_hz": 500,
                    "horizon_steps": 500,
                    "episode_instance_id": pair_id,
                    "reset_generation": 0,
                    "joint_names": list(joint_names),
                    "hard_joint_position_lower_rad": list(joint_lower),
                    "hard_joint_position_upper_rad": list(joint_upper),
                    "actuator_names": list(joint_names),
                    "floor_geom_id": 0,
                    "geom_identities": [
                        {"geom_id": 0, "geom_name": "floor", "body_id": 0, "body_name": "world"},
                        {
                            "geom_id": 1,
                            "geom_name": "left_foot",
                            "body_id": 1,
                            "body_name": "left_ankle_roll_link",
                        },
                        {
                            "geom_id": 2,
                            "geom_name": "right_foot",
                            "body_id": 2,
                            "body_name": "right_ankle_roll_link",
                        },
                    ],
                    "record_count": 500,
                    "steps": raw_steps,
                }
                raw_path = tmp_path / "traces" / f"{pair_id}-{arm}.raw.json"
                raw_refs[arm] = _json_ref(tmp_path, raw_path, raw_trace)
                raw_payloads.append({"path": raw_path, "payload": raw_trace, "arm": arm})
            report_pairs.append(
                {
                    "pair_id": pair_id,
                    "clip_id": clip_id,
                    "teacher_trace": refs["teacher_reference"],
                    "true23_trace": refs["true23_expert"],
                    "teacher_raw_trace": raw_refs["teacher_reference"],
                    "true23_raw_trace": raw_refs["true23_expert"],
                }
            )

    contract["trusted_identity_bindings"] = {
        "step1a": {
            "support_manifest_sha256": support_ref["sha256"],
            "support_manifest_payload_sha256": support_ref["payload_sha256"],
            "qualification_sha256": qualification_ref["sha256"],
            "qualification_payload_sha256": qualification_ref["payload_sha256"],
            "batch_sha256": batch_ref["sha256"],
            "batch_payload_sha256": batch_ref["payload_sha256"],
        },
        "teacher_reference": {
            "model_sha256": teacher_model["sha256"],
            "checkpoint_sha256": teacher_checkpoint["sha256"],
            "controller_config_payload_sha256": teacher_config["payload_sha256"],
            "input_action_law_payload_sha256": teacher_law["payload_sha256"],
            "semantic_reference_payload_sha256": semantic_reference["payload_sha256"],
        },
        "true23_expert": {
            "target_model_sha256": target_model["sha256"],
            "controller_sha256": expert_controller["sha256"],
            "controller_config_payload_sha256": expert_controller_config["payload_sha256"],
            "input_action_law_payload_sha256": expert_law["payload_sha256"],
            "action_law": TEST_TRUE23_ACTION_LAW,
            "clips": trusted_clip_bindings,
        },
        "runtime": {
            "simulator_name": "PinnedTestSimulator",
            "simulator_version": "1.0",
            "runner_sha256": runner["sha256"],
            "runtime_config_payload_sha256": runtime_config["payload_sha256"],
            "robot_assets": {"true23_model": robot_asset["sha256"]},
        },
    }
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)
    contract_payload_sha256 = step1b.sha256_bytes(step1b.canonical_json_bytes(contract))
    monkeypatch.setattr(step1b, "PINNED_CONTRACT_PAYLOAD_SHA256", contract_payload_sha256)
    monkeypatch.setattr(step1b, "REQUIRED_EPISODES_PER_CLIP", episodes_per_clip)

    campaign = {
        "schema_version": 1,
        "kind": step1b.CAMPAIGN_SCHEMA,
        "declared_categories": ["idle"],
        "frozen_clip_ids": list(step1b.FROZEN_CLIP_IDS),
        "clip_substitutions": False,
        "row_masks": False,
        "step1a": {
            "support_manifest": support_ref,
            "qualification": qualification_ref,
            "batch": batch_ref,
        },
        "teacher_reference": teacher,
        "true23_expert": {
            "kind": "schema_v6_true23_expert",
            "controller_semantics": TEST_TRUE23_ACTION_LAW,
            "controller": expert_controller,
            "controller_config": expert_controller_config,
            "input_action_law": expert_law,
            "target_model": target_model,
            "clips": expert_clips,
        },
        "runtime": runtime,
        "episode_schedule": schedule,
    }
    campaign_path = tmp_path / "campaign.json"
    campaign_ref = _json_ref(tmp_path, campaign_path, campaign)
    report = {
        "schema_version": 1,
        "kind": step1b.REPORT_SCHEMA,
        "contract_sha256": step1b.sha256_file(contract_path),
        "contract_payload_sha256": contract_payload_sha256,
        "campaign_manifest": campaign_ref,
        "pairs": report_pairs,
    }
    report_path = tmp_path / "report.json"
    _write_json(report_path, report)
    return {
        "contract_path": contract_path,
        "contract": contract,
        "campaign_path": campaign_path,
        "campaign": campaign,
        "report_path": report_path,
        "report": report,
        "pair_payloads": pair_payloads,
        "raw_payloads": raw_payloads,
    }


def _rewrite_trace(bundle: dict[str, object], pair_index: int, arm: str) -> None:
    report = bundle["report"]
    pair = report["pairs"][pair_index]
    key = "teacher_trace" if arm == "teacher_reference" else "true23_trace"
    path = bundle["report_path"].parent / pair[key]["file"]
    payload = next(
        item["payload"] for item in bundle["pair_payloads"] if item["path"] == path and item["arm"] == arm
    )
    pair[key] = _json_ref(bundle["report_path"].parent, path, payload)
    raw_key = "teacher_raw_trace" if arm == "teacher_reference" else "true23_raw_trace"
    raw_path = bundle["report_path"].parent / pair[raw_key]["file"]
    raw_payload = next(
        item["payload"] for item in bundle["raw_payloads"] if item["path"] == raw_path and item["arm"] == arm
    )
    raw_payload["trace_sha256"] = pair[key]["sha256"]
    raw_payload["trace_payload_sha256"] = pair[key]["payload_sha256"]
    terminal_seen = False
    for index, trace_step in enumerate(payload.get("steps", [])):
        if index >= len(raw_payload["steps"]):
            break
        raw_step = raw_payload["steps"][index]
        if "joint_positions_rad" in trace_step:
            raw_step["qpos_after"][7:] = trace_step["joint_positions_rad"]
        if "contacts" in trace_step:
            pairs = []
            if trace_step["contacts"].get("left_foot"):
                pairs.append([0, 1])
            if trace_step["contacts"].get("right_foot"):
                pairs.append([0, 2])
            raw_step["raw_contact_geom_pairs"] = pairs
        terminated = bool(trace_step.get("terminated", False))
        raw_step["torso_reference"]["measured_position_m"][2] = 0.3 if terminated else 0.0
        raw_step["termination_terms"] = _termination_terms(index=index, terminated=terminated)
        if terminal_seen:
            raw_step["termination_terms"]["time_out"] = False
            raw_step["termination_terms"]["timed_out"] = False
        terminal_seen = terminal_seen or terminated
        if "reported_nonfinite" in trace_step:
            raw_step["reported_nonfinite"] = trace_step["reported_nonfinite"]
    pair[raw_key] = _json_ref(bundle["report_path"].parent, raw_path, raw_payload)
    _write_json(bundle["report_path"], report)


def _rewrite_raw_trace(bundle: dict[str, object], pair_index: int, arm: str) -> None:
    report = bundle["report"]
    pair = report["pairs"][pair_index]
    key = "teacher_raw_trace" if arm == "teacher_reference" else "true23_raw_trace"
    path = bundle["report_path"].parent / pair[key]["file"]
    payload = next(
        item["payload"] for item in bundle["raw_payloads"] if item["path"] == path and item["arm"] == arm
    )
    pair[key] = _json_ref(bundle["report_path"].parent, path, payload)
    _write_json(bundle["report_path"], report)


def _validate(bundle: dict[str, object]) -> dict[str, object]:
    return step1b.validate_step1b_report(bundle["report_path"], contract_path=bundle["contract_path"])


def test_pinned_production_contract_rejects_empty_evidence(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    _write_json(report, {})
    result = step1b.qualify_step1b_report_fail_closed(report)
    assert result["evidence_valid"] is False
    assert result["authorization"] == "none"
    assert "Step1B evidence report keys mismatch" in result["qualification_gate_failures"][0]


def test_complete_current_evidence_remains_diagnostic_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    result = _validate(bundle)
    assert result["qualification_gate_passed"] is False
    assert result["authorization"] == "none"
    assert any("independent MuJoCo FK/COM" in failure for failure in result["qualification_gate_failures"])
    assert result["dagger_authorized"] is False
    assert result["training_authorized"] is False
    assert result["deployment_ready"] is False
    assert result["diagnostics"]["com_position_error_m"]["gate_role"] == "diagnostic_non_gating"


def test_caller_controlled_contract_cannot_relax_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    relaxed = copy.deepcopy(bundle["contract"])
    relaxed["stance_foot_thresholds"]["position_error_max_m"] = 99.0
    _write_json(bundle["contract_path"], relaxed)
    with pytest.raises(ValueError, match="not the pinned contract"):
        _validate(bundle)


def test_stale_hash_rejected_and_wrapper_authorizes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    trace_path = bundle["pair_payloads"][0]["path"]
    trace_path.write_bytes(trace_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate(bundle)
    result = step1b.qualify_step1b_report_fail_closed(bundle["report_path"], contract_path=bundle["contract_path"])
    assert result["authorization"] == "none"
    assert result["training_authorized"] is False


@pytest.mark.parametrize("mutation", ["unknown", "missing"])
def test_unknown_and_missing_fields_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    step = bundle["pair_payloads"][0]["payload"]["steps"][0]
    if mutation == "unknown":
        step["invented"] = 1
    else:
        del step["contacts"]
    _rewrite_trace(bundle, 0, "teacher_reference")
    with pytest.raises(ValueError, match="keys mismatch"):
        _validate(bundle)


@pytest.mark.parametrize("length", [499, 501])
def test_trace_length_must_be_exactly_500(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, length: int) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    payload = bundle["pair_payloads"][0]["payload"]
    if length == 499:
        payload["steps"].pop()
    else:
        payload["steps"].append(copy.deepcopy(payload["steps"][-1]))
    payload["record_count"] = length
    _rewrite_trace(bundle, 0, "teacher_reference")
    with pytest.raises(ValueError, match="record_count|exactly 500"):
        _validate(bundle)


def test_noncontiguous_and_reset_trace_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    bundle["pair_payloads"][0]["payload"]["steps"][123]["step_index"] = 124
    _rewrite_trace(bundle, 0, "teacher_reference")
    with pytest.raises(ValueError, match="noncontiguous"):
        _validate(bundle)

    bundle = _build_bundle(tmp_path / "reset", monkeypatch)
    bundle["pair_payloads"][0]["payload"]["steps"][101]["reset_generation"] = 1
    _rewrite_trace(bundle, 0, "teacher_reference")
    with pytest.raises(ValueError, match="unauthorized reset"):
        _validate(bundle)


def test_schedule_and_pair_hash_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    payload = bundle["pair_payloads"][1]["payload"]
    payload["common_initial_state_sha256"] = "a" * 64
    _rewrite_trace(bundle, 0, "true23_expert")
    with pytest.raises(ValueError, match="paired campaign schedule"):
        _validate(bundle)

    bundle = _build_bundle(tmp_path / "disturbance", monkeypatch)
    payload = bundle["pair_payloads"][1]["payload"]
    payload["steps"][10]["disturbance_delta"][0] = 0.1
    _rewrite_trace(bundle, 0, "true23_expert")
    with pytest.raises(ValueError, match="immutable schedule"):
        _validate(bundle)


def test_runner_cannot_choose_seed_or_reference_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    bundle["campaign"]["episode_schedule"][0]["seed"] += 1
    campaign_ref = _json_ref(tmp_path, bundle["campaign_path"], bundle["campaign"])
    bundle["report"]["campaign_manifest"] = campaign_ref
    _write_json(bundle["report_path"], bundle["report"])
    with pytest.raises(ValueError, match="frozen schedule"):
        _validate(bundle)

    bundle = _build_bundle(tmp_path / "start", monkeypatch)
    reference_path = tmp_path / "start" / "schedule" / f"{step1b.FROZEN_CLIP_IDS[0]}.reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["start_frame"] = 1
    first_schedule = bundle["campaign"]["episode_schedule"][0]
    first_schedule["reference_schedule"] = _json_ref(tmp_path / "start", reference_path, reference)
    campaign_ref = _json_ref(tmp_path / "start", bundle["campaign_path"], bundle["campaign"])
    bundle["report"]["campaign_manifest"] = campaign_ref
    _write_json(bundle["report_path"], bundle["report"])
    with pytest.raises(ValueError, match="exactly zero"):
        _validate(bundle)


@pytest.mark.parametrize(
    ("mutation", "failure"),
    (
        ("hard_limit", "hard joint-limit"),
        ("nonfinite", "nonfinite"),
        ("contact", "contact mismatch"),
        ("foot_position", "position"),
        ("foot_orientation", "orientation"),
    ),
)
def test_limit_nonfinite_contact_and_foot_failures_gate_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    failure: str,
) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    step = bundle["pair_payloads"][1]["payload"]["steps"][10]
    if mutation == "hard_limit":
        step["joint_positions_rad"][0] = 4.0
    elif mutation == "nonfinite":
        step["reported_nonfinite"] = True
    elif mutation == "contact":
        step["contacts"]["left_foot"] = False
    elif mutation == "foot_position":
        step["semantic_points_m"]["left_foot"] = [0.006, 0.0, 0.0]
    else:
        angle = 0.006
        step["semantic_orientations_xyzw"]["left_foot"] = [
            0.0,
            0.0,
            math.sin(angle / 2.0),
            math.cos(angle / 2.0),
        ]
    _rewrite_trace(bundle, 0, "true23_expert")
    result = _validate(bundle)
    assert result["qualification_gate_passed"] is False
    assert result["authorization"] == "none"
    assert any(failure in item for item in result["qualification_gate_failures"])


def test_post_termination_must_continue_without_reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle(
        tmp_path,
        monkeypatch,
        early_terminations={(step1b.FROZEN_CLIP_IDS[0], 0)},
    )
    assert _validate(bundle)["qualification_gate_passed"] is False
    bundle["pair_payloads"][0]["payload"]["steps"][101]["post_termination"] = False
    _rewrite_trace(bundle, 0, "teacher_reference")
    with pytest.raises(ValueError, match="not latched"):
        _validate(bundle)


def test_path_escape_and_windows_drive_paths_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    bundle["report"]["campaign_manifest"]["file"] = "../campaign.json"
    _write_json(bundle["report_path"], bundle["report"])
    with pytest.raises(ValueError, match="normalized POSIX relative path"):
        _validate(bundle)

    bundle = _build_bundle(tmp_path / "drive", monkeypatch)
    bundle["report"]["campaign_manifest"]["file"] = "Z:campaign.json"
    _write_json(bundle["report_path"], bundle["report"])
    with pytest.raises(ValueError, match="normalized POSIX relative path"):
        _validate(bundle)


def test_open_loop_pd_replay_cannot_be_authorized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    bundle["campaign"]["true23_expert"]["controller_semantics"] = "source_trajectory_pd_replay"
    campaign_ref = _json_ref(tmp_path, bundle["campaign_path"], bundle["campaign"])
    bundle["report"]["campaign_manifest"] = campaign_ref
    _write_json(bundle["report_path"], bundle["report"])
    with pytest.raises(ValueError, match="trusted approved contract|diagnostic-only"):
        _validate(bundle)


def test_trace_cannot_self_supply_wide_joint_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    trace = bundle["pair_payloads"][1]["payload"]
    trace["hard_joint_position_lower_rad"] = [-100.0] * 23
    trace["hard_joint_position_upper_rad"] = [100.0] * 23
    _rewrite_trace(bundle, 0, "true23_expert")
    with pytest.raises(ValueError, match="bounds differ from pinned model"):
        _validate(bundle)


def test_raw_sidecar_must_link_exact_derived_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    bundle["raw_payloads"][1]["payload"]["trace_sha256"] = "f" * 64
    _rewrite_raw_trace(bundle, 0, "true23_expert")
    with pytest.raises(ValueError, match="differs from paired trace/campaign"):
        _validate(bundle)


@pytest.mark.parametrize("mutation", ["termination", "torque", "contact", "initial_qpos"])
def test_raw_sidecar_observables_are_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    raw = bundle["raw_payloads"][1]["payload"]
    if mutation == "termination":
        raw["steps"][10]["termination_terms"]["anchor_pos"] = True
        expected = "termination terms differ"
    elif mutation == "torque":
        raw["steps"][10]["applied_torque_first_hardware_nm"][0] = 1.0
        expected = "applied torque differs"
    elif mutation == "contact":
        raw["steps"][10]["raw_contact_geom_pairs"] = [[0, 2]]
        expected = "derived contact flags differ"
    else:
        raw["steps"][0]["qpos_before"][7] = 0.1
        expected = "differ from pinned initial state"
    _rewrite_raw_trace(bundle, 0, "true23_expert")
    with pytest.raises(ValueError, match=expected):
        _validate(bundle)


def test_controller_descriptor_is_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    config_ref = bundle["campaign"]["true23_expert"]["controller_config"]
    config_path = tmp_path / config_ref["file"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["resolved_config"]["controller_descriptor"]["feedback_rate"] = "once_per_control_step"
    bundle["campaign"]["true23_expert"]["controller_config"] = _json_ref(tmp_path, config_path, config)
    campaign_ref = _json_ref(tmp_path, bundle["campaign_path"], bundle["campaign"])
    bundle["report"]["campaign_manifest"] = campaign_ref
    _write_json(bundle["report_path"], bundle["report"])
    with pytest.raises(ValueError, match="descriptor differs"):
        _validate(bundle)


def test_cli_never_emits_training_or_deployment_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _build_bundle(tmp_path, monkeypatch)
    output = tmp_path / "qualification.json"
    assert (
        cli.main(
            [
                "--report",
                str(bundle["report_path"]),
                "--contract",
                str(bundle["contract_path"]),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(output.read_text(encoding="utf-8"))
    assert printed["authorization"] == "none"
    assert printed["dagger_authorized"] is False
    assert printed["training_authorized"] is False
    assert printed["deployment_ready"] is False
