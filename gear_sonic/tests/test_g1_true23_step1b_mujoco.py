from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils import g1_true23_step1b_mujoco as runner
from gear_sonic.utils.g1_23dof_contract import SOURCE_IL29_JOINT_NAMES
from gear_sonic.utils.g1_29dof_low_latency_teacher import (
    DECODER_DIMS,
    ENCODER_DIMS,
    exact_fsq32,
)
from gear_sonic.utils.g1_true23_step1b_qualification import (
    CAMPAIGN_SCHEMA,
    DEFAULT_CONTRACT_PATH,
    FROZEN_CLIP_IDS,
    POST_TERMINATION_POLICY,
    REPORT_SCHEMA,
    TRUE23_CONTROLLER_DESCRIPTOR,
    TRUE23_CONTROLLER_SEMANTICS,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _clip(frame_count: int = 12) -> runner.LoadedClip:
    q = np.arange(frame_count * 29, dtype=np.float64).reshape(frame_count, 29) / 100.0
    dq = np.arange(frame_count * 29, dtype=np.float64).reshape(frame_count, 29) / 10.0
    return runner.LoadedClip(
        clip_id="clip",
        source_csv_path=Path("source.csv"),
        motion_path=Path("motion.npz"),
        expert_path=Path("expert.npz"),
        report_path=Path("report.json"),
        source_root_pos=np.zeros((frame_count, 3)),
        source_root_quat_wxyz=np.tile((1.0, 0.0, 0.0, 0.0), (frame_count, 1)),
        source_joint_pos_hardware=q,
        source_joint_vel_hardware=dq,
        target_root_pos=np.zeros((frame_count, 3)),
        target_root_quat_wxyz=np.tile((1.0, 0.0, 0.0, 0.0), (frame_count, 1)),
        target_joint_pos_hardware=np.zeros((frame_count, 23)),
        target_joint_vel_hardware=np.zeros((frame_count, 23)),
        target_action_native=np.zeros((frame_count, 23), dtype=np.float32),
        target_contact_flags=np.ones((frame_count, 2), dtype=np.bool_),
        target_body_pos_w=np.zeros((frame_count, 1, 3)),
        target_body_quat_wxyz=np.tile((1.0, 0.0, 0.0, 0.0), (frame_count, 1, 1)),
        frame_count=frame_count,
    )


def test_pinned_model_identity_and_hardware_order() -> None:
    source_path = REPOSITORY_ROOT / runner.SOURCE_MODEL_RELPATH
    target_path = REPOSITORY_ROOT / runner.TARGET_MODEL_RELPATH
    assert runner.sha256_file(source_path) == runner.SOURCE_MODEL_SHA256
    assert runner.sha256_file(target_path) == runner.TARGET_MODEL_SHA256
    assert runner.SOURCE_HARDWARE_JOINT_NAMES[:6] == (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    )
    assert (
        tuple(runner.SOURCE_HARDWARE_JOINT_NAMES[index] for index in runner.IL29_TO_HARDWARE29)
        == SOURCE_IL29_JOINT_NAMES
    )


def test_teacher_topology_fsq_and_term_major_history() -> None:
    assert ENCODER_DIMS == (267, 2048, 1024, 512, 512, 64)
    assert DECODER_DIMS == (994, 4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 29)
    torch = pytest.importorskip("torch")
    latent = torch.tensor([[0.0] * 64], dtype=torch.float32)
    token = exact_fsq32(latent)
    assert token.shape == (1, 64)
    assert torch.all(token == 0.0)

    frames = np.arange(10 * 93, dtype=np.float64).reshape(10, 93)
    flattened = runner.term_major_history(frames)
    np.testing.assert_array_equal(flattened[:30], frames[:, :3].reshape(-1))
    np.testing.assert_array_equal(flattened[30:320], frames[:, 3:32].reshape(-1))
    np.testing.assert_array_equal(flattened[320:610], frames[:, 32:61].reshape(-1))
    np.testing.assert_array_equal(flattened[610:900], frames[:, 61:90].reshape(-1))
    np.testing.assert_array_equal(flattened[900:], frames[:, 90:93].reshape(-1))


def test_encoder_267_uses_source29_future_order_and_terminal_hold() -> None:
    clip = _clip()
    reference = object.__new__(runner.ReferenceKinematics)
    reference.clip = clip
    reference._source_positions = {
        "left_wrist_yaw_link": np.zeros((clip.frame_count, 3)),
        "right_wrist_yaw_link": np.zeros((clip.frame_count, 3)),
        "torso_link": np.zeros((clip.frame_count, 3)),
    }
    reference._source_quaternions = {
        name: np.tile((1.0, 0.0, 0.0, 0.0), (clip.frame_count, 1)) for name in reference._source_positions
    }
    value = reference.teleop_encoder_input(10, (1.0, 0.0, 0.0, 0.0))
    assert value.shape == (267,)
    indices = np.asarray((10, 11, 11, 11, 11, 11, 11, 11, 11, 11))
    native_q = clip.source_joint_pos_hardware[:, runner.IL29_TO_HARDWARE29]
    native_dq = clip.source_joint_vel_hardware[:, runner.IL29_TO_HARDWARE29]
    lower = np.asarray(runner.LOWER_BODY_IL29_INDICES)
    np.testing.assert_array_equal(value[:120], native_q[indices][:, lower].reshape(-1).astype(np.float32))
    np.testing.assert_array_equal(value[120:240], native_dq[indices][:, lower].reshape(-1).astype(np.float32))
    assert not np.shares_memory(value, clip.source_joint_pos_hardware)


def test_terminal_hold_has_no_loop_or_seam_and_zero_terminal_velocity() -> None:
    assert runner.terminal_hold_indices(3, 7) == (0, 1, 2, 2, 2, 2, 2)
    values = np.asarray(((0.0, 2.0), (1.0, 4.0), (3.0, 9.0)))
    velocity = runner.terminal_hold_forward_velocity(values, 50)
    np.testing.assert_array_equal(velocity[0], (50.0, 100.0))
    np.testing.assert_array_equal(velocity[1], (100.0, 250.0))
    np.testing.assert_array_equal(velocity[-1], (0.0, 0.0))


def test_teacher_angular_velocity_is_rotated_into_body_frame() -> None:
    half = np.sqrt(0.5)
    body = runner.world_angular_velocity_to_body((half, 0.0, 0.0, half), (1.0, 0.0, 0.0))
    np.testing.assert_allclose(body, (0.0, -1.0, 0.0), atol=1.0e-15)


def test_schedule_is_deterministic_shared_and_nominal_without_contract_variation() -> None:
    first = runner.deterministic_episode_specs("clip", (1729, 1730))
    second = runner.deterministic_episode_specs("clip", (1729, 1730))
    assert first == second
    assert all(spec.root_position_delta_m == (0.0, 0.0, 0.0) for spec in first)
    assert all(spec.joint_delta_retained_hardware_rad == (0.0,) * 23 for spec in first)
    varied = runner.deterministic_episode_specs("clip", (1729, 1730), enable_initial_perturbation=True)
    assert varied[0].root_position_delta_m == (0.0, 0.0, 0.0)
    assert varied[1].root_position_delta_m != (0.0, 0.0, 0.0)
    assert len(set(spec.pair_id for spec in first)) == 2


def test_live_pd_label_and_descriptor_cannot_imply_online_ik() -> None:
    assert runner.true23_controller_descriptor() == TRUE23_CONTROLLER_DESCRIPTOR
    assert runner.true23_controller_descriptor() == {
        "action_law": "schema_v6_action_target_native_with_live_joint_pd_v1",
        "state_inputs": ["joint_position", "joint_velocity"],
        "reference_input": "schema_v6_action_target_native",
        "feedback_rate": "every_physics_substep",
        "torque_law": "clip(kp*(q_target-q)-kd*dq,effort_limits)",
        "output_space": "joint_torque",
        "online_task_space_expert": False,
    }
    law = runner.build_input_action_law("true23_expert")
    assert law["action_law"] == TRUE23_CONTROLLER_SEMANTICS
    assert law["state_dependence"] == "live_robot_state_feedback"
    assert law["output_space"] == "joint_torque"


def test_shipped_termination_uses_strict_exact_thresholds() -> None:
    ee_names = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand",
    )
    zeros = {name: (0.0, 0.0, 0.0) for name in ee_names}
    at_threshold = runner.evaluate_shipped_termination(
        measured_torso_position=(0.0, 0.0, 0.25),
        measured_torso_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        reference_torso_position=(0.0, 0.0, 0.0),
        reference_torso_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        measured_ee_positions=zeros,
        reference_ee_positions=zeros,
    )
    assert at_threshold["terminated"] is False
    above = dict(zeros)
    above[ee_names[0]] = (0.0, 0.0, 0.2500001)
    failed = runner.evaluate_shipped_termination(
        measured_torso_position=(0.0, 0.0, 0.0),
        measured_torso_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        reference_torso_position=(0.0, 0.0, 0.0),
        reference_torso_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        measured_ee_positions=above,
        reference_ee_positions=zeros,
    )
    assert failed["ee_body_pos"] is True
    assert failed["terminated"] is True


def test_shipped_orientation_term_preserves_raw_quaternion_arithmetic() -> None:
    ee_names = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_roll_rubber_hand",
        "right_wrist_roll_rubber_hand",
    )
    zeros = {name: (0.0, 0.0, 0.0) for name in ee_names}
    measured = (0.999995, 0.001, 0.002, 0.0)
    reference = (0.999994, 0.0015, 0.0025, 0.0)
    result = runner.evaluate_shipped_termination(
        measured_torso_position=(0.0, 0.0, 0.0),
        measured_torso_quaternion_wxyz=measured,
        reference_torso_position=(0.0, 0.0, 0.0),
        reference_torso_quaternion_wxyz=reference,
        measured_ee_positions=zeros,
        reference_ee_positions=zeros,
    )

    def projected_z(value: tuple[float, ...]) -> float:
        return -(1.0 - 2.0 * (value[1] ** 2 + value[2] ** 2))

    assert result["anchor_ori_projected_gravity_z_error"] == abs(projected_z(reference) - projected_z(measured))


def test_terminal_latch_continues_fixed_horizon_without_reset() -> None:
    event = runner.terminal_event_flags(already_latched=False, gate_terminated=True, final_step=False)
    assert event == (True, False, False, True)
    after = runner.terminal_event_flags(already_latched=event[-1], gate_terminated=True, final_step=False)
    assert after == (False, False, True, True)
    timeout = runner.terminal_event_flags(already_latched=False, gate_terminated=False, final_step=True)
    assert timeout == (False, True, False, True)
    concurrent = runner.terminal_event_flags(already_latched=False, gate_terminated=True, final_step=True)
    assert concurrent == (True, False, False, True)
    assert runner.shipped_time_out_term(already_latched=False, final_step=True) is True
    assert POST_TERMINATION_POLICY == "physics_no_reset_v1"


def test_trace_topology_carries_reset_and_raw_recomputation_inputs() -> None:
    assert "reset_generation" in runner.TRACE_HEADER_KEYS
    assert "sim_advanced" in runner.TRACE_STEP_KEYS
    assert "qpos_before" in runner.RAW_STEP_KEYS
    assert "qvel_before" in runner.RAW_STEP_KEYS
    assert "q_target_hardware_rad" in runner.RAW_STEP_KEYS
    assert "pd_substeps_sha256" in runner.RAW_STEP_KEYS
    assert "applied_torque_first_hardware_nm" in runner.RAW_STEP_KEYS
    assert "raw_contact_geom_pairs" in runner.RAW_STEP_KEYS
    assert "termination_terms" in runner.RAW_STEP_KEYS
    assert runner.RAW_TRACE_KIND == "g1_true23_idle_step1b_raw_trace_v1"


def test_full_shape_and_validator_envelope_topology(tmp_path: Path) -> None:
    assert runner._is_qualification_shape(
        tuple(
            runner.FrozenClipPaths(
                clip_id=clip_id,
                source_csv=tmp_path / f"{clip_id}.csv",
                motion=tmp_path / "step1a" / "motions" / f"{clip_id}.npz",
                expert=tmp_path / "step1a" / "experts" / f"{clip_id}.npz",
                report=tmp_path / "step1a" / "reports" / f"{clip_id}.json",
                expected_frame_count=1,
            )
            for clip_id in FROZEN_CLIP_IDS
        ),
        1,
        500,
    )

    root = tmp_path / "evidence"
    root.mkdir()
    step1a = tmp_path / "step1a"
    (step1a / "motions").mkdir(parents=True)
    (step1a / "experts").mkdir()
    (step1a / "reports").mkdir()
    (step1a / "qualification.json").write_text("{}", encoding="utf-8")
    (step1a / "batch.json").write_text("{}", encoding="utf-8")
    clips = []
    for clip_id in FROZEN_CLIP_IDS:
        motion = step1a / "motions" / f"{clip_id}.npz"
        expert = step1a / "experts" / f"{clip_id}.task_space.npz"
        report = step1a / "reports" / f"{clip_id}.retarget.json"
        motion.write_bytes(b"motion")
        expert.write_bytes(b"expert")
        report.write_text('{"schema_version":6}', encoding="utf-8")
        clips.append(
            runner.FrozenClipPaths(
                clip_id=clip_id,
                source_csv=tmp_path / f"{clip_id}.csv",
                motion=motion,
                expert=expert,
                report=report,
                expected_frame_count=1,
            )
        )
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    source = runner.source29_physics_parameters()
    target = runner.PhysicsParameters(
        timestep_s=0.002,
        decimation=10,
        kp=np.ones(23),
        kd=np.ones(23),
        effort=np.ones(23),
        armature=np.ones(23),
        action_scale=None,
    )
    controller_payloads = {
        "teacher_controller_config": {
            "schema_version": 1,
            "kind": "low_latency_policy_config",
            "resolved_config": {"exact": True},
        },
        "teacher_input_action_law": runner.build_input_action_law("teacher_reference"),
        "teacher_semantic_reference": {"schema_version": 1},
        "true23_controller_config": {
            "schema_version": 1,
            "kind": "state_feedback_controller_config",
            "resolved_config": {
                "controller_descriptor": runner.true23_controller_descriptor(),
                "physics_gains": {
                    "kp": [1.0] * 23,
                    "kd": [1.0] * 23,
                    "effort_limits_nm": [1.0] * 23,
                },
            },
        },
        "true23_input_action_law": runner.build_input_action_law("true23_expert"),
    }
    schedule = []
    pairs = []
    for index, clip_id in enumerate(FROZEN_CLIP_IDS):
        ref = {"file": f"schedule/{index}.json", "sha256": "0" * 64, "payload_sha256": "1" * 64}
        schedule.append(
            {
                "pair_id": f"pair-{index}",
                "clip_id": clip_id,
                "seed": (1729, 2729)[index],
                "reference_schedule": ref,
                "common_initial_state": ref,
                "teacher_initial_state": ref,
                "true23_initial_state": ref,
                "disturbance_schedule": ref,
            }
        )
        pairs.append(
            {
                "pair_id": f"pair-{index}",
                "clip_id": clip_id,
                "teacher_trace": ref,
                "true23_trace": ref,
                "teacher_raw_trace": ref,
                "true23_raw_trace": ref,
            }
        )

    report_path, campaign_ref, pin_ref, _ = runner._write_qualification_envelope(
        root=root,
        clip_paths=clips,
        checkpoint_path=checkpoint,
        contract_path=DEFAULT_CONTRACT_PATH,
        source_model_path=REPOSITORY_ROOT / runner.SOURCE_MODEL_RELPATH,
        target_model_path=REPOSITORY_ROOT / runner.TARGET_MODEL_RELPATH,
        sim_config_path=REPOSITORY_ROOT / runner.SIM_CONFIG_RELPATH,
        simulator_version="test",
        device="cpu",
        source_physics=source,
        true23_physics=target,
        controller_payloads=controller_payloads,
        schedule_records=schedule,
        pair_records=pairs,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    campaign = json.loads((root / campaign_ref["file"]).read_text(encoding="utf-8"))
    assert set(report) == {
        "schema_version",
        "kind",
        "contract_sha256",
        "contract_payload_sha256",
        "campaign_manifest",
        "pairs",
    }
    assert report["kind"] == REPORT_SCHEMA
    assert set(campaign) == {
        "schema_version",
        "kind",
        "declared_categories",
        "frozen_clip_ids",
        "clip_substitutions",
        "row_masks",
        "step1a",
        "teacher_reference",
        "true23_expert",
        "runtime",
        "episode_schedule",
    }
    assert campaign["kind"] == CAMPAIGN_SCHEMA
    assert set(campaign["teacher_reference"]) == {
        "kind",
        "controller_semantics",
        "model",
        "checkpoint",
        "controller_config",
        "input_action_law",
        "semantic_reference",
    }
    assert set(campaign["true23_expert"]) == {
        "kind",
        "controller_semantics",
        "controller",
        "controller_config",
        "input_action_law",
        "target_model",
        "clips",
    }
    assert all(
        "file" in ref and "path" not in ref
        for ref in (
            campaign["teacher_reference"]["model"],
            campaign["teacher_reference"]["controller_config"],
            campaign["true23_expert"]["target_model"],
            campaign["runtime"]["runtime_config"],
        )
    )
    assert (root / pin_ref["file"]).is_file()


def test_source29_physics_matches_released_actuator_classes() -> None:
    physics = runner.source29_physics_parameters()
    assert physics.timestep_s == 0.005
    assert physics.decimation == 4
    assert physics.kp.shape == (29,)
    # Hip pitch is 7520-22; ankle pitch is doubled 5020; wrist yaw is 4010.
    assert physics.kp[0] == pytest.approx(99.09842777666113)
    assert physics.kp[4] == pytest.approx(28.50124619574858)
    assert physics.kp[21] == pytest.approx(16.77832748089279)
    assert physics.action_scale[4] == pytest.approx(0.43857731392336724)
    assert physics.effort[4] == 50.0
    assert physics.effort[21] == 5.0


def test_source29_compiled_model_identity_when_mujoco_available() -> None:
    pytest.importorskip("mujoco")
    _, model, physics = runner.prepare_source29_model(REPOSITORY_ROOT / runner.SOURCE_MODEL_RELPATH)
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    assert runner._model_joint_names(model) == runner.SOURCE_HARDWARE_JOINT_NAMES
    assert round(1.0 / model.opt.timestep) == runner.TEACHER_PHYSICS_HZ
    assert physics.decimation == runner.TEACHER_DECIMATION
