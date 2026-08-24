from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    BalancedUpperBodyTrue23MujocoController,
    CleanSonicPolicy,
    CleanTrue23MujocoController,
    SupervisedCleanTrue23MujocoController,
    UnitreeZeroVelocityFallbackPolicy,
    encoder267_from_reference,
    motion_reference_terms,
    run_balanced_upper_body_reference_sequence,
    run_motion_replay,
    validate_reference_terms,
)
from gear_sonic.utils.g1_true23_step1b_mujoco import _quaternion_matrix

ROOT = Path(__file__).resolve().parents[2]
MOTION = ROOT / "external_dependencies/unitree_rl_mjlab/src/assets/motions/g1_23dof/B_DadDance.npz"
CAPTURE = ROOT / "artifacts/g1_true23/q200_rank256_increment_v1/seed835_clean_q409_recovery_capture_v2.npz"
STEP_TOUCH = ROOT / (
    "artifacts/g1_native124_multimotion/scaling_all61/retimed_v2/npz/J_Dance0_StepTouch_slow2.npz"
)


def _motion() -> dict[str, np.ndarray]:
    with np.load(MOTION, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def test_motion_reference_reproduces_clean_mjlab_encoder267_row0() -> None:
    motion = _motion()
    packet = motion_reference_terms(motion, 9)
    summary = validate_reference_terms(packet)
    assert summary["anchor_index"] == 9
    encoder267 = encoder267_from_reference(packet, motion["body_quat_w"][9, 0])
    with np.load(CAPTURE, allow_pickle=False) as capture:
        expected = capture["encoder267"][0]
    assert np.max(np.abs(encoder267 - expected)) <= 1.2e-7


def test_reference_validator_rejects_joint_proof_and_bool_alias() -> None:
    packet = motion_reference_terms(_motion(), 9)
    bad = copy.deepcopy(packet)
    bad["proof_joint_pos_il29"][0] += 0.01
    with pytest.raises(ValueError, match="q9/q10 proof"):
        validate_reference_terms(bad)
    bad = copy.deepcopy(packet)
    bad["control_source_frame_index"] = True
    with pytest.raises(ValueError, match="exact integers"):
        validate_reference_terms(bad)


def test_clean_student_completes_independent_cpu_mujoco_replay() -> None:
    report = run_motion_replay(repository_root=ROOT, steps=510)
    assert report["passed"] is True
    assert report["completed_transitions"] == 510
    assert report["last_q9"] == 518
    assert report["minimum_base_height_m"] >= 0.45
    assert report["maximum_base_tilt_rad"] <= 1.0
    assert report["authorization"] == {
        "simulator_only": True,
        "dds_opened": False,
        "hardware_authorized": False,
        "robot_commands_published": False,
    }


def _supervised_controller() -> SupervisedCleanTrue23MujocoController:
    return SupervisedCleanTrue23MujocoController(
        model_path=ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        physics_path=ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        policy=CleanSonicPolicy(
            ROOT / "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx",
            ROOT / "artifacts/g1_true23/steptouch_balanced_teacher_lowrank_preserve_alpha010_v1.decoder.onnx",
        ),
        fallback_policy=UnitreeZeroVelocityFallbackPolicy(
            ROOT / "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/"
            "policy/velocity/v0/exported/policy.onnx"
        ),
    )


def test_unsupervised_controller_exposes_inactive_fallback_state() -> None:
    controller = CleanTrue23MujocoController(
        model_path=ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        physics_path=ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        policy=CleanSonicPolicy(
            ROOT / "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx",
            ROOT / "artifacts/g1_true23/steptouch_balanced_teacher_lowrank_preserve_alpha010_v1.decoder.onnx",
        ),
    )
    assert controller.fallback_active is False
    assert controller.fallback_trigger is None
    assert controller.fallback_transition is None


@pytest.mark.parametrize(
    ("clip", "bundle_name", "steps"),
    [
        ("upright", "causal_packets_neutral_calibrated_v1.json", 41),
        ("standing_1806", "causal_packets_neutral_calibrated_v1.json", 38),
        ("crouch", "causal_packets_external_upright_calibration_v1.json", 62),
    ],
)
def test_balanced_upper_body_replays_real_calibrated_pico_clip(clip: str, bundle_name: str, steps: int) -> None:
    bundle_path = ROOT / f"artifacts/g1_true23/pico_saved_clip_replay_v1/{clip}/{bundle_name}"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    packets = bundle["robot_independent_reference_packets"]
    controller = BalancedUpperBodyTrue23MujocoController(
        model_path=ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        physics_path=ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        balance_policy=UnitreeZeroVelocityFallbackPolicy(
            ROOT / "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/"
            "policy/velocity/v0/exported/policy.onnx"
        ),
    )
    controller.reset()
    report = run_balanced_upper_body_reference_sequence(
        controller=controller,
        packets=packets,
        steps=steps,
    )
    assert report["passed"] is True
    assert report["completed_transitions"] == steps
    assert report["minimum_base_height_m"] >= 0.75
    assert report["maximum_base_tilt_rad"] <= 0.12
    assert report["arm_reference_blend"] == 0.4
    assert report["balance_owned_hardware_joint_count"] == 15
    assert report["pico_arm_hardware_joint_count"] == 8
    assert report["full_sonic_policy_used"] is False


def _run_supervised(*, impulse_axis: int | None, impulse: float) -> tuple[Any, float, float]:
    with np.load(STEP_TOUCH, allow_pickle=False) as archive:
        motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    controller = _supervised_controller()
    root_velocity = np.concatenate(
        (
            motion["body_lin_vel_w"][10, 0],
            _quaternion_matrix(motion["body_quat_w"][10, 0]).T @ motion["body_ang_vel_w"][10, 0],
        )
    )
    controller.reset(
        base_position=motion["body_pos_w"][10, 0],
        base_quaternion_wxyz=motion["body_quat_w"][10, 0],
        joint_position_hardware=motion["joint_pos"][10],
        root_velocity=root_velocity,
        joint_velocity_hardware=motion["joint_vel"][10],
        buffered_robot_pelvis_q9=motion["body_quat_w"][9, 0],
    )
    minimum_height = float("inf")
    maximum_tilt = 0.0
    for transition in range(160):
        if transition == 50 and impulse_axis is not None:
            controller.data.qvel[impulse_axis] += impulse
            controller.module.mj_forward(controller.model, controller.data)
        packet = motion_reference_terms(motion, 9 + transition)
        evidence = controller.step(encoder267_from_reference(packet, controller.buffered_robot_pelvis_q9))
        minimum_height = min(minimum_height, float(evidence["base_height_m"]))
        maximum_tilt = max(maximum_tilt, float(evidence["base_tilt_rad"]))
    return controller, minimum_height, maximum_tilt


def test_supervisor_does_not_false_trigger_nominal_motion() -> None:
    controller, minimum_height, maximum_tilt = _run_supervised(impulse_axis=None, impulse=0.0)
    assert controller.completed == 160
    assert controller.fallback_active is False
    assert controller.fallback_policy.query_count == 0
    assert minimum_height >= 0.45
    assert maximum_tilt <= 1.0


@pytest.mark.parametrize(
    ("axis", "impulse", "trigger"),
    [
        (0, -0.5, "linear_velocity_jump"),
        (2, 0.2, "vertical_velocity_jump"),
        (5, 0.78, "angular_velocity_jump"),
    ],
)
def test_supervisor_latches_unitree_balance_fallback(axis: int, impulse: float, trigger: str) -> None:
    controller, minimum_height, maximum_tilt = _run_supervised(impulse_axis=axis, impulse=impulse)
    assert controller.completed == 160
    assert controller.fallback_active is True
    assert controller.fallback_trigger == trigger
    assert controller.fallback_transition == 50
    assert controller.fallback_policy.query_count == 110
    assert minimum_height >= 0.45
    assert maximum_tilt <= 1.0
