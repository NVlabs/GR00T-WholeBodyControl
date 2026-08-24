#!/usr/bin/env python3
"""Run pinned native124 G1 actor against its embedded motion in MuJoCo.

This is a simulation-only gate.  It imports no Unitree SDK and cannot publish
LowCmd.  Passing proves native observation/action integration for the embedded
clip; it does not claim a clip-specific actor is task-general PICO teleop.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from gear_sonic.utils.g1_23dof_live_shadow import (
    HARDWARE_LOWER_LIMIT,
    HARDWARE_UPPER_LIMIT,
)
from gear_sonic.utils.g1_23dof_native124_acquisition import (
    MAXIMUM_LOWSTATE_AGE_MS,
    REFERENCE_RAMP_FRAMES,
    REQUIRED_MODE_MACHINE,
    WARM_START_FRAMES,
    Native23AcquisitionTransition,
)
from gear_sonic.utils.g1_23dof_native124_policy import (
    DEFAULT_Q_NATIVE,
    PUBLIC_POLICY_SHA256,
    Native124Policy,
    build_observation,
    hardware_compact_to_native,
    hardware_targets_to_raw_action,
    native_to_hardware_compact,
    raw_action_to_hardware_targets,
)

CONTROL_HZ = 50
SIMULATION_DT = 0.001
DECIMATION = 20
TORSO_REFERENCE_INDEX = 7
MINIMUM_BASE_HEIGHT_M = 0.45
MAXIMUM_TILT_RAD = 1.0
MAXIMUM_TRACKING_RMSE_RAD = 0.75

KP_NATIVE = np.asarray(
    [
        40.179,
        40.179,
        28.501,
        99.098,
        99.098,
        14.251,
        14.251,
        40.179,
        40.179,
        14.251,
        14.251,
        99.098,
        99.098,
        14.251,
        14.251,
        28.501,
        28.501,
        14.251,
        14.251,
        28.501,
        28.501,
        14.251,
        14.251,
    ],
    dtype=np.float32,
)
KD_NATIVE = np.asarray(
    [
        2.558,
        2.558,
        1.814,
        6.309,
        6.309,
        0.907,
        0.907,
        2.558,
        2.558,
        0.907,
        0.907,
        6.309,
        6.309,
        0.907,
        0.907,
        1.814,
        1.814,
        0.907,
        0.907,
        1.814,
        1.814,
        0.907,
        0.907,
    ],
    dtype=np.float32,
)
EFFORT_LIMIT_HARDWARE = np.asarray(
    [88, 139, 88, 139, 35, 35, 88, 139, 88, 139, 35, 35, 88] + [25] * 10,
    dtype=np.float32,
)
DISTURBANCES = {
    "nominal": np.zeros(6, dtype=np.float64),
    "disturbance_50": np.asarray([0.25, -0.25, 0.1, 0.26, -0.26, 0.39], dtype=np.float64),
    "disturbance_100": np.asarray([0.5, -0.5, 0.2, 0.52, -0.52, 0.78], dtype=np.float64),
}

SYNTHETIC_FULLBODY_AMPLITUDE_NATIVE = np.asarray(
    [
        0.08,
        0.08,
        0.15,
        0.03,
        0.03,
        0.35,
        0.35,
        0.03,
        0.03,
        0.20,
        0.20,
        0.10,
        0.10,
        0.15,
        0.15,
        0.05,
        0.05,
        0.25,
        0.25,
        0.02,
        0.02,
        0.20,
        0.20,
    ],
    dtype=np.float32,
)
SYNTHETIC_FULLBODY_PHASE_NATIVE = np.asarray(
    [
        0,
        0,
        0,
        0,
        math.pi,
        0,
        math.pi,
        0,
        math.pi,
        0,
        math.pi,
        0,
        0,
        0,
        math.pi,
        0,
        0,
        0,
        math.pi,
        0,
        math.pi,
        0,
        math.pi,
    ],
    dtype=np.float32,
)
MEASURED_START_OFFSET_SHAPE_NATIVE = np.sin(np.linspace(0.0, 4.0 * math.pi, 23, dtype=np.float32))
SIMULATED_LOWSTATE_AGE_MS = (0.0, 20.0, 39.999, 40.0)


def _synthetic_fullbody_reference(step: int) -> dict[str, np.ndarray]:
    angular_frequency = 2.0 * math.pi * 0.25
    seconds = step / CONTROL_HZ
    phase = angular_frequency * seconds + SYNTHETIC_FULLBODY_PHASE_NATIVE
    q = DEFAULT_Q_NATIVE + SYNTHETIC_FULLBODY_AMPLITUDE_NATIVE * np.sin(phase)
    qd = angular_frequency * SYNTHETIC_FULLBODY_AMPLITUDE_NATIVE * np.cos(phase)
    return {
        "joint_pos": q.astype(np.float32),
        "joint_vel": qd.astype(np.float32),
        "body_quat_w": np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (14, 1),
        ),
    }


def _quat_conjugate_wxyz(value: np.ndarray) -> np.ndarray:
    return np.asarray([value[0], -value[1], -value[2], -value[3]])


def _quat_mul_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w, x, y, z = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            w * rw - x * rx - y * ry - z * rz,
            w * rx + x * rw + y * rz - z * ry,
            w * ry - x * rz + y * rw + z * rx,
            w * rz + x * ry - y * rx + z * rw,
        ],
        dtype=np.float64,
    )


def _relative_rotation_6d(robot_wxyz: np.ndarray, reference_wxyz: np.ndarray) -> np.ndarray:
    relative = _quat_mul_wxyz(_quat_conjugate_wxyz(robot_wxyz), reference_wxyz)
    relative /= np.linalg.norm(relative)
    w, x, y, z = relative
    return np.asarray(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
        ],
        dtype=np.float32,
    )


def _tilt_rad(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = quaternion_wxyz
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(math.acos(np.clip(up_z, -1.0, 1.0)))


def run_scenario(
    *,
    policy: Native124Policy,
    xml_path: Path,
    scenario: str,
    steps: int,
    reference_mode: str = "embedded",
    acquisition_transition: bool = False,
    measured_start_offset_rad: float = 0.0,
    measured_start_profile: str = "default",
    bounded_external_target_envelope: bool = False,
    maximum_target_rate_rad_s: float = 0.25,
) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = SIMULATION_DT
    if model.nu != 23 or model.nq != 30 or model.nv != 29:
        raise ValueError(
            f"MuJoCo model must be free-base native23; got nq={model.nq}, nv={model.nv}, nu={model.nu}"
        )

    if acquisition_transition and reference_mode == "embedded":
        raise ValueError("acquisition transition requires neutral or synthetic_fullbody")
    if not math.isfinite(measured_start_offset_rad) or not (0.0 <= measured_start_offset_rad <= 0.05):
        raise ValueError("measured start offset must be in [0, 0.05] rad")
    if measured_start_profile not in {"default", "embedded_frame0"}:
        raise ValueError("measured_start_profile must be default or embedded_frame0")
    if not math.isfinite(maximum_target_rate_rad_s) or not (
        0.0 < maximum_target_rate_rad_s <= 2.0
    ):
        raise ValueError("maximum target rate must be in (0, 2] rad/s")

    initial_reference = policy.embedded_reference(0)
    if reference_mode == "neutral":
        initial_reference = {
            "joint_pos": DEFAULT_Q_NATIVE.copy(),
            "joint_vel": np.zeros(23, dtype=np.float32),
            "body_quat_w": np.tile(
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (14, 1),
            ),
        }
    elif reference_mode == "synthetic_fullbody":
        initial_reference = _synthetic_fullbody_reference(0)
    elif reference_mode != "embedded":
        raise ValueError("reference_mode must be embedded, neutral, or synthetic_fullbody")
    initial_anchor_wxyz = initial_reference["body_quat_w"][TORSO_REFERENCE_INDEX].astype(np.float64)
    data.qpos[:3] = [0.0, 0.0, 0.793]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    if acquisition_transition:
        measured_baseline = (
            DEFAULT_Q_NATIVE if measured_start_profile == "default" else policy.embedded_reference(0)["joint_pos"]
        )
        measured_start_native = measured_baseline + measured_start_offset_rad * MEASURED_START_OFFSET_SHAPE_NATIVE
        data.qpos[7:] = native_to_hardware_compact(measured_start_native)
    else:
        data.qpos[7:] = native_to_hardware_compact(initial_reference["joint_pos"])
    mujoco.mj_forward(model, data)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_id < 0:
        raise ValueError("MuJoCo model lacks torso_link")
    kp_hardware = native_to_hardware_compact(KP_NATIVE)
    kd_hardware = native_to_hardware_compact(KD_NATIVE)
    lower = np.asarray(HARDWARE_LOWER_LIMIT, dtype=np.float32)
    upper = np.asarray(HARDWARE_UPPER_LIMIT, dtype=np.float32)
    previous_raw_action = np.zeros(23, dtype=np.float32)
    previous_targets = np.asarray(data.qpos[7:], dtype=np.float32).copy()
    acquisition = None
    if acquisition_transition:
        acquisition = Native23AcquisitionTransition(
            measured_start_hardware=previous_targets,
            reference_start_native=hardware_compact_to_native(previous_targets),
            lower_limit_hardware=lower,
            upper_limit_hardware=upper,
            maximum_raw_target_clamps=(4 if bounded_external_target_envelope else 0),
            maximum_raw_target_excess_rad=(
                1.5 if bounded_external_target_envelope else 0.0
            ),
            maximum_target_step_rad=maximum_target_rate_rad_s / CONTROL_HZ,
        )

    heights: list[float] = []
    tilts: list[float] = []
    tracking_errors: list[float] = []
    maximum_action_abs = 0.0
    raw_target_limit_clamps = 0
    maximum_raw_target_excess_rad = 0.0
    maximum_clamped_joints_per_frame = 0
    maximum_target_step_rad = 0.0
    applied_target_limit_violations = 0
    measured_joint_limit_violations = 0
    nonfinite_count = 0
    inference_ms: list[float] = []
    terminated_step: int | None = None
    local_gate_violations = 0
    maximum_lowstate_age_ms = 0.0
    warm_start_target_hold_violations = 0
    reference_ramp_completed = False
    fail_closed_frames = 0
    safe_abort_step: int | None = None
    handoff_released = False

    for step in range(steps):
        desired_reference = (
            policy.embedded_reference(step)
            if reference_mode == "embedded"
            else (
                _synthetic_fullbody_reference(step)
                if reference_mode == "synthetic_fullbody"
                else initial_reference
            )
        )
        if acquisition is not None:
            acquired_position, acquired_velocity = acquisition.build_reference(
                desired_position_native=desired_reference["joint_pos"],
                desired_velocity_native=desired_reference["joint_vel"],
            )
            reference = {
                **desired_reference,
                "joint_pos": acquired_position,
                "joint_vel": acquired_velocity,
            }
        else:
            reference = desired_reference
        # RoboJuDo aligns embedded clip's initial heading to robot birth frame.
        aligned_reference_wxyz = _quat_mul_wxyz(
            _quat_conjugate_wxyz(initial_anchor_wxyz),
            reference["body_quat_w"][TORSO_REFERENCE_INDEX],
        )
        q_native = hardware_compact_to_native(data.qpos[7:])
        qd_native = hardware_compact_to_native(data.qvel[6:])
        observation = build_observation(
            q_ref_native=reference["joint_pos"],
            qd_ref_native=reference["joint_vel"],
            motion_anchor_ori_b=_relative_rotation_6d(data.xquat[torso_id], aligned_reference_wxyz),
            # Exact public MuJoCo runner contract uses free-base angular qvel.
            base_ang_vel=data.qvel[3:6],
            q_measured_native=q_native,
            qd_measured_native=qd_native,
            previous_raw_action_native=previous_raw_action,
        )
        started_ns = time.perf_counter_ns()
        raw_action = policy.run(observation)
        inference_ms.append((time.perf_counter_ns() - started_ns) / 1e6)
        maximum_action_abs = max(maximum_action_abs, float(np.max(np.abs(raw_action))))
        targets = raw_action_to_hardware_targets(raw_action)
        clamped_mask = (targets < lower) | (targets > upper)
        clamped_count = int(np.count_nonzero(clamped_mask))
        raw_target_limit_clamps += clamped_count
        maximum_clamped_joints_per_frame = max(maximum_clamped_joints_per_frame, clamped_count)
        target_excess = np.maximum(lower - targets, targets - upper)
        maximum_raw_target_excess_rad = max(
            maximum_raw_target_excess_rad,
            float(np.max(np.maximum(target_excess, 0.0))),
        )
        # Public actor has no embedded target clamp.  The acquisition adapter
        # applies the physical envelope only in this MuJoCo validation path.
        if acquisition is not None:
            lowstate_age_ms = SIMULATED_LOWSTATE_AGE_MS[step % len(SIMULATED_LOWSTATE_AGE_MS)]
            maximum_lowstate_age_ms = max(maximum_lowstate_age_ms, lowstate_age_ms)
            try:
                acquisition_frame = acquisition.accept_target(
                    raw_target_hardware=targets,
                    desired_position_native=desired_reference["joint_pos"],
                    desired_velocity_native=desired_reference["joint_vel"],
                    mode_machine=REQUIRED_MODE_MACHINE,
                    lowstate_age_ms=lowstate_age_ms,
                )
            except RuntimeError:
                local_gate_violations += 1
                terminated_step = step
                break
            targets = acquisition_frame.applied_target_hardware
            if not acquisition_frame.acquisition_qualified:
                fail_closed_frames += 1
                safe_abort_step = step
                terminated_step = step
                break
            if step >= WARM_START_FRAMES:
                handoff_released = True
            reference_ramp_completed = reference_ramp_completed or (acquisition_frame.phase == "tracking")
            if step < WARM_START_FRAMES and not np.allclose(targets, previous_targets, rtol=0.0, atol=1.0e-8):
                warm_start_target_hold_violations += 1
        else:
            targets = np.clip(targets, lower, upper)
        previous_raw_action = (
            hardware_targets_to_raw_action(targets)
            if acquisition is not None
            else raw_action.copy()
        )
        maximum_target_step_rad = max(
            maximum_target_step_rad,
            float(np.max(np.abs(targets - previous_targets))),
        )
        previous_targets = targets.copy()
        applied_target_limit_violations += int(np.count_nonzero((targets < lower) | (targets > upper)))

        disturbance_step = (
            WARM_START_FRAMES + REFERENCE_RAMP_FRAMES + 25
            if acquisition is not None
            else 100
        )
        if step == disturbance_step:
            data.qvel[:6] += DISTURBANCES[scenario]

        for _ in range(DECIMATION):
            torque = kp_hardware * (targets - data.qpos[7:]) - kd_hardware * data.qvel[6:]
            data.ctrl[:] = np.clip(torque, -EFFORT_LIMIT_HARDWARE, EFFORT_LIMIT_HARDWARE)
            mujoco.mj_step(model, data)

        height = float(data.qpos[2])
        tilt = _tilt_rad(data.qpos[3:7])
        tracking_error = float(np.sqrt(np.mean((q_native - reference["joint_pos"]) ** 2)))
        heights.append(height)
        tilts.append(tilt)
        tracking_errors.append(tracking_error)
        measured_joint_limit_violations += int(np.count_nonzero((data.qpos[7:] < lower) | (data.qpos[7:] > upper)))
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.ctrl).all()):
            nonfinite_count += 1
            terminated_step = step
            break
        if height < MINIMUM_BASE_HEIGHT_M or tilt > MAXIMUM_TILT_RAD:
            terminated_step = step
            break

    tracking_rmse = float(np.sqrt(np.mean(np.square(tracking_errors)))) if tracking_errors else 0.0
    minimum_height = min(heights) if heights else float(data.qpos[2])
    maximum_tilt = max(tilts) if tilts else _tilt_rad(data.qpos[3:7])
    passed = (
        terminated_step is None
        and len(heights) == steps
        and nonfinite_count == 0
        and applied_target_limit_violations == 0
        and measured_joint_limit_violations == 0
        and minimum_height >= MINIMUM_BASE_HEIGHT_M
        and maximum_tilt <= MAXIMUM_TILT_RAD
        and tracking_rmse <= MAXIMUM_TRACKING_RMSE_RAD
        and (
            not acquisition_transition
            or (
                local_gate_violations == 0
                and maximum_lowstate_age_ms <= MAXIMUM_LOWSTATE_AGE_MS
                and maximum_target_step_rad
                <= maximum_target_rate_rad_s / CONTROL_HZ + 1e-8
                and warm_start_target_hold_violations == 0
            )
        )
    )
    acquisition_completed = (
        acquisition_transition
        and acquisition is not None
        and acquisition.fail_closed_reason is None
        and reference_ramp_completed
    )
    safe_abort_before_handoff = (
        acquisition_transition
        and safe_abort_step is not None
        and not handoff_released
        and local_gate_violations == 0
        and maximum_target_step_rad
        <= maximum_target_rate_rad_s / CONTROL_HZ + 1e-8
        and minimum_height >= MINIMUM_BASE_HEIGHT_M
        and maximum_tilt <= MAXIMUM_TILT_RAD
    )
    return {
        "scenario": scenario,
        "reference_mode": reference_mode,
        "acquisition_transition": acquisition_transition,
        "measured_start_offset_rad": measured_start_offset_rad,
        "measured_start_profile": measured_start_profile,
        "bounded_external_target_envelope": bounded_external_target_envelope,
        "configured_maximum_target_rate_rad_s": maximum_target_rate_rad_s,
        "passed": passed,
        "steps_completed": len(heights),
        "terminated_step": terminated_step,
        "minimum_base_height_m": minimum_height,
        "maximum_tilt_rad": maximum_tilt,
        "tracking_rmse_rad": tracking_rmse,
        "maximum_raw_action_abs": maximum_action_abs,
        "raw_target_limit_clamps": raw_target_limit_clamps,
        "maximum_raw_target_excess_rad": maximum_raw_target_excess_rad,
        "maximum_clamped_joints_per_frame": maximum_clamped_joints_per_frame,
        "maximum_target_step_rad": maximum_target_step_rad,
        "maximum_target_rate_rad_s": maximum_target_step_rad * CONTROL_HZ,
        "applied_target_limit_violations": applied_target_limit_violations,
        "measured_joint_limit_violations": measured_joint_limit_violations,
        "nonfinite_count": nonfinite_count,
        "required_mode_machine": REQUIRED_MODE_MACHINE,
        "simulated_mode_machine": REQUIRED_MODE_MACHINE,
        "maximum_lowstate_age_ms": maximum_lowstate_age_ms,
        "maximum_allowed_lowstate_age_ms": MAXIMUM_LOWSTATE_AGE_MS,
        "local_gate_violations": local_gate_violations,
        "warm_start_frames": WARM_START_FRAMES if acquisition_transition else 0,
        "warm_start_target_hold_violations": warm_start_target_hold_violations,
        "reference_ramp_frames": (REFERENCE_RAMP_FRAMES if acquisition_transition else 0),
        "reference_ramp_completed": reference_ramp_completed,
        "acquisition_completed": acquisition_completed,
        "acquisition_qualified": (acquisition is None or acquisition.fail_closed_reason is None),
        "fail_closed_frames": fail_closed_frames,
        "fail_closed_reason": (acquisition.fail_closed_reason if acquisition is not None else None),
        "safe_abort_step": safe_abort_step,
        "safe_abort_before_handoff": safe_abort_before_handoff,
        "handoff_released": handoff_released,
        "raw_policy_contract_passed": raw_target_limit_clamps == 0,
        "inference_median_ms": float(np.median(inference_ms)),
        "inference_p99_ms": float(np.percentile(inference_ms, 99)),
    }


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=repository / "artifacts" / "unitree23_public" / "OldTownRoad_v1.onnx",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=repository / "gear_sonic" / "data" / "robots" / "g1" / "g1_23dof_rev_1_0.xml",
    )
    parser.add_argument(
        "--expected-sha256",
        default=PUBLIC_POLICY_SHA256,
        help="Required pinned SHA256 for the selected native 124-to-23 ONNX.",
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--reference-mode",
        choices=("embedded", "neutral", "synthetic_fullbody"),
        default="embedded",
        help="Embedded training clip or out-of-clip neutral standing command.",
    )
    parser.add_argument(
        "--acquisition-transition",
        action="store_true",
        help=(
            "MuJoCo-only 0.5 s measured-pose hold, 2.0 s smooth reference "
            "ramp, and 0.005 rad/frame applied-target slew limit."
        ),
    )
    parser.add_argument(
        "--measured-start-offset-rad",
        type=float,
        default=0.0,
        help="Deterministic measured-pose offset amplitude in [0, 0.05] rad.",
    )
    parser.add_argument(
        "--measured-start-profile",
        choices=("default", "embedded_frame0"),
        default="default",
        help="Measured pose used only to initialize MuJoCo acquisition.",
    )
    parser.add_argument(
        "--bounded-external-target-envelope",
        action="store_true",
        help=(
            "Permit at most four externally clamped policy targets and "
            "1.5 rad maximum raw excess while applied targets remain URDF- "
            "and 0.005 rad/frame-bounded. Gantry evaluation only."
        ),
    )
    parser.add_argument(
        "--maximum-target-rate-rad-s",
        type=float,
        default=0.25,
        help="Gantry target slew in (0, 2] rad/s; default is strict 0.25.",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    steps = int(round(arguments.seconds * CONTROL_HZ))
    if steps < 101:
        parser.error("--seconds must include disturbance step (>=2.02)")

    policy = Native124Policy(
        arguments.model,
        expected_sha256=arguments.expected_sha256,
    )
    scenarios = [
        run_scenario(
            policy=policy,
            xml_path=arguments.xml.resolve(),
            scenario=name,
            steps=steps,
            reference_mode=arguments.reference_mode,
            acquisition_transition=arguments.acquisition_transition,
            measured_start_offset_rad=arguments.measured_start_offset_rad,
            measured_start_profile=arguments.measured_start_profile,
            bounded_external_target_envelope=(
                arguments.bounded_external_target_envelope
            ),
            maximum_target_rate_rad_s=arguments.maximum_target_rate_rad_s,
        )
        for name in DISTURBANCES
    ]
    report = {
        "schema_version": 2 if arguments.acquisition_transition else 1,
        "kind": (
            "g1_true23_native124_acquisition_mujoco_validation"
            if arguments.acquisition_transition
            else "g1_true23_native124_public_policy_mujoco_validation"
        ),
        "simulation_only": True,
        "clip_specific_policy": True,
        "task_general_pico_policy": False,
        "reference_mode": arguments.reference_mode,
        "acquisition_transition": arguments.acquisition_transition,
        "measured_start_offset_rad": arguments.measured_start_offset_rad,
        "measured_start_profile": arguments.measured_start_profile,
        "bounded_external_target_envelope": (
            arguments.bounded_external_target_envelope
        ),
        "maximum_target_rate_rad_s": arguments.maximum_target_rate_rad_s,
        "policy_sha256": policy.sha256,
        "robot_model": "g1_23dof_rev_1_0",
        "control_hz": CONTROL_HZ,
        "simulation_dt_s": SIMULATION_DT,
        "acquisition_contract": (
            {
                "required_mode_machine": REQUIRED_MODE_MACHINE,
                "maximum_lowstate_age_ms": MAXIMUM_LOWSTATE_AGE_MS,
                "warm_start_frames": WARM_START_FRAMES,
                "warm_start_seconds": WARM_START_FRAMES / CONTROL_HZ,
                "reference_ramp_frames": REFERENCE_RAMP_FRAMES,
                "reference_ramp_seconds": REFERENCE_RAMP_FRAMES / CONTROL_HZ,
                "maximum_target_step_rad": (
                    arguments.maximum_target_rate_rad_s / CONTROL_HZ
                ),
                "maximum_target_rate_rad_s": arguments.maximum_target_rate_rad_s,
                "raw_policy_target_clamps_forbidden_for_promotion": True,
                "gantry_external_target_clamp_limit": (
                    4 if arguments.bounded_external_target_envelope else 0
                ),
                "gantry_raw_target_excess_limit_rad": (
                    1.5 if arguments.bounded_external_target_envelope else 0.0
                ),
            }
            if arguments.acquisition_transition
            else None
        ),
        "scenarios": scenarios,
        "passed": all(item["passed"] for item in scenarios),
        "acquisition_completed": (
            all(item["acquisition_completed"] for item in scenarios) if arguments.acquisition_transition else False
        ),
        "safe_abort_before_handoff": (
            all(item["safe_abort_before_handoff"] for item in scenarios)
            if arguments.acquisition_transition
            else False
        ),
        "raw_policy_contract_passed": all(item["raw_policy_contract_passed"] for item in scenarios),
        "promotion_eligible": all(item["passed"] for item in scenarios)
        and all(item["raw_policy_contract_passed"] for item in scenarios)
        and (
            all(item["acquisition_completed"] for item in scenarios) if arguments.acquisition_transition else True
        ),
        "bounded_gantry_eligible": (
            arguments.acquisition_transition
            and arguments.bounded_external_target_envelope
            and all(item["passed"] for item in scenarios)
            and all(item["acquisition_completed"] for item in scenarios)
        ),
        "active_motor_control_authorized": False,
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
