#!/usr/bin/env python3
"""Replay the recorded native124 first target in true-23 MuJoCo only.

This diagnostic imports neither the Unitree SDK nor a transport.  It cannot
publish a command.  It compares the exact recorded hardware target with
hold/ramp acquisition paths, and separately stress-tests the selected actor
with a target-derived reference proxy.  The proxy is deliberately labelled:
the live evidence did not log the 124-value observation or reference packet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from gear_sonic.scripts.validate_g1_23dof_native124_mujoco import (
    DECIMATION,
    EFFORT_LIMIT_HARDWARE,
    KD_NATIVE,
    KP_NATIVE,
    MINIMUM_BASE_HEIGHT_M,
    SIMULATION_DT,
    _relative_rotation_6d,
    _tilt_rad,
)
from gear_sonic.utils.g1_23dof_live_shadow import (
    HARDWARE_LOWER_LIMIT,
    HARDWARE_UPPER_LIMIT,
)
from gear_sonic.utils.g1_23dof_native124_policy import (
    DEFAULT_Q_NATIVE,
    SELECTED_GANTRY_POLICY_SHA256,
    Native124Policy,
    build_observation,
    hardware_compact_to_native,
    hardware_targets_to_raw_action,
    native_to_hardware_compact,
    raw_action_to_hardware_targets,
)

CONTROL_HZ = 50
MAXIMUM_TILT_RAD = 1.0
WARM_START_FRAMES = 25
REFERENCE_RAMP_FRAMES = 100

# native124_selected_shadow_20260804_082454.jsonl, action frame 0.
LIVE_RAW_ACTION_NATIVE = np.asarray(
    [
        1.2998430728912354,
        1.5033355951309204,
        -2.1498398780822754,
        0.10149362683296204,
        0.5569177865982056,
        -1.5150830745697021,
        -0.832035481929779,
        1.1066503524780273,
        2.3408827781677246,
        -0.41734063625335693,
        0.06555822491645813,
        -1.8317654132843018,
        1.916811227798462,
        0.48396217823028564,
        1.368984341621399,
        0.09957979619503021,
        -0.47776421904563904,
        -1.892457365989685,
        -0.23158708214759827,
        0.3642426133155823,
        -0.001703205518424511,
        1.7553603649139404,
        -2.857095241546631,
    ],
    dtype=np.float32,
)
LIVE_TARGET_HARDWARE = np.asarray(
    [
        0.40031400322914124,
        0.03562426194548607,
        0.6064444184303284,
        0.026050370186567307,
        -0.31928446888923645,
        0.15990251302719116,
        0.5118278861045837,
        0.1954781413078308,
        1.2828037738800049,
        1.3418008089065552,
        -0.5727384686470032,
        -0.0007477072067558765,
        -0.943779706954956,
        -0.46512147784233093,
        0.016787463799118996,
        0.21245940029621124,
        -0.23078875243663788,
        0.7706031799316406,
        -0.1652635782957077,
        -0.17121994495391846,
        0.6009840965270996,
        0.49833330512046814,
        -1.2542648315429688,
    ],
    dtype=np.float32,
)


def _new_simulation(xml_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if (model.nu, model.nq, model.nv) != (23, 30, 29):
        raise ValueError(
            f"expected true-23 free-base model; got nu={model.nu}, nq={model.nq}, nv={model.nv}"
        )
    model.opt.timestep = SIMULATION_DT
    data.qpos[:3] = (0.0, 0.0, 0.793)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:] = native_to_hardware_compact(DEFAULT_Q_NATIVE)
    mujoco.mj_forward(model, data)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_id < 0:
        raise ValueError("MuJoCo model lacks torso_link")
    return model, data, torso_id


def _step_pd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
) -> None:
    for _ in range(DECIMATION):
        torque = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
        data.ctrl[:] = np.clip(
            torque,
            -EFFORT_LIMIT_HARDWARE,
            EFFORT_LIMIT_HARDWARE,
        )
        mujoco.mj_step(model, data)


def _physics_result(
    *,
    name: str,
    heights: list[float],
    tilts: list[float],
    terminated_step: int | None,
    maximum_target_step_rad: float,
    final_target_error_rad: float,
    raw_target_limit_clamps: int = 0,
    maximum_raw_target_excess_rad: float = 0.0,
    maximum_raw_action_abs: float = 0.0,
    closed_loop_actor: bool,
) -> dict[str, object]:
    physics_passed = terminated_step is None
    raw_contract_passed = raw_target_limit_clamps == 0
    return {
        "strategy": name,
        "closed_loop_actor": closed_loop_actor,
        "physics_passed": physics_passed,
        "raw_policy_contract_passed": raw_contract_passed,
        "usable_acquisition_evidence": (
            closed_loop_actor and physics_passed and raw_contract_passed
        ),
        "steps_completed": len(heights),
        "terminated_step": terminated_step,
        "termination_seconds": (
            None if terminated_step is None else (terminated_step + 1) / CONTROL_HZ
        ),
        "minimum_base_height_m": min(heights),
        "maximum_tilt_rad": max(tilts),
        "maximum_target_step_rad": maximum_target_step_rad,
        "maximum_target_rate_rad_s": maximum_target_step_rad * CONTROL_HZ,
        "final_target_error_rad": final_target_error_rad,
        "raw_target_limit_clamps": raw_target_limit_clamps,
        "maximum_raw_target_excess_rad": maximum_raw_target_excess_rad,
        "maximum_raw_action_abs": maximum_raw_action_abs,
    }


def replay_recorded_target(
    *,
    xml_path: Path,
    steps: int,
    name: str,
    warm_start_frames: int,
    maximum_target_rate_rad_s: float | None,
) -> dict[str, object]:
    model, data, _ = _new_simulation(xml_path)
    kp = native_to_hardware_compact(KP_NATIVE)
    kd = native_to_hardware_compact(KD_NATIVE)
    start = native_to_hardware_compact(DEFAULT_Q_NATIVE).astype(np.float64)
    target = start.copy()
    heights: list[float] = []
    tilts: list[float] = []
    maximum_step = 0.0
    terminated_step = None
    for step in range(steps):
        requested = start if step < warm_start_frames else LIVE_TARGET_HARDWARE
        previous = target.copy()
        if maximum_target_rate_rad_s is None:
            target = np.asarray(requested, dtype=np.float64).copy()
        else:
            limit = maximum_target_rate_rad_s / CONTROL_HZ
            target += np.clip(requested - target, -limit, limit)
        maximum_step = max(maximum_step, float(np.max(np.abs(target - previous))))
        _step_pd(model, data, target, kp, kd)
        heights.append(float(data.qpos[2]))
        tilts.append(_tilt_rad(data.qpos[3:7]))
        if heights[-1] < MINIMUM_BASE_HEIGHT_M or tilts[-1] > MAXIMUM_TILT_RAD:
            terminated_step = step
            break
    return _physics_result(
        name=name,
        heights=heights,
        tilts=tilts,
        terminated_step=terminated_step,
        maximum_target_step_rad=maximum_step,
        final_target_error_rad=float(np.max(np.abs(target - LIVE_TARGET_HARDWARE))),
        closed_loop_actor=False,
    )


def replay_actor(
    *,
    policy: Native124Policy,
    xml_path: Path,
    steps: int,
    name: str,
    target_derived_reference_proxy: bool,
    maximum_target_rate_rad_s: float | None,
) -> dict[str, object]:
    model, data, torso_id = _new_simulation(xml_path)
    kp = native_to_hardware_compact(KP_NATIVE)
    kd = native_to_hardware_compact(KD_NATIVE)
    lower = np.asarray(HARDWARE_LOWER_LIMIT, dtype=np.float64)
    upper = np.asarray(HARDWARE_UPPER_LIMIT, dtype=np.float64)
    start = native_to_hardware_compact(DEFAULT_Q_NATIVE).astype(np.float64)
    desired_reference = hardware_compact_to_native(LIVE_TARGET_HARDWARE)
    previous_action = np.zeros(23, dtype=np.float32)
    target = start.copy()
    heights: list[float] = []
    tilts: list[float] = []
    maximum_step = 0.0
    raw_clamps = 0
    maximum_excess = 0.0
    maximum_action = 0.0
    terminated_step = None
    for step in range(steps):
        if target_derived_reference_proxy and step >= WARM_START_FRAMES:
            ramp_index = step - WARM_START_FRAMES + 1
            u = min(ramp_index / REFERENCE_RAMP_FRAMES, 1.0)
            alpha = u * u * (3.0 - 2.0 * u)
            alpha_rate = (
                0.0
                if u >= 1.0
                else 6.0 * u * (1.0 - u) / (REFERENCE_RAMP_FRAMES / CONTROL_HZ)
            )
            q_ref = DEFAULT_Q_NATIVE + alpha * (desired_reference - DEFAULT_Q_NATIVE)
            qd_ref = alpha_rate * (desired_reference - DEFAULT_Q_NATIVE)
        else:
            q_ref = DEFAULT_Q_NATIVE
            qd_ref = np.zeros(23, dtype=np.float32)
        observation = build_observation(
            q_ref_native=q_ref,
            qd_ref_native=qd_ref,
            motion_anchor_ori_b=_relative_rotation_6d(
                data.xquat[torso_id], np.asarray([1.0, 0.0, 0.0, 0.0])
            ),
            base_ang_vel=data.qvel[3:6],
            q_measured_native=hardware_compact_to_native(data.qpos[7:]),
            qd_measured_native=hardware_compact_to_native(data.qvel[6:]),
            previous_raw_action_native=previous_action,
        )
        action = policy.run(observation)
        maximum_action = max(maximum_action, float(np.max(np.abs(action))))
        raw_target = raw_action_to_hardware_targets(action).astype(np.float64)
        excess = np.maximum(np.maximum(lower - raw_target, raw_target - upper), 0.0)
        raw_clamps += int(np.count_nonzero(excess > 0.0))
        maximum_excess = max(maximum_excess, float(np.max(excess)))
        requested = np.clip(raw_target, lower, upper)
        if target_derived_reference_proxy and step < WARM_START_FRAMES:
            requested = start
        previous_target = target.copy()
        if maximum_target_rate_rad_s is None:
            target = requested.copy()
            previous_action = action.copy()
        else:
            limit = maximum_target_rate_rad_s / CONTROL_HZ
            target += np.clip(requested - target, -limit, limit)
            previous_action = hardware_targets_to_raw_action(target)
        maximum_step = max(
            maximum_step,
            float(np.max(np.abs(target - previous_target))),
        )
        _step_pd(model, data, target, kp, kd)
        heights.append(float(data.qpos[2]))
        tilts.append(_tilt_rad(data.qpos[3:7]))
        if heights[-1] < MINIMUM_BASE_HEIGHT_M or tilts[-1] > MAXIMUM_TILT_RAD:
            terminated_step = step
            break
    return _physics_result(
        name=name,
        heights=heights,
        tilts=tilts,
        terminated_step=terminated_step,
        maximum_target_step_rad=maximum_step,
        final_target_error_rad=float(np.max(np.abs(target - LIVE_TARGET_HARDWARE))),
        raw_target_limit_clamps=raw_clamps,
        maximum_raw_target_excess_rad=maximum_excess,
        maximum_raw_action_abs=maximum_action,
        closed_loop_actor=True,
    )


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=repository
        / "artifacts/unitree23_candidates/assets/models/g1/beyondmimic/23dof_50fps/fightAndSports1_subject1.onnx",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=repository / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    steps = int(round(args.seconds * CONTROL_HZ))
    if steps < 150:
        parser.error("--seconds must be at least 3.0")
    policy = Native124Policy(
        args.model,
        expected_sha256=SELECTED_GANTRY_POLICY_SHA256,
    )

    strategies = [
        replay_actor(
            policy=policy,
            xml_path=args.xml,
            steps=steps,
            name="neutral_reference_actor_direct_baseline",
            target_derived_reference_proxy=False,
            maximum_target_rate_rad_s=None,
        ),
        replay_recorded_target(
            xml_path=args.xml,
            steps=steps,
            name="recorded_live_target_direct",
            warm_start_frames=0,
            maximum_target_rate_rad_s=None,
        ),
        replay_recorded_target(
            xml_path=args.xml,
            steps=steps,
            name="hold_0p5s_then_recorded_live_target_direct",
            warm_start_frames=WARM_START_FRAMES,
            maximum_target_rate_rad_s=None,
        ),
    ]
    for rate in (0.25, 0.5, 1.0, 2.0):
        strategies.append(
            replay_recorded_target(
                xml_path=args.xml,
                steps=steps,
                name=f"recorded_live_target_ramp_{rate:g}_rad_s",
                warm_start_frames=0,
                maximum_target_rate_rad_s=rate,
            )
        )
    strategies.append(
        replay_actor(
            policy=policy,
            xml_path=args.xml,
            steps=steps,
            name="warm_start_reference_ramp_proxy_actor_direct",
            target_derived_reference_proxy=True,
            maximum_target_rate_rad_s=None,
        )
    )
    for rate in (0.25, 0.5, 1.0, 2.0):
        strategies.append(
            replay_actor(
                policy=policy,
                xml_path=args.xml,
                steps=steps,
                name=f"warm_start_reference_ramp_proxy_actor_target_ramp_{rate:g}_rad_s",
                target_derived_reference_proxy=True,
                maximum_target_rate_rad_s=rate,
            )
        )

    acquisition_candidates = [
        item
        for item in strategies
        if item["strategy"] != "neutral_reference_actor_direct_baseline"
    ]
    report = {
        "schema_version": 1,
        "kind": "g1_true23_native124_recorded_live_first_frame_mujoco_replay",
        "simulation_only": True,
        "robot_mutation_authorized": False,
        "model_dof": 23,
        "policy_sha256": policy.sha256,
        "source_evidence": "/root/g1_true23_runs/live/native124_selected_shadow_20260804_082454.jsonl",
        "source_action_frame_index": 0,
        "source_reported_target_slew_rad": 1.2954457234591246,
        "source_lowstate_age_ms": 0.713625,
        "source_pico_packet_age_ms": 70.451489,
        "source_end_to_end_age_ms": 71.020372,
        "recorded_raw_action_native": LIVE_RAW_ACTION_NATIVE.tolist(),
        "recorded_target_hardware": LIVE_TARGET_HARDWARE.tolist(),
        "default_start_maximum_target_delta_rad": float(
            np.max(
                np.abs(
                    LIVE_TARGET_HARDWARE
                    - native_to_hardware_compact(DEFAULT_Q_NATIVE)
                )
            )
        ),
        "exact_observation_reconstructable": False,
        "observation_limitation": (
            "The evidence logs the action and target but not q_ref, qd_ref, "
            "motion-anchor orientation, measured q/qd, or the 124-value observation."
        ),
        "reference_proxy_contract": (
            "Reference-ramp stress cases use the recorded hardware target, reordered "
            "to native23, only as a diagnostic pose proxy. It is neither an estimate "
            "nor a bound on the missing live standing reference."
        ),
        "supporting_reports": [
            "artifacts/unitree23_candidates/acquisition_applied_history_sports_neutral_20s.json",
            "artifacts/unitree23_candidates/gantry_acquisition_sports_neutral_20s.json",
            "artifacts/unitree23_candidates/gantry_sports_neutral_rate_0.5.json",
            "artifacts/unitree23_candidates/gantry_sports_neutral_rate_1.0.json",
            "artifacts/unitree23_candidates/gantry_sports_neutral_rate_2.0.json",
            "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.mujoco_smoke_diagnostic.json",
        ],
        "strategies": strategies,
        "acquisition_strategy_passed": any(
            bool(item["usable_acquisition_evidence"])
            for item in acquisition_candidates
        ),
        "go_for_robot_actuation": False,
        "go_no_go": "NO_GO",
        "reason": (
            "Every tested recorded-target or reference-ramp acquisition path falls "
            "in true-23 MuJoCo or violates the raw actor target envelope. The benign "
            "neutral baseline does not reproduce the live observation, and the exact "
            "live observation was not logged."
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
