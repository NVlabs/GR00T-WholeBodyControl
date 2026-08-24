"""Run hash-bound clean SONIC true23 teleoperation in local CPU MuJoCo."""

from __future__ import annotations

import argparse
import hashlib
from itertools import chain
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import zmq

from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_DEFAULT_Q_HARDWARE,
)
from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    BalancedUpperBodyTrue23MujocoController,
    CleanSonicPolicy,
    CleanTrue23MujocoController,
    SupervisedCleanTrue23MujocoController,
    UnitreeZeroVelocityFallbackPolicy,
    pico_reference_policy_history,
    reference_initial_state,
    run_balanced_upper_body_reference_sequence,
    run_motion_replay,
    run_reference_sequence,
    run_supervisor_disturbance_qualification,
    validate_reference_terms,
)
from gear_sonic.utils.g1_true23_pico_sonic_mode_registry import (
    PROFILE_NAMES,
    load_native23_mode_profile,
)
from gear_sonic.utils.g1_true23_sonic_library_replay import run_library_motion_replay


def _exclusive_json(path: Path, report: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _zmq_packets(endpoint: str, timeout_ms: int) -> Iterator[dict]:
    context = zmq.Context(io_threads=1)
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, 8)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    try:
        while True:
            if not dict(poller.poll(timeout_ms)).get(socket):
                raise TimeoutError("timed out waiting for local PICO causal packet")
            value = socket.recv_json()
            if not isinstance(value, dict):
                raise ValueError("PICO causal ZMQ payload is not a JSON object")
            yield value
    finally:
        socket.close()
        context.term()


def _saved_packets(path: Path) -> tuple[list[dict], str]:
    """Load immutable offline causal packets without claiming live freshness."""

    raw = path.resolve().read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {
        "robot_independent_reference_packets",
        "semantic_packets",
    }:
        raise ValueError("saved PICO packet bundle schema mismatch")
    packets = payload["robot_independent_reference_packets"]
    if not isinstance(packets, list) or not packets:
        raise ValueError("saved PICO packet bundle has no reference packets")
    if any(not isinstance(packet, dict) for packet in packets):
        raise ValueError("saved PICO reference packet is not an object")
    for packet in packets:
        validate_reference_terms(packet)
    return packets, hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "replay",
            "library-replay",
            "saved",
            "balanced-upper-saved",
            "zmq",
            "balanced-upper-zmq",
            "supervisor-screen",
        ),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=510)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--packets", type=Path)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--trajectory-output", type=Path)
    parser.add_argument("--receive-timeout-ms", type=int, default=2000)
    parser.add_argument("--maximum-age-ms", type=int, default=100)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--disable-safety-fallback", action="store_true")
    parser.add_argument("--native23-profile", choices=PROFILE_NAMES)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.native23_profile is not None and args.mode not in {"library-replay", "saved", "zmq"}:
        raise ValueError("native23 profile is only valid for library-replay, saved, or zmq mode")
    if args.mode != "library-replay" and (args.motion is not None or args.trajectory_output is not None):
        raise ValueError("motion and trajectory output are only valid for library-replay mode")
    if args.mode == "replay":
        report = run_motion_replay(repository_root=root, steps=args.steps, viewer=args.viewer)
    elif args.mode == "library-replay":
        if args.viewer:
            raise ValueError("library replay does not open a viewer")
        if args.native23_profile is None or args.motion is None:
            raise ValueError("library-replay requires --native23-profile and --motion")
        if not args.disable_safety_fallback:
            raise ValueError("library-replay requires the native23 fail-closed controller")
        native23_profile = load_native23_mode_profile(root, args.native23_profile)
        if "library_replay" not in native23_profile.simulator_modes:
            raise ValueError("native23 profile does not permit library replay")
        motion_path = args.motion if args.motion.is_absolute() else root / args.motion
        report, arrays = run_library_motion_replay(
            repository_root=root,
            motion_path=motion_path,
            maximum_steps=args.steps,
            decoder_path=native23_profile.decoder_path,
            expected_decoder_sha256=native23_profile.decoder_sha256,
            controller_mode="sonic",
            gain_profile="released_retained",
        )
        report["mode"] = "sonic_native23_library_replay"
        report["native23_profile"] = native23_profile.name
        report["native23_profile_purpose"] = native23_profile.purpose
        report["live_transport_proven"] = False
        report["live_headset_source_proven"] = False
        report["hardware_authorized"] = False
        if args.trajectory_output is not None:
            trajectory_path = (
                args.trajectory_output if args.trajectory_output.is_absolute() else root / args.trajectory_output
            )
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            with trajectory_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            report["trajectory_npz"] = str(trajectory_path.resolve())
    elif args.mode == "supervisor-screen":
        if args.viewer:
            raise ValueError("supervisor screen does not open a viewer")
        report = run_supervisor_disturbance_qualification(repository_root=root, steps=args.steps)
    else:
        native23_profile = None
        if args.native23_profile is not None:
            if not args.disable_safety_fallback:
                raise ValueError("native23 profile requires the native23 fail-closed controller")
            native23_profile = load_native23_mode_profile(root, args.native23_profile)
            if args.mode not in native23_profile.simulator_modes:
                raise ValueError("native23 profile does not permit this simulator mode")
        controller_kwargs = {
            "model_path": root / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
            "physics_path": root / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        }
        if args.mode in {"balanced-upper-saved", "balanced-upper-zmq"}:
            if args.disable_safety_fallback:
                raise ValueError("balanced upper-body mode requires the balance policy")
            controller = BalancedUpperBodyTrue23MujocoController(
                model_path=controller_kwargs["model_path"],
                physics_path=controller_kwargs["physics_path"],
                balance_policy=UnitreeZeroVelocityFallbackPolicy(
                    root / "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/"
                    "policy/velocity/v0/exported/policy.onnx"
                ),
            )
        elif args.disable_safety_fallback:
            encoder_path = (
                native23_profile.encoder_path
                if native23_profile is not None
                else root / "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx"
            )
            decoder_path = (
                native23_profile.decoder_path
                if native23_profile is not None
                else root
                / "artifacts/g1_true23/steptouch_balanced_teacher_lowrank_preserve_alpha010_v1.decoder.onnx"
            )
            controller = CleanTrue23MujocoController(
                **controller_kwargs,
                minimum_base_height_m=(
                    native23_profile.minimum_base_height_m if native23_profile is not None else 0.45
                ),
                maximum_base_tilt_rad=(
                    native23_profile.maximum_base_tilt_rad if native23_profile is not None else 1.0
                ),
                policy=CleanSonicPolicy(
                    encoder_path,
                    decoder_path,
                    expected_decoder_sha256=(
                        native23_profile.decoder_sha256
                        if native23_profile is not None
                        else "dc4b6cf4681eafaff6bb6d70d0aad136e9a3a184337490dc32a511e45b31ec9a"
                    ),
                ),
            )
            if native23_profile is not None:
                controller.use_released_retained_gains()
        else:
            controller = SupervisedCleanTrue23MujocoController(
                **controller_kwargs,
                policy=CleanSonicPolicy(
                    root / "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx",
                    root
                    / "artifacts/g1_true23/steptouch_balanced_teacher_lowrank_preserve_alpha010_v1.decoder.onnx",
                ),
                fallback_policy=UnitreeZeroVelocityFallbackPolicy(
                    root / "artifacts/external/unitree_rl_mjlab/deploy/robots/g1/config/"
                    "policy/velocity/v0/exported/policy.onnx"
                ),
            )
        saved_packet_sha256 = None
        if args.mode in {"saved", "balanced-upper-saved"}:
            if args.packets is None:
                raise ValueError("saved mode requires --packets")
            saved, saved_packet_sha256 = _saved_packets(args.packets)
            if args.steps != len(saved):
                raise ValueError(f"saved mode --steps must equal packet count {len(saved)}")
            packets = iter(saved)
        else:
            if args.packets is not None:
                raise ValueError("--packets is only valid in saved mode")
            packets = _zmq_packets(args.endpoint, args.receive_timeout_ms)
        first = next(packets)
        validate_reference_terms(first)
        startup_second = None
        if native23_profile is not None:
            startup_second = next(packets)
            first_summary = validate_reference_terms(first)
            second_summary = validate_reference_terms(startup_second)
            if (
                second_summary["anchor_index"] != first_summary["anchor_index"] + 1
                or second_summary["anchor_monotonic_ns"] != first_summary["anchor_monotonic_ns"] + 20_000_000
            ):
                raise ValueError("native23 startup packets are not contiguous")
            packets = chain((startup_second,), packets)
        if args.mode in {"balanced-upper-saved", "balanced-upper-zmq"}:
            controller.reset()
        elif args.mode == "saved" and native23_profile is None:
            q10 = SAFE_TARGET_DEFAULT_Q_HARDWARE
            qd10 = [0.0] * len(SAFE_TARGET_DEFAULT_Q_HARDWARE)
        else:
            q10, qd10 = reference_initial_state(first)
            if startup_second is not None:
                qd10 = reference_initial_state(startup_second)[1]
        if args.mode not in {"balanced-upper-saved", "balanced-upper-zmq"}:
            base_height = controller.reference_root_height(q10) if native23_profile is not None else 0.76
            root_velocity = None
            if native23_profile is not None:
                if startup_second is None:
                    raise RuntimeError("native23 startup second packet missing")
                q11_hardware = reference_initial_state(startup_second)[0]
                height_q11 = controller.reference_root_height(q11_hardware)
                root_velocity = [0.0, 0.0, (height_q11 - base_height) / 0.02, 0.0, 0.0, 0.0]
            controller.reset(
                base_position=[0.0, 0.0, base_height],
                base_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
                joint_position_hardware=q10,
                root_velocity=root_velocity,
                joint_velocity_hardware=qd10,
            )
            if native23_profile is not None:
                controller.history = pico_reference_policy_history(first)
        passive = None
        if args.viewer:
            import mujoco.viewer

            passive = mujoco.viewer.launch_passive(controller.model, controller.data)
        try:
            try:
                if args.mode in {"balanced-upper-saved", "balanced-upper-zmq"}:
                    report = run_balanced_upper_body_reference_sequence(
                        controller=controller,
                        packets=chain((first,), packets),
                        steps=args.steps,
                        maximum_age_ns=(
                            None if args.mode == "balanced-upper-saved" else args.maximum_age_ms * 1_000_000
                        ),
                        step_callback=None if passive is None else passive.sync,
                    )
                else:
                    report = run_reference_sequence(
                        controller=controller,
                        packets=chain((first,), packets),
                        steps=args.steps,
                        maximum_age_ns=(None if args.mode == "saved" else args.maximum_age_ms * 1_000_000),
                        step_callback=None if passive is None else passive.sync,
                        retarget_native23_fk=native23_profile is not None,
                    )
            except Exception as error:
                if args.mode not in {"saved", "balanced-upper-saved"}:
                    raise
                report = {
                    "schema_version": 1,
                    "kind": "g1_true23_clean_mujoco_saved_pico_failure",
                    "mode": "pico_saved_causal_packet_replay",
                    "completed_transition_calls": controller.completed,
                    "failed_transition_index": controller.completed,
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "fallback_active": bool(getattr(controller, "fallback_active", False)),
                    "fallback_trigger": getattr(controller, "fallback_trigger", None),
                    "fallback_first_transition": getattr(controller, "fallback_transition", None),
                    "terminal_base_height_m": float(controller.data.qpos[2]),
                    "passed": False,
                    "authorization": {
                        "simulator_only": True,
                        "dds_opened": False,
                        "hardware_authorized": False,
                        "robot_commands_published": False,
                    },
                }
        finally:
            if passive is not None:
                passive.close()
        if args.mode in {"saved", "balanced-upper-saved"}:
            report["mode"] = (
                "pico_saved_balanced_upper_body_replay"
                if args.mode == "balanced-upper-saved"
                else "pico_saved_causal_packet_replay"
            )
            report["saved_packet_bundle"] = str(args.packets.resolve())
            report["saved_packet_bundle_sha256"] = saved_packet_sha256
            report["offline_saved_capture"] = True
            report["physical_initialization"] = (
                "unitree_zero_velocity_standing_default"
                if args.mode == "balanced-upper-saved"
                else ("first_packet_causal_q10" if native23_profile is not None else "safe_target_default")
            )
            report["live_freshness_checked"] = False
            report["live_transport_proven"] = False
        elif args.mode == "balanced-upper-zmq":
            report["mode"] = "pico_balanced_upper_body_zmq"
            report["endpoint"] = args.endpoint
            report["offline_saved_capture"] = False
            report["live_freshness_checked"] = True
            report["live_transport_proven"] = True
        else:
            report["endpoint"] = args.endpoint
        report["safety_fallback_enabled"] = not args.disable_safety_fallback
        if native23_profile is not None:
            report["native23_profile"] = native23_profile.name
            report["native23_profile_purpose"] = native23_profile.purpose
            report["decoder_sha256"] = native23_profile.decoder_sha256
            report["physical_dof"] = 23
            report["decoder_output_dof"] = 23
            report["source_29dof_physics_used"] = False
            report["gain_profile"] = "released_retained"
            report["minimum_base_height_gate_m"] = native23_profile.minimum_base_height_m
            report["maximum_base_tilt_gate_rad"] = native23_profile.maximum_base_tilt_rad
            report["live_transport_proven"] = args.mode == "zmq"
            report["hardware_authorized"] = False
    _exclusive_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
