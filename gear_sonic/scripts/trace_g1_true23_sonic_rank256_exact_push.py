"""Trace causal nominal-vs-exact-push divergence for rank256 student."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from gear_sonic.scripts import (
    train_g1_true23_sonic_rank256_disturbance_survival_score as rank256,
    train_g1_true23_sonic_rank256_exact_impulse_survival_score as exact,
)
from gear_sonic.trl.mjlab import (
    sonic_task_space_ppo_full_support_runner as fs,
    sonic_task_space_ppo_runner as task_space,
)
from gear_sonic.utils import g1_true23_sonic_task_space_checkpoint_trace as trace
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_23dof_contract import NATIVE_IL23_TO_CANONICAL_IL29, SOURCE_IL29_JOINT_NAMES

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_exact_push_causal_trace_v1.json"
)
CONTRACT_SHA256 = "066a8c39c120edd430317660395d8a4962b3e211e26e9e7e52ae20b2d6ac6abe"
KIND = "g1_true23_sonic_rank256_exact_push_causal_trace_result_v1"
FAILURE_KIND = "g1_true23_sonic_rank256_exact_push_causal_trace_failure_v1"
NATIVE_JOINT_NAMES = tuple(SOURCE_IL29_JOINT_NAMES[index] for index in NATIVE_IL23_TO_CANONICAL_IL29)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("exact-push causal trace contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    scenario = body.get("scenario", {})
    boundaries = body.get("boundaries", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_exact_push_causal_trace_contract_v1"
        or scenario.get("seed") != rank256.DISTURBANCE_SEED
        or scenario.get("steps") != rank256.COLLECTION_STEPS
        or scenario.get("impulse_transition") != rank256.IMPULSE_TRANSITION
        or scenario.get("scenario_order") != ["nominal", "disturbance"]
        or boundaries.get("training_transitions") != 0
        or boundaries.get("robot_or_network_commands_permitted") is not False
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("exact-push causal trace contract semantic mismatch")
    for name, entry in body["parents"].items():
        raw = Path(entry["path"])
        artifact = (root / raw).resolve(strict=True) if not raw.is_absolute() else raw.resolve(strict=True)
        if artifact.is_symlink() or sha256_file(artifact) != entry["sha256"]:
            raise ValueError(f"exact-push causal trace parent mismatch: {name}")
    return body


def preflight(repository_root: Path) -> Mapping[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract = _load_contract(root)
        with exact._scope(root):  # noqa: SLF001
            base = rank256.preflight(root)
        if base.get("ready") is not True:
            raise RuntimeError("exact-push causal trace base preflight failed")
        sources = {
            CONTRACT_RELATIVE_PATH.as_posix(): CONTRACT_SHA256,
            "gear_sonic/scripts/trace_g1_true23_sonic_rank256_exact_push.py": sha256_file(
                root / "gear_sonic/scripts/trace_g1_true23_sonic_rank256_exact_push.py"
            ),
        }
        material = {
            "base_material_manifest_sha256": base["material_manifest_sha256"],
            "parents": {name: entry["sha256"] for name, entry in contract["parents"].items()},
            "sources": sources,
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_exact_push_causal_trace_preflight_v1",
            "ready": True,
            "contract": contract,
            "base": base,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha(material),
            "simulator_constructed": False,
            "simulator_steps": 0,
            "training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_exact_push_causal_trace_preflight_v1",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "simulator_steps": 0,
            "training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }


class _CausalRecorder(trace._RewardComputeTraceRecorder):  # noqa: SLF001
    def _snapshot(self, q9: int, actor_action: torch.Tensor, reward: torch.Tensor, dt: float) -> dict[str, Any]:
        snapshot = super()._snapshot(q9, actor_action, reward, dt)
        command = self.raw_env.command_manager.get_term("motion")
        robot = self.raw_env.scene["robot"].data
        snapshot["anchor_error_xyz"] = trace._clone_tensor_snapshot(  # noqa: SLF001
            command.anchor_pos_w - command.robot_anchor_pos_w,
            shape=(1, 3),
            context="anchor position error",
            floating=True,
        )
        for name, value in (
            ("projected_gravity", robot.projected_gravity_b),
            ("root_linear_velocity_b", robot.root_link_lin_vel_b),
            ("root_angular_velocity_b", robot.root_link_ang_vel_b),
        ):
            snapshot[name] = trace._clone_tensor_snapshot(  # noqa: SLF001
                value,
                shape=(1, 3),
                context=name,
                floating=True,
            )
        return snapshot

    def _finalize(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        result = super()._finalize(snapshot)
        for name in (
            "anchor_error_xyz",
            "projected_gravity",
            "root_linear_velocity_b",
            "root_angular_velocity_b",
        ):
            result[name] = trace._tensor_row(snapshot[name], 3, name)  # noqa: SLF001
        return result


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_scenario(
    *,
    scenario: str,
    state: Mapping[str, torch.Tensor],
    topology: Path,
    motion: Path,
    vector: torch.Tensor,
    expected_policy_sha: str,
) -> Mapping[str, Any]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment

    _seed(rank256.DISTURBANCE_SEED)
    cfg = task_space.make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=1)
    cfg.seed = rank256.DISTURBANCE_SEED
    task_audit = task_space.audit_task_space_ppo_env_cfg(cfg, expected_num_envs=1)
    env = ManagerBasedRlEnv(cfg=cfg, device=rank256.parent.DEVICE)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        prime = prime_sonic_true23_training_environment(wrapped)
        actor = rank256._actor(state, topology)  # noqa: SLF001
        actor.eval()
        before = inspect_true23_policy_state(
            {"policy_state_dict": actor.export_true23_policy_state()}, reference_profile=fs.REFERENCE_PROFILE
        )
        if before != expected_policy_sha:
            raise RuntimeError("causal trace actor identity mismatch")
        observations = wrapped.get_observations()
        reward_layout = trace._reward_layout(env)  # noqa: SLF001
        recorder = _CausalRecorder(env, reward_layout)
        frames: list[Mapping[str, Any]] = []
        env.extras["log"] = {}
        try:
            with torch.inference_mode():
                for transition in range(rank256.COLLECTION_STEPS):
                    q9 = trace._q9(env.command_manager.get_term("motion"))  # noqa: SLF001
                    if q9 != 9 + transition:
                        raise RuntimeError("causal trace q9 discontinuity")
                    action = actor(observations, stochastic_output=False)
                    if scenario == "disturbance" and transition == rank256.IMPULSE_TRANSITION:
                        rank256._apply_impulse(wrapped, vector)  # noqa: SLF001
                    recorder.arm(q9, action)
                    observations, rewards, dones, extras = wrapped.step(action)
                    frame = recorder.finish()
                    extras["log"] = {}
                    if not math.isclose(
                        frame["reward_total"], float(rewards[0].detach().cpu()), rel_tol=0.0, abs_tol=1.0e-7
                    ):
                        raise RuntimeError("causal trace wrapped reward mismatch")
                    frames.append(frame)
                    if bool(dones[0].detach().cpu()):
                        break
        finally:
            recorder.restore()
        after = inspect_true23_policy_state(
            {"policy_state_dict": actor.export_true23_policy_state()}, reference_profile=fs.REFERENCE_PROFILE
        )
        if after != before:
            raise RuntimeError("causal trace actor mutated")
    finally:
        env.close()
    series = trace.frames_to_series(frames)
    for name in (
        "anchor_error_xyz",
        "projected_gravity",
        "root_linear_velocity_b",
        "root_angular_velocity_b",
    ):
        series[name] = [frame[name] for frame in frames]
    return {
        "scenario": scenario,
        "policy_state_sha256": before,
        "completed_transitions": len(frames),
        "terminal_q9": frames[-1]["q9"],
        "termination_names": frames[-1]["termination_names"],
        "episode_return": sum(float(frame["reward_total"]) for frame in frames),
        "task_audit": task_audit,
        "prime": prime,
        "series": series,
    }


def _first_q9(q9: Sequence[int], values: Sequence[float], threshold: float) -> int | None:
    return next((q for q, value in zip(q9, values, strict=True) if value >= threshold), None)


def _comparison(
    nominal: Mapping[str, Any], disturbance: Mapping[str, Any], contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    left = nominal["series"]
    right = disturbance["series"]
    common = min(len(left["q9"]), len(right["q9"]))
    q9 = right["q9"][:common]
    action_delta = [
        max(abs(a - b) for a, b in zip(left["raw_native_action"][i], right["raw_native_action"][i], strict=True))
        for i in range(common)
    ]
    anchor_norm = [math.sqrt(sum(value * value for value in row)) for row in right["anchor_error_xyz"]]
    gravity_xy = [math.hypot(row[0], row[1]) for row in right["projected_gravity"]]
    angular_norm = [math.sqrt(sum(value * value for value in row)) for row in right["root_angular_velocity_b"]]
    names = [term["name"] for term in nominal["task_audit"]["rewards"].get("reward_terms", [])]
    # Task audit is a contract mapping, while recorder layout follows manager order.
    reward_names = (
        list(trace._reward_layout)
        if False
        else [
            "motion_global_root_pos",
            "motion_global_root_ori",
            "motion_body_pos",
            "motion_body_ori",
            "motion_body_lin_vel",
            "motion_body_ang_vel",
            "action_rate_l2",
            "joint_limit",
            "self_collisions",
            "action_target_soft_limit_barrier",
            "motion_ankle_pos",
            "motion_ankle_ori",
            "joint_torques_l2",
            "actuator_saturation",
            "v2_ankle_projection_l2",
            "evaluator_aligned_recovery",
            "feet_slip",
            "soft_landing",
            "alive",
            "non_timeout_termination",
            "causal_proof_frame_guard",
            "worst_ee_z_normalized_squared",
            "right_wrist_prethreshold_barrier",
        ]
    )
    del names
    post = [i for i, value in enumerate(q9) if value >= contract["scenario"]["impulse_q9"]]
    reward_delta = {}
    for term_index, name in enumerate(reward_names):
        reward_delta[name] = sum(
            right["reward_weighted"][i][term_index] - left["reward_weighted"][i][term_index] for i in post
        )
    joint_max = []
    for joint_index, name in enumerate(NATIVE_JOINT_NAMES):
        deltas = [
            abs(left["raw_native_action"][i][joint_index] - right["raw_native_action"][i][joint_index])
            for i in range(common)
        ]
        joint_max.append(
            {
                "name": name,
                "maximum_raw_action_abs_delta": max(deltas),
                "first_q9_above_0p01": _first_q9(q9, deltas, 0.01),
            }
        )
    joint_max.sort(key=lambda item: item["maximum_raw_action_abs_delta"], reverse=True)
    thresholds = contract["thresholds"]
    return {
        "common_q9_first": q9[0],
        "common_q9_last": q9[-1],
        "impulse_q9": contract["scenario"]["impulse_q9"],
        "first_raw_action_delta_q9": {
            str(value): _first_q9(q9, action_delta, float(value)) for value in thresholds["raw_action_linf"]
        },
        "disturbance_anchor_error_threshold_q9": {
            str(value): _first_q9(q9, anchor_norm, float(value)) for value in thresholds["anchor_position_error_m"]
        },
        "disturbance_gravity_xy_threshold_q9": {
            str(value): _first_q9(q9, gravity_xy, float(value)) for value in thresholds["gravity_xy_norm"]
        },
        "disturbance_angular_velocity_threshold_q9": {
            str(value): _first_q9(q9, angular_norm, float(value))
            for value in thresholds["root_angular_velocity_norm_radps"]
        },
        "top_raw_action_joint_deltas": joint_max[:10],
        "post_impulse_reward_delta_by_term": dict(
            sorted(reward_delta.items(), key=lambda item: abs(item[1]), reverse=True)
        ),
        "terminal_disturbance_anchor_error_xyz": right["anchor_error_xyz"][-1],
        "terminal_disturbance_projected_gravity": right["projected_gravity"][-1],
        "terminal_disturbance_root_linear_velocity_b": right["root_linear_velocity_b"][-1],
        "terminal_disturbance_root_angular_velocity_b": right["root_angular_velocity_b"][-1],
        "terminal_disturbance_ee_z_error_m": dict(
            zip(trace.EE_BODY_NAMES, right["ee_z_error_m"][-1], strict=True)
        ),
    }


def run(repository_root: Path, output: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("exact-push causal trace preflight failed")
    target = output if output.is_absolute() else root / output
    if os.path.lexists(target):
        raise FileExistsError("exact-push causal trace output exists")
    os.environ.update({"CUDA_VISIBLE_DEVICES": "0", "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0"})
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    with exact._scope(root):  # noqa: SLF001
        state, overlay = rank256._rank256_state(root, audit["base"]["contract"])  # noqa: SLF001
    base_contract = task_space.load_task_space_ppo_contract(root)
    topology = (root / base_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
        strict=True
    )
    motion = (root / base_contract["environment"]["motion_relative_path"]).resolve(strict=True)
    vector = torch.tensor(
        audit["contract"]["scenario"]["exact_vector"], dtype=torch.float32, device=rank256.parent.DEVICE
    ).reshape(1, 6)
    traces = {
        scenario: _run_scenario(
            scenario=scenario,
            state=state,
            topology=topology,
            motion=motion,
            vector=vector,
            expected_policy_sha=overlay["policy_state_sha256"],
        )
        for scenario in ("nominal", "disturbance")
    }
    comparison = _comparison(traces["nominal"], traces["disturbance"], audit["contract"])
    result = {
        "schema_version": 1,
        "kind": KIND,
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "source_overlay": overlay,
        "traces": traces,
        "comparison": comparison,
        "training_transitions": 0,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "candidate_selected": False,
        "support_qualified": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    current = preflight(root)
    if (
        current.get("ready") is not True
        or current.get("material_manifest_sha256") != audit["material_manifest_sha256"]
    ):
        raise RuntimeError("exact-push causal trace materials changed")
    trace.write_json_exclusive(target, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=ROOT)
    execute = sub.add_parser("trace")
    execute.add_argument("--repository-root", type=Path, default=ROOT)
    execute.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = (
        preflight(args.repository_root) if args.command == "preflight" else run(args.repository_root, args.output)
    )
    print(
        json.dumps(
            report if args.command == "preflight" else report["comparison"],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
