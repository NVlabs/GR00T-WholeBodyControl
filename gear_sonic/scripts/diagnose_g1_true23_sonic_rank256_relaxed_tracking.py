"""Test whether rank256 physically recovers when only tracking aborts are disabled."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
from types import MethodType
from typing import Any

import numpy as np
import torch

from gear_sonic.scripts import train_g1_true23_sonic_rank256_disturbance_survival_score as rank256
from gear_sonic.trl.mjlab import (
    sonic_task_space_ppo_full_support_runner as fs,
    sonic_task_space_ppo_runner as task_space,
)
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file
from gear_sonic.utils.g1_true23_native124_21204_composite_mjlab import load_composite_mjlab_contract
import gear_sonic.utils.g1_true23_native124_selected_v2_ankle_evaluation as evidence

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_relaxed_tracking_diagnostic_v1.json"
)
CONTRACT_SHA256 = "eb3b421eaef6431c14e5af8d63bc00b769bba86af24a2af3e77b32aa02e6f568"
KIND = "g1_true23_sonic_rank256_relaxed_tracking_diagnostic_v1"
DISABLED_TERMINATIONS = ("anchor_pos", "anchor_ori", "ee_body_pos")
PRESERVED_TERMINATIONS = ("time_out", "v2_raw_clip")
STEPS = 510
IMPULSE_TRANSITION = 241
SEED = 2069156915


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("relaxed diagnostic contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    scenario = body.get("scenario", {})
    override = body.get("tracking_abort_override", {})
    boundaries = body.get("boundaries", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_relaxed_tracking_diagnostic_contract_v1"
        or scenario.get("seed") != SEED
        or scenario.get("steps") != STEPS
        or scenario.get("impulse_transition") != IMPULSE_TRANSITION
        or override.get("disabled_only") != list(DISABLED_TERMINATIONS)
        or override.get("preserved") != list(PRESERVED_TERMINATIONS)
        or boundaries.get("policy_mutations") != 0
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("relaxed diagnostic contract semantic mismatch")
    candidate = body["candidate"]
    manifest = (root / candidate["manifest_path"]).resolve(strict=True)
    if manifest.is_symlink() or sha256_file(manifest) != candidate["manifest_sha256"]:
        raise ValueError("relaxed diagnostic candidate manifest mismatch")
    parent = body["parent_evidence"]
    result = Path(parent["grouped_result_path"]).resolve(strict=True)
    if result.is_symlink() or sha256_file(result) != parent["grouped_result_sha256"]:
        raise ValueError("relaxed diagnostic parent evidence mismatch")
    return body


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract = _load_contract(root)
        source_contract = rank256._load_contract(root)  # noqa: SLF001
        baseline, overlay = rank256._rank256_state(root, source_contract)  # noqa: SLF001
        base_contract = task_space.load_task_space_ppo_contract(root)
        topology = (root / base_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
            strict=True
        )
        motion = (root / base_contract["environment"]["motion_relative_path"]).resolve(strict=True)
        sources = {
            CONTRACT_RELATIVE_PATH.as_posix(): sha256_file(root / CONTRACT_RELATIVE_PATH),
            "gear_sonic/scripts/diagnose_g1_true23_sonic_rank256_relaxed_tracking.py": sha256_file(
                Path(__file__).resolve()
            ),
        }
        material = {
            "contract_sha256": CONTRACT_SHA256,
            "overlay": overlay,
            "topology_sha256": sha256_file(topology),
            "motion_sha256": sha256_file(motion),
            "sources": sources,
        }
        del baseline
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_relaxed_tracking_preflight_v1",
            "ready": True,
            "contract": contract,
            "overlay": overlay,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha(material),
            "simulator_constructed": False,
            "simulator_steps": 0,
            "policy_mutations": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_relaxed_tracking_preflight_v1",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "simulator_steps": 0,
            "policy_mutations": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "hardware_authorized": False,
        }


def _tracking_errors(raw_env: Any) -> dict[str, float]:
    from mjlab.utils.lab_api.math import quat_apply_inverse

    command = raw_env.command_manager.get_term("motion")
    anchor_position = float(
        torch.linalg.vector_norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1)[0].detach().cpu()
    )
    motion_gravity = quat_apply_inverse(
        command.anchor_quat_w,
        raw_env.scene["robot"].data.gravity_vec_w,
    )
    robot_gravity = quat_apply_inverse(
        command.robot_anchor_quat_w,
        raw_env.scene["robot"].data.gravity_vec_w,
    )
    anchor_orientation = float(torch.abs(motion_gravity[:, 2] - robot_gravity[:, 2])[0].detach().cpu())
    ee = task_space._body_z_errors(  # noqa: SLF001
        raw_env,
        "motion",
        task_space.EE_TERMINATION_BODY_NAMES,
    )[0]
    ee_cpu = ee.detach().cpu().to(torch.float64)
    return {
        "anchor_position_m": anchor_position,
        "anchor_orientation_gravity_z_difference": anchor_orientation,
        "maximum_end_effector_z_m": float(torch.amax(ee_cpu)),
        "left_ankle_z_m": float(ee_cpu[0]),
        "right_ankle_z_m": float(ee_cpu[1]),
        "left_wrist_z_m": float(ee_cpu[2]),
        "right_wrist_z_m": float(ee_cpu[3]),
    }


def _never_terminate(raw_env: Any, **_kwargs: Any) -> torch.Tensor:
    """Diagnostic override: preserve term object/primer state but never abort."""

    return torch.zeros(int(raw_env.num_envs), dtype=torch.bool, device=raw_env.device)


class _TerminalRecorder:
    def __init__(self, raw_env: Any, velocity_limits: np.ndarray) -> None:
        self.raw_env = raw_env
        self.velocity_limits = velocity_limits
        self._original_reset = raw_env._reset_idx
        self._armed: dict[str, Any] | None = None
        self.captured: dict[str, Any] | None = None

        def observed_reset(_env: Any, env_ids: torch.Tensor | None = None) -> None:
            if self._armed is not None and int(_env.common_step_counter) > 0:
                if self.captured is not None:
                    raise RuntimeError("relaxed terminal captured twice")
                self.captured = {
                    **self._armed,
                    "episode_length_pre_reset": int(_env.episode_length_buf[0].detach().cpu()),
                    "termination_names": evidence._termination_names(_env),  # noqa: SLF001
                    "evidence": evidence._step_evidence(_env, self.velocity_limits),  # noqa: SLF001
                    "action_chain": task_space.capture_student_action_chain(_env),
                    "tracking_errors": _tracking_errors(_env),
                }
            self._original_reset(env_ids)

        raw_env._reset_idx = MethodType(observed_reset, raw_env)

    def arm(self, *, transition: int, q9: int) -> None:
        if self._armed is not None:
            raise RuntimeError("relaxed terminal recorder already armed")
        self._armed = {"transition": transition, "q9": q9}
        self.captured = None

    def finish(self, *, done: bool) -> dict[str, Any] | None:
        captured = self.captured
        self._armed = None
        self.captured = None
        if done != (captured is not None):
            raise RuntimeError("relaxed done/terminal capture mismatch")
        return captured

    def restore(self) -> None:
        self.raw_env._reset_idx = self._original_reset


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError("relaxed diagnostic output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(repository_root: Path, output: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("relaxed diagnostic preflight failed")
    target = output if output.is_absolute() else root / output
    if os.path.lexists(target):
        raise FileExistsError("relaxed diagnostic output exists")
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
        }
    )
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    random.seed(SEED)
    np.random.seed(SEED % (2**32))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    contract = audit["contract"]
    source_contract = rank256._load_contract(root)  # noqa: SLF001
    state, overlay = rank256._rank256_state(root, source_contract)  # noqa: SLF001
    base_contract = task_space.load_task_space_ppo_contract(root)
    topology = (root / base_contract["actor_initialization"]["topology_checkpoint_relative_path"]).resolve(
        strict=True
    )
    motion = (root / base_contract["environment"]["motion_relative_path"]).resolve(strict=True)
    cfg = task_space.make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=1)
    cfg.seed = SEED
    strict_audit = task_space.audit_task_space_ppo_env_cfg(cfg, expected_num_envs=1)
    original_terminations = tuple(cfg.terminations)
    for name in DISABLED_TERMINATIONS:
        cfg.terminations[name].func = _never_terminate
    if tuple(cfg.terminations) != original_terminations:
        raise RuntimeError("relaxed diagnostic termination override drift")
    composite = load_composite_mjlab_contract(root)
    velocity_limits = np.asarray(composite["nominal_gate"]["velocity_limit_hardware_radps"], dtype=np.float32)
    env = ManagerBasedRlEnv(cfg=cfg, device=rank256.parent.DEVICE)
    accumulator = evidence.RolloutEvidenceAccumulator()
    tracking_max = {
        "anchor_position_m": 0.0,
        "anchor_orientation_gravity_z_difference": 0.0,
        "maximum_end_effector_z_m": 0.0,
        "left_ankle_z_m": 0.0,
        "right_ankle_z_m": 0.0,
        "left_wrist_z_m": 0.0,
        "right_wrist_z_m": 0.0,
    }
    first_breach: int | None = None
    last_breach: int | None = None
    recovered_transition: int | None = None
    stable_streak = 0
    completed = 0
    terminal_q9: int | None = None
    termination_names: list[str] = []
    pull_plug_reason: str | None = None
    action_mismatch_count = 0
    raw_clip_count = 0
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
        prime = prime_sonic_true23_training_environment(wrapped)
        actor = rank256._actor(state, topology)  # noqa: SLF001
        actor.eval()
        actor_hash_before = inspect_true23_policy_state(
            {"policy_state_dict": actor.export_true23_policy_state()},
            reference_profile=fs.REFERENCE_PROFILE,
        )
        observations = wrapped.get_observations()
        recorder = _TerminalRecorder(env, velocity_limits)
        vector = torch.from_numpy(np.asarray(contract["scenario"]["exact_vector"], dtype=np.float32)).reshape(1, 6)
        vector = vector.to(device=rank256.parent.DEVICE)
        try:
            with torch.inference_mode():
                for transition in range(STEPS):
                    q9 = evidence._q9(env.command_manager.get_term("motion"))  # noqa: SLF001
                    if q9 != 9 + transition:
                        pull_plug_reason = "q9_discontinuity"
                        break
                    action = actor(observations, stochastic_output=False)
                    if transition == IMPULSE_TRANSITION:
                        rank256._apply_impulse(wrapped, vector)  # noqa: SLF001
                    recorder.arm(transition=transition, q9=q9)
                    observations, _reward, dones, _extras = wrapped.step(action)
                    done = bool(dones[0].detach().cpu())
                    terminal = recorder.finish(done=done)
                    if terminal is None:
                        step = evidence._step_evidence(env, velocity_limits)  # noqa: SLF001
                        chain = task_space.capture_student_action_chain(env)
                        errors = _tracking_errors(env)
                    else:
                        step = terminal["evidence"]
                        chain = terminal["action_chain"]
                        errors = terminal["tracking_errors"]
                    mismatch, clips = task_space._action_chain_mismatch_count(action, chain)  # noqa: SLF001
                    action_mismatch_count += mismatch
                    raw_clip_count += clips
                    accumulator.add(step)
                    completed += 1
                    for name, value in errors.items():
                        tracking_max[name] = max(tracking_max[name], value)
                    thresholds = contract["tracking_abort_override"]["would_abort_thresholds"]
                    breached = bool(
                        errors["anchor_position_m"] > thresholds["anchor_position_m"]
                        or errors["anchor_orientation_gravity_z_difference"]
                        > thresholds["anchor_orientation_gravity_z_difference"]
                        or errors["maximum_end_effector_z_m"] > thresholds["end_effector_z_m"]
                    )
                    if breached:
                        first_breach = transition if first_breach is None else first_breach
                        last_breach = transition
                        stable_streak = 0
                    elif first_breach is not None and recovered_transition is None:
                        stable_streak += 1
                        if stable_streak >= contract["tracking_abort_override"]["stable_recovery_steps"]:
                            recovered_transition = transition
                    summary = accumulator.report()
                    scalars = summary["safety_scalars"]
                    if (
                        summary["hard_safety_violation_count"] > 0
                        or float(scalars["base_height_m"]["minimum"]) < 0.45
                        or float(scalars["base_tilt_rad"]["maximum"]) > 1.0
                    ):
                        pull_plug_reason = "physical_safety_gate"
                        break
                    if done:
                        terminal_q9 = q9
                        termination_names = terminal["termination_names"]
                        break
        finally:
            recorder.restore()
        actor_hash_after = inspect_true23_policy_state(
            {"policy_state_dict": actor.export_true23_policy_state()},
            reference_profile=fs.REFERENCE_PROFILE,
        )
    finally:
        env.close()
    rollout = accumulator.report()
    physically_safe = bool(
        pull_plug_reason is None
        and completed == STEPS
        and terminal_q9 == 518
        and termination_names == ["time_out"]
        and rollout["hard_safety_violation_count"] == 0
        and raw_clip_count == 0
        and action_mismatch_count == 0
    )
    report = {
        "schema_version": 1,
        "kind": KIND,
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "strict_task_audit_before_override": strict_audit,
        "prime": prime,
        "original_terminations": list(original_terminations),
        "disabled_tracking_terminations": list(DISABLED_TERMINATIONS),
        "preserved_terminations": list(PRESERVED_TERMINATIONS),
        "completed_transitions": completed,
        "terminal_q9": terminal_q9,
        "termination_names": termination_names,
        "impulse_transition": IMPULSE_TRANSITION,
        "impulse_q9": 250,
        "tracking": {
            "maximum_errors": tracking_max,
            "first_would_abort_transition": first_breach,
            "first_would_abort_q9": None if first_breach is None else 9 + first_breach,
            "last_would_abort_transition": last_breach,
            "last_would_abort_q9": None if last_breach is None else 9 + last_breach,
            "stable_recovery_transition": recovered_transition,
            "stable_recovery_q9": None if recovered_transition is None else 9 + recovered_transition,
        },
        "rollout": rollout,
        "action_semantics_mismatch_count": action_mismatch_count,
        "raw_clip_required_count": raw_clip_count,
        "pull_plug_reason": pull_plug_reason,
        "physically_safe_through_q518": physically_safe,
        "strict_tracking_qualification": False,
        "actor_state_sha256_before": actor_hash_before,
        "actor_state_sha256_after": actor_hash_after,
        "actor_unchanged": actor_hash_after == actor_hash_before == overlay["policy_state_sha256"],
        "policy_mutations": 0,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "training_performed": False,
        "support_qualified": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    _write_json_exclusive(target, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=ROOT)
    execute = sub.add_parser("run")
    execute.add_argument("--repository-root", type=Path, default=ROOT)
    execute.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        report = preflight(args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
        return 0 if report.get("ready") is True else 1
    report = run(args.repository_root, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    return 0 if report["physically_safe_through_q518"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
