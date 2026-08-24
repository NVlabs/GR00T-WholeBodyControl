"""Qualify fixed-motion SONIC plus bounded phase ankle-balance residual."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from gear_sonic.scripts import (
    collect_g1_true23_sonic_rank256_shifted_base_causal_recovery_score as shifted,
    screen_g1_true23_sonic_rank256_balance_residual as balance,
    screen_g1_true23_sonic_rank256_shifted_base_causal_recovery as shifted_screen,
    train_g1_true23_sonic_rank256_disturbance_survival_score as disturbance,
    train_g1_true23_sonic_task_space_ppo_full_support as publication,
)
from gear_sonic.trl.mjlab import (
    sonic_task_space_ppo_full_support_runner as fs,
    sonic_task_space_ppo_runner as task_space,
)
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_phase_balance_qualification_v1.json"
)
CONTRACT_SHA256 = "c9224bb9bbf360bdc3a1dcc4918a008a149bda26b5b0e08943e58cb3483e96c7"
CONTRACT_KIND = "g1_true23_sonic_rank256_phase_balance_qualification_contract_v1"
SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    Path("gear_sonic/scripts/qualify_g1_true23_sonic_rank256_phase_balance.py"),
    Path("gear_sonic/scripts/screen_g1_true23_sonic_rank256_balance_residual.py"),
    Path("gear_sonic/scripts/collect_g1_true23_sonic_rank256_shifted_base_causal_recovery_score.py"),
    Path("gear_sonic/scripts/screen_g1_true23_sonic_rank256_shifted_base_causal_recovery.py"),
)
MODES = ("nominal", "disturbance")
RUN_DIR_DEFAULT = Path("/root/g1_true23_runs/sonic_rank256_phase_balance_qualification_v1")
RESULT_FILENAME = "rank256_phase_balance_qualification_result.json"


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("phase-balance contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    controller = body.get("controller", {})
    qualification = body.get("qualification", {})
    boundaries = body.get("boundaries", {})
    if (
        body.get("kind") != CONTRACT_KIND
        or body.get("parents", {}).get("shifted_base_policy_state_sha256")
        != "6acada92dd6e2700499c64255e077c7e844aa0f96b0700e2e2c5640606ae0650"
        or controller.get("phase_window_q9_inclusive") != [180, 300]
        or controller.get("residual_gain") != 0.4
        or controller.get("ankle_pitch_coefficient") != -8.0
        or controller.get("hip_pitch_coefficient") != 0.0
        or controller.get("maximum_applied_raw_residual_abs") != 1.0
        or controller.get("native_ankle_pitch_indices") != [15, 16]
        or controller.get("safe_target_transform_application_count") != 1
        or qualification.get("steps") != 510
        or qualification.get("impulse_transition") != disturbance.IMPULSE_TRANSITION
        or len(qualification.get("nominal_seeds", ())) != 10
        or len(qualification.get("disturbance_seeds", ())) != 10
        or len(qualification.get("disturbance_vectors", ())) != 10
        or qualification.get("required_completed_transitions") != 510
        or qualification.get("required_terminal_q9") != 518
        or qualification.get("required_termination_names") != ["time_out"]
        or boundaries.get("training_transitions") != 0
        or boundaries.get("robot_or_network_commands_permitted") is not False
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("phase-balance contract semantic mismatch")
    late = controller.get("late_policy")
    if late is not None and (
        not isinstance(late, Mapping)
        or late.get("switch_after_q9") != 400
        or late.get("direction_scale") != -1.0
        or late.get("policy_state_sha256") != "044df762012a7f0e2e3bf1dc57542a8399fa27d1bbf6f8eacb099a0191f3d2eb"
        or late.get("no_blend") is not True
    ):
        raise ValueError("phase-balance late policy mismatch")
    for name, entry in body["parents"].items():
        if name == "shifted_base_policy_state_sha256":
            continue
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"phase-balance parent schema mismatch: {name}")
        path = Path(entry["path"]).expanduser().resolve(strict=True)
        if path.is_symlink() or not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"phase-balance parent mismatch: {name}")
    vectors = np.asarray(qualification["disturbance_vectors"], dtype=np.float32)
    if vectors.shape != (10, 6) or not np.isfinite(vectors).all():
        raise ValueError("phase-balance disturbance vector matrix mismatch")
    return body


def _inputs(root: Path) -> tuple[Mapping[str, Any], dict[str, torch.Tensor], Mapping[str, Any], Path, Path]:
    contract = _load_contract(root)
    _parent, state, overlay, topology, motion = shifted._inputs(root)  # noqa: SLF001
    observed = inspect_true23_policy_state({"policy_state_dict": state}, reference_profile=fs.REFERENCE_PROFILE)
    if observed != contract["parents"]["shifted_base_policy_state_sha256"]:
        raise ValueError("phase-balance base policy mismatch")
    return contract, state, overlay, topology, motion


def preflight(repository_root: Path) -> dict[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract, _state, overlay, topology, motion = _inputs(root)
        sources = {
            path.as_posix(): sha256_file((root / path).resolve(strict=True)) for path in SOURCE_RELATIVE_PATHS
        }
        material = {
            "contract_sha256": CONTRACT_SHA256,
            "base_policy_state_sha256": contract["parents"]["shifted_base_policy_state_sha256"],
            "late_policy_state_sha256": contract["controller"].get("late_policy", {}).get("policy_state_sha256"),
            "parent_sha256": {
                name: entry["sha256"]
                for name, entry in contract["parents"].items()
                if name != "shifted_base_policy_state_sha256"
            },
            "source_overlay": overlay,
            "topology_sha256": sha256_file(topology),
            "motion_sha256": sha256_file(motion),
            "sources": sources,
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_phase_balance_qualification_preflight_v1",
            "ready": True,
            "contract": contract,
            "material_manifest": material,
            "material_manifest_sha256": _canonical_sha(material),
            "simulator_constructed": False,
            "evaluation_runs": 0,
            "training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "deployment_ready": False,
            "hardware_authorized": False,
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_phase_balance_qualification_preflight_v1",
            "ready": False,
            "error_type": type(error).__name__,
            "error_message_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
            "simulator_constructed": False,
            "evaluation_runs": 0,
            "training_transitions": 0,
            "optimizer_steps": 0,
            "critic_updates": 0,
            "deployment_ready": False,
            "hardware_authorized": False,
        }


class PhaseBalancePolicy(balance._BalanceResidualPolicy):  # noqa: SLF001
    """Apply exact residual only during contract-bound motion phase."""

    def __init__(self, *args: Any, start_q9: int, stop_q9: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.start_q9 = start_q9
        self.stop_q9 = stop_q9
        self.active_q9_first: int | None = None
        self.active_q9_last: int | None = None

    def forward(self, observations: Any, stochastic_output: bool = False) -> torch.Tensor:
        command = self.raw_env.command_manager.get_term("motion")
        time_steps = command.time_steps
        if type(time_steps) is not torch.Tensor or time_steps.shape != (1,) or time_steps.dtype != torch.long:
            raise ValueError("phase-balance q9 must be int64 [1]")
        q9 = int(time_steps[0].detach().cpu())
        ankle_sign, hip_sign = self.ankle_sign, self.hip_sign
        enabled = self.start_q9 <= q9 <= self.stop_q9
        if not enabled:
            self.ankle_sign = 0.0
            self.hip_sign = 0.0
        try:
            output = super().forward(observations, stochastic_output=stochastic_output)
        finally:
            self.ankle_sign = ankle_sign
            self.hip_sign = hip_sign
        if enabled:
            self.active_q9_first = q9 if self.active_q9_first is None else self.active_q9_first
            self.active_q9_last = q9
        return output


class SwitchedBasePolicy(torch.nn.Module):
    """Switch once from recovery base to nominal-stable late actor."""

    def __init__(self, early: torch.nn.Module, late: torch.nn.Module, raw_env: Any, switch_after_q9: int) -> None:
        super().__init__()
        self.early = early
        self.late = late
        self.raw_env = raw_env
        self.switch_after_q9 = switch_after_q9
        self.late_policy_first_q9: int | None = None
        self.late_policy_last_q9: int | None = None

    def forward(self, observations: Any, stochastic_output: bool = False) -> torch.Tensor:
        command = self.raw_env.command_manager.get_term("motion")
        q9 = int(command.time_steps[0].detach().cpu())
        if q9 <= self.switch_after_q9:
            actor = self.early
        else:
            actor = self.late
            self.late_policy_first_q9 = q9 if self.late_policy_first_q9 is None else self.late_policy_first_q9
            self.late_policy_last_q9 = q9
        return actor(observations, stochastic_output=stochastic_output)

    def export_true23_policy_state(self) -> Mapping[str, torch.Tensor]:
        return self.early.export_true23_policy_state()  # type: ignore[attr-defined, no-any-return]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate(
    *,
    mode: str,
    index: int,
    contract: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    topology: Path,
    motion: Path,
    material_sha: str,
    late_state: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment

    qualification = contract["qualification"]
    controller = contract["controller"]
    seed = int(qualification[f"{mode}_seeds"][index])
    vector = np.zeros(6, dtype=np.float32)
    if mode == "disturbance":
        vector = np.asarray(qualification["disturbance_vectors"][index], dtype=np.float32)
    _seed(seed)
    cfg = task_space.make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=1)
    cfg.seed = seed
    task_audit = task_space.audit_task_space_ppo_env_cfg(cfg, expected_num_envs=1)
    env = ManagerBasedRlEnv(cfg=cfg, device=disturbance.parent.DEVICE)
    original_seed = fs.FIXED_SEED
    try:
        wrapped_base = RslRlVecEnvWrapper(env, clip_actions=None)
        wrapped = (
            disturbance._ImpulseProxy(  # noqa: SLF001
                wrapped_base,
                torch.from_numpy(vector.reshape(1, 6)).to(disturbance.parent.DEVICE),
            )
            if mode == "disturbance"
            else wrapped_base
        )
        prime = prime_sonic_true23_training_environment(wrapped)
        base_actor = disturbance._actor(state, topology)  # noqa: SLF001
        switched: SwitchedBasePolicy | None = None
        late = controller.get("late_policy")
        if late is not None:
            if late_state is None:
                raise ValueError("phase-balance late policy state missing")
            late_actor = disturbance._actor(late_state, topology)  # noqa: SLF001
            switched = SwitchedBasePolicy(
                base_actor,
                late_actor,
                env,
                int(late["switch_after_q9"]),
            ).to(disturbance.parent.DEVICE)
            base_actor = switched
        start_q9, stop_q9 = controller["phase_window_q9_inclusive"]
        policy = PhaseBalancePolicy(
            base_actor,
            env,
            controller["signal"],
            float(controller["residual_gain"]),
            float(controller["residual_gain"]),
            float(controller["maximum_applied_raw_residual_abs"]),
            float(controller["ankle_pitch_coefficient"]),
            float(controller["hip_pitch_coefficient"]),
            start_q9=int(start_q9),
            stop_q9=int(stop_q9),
        ).to(disturbance.parent.DEVICE)
        fs.FIXED_SEED = seed
        record = fs.evaluate_full_support_policy(
            policy=policy,
            wrapped_env=wrapped,
            update_count=1,
            evaluation_seed=seed,
        )
    finally:
        fs.FIXED_SEED = original_seed
        env.close()
    return {
        **record,
        "mode": mode,
        "qualification_index": index,
        "evaluation_seed": seed,
        "disturbance_vector": vector.tolist(),
        "disturbance_vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
        "impulse_applied": mode == "disturbance" and int(record["completed_transitions"]) > 241,
        "residual_activation_count": policy.activation_count,
        "maximum_abs_residual": policy.maximum_abs_residual,
        "active_q9_first": policy.active_q9_first,
        "active_q9_last": policy.active_q9_last,
        "late_policy_first_q9": None if switched is None else switched.late_policy_first_q9,
        "late_policy_last_q9": None if switched is None else switched.late_policy_last_q9,
        "late_policy_state_sha256": None if late is None else late["policy_state_sha256"],
        "task_audit": task_audit,
        "prime": prime,
        "material_manifest_sha256": material_sha,
        "controller_sha256": _canonical_sha(controller),
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def _clean(record: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    qualification = contract["qualification"]
    late = contract["controller"].get("late_policy")
    return (
        record.get("completed_transitions") == qualification["required_completed_transitions"]
        and record.get("terminal_q9") == qualification["required_terminal_q9"]
        and record.get("termination_names") == qualification["required_termination_names"]
        and all(record.get(name) == 0 for name in qualification["required_zero_counts"])
        and record.get("active_q9_first") == contract["controller"]["phase_window_q9_inclusive"][0]
        and record.get("active_q9_last") == contract["controller"]["phase_window_q9_inclusive"][1]
        and record.get("residual_activation_count", 0) > 0
        and 0.0 < float(record.get("maximum_abs_residual", 0.0)) <= 1.0
        and (record.get("mode") != "disturbance" or record.get("impulse_applied") is True)
        and (
            late is None
            or (
                record.get("late_policy_first_q9") == int(late["switch_after_q9"]) + 1
                and record.get("late_policy_last_q9") == qualification["last_action_q9"]
                and record.get("late_policy_state_sha256") == late["policy_state_sha256"]
            )
        )
    )


def initialize(repository_root: Path, requested_run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("phase-balance preflight failed")
    run_dir = publication._create_run_dir_exclusive(requested_run_dir)  # noqa: SLF001
    publication._write_json_exclusive(run_dir / "preflight.json", audit)  # noqa: SLF001
    publication._write_json_exclusive(run_dir / "material_manifest.json", audit["material_manifest"])  # noqa: SLF001
    return audit


def evaluate_one(repository_root: Path, run_dir: Path, mode: str, index: int) -> Mapping[str, Any]:
    if mode not in MODES or index not in range(10):
        raise ValueError("phase-balance evaluation request mismatch")
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("phase-balance preflight failed")
    expected_dir = run_dir.expanduser().resolve(strict=True)
    if expected_dir.is_symlink():
        raise ValueError("phase-balance run directory symlink forbidden")
    output = expected_dir / "evaluations" / f"{mode}_{index:02d}.json"
    if os.path.lexists(output):
        raise FileExistsError("phase-balance evaluation exists")
    contract, state, _overlay, topology, motion = _inputs(root)
    late_state: Mapping[str, torch.Tensor] | None = None
    late = contract["controller"].get("late_policy")
    if late is not None:
        _screen_contract, baseline, direction, _screen_overlay, _screen_topology, _screen_motion = (
            shifted_screen._inputs(root)  # noqa: SLF001
        )
        late_state = shifted_screen._state(baseline, direction, float(late["direction_scale"]))  # noqa: SLF001
        observed_late = inspect_true23_policy_state(
            {"policy_state_dict": late_state}, reference_profile=fs.REFERENCE_PROFILE
        )
        if observed_late != late["policy_state_sha256"]:
            raise ValueError("phase-balance late policy reconstruction mismatch")
    os.environ.update({"CUDA_VISIBLE_DEVICES": "0", "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0"})
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False, deterministic=True)
    torch.cuda.set_device(0)
    record = _evaluate(
        mode=mode,
        index=index,
        contract=contract,
        state=state,
        topology=topology,
        motion=motion,
        material_sha=audit["material_manifest_sha256"],
        late_state=late_state,
    )
    publication._write_json_exclusive(output, record)  # noqa: SLF001
    return record


def finalize(repository_root: Path, run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("phase-balance preflight failed")
    resolved = run_dir.expanduser().resolve(strict=True)
    records: list[Mapping[str, Any]] = []
    for mode in MODES:
        for index in range(10):
            path = (resolved / "evaluations" / f"{mode}_{index:02d}.json").resolve(strict=True)
            records.append(json.loads(path.read_text(encoding="utf-8")))
    contract = audit["contract"]
    expected = {(mode, index) for mode in MODES for index in range(10)}
    observed = {(row.get("mode"), row.get("qualification_index")) for row in records}
    all_pass = observed == expected and all(
        row.get("material_manifest_sha256") == audit["material_manifest_sha256"]
        and row.get("policy_state_sha256") == contract["parents"]["shifted_base_policy_state_sha256"]
        and _clean(row, contract)
        for row in records
    )
    result = {
        "schema_version": 1,
        "kind": "g1_true23_sonic_rank256_phase_balance_qualification_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "controller": contract["controller"],
        "base_policy_state_sha256": contract["parents"]["shifted_base_policy_state_sha256"],
        "evaluation_count": len(records),
        "nominal_pass_count": sum(_clean(row, contract) for row in records if row["mode"] == "nominal"),
        "disturbance_pass_count": sum(_clean(row, contract) for row in records if row["mode"] == "disturbance"),
        "minimum_completed_transitions": min(int(row["completed_transitions"]) for row in records),
        "maximum_abs_residual": max(float(row["maximum_abs_residual"]) for row in records),
        "all_twenty_passed": all_pass,
        "simulator_teleop_candidate": all_pass,
        "fixed_motion_phase_specific": True,
        "training_transitions": 0,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    publication._write_json_exclusive(resolved / RESULT_FILENAME, result)  # noqa: SLF001
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "init", "evaluate-one", "finalize"):
        item = sub.add_parser(name)
        item.add_argument("--repository-root", type=Path, default=ROOT)
        if name != "preflight":
            item.add_argument("--run-dir", type=Path, default=RUN_DIR_DEFAULT)
        if name == "evaluate-one":
            item.add_argument("--mode", choices=MODES, required=True)
            item.add_argument("--index", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.repository_root)
        elif args.command == "init":
            result = initialize(args.repository_root, args.run_dir)
        elif args.command == "evaluate-one":
            result = evaluate_one(args.repository_root, args.run_dir, args.mode, args.index)
        else:
            result = finalize(args.repository_root, args.run_dir)
    except Exception as error:
        print(f"phase-balance qualification failed: {type(error).__name__}: {error}")  # noqa: T201
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))  # noqa: T201
    if args.command == "preflight":
        return 0 if result.get("ready") is True else 1
    if args.command == "finalize":
        return 0 if result.get("all_twenty_passed") is True else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
