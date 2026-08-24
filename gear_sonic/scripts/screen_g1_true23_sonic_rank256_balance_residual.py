"""Screen bounded pitch-balance residuals on exact-push rank256 failure."""

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
    train_g1_true23_sonic_rank256_disturbance_survival_score as rank256,
    train_g1_true23_sonic_rank256_exact_impulse_survival_score as exact,
)
from gear_sonic.trl.mjlab import (
    sonic_task_space_ppo_full_support_runner as fs,
    sonic_task_space_ppo_runner as task_space,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_rank256_balance_residual_screen_v1.json"
)
CONTRACT_SHA256 = "ac9dfba6ce4652faa7b9af65dd12c6fc1f6b3f1c171d33fda5ddce083076c113"
RESULT_FILENAME = "rank256_balance_residual_screen_result.json"
EXTRA_SOURCE_RELATIVE_PATHS: tuple[Path, ...] = ()
VARIANTS = (
    ("baseline", 0.0, 0.0),
    ("ankle_positive", 1.0, 0.0),
    ("ankle_negative", -1.0, 0.0),
    ("hip_positive", 0.0, 1.0),
    ("hip_negative", 0.0, -1.0),
    ("ankle_positive_hip_negative", 1.0, -1.0),
    ("ankle_negative_hip_positive", -1.0, 1.0),
    ("ankle_hip_positive", 1.0, 1.0),
    ("ankle_hip_negative", -1.0, -1.0),
)
HIP_PITCH_INDICES = (0, 1)
ANKLE_PITCH_INDICES = (15, 16)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _load_contract(root: Path) -> Mapping[str, Any]:
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("balance residual contract mismatch")
    body = json.loads(path.read_text(encoding="utf-8"))
    controller = body.get("controller", {})
    evaluation = body.get("evaluation", {})
    boundaries = body.get("boundaries", {})
    indices = controller.get("native_pitch_indices", {})
    if (
        body.get("kind") != "g1_true23_sonic_rank256_balance_residual_screen_contract_v1"
        or controller.get("variants") != [list(value) for value in VARIANTS]
        or tuple(indices.values()) != (*HIP_PITCH_INDICES, *ANKLE_PITCH_INDICES)
        or controller.get("residual_gain") != 0.4
        or controller.get("residual_abs_clip") != 0.4
        or evaluation.get("disturbance_first") is not True
        or evaluation.get("nominal_top_improving_count") != 3
        or boundaries.get("training_transitions") != 0
        or boundaries.get("robot_or_network_commands_permitted") is not False
        or boundaries.get("hardware_authorized") is not False
    ):
        raise ValueError("balance residual contract semantic mismatch")
    for name, entry in body["parents"].items():
        artifact = Path(entry["path"])
        artifact = (
            (root / artifact).resolve(strict=True) if not artifact.is_absolute() else artifact.resolve(strict=True)
        )
        if artifact.is_symlink() or sha256_file(artifact) != entry["sha256"]:
            raise ValueError(f"balance residual parent mismatch: {name}")
    return body


def preflight(repository_root: Path) -> Mapping[str, Any]:
    try:
        root = repository_root.expanduser().resolve(strict=True)
        contract = _load_contract(root)
        with exact._scope(root):  # noqa: SLF001
            base = rank256.preflight(root)
        if base.get("ready") is not True:
            raise RuntimeError("balance residual base preflight failed")
        sources = {
            CONTRACT_RELATIVE_PATH.as_posix(): CONTRACT_SHA256,
            "gear_sonic/scripts/screen_g1_true23_sonic_rank256_balance_residual.py": sha256_file(
                root / "gear_sonic/scripts/screen_g1_true23_sonic_rank256_balance_residual.py"
            ),
        }
        sources.update(
            {
                relative.as_posix(): sha256_file((root / relative).resolve(strict=True))
                for relative in EXTRA_SOURCE_RELATIVE_PATHS
            }
        )
        material = {
            "base_material_manifest_sha256": base["material_manifest_sha256"],
            "parents": {name: entry["sha256"] for name, entry in contract["parents"].items()},
            "sources": sources,
        }
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_rank256_balance_residual_screen_preflight_v1",
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
            "kind": "g1_true23_sonic_rank256_balance_residual_screen_preflight_v1",
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


class _BalanceResidualPolicy(torch.nn.Module):
    def __init__(
        self,
        base: torch.nn.Module,
        raw_env: Any,
        signal_cfg: Mapping[str, Any],
        residual_gain: float,
        residual_clip: float,
        applied_residual_clip: float,
        ankle_sign: float,
        hip_sign: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.raw_env = raw_env
        self.signal_cfg = dict(signal_cfg)
        self.residual_gain = residual_gain
        self.residual_clip = residual_clip
        self.applied_residual_clip = applied_residual_clip
        self.ankle_sign = ankle_sign
        self.hip_sign = hip_sign
        self.activation_count = 0
        self.maximum_abs_residual = 0.0

    def export_true23_policy_state(self) -> Mapping[str, torch.Tensor]:
        return self.base.export_true23_policy_state()  # type: ignore[attr-defined, no-any-return]

    def forward(self, observations: Any, stochastic_output: bool = False) -> torch.Tensor:
        from mjlab.utils.lab_api.math import quat_apply_inverse

        action = self.base(observations, stochastic_output=stochastic_output)
        if self.ankle_sign == 0.0 and self.hip_sign == 0.0:
            return action
        command = self.raw_env.command_manager.get_term("motion")
        robot = self.raw_env.scene["robot"].data
        gravity_w = robot.gravity_vec_w
        reference_gravity = quat_apply_inverse(command.anchor_quat_w, gravity_w)
        measured_gravity = quat_apply_inverse(command.robot_anchor_quat_w, gravity_w)
        linear_error = quat_apply_inverse(
            command.robot_anchor_quat_w,
            command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w,
        )
        angular_error = quat_apply_inverse(
            command.robot_anchor_quat_w,
            command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w,
        )
        cfg = self.signal_cfg
        signal = (
            float(cfg["orientation_error_gain"]) * (reference_gravity[:, 0] - measured_gravity[:, 0])
            + float(cfg["angular_velocity_error_gain"]) * angular_error[:, 1]
            + float(cfg["linear_velocity_error_gain"]) * linear_error[:, 0]
        )
        deadband = float(cfg["deadband"])
        signal = torch.where(torch.abs(signal) >= deadband, signal, torch.zeros_like(signal))
        signal = torch.clamp(signal, -float(cfg["signal_clip"]), float(cfg["signal_clip"]))
        residual = torch.clamp(signal * self.residual_gain, -self.residual_clip, self.residual_clip)
        result = action.clone()
        ankle_residual = torch.clamp(
            self.ankle_sign * residual,
            -self.applied_residual_clip,
            self.applied_residual_clip,
        )
        hip_residual = torch.clamp(
            self.hip_sign * residual,
            -self.applied_residual_clip,
            self.applied_residual_clip,
        )
        for index in ANKLE_PITCH_INDICES:
            result[:, index] += ankle_residual
        for index in HIP_PITCH_INDICES:
            result[:, index] += hip_residual
        maximum = torch.maximum(torch.abs(ankle_residual), torch.abs(hip_residual))
        if bool(torch.any(maximum != 0.0)):
            self.activation_count += 1
            self.maximum_abs_residual = max(self.maximum_abs_residual, float(torch.amax(maximum).detach().cpu()))
        return result


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _evaluate(
    *,
    variant: tuple[str, float, float],
    scenario: str,
    state: Mapping[str, torch.Tensor],
    topology: Path,
    motion: Path,
    contract: Mapping[str, Any],
    material_sha: str,
) -> Mapping[str, Any]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment

    _seed(rank256.DISTURBANCE_SEED)
    cfg = task_space.make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=1)
    cfg.seed = rank256.DISTURBANCE_SEED
    audit = task_space.audit_task_space_ppo_env_cfg(cfg, expected_num_envs=1)
    env = ManagerBasedRlEnv(cfg=cfg, device=rank256.parent.DEVICE)
    try:
        wrapped_base = RslRlVecEnvWrapper(env, clip_actions=None)
        vector = torch.tensor(rank256.FAILED_VECTOR, dtype=torch.float32, device=rank256.parent.DEVICE).reshape(
            1, 6
        )
        wrapped = rank256._ImpulseProxy(wrapped_base, vector) if scenario == "disturbance" else wrapped_base  # noqa: SLF001
        prime = prime_sonic_true23_training_environment(wrapped)
        base_actor = rank256._actor(state, topology)  # noqa: SLF001
        policy = _BalanceResidualPolicy(
            base_actor,
            env,
            contract["controller"]["signal"],
            float(contract["controller"]["residual_gain"]),
            float(contract["controller"]["residual_abs_clip"]),
            float(
                contract["controller"].get(
                    "maximum_applied_residual_abs",
                    contract["controller"]["residual_abs_clip"],
                )
            ),
            variant[1],
            variant[2],
        ).to(rank256.parent.DEVICE)
        original_seed = fs.FIXED_SEED
        try:
            fs.FIXED_SEED = rank256.DISTURBANCE_SEED
            record = fs.evaluate_full_support_policy(
                policy=policy,
                wrapped_env=wrapped,
                update_count=1,
                evaluation_seed=rank256.DISTURBANCE_SEED,
            )
        finally:
            fs.FIXED_SEED = original_seed
    finally:
        env.close()
    return {
        **record,
        "variant": variant[0],
        "ankle_sign": variant[1],
        "hip_sign": variant[2],
        "scenario": scenario,
        "residual_activation_count": policy.activation_count,
        "maximum_abs_residual": policy.maximum_abs_residual,
        "task_audit": audit,
        "prime": prime,
        "material_manifest_sha256": material_sha,
    }


def run(repository_root: Path, requested_run_dir: Path) -> Mapping[str, Any]:
    root = repository_root.expanduser().resolve(strict=True)
    audit = preflight(root)
    if audit.get("ready") is not True:
        raise RuntimeError("balance residual preflight failed")
    if os.path.lexists(requested_run_dir):
        raise FileExistsError("balance residual run directory exists")
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
    from gear_sonic.scripts import train_g1_true23_sonic_task_space_ppo_full_support as publication

    run_dir = publication._create_run_dir_exclusive(requested_run_dir)  # noqa: SLF001
    publication._write_json_exclusive(run_dir / "preflight.json", audit)  # noqa: SLF001
    records = []
    for variant in VARIANTS:
        record = _evaluate(
            variant=variant,
            scenario="disturbance",
            state=state,
            topology=topology,
            motion=motion,
            contract=audit["contract"],
            material_sha=audit["material_manifest_sha256"],
        )
        publication._write_json_exclusive(run_dir / "evaluations" / f"{variant[0]}_disturbance.json", record)  # noqa: SLF001
        records.append(record)
        print(
            f"{variant[0]} disturbance steps={record['completed_transitions']} q9={record['terminal_q9']}",
            flush=True,
        )  # noqa: T201
    baseline_disturbance = next(record for record in records if record["variant"] == "baseline")
    improving = sorted(
        (
            record
            for record in records
            if record["variant"] != "baseline"
            and int(record["completed_transitions"]) > int(baseline_disturbance["completed_transitions"])
        ),
        key=lambda record: int(record["completed_transitions"]),
        reverse=True,
    )[: int(audit["contract"]["evaluation"]["nominal_top_improving_count"])]
    nominal_variants = [
        VARIANTS[0],
        *(next(value for value in VARIANTS if value[0] == record["variant"]) for record in improving),
    ]
    for variant in nominal_variants:
        record = _evaluate(
            variant=variant,
            scenario="nominal",
            state=state,
            topology=topology,
            motion=motion,
            contract=audit["contract"],
            material_sha=audit["material_manifest_sha256"],
        )
        publication._write_json_exclusive(run_dir / "evaluations" / f"{variant[0]}_nominal.json", record)  # noqa: SLF001
        records.append(record)
        print(
            f"{variant[0]} nominal steps={record['completed_transitions']} q9={record['terminal_q9']}", flush=True
        )  # noqa: T201
    baseline_nominal = next(
        record for record in records if record["variant"] == "baseline" and record["scenario"] == "nominal"
    )
    candidates = []
    for disturbance in improving:
        nominal = next(
            (
                record
                for record in records
                if record["variant"] == disturbance["variant"] and record["scenario"] == "nominal"
            ),
            None,
        )
        if nominal is not None and int(nominal["completed_transitions"]) >= int(
            baseline_nominal["completed_transitions"]
        ):
            candidates.append((disturbance, nominal))
    selected = max(
        candidates,
        key=lambda pair: (pair[0]["completed_transitions"], pair[1]["completed_transitions"]),
        default=None,
    )
    assessment = {
        "baseline_disturbance_completed_transitions": baseline_disturbance["completed_transitions"],
        "baseline_nominal_completed_transitions": baseline_nominal["completed_transitions"],
        "candidate_selected": selected is not None,
        "selected_variant": None if selected is None else selected[0]["variant"],
        "selected_disturbance_completed_transitions": None
        if selected is None
        else selected[0]["completed_transitions"],
        "selected_nominal_completed_transitions": None
        if selected is None
        else selected[1]["completed_transitions"],
        "support_qualified": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    result = {
        "schema_version": 1,
        "kind": "g1_true23_sonic_rank256_balance_residual_screen_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": audit["material_manifest_sha256"],
        "source_overlay": overlay,
        "evaluations": records,
        "assessment": assessment,
        "training_transitions": 0,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "candidate_selected": assessment["candidate_selected"],
        "support_qualified": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }
    current = preflight(root)
    if (
        current.get("ready") is not True
        or current["material_manifest_sha256"] != audit["material_manifest_sha256"]
    ):
        raise RuntimeError("balance residual materials changed")
    publication._write_json_exclusive(run_dir / RESULT_FILENAME, result)  # noqa: SLF001
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--repository-root", type=Path, default=ROOT)
    screen = sub.add_parser("screen")
    screen.add_argument("--repository-root", type=Path, default=ROOT)
    screen.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = (
        preflight(args.repository_root) if args.command == "preflight" else run(args.repository_root, args.run_dir)
    )
    print(
        json.dumps(
            report if args.command == "preflight" else report["assessment"],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )  # noqa: T201
    return 0 if args.command == "preflight" and report.get("ready") is True or args.command == "screen" else 1


if __name__ == "__main__":
    raise SystemExit(main())
