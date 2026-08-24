"""Deterministic, simulator-only trace of the immutable task-space PPO v2 pair.

This diagnostic never trains, exports, promotes, or talks to hardware.  It
replays the exact update-0 and update-5 policies from the admitted v2 pilot in
fresh one-environment MJLab instances and records the reward/action/contact
state that existed before terminal autoreset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
from types import MethodType
from typing import Any

import numpy as np
import torch

from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    inspect_true23_policy_state,
    sha256_file,
)
from gear_sonic.utils.g1_23dof_mjlab_training import validate_file_manifest

SCHEMA_VERSION = 1
TRACE_KIND = "g1_true23_sonic_task_space_checkpoint_trace_v1"
FAILURE_KIND = "g1_true23_sonic_task_space_checkpoint_trace_failure_v1"
PREFLIGHT_KIND = "g1_true23_sonic_task_space_checkpoint_trace_preflight_v1"

FIXED_SEED = 20260805
DEVICE = "cuda:0"
ACTION_DIM = 23
EE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
)
CONTACT_SITE_NAMES = ("left_foot", "right_foot")
EXPECTED_UPDATES = (0, 5)
EXPECTED_CONTRACT_SHA256 = "a0225ad7d08d7037ec3782d545961a1340b1efef540bda5737b37b7c987d20eb"
EXPECTED_MATERIAL_SHA256 = "92b934c91fb58fd8955054e1ccbfa92b3d49e56aa3c8948e64e791fce17a3ef2"
EXPECTED_INITIAL_CRITIC_SHA256 = "95d31b41ec92194cee23490695c0f052e74df169d640612f9d55e176540eccf0"
EXPECTED_POLICY_SHA256 = {
    0: "358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61",
    5: "fd0ea210bec6c403eb18fa0466ae11030b075ba1a5aa028cef94a458905c6f18",
}
EXPECTED_COMPLETED_TRANSITIONS = {0: 155, 5: 125}
EXPECTED_TERMINAL_Q9 = {0: 163, 5: 133}
EXPECTED_EPISODE_RETURN = {
    0: -123.90045524947345,
    5: -115.26687344536185,
}
EXPECTED_TERMINATION_NAMES = ("ee_body_pos",)
PRIOR_FAILED_TRACE_FILENAME = "sonic_task_space_ckpt0_vs_ckpt5_trace_v1.json"
PRIOR_FAILED_TRACE_SHA256 = "f982aceb0a2ba996193ae8c02e8f27c75b6b86f12bd6a96dd66a71e55d961401"
SECOND_FAILED_TRACE_FILENAME = "sonic_task_space_ckpt0_vs_ckpt5_trace_v2.json"
SECOND_FAILED_TRACE_SHA256 = "eef3441cf4e9af1c343bbc43862eec40ed52c20ceb31bf553209dc92bc852059"
RETRY_TRACE_FILENAME = "sonic_task_space_ckpt0_vs_ckpt5_trace_v3.json"
CAPTURE_CONTRACT = "gpu_clone_at_reward_compute_cpu_materialize_after_wrapped_step_v1"

PARENT_RUN_FILE_SHA256 = {
    "pilot_result.json": "f78ee6758fd7e947339422a51e1ebb85fe0de5200d9b75bd04ac962847d74d0e",
    "initialization.json": "f6534b75122ed7f213b396a09eab9778b7c40e4b55a1eb33b574e53d74daaca9",
    "sonic_task_space_ppo_runtime.json": "c174db0c926728d1466ec98b070001a22fd3967be0af2ea99eaaa5afe4ffd8e3",
    "resolved_training.json": "cc7643d2b71d455896ff0615078fc2c7341c29cd95f48e21647aa831a92f4b47",
    "environment_prime.json": "1ea82fc1425508208830b9ae085f3e8ea9395e27fe1b91cac2450e810c0fb9bc",
    "material_manifest.json": "c84e7196ea2d7e83598b4dfaf0e2bf28bf2ae3f6ea4b7eaea8f4fea02a73bee4",
    "preflight.json": "32988f296dc7ce956b4475d845d8991fd9010b84fff418076c2686dc32982aab",
    "evaluations/evaluation_update_0.json": ("8c983b4c8e0a6cfa65d5b548e35842afb172a0f5f1cffabcc71c4e079ba7866f"),
    "evaluations/evaluation_update_5.json": ("a530f448314fb3d71d01bd95676413f504e1e7e03283b627d0947159e88fcea5"),
    "checkpoints/sonic_task_space_model_0.pt": (
        "c448882d753f6a594dee578fac437ed5cc17889378a037d47095e9a1958ec8f2"
    ),
    "checkpoints/sonic_task_space_model_5.pt": (
        "99b9a19d26f9a1ae9d685bde73faaf2842983b64551d447c65879ba97071cdbd"
    ),
}

KNOWN_CONTROL_PLANE_PATHS = (
    "gear_sonic/config/sim_validation/g1_true23_sonic_task_space_ppo_v1.json",
    "gear_sonic/trl/mjlab/sonic_task_space_ppo_runner.py",
    "gear_sonic/tests/test_g1_true23_sonic_task_space_ppo.py",
)
EXPECTED_UNCHANGED_CONTROL_PATHS = ("gear_sonic/scripts/train_g1_true23_sonic_task_space_ppo.py",)
SEALED_CURRENT_CONTROL_SHA256 = {
    "gear_sonic/config/sim_validation/g1_true23_sonic_task_space_ppo_v1.json": (
        "a42d661e68e583e9f7e750e7fd8211ba36747172b8ff9be031414b2fe0f03044"
    ),
    "gear_sonic/trl/mjlab/sonic_task_space_ppo_runner.py": (
        "b92bce4d9c756b9fe5d97b7eedd2017a11deb23c193b3ad9531e6f5edfc41ded"
    ),
    "gear_sonic/tests/test_g1_true23_sonic_task_space_ppo.py": (
        "8d570b1999f82e256177ebe8e502016b52ecf2a86efe4400a8c40e37b882f832"
    ),
    "gear_sonic/scripts/train_g1_true23_sonic_task_space_ppo.py": (
        "2295ab37f8ba44ae8138739ec8a1e3d7f50c10bda0f3c3d28c2dad12e416b8ee"
    ),
}
TRACE_SOURCE_RELATIVE_PATHS = (
    "gear_sonic/config/sim_validation/g1_true23_sonic_task_space_ppo_v1.json",
    "gear_sonic/trl/mjlab/sonic_task_space_ppo_runner.py",
    "gear_sonic/trl/mjlab/true23_actor.py",
    "gear_sonic/trl/mjlab/causal_history_runner.py",
    "gear_sonic/utils/g1_true23_sonic_task_space_checkpoint_trace.py",
    "gear_sonic/scripts/trace_g1_true23_sonic_task_space_checkpoints.py",
)

PHYSICAL_PARENT_PREFIXES = (
    "gear_sonic/envs/mjlab/",
    "gear_sonic/utils/g1_23dof_",
    "gear_sonic/utils/g1_true23_",
    "gear_sonic/trl/mjlab/true23_actor.py",
    "gear_sonic/trl/mjlab/causal_history_runner.py",
    "unitree_rl_mjlab/src/",
    "mjlab/src/",
)

ACTION_DIVERGENCE_ATOL = 1.0e-6
EE_DIVERGENCE_ATOL_M = 1.0e-4
REWARD_DIVERGENCE_ATOL = 1.0e-6
CONTACT_FORCE_DIVERGENCE_ATOL_N = 1.0e-3
LANDING_FORCE_DIVERGENCE_ATOL_N = 1.0e-3


@dataclass(frozen=True)
class TraceInputs:
    repository_root: Path
    run_dir: Path
    run_files: Mapping[str, Path]
    parent_json: Mapping[str, Mapping[str, Any]]
    parent_material: Mapping[str, Any]
    provenance: Mapping[str, Any]


class TraceReproductionError(ValueError):
    """Parent evaluation mismatch carrying publication-safe scalar evidence."""

    def __init__(self, message: str, partial_evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        _assert_scalar_evidence(partial_evidence)
        self.partial_evidence = dict(partial_evidence)


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _sha256(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be lowercase SHA256")
    return value


def _canonical_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _regular_file(path: Path, context: str) -> Path:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{context} may not traverse symlinks")
    resolved = absolute.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{context} must be a regular file: {resolved}")
    return resolved


def _regular_directory(path: Path, context: str) -> Path:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{context} may not traverse symlinks")
    resolved = absolute.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{context} must be a regular directory: {resolved}")
    return resolved


def _strict_json(path: Path, context: str) -> Mapping[str, Any]:
    resolved = _regular_file(path, context)
    value = json.loads(
        resolved.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{context} contains non-finite JSON token {token}")
        ),
    )
    return _mapping(value, context)


def _logical_parent_path(root: Path, logical: str) -> Path:
    normalized = PurePosixPath(logical)
    if normalized.is_absolute() or ".." in normalized.parts or normalized.as_posix() != logical:
        raise ValueError(f"invalid parent logical path: {logical}")
    if logical.startswith("unitree_rl_mjlab/src/"):
        suffix = logical.removeprefix("unitree_rl_mjlab/src/")
        return root / "external_dependencies" / "unitree_rl_mjlab" / "src" / suffix
    if logical.startswith("mjlab/src/"):
        suffix = logical.removeprefix("mjlab/src/")
        return root / "external_dependencies" / "mjlab" / "src" / suffix
    if logical.startswith("gear_sonic/"):
        return root / logical
    raise ValueError(f"unsupported parent source logical path: {logical}")


def _manifest_record_map(manifest: Mapping[str, Any], kind: str) -> dict[str, Mapping[str, Any]]:
    validated = validate_file_manifest(manifest, kind=kind)
    return {record["logical_path"]: record for record in validated["files"]}


def _verify_record(path: Path, record: Mapping[str, Any], context: str) -> dict[str, Any]:
    resolved = _regular_file(path, context)
    expected_size = record.get("size_bytes")
    expected_hash = _sha256(record.get("sha256"), f"{context} SHA256")
    actual_size = resolved.stat().st_size
    actual_hash = sha256_file(resolved)
    if actual_size != expected_size or actual_hash != expected_hash:
        raise ValueError(f"{context} differs from parent material manifest")
    return {
        "logical_path": record["logical_path"],
        "size_bytes": actual_size,
        "sha256": actual_hash,
    }


def classify_control_plane_drift(
    parent_records: Mapping[str, Mapping[str, Any]],
    current_hashes: Mapping[str, str],
) -> dict[str, Any]:
    required = set(KNOWN_CONTROL_PLANE_PATHS) | set(EXPECTED_UNCHANGED_CONTROL_PATHS)
    if set(current_hashes) != required or not required.issubset(parent_records):
        raise ValueError("control-plane comparison path coverage mismatch")
    drift: list[dict[str, str]] = []
    unchanged: list[str] = []
    for logical in sorted(required):
        parent_hash = _sha256(parent_records[logical].get("sha256"), f"parent {logical}")
        current_hash = _sha256(current_hashes[logical], f"current {logical}")
        if current_hash != SEALED_CURRENT_CONTROL_SHA256[logical]:
            raise ValueError(f"current sealed control-plane source drift: {logical}")
        if current_hash == parent_hash:
            unchanged.append(logical)
        elif logical in KNOWN_CONTROL_PLANE_PATHS:
            drift.append(
                {
                    "logical_path": logical,
                    "parent_sha256": parent_hash,
                    "current_sha256": current_hash,
                    "classification": "known_post_pilot_control_plane_drift_not_parent_execution_identity",
                }
            )
        else:
            raise ValueError(f"unexpected control-plane drift: {logical}")
    return {
        "known_drift": drift,
        "unchanged": unchanged,
        "historical_full_source_identity_claimed": False,
    }


def decode_reward_terms(
    *,
    weighted_rates: Sequence[float],
    weights: Sequence[float],
    dt: float,
    reward_total: float,
) -> tuple[list[float], list[float]]:
    if (
        len(weighted_rates) != len(weights)
        or not weighted_rates
        or not math.isfinite(dt)
        or dt <= 0.0
        or not math.isfinite(reward_total)
    ):
        raise ValueError("reward trace inputs are invalid")
    raw: list[float] = []
    weighted: list[float] = []
    for index, (rate, weight) in enumerate(zip(weighted_rates, weights, strict=True)):
        if not math.isfinite(rate) or not math.isfinite(weight):
            raise ValueError(f"reward term {index} rate/weight is invalid")
        if weight == 0.0:
            raise ValueError(f"reward term {index} has zero-weight")
        raw.append(float(rate / weight))
        weighted.append(float(rate * dt))
    if not math.isclose(sum(weighted), reward_total, rel_tol=0.0, abs_tol=2.0e-5):
        raise ValueError("reward term contributions do not sum to reward total")
    return raw, weighted


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{context} must be finite scalar tensor")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context} must be finite scalar")
    return float(value)


def _tensor_row(value: Any, width: int, context: str) -> list[float]:
    if (
        type(value) is not torch.Tensor
        or value.shape != (1, width)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{context} must be finite [1,{width}] tensor")
    return [float(item) for item in value[0].detach().cpu().tolist()]


def _q9(command: Any) -> int:
    value = getattr(command, "time_steps", None)
    if type(value) is not torch.Tensor or value.shape != (1,) or value.dtype != torch.long:
        raise ValueError("trace q9 must be int64 [1]")
    return int(value.detach().cpu().item())


def _clone_tensor_snapshot(
    value: Any,
    *,
    shape: tuple[int, ...],
    context: str,
    dtype: torch.dtype | None = None,
    floating: bool = False,
) -> torch.Tensor:
    """Clone one CUDA value without a device-to-host read at the reward seam."""

    if (
        type(value) is not torch.Tensor
        or value.shape != shape
        or (dtype is not None and value.dtype != dtype)
        or (floating and not value.is_floating_point())
    ):
        raise ValueError(f"{context} tensor metadata drift")
    return value.detach().clone()


def _snapshot_termination_state(raw_env: Any) -> dict[str, Any]:
    manager = raw_env.termination_manager
    names = list(manager.active_terms)
    if not names or len(names) != len(set(names)) or any(type(name) is not str or not name for name in names):
        raise ValueError("termination snapshot names drift")
    values = torch.stack([manager.get_term(name)[0] for name in names]).unsqueeze(0)
    return {
        "names": names,
        "values": _clone_tensor_snapshot(
            values,
            shape=(1, len(names)),
            context="termination snapshot",
            dtype=torch.bool,
        ),
    }


def _finalize_termination_state(snapshot: Mapping[str, Any]) -> list[str]:
    names = snapshot.get("names")
    if (
        not isinstance(names, list)
        or not names
        or len(names) != len(set(names))
        or any(type(name) is not str or not name for name in names)
    ):
        raise ValueError("termination snapshot names drift")
    values = snapshot.get("values")
    if type(values) is not torch.Tensor or values.shape != (1, len(names)) or values.dtype != torch.bool:
        raise ValueError("termination snapshot values drift")
    active = values[0].detach().cpu().tolist()
    return sorted(name for name, enabled in zip(names, active, strict=True) if bool(enabled))


def _snapshot_ee_z_errors(raw_env: Any) -> torch.Tensor:
    command = raw_env.command_manager.get_term("motion")
    configured = tuple(command.cfg.body_names)
    if any(name not in configured for name in EE_BODY_NAMES):
        raise ValueError("trace EE body mapping drift")
    indices = [configured.index(name) for name in EE_BODY_NAMES]
    reference = command.body_pos_relative_w[:, indices, -1]
    measured = command.robot_body_pos_w[:, indices, -1]
    errors = torch.abs(reference - measured)
    return _clone_tensor_snapshot(
        errors,
        shape=(1, len(EE_BODY_NAMES)),
        context="EE z errors",
        floating=True,
    )


def _snapshot_action_chain(raw_env: Any, actor_action: torch.Tensor) -> dict[str, torch.Tensor]:
    from gear_sonic.envs.mjlab.sonic_true23_student_qualification import (
        DiagnosticSafeTargetNativeIl23JointPositionAction,
    )

    term = raw_env.action_manager.get_term("joint_pos")
    if type(term) is not DiagnosticSafeTargetNativeIl23JointPositionAction:
        raise TypeError("trace runtime lacks exact diagnostic student action term")
    values = {
        "raw_native": term.plain_sonic_raw_action_native,
        "candidate_target_hardware": term.candidate_target_hardware,
        "safe_native": term.safe_native_action,
        "final_target_hardware": term.final_target_hardware,
        "raw_clip_mask_native": term.raw_clip_mask_native,
    }
    snapshot: dict[str, torch.Tensor] = {
        "actor_action": _clone_tensor_snapshot(
            actor_action,
            shape=(1, ACTION_DIM),
            context="actor action snapshot",
            floating=True,
        )
    }
    for name, value in values.items():
        snapshot[name] = _clone_tensor_snapshot(
            value,
            shape=(1, ACTION_DIM),
            context=f"action-chain {name} snapshot",
            dtype=torch.bool if name == "raw_clip_mask_native" else torch.float32,
        )
    return snapshot


def _finalize_action_chain(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    from gear_sonic.envs.mjlab.sonic_true23 import native_actions_to_hardware_targets
    from gear_sonic.utils.g1_23dof_safe_target_transform import (
        SAFE_TARGET_DEFAULT_Q_HARDWARE,
        SAFE_TARGET_RAW_ACTION_CLIP,
        safe_target_transform_torch,
    )

    actor_action = snapshot.get("actor_action")
    raw_native = snapshot.get("raw_native")
    if type(actor_action) is not torch.Tensor or type(raw_native) is not torch.Tensor:
        raise ValueError("trace action snapshot is incomplete")
    raw = actor_action.detach().to(dtype=torch.float32, device=raw_native.device)
    expected_safe, expected_target = safe_target_transform_torch(raw)
    expected_candidate = native_actions_to_hardware_targets(raw, SAFE_TARGET_DEFAULT_Q_HARDWARE)
    expected_mask = torch.abs(raw) >= SAFE_TARGET_RAW_ACTION_CLIP
    expected = {
        "raw_native": raw,
        "candidate_target_hardware": expected_candidate,
        "safe_native": expected_safe,
        "final_target_hardware": expected_target,
    }
    for name, value in expected.items():
        actual = snapshot.get(name)
        if (
            type(actual) is not torch.Tensor
            or actual.shape != value.shape
            or not bool(torch.isfinite(actual).all())
            or not torch.allclose(actual, value, atol=1.0e-6, rtol=0.0)
        ):
            raise ValueError(f"trace action-chain mismatch: {name}")
    actual_mask = snapshot.get("raw_clip_mask_native")
    if type(actual_mask) is not torch.Tensor or not torch.equal(actual_mask, expected_mask):
        raise ValueError("trace raw-clip mask mismatch")
    clip_count = int(torch.count_nonzero(expected_mask).detach().cpu().item())
    if clip_count != 0:
        raise ValueError("trace policy required raw clipping")
    return {
        "raw_native": _tensor_row(snapshot["raw_native"], ACTION_DIM, "raw native action"),
        "safe_native": _tensor_row(snapshot["safe_native"], ACTION_DIM, "safe native action"),
        "final_target_hardware": _tensor_row(
            snapshot["final_target_hardware"],
            ACTION_DIM,
            "final hardware target",
        ),
    }


def _snapshot_contact_state(raw_env: Any) -> dict[str, torch.Tensor]:
    sensor = raw_env.scene["feet_ground_contact"].data
    force = getattr(sensor, "force", None)
    found = getattr(sensor, "found", None)
    if (
        type(force) is not torch.Tensor
        or force.shape != (1, len(CONTACT_SITE_NAMES), 3)
        or type(found) is not torch.Tensor
        or found.shape != (1, len(CONTACT_SITE_NAMES))
    ):
        raise ValueError("trace foot-contact tensors drifted")
    log = _mapping(raw_env.extras.get("log"), "trace environment log")
    landing = log.get("Metrics/landing_force_mean")
    if type(landing) is not torch.Tensor or landing.numel() != 1 or not landing.is_floating_point():
        raise ValueError("landing force mean tensor metadata drift")
    return {
        "found": _clone_tensor_snapshot(
            found,
            shape=(1, len(CONTACT_SITE_NAMES)),
            context="contact found snapshot",
            floating=True,
        ),
        "force": _clone_tensor_snapshot(
            force,
            shape=(1, len(CONTACT_SITE_NAMES), 3),
            context="contact force snapshot",
            floating=True,
        ),
        "landing_force_mean": landing.detach().reshape(1).clone(),
    }


def _finalize_contact_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    found = snapshot.get("found")
    force = snapshot.get("force")
    landing = snapshot.get("landing_force_mean")
    if (
        type(found) is not torch.Tensor
        or found.shape != (1, len(CONTACT_SITE_NAMES))
        or not found.is_floating_point()
        or not bool(torch.isfinite(found).all())
        or type(force) is not torch.Tensor
        or force.shape != (1, len(CONTACT_SITE_NAMES), 3)
        or not force.is_floating_point()
        or not bool(torch.isfinite(force).all())
        or type(landing) is not torch.Tensor
        or landing.shape != (1,)
        or not landing.is_floating_point()
        or not bool(torch.isfinite(landing).all())
    ):
        raise ValueError("trace contact snapshot drift")
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    return {
        "found": [bool(item > 0) for item in found[0].detach().cpu().tolist()],
        "force_magnitude_n": _tensor_row(magnitude, len(CONTACT_SITE_NAMES), "contact force"),
        "landing_force_mean_n": _finite_float(landing[0], "landing force mean"),
    }


def _reward_layout(raw_env: Any) -> dict[str, Any]:
    names = list(raw_env.reward_manager.active_terms)
    if not names or len(names) != len(set(names)):
        raise ValueError("trace reward-term names drifted")
    weights = []
    for name in names:
        weight = _finite_float(raw_env.reward_manager.get_term_cfg(name).weight, f"reward weight {name}")
        if weight == 0.0:
            raise ValueError("trace cannot recover raw value for zero-weight reward")
        weights.append(weight)
    dt = _finite_float(raw_env.step_dt, "trace control dt")
    if dt != 0.02 or raw_env.cfg.scale_rewards_by_dt is not True:
        raise ValueError("trace reward dt-scaling contract drift")
    return {
        "names": names,
        "weights": weights,
        "control_dt_s": dt,
    }


class _RewardComputeTraceRecorder:
    """Capture transition evidence after rewards, before terminal autoreset."""

    def __init__(self, raw_env: Any, layout: Mapping[str, Any]) -> None:
        self.raw_env = raw_env
        self.layout = layout
        self._original_compute = raw_env.reward_manager.compute
        self._armed: tuple[int, torch.Tensor] | None = None
        self._captured_snapshot: dict[str, Any] | None = None

        def observed_compute(_manager: Any, dt: float) -> torch.Tensor:
            reward = self._original_compute(dt)
            if self._armed is None or self._captured_snapshot is not None:
                raise RuntimeError("trace reward compute occurred outside one armed step")
            q9, actor_action = self._armed
            self._captured_snapshot = self._snapshot(q9, actor_action, reward, dt)
            return reward

        raw_env.reward_manager.compute = MethodType(observed_compute, raw_env.reward_manager)

    def _snapshot(
        self,
        q9: int,
        actor_action: torch.Tensor,
        reward: torch.Tensor,
        dt: float,
    ) -> dict[str, Any]:
        if dt != self.layout["control_dt_s"]:
            raise ValueError("trace reward compute dt drift")
        names = list(self.raw_env.reward_manager.active_terms)
        if names != self.layout["names"]:
            raise ValueError("trace reward iterable order drift")
        rates = getattr(self.raw_env.reward_manager, "_step_reward", None)
        return {
            "q9": q9,
            "reward_total": _clone_tensor_snapshot(
                reward,
                shape=(1,),
                context="reward total snapshot",
                floating=True,
            ),
            "reward_rates": _clone_tensor_snapshot(
                rates,
                shape=(1, len(names)),
                context="reward-rate snapshot",
                floating=True,
            ),
            "ee_z_error_m": _snapshot_ee_z_errors(self.raw_env),
            "actions": _snapshot_action_chain(self.raw_env, actor_action),
            "contact": _snapshot_contact_state(self.raw_env),
            "terminations": _snapshot_termination_state(self.raw_env),
            "episode_length_pre_reset": _clone_tensor_snapshot(
                self.raw_env.episode_length_buf,
                shape=(1,),
                context="episode length snapshot",
                dtype=torch.long,
            ),
        }

    def _finalize(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        reward_total_tensor = snapshot.get("reward_total")
        reward_rates = snapshot.get("reward_rates")
        if type(reward_total_tensor) is not torch.Tensor or type(reward_rates) is not torch.Tensor:
            raise ValueError("trace reward snapshot is incomplete")
        reward_total = _finite_float(reward_total_tensor[0], "trace reward total")
        rates = _tensor_row(reward_rates, len(self.layout["names"]), "trace reward rates")
        raw_values, weighted = decode_reward_terms(
            weighted_rates=rates,
            weights=self.layout["weights"],
            dt=self.layout["control_dt_s"],
            reward_total=reward_total,
        )
        q9 = snapshot.get("q9")
        if type(q9) is not int:
            raise ValueError("trace q9 snapshot drift")
        episode_length = snapshot.get("episode_length_pre_reset")
        if (
            type(episode_length) is not torch.Tensor
            or episode_length.shape != (1,)
            or episode_length.dtype != torch.long
        ):
            raise ValueError("trace episode length snapshot drift")
        return {
            "q9": q9,
            "reward_total": reward_total,
            "reward_raw": raw_values,
            "reward_weighted": weighted,
            "ee_z_error_m": _tensor_row(
                snapshot["ee_z_error_m"],
                len(EE_BODY_NAMES),
                "EE z errors",
            ),
            "actions": _finalize_action_chain(_mapping(snapshot.get("actions"), "action snapshot")),
            "contact": _finalize_contact_state(_mapping(snapshot.get("contact"), "contact snapshot")),
            "termination_names": _finalize_termination_state(
                _mapping(snapshot.get("terminations"), "termination snapshot")
            ),
            "episode_length_pre_reset": int(episode_length[0].detach().cpu().item()),
        }

    def arm(self, q9: int, actor_action: torch.Tensor) -> None:
        if self._armed is not None or self._captured_snapshot is not None:
            raise RuntimeError("trace recorder was not consumed")
        self._armed = (q9, actor_action.detach().clone())

    def finish(self) -> dict[str, Any]:
        if self._armed is None or self._captured_snapshot is None:
            raise RuntimeError("trace reward compute was not captured")
        captured = self._finalize(self._captured_snapshot)
        self._armed = None
        self._captured_snapshot = None
        return captured

    def restore(self) -> None:
        self.raw_env.reward_manager.compute = self._original_compute


def frames_to_series(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not frames:
        raise ValueError("trace requires at least one frame")
    return {
        "q9": [frame["q9"] for frame in frames],
        "reward_total": [frame["reward_total"] for frame in frames],
        "reward_raw": [frame["reward_raw"] for frame in frames],
        "reward_weighted": [frame["reward_weighted"] for frame in frames],
        "ee_z_error_m": [frame["ee_z_error_m"] for frame in frames],
        "raw_native_action": [frame["actions"]["raw_native"] for frame in frames],
        "safe_native_action": [frame["actions"]["safe_native"] for frame in frames],
        "final_target_hardware": [frame["actions"]["final_target_hardware"] for frame in frames],
        "contact_found": [frame["contact"]["found"] for frame in frames],
        "contact_force_magnitude_n": [frame["contact"]["force_magnitude_n"] for frame in frames],
        "landing_force_mean_n": [frame["contact"]["landing_force_mean_n"] for frame in frames],
        "termination_names": [frame["termination_names"] for frame in frames],
    }


def _all_finite(values: Any) -> bool:
    if isinstance(values, bool):
        return True
    if isinstance(values, (int, float)):
        return math.isfinite(float(values))
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return all(_all_finite(value) for value in values)
    return False


def validate_trace_layout(layout: Mapping[str, Any]) -> None:
    if set(layout) != {
        "reward_terms",
        "control_dt_s",
        "ee_body_names",
        "contact_site_names",
        "action_orders",
    }:
        raise ValueError("trace layout schema mismatch")
    reward_terms = layout.get("reward_terms")
    if not isinstance(reward_terms, list) or not reward_terms:
        raise ValueError("trace reward layout must be non-empty list")
    names: list[str] = []
    for index, term_value in enumerate(reward_terms):
        term = _mapping(term_value, f"trace reward layout term {index}")
        if set(term) != {"name", "weight"}:
            raise ValueError("trace reward layout term schema mismatch")
        name = term.get("name")
        weight = term.get("weight")
        if type(name) is not str or not name or isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("trace reward layout term value mismatch")
        if not math.isfinite(float(weight)) or float(weight) == 0.0:
            raise ValueError("trace reward layout term weight mismatch")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("trace reward layout names must be unique")
    if layout.get("control_dt_s") != 0.02:
        raise ValueError("trace layout control dt mismatch")
    if tuple(layout.get("ee_body_names", ())) != EE_BODY_NAMES:
        raise ValueError("trace layout EE bodies mismatch")
    if tuple(layout.get("contact_site_names", ())) != CONTACT_SITE_NAMES:
        raise ValueError("trace layout contact sites mismatch")
    if layout.get("action_orders") != {
        "raw_native_action": "native_isaaclab_23",
        "safe_native_action": "native_isaaclab_23",
        "final_target_hardware": "hardware_mujoco_23",
    }:
        raise ValueError("trace layout action orders mismatch")


def validate_trace_series(
    trace: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> None:
    validate_trace_layout(layout)
    update = trace.get("update_count")
    if update not in EXPECTED_UPDATES:
        raise ValueError("trace update is not the bound pair")
    series = _mapping(trace.get("series"), "trace series")
    required = {
        "q9",
        "reward_total",
        "reward_raw",
        "reward_weighted",
        "ee_z_error_m",
        "raw_native_action",
        "safe_native_action",
        "final_target_hardware",
        "contact_found",
        "contact_force_magnitude_n",
        "landing_force_mean_n",
        "termination_names",
    }
    if set(series) != required:
        raise ValueError("trace series schema mismatch")
    q9s = series["q9"]
    completed = trace.get("completed_transitions")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed <= 0
        or not isinstance(q9s, list)
        or q9s != list(range(9, 9 + completed))
    ):
        raise ValueError("trace q9 series is discontinuous")
    for name in required - {"q9"}:
        values = series[name]
        if not isinstance(values, list) or len(values) != completed:
            raise ValueError(f"trace {name} length mismatch")
    reward_width = len(layout["reward_terms"])
    widths = {
        "reward_raw": reward_width,
        "reward_weighted": reward_width,
        "ee_z_error_m": len(EE_BODY_NAMES),
        "raw_native_action": ACTION_DIM,
        "safe_native_action": ACTION_DIM,
        "final_target_hardware": ACTION_DIM,
        "contact_found": len(CONTACT_SITE_NAMES),
        "contact_force_magnitude_n": len(CONTACT_SITE_NAMES),
    }
    for name, width in widths.items():
        if any(not isinstance(row, list) or len(row) != width or not _all_finite(row) for row in series[name]):
            raise ValueError(f"trace {name} row mismatch")
    if any(
        not isinstance(row, list) or any(type(value) is not bool for value in row)
        for row in series["contact_found"]
    ):
        raise ValueError("trace contact state must be boolean")
    if not _all_finite(series["reward_total"]) or not _all_finite(series["landing_force_mean_n"]):
        raise ValueError("trace scalar series is nonfinite")
    if any(
        not isinstance(names, list)
        or names != sorted(names)
        or len(names) != len(set(names))
        or any(type(name) is not str or not name for name in names)
        for names in series["termination_names"]
    ):
        raise ValueError("trace termination series is invalid")
    weights = [term["weight"] for term in layout["reward_terms"]]
    for index in range(completed):
        expected = sum(series["reward_weighted"][index])
        if not math.isclose(expected, series["reward_total"][index], rel_tol=0.0, abs_tol=2.0e-5):
            raise ValueError("trace weighted rewards do not sum to total")
        for raw, weighted, weight in zip(
            series["reward_raw"][index],
            series["reward_weighted"][index],
            weights,
            strict=True,
        ):
            if not math.isclose(raw * weight * layout["control_dt_s"], weighted, rel_tol=0.0, abs_tol=2.0e-5):
                raise ValueError("trace raw/weighted reward relation mismatch")
    terminal = [index for index, names in enumerate(series["termination_names"]) if names]
    if terminal != [completed - 1]:
        raise ValueError("trace must contain one final terminal frame")
    if trace.get("terminal_q9") != q9s[-1] or trace.get("termination_names") != series["termination_names"][-1]:
        raise ValueError("trace terminal summary mismatch")
    if not math.isclose(sum(series["reward_total"]), trace.get("episode_return"), rel_tol=0.0, abs_tol=2.0e-5):
        raise ValueError("trace episode return summary mismatch")


def _first_q9(mask: Sequence[bool], q9s: Sequence[int]) -> int | None:
    return next((q9 for q9, active in zip(q9s, mask, strict=True) if active), None)


def _row_linf(left: Sequence[float], right: Sequence[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def compare_trace_pair(
    update0: Mapping[str, Any],
    update5: Mapping[str, Any],
    layout: Mapping[str, Any],
) -> dict[str, Any]:
    validate_trace_series(update0, layout)
    validate_trace_series(update5, layout)
    if update0["update_count"] != 0 or update5["update_count"] != 5:
        raise ValueError("trace comparison order must be update0 then update5")
    left = update0["series"]
    right = update5["series"]
    common = min(update0["completed_transitions"], update5["completed_transitions"])
    if common <= 1 or left["q9"][:common] != right["q9"][:common]:
        raise ValueError("trace pair has no exact common q9 horizon")
    q9s = left["q9"][:common]
    raw_action_delta = [
        _row_linf(left["raw_native_action"][i], right["raw_native_action"][i]) for i in range(common)
    ]
    safe_action_delta = [
        _row_linf(left["safe_native_action"][i], right["safe_native_action"][i]) for i in range(common)
    ]
    final_target_delta = [
        _row_linf(left["final_target_hardware"][i], right["final_target_hardware"][i]) for i in range(common)
    ]
    ee_delta = [_row_linf(left["ee_z_error_m"][i], right["ee_z_error_m"][i]) for i in range(common)]
    reward_delta = [abs(left["reward_total"][i] - right["reward_total"][i]) for i in range(common)]
    contact_state_delta = [left["contact_found"][i] != right["contact_found"][i] for i in range(common)]
    contact_force_delta = [
        _row_linf(left["contact_force_magnitude_n"][i], right["contact_force_magnitude_n"][i])
        for i in range(common)
    ]
    landing_delta = [
        abs(left["landing_force_mean_n"][i] - right["landing_force_mean_n"][i]) for i in range(common)
    ]
    termination_delta = [left["termination_names"][i] != right["termination_names"][i] for i in range(common)]
    category_first = {
        "raw_action": _first_q9([value > ACTION_DIVERGENCE_ATOL for value in raw_action_delta], q9s),
        "safe_action": _first_q9([value > ACTION_DIVERGENCE_ATOL for value in safe_action_delta], q9s),
        "final_target": _first_q9([value > ACTION_DIVERGENCE_ATOL for value in final_target_delta], q9s),
        "ee_z_error": _first_q9([value > EE_DIVERGENCE_ATOL_M for value in ee_delta], q9s),
        "reward_total": _first_q9([value > REWARD_DIVERGENCE_ATOL for value in reward_delta], q9s),
        "contact_state": _first_q9(contact_state_delta, q9s),
        "contact_force": _first_q9(
            [value > CONTACT_FORCE_DIVERGENCE_ATOL_N for value in contact_force_delta],
            q9s,
        ),
        "landing_force": _first_q9(
            [value > LANDING_FORCE_DIVERGENCE_ATOL_N for value in landing_delta],
            q9s,
        ),
        "termination": _first_q9(termination_delta, q9s),
    }
    present = [value for value in category_first.values() if value is not None]
    reward_names = [term["name"] for term in layout["reward_terms"]]

    def term_sums(series: Mapping[str, Any], count: int) -> list[float]:
        return [sum(row[index] for row in series["reward_weighted"][:count]) for index in range(len(reward_names))]

    full0 = term_sums(left, common)
    full5 = term_sums(right, common)
    preterminal = common - 1
    pre0 = term_sums(left, preterminal)
    pre5 = term_sums(right, preterminal)
    terminal_ee = right["ee_z_error_m"][common - 1]
    worst_index = max(range(len(terminal_ee)), key=terminal_ee.__getitem__)
    return {
        "common_transition_count": common,
        "common_q9_first": q9s[0],
        "common_q9_last": q9s[-1],
        "preterminal_common_transition_count": preterminal,
        "preterminal_common_q9_last": q9s[-2],
        "first_divergence_q9": min(present) if present else None,
        "first_divergence_by_category_q9": category_first,
        "maximum_common_linf": {
            "raw_action": max(raw_action_delta),
            "safe_action": max(safe_action_delta),
            "final_target": max(final_target_delta),
            "ee_z_error_m": max(ee_delta),
            "reward_total": max(reward_delta),
            "contact_force_n": max(contact_force_delta),
            "landing_force_n": max(landing_delta),
        },
        "common_reward_by_term": {
            "names": reward_names,
            "through_terminal_update5": {
                "update0": full0,
                "update5": full5,
                "update5_minus_update0": [b - a for a, b in zip(full0, full5, strict=True)],
            },
            "preterminal_common_prefix": {
                "update0": pre0,
                "update5": pre5,
                "update5_minus_update0": [b - a for a, b in zip(pre0, pre5, strict=True)],
            },
        },
        "update5_terminal_worst_ee": {
            "body_name": EE_BODY_NAMES[worst_index],
            "z_error_m": terminal_ee[worst_index],
        },
    }


def _assert_scalar_evidence(value: Any, path: str = "partial_evidence") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{path} keys must be non-empty strings")
            _assert_scalar_evidence(child, f"{path}.{key}")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{path} must contain scalar JSON evidence only")


def parent_episode_return_diagnostic(
    *,
    result: Mapping[str, Any],
    expected_evaluation: Mapping[str, Any],
    update_count: int,
) -> dict[str, Any]:
    """Disclose historical return drift without converting it into a tolerance gate."""

    if update_count not in EXPECTED_UPDATES or result.get("update_count") != update_count:
        raise ValueError("episode-return diagnostic update identity mismatch")
    observed = _finite_float(result.get("episode_return"), "observed episode return")
    historical = _finite_float(expected_evaluation.get("episode_return"), "historical episode return")
    return {
        "update_count": update_count,
        "observed_episode_return": observed,
        "historical_parent_episode_return": historical,
        "delta_observed_minus_historical": observed - historical,
        "exact_match": observed == historical,
        "gate_applied": False,
        "reason": "mujoco_warp_cuda_return_is_not_cross_replay_deterministic",
    }


def reproduction_scalar_evidence(
    *,
    result: Mapping[str, Any],
    layout: Mapping[str, Any],
    expected_evaluation: Mapping[str, Any],
    update_count: int,
    provenance_snapshot_sha256: str,
) -> dict[str, Any]:
    """Summarize a failed full trace without publishing trajectory arrays."""

    validate_trace_series(result, layout)
    if update_count not in EXPECTED_UPDATES or result.get("update_count") != update_count:
        raise ValueError("partial trace update identity mismatch")
    provenance_hash = _sha256(provenance_snapshot_sha256, "partial trace provenance snapshot")
    series = _mapping(result["series"], "partial trace series")
    reward_names = [term["name"] for term in layout["reward_terms"]]
    reward_totals = {
        name: float(sum(row[index] for row in series["reward_weighted"]))
        for index, name in enumerate(reward_names)
    }
    terminal_ee = series["ee_z_error_m"][-1]
    terminal_contact = series["contact_force_magnitude_n"][-1]
    termination_names = result.get("termination_names")
    expected_names = expected_evaluation.get("termination_names")
    if not isinstance(termination_names, list) or not isinstance(expected_names, list):
        raise ValueError("partial trace termination names are invalid")
    evidence = {
        "stage": f"update{update_count}_parent_evaluation_reproduction",
        "update_count": update_count,
        "observed": {
            "completed_transitions": int(result["completed_transitions"]),
            "first_q9": int(series["q9"][0]),
            "terminal_q9": int(result["terminal_q9"]),
            "termination_names": ",".join(termination_names),
            "termination_name_count": len(termination_names),
            "episode_return": float(result["episode_return"]),
            "policy_state_sha256": _sha256(result["policy_state_sha256"], "observed policy SHA256"),
        },
        "expected": {
            "completed_transitions": int(expected_evaluation["completed_transitions"]),
            "first_q9": 9,
            "terminal_q9": int(expected_evaluation["terminal_q9"]),
            "termination_names": ",".join(expected_names),
            "termination_name_count": len(expected_names),
            "episode_return": float(expected_evaluation["episode_return"]),
            "policy_state_sha256": _sha256(
                expected_evaluation["policy_state_sha256"],
                "expected policy SHA256",
            ),
        },
        "delta_observed_minus_expected": {
            "completed_transitions": int(result["completed_transitions"])
            - int(expected_evaluation["completed_transitions"]),
            "terminal_q9": int(result["terminal_q9"]) - int(expected_evaluation["terminal_q9"]),
        },
        "mismatch": {
            "completed_transitions": result["completed_transitions"]
            != expected_evaluation["completed_transitions"],
            "terminal_q9": result["terminal_q9"] != expected_evaluation["terminal_q9"],
            "termination_names": termination_names != expected_names,
            "policy_state_sha256": result["policy_state_sha256"] != expected_evaluation["policy_state_sha256"],
        },
        "episode_return_diagnostic": parent_episode_return_diagnostic(
            result=result,
            expected_evaluation=expected_evaluation,
            update_count=update_count,
        ),
        "cumulative_reward_by_term": reward_totals,
        "terminal_ee_z_error_m": {name: float(terminal_ee[index]) for index, name in enumerate(EE_BODY_NAMES)},
        "terminal_contact": {
            "left_found": bool(series["contact_found"][-1][0]),
            "right_found": bool(series["contact_found"][-1][1]),
            "left_force_magnitude_n": float(terminal_contact[0]),
            "right_force_magnitude_n": float(terminal_contact[1]),
            "landing_force_mean_n": float(series["landing_force_mean_n"][-1]),
        },
        "hashes": {
            "checkpoint_file_sha256": PARENT_RUN_FILE_SHA256[
                f"checkpoints/sonic_task_space_model_{update_count}.pt"
            ],
            "parent_evaluation_file_sha256": PARENT_RUN_FILE_SHA256[
                f"evaluations/evaluation_update_{update_count}.json"
            ],
            "provenance_snapshot_sha256": provenance_hash,
            "q9_series_sha256": _canonical_sha256(series["q9"]),
            "reward_series_sha256": _canonical_sha256(
                {
                    "total": series["reward_total"],
                    "raw": series["reward_raw"],
                    "weighted": series["reward_weighted"],
                }
            ),
            "ee_series_sha256": _canonical_sha256(series["ee_z_error_m"]),
            "raw_action_series_sha256": _canonical_sha256(series["raw_native_action"]),
            "safe_action_series_sha256": _canonical_sha256(series["safe_native_action"]),
            "final_target_series_sha256": _canonical_sha256(series["final_target_hardware"]),
        },
    }
    _assert_scalar_evidence(evidence)
    evidence["scalar_evidence_sha256"] = _canonical_sha256(evidence)
    return evidence


def validate_parent_evaluation_reproduction(
    *,
    result: Mapping[str, Any],
    layout: Mapping[str, Any],
    expected_evaluation: Mapping[str, Any],
    update_count: int,
    provenance_snapshot_sha256: str,
) -> dict[str, Any]:
    """Require exact structural identity; return historical reward drift as diagnostics."""

    _sha256(provenance_snapshot_sha256, "reproduction provenance snapshot")
    validate_trace_series(result, layout)
    observed_names = result.get("termination_names")
    expected_names = expected_evaluation.get("termination_names")
    structural_mismatch = (
        result.get("completed_transitions") != expected_evaluation.get("completed_transitions")
        or result.get("terminal_q9") != expected_evaluation.get("terminal_q9")
        or observed_names != expected_names
        or result.get("policy_state_sha256") != expected_evaluation.get("policy_state_sha256")
    )
    if structural_mismatch:
        evidence = reproduction_scalar_evidence(
            result=result,
            layout=layout,
            expected_evaluation=expected_evaluation,
            update_count=update_count,
            provenance_snapshot_sha256=provenance_snapshot_sha256,
        )
        raise TraceReproductionError(
            f"update{update_count} trace did not reproduce parent evaluation structure",
            evidence,
        )
    return parent_episode_return_diagnostic(
        result=result,
        expected_evaluation=expected_evaluation,
        update_count=update_count,
    )


def _parent_material_without_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("material_manifest_sha256", None)
    return result


def _physical_parent_records(
    root: Path,
    source_records: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for logical, record in source_records.items():
        if logical in KNOWN_CONTROL_PLANE_PATHS or not logical.startswith(PHYSICAL_PARENT_PREFIXES):
            continue
        records.append(_verify_record(_logical_parent_path(root, logical), record, f"physical source {logical}"))
    if not records:
        raise ValueError("parent physical source selection is empty")
    return sorted(records, key=lambda item: item["logical_path"])


def _verify_robot_assets(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = _manifest_record_map(manifest, "robot_assets")
    base = root / "external_dependencies" / "unitree_rl_mjlab" / "src" / "assets" / "robots" / "unitree_g1"
    return [
        _verify_record(base / logical.removeprefix("unitree_g1/"), record, f"robot asset {logical}")
        for logical, record in sorted(records.items())
    ]


def _bound_input_paths(root: Path) -> dict[str, Path]:
    return {
        "campaign/recovery_campaign.json": (
            root / "artifacts/g1_true23/g1_true23_sonic_recovery_qualification_campaign_v1.json"
        ),
        "actor/topology.pt": root / "sonic_release/g1_23dof_rev_1_0_low_latency_init.pt",
        "actor/model250.pt": Path(
            "/root/g1_true23_runs/causal_history_recovery_stand_transition_dance_v1/checkpoints/causal_model_250.pt"
        ),
        "actor/encoder.onnx": (
            root / "artifacts/g1_true23/causal_model_250_20260803/causal_model_250.encoder.onnx"
        ),
        "actor/v2_manifest.json": (
            root / "artifacts/g1_true23/sonic_native124_21204_bc_last_affine_ridge_seed20260805_v2.manifest.json"
        ),
        "actor/v2_decoder.onnx": (
            root / "artifacts/g1_true23/sonic_native124_21204_bc_last_affine_ridge_seed20260805_v2.decoder.onnx"
        ),
        "motion/B_DadDance.npz": (
            root / "artifacts/g1_native124_multimotion/scaling_all61/feasible_v1/npz/B_DadDance.npz"
        ),
    }


def _verify_bound_inputs(root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = _manifest_record_map(manifest, "motion_dataset")
    paths = _bound_input_paths(root)
    if set(records) != set(paths):
        raise ValueError("parent bound-input logical paths drifted")
    return [
        _verify_record(paths[logical], records[logical], f"bound input {logical}") for logical in sorted(paths)
    ]


def _trace_source_binding(root: Path) -> dict[str, Any]:
    files: dict[str, Path] = {logical: root / logical for logical in TRACE_SOURCE_RELATIVE_PATHS}
    for tree, prefix in ((root / "gear_sonic" / "envs" / "mjlab", "gear_sonic/envs/mjlab"),):
        for path in sorted(tree.rglob("*.py")):
            if "__pycache__" not in path.parts:
                files[f"{prefix}/{path.relative_to(tree).as_posix()}"] = path
    records = []
    for logical in sorted(files):
        resolved = _regular_file(files[logical], f"trace source {logical}")
        records.append(
            {
                "logical_path": logical,
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return {
        "schema_version": 1,
        "kind": "g1_true23_sonic_task_space_trace_executed_sources_v1",
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "manifest_sha256": _canonical_sha256(records),
        "files": records,
    }


def _parent_run_files(run_dir: Path) -> dict[str, Path]:
    if run_dir.is_symlink() or run_dir.name != "sonic_task_space_ppo_seed20260805_v2":
        raise ValueError("trace parent run directory identity mismatch")
    return {
        logical: _regular_file(run_dir / logical, f"parent run file {logical}")
        for logical in PARENT_RUN_FILE_SHA256
    }


def _load_parent_json(run_files: Mapping[str, Path]) -> dict[str, Mapping[str, Any]]:
    return {
        logical: _strict_json(path, f"parent {logical}")
        for logical, path in run_files.items()
        if path.suffix == ".json"
    }


def _validate_failed_trace_artifact(
    path: Path,
    *,
    expected_sha256: str,
    contains_partial_scalar_evidence: bool,
    context: str,
) -> dict[str, Any]:
    path = _regular_file(
        path,
        context,
    )
    observed_hash = sha256_file(path)
    if observed_hash != expected_sha256:
        raise ValueError(f"{context} identity mismatch")
    value = _strict_json(path, context)
    partial = value.get("partial_scalar_evidence")
    if (
        value.get("kind") != FAILURE_KIND
        or value.get("simulator_trace_complete") is not False
        or value.get("training_updates") != 0
        or value.get("hardware_authorized") is not False
        or _mapping(value.get("error"), "prior failed trace error").get("message")
        != "update0 trace did not reproduce parent evaluation"
        or (partial is not None) != contains_partial_scalar_evidence
    ):
        raise ValueError(f"{context} semantic mismatch")
    if contains_partial_scalar_evidence:
        scalar = _mapping(partial, f"{context} partial scalar evidence")
        mismatch = _mapping(scalar.get("mismatch"), f"{context} mismatch evidence")
        if (
            scalar.get("stage") != "update0_parent_evaluation_reproduction"
            or mismatch.get("completed_transitions") is not False
            or mismatch.get("terminal_q9") is not False
            or mismatch.get("termination_names") is not False
            or mismatch.get("policy_state_sha256") is not False
            or mismatch.get("episode_return") is not True
        ):
            raise ValueError(f"{context} partial scalar evidence mismatch")
    return {
        "path": str(path),
        "sha256": observed_hash,
        "immutable": True,
        "contains_partial_scalar_evidence": contains_partial_scalar_evidence,
    }


def _validate_prior_failed_traces(run_dir: Path) -> dict[str, Any]:
    return {
        "v1": _validate_failed_trace_artifact(
            run_dir.parent / PRIOR_FAILED_TRACE_FILENAME,
            expected_sha256=PRIOR_FAILED_TRACE_SHA256,
            contains_partial_scalar_evidence=False,
            context="prior v1 failed checkpoint trace",
        ),
        "v2": _validate_failed_trace_artifact(
            run_dir.parent / SECOND_FAILED_TRACE_FILENAME,
            expected_sha256=SECOND_FAILED_TRACE_SHA256,
            contains_partial_scalar_evidence=True,
            context="prior v2 failed checkpoint trace",
        ),
    }


def _validate_parent_semantics(parent: Mapping[str, Mapping[str, Any]]) -> None:
    pilot = parent["pilot_result.json"]
    if (
        pilot.get("contract_sha256") != EXPECTED_CONTRACT_SHA256
        or pilot.get("material_manifest_sha256") != EXPECTED_MATERIAL_SHA256
        or pilot.get("completed_update_count") != 5
        or pilot.get("executed_training_transitions") != 10_240
        or pilot.get("optimizer_step_count") != 40
        or pilot.get("selected_candidate_checkpoint_path") is not None
    ):
        raise ValueError("parent pilot result semantic identity mismatch")
    for update in EXPECTED_UPDATES:
        evaluation = parent[f"evaluations/evaluation_update_{update}.json"]
        if (
            evaluation.get("update_count") != update
            or evaluation.get("evaluation_seed") != FIXED_SEED
            or evaluation.get("policy_state_sha256") != EXPECTED_POLICY_SHA256[update]
            or evaluation.get("completed_transitions") != EXPECTED_COMPLETED_TRANSITIONS[update]
            or evaluation.get("terminal_q9") != EXPECTED_TERMINAL_Q9[update]
            or tuple(evaluation.get("termination_names", ())) != EXPECTED_TERMINATION_NAMES
            or not math.isclose(
                float(evaluation.get("episode_return", math.nan)),
                EXPECTED_EPISODE_RETURN[update],
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or any(
                evaluation.get(name) != 0
                for name in (
                    "nonfinite_count",
                    "raw_clip_required_count",
                    "action_semantics_mismatch_count",
                    "q9_discontinuity_count",
                )
            )
        ):
            raise ValueError(f"parent update{update} evaluation semantic identity mismatch")


def resolve_trace_inputs(repository_root: Path, parent_run_dir: Path) -> TraceInputs:
    root = repository_root.expanduser().resolve(strict=True)
    run_dir = parent_run_dir.expanduser().resolve(strict=True)
    run_files = _parent_run_files(run_dir)
    prior_failures = _validate_prior_failed_traces(run_dir)
    actual_hashes = {logical: sha256_file(path) for logical, path in run_files.items()}
    if actual_hashes != PARENT_RUN_FILE_SHA256:
        raise ValueError("parent run artifact SHA256 binding mismatch")
    parent = _load_parent_json(run_files)
    _validate_parent_semantics(parent)
    material = parent["material_manifest.json"]
    if (
        sha256_file(run_files["material_manifest.json"]) != PARENT_RUN_FILE_SHA256["material_manifest.json"]
        or material.get("material_manifest_sha256") != EXPECTED_MATERIAL_SHA256
        or _canonical_sha256(_parent_material_without_self_hash(material)) != EXPECTED_MATERIAL_SHA256
        or material.get("contract_sha256") != EXPECTED_CONTRACT_SHA256
        or parent["preflight.json"].get("material_manifest") != material
    ):
        raise ValueError("parent material manifest identity mismatch")
    source_records = _manifest_record_map(material["source_files"], "source_files")
    current_control_hashes = {
        logical: sha256_file(_regular_file(root / logical, f"control source {logical}"))
        for logical in (*KNOWN_CONTROL_PLANE_PATHS, *EXPECTED_UNCHANGED_CONTROL_PATHS)
    }
    control_drift = classify_control_plane_drift(source_records, current_control_hashes)
    physical_records = _physical_parent_records(root, source_records)
    robot_records = _verify_robot_assets(root, material["robot_assets"])
    bound_records = _verify_bound_inputs(root, material["bound_inputs"])
    from gear_sonic.scripts.train_g1_true23_native124_selected_v2_ankle import (
        resolve_rsl_runtime_binding,
    )

    current_rsl = resolve_rsl_runtime_binding()
    if current_rsl != material["rsl_runtime"]:
        raise ValueError("current RSL runtime differs from parent material binding")
    trace_sources = _trace_source_binding(root)
    provenance = {
        "parent_run_file_sha256": actual_hashes,
        "prior_failed_traces": prior_failures,
        "parent_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "parent_material_manifest_sha256": EXPECTED_MATERIAL_SHA256,
        "parent_physical_source_subset": {
            "file_count": len(physical_records),
            "manifest_sha256": _canonical_sha256(physical_records),
        },
        "parent_robot_assets": {
            "file_count": len(robot_records),
            "manifest_sha256": _canonical_sha256(robot_records),
        },
        "parent_bound_inputs": {
            "file_count": len(bound_records),
            "manifest_sha256": _canonical_sha256(bound_records),
        },
        "rsl_runtime_binding_sha256": current_rsl["runtime_binding_sha256"],
        "control_plane": control_drift,
        "current_executed_trace_sources": trace_sources,
    }
    provenance["snapshot_sha256"] = _canonical_sha256(provenance)
    return TraceInputs(
        repository_root=root,
        run_dir=run_dir,
        run_files=run_files,
        parent_json=parent,
        parent_material=material,
        provenance=provenance,
    )


def trace_preflight(repository_root: Path, parent_run_dir: Path) -> dict[str, Any]:
    try:
        inputs = resolve_trace_inputs(repository_root, parent_run_dir)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PREFLIGHT_KIND,
            "ready": True,
            "parent_run_dir": str(inputs.run_dir),
            "provenance": inputs.provenance,
            "simulator_constructed": False,
            "training_updates": 0,
            "teacher_labels_used": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PREFLIGHT_KIND,
            "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "simulator_constructed": False,
            "training_updates": 0,
            "teacher_labels_used": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }


def _checkpoint_body(inputs: TraceInputs, update: int) -> Mapping[str, Any]:
    path = inputs.run_files[f"checkpoints/sonic_task_space_model_{update}.pt"]
    value = torch.load(path, map_location="cpu", weights_only=True)
    body = _mapping(value, f"update{update} checkpoint")
    if (
        body.get("update_count") != update
        or body.get("contract_sha256") != EXPECTED_CONTRACT_SHA256
        or body.get("run_materials_sha256") != EXPECTED_MATERIAL_SHA256
        or body.get("policy_state_sha256") != EXPECTED_POLICY_SHA256[update]
        or body.get("initial_critic_state_sha256") != EXPECTED_INITIAL_CRITIC_SHA256
        or body.get("initial_overlay_policy_state_sha256") != EXPECTED_POLICY_SHA256[0]
    ):
        raise ValueError(f"update{update} checkpoint internal binding mismatch")
    policy = _mapping(body.get("policy_state_dict"), f"update{update} policy")
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy},
        reference_profile="released_low_latency_step1_0p02s",
    )
    if policy_hash != EXPECTED_POLICY_SHA256[update]:
        raise ValueError(f"update{update} checkpoint policy state mismatch")
    return body


def _load_actor_policy(actor: Any, checkpoint: Mapping[str, Any], update: int) -> None:
    policy = _mapping(checkpoint["policy_state_dict"], f"update{update} policy")
    network = {key: value for key, value in policy.items() if key != "std"}
    missing, unexpected = actor.core.load_state_dict(network, strict=True)
    if missing or unexpected:
        raise ValueError(f"update{update} actor load mismatch")
    std = policy.get("std")
    if type(std) is not torch.Tensor or std.shape != (ACTION_DIM,):
        raise ValueError(f"update{update} std tensor mismatch")
    with torch.no_grad():
        actor.distribution.std_param.copy_(std.to(actor.distribution.std_param.device))
    exported = actor.export_true23_policy_state()
    observed = inspect_true23_policy_state(
        {"policy_state_dict": exported},
        reference_profile="released_low_latency_step1_0p02s",
    )
    if observed != EXPECTED_POLICY_SHA256[update]:
        raise ValueError(f"update{update} live actor identity mismatch")


def run_policy_trace(
    *,
    policy: Any,
    wrapped_env: Any,
    update_count: int,
    expected_evaluation: Mapping[str, Any],
    provenance_snapshot_sha256: str,
) -> dict[str, Any]:
    raw_env = getattr(wrapped_env, "unwrapped", None)
    if (
        raw_env is None
        or int(getattr(raw_env, "num_envs", -1)) != 1
        or int(getattr(getattr(raw_env, "cfg", None), "seed", -1)) != FIXED_SEED
        or getattr(wrapped_env, "clip_actions", "missing") is not None
        or int(getattr(wrapped_env, "max_episode_length", -1)) != 510
        or int(raw_env.common_step_counter) != 0
        or int(raw_env._sim_step_counter) != 0
    ):
        raise ValueError("trace environment/wrapper is not exact freshly primed one-env task")
    command = raw_env.command_manager.get_term("motion")
    if _q9(command) != 9:
        raise ValueError("trace did not start at q9=9")
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy.export_true23_policy_state()},
        reference_profile="released_low_latency_step1_0p02s",
    )
    if policy_hash != EXPECTED_POLICY_SHA256[update_count]:
        raise ValueError("trace live policy hash mismatch")
    reward = _reward_layout(raw_env)
    layout = {
        "reward_terms": [
            {"name": name, "weight": weight}
            for name, weight in zip(reward["names"], reward["weights"], strict=True)
        ],
        "control_dt_s": reward["control_dt_s"],
        "ee_body_names": list(EE_BODY_NAMES),
        "contact_site_names": list(CONTACT_SITE_NAMES),
        "action_orders": {
            "raw_native_action": "native_isaaclab_23",
            "safe_native_action": "native_isaaclab_23",
            "final_target_hardware": "hardware_mujoco_23",
        },
    }
    observations = wrapped_env.get_observations()
    recorder = _RewardComputeTraceRecorder(raw_env, reward)
    was_training = bool(policy.training)
    policy.eval()
    frames: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for transition in range(510):
                q9 = _q9(command)
                if q9 != 9 + transition:
                    raise ValueError("trace q9 discontinuity")
                actor_action = policy(observations, stochastic_output=False)
                if (
                    type(actor_action) is not torch.Tensor
                    or actor_action.shape != (1, ACTION_DIM)
                    or not bool(torch.isfinite(actor_action).all())
                ):
                    raise ValueError("trace actor emitted invalid action")
                recorder.arm(q9, actor_action)
                observations, rewards, dones, _extras = wrapped_env.step(actor_action.to(wrapped_env.device))
                frame = recorder.finish()
                if (
                    type(rewards) is not torch.Tensor
                    or rewards.shape != (1,)
                    or not math.isclose(
                        frame["reward_total"],
                        _finite_float(rewards[0], "wrapped trace reward"),
                        rel_tol=0.0,
                        abs_tol=1.0e-7,
                    )
                    or type(dones) is not torch.Tensor
                    or dones.shape != (1,)
                ):
                    raise ValueError("trace wrapper step result drift")
                done = bool(int(dones[0].detach().cpu().item()))
                if done != bool(frame["termination_names"]):
                    raise ValueError("trace done/termination mismatch")
                frames.append(frame)
                if done:
                    if frame["episode_length_pre_reset"] != len(frames):
                        raise ValueError("trace terminal episode length mismatch")
                    break
    finally:
        recorder.restore()
        policy.train(was_training)
    series = frames_to_series(frames)
    result = {
        "update_count": update_count,
        "evaluation_seed": FIXED_SEED,
        "controller": "deterministic_actor_mean",
        "policy_state_sha256": policy_hash,
        "completed_transitions": len(frames),
        "terminal_q9": frames[-1]["q9"],
        "termination_names": frames[-1]["termination_names"],
        "episode_return": sum(frame["reward_total"] for frame in frames),
        "series": series,
    }
    validate_trace_series(result, layout)
    return_diagnostic = validate_parent_evaluation_reproduction(
        result=result,
        layout=layout,
        expected_evaluation=expected_evaluation,
        update_count=update_count,
        provenance_snapshot_sha256=provenance_snapshot_sha256,
    )
    return {
        "layout": layout,
        "trace": result,
        "parent_episode_return_diagnostic": return_diagnostic,
    }


def _seed_everything() -> None:
    random.seed(FIXED_SEED)
    np.random.seed(FIXED_SEED % (2**32))
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed_all(FIXED_SEED)


def _provenance_snapshot(inputs: TraceInputs) -> str:
    refreshed = resolve_trace_inputs(inputs.repository_root, inputs.run_dir)
    return str(refreshed.provenance["snapshot_sha256"])


def execute_checkpoint_trace(repository_root: Path, parent_run_dir: Path) -> dict[str, Any]:
    """Execute two deterministic evaluations.  No optimizer or training exists here."""

    inputs = resolve_trace_inputs(repository_root, parent_run_dir)
    starting_snapshot = inputs.provenance["snapshot_sha256"]
    checkpoints = {update: _checkpoint_body(inputs, update) for update in EXPECTED_UPDATES}

    from gear_sonic.trl.mjlab.true23_actor import True23SonicActorModel

    topology = _bound_input_paths(inputs.repository_root)["actor/topology.pt"]
    fake_observations = {
        "tokenizer": torch.zeros((1, 268), dtype=torch.float32),
        "policy": torch.zeros((1, 930), dtype=torch.float32),
    }
    # Actor construction consumes Torch RNG.  It therefore happens before the
    # final per-evaluation reseed/environment construction.
    actor = True23SonicActorModel(
        fake_observations,
        {"actor": ["tokenizer", "policy"]},
        "actor",
        ACTION_DIM,
        warm_start_path=str(topology),
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"},
        hidden_dims=(),
        activation="silu",
        obs_normalization=False,
    ).to(DEVICE)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
    from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
        audit_task_space_ppo_env_cfg,
        make_task_space_ppo_env_cfg,
    )

    motion = _bound_input_paths(inputs.repository_root)["motion/B_DadDance.npz"]
    traces: dict[int, Mapping[str, Any]] = {}
    return_diagnostics: dict[int, Mapping[str, Any]] = {}
    common_layout: Mapping[str, Any] | None = None
    task_audit: Mapping[str, Any] | None = None
    prime_evidence: Mapping[str, Any] | None = None
    for update in EXPECTED_UPDATES:
        if _provenance_snapshot(inputs) != starting_snapshot:
            raise RuntimeError(f"trace provenance changed before update{update} evaluation")
        _load_actor_policy(actor, checkpoints[update], update)
        _seed_everything()
        cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=1)
        cfg.seed = FIXED_SEED
        audit = audit_task_space_ppo_env_cfg(cfg, expected_num_envs=1)
        env = ManagerBasedRlEnv(cfg=cfg, device=DEVICE)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
            prime = prime_sonic_true23_training_environment(wrapped)
            expected = inputs.parent_json[f"evaluations/evaluation_update_{update}.json"]
            if prime != expected["prime"] or audit != expected["task_audit"]:
                raise ValueError(f"update{update} trace env/prime differs from parent evaluation")
            executed = run_policy_trace(
                policy=actor,
                wrapped_env=wrapped,
                update_count=update,
                expected_evaluation=expected,
                provenance_snapshot_sha256=starting_snapshot,
            )
        finally:
            env.close()
        if common_layout is None:
            common_layout = executed["layout"]
            task_audit = audit
            prime_evidence = prime
        elif executed["layout"] != common_layout or audit != task_audit or prime != prime_evidence:
            raise ValueError("trace pair environment/layout identity mismatch")
        traces[update] = executed["trace"]
        return_diagnostics[update] = executed["parent_episode_return_diagnostic"]
        if _provenance_snapshot(inputs) != starting_snapshot:
            raise RuntimeError(f"trace provenance changed after update{update} evaluation")
    assert common_layout is not None and task_audit is not None and prime_evidence is not None
    comparison = compare_trace_pair(traces[0], traces[5], common_layout)
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": TRACE_KIND,
        "purpose": "deterministic_update0_vs_update5_diagnostic_only",
        "evaluation_seed": FIXED_SEED,
        "parent_run_dir": str(inputs.run_dir),
        "provenance": inputs.provenance,
        "layout": common_layout,
        "task_audit": task_audit,
        "prime": prime_evidence,
        "capture_contract": CAPTURE_CONTRACT,
        "traces": {"update0": traces[0], "update5": traces[5]},
        "historical_parent_episode_return_diagnostics": {
            "update0": return_diagnostics[0],
            "update5": return_diagnostics[5],
        },
        "comparison": comparison,
        "training_updates": 0,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    assert_publication_boundary(report)
    if _provenance_snapshot(inputs) != starting_snapshot:
        raise RuntimeError("trace provenance changed before publication")
    return report


def assert_publication_boundary(value: Any) -> None:
    forbidden_true = {
        "teacher_labels_used",
        "support_qualified",
        "promotion_eligible",
        "hardware_authorized",
        "deployment_ready",
    }

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError(f"trace publication key is not string at {path}")
                if key in forbidden_true and child is not False:
                    raise ValueError(f"trace publication boundary violated at {path}.{key}")
                if key == "training_updates" and child != 0:
                    raise ValueError(f"trace training boundary violated at {path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif item is None or isinstance(item, (str, bool, int)):
            return
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"trace publication contains nonfinite float at {path}")
        else:
            raise ValueError(f"trace publication contains unsupported value at {path}")

    visit(value, "trace")


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    assert_publication_boundary(value)
    output = path.expanduser().absolute()
    _regular_directory(output.parent, "trace output parent")
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite trace output: {output}")
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def failure_report(error: Exception) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": FAILURE_KIND,
        "error": {"type": type(error).__name__, "message": str(error)},
        "simulator_trace_complete": False,
        "training_updates": 0,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    partial = getattr(error, "partial_evidence", None)
    if partial is not None:
        _assert_scalar_evidence(partial)
        report["partial_scalar_evidence"] = dict(partial)
    return report


__all__ = [
    "ACTION_DIVERGENCE_ATOL",
    "CAPTURE_CONTRACT",
    "CONTACT_FORCE_DIVERGENCE_ATOL_N",
    "EE_DIVERGENCE_ATOL_M",
    "EXPECTED_CONTRACT_SHA256",
    "EXPECTED_MATERIAL_SHA256",
    "EXPECTED_POLICY_SHA256",
    "FIXED_SEED",
    "KNOWN_CONTROL_PLANE_PATHS",
    "LANDING_FORCE_DIVERGENCE_ATOL_N",
    "PARENT_RUN_FILE_SHA256",
    "PRIOR_FAILED_TRACE_FILENAME",
    "PRIOR_FAILED_TRACE_SHA256",
    "REWARD_DIVERGENCE_ATOL",
    "RETRY_TRACE_FILENAME",
    "SECOND_FAILED_TRACE_FILENAME",
    "SECOND_FAILED_TRACE_SHA256",
    "TraceReproductionError",
    "assert_publication_boundary",
    "classify_control_plane_drift",
    "compare_trace_pair",
    "decode_reward_terms",
    "execute_checkpoint_trace",
    "failure_report",
    "frames_to_series",
    "parent_episode_return_diagnostic",
    "resolve_trace_inputs",
    "reproduction_scalar_evidence",
    "run_policy_trace",
    "trace_preflight",
    "validate_parent_evaluation_reproduction",
    "validate_trace_layout",
    "validate_trace_series",
    "write_json_exclusive",
]
