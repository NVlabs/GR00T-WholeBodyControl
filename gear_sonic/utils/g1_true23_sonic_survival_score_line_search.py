"""Pure contract, gradient, policy-state, and admission helpers for survival search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import math
from pathlib import Path
from typing import Any

import torch

from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state, sha256_file

CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_survival_score_line_search_v1.json"
)
CONTRACT_KIND = "g1_true23_sonic_survival_score_line_search_contract_v1"
CONTRACT_SHA256 = "5c307f7060deb0ba0e57b09d3300e5c0b129b31bc1d2b2bc04e4f40b05277bfa"
PPO_LINE_SEARCH_SHA256 = "8d9ecb6adc8437efdee24a42b13e5200626628be06c2a295f161b461b00ad1cc"
QUARTER_RESULT_SHA256 = "3d310e57a1a168437d68c7675760ce15ab32594f87ef6526d656f31a48831953"
TARGET_DIRECTION_L2 = 0.002365529660531515
SCALES = (-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0)
GRADIENT_BATCH_SIZE = 512
GRADIENT_BATCHES = 40
STATE_PARAMETER_NAMES = tuple(name.removeprefix("core.") for name in fs.TRAINABLE_ACTOR_PARAMETERS)
ZERO_COUNTS = (
    "nonfinite_count",
    "raw_clip_required_count",
    "action_semantics_mismatch_count",
    "q9_discontinuity_count",
)


def load_survival_contract(repository_root: str | Path) -> Mapping[str, Any]:
    root = Path(repository_root).expanduser().resolve(strict=True)
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or path.parent != (root / CONTRACT_RELATIVE_PATH.parent).resolve(strict=True):
        raise ValueError("survival-score contract path drift")
    contract = fs._strict_json(path, "survival-score contract")
    if sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("survival-score contract SHA256 mismatch")
    validate_survival_contract(contract)
    return contract


def validate_survival_contract(contract: Mapping[str, Any]) -> None:
    if set(contract) != {
        "schema_version",
        "kind",
        "role",
        "seed",
        "parent_full_support_contract",
        "negative_evidence",
        "rollout",
        "survival_score",
        "evaluation_gate",
        "boundaries",
    }:
        raise ValueError("survival-score contract schema mismatch")
    parent = fs._mapping(contract.get("parent_full_support_contract"), "survival parent")
    negative = fs._mapping(contract.get("negative_evidence"), "survival negative evidence")
    rollout = fs._mapping(contract.get("rollout"), "survival rollout")
    score = fs._mapping(contract.get("survival_score"), "survival score")
    gate = fs._mapping(contract.get("evaluation_gate"), "survival gate")
    boundary = fs._mapping(contract.get("boundaries"), "survival boundary")
    if (
        contract.get("schema_version") != 1
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("seed") != fs.FIXED_SEED
        or parent
        != {
            "relative_path": fs.CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": fs.CONTRACT_SHA256,
        }
        or negative.get("ppo_delta_line_search_sha256") != PPO_LINE_SEARCH_SHA256
        or negative.get("quarter_step_result_sha256") != QUARTER_RESULT_SHA256
        or negative.get("quarter_step_terminal_q9") != 160
        or negative.get("quarter_step_completed_transitions") != 152
        or negative.get("quarter_step_candidate_selected") is not False
        or negative.get("quarter_vs_full_delta_cosine") != 0.2840025806087036
        or rollout
        != {
            "num_envs": fs.NUM_ENVS,
            "num_steps_per_env": fs.NUM_STEPS_PER_ENV,
            "training_transitions": fs.TRAINING_TRANSITIONS,
            "first_episode_only": True,
            "autoreset_rows_excluded": True,
            "pre_adam_full_support_gate_required": True,
            "optimizer_steps": 0,
            "critic_updates": 0,
        }
        or score.get("gradient_parameters") != list(fs.TRAINABLE_ACTOR_PARAMETERS)
        or score.get("gradient_batches") != GRADIENT_BATCHES
        or score.get("gradient_batch_size") != GRADIENT_BATCH_SIZE
        or score.get("direction") != "negative_gradient"
        or score.get("direction_l2_target") != TARGET_DIRECTION_L2
        or score.get("scales") != list(SCALES)
        or gate.get("baseline_completed_transitions") != 155
        or gate.get("baseline_terminal_q9") != 163
        or gate.get("candidate_minimum_completed_transitions") != 156
        or gate.get("candidate_minimum_terminal_q9") != 164
        or gate.get("required_termination_names") != ["ee_body_pos"]
        or gate.get("required_zero_counts") != list(ZERO_COUNTS)
        or boundary
        != {
            "diagnostic_candidate_only": True,
            "teacher_labels_used": False,
            "dagger_data": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
            "robot_or_network_commands_permitted": False,
        }
    ):
        raise ValueError("survival-score contract semantic mismatch")


def first_episode_mask_and_success(dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        type(dones) is not torch.Tensor
        or tuple(dones.shape) != (fs.NUM_STEPS_PER_ENV, fs.NUM_ENVS, 1)
        or dones.dtype != torch.uint8
    ):
        raise ValueError("survival-score dones must be uint8 [160,128,1]")
    done = dones.squeeze(-1).to(dtype=torch.bool)
    active = torch.ones_like(done)
    if fs.NUM_STEPS_PER_ENV > 1:
        active[1:] = torch.cumprod((~done[:-1]).to(dtype=torch.int64), dim=0).to(dtype=torch.bool)
    success = active[-1] & ~done[-1]
    if not torch.equal(active[0], torch.ones_like(active[0])):
        raise RuntimeError("survival-score first row must be active")
    if bool(torch.any(done & ~active)):
        # Autoreset rows can contain later dones, but they are deliberately outside mask.
        pass
    return active, success


def normalized_negative_gradient_direction(
    gradients: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if set(gradients) != set(STATE_PARAMETER_NAMES):
        raise ValueError("survival-score gradient names mismatch")
    ordered: list[torch.Tensor] = []
    for name in STATE_PARAMETER_NAMES:
        value = gradients[name]
        if (
            type(value) is not torch.Tensor
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"survival-score gradient invalid: {name}")
        ordered.append(value.detach().cpu().contiguous())
    raw_norm = torch.linalg.vector_norm(torch.cat([value.reshape(-1).double() for value in ordered])).item()
    if not math.isfinite(raw_norm) or raw_norm <= 0.0:
        raise ValueError("survival-score gradient norm must be positive finite")
    factor = -TARGET_DIRECTION_L2 / raw_norm
    direction = {
        name: (gradients[name].detach().cpu().float() * factor).contiguous() for name in STATE_PARAMETER_NAMES
    }
    observed_norm = torch.linalg.vector_norm(
        torch.cat([direction[name].reshape(-1).double() for name in STATE_PARAMETER_NAMES])
    ).item()
    if abs(observed_norm - TARGET_DIRECTION_L2) > 1e-9:
        raise RuntimeError("survival-score normalized direction norm mismatch")
    return direction, {
        "raw_gradient_l2": raw_norm,
        "normalization_factor": factor,
        "normalized_direction_l2": observed_norm,
        "target_direction_l2": TARGET_DIRECTION_L2,
    }


def construct_scaled_policy_state(
    baseline: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
    scale: float,
) -> dict[str, torch.Tensor]:
    if scale not in SCALES:
        raise ValueError("survival-score policy scale not predeclared")
    if set(direction) != set(STATE_PARAMETER_NAMES):
        raise ValueError("survival-score direction keys mismatch")
    state = {name: value.detach().cpu().contiguous().clone() for name, value in baseline.items()}
    for name in STATE_PARAMETER_NAMES:
        if name not in state or state[name].shape != direction[name].shape:
            raise ValueError(f"survival-score policy tensor mismatch: {name}")
        state[name] = torch.add(state[name], direction[name], alpha=float(scale)).contiguous()
    observed = inspect_true23_policy_state({"policy_state_dict": state}, reference_profile=fs.REFERENCE_PROFILE)
    if scale == 0.0 and observed != fs.INITIAL_OVERLAY_POLICY_STATE_SHA256:
        raise RuntimeError("survival-score zero scale differs from baseline")
    return state


def assess_survival_evaluations(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_scales = list(SCALES)
    if len(records) != len(expected_scales) or [record.get("scale") for record in records] != expected_scales:
        raise ValueError("survival-score evaluation scale order mismatch")
    baseline = records[expected_scales.index(0.0)]
    base_return = baseline.get("episode_return")
    baseline_ok = (
        baseline.get("completed_transitions") == 155
        and baseline.get("terminal_q9") == 163
        and baseline.get("termination_names") == ["ee_body_pos"]
        and all(baseline.get(name) == 0 for name in ZERO_COUNTS)
        and not isinstance(base_return, bool)
        and isinstance(base_return, (int, float))
        and math.isfinite(float(base_return))
    )
    floor = float(base_return) - max(0.2 * abs(float(base_return)), 50.0) if baseline_ok else math.nan
    admissible: list[Mapping[str, Any]] = []
    for record in records:
        value = record.get("episode_return")
        if (
            record.get("scale") != 0.0
            and record.get("completed_transitions", -1) >= 156
            and record.get("terminal_q9", -1) >= 164
            and record.get("termination_names") == ["ee_body_pos"]
            and all(record.get(name) == 0 for name in ZERO_COUNTS)
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= floor
        ):
            admissible.append(record)
    selected: Mapping[str, Any] | None = None
    if baseline_ok and admissible:
        selected = max(
            admissible,
            key=lambda record: (
                int(record["completed_transitions"]),
                float(record["episode_return"]),
                -abs(float(record["scale"])),
            ),
        )
    return {
        "baseline_passed": baseline_ok,
        "candidate_selected": selected is not None,
        "selected_scale": float(selected["scale"]) if selected is not None else None,
        "selected_policy_state_sha256": selected.get("policy_state_sha256") if selected is not None else None,
        "selected_completed_transitions": selected.get("completed_transitions") if selected is not None else None,
        "selected_terminal_q9": selected.get("terminal_q9") if selected is not None else None,
        "candidate_reward_floor": floor if baseline_ok else None,
        "stop_reason": "survival_score_q164_pass" if selected is not None else "survival_score_no_q164_candidate",
        "diagnostic_candidate_only": True,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def candidate_checkpoint(
    *,
    state: Mapping[str, torch.Tensor],
    assessment: Mapping[str, Any],
    gradient_evidence: Mapping[str, Any],
    rollout_evidence_sha256: str,
    material_manifest_sha256: str,
) -> dict[str, Any]:
    if assessment.get("candidate_selected") is not True:
        raise ValueError("survival-score checkpoint requires selected candidate")
    policy = {name: value.detach().cpu().contiguous().clone() for name, value in state.items()}
    policy_sha = inspect_true23_policy_state({"policy_state_dict": policy}, reference_profile=fs.REFERENCE_PROFILE)
    if policy_sha != assessment.get("selected_policy_state_sha256"):
        raise ValueError("survival-score selected policy identity mismatch")
    return {
        "schema_version": 1,
        "kind": "g1_true23_sonic_survival_score_candidate_checkpoint_v1",
        "contract_sha256": CONTRACT_SHA256,
        "material_manifest_sha256": material_manifest_sha256,
        "rollout_evidence_sha256": rollout_evidence_sha256,
        "policy_state_dict": policy,
        "policy_state_sha256": policy_sha,
        "selected_scale": assessment["selected_scale"],
        "gradient_evidence": copy.deepcopy(dict(gradient_evidence)),
        "training_transitions": fs.TRAINING_TRANSITIONS,
        "gradient_computations": 1,
        "optimizer_steps": 0,
        "critic_updates": 0,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


__all__ = [
    "CONTRACT_SHA256",
    "GRADIENT_BATCH_SIZE",
    "SCALES",
    "STATE_PARAMETER_NAMES",
    "TARGET_DIRECTION_L2",
    "assess_survival_evaluations",
    "candidate_checkpoint",
    "construct_scaled_policy_state",
    "first_episode_mask_and_success",
    "load_survival_contract",
    "normalized_negative_gradient_direction",
    "validate_survival_contract",
]
