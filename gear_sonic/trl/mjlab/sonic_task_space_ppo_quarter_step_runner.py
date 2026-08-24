"""One bounded quarter-rate full-support PPO update for genuine SONIC.

This runner reuses the already-audited 128 x 160 collection and pre-Adam
coverage gate.  Only executed optimizer learning rate changes: 1e-5 to
2.5e-6.  It starts from exact update-0 overlay, never resumes model1/model5,
and permits exactly one update/evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from gear_sonic.trl.mjlab import sonic_task_space_ppo_full_support_runner as fs
from gear_sonic.utils.g1_23dof_artifact import inspect_true23_policy_state

CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_task_space_ppo_quarter_step_v1.json"
)
CONTRACT_KIND = "g1_true23_sonic_task_space_ppo_quarter_step_contract_v1"
CONTRACT_SHA256 = "07d6c426614e05cd9c4b4207777c6bc4caa9ba16d6874f6e693c49f1f3ac335d"
QUARTER_LEARNING_RATE = 2.5e-6
LINE_SEARCH_REPORT_SHA256 = "8d9ecb6adc8437efdee24a42b13e5200626628be06c2a295f161b461b00ad1cc"
LINE_SEARCH_POLICY_SHA256 = "bb14ccc4fc465d3c61580c3f3b2f2512ddcd3fc0d9bf49ed1fa7cbd048585113"
EVALUATION_ZERO_COUNTS = (
    "nonfinite_count",
    "raw_clip_required_count",
    "action_semantics_mismatch_count",
    "q9_discontinuity_count",
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def load_quarter_step_contract(repository_root: str | Path) -> Mapping[str, Any]:
    root = Path(repository_root).expanduser().resolve(strict=True)
    path = (root / CONTRACT_RELATIVE_PATH).resolve(strict=True)
    if path.is_symlink() or path.parent != (root / CONTRACT_RELATIVE_PATH.parent).resolve(strict=True):
        raise ValueError("quarter-step contract path drift")
    payload = fs._strict_json(path, "quarter-step contract")
    if fs.sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("quarter-step contract SHA256 mismatch")
    validate_quarter_step_contract(payload)
    return payload


def validate_quarter_step_contract(contract: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "kind",
        "role",
        "seed",
        "parent_full_support_contract",
        "line_search_evidence",
        "update",
        "evaluation_gate",
        "boundaries",
    }
    if set(contract) != expected_keys:
        raise ValueError("quarter-step contract schema mismatch")
    parent = fs._mapping(contract.get("parent_full_support_contract"), "quarter-step parent contract")
    line = fs._mapping(contract.get("line_search_evidence"), "quarter-step line search")
    update = fs._mapping(contract.get("update"), "quarter-step update")
    gate = fs._mapping(contract.get("evaluation_gate"), "quarter-step evaluation gate")
    boundaries = fs._mapping(contract.get("boundaries"), "quarter-step boundaries")
    if (
        contract.get("schema_version") != 1
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("seed") != fs.FIXED_SEED
        or parent
        != {
            "relative_path": str(fs.CONTRACT_RELATIVE_PATH).replace("\\", "/"),
            "sha256": fs.CONTRACT_SHA256,
        }
        or line.get("linux_path")
        != "/root/g1_true23_runs/sonic_task_space_ppo_full_support_delta_line_search_v1_retry1.json"
        or line.get("sha256") != LINE_SEARCH_REPORT_SHA256
        or line.get("classification") != "excessive_update_magnitude"
        or line.get("selected_probe") != "alpha_plus_0_25"
        or line.get("selected_probe_policy_sha256") != LINE_SEARCH_POLICY_SHA256
        or line.get("selected_probe_completed_transitions") != 155
        or line.get("selected_probe_terminal_q9") != 163
        or line.get("selected_probe_aligned_reward_delta") != 0.08949041042074413
        or update
        != {
            "num_envs": fs.NUM_ENVS,
            "num_steps_per_env": fs.NUM_STEPS_PER_ENV,
            "training_transitions": fs.TRAINING_TRANSITIONS,
            "maximum_updates": 1,
            "optimizer_steps": fs.OPTIMIZER_STEPS,
            "learning_rate": QUARTER_LEARNING_RATE,
            "parent_learning_rate": fs.FIXED_LEARNING_RATE,
            "learning_rate_ratio": 0.25,
            "initial_policy_sha256": fs.INITIAL_OVERLAY_POLICY_STATE_SHA256,
            "fresh_critic_sha256": fs.INITIAL_CRITIC_STATE_SHA256,
        }
        or gate
        != {
            "baseline_completed_transitions": 155,
            "baseline_terminal_q9": 163,
            "candidate_minimum_completed_transitions": 156,
            "candidate_minimum_terminal_q9": 164,
            "required_termination_names": ["ee_body_pos"],
            "candidate_reward_floor": "paired_baseline_minus_max_0p2_abs_or_50",
        }
        or boundaries
        != {
            "diagnostic_candidate_only": True,
            "teacher_labels_used": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
            "robot_or_network_commands_permitted": False,
        }
    ):
        raise ValueError("quarter-step contract semantic mismatch")


class SonicTaskSpacePpoQuarterStepRunner(fs.SonicTaskSpacePpoFullSupportRunner):
    """Full-support runner with one predeclared quarter-rate Adam update."""

    def __init__(self, *args: Any, quarter_step_contract: Mapping[str, Any], **kwargs: Any) -> None:
        validate_quarter_step_contract(quarter_step_contract)
        self._quarter_step_bootstrap = True
        self._quarter_step_contract = copy.deepcopy(dict(quarter_step_contract))
        super().__init__(*args, **kwargs)
        self.alg.learning_rate = QUARTER_LEARNING_RATE
        for group in self.alg.optimizer.param_groups:
            group["lr"] = QUARTER_LEARNING_RATE
        self._quarter_step_bootstrap = False
        self._assert_execution_boundary()

    def _assert_execution_boundary(self) -> None:
        if getattr(self, "_quarter_step_bootstrap", False):
            fs.SonicTaskSpacePpoFullSupportRunner._assert_execution_boundary(self)
            return
        actor = self.alg.get_policy()
        named_actor = dict(actor.named_parameters())
        for name, initial in self._frozen_actor_initial.items():
            if name not in named_actor or not torch.equal(named_actor[name].detach(), initial):
                raise RuntimeError(f"quarter-step frozen actor changed: {name}")
        if not torch.all(named_actor[fs.STD_PARAMETER].detach() == fs.EXPLORATION_STD):
            raise RuntimeError("quarter-step fixed std changed")
        optimizer_ids = {
            id(parameter) for group in self.alg.optimizer.param_groups for parameter in group["params"]
        }
        if (
            optimizer_ids != self._optimizer_parameter_ids
            or bool(optimizer_ids & self._frozen_parameter_ids)
            or type(self.alg.optimizer) is not torch.optim.Adam
            or float(self.alg.learning_rate) != QUARTER_LEARNING_RATE
            or any(float(group["lr"]) != QUARTER_LEARNING_RATE for group in self.alg.optimizer.param_groups)
        ):
            raise RuntimeError("quarter-step optimizer boundary changed")
        update = self._require_counter_coherence()
        expected = {
            "fresh": (0, 0, 0, 0),
            "rollout_ready": (0, 0, fs.TRAINING_TRANSITIONS, fs.NUM_STEPS_PER_ENV),
            "rollout_failed": (0, 0, fs.TRAINING_TRANSITIONS, fs.NUM_STEPS_PER_ENV),
            "updated": (1, fs.OPTIMIZER_STEPS, fs.TRAINING_TRANSITIONS, 0),
        }
        if self._phase not in expected:
            raise RuntimeError("quarter-step runner phase invalid")
        if (
            update,
            self._optimizer_step_count,
            self._executed_training_transitions,
            self._storage_step(),
        ) != expected[self._phase]:
            raise RuntimeError("quarter-step phase counters diverged")
        policy = self._policy_state_adapter.state_dict()
        policy_sha = inspect_true23_policy_state(
            {"policy_state_dict": policy}, reference_profile=fs.REFERENCE_PROFILE
        )
        _, critic_mlp = fs._critic_state_subsets(self.alg.critic.state_dict())
        if self._phase in {"fresh", "rollout_ready", "rollout_failed"}:
            if (
                policy_sha != fs.INITIAL_OVERLAY_POLICY_STATE_SHA256
                or fs._state_sha256(critic_mlp) != self._initial_critic_mlp_state_sha256
                or self.alg.optimizer.state
            ):
                raise RuntimeError("quarter-step pre-Adam state mutated")
        elif self._phase == "updated":
            _, trainable = fs._policy_state_subsets(policy)
            if (
                policy_sha == fs.INITIAL_OVERLAY_POLICY_STATE_SHA256
                or fs._state_sha256(trainable) == fs.INITIAL_TRAINABLE_ACTOR_STATE_SHA256
                or len(self.alg.optimizer.state) != fs.EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT
            ):
                raise RuntimeError("quarter-step update did not adapt actor exactly once")

    def quarter_step_checkpoint(self) -> dict[str, Any]:
        self._assert_execution_boundary()
        if self._phase != "updated":
            raise RuntimeError("quarter-step checkpoint requires completed update")
        policy = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in self._policy_state_adapter.state_dict().items()
        }
        frozen, trainable = fs._policy_state_subsets(policy)
        return {
            "schema_version": 1,
            "kind": "g1_true23_sonic_task_space_ppo_quarter_step_checkpoint_v1",
            "contract_sha256": CONTRACT_SHA256,
            "parent_full_support_contract_sha256": fs.CONTRACT_SHA256,
            "line_search_report_sha256": LINE_SEARCH_REPORT_SHA256,
            "policy_state_dict": policy,
            "policy_state_sha256": inspect_true23_policy_state(
                {"policy_state_dict": policy}, reference_profile=fs.REFERENCE_PROFILE
            ),
            "frozen_actor_state_sha256": fs._state_sha256(frozen),
            "trainable_actor_state_sha256": fs._state_sha256(trainable),
            "learning_rate": QUARTER_LEARNING_RATE,
            "completed_update_count": self.completed_update_count,
            "executed_training_transitions": self._executed_training_transitions,
            "optimizer_step_count": self._optimizer_step_count,
            "rollout_evidence_sha256": self._rollout_evidence_sha256,
            "candidate_selected": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "deployment_ready": False,
            "hardware_authorized": False,
        }


def assess_quarter_step_evaluations(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if [record.get("update_count") for record in records] != [0, 1]:
        raise ValueError("quarter-step evaluations must be exact [0,1]")
    baseline, candidate = records
    baseline_ok = (
        baseline.get("completed_transitions") == 155
        and baseline.get("terminal_q9") == 163
        and baseline.get("termination_names") == ["ee_body_pos"]
        and all(baseline.get(name) == 0 for name in EVALUATION_ZERO_COUNTS)
    )
    base_return = baseline.get("episode_return")
    candidate_return = candidate.get("episode_return")
    finite_returns = (
        not isinstance(base_return, bool)
        and isinstance(base_return, (int, float))
        and math.isfinite(float(base_return))
        and not isinstance(candidate_return, bool)
        and isinstance(candidate_return, (int, float))
        and math.isfinite(float(candidate_return))
    )
    floor = float(base_return) - max(0.2 * abs(float(base_return)), 50.0) if finite_returns else math.nan
    candidate_ok = (
        candidate.get("completed_transitions", -1) >= 156
        and candidate.get("terminal_q9", -1) >= 164
        and candidate.get("termination_names") == ["ee_body_pos"]
        and all(candidate.get(name) == 0 for name in EVALUATION_ZERO_COUNTS)
        and finite_returns
        and float(candidate_return) >= floor
    )
    selected = baseline_ok and candidate_ok
    return {
        "baseline_passed": baseline_ok,
        "candidate_passed": candidate_ok,
        "candidate_selected": selected,
        "candidate_reward_floor": floor if finite_returns else None,
        "stop_reason": "quarter_step_q164_pass" if selected else "quarter_step_no_q164_or_reward_improvement",
        "diagnostic_candidate_only": True,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


__all__ = [
    "CONTRACT_SHA256",
    "LINE_SEARCH_REPORT_SHA256",
    "QUARTER_LEARNING_RATE",
    "SonicTaskSpacePpoQuarterStepRunner",
    "assess_quarter_step_evaluations",
    "load_quarter_step_contract",
    "validate_quarter_step_contract",
]
