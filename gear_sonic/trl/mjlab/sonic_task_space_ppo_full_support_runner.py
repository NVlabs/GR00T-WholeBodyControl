"""One-rollout temporal-support ablation for genuine SONIC task-space PPO.

This module is additive, simulator-only, and deliberately cannot resume the
failed five-update pilot.  It reconstructs the exact update-0 overlay, gathers
one 128 x 160 rollout, publishes and rechecks first-episode coverage before the
first Adam step, performs exactly one PPO update, and never marks a checkpoint
as support- or deployment-qualified.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import hashlib
import math
import os
from pathlib import Path
import random
import tempfile
import time
from types import MethodType
from typing import Any

import numpy as np
import torch

from gear_sonic.envs.mjlab.sonic_true23_student_qualification import (
    DiagnosticSafeTargetNativeIl23JointPositionAction,
    capture_student_action_chain,
)
from gear_sonic.trl.mjlab.causal_history_runner import (
    CausalHistoryMjlabOnPolicyRunner,
)
from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
    ACTION_DIM,
    EXPECTED_CRITIC_PARAMETER_TENSOR_COUNT,
    EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT,
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    EXPLORATION_STD,
    FIXED_LEARNING_RATE,
    INITIAL_FROZEN_ACTOR_STATE_SHA256,
    INITIAL_TRAINABLE_ACTOR_STATE_SHA256,
    OPTIMIZER_PARAMETER_SHAPES,
    REFERENCE_PROFILE,
    RIGHT_WRIST_BODY_NAME,
    STD_PARAMETER,
    TRAINABLE_ACTOR_PARAMETERS,
    _action_chain_mismatch_count,
    _evaluation_q9,
    _finite_observations,
    _mapping,
    _nested_state_equal,
    _policy_state_subsets,
    _require_sha256,
    _state_sha256,
    _strict_json,
    _TaskSpaceEvaluationTerminalRecorder,
    _validate_critic_state_schema,
    _write_json_new,
    audit_task_space_ppo_env_cfg,
    load_overlay_policy_state,
    load_task_space_ppo_contract,
    make_task_space_ppo_env_cfg,
    validate_task_space_ppo_contract,
)
from gear_sonic.utils.g1_23dof_artifact import (
    inspect_true23_policy_state,
    sha256_file,
)

SCHEMA_VERSION = 1
CONTRACT_KIND = "g1_true23_sonic_task_space_ppo_full_support_contract_v1"
CHECKPOINT_KIND = "g1_true23_sonic_task_space_ppo_full_support_checkpoint_v1"
CONTRACT_RELATIVE_PATH = Path(
    "gear_sonic/config/sim_validation/g1_true23_sonic_task_space_ppo_full_support_v1.json"
)
CONTRACT_SHA256 = "7454d22296f2f1415531fd02d01ea3a18f8907270687a96434ea84be51051f3d"
BASE_CONTRACT_SHA256 = "a42d661e68e583e9f7e750e7fd8211ba36747172b8ff9be031414b2fe0f03044"
TRACE_SHA256 = "608e126d61d14225149706d90a876445489982c7d88a49a02088d76f453fb22d"
INITIAL_CRITIC_STATE_SHA256 = "95d31b41ec92194cee23490695c0f052e74df169d640612f9d55e176540eccf0"
INITIAL_OVERLAY_POLICY_STATE_SHA256 = "358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61"
FAILED_UPDATE5_POLICY_STATE_SHA256 = "fd0ea210bec6c403eb18fa0466ae11030b075ba1a5aa028cef94a458905c6f18"

FIXED_SEED = 20260805
NUM_ENVS = 128
NUM_STEPS_PER_ENV = 160
TRAINING_TRANSITIONS = NUM_ENVS * NUM_STEPS_PER_ENV
MAXIMUM_UPDATES = 1
PPO_LEARNING_EPOCHS = 2
PPO_MINI_BATCHES = 4
MINI_BATCH_SIZE = TRAINING_TRANSITIONS // PPO_MINI_BATCHES
OPTIMIZER_STEPS = PPO_LEARNING_EPOCHS * PPO_MINI_BATCHES
MINIMUM_FREE_CUDA_BYTES = 2_147_483_648
EXPECTED_CUDA_DEVICE_NAME = "NVIDIA GeForce RTX 3070 Laptop GPU"
EXPECTED_CUDA_TOTAL_MEMORY_BYTES = 8_589_410_304
EXPECTED_Q9_FIRST = 9
EXPECTED_Q9_LAST = EXPECTED_Q9_FIRST + NUM_STEPS_PER_ENV - 1
EXACT_SURVIVORS_THROUGH_Q9 = 88
MINIMUM_SURVIVORS = {126: 96, 133: 64, 154: 32, 163: 16}
EVALUATION_UPDATES = (0, 1)
ROLLOUT_EVIDENCE_FILENAME = "full_support_rollout_evidence.json"
RUNTIME_FILENAME = "sonic_task_space_ppo_full_support_runtime.json"

_SOURCE_CHECKPOINT_SHA256 = "85bd6de646905a44190dbf32c79737082bb604ab007a90a62e4fd2fdeeee6bd9"
_SOURCE_POLICY_SHA256 = "c3bfcb5c42929293b62425f155b59ccb731f57c98e8852c7f1e97094525684af"
_SOURCE_LINEAGE_SHA256 = "08bbd03d0df751328e449d3624d79167587b461d6835fa1f4a8742aad9ffa82a"
_V2_DECODER_SHA256 = "011740f86483323fc0f1c39ab25b784cf9411b401e56fee8b7a716664e921ee1"


def _regular_external_file(path_value: Any, expected_sha256: Any, context: str) -> Path:
    expected = _require_sha256(expected_sha256, f"{context} SHA256")
    if type(path_value) is not str or not path_value:
        raise ValueError(f"{context} path must be non-empty string")
    lexical = Path(path_value).expanduser()
    if lexical.is_symlink():
        raise ValueError(f"{context} may not be a symlink")
    try:
        path = lexical.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"{context} is missing") from error
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{context} identity mismatch")
    return path


def load_full_support_contract(repository_root: str | Path | None = None) -> Mapping[str, Any]:
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    path = root / CONTRACT_RELATIVE_PATH
    if path.is_symlink() or not path.is_file():
        raise ValueError("full-support PPO contract is missing or symlinked")
    actual = sha256_file(path)
    if actual != CONTRACT_SHA256:
        raise ValueError(f"full-support PPO contract SHA256 mismatch: expected {CONTRACT_SHA256}, got {actual}")
    contract = _strict_json(path, "full-support PPO contract")
    validate_full_support_contract(contract)
    return contract


def validate_full_support_contract(contract: Mapping[str, Any]) -> None:
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("role") != "offline_mjlab_genuine_sonic_single_full_support_rollout_single_update_only"
        or contract.get("seed") != FIXED_SEED
    ):
        raise ValueError("full-support PPO contract identity mismatch")
    base = _mapping(contract.get("base_task_space_contract"), "base task-space contract")
    if (
        base.get("relative_path") != "gear_sonic/config/sim_validation/g1_true23_sonic_task_space_ppo_v1.json"
        or base.get("sha256") != BASE_CONTRACT_SHA256
    ):
        raise ValueError("full-support base contract binding mismatch")
    evidence = _mapping(contract.get("diagnostic_evidence"), "diagnostic evidence")
    if (
        evidence.get("trace_sha256") != TRACE_SHA256
        or evidence.get("trace_training_updates") != 0
        or _mapping(evidence.get("update0"), "trace update0").get("terminal_q9") != 163
        or _mapping(evidence.get("failed_update5"), "trace update5").get("terminal_q9") != 133
    ):
        raise ValueError("full-support trace binding mismatch")
    parent = _mapping(contract.get("failed_parent_pilot"), "failed parent pilot")
    if (
        parent.get("checkpoint_resume_permitted") is not False
        or parent.get("model0_is_evidence_only") is not True
        or parent.get("model5_is_failed_evidence_only") is not True
        or any(parent.get(name) is not False for name in ("critic_reused", "optimizer_reused", "counters_reused"))
    ):
        raise ValueError("failed pilot reuse boundary mismatch")
    actor = _mapping(contract.get("actor_initialization"), "actor initialization")
    if (
        actor.get("overlay_policy_state_sha256") != INITIAL_OVERLAY_POLICY_STATE_SHA256
        or actor.get("initial_frozen_actor_state_sha256") != INITIAL_FROZEN_ACTOR_STATE_SHA256
        or actor.get("initial_trainable_actor_state_sha256") != INITIAL_TRAINABLE_ACTOR_STATE_SHA256
        or actor.get("initial_fresh_critic_state_sha256") != INITIAL_CRITIC_STATE_SHA256
        or actor.get("failed_model5_loaded") is not False
        or actor.get("failed_model5_resumed") is not False
        or actor.get("exploration_std") != EXPLORATION_STD
        or actor.get("exploration_std_trainable") is not False
        or actor.get("trainable_actor_parameters") != list(TRAINABLE_ACTOR_PARAMETERS)
    ):
        raise ValueError("full-support actor initialization mismatch")
    single = _mapping(contract.get("single_update"), "single update")
    expected_single = {
        "num_envs": NUM_ENVS,
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "training_transitions": TRAINING_TRANSITIONS,
        "maximum_updates": MAXIMUM_UPDATES,
        "num_learning_epochs": PPO_LEARNING_EPOCHS,
        "num_mini_batches": PPO_MINI_BATCHES,
        "mini_batch_size": MINI_BATCH_SIZE,
        "optimizer_steps": OPTIMIZER_STEPS,
        "learning_rate": FIXED_LEARNING_RATE,
        "schedule": "fixed",
        "clip_param": 0.1,
        "entropy_coef": 0.0,
        "gamma": 0.99,
        "lam": 0.95,
        "max_grad_norm": 0.25,
        "rollout_must_complete_before_optimizer": True,
        "rollout_reseeded_after_runner_construction": False,
        "pre_rollout_cpu_and_cuda_rng_state_hashes_published": True,
        "coverage_and_memory_probes_may_consume_rng": False,
        "random_episode_length_initialization": False,
    }
    if any(single.get(name) != value for name, value in expected_single.items()):
        raise ValueError("full-support single-update contract mismatch")
    coverage = _mapping(contract.get("pre_optimizer_coverage_gate"), "coverage gate")
    if (
        coverage.get("histogram_scope") != "first_episode_per_env_only"
        or coverage.get("autoreset_repeats_may_not_satisfy_gate") is not True
        or coverage.get("anchor_q9") != EXPECTED_Q9_FIRST
        or coverage.get("rollout_action_q9_last_if_uninterrupted") != EXPECTED_Q9_LAST
        or coverage.get("required_exact_survivors_through_q9") != {str(EXACT_SURVIVORS_THROUGH_Q9): NUM_ENVS}
        or coverage.get("required_minimum_survivors_at_q9")
        != {str(key): value for key, value in MINIMUM_SURVIVORS.items()}
        or coverage.get("required_storage_step") != NUM_STEPS_PER_ENV
        or coverage.get("required_training_transitions") != TRAINING_TRANSITIONS
        or coverage.get("required_critic_observation_normalizer_sample_count") != TRAINING_TRANSITIONS
    ):
        raise ValueError("full-support coverage contract mismatch")
    memory = _mapping(contract.get("pre_optimizer_memory_gate"), "memory gate")
    if (
        memory.get("device_name") != EXPECTED_CUDA_DEVICE_NAME
        or memory.get("device_total_memory_bytes") != EXPECTED_CUDA_TOTAL_MEMORY_BYTES
        or memory.get("mini_batch_size") != MINI_BATCH_SIZE
        or memory.get("minimum_free_cuda_bytes_before_update") != MINIMUM_FREE_CUDA_BYTES
        or memory.get("cuda_cache_released_before_measurement") is not True
        or memory.get("out_of_memory_is_fatal") is not True
    ):
        raise ValueError("full-support memory contract mismatch")
    gates = _mapping(contract.get("evaluation_gates"), "evaluation gates")
    if (
        gates.get("baseline_expected_completed_transitions") != 155
        or gates.get("baseline_expected_terminal_q9") != 163
        or gates.get("baseline_expected_termination_names") != ["ee_body_pos"]
        or gates.get("candidate_update") != 1
        or gates.get("candidate_minimum_completed_transitions") != 156
        or gates.get("candidate_minimum_terminal_q9") != 164
        or gates.get("candidate_reward_relative_tolerance") != 0.2
        or gates.get("candidate_reward_absolute_tolerance") != 50.0
    ):
        raise ValueError("full-support evaluation gate mismatch")
    checkpoint = _mapping(contract.get("checkpoint"), "checkpoint")
    if (
        checkpoint.get("kind") != CHECKPOINT_KIND
        or checkpoint.get("updates") != [0, 1]
        or checkpoint.get("resume_permitted") is not False
        or checkpoint.get("stock_export_forbidden") is not True
    ):
        raise ValueError("full-support checkpoint contract mismatch")
    boundaries = _mapping(contract.get("boundaries"), "boundaries")
    if boundaries.get("simulator_training_only") is not True or any(
        boundaries.get(name) is not False
        for name in (
            "teacher_labels_used",
            "dagger_data",
            "support_qualified",
            "promotion_eligible",
            "deployment_ready",
            "hardware_authorized",
            "robot_or_network_commands_permitted",
        )
    ):
        raise ValueError("full-support publication boundary mismatch")


def validate_bound_full_support_evidence(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Hash-check trace and every failed-pilot artifact without loading weights."""

    validate_full_support_contract(contract)
    evidence = _mapping(contract["diagnostic_evidence"], "diagnostic evidence")
    trace_path = _regular_external_file(evidence["trace_linux_path"], evidence["trace_sha256"], "trace v3")
    trace = _strict_json(trace_path, "trace v3")
    if (
        trace.get("kind") != evidence["trace_kind"]
        or trace.get("training_updates") != 0
        or any(
            trace.get(name) is not False
            for name in (
                "teacher_labels_used",
                "support_qualified",
                "promotion_eligible",
                "deployment_ready",
                "hardware_authorized",
            )
        )
    ):
        raise ValueError("trace v3 boundary mismatch")
    traces = _mapping(trace.get("traces"), "trace pair")
    for name, expected in (("update0", evidence["update0"]), ("update5", evidence["failed_update5"])):
        actual = _mapping(traces.get(name), f"trace {name}")
        if (
            actual.get("policy_state_sha256") != expected["policy_state_sha256"]
            or actual.get("completed_transitions") != expected["completed_transitions"]
            or actual.get("terminal_q9") != expected["terminal_q9"]
            or actual.get("termination_names") != expected["termination_names"]
        ):
            raise ValueError(f"trace {name} scalar identity mismatch")
    reward_layout = trace.get("layout", {}).get("reward_terms")
    if not isinstance(reward_layout, list):
        raise ValueError("trace reward layout missing")
    names = [item.get("name") if isinstance(item, Mapping) else None for item in reward_layout]
    try:
        barrier_index = names.index("right_wrist_prethreshold_barrier")
    except ValueError as error:
        raise ValueError("trace right-wrist barrier missing") from error
    for name, expected in (("update0", evidence["update0"]), ("update5", evidence["failed_update5"])):
        series = _mapping(_mapping(traces[name], f"trace {name}").get("series"), f"trace {name} series")
        q9 = series.get("q9")
        raw = series.get("reward_raw")
        if not isinstance(q9, list) or not isinstance(raw, list) or len(q9) != len(raw):
            raise ValueError("trace reward series mismatch")
        first = next(
            (
                int(frame_q9)
                for frame_q9, row in zip(q9, raw, strict=True)
                if isinstance(row, list) and len(row) == len(names) and float(row[barrier_index]) > 0.0
            ),
            None,
        )
        if first != expected["right_wrist_barrier_first_q9"]:
            raise ValueError(f"trace {name} barrier onset mismatch")
    comparison = _mapping(trace.get("comparison"), "trace comparison")
    if (
        _mapping(comparison.get("update5_terminal_worst_ee"), "trace terminal EE").get("body_name")
        != RIGHT_WRIST_BODY_NAME
    ):
        raise ValueError("trace terminal EE identity mismatch")
    parent = _mapping(contract["failed_parent_pilot"], "failed parent pilot")
    parent_dir = Path(parent["run_dir_linux_path"]).expanduser().resolve(strict=True)
    if not parent_dir.is_dir() or parent_dir.is_symlink():
        raise ValueError("failed parent pilot directory mismatch")
    artifacts = _mapping(parent.get("artifact_sha256"), "failed pilot artifact hashes")
    for relative, expected_hash in artifacts.items():
        if type(relative) is not str or Path(relative).is_absolute():
            raise ValueError("failed pilot artifact path must be relative")
        path = parent_dir / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(parent_dir):
            raise ValueError(f"failed pilot artifact path mismatch: {relative}")
        if sha256_file(path) != _require_sha256(expected_hash, f"failed pilot {relative} SHA256"):
            raise ValueError(f"failed pilot artifact SHA256 mismatch: {relative}")
    provenance_hashes = _mapping(
        _mapping(trace.get("provenance"), "trace provenance").get("parent_run_file_sha256"),
        "trace parent run hashes",
    )
    if dict(provenance_hashes) != dict(artifacts):
        raise ValueError("trace and contract failed-pilot artifact bindings diverge")
    return {
        "trace_path": str(trace_path),
        "trace_sha256": TRACE_SHA256,
        "failed_parent_run_dir": str(parent_dir),
        "failed_parent_artifact_count": len(artifacts),
        "failed_model5_loaded": False,
        "failed_model5_resumed": False,
        "training_updates": 0,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def _rng_state_sha256(value: torch.Tensor) -> str:
    if type(value) is not torch.Tensor or value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError("RNG state must be uint8 vector")
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def capture_rollout_rng_state_hashes() -> dict[str, Any]:
    """Hash RNG states without drawing from any stream."""

    python_hash = hashlib.sha256(repr(random.getstate()).encode("utf-8")).hexdigest()
    numpy_hash = hashlib.sha256(repr(np.random.get_state()).encode("utf-8")).hexdigest()
    cuda_states = torch.cuda.get_rng_state_all()
    return {
        "python_state_sha256": python_hash,
        "numpy_state_sha256": numpy_hash,
        "torch_cpu_state_sha256": _rng_state_sha256(torch.random.get_rng_state()),
        "torch_cuda_state_sha256": [_rng_state_sha256(state) for state in cuda_states],
        "cuda_device_count": len(cuda_states),
        "rng_draws_performed": 0,
    }


def _critic_state_subsets(
    state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    _validate_critic_state_schema(state)
    normalizer = {name: value for name, value in state.items() if name.startswith("obs_normalizer.")}
    mlp = {name: value for name, value in state.items() if name.startswith("mlp.")}
    if len(normalizer) != 4 or len(mlp) != EXPECTED_CRITIC_PARAMETER_TENSOR_COUNT:
        raise ValueError("critic state subset schema mismatch")
    return normalizer, mlp


def _clone_tensor_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if type(name) is not str or type(value) is not torch.Tensor:
            raise TypeError("tensor state snapshot must map strings to tensors")
        cloned[name] = value.detach().cpu().contiguous().clone()
    return cloned


def assess_full_support_rollout_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Pure fail-closed first-episode coverage and pre-Adam state gate."""

    violations: list[str] = []
    first_histogram = record.get("first_episode_q9_histogram")
    if not isinstance(first_histogram, list) or len(first_histogram) != NUM_STEPS_PER_ENV:
        violations.append("first_episode_histogram_shape")
        counts: dict[int, int] = {}
    else:
        counts = {}
        for expected_q9, item in zip(range(EXPECTED_Q9_FIRST, EXPECTED_Q9_LAST + 1), first_histogram, strict=True):
            if (
                not isinstance(item, Mapping)
                or item.get("q9") != expected_q9
                or isinstance(item.get("count"), bool)
                or not isinstance(item.get("count"), int)
                or not 0 <= item["count"] <= NUM_ENVS
            ):
                violations.append("first_episode_histogram_entry")
                counts = {}
                break
            counts[expected_q9] = int(item["count"])
    if counts:
        if any(counts[q9] != NUM_ENVS for q9 in range(EXPECTED_Q9_FIRST, EXACT_SURVIVORS_THROUGH_Q9 + 1)):
            violations.append("exact_supported_prefix_survivors")
        if any(counts[q9] < minimum for q9, minimum in MINIMUM_SURVIVORS.items()):
            violations.append("failure_region_survivors")
        ordered = [counts[q9] for q9 in range(EXPECTED_Q9_FIRST, EXPECTED_Q9_LAST + 1)]
        if any(right > left for left, right in zip(ordered, ordered[1:])):
            violations.append("first_episode_survivors_not_monotone")
    state = record.get("pre_adam_state")
    if not isinstance(state, Mapping):
        violations.append("pre_adam_state_missing")
    else:
        exact_state = {
            "storage_step": NUM_STEPS_PER_ENV,
            "executed_training_transitions": TRAINING_TRANSITIONS,
            "optimizer_step_count": 0,
            "optimizer_state_entry_count": 0,
            "completed_update_count": 0,
            "current_learning_iteration": 0,
            "critic_observation_normalizer_sample_count": TRAINING_TRANSITIONS,
        }
        if any(state.get(name) != value for name, value in exact_state.items()):
            violations.append("pre_adam_counters")
        if any(
            state.get(name) is not True
            for name in (
                "policy_state_unchanged",
                "frozen_actor_state_unchanged",
                "trainable_actor_state_unchanged",
                "critic_mlp_state_unchanged",
                "returns_finite",
                "advantages_finite",
            )
        ):
            violations.append("pre_adam_tensor_state")
        for name in (
            "policy_state_before_sha256",
            "policy_state_after_sha256",
            "frozen_actor_before_sha256",
            "frozen_actor_after_sha256",
            "trainable_actor_before_sha256",
            "trainable_actor_after_sha256",
            "critic_mlp_before_sha256",
            "critic_mlp_after_sha256",
            "critic_normalizer_before_sha256",
            "critic_normalizer_after_sha256",
        ):
            try:
                _require_sha256(state.get(name), name)
            except ValueError:
                violations.append("pre_adam_hash_schema")
                break
        if (
            state.get("policy_state_before_sha256") != state.get("policy_state_after_sha256")
            or state.get("policy_state_after_sha256") != INITIAL_OVERLAY_POLICY_STATE_SHA256
            or state.get("frozen_actor_before_sha256") != state.get("frozen_actor_after_sha256")
            or state.get("frozen_actor_after_sha256") != INITIAL_FROZEN_ACTOR_STATE_SHA256
            or state.get("trainable_actor_before_sha256") != state.get("trainable_actor_after_sha256")
            or state.get("trainable_actor_after_sha256") != INITIAL_TRAINABLE_ACTOR_STATE_SHA256
            or state.get("critic_mlp_before_sha256") != state.get("critic_mlp_after_sha256")
            or state.get("critic_normalizer_before_sha256") == state.get("critic_normalizer_after_sha256")
        ):
            violations.append("pre_adam_direct_hash_invariant")
        for name in ("std_before_sha256", "std_after_sha256"):
            try:
                _require_sha256(state.get(name), name)
            except ValueError:
                violations.append("pre_adam_std_hash_schema")
                break
        if state.get("std_before_sha256") != state.get("std_after_sha256"):
            violations.append("pre_adam_std_changed")
    rewards = record.get("first_episode_reward_and_termination_evidence")
    if not isinstance(rewards, Mapping):
        violations.append("reward_coverage_missing")
    else:
        numeric = (
            rewards.get("right_wrist_barrier_raw_sum"),
            rewards.get("right_wrist_barrier_weighted_sum"),
            rewards.get("worst_ee_raw_sum"),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in numeric
        ):
            violations.append("reward_coverage_nonfinite")
        elif (
            rewards.get("right_wrist_barrier_active_count", 0) < 1
            or float(rewards["right_wrist_barrier_raw_sum"]) <= 0.0
            or float(rewards["right_wrist_barrier_weighted_sum"]) >= 0.0
            or float(rewards["worst_ee_raw_sum"]) < 0.0
            or rewards.get("ee_body_pos_terminal_count", 0) < 1
        ):
            violations.append("reward_or_terminal_region_not_observed")
    memory = record.get("pre_optimizer_memory")
    if not isinstance(memory, Mapping):
        violations.append("memory_evidence_missing")
    elif (
        memory.get("device_name") != EXPECTED_CUDA_DEVICE_NAME
        or memory.get("device_total_memory_bytes") != EXPECTED_CUDA_TOTAL_MEMORY_BYTES
        or memory.get("mini_batch_size") != MINI_BATCH_SIZE
        or memory.get("free_cuda_bytes_before_update", 0) < MINIMUM_FREE_CUDA_BYTES
        or memory.get("minimum_required_free_cuda_bytes") != MINIMUM_FREE_CUDA_BYTES
        or memory.get("cuda_cache_released") is not True
    ):
        violations.append("pre_optimizer_memory_gate")
    rng = record.get("rng_state_hashes")
    if (
        not isinstance(rng, Mapping)
        or rng.get("pre_rollout_pre_probe") != rng.get("pre_rollout")
        or rng.get("post_rollout_pre_probe") != rng.get("post_probe")
        or not isinstance(rng.get("pre_rollout"), Mapping)
        or not isinstance(rng.get("post_rollout_pre_probe"), Mapping)
    ):
        violations.append("probe_rng_state_changed")
    for name in (
        "nonfinite_count",
        "q9_discontinuity_count",
        "action_semantics_mismatch_count",
    ):
        if record.get(name) != 0:
            violations.append(name)
    if record.get("total_inserted_transitions") != TRAINING_TRANSITIONS:
        violations.append("inserted_transition_count")
    return {
        "gate_passed": not violations,
        "violations": violations,
        "first_episode_scope_only": True,
        "autoreset_repeats_admitted_to_gate": False,
        "optimizer_permitted": not violations,
        "candidate_selected": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def validate_full_support_rollout_evidence(record: Mapping[str, Any]) -> Mapping[str, Any]:
    body = _mapping(record, "full-support rollout evidence")
    if (
        body.get("schema_version") != SCHEMA_VERSION
        or body.get("kind") != "g1_true23_sonic_task_space_full_support_rollout_evidence_v1"
        or body.get("contract_sha256") != CONTRACT_SHA256
        or body.get("trace_sha256") != TRACE_SHA256
        or body.get("failed_model5_loaded") is not False
        or body.get("failed_model5_resumed") is not False
        or body.get("optimizer_steps_at_publication") != 0
        or body.get("training_updates_at_publication") != 0
    ):
        raise ValueError("full-support rollout evidence identity mismatch")
    assessment = assess_full_support_rollout_evidence(body)
    if body.get("coverage_assessment") != assessment:
        raise ValueError("full-support rollout assessment mismatch")
    return body


def _checkpoint_header() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "training_state_evidence_included": True,
        "resume_permitted": False,
        "weights_only_load_validation_required": True,
        "support_qualified": False,
        "promotion_eligible": False,
        "deployment_ready": False,
        "hardware_authorized": False,
    }


def validate_full_support_checkpoint(
    checkpoint: Any,
    *,
    expected_run_materials_sha256: str | None = None,
) -> Mapping[str, Any]:
    body = _mapping(checkpoint, "full-support checkpoint")
    expected_keys = {
        "g1_true23_sonic_task_space_ppo_full_support_checkpoint",
        "policy_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "update_count",
        "trainer_state",
        "initial_critic_state_sha256",
        "optimizer_parameter_tensor_count",
        "optimizer_step_count",
        "executed_training_transitions",
        "storage_step",
        "policy_state_sha256",
        "critic_state_sha256",
        "frozen_actor_state_sha256",
        "trainable_actor_state_sha256",
        "initial_overlay_policy_state_sha256",
        "contract_sha256",
        "trace_sha256",
        "run_materials_sha256",
        "rollout_evidence_sha256",
        "source_actor",
        "training_boundary",
    }
    if (
        set(body) != expected_keys
        or body.get("g1_true23_sonic_task_space_ppo_full_support_checkpoint") != _checkpoint_header()
    ):
        raise ValueError("full-support checkpoint root/header mismatch")
    if body.get("contract_sha256") != CONTRACT_SHA256 or body.get("trace_sha256") != TRACE_SHA256:
        raise ValueError("full-support checkpoint evidence binding mismatch")
    run_materials = _require_sha256(body.get("run_materials_sha256"), "checkpoint materials SHA256")
    if expected_run_materials_sha256 is not None and run_materials != _require_sha256(
        expected_run_materials_sha256, "expected checkpoint materials SHA256"
    ):
        raise ValueError("full-support checkpoint materials mismatch")
    update = body.get("update_count")
    if isinstance(update, bool) or not isinstance(update, int) or update not in EVALUATION_UPDATES:
        raise ValueError("full-support checkpoint update mismatch")
    policy = _mapping(body.get("policy_state_dict"), "checkpoint policy")
    policy_hash = inspect_true23_policy_state({"policy_state_dict": policy}, reference_profile=REFERENCE_PROFILE)
    if body.get("policy_state_sha256") != policy_hash:
        raise ValueError("full-support checkpoint policy hash mismatch")
    if body.get("initial_overlay_policy_state_sha256") != INITIAL_OVERLAY_POLICY_STATE_SHA256:
        raise ValueError("full-support initial overlay binding mismatch")
    frozen, trainable = _policy_state_subsets(policy)
    frozen_hash = _state_sha256(frozen)
    trainable_hash = _state_sha256(trainable)
    if (
        body.get("frozen_actor_state_sha256") != frozen_hash
        or frozen_hash != INITIAL_FROZEN_ACTOR_STATE_SHA256
        or body.get("trainable_actor_state_sha256") != trainable_hash
    ):
        raise ValueError("full-support actor subset hash mismatch")
    if update == 0:
        if (
            policy_hash != INITIAL_OVERLAY_POLICY_STATE_SHA256
            or trainable_hash != INITIAL_TRAINABLE_ACTOR_STATE_SHA256
        ):
            raise ValueError("full-support update0 actor is not exact overlay")
    elif (
        policy_hash == INITIAL_OVERLAY_POLICY_STATE_SHA256
        or trainable_hash == INITIAL_TRAINABLE_ACTOR_STATE_SHA256
    ):
        raise ValueError("full-support update1 actor did not change")
    critic = _mapping(body.get("critic_state_dict"), "checkpoint critic")
    _validate_critic_state_schema(critic)
    critic_hash = _state_sha256(critic)
    critic_normalizer_count = int(critic["obs_normalizer.count"].detach().cpu().item())
    if (
        body.get("critic_state_sha256") != critic_hash
        or body.get("initial_critic_state_sha256") != INITIAL_CRITIC_STATE_SHA256
        or (update == 0 and critic_hash != INITIAL_CRITIC_STATE_SHA256)
        or critic_normalizer_count != update * TRAINING_TRANSITIONS
    ):
        raise ValueError("full-support critic identity mismatch")
    optimizer = _mapping(body.get("optimizer_state_dict"), "checkpoint optimizer")
    if set(optimizer) != {"state", "param_groups"}:
        raise ValueError("full-support optimizer schema mismatch")
    optimizer_state = _mapping(optimizer.get("state"), "checkpoint Adam state")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("full-support checkpoint requires one Adam group")
    group = _mapping(groups[0], "checkpoint Adam group")
    parameter_ids = group.get("params")
    if (
        not isinstance(parameter_ids, list)
        or parameter_ids != list(range(EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT))
        or body.get("optimizer_parameter_tensor_count") != EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT
        or float(group.get("lr", math.nan)) != FIXED_LEARNING_RATE
        or tuple(group.get("betas", ())) != (0.9, 0.999)
        or float(group.get("eps", math.nan)) != 1.0e-8
        or float(group.get("weight_decay", math.nan)) != 0.0
        or group.get("amsgrad") is not False
        or group.get("maximize") is not False
        or group.get("capturable") is not False
        or group.get("differentiable") is not False
    ):
        raise ValueError("full-support Adam group mismatch")
    expected_steps = update * OPTIMIZER_STEPS
    expected_transitions = update * TRAINING_TRANSITIONS
    if (
        body.get("optimizer_step_count") != expected_steps
        or body.get("executed_training_transitions") != expected_transitions
        or body.get("storage_step") != 0
    ):
        raise ValueError("full-support checkpoint execution counters mismatch")
    if update == 0:
        if optimizer_state or body.get("rollout_evidence_sha256") is not None:
            raise ValueError("full-support update0 must be fresh")
    else:
        _require_sha256(body.get("rollout_evidence_sha256"), "rollout evidence SHA256")
        if set(optimizer_state) != set(parameter_ids):
            raise ValueError("full-support Adam parameter coverage mismatch")
        for parameter_id, entry_value in optimizer_state.items():
            entry = _mapping(entry_value, f"Adam state {parameter_id}")
            if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("full-support Adam state schema mismatch")
            step = entry["step"]
            if isinstance(step, torch.Tensor):
                if step.numel() != 1 or not step.is_floating_point() or not bool(torch.isfinite(step).all()):
                    raise ValueError("full-support Adam step tensor mismatch")
                step = step.detach().cpu().item()
            if isinstance(step, bool) or not isinstance(step, (int, float)) or float(step) != OPTIMIZER_STEPS:
                raise ValueError("full-support Adam step count mismatch")
            expected_shape = OPTIMIZER_PARAMETER_SHAPES[int(parameter_id)]
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = entry[moment_name]
                if (
                    type(moment) is not torch.Tensor
                    or tuple(moment.shape) != expected_shape
                    or moment.dtype != torch.float32
                    or not bool(torch.isfinite(moment).all())
                ):
                    raise ValueError(f"full-support Adam {moment_name} mismatch")
    trainer = _mapping(body.get("trainer_state"), "checkpoint trainer state")
    if (
        trainer.get("completed_update_count") != update
        or trainer.get("current_learning_iteration") != update
        or trainer.get("env_common_step_counter") != update * NUM_STEPS_PER_ENV
        or trainer.get("algorithm_learning_rate") != FIXED_LEARNING_RATE
    ):
        raise ValueError("full-support trainer counters mismatch")
    source = _mapping(body.get("source_actor"), "checkpoint source actor")
    if source != {
        "checkpoint_sha256": _SOURCE_CHECKPOINT_SHA256,
        "policy_state_sha256": _SOURCE_POLICY_SHA256,
        "lineage_sha256": _SOURCE_LINEAGE_SHA256,
        "checkpoint_update_count": 250,
        "v2_decoder_sha256": _V2_DECODER_SHA256,
        "critic_reused": False,
        "optimizer_reused": False,
        "counters_reused": False,
        "failed_model5_loaded": False,
        "failed_model5_resumed": False,
    }:
        raise ValueError("full-support checkpoint source boundary mismatch")
    boundary = _mapping(body.get("training_boundary"), "checkpoint training boundary")
    if (
        boundary.get("trainable_actor_parameters") != list(TRAINABLE_ACTOR_PARAMETERS)
        or boundary.get("trainable_actor_parameter_count") != EXPECTED_TRAINABLE_PARAMETER_COUNT
        or boundary.get("exploration_std") != EXPLORATION_STD
        or boundary.get("exploration_std_trainable") is not False
        or any(
            boundary.get(name) is not False
            for name in (
                "teacher_labels_used",
                "support_qualified",
                "promotion_eligible",
                "hardware_authorized",
                "deployment_ready",
            )
        )
    ):
        raise ValueError("full-support checkpoint boundary mismatch")
    return body


class _FullSupportRewardEvidenceRecorder:
    """Clone pre-reset GPU evidence; materialize only after complete rollout."""

    def __init__(self, raw_env: Any) -> None:
        self.raw_env = raw_env
        self.manager = raw_env.reward_manager
        self._original_compute = self.manager.compute
        self.names = list(self.manager.active_terms)
        if (
            len(self.names) != len(set(self.names))
            or "right_wrist_prethreshold_barrier" not in self.names
            or "worst_ee_z_normalized_squared" not in self.names
        ):
            raise ValueError("full-support reward layout mismatch")
        self.barrier_index = self.names.index("right_wrist_prethreshold_barrier")
        self.worst_index = self.names.index("worst_ee_z_normalized_squared")
        self.barrier_weight = float(self.manager.get_term_cfg("right_wrist_prethreshold_barrier").weight)
        self.worst_weight = float(self.manager.get_term_cfg("worst_ee_z_normalized_squared").weight)
        self.dt = float(raw_env.step_dt)
        if self.barrier_weight != -25.0 or self.worst_weight != -20.0 or self.dt != 0.02:
            raise ValueError("full-support reward constants drifted")
        self.termination_names = list(raw_env.termination_manager.active_terms)
        if not self.termination_names or len(self.termination_names) != len(set(self.termination_names)):
            raise ValueError("full-support termination layout mismatch")
        self._armed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._pending_snapshot: dict[str, Any] | None = None
        self._snapshots: list[dict[str, Any]] = []
        self._materialized: dict[str, Any] | None = None

        def observed_compute(_manager: Any, dt: float) -> torch.Tensor:
            reward = self._original_compute(dt)
            self._capture(reward, dt)
            return reward

        self.manager.compute = MethodType(observed_compute, self.manager)

    def arm(self, q9: torch.Tensor, first_episode_active: torch.Tensor, action: torch.Tensor) -> None:
        if self._armed is not None or self._pending_snapshot is not None or self._materialized is not None:
            raise RuntimeError("full-support reward recorder was not consumed")
        self._armed = (q9.detach().clone(), first_episode_active.detach().clone(), action.detach().clone())

    def _capture(self, reward: torch.Tensor, dt: float) -> None:
        if self._armed is None or self._pending_snapshot is not None or dt != self.dt:
            raise RuntimeError("full-support reward compute occurred outside armed step")
        q9, active, action = self._armed
        rates = getattr(self.manager, "_step_reward", None)
        if (
            type(reward) is not torch.Tensor
            or reward.shape != (NUM_ENVS,)
            or not reward.is_floating_point()
            or type(rates) is not torch.Tensor
            or rates.shape != (NUM_ENVS, len(self.names))
            or not rates.is_floating_point()
            or type(q9) is not torch.Tensor
            or q9.shape != (NUM_ENVS,)
            or q9.dtype != torch.long
            or type(active) is not torch.Tensor
            or active.shape != (NUM_ENVS,)
            or active.dtype != torch.bool
            or type(action) is not torch.Tensor
            or action.shape != (NUM_ENVS, ACTION_DIM)
            or not action.is_floating_point()
        ):
            raise RuntimeError("full-support reward/action tensor mismatch")
        termination_manager = self.raw_env.termination_manager
        if list(termination_manager.active_terms) != self.termination_names:
            raise RuntimeError("full-support termination layout drifted")
        term_values: list[torch.Tensor] = []
        for name in self.termination_names:
            value = termination_manager.get_term(name)
            if type(value) is not torch.Tensor or value.shape != (NUM_ENVS,) or value.dtype != torch.bool:
                raise RuntimeError("full-support termination tensor mismatch")
            term_values.append(value.detach().clone())
        action_term = self.raw_env.action_manager.get_term("joint_pos")
        if type(action_term) is not DiagnosticSafeTargetNativeIl23JointPositionAction:
            raise TypeError("full-support runtime action term mismatch")
        action_values = {
            "raw_native": action_term.plain_sonic_raw_action_native,
            "candidate_target_hardware": action_term.candidate_target_hardware,
            "safe_native": action_term.safe_native_action,
            "final_target_hardware": action_term.final_target_hardware,
            "raw_clip_mask_native": action_term.raw_clip_mask_native,
        }
        for name, value in action_values.items():
            if (
                type(value) is not torch.Tensor
                or value.shape != (NUM_ENVS, ACTION_DIM)
                or (name == "raw_clip_mask_native" and value.dtype != torch.bool)
                or (name != "raw_clip_mask_native" and not value.is_floating_point())
            ):
                raise ValueError(f"full-support action snapshot metadata mismatch: {name}")
        self._pending_snapshot = {
            "q9": q9.detach().clone(),
            "active": active.detach().clone(),
            "action": action.detach().clone(),
            "reward": reward.detach().clone(),
            "rates": rates.detach().clone(),
            "terminations": [value.detach().clone() for value in term_values],
            "action_chain": {name: value.detach().clone() for name, value in action_values.items()},
        }

    def finish(self, dones: torch.Tensor) -> None:
        if self._armed is None or self._pending_snapshot is None:
            raise RuntimeError("full-support reward recorder did not capture step")
        if type(dones) is not torch.Tensor or dones.shape != (NUM_ENVS,):
            raise RuntimeError("full-support done tensor mismatch")
        self._pending_snapshot["dones"] = dones.detach().clone()
        self._snapshots.append(self._pending_snapshot)
        self._armed = None
        self._pending_snapshot = None

    def materialize(self) -> dict[str, Any]:
        if (
            self._armed is not None
            or self._pending_snapshot is not None
            or len(self._snapshots) != NUM_STEPS_PER_ENV
        ):
            raise RuntimeError("full-support reward recorder rollout is incomplete")
        if self._materialized is not None:
            return copy.deepcopy(self._materialized)
        barrier_active_count = 0
        barrier_raw_sum = 0.0
        barrier_weighted_sum = 0.0
        worst_raw_sum = 0.0
        termination_counts = {name: 0 for name in self.termination_names}
        barrier_q9_histogram: dict[int, int] = {}
        nonfinite_count = 0
        action_semantics_mismatch_count = 0
        raw_clip_required_count = 0
        for snapshot in self._snapshots:
            q9 = snapshot["q9"]
            active = snapshot["active"]
            action = snapshot["action"]
            reward = snapshot["reward"]
            rates = snapshot["rates"]
            dones = snapshot["dones"].to(dtype=torch.bool)
            terminations = snapshot["terminations"]
            tensors = (action, reward, rates, *snapshot["action_chain"].values())
            if any(tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()) for tensor in tensors):
                nonfinite_count += 1
            derived_done = torch.stack(terminations, dim=0).any(dim=0)
            if not torch.equal(dones, derived_done):
                raise RuntimeError("full-support done and termination evidence diverged")
            for name, values in zip(self.termination_names, terminations, strict=True):
                termination_counts[name] += int(torch.count_nonzero(values & active).detach().cpu().item())
            barrier_rate = rates[:, self.barrier_index]
            worst_rate = rates[:, self.worst_index]
            active_barrier = active & (barrier_rate < 0.0)
            barrier_active_count += int(torch.count_nonzero(active_barrier).detach().cpu().item())
            barrier_raw_sum += float((barrier_rate[active] / self.barrier_weight).sum().detach().cpu().item())
            barrier_weighted_sum += float((barrier_rate[active] * self.dt).sum().detach().cpu().item())
            worst_raw_sum += float((worst_rate[active] / self.worst_weight).sum().detach().cpu().item())
            for value in q9[active_barrier].detach().cpu().tolist():
                key = int(value)
                barrier_q9_histogram[key] = barrier_q9_histogram.get(key, 0) + 1
            mismatch, clip_count = _action_chain_mismatch_count(action, snapshot["action_chain"])
            action_semantics_mismatch_count += mismatch
            raw_clip_required_count += clip_count
        self._materialized = {
            "right_wrist_barrier_active_count": barrier_active_count,
            "right_wrist_barrier_raw_sum": barrier_raw_sum,
            "right_wrist_barrier_weighted_sum": barrier_weighted_sum,
            "worst_ee_raw_sum": worst_raw_sum,
            "ee_body_pos_terminal_count": termination_counts.get("ee_body_pos", 0),
            "termination_counts": termination_counts,
            "right_wrist_barrier_q9_histogram": [
                {"q9": q9, "count": count} for q9, count in sorted(barrier_q9_histogram.items())
            ],
            "nonfinite_count": nonfinite_count,
            "action_semantics_mismatch_count": action_semantics_mismatch_count,
            "raw_clip_required_count": raw_clip_required_count,
        }
        return copy.deepcopy(self._materialized)

    def restore(self) -> None:
        self.manager.compute = self._original_compute


def _vector_q9(raw_env: Any) -> torch.Tensor:
    command = raw_env.command_manager.get_term("motion")
    value = getattr(command, "time_steps", None)
    if type(value) is not torch.Tensor or value.shape != (NUM_ENVS,) or value.dtype != torch.long:
        raise ValueError("full-support rollout q9 must be int64 [128]")
    return value.detach().clone()


class SonicTaskSpacePpoFullSupportRunner(CausalHistoryMjlabOnPolicyRunner):
    """Exact update0 overlay with split collect/guard/update lifecycle."""

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
        *,
        source_actor_checkpoint_path: str | Path,
        overlay_encoder_path: str | Path,
        overlay_decoder_path: str | Path,
        base_task_space_contract: Mapping[str, Any],
        full_support_contract: Mapping[str, Any],
        run_materials_sha256: str,
        **kwargs: Any,
    ) -> None:
        validate_task_space_ppo_contract(base_task_space_contract)
        validate_full_support_contract(full_support_contract)
        raw_env = getattr(env, "unwrapped", None)
        if (
            raw_env is None
            or getattr(env, "clip_actions", "missing") is not None
            or int(getattr(raw_env, "num_envs", -1)) != NUM_ENVS
        ):
            raise ValueError("full-support wrapped environment boundary mismatch")
        audit_task_space_ppo_env_cfg(raw_env.cfg, expected_num_envs=NUM_ENVS)
        algorithm_cfg = _mapping(train_cfg.get("algorithm"), "full-support PPO algorithm config")
        actor_cfg = _mapping(train_cfg.get("actor"), "full-support PPO actor config")
        critic_cfg = _mapping(train_cfg.get("critic"), "full-support PPO critic config")
        obs_groups = _mapping(train_cfg.get("obs_groups"), "full-support observation groups")
        distribution_cfg = _mapping(actor_cfg.get("distribution_cfg"), "full-support actor distribution")
        if (
            int(getattr(env, "num_envs", -1)) != NUM_ENVS
            or train_cfg.get("seed") != FIXED_SEED
            or train_cfg.get("num_steps_per_env") != NUM_STEPS_PER_ENV
            or train_cfg.get("max_iterations") != MAXIMUM_UPDATES
            or train_cfg.get("save_interval") != 1
            or train_cfg.get("clip_actions") is not None
            or train_cfg.get("upload_model") is not False
            or train_cfg.get("class_name") != "OnPolicyRunner"
            or train_cfg.get("logger") != "tensorboard"
            or tuple(obs_groups.get("actor", ())) != ("tokenizer", "policy")
            or tuple(obs_groups.get("critic", ())) != ("critic",)
            or actor_cfg.get("class_name") != "gear_sonic.trl.mjlab.true23_actor:True23SonicActorModel"
            or actor_cfg.get("obs_normalization") is not False
            or tuple(actor_cfg.get("hidden_dims", ())) != ()
            or actor_cfg.get("activation") != "silu"
            or actor_cfg.get("tokenizer_obs_group") != "tokenizer"
            or actor_cfg.get("proprioception_obs_group") != "policy"
            or distribution_cfg != {"class_name": "GaussianDistribution", "init_std": 0.1, "std_type": "scalar"}
            or tuple(critic_cfg.get("hidden_dims", ())) != (512, 256, 128)
            or critic_cfg.get("class_name") != "MLPModel"
            or critic_cfg.get("activation") != "elu"
            or critic_cfg.get("obs_normalization") is not True
            or critic_cfg.get("distribution_cfg") is not None
            or algorithm_cfg.get("class_name") != "PPO"
            or algorithm_cfg.get("num_learning_epochs") != PPO_LEARNING_EPOCHS
            or algorithm_cfg.get("num_mini_batches") != PPO_MINI_BATCHES
            or algorithm_cfg.get("learning_rate") != FIXED_LEARNING_RATE
            or algorithm_cfg.get("schedule") != "fixed"
            or algorithm_cfg.get("desired_kl") is not None
            or algorithm_cfg.get("clip_param") != 0.1
            or algorithm_cfg.get("entropy_coef") != 0.0
            or algorithm_cfg.get("gamma") != 0.99
            or algorithm_cfg.get("lam") != 0.95
            or algorithm_cfg.get("max_grad_norm") != 0.25
            or algorithm_cfg.get("value_loss_coef") != 1.0
            or algorithm_cfg.get("use_clipped_value_loss") is not True
            or algorithm_cfg.get("normalize_advantage_per_mini_batch") is not False
            or algorithm_cfg.get("optimizer") != "adam"
            or algorithm_cfg.get("share_cnn_encoders") is not False
            or algorithm_cfg.get("rnd_cfg") is not None
            or algorithm_cfg.get("symmetry_cfg") is not None
        ):
            raise ValueError("full-support executed PPO configuration drift")
        if log_dir is None:
            raise ValueError("full-support runner requires log directory")
        self._base_task_space_contract = copy.deepcopy(dict(base_task_space_contract))
        self._full_support_contract = copy.deepcopy(dict(full_support_contract))
        self._run_materials_sha256 = _require_sha256(run_materials_sha256, "full-support run materials SHA256")
        self._source_actor_checkpoint_path = Path(source_actor_checkpoint_path).expanduser().resolve()
        self._overlay_encoder_path = Path(overlay_encoder_path).expanduser().resolve()
        self._overlay_decoder_path = Path(overlay_decoder_path).expanduser().resolve()
        self._runtime_dir = Path(log_dir).expanduser().resolve()
        super().__init__(
            env,
            train_cfg,
            log_dir,
            device,
            registry_name,
            **kwargs,
        )

        state, overlay = load_overlay_policy_state(
            source_checkpoint_path=self._source_actor_checkpoint_path,
            encoder_path=self._overlay_encoder_path,
            decoder_path=self._overlay_decoder_path,
            contract=self._base_task_space_contract,
        )
        self._policy_state_adapter.load_state_dict(state, strict=True)
        actor = self.alg.get_policy()
        named_actor = dict(actor.named_parameters())
        if set(TRAINABLE_ACTOR_PARAMETERS) - set(named_actor) or STD_PARAMETER not in named_actor:
            raise ValueError("full-support actor parameter namespace mismatch")
        for name, parameter in named_actor.items():
            parameter.requires_grad_(name in TRAINABLE_ACTOR_PARAMETERS)
        trainable_actor = [named_actor[name] for name in TRAINABLE_ACTOR_PARAMETERS]
        if sum(parameter.numel() for parameter in trainable_actor) != EXPECTED_TRAINABLE_PARAMETER_COUNT:
            raise ValueError("full-support trainable actor parameter count mismatch")
        if not torch.all(named_actor[STD_PARAMETER].detach() == EXPLORATION_STD):
            raise ValueError("full-support exploration std mismatch")
        self._frozen_actor_initial = {
            name: parameter.detach().clone()
            for name, parameter in named_actor.items()
            if name not in TRAINABLE_ACTOR_PARAMETERS
        }
        policy_state = self._policy_state_adapter.state_dict()
        self._initial_overlay_policy_state_sha256 = inspect_true23_policy_state(
            {"policy_state_dict": policy_state}, reference_profile=REFERENCE_PROFILE
        )
        frozen_export, trainable_export = _policy_state_subsets(policy_state)
        if (
            self._initial_overlay_policy_state_sha256 != INITIAL_OVERLAY_POLICY_STATE_SHA256
            or self._initial_overlay_policy_state_sha256 != overlay["overlay_policy_state_sha256"]
            or _state_sha256(frozen_export) != INITIAL_FROZEN_ACTOR_STATE_SHA256
            or _state_sha256(trainable_export) != INITIAL_TRAINABLE_ACTOR_STATE_SHA256
        ):
            raise RuntimeError("full-support update0 overlay identity drifted")

        trainable_critic = list(self.alg.critic.parameters())
        critic_state = self.alg.critic.state_dict()
        _validate_critic_state_schema(critic_state)
        if len(trainable_critic) != EXPECTED_CRITIC_PARAMETER_TENSOR_COUNT or any(
            not parameter.requires_grad for parameter in trainable_critic
        ):
            raise ValueError("full-support critic must be fresh and fully trainable")
        self._initial_critic_state_sha256 = _state_sha256(critic_state)
        if self._initial_critic_state_sha256 != INITIAL_CRITIC_STATE_SHA256:
            raise RuntimeError("full-support fresh critic differs from failed-parent update0")
        critic_normalizer, critic_mlp = _critic_state_subsets(critic_state)
        self._initial_critic_normalizer_state_sha256 = _state_sha256(critic_normalizer)
        self._initial_critic_mlp_state_sha256 = _state_sha256(critic_mlp)
        self.alg.optimizer = torch.optim.Adam([*trainable_actor, *trainable_critic], lr=FIXED_LEARNING_RATE)
        self.alg.learning_rate = FIXED_LEARNING_RATE
        if self.alg.optimizer.state:
            raise RuntimeError("fresh full-support Adam unexpectedly has state")
        self._optimizer_parameter_tensor_count = len([*trainable_actor, *trainable_critic])
        if self._optimizer_parameter_tensor_count != EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT:
            raise RuntimeError("full-support optimizer parameter count drifted")
        self._optimizer_parameter_ids = {id(parameter) for parameter in (*trainable_actor, *trainable_critic)}
        self._frozen_parameter_ids = {
            id(parameter) for name, parameter in named_actor.items() if name not in TRAINABLE_ACTOR_PARAMETERS
        }
        self._optimizer_step_count = 0
        self._executed_training_transitions = 0
        self._rollout_evidence_sha256: str | None = None
        self._rollout_evidence: dict[str, Any] | None = None
        self._phase = "fresh"
        self._logger_started = False
        if self.completed_update_count != 0 or self.current_learning_iteration != 0:
            raise RuntimeError("full-support counters must start at zero")
        self._assert_execution_boundary()
        _write_json_new(
            self._runtime_dir / RUNTIME_FILENAME,
            {
                "schema_version": 1,
                "kind": "g1_true23_sonic_task_space_ppo_full_support_runtime_v1",
                "contract_sha256": CONTRACT_SHA256,
                "base_contract_sha256": BASE_CONTRACT_SHA256,
                "trace_sha256": TRACE_SHA256,
                "run_materials_sha256": self._run_materials_sha256,
                "source_actor": overlay,
                "initial_overlay_policy_state_sha256": self._initial_overlay_policy_state_sha256,
                "initial_fresh_critic_state_sha256": self._initial_critic_state_sha256,
                "initial_critic_mlp_state_sha256": self._initial_critic_mlp_state_sha256,
                "initial_critic_normalizer_state_sha256": self._initial_critic_normalizer_state_sha256,
                "optimizer_state_entry_count": 0,
                "optimizer_step_count": 0,
                "executed_training_transitions": 0,
                "failed_model5_loaded": False,
                "failed_model5_resumed": False,
                "teacher_labels_used": False,
                "support_qualified": False,
                "promotion_eligible": False,
                "hardware_authorized": False,
                "deployment_ready": False,
            },
        )

    def _storage_step(self) -> int:
        value = getattr(getattr(self.alg, "storage", None), "step", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("full-support rollout storage step drifted")
        return value

    def _assert_execution_boundary(self) -> None:
        actor = self.alg.get_policy()
        named_actor = dict(actor.named_parameters())
        for name, initial in self._frozen_actor_initial.items():
            if name not in named_actor or not torch.equal(named_actor[name].detach(), initial):
                raise RuntimeError(f"full-support frozen actor changed: {name}")
        if not torch.all(named_actor[STD_PARAMETER].detach() == EXPLORATION_STD):
            raise RuntimeError("full-support fixed std changed")
        optimizer_ids = {
            id(parameter) for group in self.alg.optimizer.param_groups for parameter in group["params"]
        }
        if (
            optimizer_ids != self._optimizer_parameter_ids
            or bool(optimizer_ids & self._frozen_parameter_ids)
            or type(self.alg.optimizer) is not torch.optim.Adam
            or float(self.alg.learning_rate) != FIXED_LEARNING_RATE
            or any(float(group["lr"]) != FIXED_LEARNING_RATE for group in self.alg.optimizer.param_groups)
        ):
            raise RuntimeError("full-support optimizer boundary changed")
        update = self._require_counter_coherence()
        expected = {
            "fresh": (0, 0, 0, 0),
            "rollout_ready": (0, 0, TRAINING_TRANSITIONS, NUM_STEPS_PER_ENV),
            "rollout_failed": (0, 0, TRAINING_TRANSITIONS, NUM_STEPS_PER_ENV),
            "updated": (1, OPTIMIZER_STEPS, TRAINING_TRANSITIONS, 0),
        }
        if self._phase not in expected:
            raise RuntimeError("full-support runner phase is invalid")
        expected_update, expected_optimizer, expected_transitions, expected_storage = expected[self._phase]
        if (
            update != expected_update
            or self._optimizer_step_count != expected_optimizer
            or self._executed_training_transitions != expected_transitions
            or self._storage_step() != expected_storage
        ):
            raise RuntimeError("full-support phase counters diverged")
        if self._phase in {"fresh", "rollout_ready", "rollout_failed"}:
            current_policy = inspect_true23_policy_state(
                {"policy_state_dict": self._policy_state_adapter.state_dict()},
                reference_profile=REFERENCE_PROFILE,
            )
            _, current_mlp = _critic_state_subsets(self.alg.critic.state_dict())
            if (
                current_policy != INITIAL_OVERLAY_POLICY_STATE_SHA256
                or _state_sha256(current_mlp) != self._initial_critic_mlp_state_sha256
                or self.alg.optimizer.state
                or self._optimizer_step_count != 0
            ):
                raise RuntimeError("full-support pre-Adam state mutated")
        if self._phase == "updated":
            current_policy = inspect_true23_policy_state(
                {"policy_state_dict": self._policy_state_adapter.state_dict()},
                reference_profile=REFERENCE_PROFILE,
            )
            _, current_trainable = _policy_state_subsets(self._policy_state_adapter.state_dict())
            if (
                current_policy == INITIAL_OVERLAY_POLICY_STATE_SHA256
                or _state_sha256(current_trainable) == INITIAL_TRAINABLE_ACTOR_STATE_SHA256
                or len(self.alg.optimizer.state) != EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT
            ):
                raise RuntimeError("full-support update1 did not adapt actor exactly once")

    def _numbered_checkpoint_path(self, update_count: int) -> Path:
        if (
            isinstance(update_count, bool)
            or not isinstance(update_count, int)
            or update_count not in EVALUATION_UPDATES
        ):
            raise ValueError("full-support checkpoint update must be 0 or 1")
        return self.checkpoint_dir / f"sonic_task_space_full_support_model_{update_count}.pt"

    def _checkpoint_body(self) -> dict[str, Any]:
        self._assert_execution_boundary()
        update = self._require_counter_coherence()
        if self._phase not in {"fresh", "updated"}:
            raise RuntimeError("full-support checkpoint forbidden between rollout and update")
        policy = self._policy_state_adapter.state_dict()
        critic = {
            name: value.detach().cpu().contiguous().clone() for name, value in self.alg.critic.state_dict().items()
        }
        frozen, trainable = _policy_state_subsets(policy)
        return {
            "g1_true23_sonic_task_space_ppo_full_support_checkpoint": _checkpoint_header(),
            "policy_state_dict": policy,
            "critic_state_dict": critic,
            "optimizer_state_dict": copy.deepcopy(self.alg.optimizer.state_dict()),
            "update_count": update,
            "trainer_state": self._trainer_state(),
            "initial_critic_state_sha256": self._initial_critic_state_sha256,
            "optimizer_parameter_tensor_count": self._optimizer_parameter_tensor_count,
            "optimizer_step_count": self._optimizer_step_count,
            "executed_training_transitions": self._executed_training_transitions,
            "storage_step": self._storage_step(),
            "policy_state_sha256": inspect_true23_policy_state(
                {"policy_state_dict": policy}, reference_profile=REFERENCE_PROFILE
            ),
            "critic_state_sha256": _state_sha256(critic),
            "frozen_actor_state_sha256": _state_sha256(frozen),
            "trainable_actor_state_sha256": _state_sha256(trainable),
            "initial_overlay_policy_state_sha256": self._initial_overlay_policy_state_sha256,
            "contract_sha256": CONTRACT_SHA256,
            "trace_sha256": TRACE_SHA256,
            "run_materials_sha256": self._run_materials_sha256,
            "rollout_evidence_sha256": self._rollout_evidence_sha256 if update == 1 else None,
            "source_actor": {
                "checkpoint_sha256": _SOURCE_CHECKPOINT_SHA256,
                "policy_state_sha256": _SOURCE_POLICY_SHA256,
                "lineage_sha256": _SOURCE_LINEAGE_SHA256,
                "checkpoint_update_count": 250,
                "v2_decoder_sha256": _V2_DECODER_SHA256,
                "critic_reused": False,
                "optimizer_reused": False,
                "counters_reused": False,
                "failed_model5_loaded": False,
                "failed_model5_resumed": False,
            },
            "training_boundary": {
                "trainable_actor_parameters": list(TRAINABLE_ACTOR_PARAMETERS),
                "trainable_actor_parameter_count": EXPECTED_TRAINABLE_PARAMETER_COUNT,
                "exploration_std": EXPLORATION_STD,
                "exploration_std_trainable": False,
                "teacher_labels_used": False,
                "support_qualified": False,
                "promotion_eligible": False,
                "hardware_authorized": False,
                "deployment_ready": False,
            },
        }

    def save(self, path: str, infos: dict | None = None) -> None:
        if infos is not None:
            raise ValueError("full-support checkpoint forbids stock infos")
        output = Path(path).expanduser().resolve()
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"refusing to overwrite full-support checkpoint: {output}")
        if output.parent != self.checkpoint_dir:
            raise ValueError("full-support checkpoint must remain in exact checkpoint directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        body = self._checkpoint_body()
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            torch.save(body, temporary)
            loaded = torch.load(temporary, map_location="cpu", weights_only=True)
            validate_full_support_checkpoint(loaded, expected_run_materials_sha256=self._run_materials_sha256)
            os.link(temporary, output)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._last_checkpoint_path = output
        self._last_checkpoint_update_count = body["update_count"]

    def load(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("full-support temporal-support ablation does not permit resume")

    def save_current_checkpoint(self) -> Path:
        self._save_numbered_checkpoint()
        if self._last_checkpoint_path is None:
            raise RuntimeError("full-support checkpoint publication failed")
        return self._last_checkpoint_path

    def _save_numbered_checkpoint(self) -> None:
        self._assert_execution_boundary()
        if self._phase not in {"fresh", "updated"}:
            return
        super()._save_numbered_checkpoint()

    def collect_full_support_rollout(self) -> Mapping[str, Any]:
        """Collect 20,480 transitions and publish gate evidence with zero Adam steps."""

        if self._phase != "fresh":
            raise RuntimeError("full-support rollout may be collected exactly once")
        self._assert_execution_boundary()
        raw_env = getattr(self.env, "unwrapped", None)
        if raw_env is None or int(getattr(raw_env, "num_envs", -1)) != NUM_ENVS:
            raise ValueError("full-support raw environment mismatch")
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        self._logger_started = True
        rng_pre_rollout_pre_probe = capture_rollout_rng_state_hashes()
        policy_before = _clone_tensor_state(self._policy_state_adapter.state_dict())
        frozen_before, trainable_before = _policy_state_subsets(policy_before)
        std_before = {"std": policy_before["std"]}
        critic_before = _clone_tensor_state(self.alg.critic.state_dict())
        normalizer_before, mlp_before = _critic_state_subsets(critic_before)
        normalizer_count_before = int(normalizer_before["obs_normalizer.count"].detach().cpu().item())
        if normalizer_count_before != 0:
            raise RuntimeError("full-support critic normalizer was not fresh")
        first_episode_active = torch.ones(NUM_ENVS, dtype=torch.bool, device=self.env.device)
        q9_snapshots: list[torch.Tensor] = []
        active_snapshots: list[torch.Tensor] = []
        done_snapshots: list[torch.Tensor] = []
        recorder = _FullSupportRewardEvidenceRecorder(raw_env)
        rng_pre_rollout = capture_rollout_rng_state_hashes()
        if rng_pre_rollout != rng_pre_rollout_pre_probe:
            recorder.restore()
            self.logger.stop_logging_writer()
            self._logger_started = False
            raise RuntimeError("full-support pre-rollout state probes consumed RNG")
        collect_start = time.time()
        try:
            with torch.inference_mode():
                for _step in range(NUM_STEPS_PER_ENV):
                    q9 = _vector_q9(raw_env)
                    q9_snapshots.append(q9.detach().clone())
                    active_snapshots.append(first_episode_active.detach().clone())
                    actions = self.alg.act(obs)
                    recorder.arm(q9, first_episode_active, actions)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    recorder.finish(dones)
                    if self.cfg.get("check_for_nan", True):
                        from rsl_rl.utils import check_nan

                        check_nan(obs, rewards, dones)
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    done_snapshots.append(dones.detach().clone())
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards if bool(getattr(self.alg, "rnd", False)) else None
                    )
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                    extras["log"] = {}
                    first_episode_active = first_episode_active & ~dones.to(
                        dtype=torch.bool, device=first_episode_active.device
                    )
                    self._executed_training_transitions += NUM_ENVS
                self.alg.compute_returns(obs)
        except BaseException:
            self._training_state_poisoned = True
            self._phase = "rollout_failed"
            if self._logger_started:
                self.logger.stop_logging_writer()
                self._logger_started = False
            raise
        finally:
            recorder.restore()
        collect_time_s = time.time() - collect_start
        rng_post_rollout_pre_probe = capture_rollout_rng_state_hashes()

        reward_evidence = recorder.materialize()
        q9_discontinuity_count = 0
        first_histogram: list[dict[str, int]] = []
        all_histogram: dict[int, int] = {}
        first_termination_histogram: dict[int, int] = {}
        for step, (q9_gpu, active_gpu, done_gpu) in enumerate(
            zip(q9_snapshots, active_snapshots, done_snapshots, strict=True)
        ):
            q9 = q9_gpu.detach().cpu()
            active = active_gpu.detach().cpu()
            done = done_gpu.detach().to(dtype=torch.bool).cpu()
            expected_q9 = EXPECTED_Q9_FIRST + step
            q9_discontinuity_count += int(torch.count_nonzero(active & (q9 != expected_q9)).item())
            first_count = int(torch.count_nonzero(active & (q9 == expected_q9)).item())
            first_histogram.append({"q9": expected_q9, "count": first_count})
            for value, count in zip(*torch.unique(q9, return_counts=True), strict=True):
                key = int(value.item())
                all_histogram[key] = all_histogram.get(key, 0) + int(count.item())
            terminal_q9 = q9[active & done]
            for value, count in zip(*torch.unique(terminal_q9, return_counts=True), strict=True):
                key = int(value.item())
                first_termination_histogram[key] = first_termination_histogram.get(key, 0) + int(count.item())

        policy_after = _clone_tensor_state(self._policy_state_adapter.state_dict())
        frozen_after, trainable_after = _policy_state_subsets(policy_after)
        std_after = {"std": policy_after["std"]}
        critic_after = _clone_tensor_state(self.alg.critic.state_dict())
        normalizer_after, mlp_after = _critic_state_subsets(critic_after)
        normalizer_count_after = int(normalizer_after["obs_normalizer.count"].detach().cpu().item())
        storage_step = self._storage_step()
        returns = self.alg.storage.returns
        advantages = self.alg.storage.advantages
        if type(returns) is not torch.Tensor or type(advantages) is not torch.Tensor:
            raise RuntimeError("full-support storage return tensors missing")

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("full-support memory gate requires one visible CUDA device")
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        free_cuda_bytes, total_cuda_bytes = torch.cuda.mem_get_info(self.device)
        device_index = torch.cuda.current_device()
        memory = {
            "device_name": torch.cuda.get_device_name(device_index),
            "device_total_memory_bytes": int(total_cuda_bytes),
            "free_cuda_bytes_before_update": int(free_cuda_bytes),
            "minimum_required_free_cuda_bytes": MINIMUM_FREE_CUDA_BYTES,
            "allocated_cuda_bytes": int(torch.cuda.memory_allocated(device_index)),
            "reserved_cuda_bytes": int(torch.cuda.memory_reserved(device_index)),
            "mini_batch_size": MINI_BATCH_SIZE,
            "cuda_cache_released": True,
            "out_of_memory_is_fatal": True,
        }
        rng_post_probe = capture_rollout_rng_state_hashes()
        first_transition_count = sum(item["count"] for item in first_histogram)
        pre_adam_state = {
            "storage_step": storage_step,
            "executed_training_transitions": self._executed_training_transitions,
            "optimizer_step_count": self._optimizer_step_count,
            "optimizer_state_entry_count": len(self.alg.optimizer.state),
            "completed_update_count": self.completed_update_count,
            "current_learning_iteration": self.current_learning_iteration,
            "policy_state_before_sha256": inspect_true23_policy_state(
                {"policy_state_dict": policy_before}, reference_profile=REFERENCE_PROFILE
            ),
            "policy_state_after_sha256": inspect_true23_policy_state(
                {"policy_state_dict": policy_after}, reference_profile=REFERENCE_PROFILE
            ),
            "policy_state_unchanged": all(
                torch.equal(policy_before[name], policy_after[name]) for name in policy_before
            ),
            "frozen_actor_before_sha256": _state_sha256(frozen_before),
            "frozen_actor_after_sha256": _state_sha256(frozen_after),
            "frozen_actor_state_unchanged": all(
                torch.equal(frozen_before[name], frozen_after[name]) for name in frozen_before
            ),
            "trainable_actor_before_sha256": _state_sha256(trainable_before),
            "trainable_actor_after_sha256": _state_sha256(trainable_after),
            "trainable_actor_state_unchanged": all(
                torch.equal(trainable_before[name], trainable_after[name]) for name in trainable_before
            ),
            "std_before_sha256": _state_sha256(std_before),
            "std_after_sha256": _state_sha256(std_after),
            "critic_mlp_before_sha256": _state_sha256(mlp_before),
            "critic_mlp_after_sha256": _state_sha256(mlp_after),
            "critic_mlp_state_unchanged": all(
                torch.equal(mlp_before[name], mlp_after[name]) for name in mlp_before
            ),
            "critic_normalizer_before_sha256": _state_sha256(normalizer_before),
            "critic_normalizer_after_sha256": _state_sha256(normalizer_after),
            "critic_observation_normalizer_sample_count_before": normalizer_count_before,
            "critic_observation_normalizer_sample_count": normalizer_count_after,
            "returns_finite": bool(torch.isfinite(returns).all()),
            "advantages_finite": bool(torch.isfinite(advantages).all()),
        }
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": "g1_true23_sonic_task_space_full_support_rollout_evidence_v1",
            "contract_sha256": CONTRACT_SHA256,
            "trace_sha256": TRACE_SHA256,
            "run_materials_sha256": self._run_materials_sha256,
            "histogram_scope": "first_episode_per_env_only",
            "first_episode_q9_histogram": first_histogram,
            "all_rollout_q9_histogram": [
                {"q9": q9, "count": count} for q9, count in sorted(all_histogram.items())
            ],
            "first_episode_termination_q9_histogram": [
                {"q9": q9, "count": count} for q9, count in sorted(first_termination_histogram.items())
            ],
            "first_episode_transition_count": first_transition_count,
            "autoreset_transition_count": TRAINING_TRANSITIONS - first_transition_count,
            "total_inserted_transitions": self._executed_training_transitions,
            "first_episode_reward_and_termination_evidence": {
                key: value
                for key, value in reward_evidence.items()
                if key
                not in {
                    "nonfinite_count",
                    "action_semantics_mismatch_count",
                    "raw_clip_required_count",
                }
            },
            "pre_adam_state": pre_adam_state,
            "rng_state_hashes": {
                "pre_rollout_pre_probe": rng_pre_rollout_pre_probe,
                "pre_rollout": rng_pre_rollout,
                "post_rollout_pre_probe": rng_post_rollout_pre_probe,
                "post_probe": rng_post_probe,
            },
            "pre_optimizer_memory": memory,
            "collect_time_s": collect_time_s,
            "nonfinite_count": reward_evidence["nonfinite_count"],
            "q9_discontinuity_count": q9_discontinuity_count,
            "action_semantics_mismatch_count": reward_evidence["action_semantics_mismatch_count"],
            "raw_clip_required_count": reward_evidence["raw_clip_required_count"],
            "optimizer_steps_at_publication": self._optimizer_step_count,
            "training_updates_at_publication": self.completed_update_count,
            "failed_model5_loaded": False,
            "failed_model5_resumed": False,
            "teacher_labels_used": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
        record["coverage_assessment"] = assess_full_support_rollout_evidence(record)
        validate_full_support_rollout_evidence(record)
        output = self._runtime_dir / ROLLOUT_EVIDENCE_FILENAME
        _write_json_new(output, record)
        self._rollout_evidence_sha256 = sha256_file(output)
        self._rollout_evidence = copy.deepcopy(record)
        self._phase = "rollout_ready" if record["coverage_assessment"]["gate_passed"] is True else "rollout_failed"
        self._assert_execution_boundary()
        if self._phase == "rollout_failed" and self._logger_started:
            self.logger.stop_logging_writer()
            self._logger_started = False
        return copy.deepcopy(record)

    def _revalidate_pre_adam_gate(self) -> None:
        if self._phase != "rollout_ready" or self._rollout_evidence is None:
            raise RuntimeError("full-support pre-Adam gate was not admitted")
        path = self._runtime_dir / ROLLOUT_EVIDENCE_FILENAME
        if (
            self._rollout_evidence_sha256 is None
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != self._rollout_evidence_sha256
        ):
            raise RuntimeError("full-support rollout evidence changed before Adam")
        disk = _strict_json(path, "full-support rollout evidence")
        validate_full_support_rollout_evidence(disk)
        if dict(disk) != self._rollout_evidence or disk["coverage_assessment"]["gate_passed"] is not True:
            raise RuntimeError("full-support rollout admission changed before Adam")
        self._assert_execution_boundary()
        expected_state = _mapping(disk.get("pre_adam_state"), "published pre-Adam state")
        live_policy = _clone_tensor_state(self._policy_state_adapter.state_dict())
        live_frozen, live_trainable = _policy_state_subsets(live_policy)
        live_std = {"std": live_policy["std"]}
        live_normalizer, live_mlp = _critic_state_subsets(_clone_tensor_state(self.alg.critic.state_dict()))
        live_normalizer_count = int(live_normalizer["obs_normalizer.count"].detach().cpu().item())
        live_hashes = {
            "policy_state_after_sha256": inspect_true23_policy_state(
                {"policy_state_dict": live_policy}, reference_profile=REFERENCE_PROFILE
            ),
            "frozen_actor_after_sha256": _state_sha256(live_frozen),
            "trainable_actor_after_sha256": _state_sha256(live_trainable),
            "std_after_sha256": _state_sha256(live_std),
            "critic_mlp_after_sha256": _state_sha256(live_mlp),
            "critic_normalizer_after_sha256": _state_sha256(live_normalizer),
        }
        if (
            any(expected_state.get(name) != value for name, value in live_hashes.items())
            or live_normalizer_count != TRAINING_TRANSITIONS
            or expected_state.get("critic_observation_normalizer_sample_count") != live_normalizer_count
            or self._storage_step() != NUM_STEPS_PER_ENV
            or not bool(torch.isfinite(self.alg.storage.returns).all())
            or not bool(torch.isfinite(self.alg.storage.advantages).all())
        ):
            raise RuntimeError("full-support live pre-Adam tensor state changed after publication")
        if capture_rollout_rng_state_hashes() != disk["rng_state_hashes"]["post_probe"]:
            raise RuntimeError("full-support RNG state changed between evidence and Adam")
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        free_cuda_bytes, total_cuda_bytes = torch.cuda.mem_get_info(self.device)
        if (
            int(total_cuda_bytes) != EXPECTED_CUDA_TOTAL_MEMORY_BYTES
            or int(free_cuda_bytes) < MINIMUM_FREE_CUDA_BYTES
        ):
            raise RuntimeError("full-support pre-Adam CUDA memory gate failed")

    def optimize_collected_rollout(self) -> Mapping[str, float]:
        """Run one guarded RSL PPO update and exactly eight Adam steps."""

        if self._phase != "rollout_ready":
            raise RuntimeError("full-support optimizer requires admitted rollout")
        original_update = self.alg.update
        original_optimizer_step = self.alg.optimizer.step

        def bounded_optimizer_step(*args: Any, **kwargs: Any) -> Any:
            if self._optimizer_step_count >= OPTIMIZER_STEPS:
                raise RuntimeError("full-support optimizer-step cap reached")
            result = original_optimizer_step(*args, **kwargs)
            self._optimizer_step_count += 1
            return result

        def guarded_update() -> Mapping[str, float]:
            rng_before_gate = capture_rollout_rng_state_hashes()
            self._revalidate_pre_adam_gate()
            if capture_rollout_rng_state_hashes() != rng_before_gate:
                raise RuntimeError("full-support immediate pre-Adam gate consumed RNG")
            result = original_update()
            if self._optimizer_step_count != OPTIMIZER_STEPS:
                raise RuntimeError("full-support PPO optimizer cadence drifted")
            return result

        self.alg.optimizer.step = bounded_optimizer_step
        self.alg.update = guarded_update
        learn_start = time.time()
        try:
            loss_dict = self.alg.update()
        except BaseException:
            self._training_state_poisoned = True
            self._phase = "poisoned"
            if self._logger_started:
                self.logger.stop_logging_writer()
                self._logger_started = False
            raise
        finally:
            self.alg.optimizer.step = original_optimizer_step
            self.alg.update = original_update
        if (
            not isinstance(loss_dict, Mapping)
            or set(loss_dict) != {"value", "surrogate", "entropy"}
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in loss_dict.values()
            )
        ):
            self._training_state_poisoned = True
            self._phase = "poisoned"
            raise RuntimeError("full-support PPO loss dictionary mismatch")
        if self._storage_step() != 0:
            self._training_state_poisoned = True
            self._phase = "poisoned"
            raise RuntimeError("full-support PPO storage did not clear")
        self.completed_update_count = 1
        self.current_learning_iteration = 1
        self._phase = "updated"
        self._assert_execution_boundary()
        if self._logger_started:
            self.logger.log(
                it=0,
                start_it=0,
                total_it=1,
                collect_time=0.0,
                learn_time=time.time() - learn_start,
                loss_dict=dict(loss_dict),
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )
            self.logger.stop_logging_writer()
            self._logger_started = False
        return {name: float(value) for name, value in loss_dict.items()}

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        del num_learning_iterations, init_at_random_ep_len
        raise RuntimeError(
            "full-support runner forbids coupled learn(); use "
            "collect_full_support_rollout() then optimize_collected_rollout()"
        )


def evaluate_full_support_policy(
    *,
    policy: Any,
    wrapped_env: Any,
    update_count: int,
    evaluation_seed: int,
) -> dict[str, Any]:
    """One deterministic mean-action episode at update0 or update1."""

    if update_count not in EVALUATION_UPDATES or evaluation_seed != FIXED_SEED:
        raise ValueError("full-support evaluation identity mismatch")
    raw_env = getattr(wrapped_env, "unwrapped", None)
    if (
        raw_env is None
        or int(getattr(raw_env, "num_envs", -1)) != 1
        or int(getattr(getattr(raw_env, "cfg", None), "seed", -1)) != evaluation_seed
        or getattr(wrapped_env, "clip_actions", "missing") is not None
        or int(getattr(wrapped_env, "max_episode_length", -1)) != 510
    ):
        raise ValueError("full-support evaluation environment mismatch")
    if (
        int(raw_env.common_step_counter) != 0
        or int(raw_env._sim_step_counter) != 0
        or int(raw_env.episode_length_buf[0].detach().cpu().item()) != 0
    ):
        raise ValueError("full-support evaluation environment was not freshly primed")
    command = raw_env.command_manager.get_term("motion")
    if _evaluation_q9(command) != EXPECTED_Q9_FIRST:
        raise ValueError("full-support evaluation did not start at q9=9")
    policy_state = policy.export_true23_policy_state()
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy_state}, reference_profile=REFERENCE_PROFILE
    )
    if update_count == 0 and policy_hash != INITIAL_OVERLAY_POLICY_STATE_SHA256:
        raise ValueError("full-support update0 evaluation policy mismatch")
    if update_count == 1 and policy_hash == INITIAL_OVERLAY_POLICY_STATE_SHA256:
        raise ValueError("full-support update1 evaluation actor did not change")
    observations = wrapped_env.get_observations()
    was_training = bool(policy.training)
    policy.eval()
    recorder = _TaskSpaceEvaluationTerminalRecorder(raw_env)
    completed = 0
    episode_return = 0.0
    terminal_q9: int | None = None
    termination_names: list[str] = []
    nonfinite_count = 0
    raw_clip_required_count = 0
    action_semantics_mismatch_count = 0
    q9_discontinuity_count = 0
    try:
        with torch.inference_mode():
            for transition in range(510):
                q9 = _evaluation_q9(command)
                if q9 != EXPECTED_Q9_FIRST + transition:
                    q9_discontinuity_count += 1
                    break
                if not _finite_observations(observations):
                    nonfinite_count += 1
                    break
                raw_action = policy(observations, stochastic_output=False)
                if (
                    type(raw_action) is not torch.Tensor
                    or raw_action.shape != (1, ACTION_DIM)
                    or not bool(torch.isfinite(raw_action).all())
                ):
                    nonfinite_count += 1
                    break
                recorder.arm()
                observations, rewards, dones, _extras = wrapped_env.step(raw_action.to(wrapped_env.device))
                if (
                    type(rewards) is not torch.Tensor
                    or rewards.shape != (1,)
                    or not bool(torch.isfinite(rewards).all())
                    or type(dones) is not torch.Tensor
                    or dones.shape != (1,)
                ):
                    nonfinite_count += 1
                    raise RuntimeError("full-support evaluation step tensors drifted")
                done = bool(int(dones[0].detach().cpu().item()))
                terminal = recorder.finish(done=done)
                chain = terminal["action_chain"] if terminal is not None else capture_student_action_chain(raw_env)
                mismatch, clip_count = _action_chain_mismatch_count(raw_action, chain)
                action_semantics_mismatch_count += mismatch
                raw_clip_required_count += clip_count
                episode_return += float(rewards[0].detach().cpu().item())
                completed += 1
                if done:
                    terminal_q9 = q9
                    termination_names = list(terminal["termination_names"])
                    if terminal["episode_length_pre_reset"] != completed:
                        raise RuntimeError("full-support evaluation terminal length drifted")
                    break
    finally:
        recorder.restore()
        policy.train(was_training)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "g1_true23_sonic_task_space_ppo_full_support_evaluation_v1",
        "update_count": update_count,
        "evaluation_seed": evaluation_seed,
        "controller": "deterministic_actor_mean",
        "policy_state_sha256": policy_hash,
        "completed_transitions": completed,
        "terminal_q9": terminal_q9,
        "termination_names": termination_names,
        "episode_return": episode_return,
        "nonfinite_count": nonfinite_count,
        "raw_clip_required_count": raw_clip_required_count,
        "action_semantics_mismatch_count": action_semantics_mismatch_count,
        "q9_discontinuity_count": q9_discontinuity_count,
        "historical_parent_episode_return_gate_applied": False,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def assess_full_support_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not evaluations or len(evaluations) > 2:
        raise ValueError("full-support evaluation sequence must contain one or two records")
    updates = [record.get("update_count") for record in evaluations]
    if updates != list(EVALUATION_UPDATES[: len(evaluations)]):
        raise ValueError("full-support evaluation order mismatch")
    allowed_terminations = {"time_out", "anchor_pos", "anchor_ori", "ee_body_pos"}
    baseline_return: float | None = None
    for record in evaluations:
        update = record["update_count"]
        completed = record.get("completed_transitions")
        terminal_q9 = record.get("terminal_q9")
        names = record.get("termination_names")
        episode_return = record.get("episode_return")
        if (
            record.get("schema_version") != SCHEMA_VERSION
            or record.get("kind") != "g1_true23_sonic_task_space_ppo_full_support_evaluation_v1"
            or record.get("evaluation_seed") != FIXED_SEED
            or record.get("controller") != "deterministic_actor_mean"
            or any(
                record.get(name) != 0
                for name in (
                    "nonfinite_count",
                    "raw_clip_required_count",
                    "action_semantics_mismatch_count",
                    "q9_discontinuity_count",
                )
            )
            or isinstance(completed, bool)
            or not isinstance(completed, int)
            or not 1 <= completed <= 510
            or terminal_q9 != EXPECTED_Q9_FIRST + completed - 1
            or not isinstance(names, list)
            or not names
            or len(names) != len(set(names))
            or any(type(name) is not str or name not in allowed_terminations for name in names)
            or (completed == 510 and names != ["time_out"])
            or (completed < 510 and "time_out" in names)
            or isinstance(episode_return, bool)
            or not isinstance(episode_return, (int, float))
            or not math.isfinite(float(episode_return))
        ):
            return {
                "passed": False,
                "stop": True,
                "stop_reason": f"update{update}_structural_or_safety_failure",
                "candidate_selected": False,
                "support_qualified": False,
                "promotion_eligible": False,
                "hardware_authorized": False,
                "deployment_ready": False,
            }
        if update == 0:
            baseline_return = float(episode_return)
            if (
                completed != 155
                or terminal_q9 != 163
                or names != ["ee_body_pos"]
                or record.get("policy_state_sha256") != INITIAL_OVERLAY_POLICY_STATE_SHA256
            ):
                return {
                    "passed": False,
                    "stop": True,
                    "stop_reason": "update0_structural_baseline_mismatch",
                    "candidate_selected": False,
                    "support_qualified": False,
                    "promotion_eligible": False,
                    "hardware_authorized": False,
                    "deployment_ready": False,
                }
        else:
            if baseline_return is None:
                raise RuntimeError("full-support baseline return missing")
            reward_floor = baseline_return - max(0.2 * abs(baseline_return), 50.0)
            if (
                completed < 156
                or terminal_q9 < 164
                or float(episode_return) < reward_floor
                or record.get("policy_state_sha256") == INITIAL_OVERLAY_POLICY_STATE_SHA256
            ):
                return {
                    "passed": False,
                    "stop": True,
                    "stop_reason": "update1_no_q164_or_reward_improvement",
                    "candidate_reward_floor": reward_floor,
                    "candidate_selected": False,
                    "support_qualified": False,
                    "promotion_eligible": False,
                    "hardware_authorized": False,
                    "deployment_ready": False,
                }
    complete = len(evaluations) == 2
    return {
        "passed": complete,
        "stop": False,
        "stop_reason": None,
        "baseline_passed": True,
        "experiment_complete": complete,
        "candidate_selected": complete,
        "candidate_reward_floor": (
            baseline_return - max(0.2 * abs(baseline_return), 50.0)
            if complete and baseline_return is not None
            else None
        ),
        "historical_parent_absolute_return_gate_applied": False,
        "paired_current_run_baseline_return_used_for_catastrophe_floor": complete,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def _full_support_schedule_report(
    runner: SonicTaskSpacePpoFullSupportRunner,
    evaluations: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Path],
    assessment: Mapping[str, Any],
    rollout: Mapping[str, Any] | None,
    loss: Mapping[str, float] | None,
) -> dict[str, Any]:
    candidate_path: str | None = None
    candidate_sha256: str | None = None
    if assessment.get("candidate_selected") is True:
        if len(checkpoints) != 2 or runner.completed_update_count != 1:
            raise RuntimeError("full-support candidate checkpoint sequence mismatch")
        candidate_path = str(checkpoints[-1])
        candidate_sha256 = sha256_file(checkpoints[-1])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "g1_true23_sonic_task_space_ppo_full_support_result_v1",
        "contract_sha256": CONTRACT_SHA256,
        "trace_sha256": TRACE_SHA256,
        "run_materials_sha256": runner._run_materials_sha256,
        "completed_update_count": runner.completed_update_count,
        "optimizer_step_count": runner._optimizer_step_count,
        "executed_training_transitions": runner._executed_training_transitions,
        "evaluations": [dict(record) for record in evaluations],
        "checkpoint_paths": [str(path) for path in checkpoints],
        "rollout_evidence_sha256": runner._rollout_evidence_sha256,
        "rollout_coverage_assessment": (
            copy.deepcopy(rollout.get("coverage_assessment")) if rollout is not None else None
        ),
        "update1_loss": dict(loss) if loss is not None else None,
        "assessment": dict(assessment),
        "candidate": (
            {
                "path": candidate_path,
                "sha256": candidate_sha256,
                "update_count": 1,
            }
            if candidate_path is not None
            else None
        ),
        "failed_model5_loaded": False,
        "failed_model5_resumed": False,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }


def execute_full_support_schedule(
    runner: SonicTaskSpacePpoFullSupportRunner,
    evaluator: Callable[[SonicTaskSpacePpoFullSupportRunner, int], Mapping[str, Any]],
    *,
    phase_boundary: Callable[[str], None] | None = None,
    evaluation_publisher: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not callable(evaluator):
        raise TypeError("full-support evaluator must be callable")
    boundary = phase_boundary or (lambda _phase: None)
    evaluations: list[dict[str, Any]] = []
    checkpoints: list[Path] = []

    def checkpoint_and_evaluate(update: int) -> dict[str, Any]:
        if runner.completed_update_count != update:
            raise RuntimeError("full-support schedule reached wrong update")
        checkpoint = runner.save_current_checkpoint()
        checkpoints.append(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        validated = validate_full_support_checkpoint(
            saved, expected_run_materials_sha256=runner._run_materials_sha256
        )
        if validated["update_count"] != update:
            raise RuntimeError("full-support published checkpoint update drifted")
        policy_before = {
            name: value.detach().clone() for name, value in runner._policy_state_adapter.state_dict().items()
        }
        critic_before = copy.deepcopy(runner.alg.critic.state_dict())
        optimizer_before = copy.deepcopy(runner.alg.optimizer.state_dict())
        counters_before = (
            runner.completed_update_count,
            runner.current_learning_iteration,
            runner._optimizer_step_count,
            runner._executed_training_transitions,
            runner._storage_step(),
            runner._phase,
            runner._rollout_evidence_sha256,
        )
        rng_before = capture_rollout_rng_state_hashes()
        boundary(f"before_evaluation_update_{update}")
        record = dict(evaluator(runner, update))
        boundary(f"after_evaluation_update_{update}")
        if capture_rollout_rng_state_hashes() != rng_before:
            raise RuntimeError("full-support deterministic evaluation consumed training RNG")
        if (
            record.get("update_count") != update
            or record.get("policy_state_sha256") != validated["policy_state_sha256"]
            or not _nested_state_equal(runner._policy_state_adapter.state_dict(), policy_before)
            or not _nested_state_equal(runner.alg.critic.state_dict(), critic_before)
            or not _nested_state_equal(runner.alg.optimizer.state_dict(), optimizer_before)
            or counters_before
            != (
                runner.completed_update_count,
                runner.current_learning_iteration,
                runner._optimizer_step_count,
                runner._executed_training_transitions,
                runner._storage_step(),
                runner._phase,
                runner._rollout_evidence_sha256,
            )
        ):
            raise RuntimeError("full-support evaluation mutated training state")
        runner._assert_execution_boundary()
        if evaluation_publisher is not None:
            evaluation_publisher(record)
        evaluations.append(record)
        return assess_full_support_evaluations(evaluations)

    assessment = checkpoint_and_evaluate(0)
    if assessment["stop"]:
        return _full_support_schedule_report(runner, evaluations, checkpoints, assessment, None, None)
    boundary("before_full_support_rollout")
    rollout = runner.collect_full_support_rollout()
    boundary("after_full_support_rollout")
    if rollout["coverage_assessment"]["gate_passed"] is not True:
        failed = {
            "passed": False,
            "stop": True,
            "stop_reason": "pre_adam_rollout_coverage_or_memory_gate_failed",
            "candidate_selected": False,
            "support_qualified": False,
            "promotion_eligible": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        }
        return _full_support_schedule_report(runner, evaluations, checkpoints, failed, rollout, None)
    boundary("immediately_before_guarded_update1")
    loss = runner.optimize_collected_rollout()
    boundary("after_guarded_update1")
    assessment = checkpoint_and_evaluate(1)
    return _full_support_schedule_report(runner, evaluations, checkpoints, assessment, rollout, loss)


__all__ = [
    "CONTRACT_SHA256",
    "FIXED_SEED",
    "MAXIMUM_UPDATES",
    "MINIMUM_SURVIVORS",
    "NUM_ENVS",
    "NUM_STEPS_PER_ENV",
    "OPTIMIZER_STEPS",
    "SonicTaskSpacePpoFullSupportRunner",
    "TRAINING_TRANSITIONS",
    "assess_full_support_evaluations",
    "assess_full_support_rollout_evidence",
    "audit_task_space_ppo_env_cfg",
    "evaluate_full_support_policy",
    "execute_full_support_schedule",
    "load_full_support_contract",
    "load_task_space_ppo_contract",
    "make_task_space_ppo_env_cfg",
    "validate_bound_full_support_evidence",
    "validate_full_support_checkpoint",
    "validate_full_support_contract",
    "validate_full_support_rollout_evidence",
]
