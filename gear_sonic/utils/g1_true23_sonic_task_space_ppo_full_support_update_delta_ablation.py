"""In-memory 2x2 attribution of the full-support PPO update delta.

This diagnostic crosses decoder module-14 and module-16 tensors from immutable
model 0 and model 1 checkpoints. It never creates an optimizer, trains, saves a
checkpoint, selects a candidate, or communicates with robot hardware.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

import numpy as np
import torch

from gear_sonic.utils import (
    g1_true23_sonic_task_space_ppo_full_support_checkpoint_trace as paired,
)
from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    inspect_true23_policy_state,
    sha256_file,
)
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    NATIVE_IL23_JOINT_NAMES,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import (
    SAFE_TARGET_RAW_ACTION_CLIP,
    SAFE_TARGET_TRANSFORM_KIND,
    safe_target_transform_numpy,
)

SCHEMA_VERSION = 1
ABLATION_KIND = "g1_true23_sonic_task_space_ppo_full_support_update_delta_ablation_v1"
PREFLIGHT_KIND = f"{ABLATION_KIND}_preflight"
FAILURE_KIND = f"{ABLATION_KIND}_failure"
CONTRACT_KIND = "g1_true23_sonic_task_space_ppo_full_support_update_delta_ablation_contract_v1"
CONTRACT_SHA256 = "542eb733f631349e0117f8a61d7ae6221ea13ab54fe8cb837a02745e106cec1e"
SAFE_TARGET_TRANSFORM_APPLICATION_COUNT = 1
SOURCE_PAIR_TRACE_SHA256 = "0fb71869eb1249958f7f4137df52b9b0bd92b4666d07c74a614ee11622a878d3"
SOURCE_PAIR_PROVENANCE_SHA256 = "3733a8f1248392f1e64ed7af01ef8a75c5f1e1409af343e30a6f9cee237ce849"
OUTPUT_FILENAME = "sonic_task_space_ppo_full_support_update_delta_ablation_v1.json"

CONTRACT_RELATIVE_PATH = (
    "gear_sonic/config/sim_validation/g1_true23_sonic_task_space_ppo_full_support_update_delta_ablation_v1.json"
)
SOURCE_PAIR_TRACE_LINUX_PATH = (
    "/root/g1_true23_runs/sonic_task_space_ppo_full_support_model0_vs_model1_trace_v1.json"
)
SOURCE_PAIR_TRACE_PATH = Path(SOURCE_PAIR_TRACE_LINUX_PATH)
SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    "gear_sonic/utils/g1_true23_sonic_task_space_ppo_full_support_update_delta_ablation.py",
    "gear_sonic/scripts/ablate_g1_true23_sonic_task_space_ppo_full_support_update_delta.py",
    "gear_sonic/tests/test_g1_true23_sonic_task_space_ppo_full_support_update_delta_ablation.py",
)
POLICY_ORDER = ("baseline", "block_only", "head_only", "full")
MODULE14_TENSORS = (
    "actor_module.decoders.g1_dyn.module.14.bias",
    "actor_module.decoders.g1_dyn.module.14.weight",
)
MODULE16_TENSORS = (
    "actor_module.decoders.g1_dyn.module.16.bias",
    "actor_module.decoders.g1_dyn.module.16.weight",
)
UPDATED_TENSORS = frozenset((*MODULE14_TENSORS, *MODULE16_TENSORS))
POLICY_SOURCES = {
    "baseline": (0, 0),
    "block_only": (1, 0),
    "head_only": (0, 1),
    "full": (1, 1),
}
POLICY_SHA256 = {
    "baseline": "358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61",
    "block_only": "1208a23d58f2476ad5513b9e1eb1afec257c14f1e07bac3e87415186fd12e58b",
    "head_only": "5a0880255a0f1f97a964548883dd15b39ed60b79a28941072b196f15d2f18367",
    "full": "7299df1851c5b42256f170334e2d5afdc81603b17ae832381abadadd9ea48639",
}


@dataclass(frozen=True)
class AblationInputs:
    repository_root: Path
    run_dir: Path
    contract: Mapping[str, Any]
    parent_inputs: paired.TraceInputs
    source_trace_path: Path
    source_trace: Mapping[str, Any]
    provenance: Mapping[str, Any]


class EndpointReproductionError(ValueError):
    """Endpoint replay mismatch with publication-safe scalar evidence."""

    def __init__(self, message: str, partial_evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        paired._assert_scalar_evidence(partial_evidence)
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


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int/float equality aliases."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        if (
            any(type(key) is not str for key in left)
            or any(type(key) is not str for key in right)
            or set(left) != set(right)
        ):
            return False
        return all(_exact_json_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_equal(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    if left is None:
        return True
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and left == right
    if type(left) in {bool, int, str}:
        return left == right
    return False


def _all_finite_non_bool(value: Any) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_finite_non_bool(item) for item in value)
    return False


def _as_strict_float_vector(value: Any, *, expected: int, context: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"ablation {context} row mismatch")
    parsed: list[float] = []
    for item in value:
        if type(item) is not float or not math.isfinite(item):
            raise ValueError(f"ablation {context} drift")
        parsed.append(item)
    return parsed


def _validate_action_semantics(
    raw_native_action: Any,
    safe_native_action: Any,
    final_target_hardware: Any,
    *,
    context: str,
) -> None:
    raw = _as_strict_float_vector(
        raw_native_action, expected=paired.ACTION_DIM, context=f"{context} raw_native_action"
    )
    safe = _as_strict_float_vector(
        safe_native_action, expected=paired.ACTION_DIM, context=f"{context} safe_native_action"
    )
    final = _as_strict_float_vector(
        final_target_hardware,
        expected=paired.ACTION_DIM,
        context=f"{context} final_target_hardware",
    )
    if any(abs(value) >= SAFE_TARGET_RAW_ACTION_CLIP for value in raw):
        raise ValueError(f"{context} raw_native_action strict clip threshold exceeded")
    expected_safe_native, expected_final_target = safe_target_transform_numpy(np.array(raw, dtype=np.float32))
    if not np.allclose(np.asarray(safe, dtype=np.float32), expected_safe_native, rtol=0.0, atol=1e-6):
        raise ValueError(f"{context} safe_native_action action transform mismatch")
    if not np.allclose(np.asarray(final, dtype=np.float32), expected_final_target, rtol=0.0, atol=1e-6):
        raise ValueError(f"{context} final_target_hardware action transform mismatch")


def _validate_publication_receipt_boundary(
    value: Mapping[str, Any], *, simulator_constructed: bool | None
) -> None:
    expected = paired._boundary_receipt(simulator_constructed=simulator_constructed)
    for key, expected_value in expected.items():
        if key not in value or type(value[key]) is not type(expected_value) or value[key] != expected_value:
            raise ValueError("ablation publication boundary receipt mismatch")


def _validate_success_publication(value: Mapping[str, Any]) -> str:
    if type(value.get("schema_version")) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("ablation publication schema mismatch")
    if value.get("kind") != ABLATION_KIND:
        raise ValueError("ablation publication kind mismatch")
    if type(value.get("evaluation_seed")) is not int or value["evaluation_seed"] != paired.FIXED_SEED:
        raise ValueError("ablation publication seed drift")
    provenance = _mapping(value.get("provenance"), "ablation report provenance")
    snapshot = _sha256(provenance.get("snapshot_sha256"), "ablation report provenance snapshot")
    _validate_publication_receipt_boundary(value, simulator_constructed=True)
    paired.assert_publication_boundary(value)
    return snapshot


def _validate_success_publication_report(
    value: Mapping[str, Any],
    inputs: AblationInputs,
) -> None:
    boundary = paired._boundary_receipt(simulator_constructed=True)
    expected_keys = {
        "schema_version",
        "kind",
        "purpose",
        "evaluation_seed",
        "parent_run_dir",
        "source_pair_trace",
        "provenance",
        "capture_contract",
        "policy_construction",
        "initial_actor_observation",
        "layout",
        "task_audit",
        "prime",
        "traces",
        "endpoint_reproduction",
        "pairwise_comparisons",
        "attribution_summary",
    } | set(boundary)
    if set(value) != expected_keys:
        raise ValueError("ablation success publication schema mismatch")

    if value.get("purpose") != inputs.contract["purpose"]:
        raise ValueError("ablation publication purpose mismatch")
    if type(value.get("parent_run_dir")) is not str or value["parent_run_dir"] != str(inputs.run_dir):
        raise ValueError("ablation publication parent run mismatch")
    source_pair = _mapping(value.get("source_pair_trace"), "ablation source-pair receipt")
    if set(source_pair) != {
        "path",
        "sha256",
        "provenance_snapshot_sha256",
        "validated_comparison_exact",
    }:
        raise ValueError("ablation source-pair receipt schema mismatch")
    if (
        source_pair.get("path") != str(inputs.source_trace_path)
        or _sha256(source_pair.get("sha256"), "ablation source-pair trace") != SOURCE_PAIR_TRACE_SHA256
        or _sha256(
            source_pair.get("provenance_snapshot_sha256"),
            "ablation source-pair provenance",
        )
        != SOURCE_PAIR_PROVENANCE_SHA256
        or source_pair.get("validated_comparison_exact") is not True
    ):
        raise ValueError("ablation source-pair receipt drift")

    provenance = _mapping(value.get("provenance"), "ablation report provenance")
    if not _exact_json_equal(provenance, inputs.provenance):
        raise ValueError("ablation report provenance drift")
    if not _exact_json_equal(value.get("capture_contract"), paired.CAPTURE_CONTRACT):
        raise ValueError("ablation capture contract drift")
    construction = _mapping(value.get("policy_construction"), "ablation policy construction")
    if not _exact_json_equal(construction, inputs.provenance["hybrid_policy_construction"]):
        raise ValueError("ablation policy construction drift")
    contract = inputs.contract
    source_layout = _mapping(inputs.source_trace.get("layout"), "source pair layout")
    layout = _mapping(value.get("layout"), "ablation report layout")
    if not _exact_json_equal(source_layout, layout):
        raise ValueError("ablation report layout drift")
    source_task_audit = _mapping(inputs.source_trace.get("task_audit"), "source pair task audit")
    source_prime = _mapping(inputs.source_trace.get("prime"), "source pair prime")
    task_audit = _mapping(value.get("task_audit"), "ablation task audit")
    prime = _mapping(value.get("prime"), "ablation prime")
    if not _exact_json_equal(task_audit, source_task_audit):
        raise ValueError("ablation task audit drift")
    if not _exact_json_equal(prime, source_prime):
        raise ValueError("ablation prime drift")

    traces = _mapping(value.get("traces"), "ablation traces")
    if set(traces) != set(POLICY_ORDER):
        raise ValueError("ablation trace set mismatch")
    for policy_name in POLICY_ORDER:
        trace = _mapping(traces[policy_name], f"ablation {policy_name} trace")
        validate_ablation_trace(trace, source_layout, contract)

    trace_initials = {
        name: _sha256(trace["initial_actor_observation_sha256"], f"ablation {name} initial observation")
        for name, trace in traces.items()
    }
    initial = _mapping(value.get("initial_actor_observation"), "ablation initial actor observation")
    if set(initial) != {"identical_across_all_four_policies", "sha256"}:
        raise ValueError("ablation initial actor observation schema mismatch")
    if (
        initial["identical_across_all_four_policies"] is not True
        or len(set(trace_initials.values())) != 1
        or _sha256(initial.get("sha256"), "ablation initial actor observation")
        != next(iter(trace_initials.values()))
    ):
        raise ValueError("ablation initial actor observation drift")

    expected_endpoints = _mapping(contract.get("expected_endpoint_reproductions"), "ablation expected endpoints")
    endpoint = _mapping(value.get("endpoint_reproduction"), "ablation endpoint reproduction")
    if set(endpoint) != set(expected_endpoints):
        raise ValueError("ablation endpoint reproduction key mismatch")
    for policy_name in expected_endpoints:
        expected = _mapping(expected_endpoints[policy_name], f"ablation expected endpoint for {policy_name}")
        observed = _mapping(endpoint[policy_name], f"ablation observed endpoint for {policy_name}")
        trace = _mapping(traces[policy_name], f"ablation report trace {policy_name}")
        recomputed = validate_endpoint_reproduction(trace, expected, policy_name)
        if not _exact_json_equal(observed, recomputed):
            raise ValueError("ablation endpoint reproduction drift")

    expected_pairwise_order = list(contract["comparison"]["pairwise_order"])
    pairwise = _mapping(value.get("pairwise_comparisons"), "ablation pairwise comparisons")
    if set(pairwise) != set(expected_pairwise_order):
        raise ValueError("ablation pairwise comparison key mismatch")
    recomputed_pairwise: dict[str, Any] = {}
    for label in expected_pairwise_order:
        left_name, right_name = label.split("_vs_", maxsplit=1)
        recomputed_pairwise[label] = compare_ablation_pair(
            left_name,
            _mapping(traces[left_name], f"ablation pairwise left {left_name}"),
            right_name,
            _mapping(traces[right_name], f"ablation pairwise right {right_name}"),
            source_layout,
            contract,
        )
        if not _exact_json_equal(pairwise[label], recomputed_pairwise[label]):
            raise ValueError("ablation pairwise comparison drift")

    attribution = _mapping(value.get("attribution_summary"), "ablation attribution summary")
    if not _exact_json_equal(
        attribution,
        _attribution_summary(traces, recomputed_pairwise, source_layout),
    ):
        raise ValueError("ablation attribution summary drift")


def _validate_failure_publication(value: Mapping[str, Any]) -> None:
    if type(value.get("schema_version")) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("ablation publication schema mismatch")
    if value.get("kind") != FAILURE_KIND:
        raise ValueError("ablation publication kind mismatch")
    if (
        type(value.get("simulator_ablation_complete")) is not bool
        or value["simulator_ablation_complete"] is not False
    ):
        raise ValueError("ablation publication failure kind drift")
    boundary = paired._boundary_receipt(simulator_constructed=None)
    expected = {
        "schema_version",
        "kind",
        "ablation_contract_sha256",
        "source_pair_trace_sha256",
        "failure_stage",
        "parent_run_dir",
        "preexecution_provenance_snapshot_sha256",
        "error",
        "simulator_ablation_complete",
        *boundary,
    }
    if set(value) != expected and set(value) != expected | {"partial_scalar_evidence"}:
        raise ValueError("ablation publication schema mismatch")
    if "simulator_constructed" in value:
        raise ValueError("ablation publication boundary receipt mismatch")
    if _sha256(value.get("ablation_contract_sha256"), "ablation failure contract") != CONTRACT_SHA256:
        raise ValueError("ablation failure contract mismatch")
    if (
        _sha256(value.get("source_pair_trace_sha256"), "ablation failure source pair trace")
        != SOURCE_PAIR_TRACE_SHA256
    ):
        raise ValueError("ablation failure source pair trace mismatch")
    if type(value.get("failure_stage")) is not str or not value["failure_stage"]:
        raise ValueError("ablation failure stage must be non-empty string")
    parent_run = value.get("parent_run_dir")
    if parent_run is not None and (type(parent_run) is not str or not parent_run):
        raise ValueError("ablation failure parent run must be non-empty string")
    preexecution = value.get("preexecution_provenance_snapshot_sha256")
    if preexecution is not None:
        _sha256(preexecution, "ablation preexecution snapshot")
    error = _mapping(value.get("error"), "ablation failure error")
    if set(error) != {"type", "message"}:
        raise ValueError("ablation failure error schema mismatch")
    if type(error.get("type")) is not str or not error["type"]:
        raise ValueError("ablation failure error type must be non-empty string")
    if type(error.get("message")) is not str:
        raise ValueError("ablation failure error message must be string")
    if "partial_scalar_evidence" in value:
        partial = value.get("partial_scalar_evidence")
        paired._assert_scalar_evidence(_mapping(partial, "ablation partial scalar evidence"))
    _validate_publication_receipt_boundary(value, simulator_constructed=None)
    paired.assert_publication_boundary(value)


def _validate_reward_internal_identity(
    contract_terms: Sequence[Mapping[str, Any]],
    internal: Mapping[str, Any],
) -> list[list[Any]]:
    required_outer_term = {"name", "weight", "callable_identity"}
    expected: list[list[Any]] = []
    for index, outer_value in enumerate(contract_terms):
        outer = _mapping(outer_value, f"reward term {index}")
        if set(outer) != required_outer_term:
            raise ValueError("ablation trace reward term schema mismatch")
        name = outer["name"]
        weight = outer["weight"]
        callable_identity = outer["callable_identity"]
        if type(name) is not str or not name:
            raise ValueError("ablation trace reward term name must be non-empty string")
        if type(weight) is not float or not math.isfinite(weight):
            raise ValueError("ablation trace reward term weight must be finite float")
        if type(callable_identity) is not str or not callable_identity:
            raise ValueError("ablation trace reward term callable must be non-empty string")
        expected.append([name, weight, callable_identity])
    required_internal = {
        "manager_class",
        "step_reward_shape",
        "column_order_verified",
        "terms",
    }
    required_term = {
        "column_index",
        "name",
        "weight",
        "callable_identity",
        "parameter_names",
        "public_internal_cfg_object_identical",
    }
    if set(internal) != required_internal:
        raise ValueError("ablation trace reward_internal_identity schema mismatch")
    manager_class = internal["manager_class"]
    if type(manager_class) is not str or not manager_class:
        raise ValueError("ablation trace reward_internal_identity manager drift")
    step_shape = internal["step_reward_shape"]
    if (
        type(step_shape) is not list
        or len(step_shape) != 2
        or any(type(dimension) is not int for dimension in step_shape)
        or step_shape != [1, len(expected)]
    ):
        raise ValueError("ablation trace reward_internal_identity step_reward_shape mismatch")
    if internal["column_order_verified"] is not True:
        raise ValueError("ablation trace reward_internal_identity column order drift")
    terms = internal.get("terms")
    if not isinstance(terms, list) or len(terms) != len(expected):
        raise ValueError("ablation trace reward_internal_identity term count mismatch")
    for index, term in enumerate(terms):
        term = _mapping(term, "reward_internal_identity term")
        if set(term) != required_term:
            raise ValueError("ablation trace reward_internal_identity term schema mismatch")
        if type(term["column_index"]) is not int or term["column_index"] != index:
            raise ValueError("ablation trace reward_internal_identity term order drift")
        name = term["name"]
        weight = term["weight"]
        callable_identity = term["callable_identity"]
        if type(name) is not str or not name:
            raise ValueError("ablation trace reward_internal_identity term name drift")
        if type(weight) is not float or not math.isfinite(weight):
            raise ValueError("ablation trace reward_internal_identity term weight drift")
        if type(callable_identity) is not str or not callable_identity:
            raise ValueError("ablation trace reward_internal_identity term callable drift")
        if not _exact_json_equal([name, weight, callable_identity], expected[index]):
            raise ValueError("ablation trace reward_internal_identity term drift")
        params = term["parameter_names"]
        if (
            not isinstance(params, list)
            or any(type(name) is not str for name in params)
            or sorted(params) != params
            or len(set(params)) != len(params)
        ):
            raise ValueError("ablation trace reward_internal_identity parameter_names drift")
        if term["public_internal_cfg_object_identical"] is not True:
            raise ValueError("ablation trace reward_internal_identity cfg binding drift")
    return expected


def _strict_json(path: Path, context: str) -> Mapping[str, Any]:
    value = json.loads(
        paired._regular_file(path, context).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{context} contains non-finite JSON token {token}")
        ),
    )
    return _mapping(value, context)


def _remove_publication_target_if_tied_to_temp(target_path: Path, temp_path: Path) -> None:
    try:
        target = target_path.lstat()
        temporary = temp_path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(target.st_mode) or not stat.S_ISREG(temporary.st_mode):
        return
    if target.st_dev == temporary.st_dev and target.st_ino == temporary.st_ino:
        os.unlink(target_path)


def _expected_policy_specs() -> dict[str, Any]:
    return {
        name: {
            "module14_source_update": POLICY_SOURCES[name][0],
            "module16_source_update": POLICY_SOURCES[name][1],
            "policy_state_sha256": POLICY_SHA256[name],
        }
        for name in POLICY_ORDER
    }


def load_ablation_contract(repository_root: str | Path | None = None) -> Mapping[str, Any]:
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    path = paired._regular_file(root / CONTRACT_RELATIVE_PATH, "update-delta ablation contract")
    if sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("update-delta ablation contract SHA256 mismatch")
    contract = _strict_json(path, "update-delta ablation contract")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("purpose")
        != "deterministic_diagnostic_only_attribute_update1_regression_between_decoder_module14_and_module16"
    ):
        raise ValueError("update-delta ablation contract identity mismatch")

    parent_contract = paired.load_trace_contract(root)
    parent = _mapping(contract.get("parent_run"), "ablation parent run")
    parent_reference = parent_contract["parent_run"]
    if (
        parent.get("linux_path") != parent_reference["linux_path"]
        or parent.get("directory_name") != parent_reference["directory_name"]
        or parent.get("artifact_sha256") != parent_reference["artifact_sha256"]
        or parent.get("full_support_contract_sha256") != paired.FULL_SUPPORT_CONTRACT_SHA256
        or parent.get("run_materials_sha256") != paired.RUN_MATERIALS_SHA256
        or parent.get("rollout_evidence_sha256") != paired.ROLLOUT_EVIDENCE_SHA256
    ):
        raise ValueError("update-delta ablation parent binding mismatch")

    source = _mapping(contract.get("source_pair_trace"), "source pair trace")
    if source != {
        "linux_path": SOURCE_PAIR_TRACE_LINUX_PATH,
        "sha256": SOURCE_PAIR_TRACE_SHA256,
        "kind": paired.TRACE_KIND,
        "provenance_snapshot_sha256": SOURCE_PAIR_PROVENANCE_SHA256,
        "contract_sha256": paired.TRACE_CONTRACT_SHA256,
        "utility_sha256": "f05e53a0eaaafd53fe718b6f530cee01be163ba8496642a04c1db9ee65f21104",
        "cli_sha256": "e07f51fabc49c37da137c0739fd93966cc43aea235e001cef04264c5bf878ad8",
        "test_sha256": "23043064d260d7544a90029813a748659381814412d97d2849b56cc30dfbc7bf",
    }:
        raise ValueError("source pair trace binding mismatch")

    sealed = _mapping(contract.get("sealed_sources"), "ablation sealed sources")
    if {
        logical: sha256_file(paired._regular_file(root / logical, f"sealed source {logical}"))
        for logical in sealed
    } != sealed:
        raise ValueError("ablation sealed source SHA256 mismatch")

    construction = _mapping(contract.get("policy_construction"), "policy construction")
    if (
        construction.get("policy_order") != list(POLICY_ORDER)
        or construction.get("module14_tensor_names") != list(MODULE14_TENSORS)
        or construction.get("module16_tensor_names") != list(MODULE16_TENSORS)
        or construction.get("all_other_policy_tensors_identical_between_model0_and_model1") is not True
        or construction.get("hybrids_are_in_memory_only") is not True
        or construction.get("policies") != _expected_policy_specs()
    ):
        raise ValueError("update-delta policy construction mismatch")

    endpoints = _mapping(contract.get("expected_endpoint_reproductions"), "endpoint reproductions")
    if endpoints != {
        "baseline": {
            "completed_transitions": 155,
            "terminal_q9": 163,
            "termination_names": ["ee_body_pos"],
            "source_trace_episode_return": -119.28385989740491,
        },
        "full": {
            "completed_transitions": 151,
            "terminal_q9": 159,
            "termination_names": ["ee_body_pos"],
            "source_trace_episode_return": -115.48705283179879,
        },
    }:
        raise ValueError("update-delta endpoint contract mismatch")

    execution = _mapping(contract.get("execution"), "ablation execution")
    if execution != {
        "evaluation_seed": paired.FIXED_SEED,
        "num_envs": 1,
        "device": paired.DEVICE,
        "controller": "deterministic_actor_mean",
        "max_transitions": 510,
        "initial_q9": 9,
        "expected_max_episode_length": 510,
        "wrapper_clip_actions": None,
        "capture_contract": paired.CAPTURE_CONTRACT,
        "fresh_environment_per_policy": True,
        "fresh_actor_instance_per_policy": True,
        "allowed_terminal_terms": ["anchor_ori", "anchor_pos", "ee_body_pos", "time_out"],
        "all_absolute_episode_returns_are_diagnostic_only": True,
        "safe_target_transform_kind": SAFE_TARGET_TRANSFORM_KIND,
        "safe_target_transform_application_count": SAFE_TARGET_TRANSFORM_APPLICATION_COUNT,
        "raw_native_action_strict_abs_max": SAFE_TARGET_RAW_ACTION_CLIP,
        "initial_actor_observation_clone_only_before_first_action": True,
        "initial_actor_observation_host_materialization_after_episode": True,
        "require_identical_initial_actor_observation_sha256": True,
        "full_provenance_rehash_before_and_after_every_evaluation": True,
        "full_provenance_rehash_immediately_before_publication": True,
    }:
        raise ValueError("update-delta execution contract mismatch")

    comparison = _mapping(contract.get("comparison"), "ablation comparison")
    if (
        comparison.get("pairwise_all_six") is not True
        or comparison.get("pairwise_order")
        != [
            "baseline_vs_block_only",
            "baseline_vs_head_only",
            "baseline_vs_full",
            "block_only_vs_head_only",
            "block_only_vs_full",
            "head_only_vs_full",
        ]
        or comparison.get("reward_attribution_uses_preterminal_common_q9_only") is not True
        or comparison.get("global_four_way_preterminal_reward_factorial") is not True
        or comparison.get("global_reward_factorial_q9_first") != 9
        or comparison.get("global_reward_factorial_q9_last_rule") != "minimum_of_all_four_terminal_q9_minus_one"
        or comparison.get("absolute_episode_return_never_authorizes_attribution_or_candidate") is not True
        or comparison.get("action_linf_thresholds") != [1e-6, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
        or comparison.get("ee_error_delta_thresholds_m") != [0.0001, 0.001, 0.01, 0.025, 0.05]
        or comparison.get("reward_delta_atol") != 1e-6
        or comparison.get("contact_force_delta_atol_n") != 0.001
        or comparison.get("landing_force_delta_atol_n") != 0.001
        or comparison.get("ee_body_pos_threshold_m") != 0.25
        or comparison.get("barrier_term_name") != "right_wrist_prethreshold_barrier"
        or comparison.get("worst_ee_term_name") != "worst_ee_z_normalized_squared"
        or contract.get("expected_reward_terms") != parent_contract["expected_reward_terms"]
        or contract.get("boundaries") != paired._boundary_receipt(simulator_constructed=None)
        or contract.get("output")
        != {
            "linux_path": f"/root/g1_true23_runs/{OUTPUT_FILENAME}",
            "filename": OUTPUT_FILENAME,
            "exclusive_create": True,
        }
    ):
        raise ValueError("update-delta comparison/boundary/output mismatch")
    return contract


def _source_binding(root: Path) -> dict[str, Any]:
    records = []
    for logical in sorted(SOURCE_RELATIVE_PATHS):
        path = paired._regular_file(root / logical, f"ablation source {logical}")
        records.append(
            {
                "logical_path": logical,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "kind": "g1_true23_full_support_update_delta_ablation_executed_sources_v1",
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "manifest_sha256": _canonical_sha256(records),
        "files": records,
    }


def _validate_source_pair_trace(
    report: Mapping[str, Any],
    parent_inputs: paired.TraceInputs,
    contract: Mapping[str, Any],
) -> None:
    if (
        report.get("schema_version") != paired.SCHEMA_VERSION
        or report.get("kind") != paired.TRACE_KIND
        or report.get("evaluation_seed") != paired.FIXED_SEED
        or report.get("parent_run_dir") != str(parent_inputs.run_dir)
        or report.get("capture_contract") != paired.CAPTURE_CONTRACT
        or _mapping(report.get("provenance"), "source trace provenance").get("snapshot_sha256")
        != SOURCE_PAIR_PROVENANCE_SHA256
    ):
        raise ValueError("source pair trace identity mismatch")
    paired.assert_publication_boundary(report)
    source_contract = parent_inputs.contract
    layout = _mapping(report.get("layout"), "source trace layout")
    traces = _mapping(report.get("traces"), "source pair traces")
    if set(traces) != {"model0", "model1"}:
        raise ValueError("source pair trace set mismatch")
    for update, name in enumerate(("model0", "model1")):
        trace = _mapping(traces[name], f"source {name}")
        paired.validate_trace_series(trace, layout, source_contract)
        paired.validate_structural_reproduction(
            trace,
            source_contract["expected_reproductions"][str(update)],
            update,
        )
    recomputed = paired.compare_trace_pair(traces["model0"], traces["model1"], layout, source_contract)
    if not _exact_json_equal(recomputed, report.get("comparison")):
        raise ValueError("source pair trace comparison mismatch")
    endpoints = contract["expected_endpoint_reproductions"]
    if any(
        not _exact_json_equal(
            traces[source_name].get(field),
            endpoints[target_name].get(field),
        )
        for source_name, target_name in (("model0", "baseline"), ("model1", "full"))
        for field in ("completed_transitions", "terminal_q9", "termination_names")
    ):
        raise ValueError("source pair trace endpoint mismatch")


def _policy_mapping(checkpoint: Mapping[str, Any], context: str) -> Mapping[str, torch.Tensor]:
    policy = _mapping(checkpoint.get("policy_state_dict"), context)
    if not policy or any(
        type(name) is not str or type(value) is not torch.Tensor for name, value in policy.items()
    ):
        raise ValueError(f"{context} tensor schema mismatch")
    return policy  # type: ignore[return-value]


def construct_hybrid_policy_state(
    checkpoints: Mapping[int, Mapping[str, Any]],
    policy_name: str,
    contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Build one exact, fully unaliased CPU policy state."""

    if policy_name not in POLICY_SOURCES:
        raise ValueError("unknown update-delta ablation policy")
    policy0 = _policy_mapping(checkpoints[0], "model0 policy")
    policy1 = _policy_mapping(checkpoints[1], "model1 policy")
    if set(policy0) != set(policy1):
        raise ValueError("model0/model1 policy namespaces differ")
    changed = {name for name in policy0 if not torch.equal(policy0[name], policy1[name])}
    if changed != UPDATED_TENSORS:
        raise ValueError("model0/model1 changed tensor set is not exact module14+module16")
    module14_source, module16_source = POLICY_SOURCES[policy_name]
    sources = {0: policy0, 1: policy1}
    result: dict[str, torch.Tensor] = {}
    for name in sorted(policy0):
        source_update = (
            module14_source if name in MODULE14_TENSORS else module16_source if name in MODULE16_TENSORS else 0
        )
        source = sources[source_update][name]
        clone = source.detach().cpu().contiguous().clone()
        if clone.data_ptr() == source.data_ptr():
            raise RuntimeError("hybrid policy tensor aliases immutable checkpoint")
        result[name] = clone
    if len({value.data_ptr() for value in result.values()}) != len(result):
        raise RuntimeError("hybrid policy tensors alias each other")
    expected = contract["policy_construction"]["policies"][policy_name]
    if (
        expected["module14_source_update"] != module14_source
        or expected["module16_source_update"] != module16_source
    ):
        raise ValueError("hybrid policy source contract mismatch")
    observed = inspect_true23_policy_state(
        {"policy_state_dict": result},
        reference_profile="released_low_latency_step1_0p02s",
    )
    if observed != expected["policy_state_sha256"] or observed != POLICY_SHA256[policy_name]:
        raise ValueError(f"{policy_name} canonical policy SHA256 mismatch")
    return result


def _verify_hybrid_policies(
    checkpoints: Mapping[int, Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = {}
    for name in POLICY_ORDER:
        state = construct_hybrid_policy_state(checkpoints, name, contract)
        receipt[name] = {
            "module14_source_update": POLICY_SOURCES[name][0],
            "module16_source_update": POLICY_SOURCES[name][1],
            "policy_state_sha256": POLICY_SHA256[name],
            "tensor_count": len(state),
            "fully_cloned_no_checkpoint_alias": True,
        }
        del state
    return {
        "changed_tensor_names": sorted(UPDATED_TENSORS),
        "all_other_policy_tensors_identical": True,
        "policies": receipt,
    }


def _current_provenance_snapshot(inputs: AblationInputs) -> str:
    parent_snapshot = paired._current_provenance_snapshot(inputs.parent_inputs)
    if parent_snapshot != SOURCE_PAIR_PROVENANCE_SHA256:
        raise RuntimeError("parent/source-pair provenance changed")
    source_trace_sha = sha256_file(paired._regular_file(inputs.source_trace_path, "source pair trace"))
    if source_trace_sha != SOURCE_PAIR_TRACE_SHA256:
        raise RuntimeError("source pair trace changed")
    sealed = inputs.contract["sealed_sources"]
    actual_sealed = {
        logical: sha256_file(paired._regular_file(inputs.repository_root / logical, f"sealed source {logical}"))
        for logical in sorted(sealed)
    }
    if actual_sealed != sealed:
        raise RuntimeError("ablation sealed source changed")
    return _canonical_sha256(
        {
            "contract_sha256": CONTRACT_SHA256,
            "parent_provenance_snapshot_sha256": parent_snapshot,
            "source_pair_trace_sha256": source_trace_sha,
            "sealed_source_sha256": actual_sealed,
            "executed_ablation_sources": _source_binding(inputs.repository_root),
            "canonical_hybrid_policy_sha256": POLICY_SHA256,
        }
    )


def resolve_ablation_inputs(repository_root: Path, parent_run_dir: Path) -> AblationInputs:
    root = repository_root.expanduser().resolve(strict=True)
    contract = load_ablation_contract(root)
    run_dir = parent_run_dir.expanduser().resolve(strict=True)
    expected_run = Path(contract["parent_run"]["linux_path"]).expanduser().resolve(strict=True)
    if run_dir != expected_run or run_dir.is_symlink():
        raise ValueError("update-delta ablation parent run mismatch")
    parent_inputs = paired.resolve_trace_inputs(root, run_dir)
    if parent_inputs.provenance["snapshot_sha256"] != SOURCE_PAIR_PROVENANCE_SHA256:
        raise ValueError("source pair parent provenance mismatch")
    source_path = paired._regular_file(SOURCE_PAIR_TRACE_PATH, "source pair trace")
    if sha256_file(source_path) != SOURCE_PAIR_TRACE_SHA256:
        raise ValueError("source pair trace SHA256 mismatch")
    source_trace = _strict_json(source_path, "source pair trace")
    _validate_source_pair_trace(source_trace, parent_inputs, contract)
    checkpoints = paired._load_and_validate_checkpoints(parent_inputs)
    hybrid_receipt = _verify_hybrid_policies(checkpoints, contract)
    del checkpoints
    provenance: dict[str, Any] = {
        "ablation_contract_sha256": CONTRACT_SHA256,
        "parent_run_dir": str(run_dir),
        "parent_provenance_snapshot_sha256": SOURCE_PAIR_PROVENANCE_SHA256,
        "source_pair_trace_sha256": SOURCE_PAIR_TRACE_SHA256,
        "source_pair_trace_provenance_sha256": SOURCE_PAIR_PROVENANCE_SHA256,
        "executed_ablation_sources": _source_binding(root),
        "hybrid_policy_construction": hybrid_receipt,
    }
    provisional = AblationInputs(
        repository_root=root,
        run_dir=run_dir,
        contract=contract,
        parent_inputs=parent_inputs,
        source_trace_path=source_path,
        source_trace=source_trace,
        provenance=provenance,
    )
    provenance["snapshot_sha256"] = _current_provenance_snapshot(provisional)
    return provisional


def ablation_preflight(repository_root: Path, parent_run_dir: Path) -> dict[str, Any]:
    try:
        inputs = resolve_ablation_inputs(repository_root, parent_run_dir)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PREFLIGHT_KIND,
            "ready": True,
            "parent_run_dir": str(inputs.run_dir),
            "source_pair_trace": str(inputs.source_trace_path),
            "provenance": inputs.provenance,
            **paired._boundary_receipt(simulator_constructed=False),
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PREFLIGHT_KIND,
            "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            **paired._boundary_receipt(simulator_constructed=False),
        }


def _load_live_policy(
    actor: Any,
    policy_state: Mapping[str, torch.Tensor],
    expected_sha256: str,
) -> None:
    network = {key: value for key, value in policy_state.items() if key != "std"}
    missing, unexpected = actor.core.load_state_dict(network, strict=True)
    if missing or unexpected:
        raise ValueError("hybrid actor load mismatch")
    std = policy_state.get("std")
    if type(std) is not torch.Tensor or tuple(std.shape) != (paired.ACTION_DIM,):
        raise ValueError("hybrid actor std mismatch")
    with torch.no_grad():
        actor.distribution.std_param.copy_(std.to(actor.distribution.std_param.device))
    observed = inspect_true23_policy_state(
        {"policy_state_dict": actor.export_true23_policy_state()},
        reference_profile="released_low_latency_step1_0p02s",
    )
    if observed != expected_sha256:
        raise ValueError("live hybrid actor identity mismatch")


def _clone_initial_actor_observation(observations: Any) -> dict[str, torch.Tensor]:
    snapshots: dict[str, torch.Tensor] = {}
    for name, width in (("tokenizer", 268), ("policy", 930)):
        value = observations.get(name)
        if type(value) is not torch.Tensor or tuple(value.shape) != (1, width) or value.dtype is not torch.float32:
            raise ValueError(f"initial actor observation {name} metadata mismatch")
        snapshots[name] = value.detach().clone()
    return snapshots


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("initial actor observation is nonfinite")
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _trace_layout(raw_env: Any, contract: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any]]:
    reward = paired._reward_layout(raw_env)
    internal = paired.reward_internal_identity(raw_env, contract)
    terms = [
        {
            "name": term["name"],
            "weight": term["weight"],
            "callable_identity": term["callable_identity"],
        }
        for term in internal["terms"]
    ]
    return (
        {
            "reward_terms": terms,
            "reward_internal_identity": internal,
            "control_dt_s": reward["control_dt_s"],
            "ee_body_names": list(paired.EE_BODY_NAMES),
            "contact_site_names": list(paired.CONTACT_SITE_NAMES),
            "native_joint_names": list(NATIVE_IL23_JOINT_NAMES),
            "hardware_joint_names": list(HARDWARE_23_JOINT_NAMES),
            "action_orders": {
                "raw_native_action": "native_isaaclab_23",
                "safe_native_action": "native_isaaclab_23",
                "final_target_hardware": "hardware_mujoco_23",
            },
            "ee_body_pos_threshold_m": 0.25,
        },
        reward,
    )


def validate_ablation_trace(
    trace: Mapping[str, Any], layout: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    required_trace_keys = {
        "policy_name",
        "module14_source_update",
        "module16_source_update",
        "evaluation_seed",
        "controller",
        "policy_state_sha256",
        "initial_actor_observation_sha256",
        "completed_transitions",
        "terminal_q9",
        "termination_names",
        "episode_return",
        "series",
    }
    if set(trace) != required_trace_keys:
        raise ValueError("ablation trace schema mismatch")
    policy_name = trace.get("policy_name")
    if type(policy_name) is not str or policy_name not in POLICY_ORDER:
        raise ValueError("ablation trace policy name mismatch")
    spec = contract["policy_construction"]["policies"][policy_name]
    if (
        type(trace.get("module14_source_update")) is not int
        or type(trace.get("module16_source_update")) is not int
        or type(trace.get("evaluation_seed")) is not int
        or trace.get("module14_source_update") != spec["module14_source_update"]
        or trace.get("module16_source_update") != spec["module16_source_update"]
        or trace.get("policy_state_sha256") != spec["policy_state_sha256"]
        or trace.get("evaluation_seed") != paired.FIXED_SEED
        or trace.get("controller") != "deterministic_actor_mean"
    ):
        raise ValueError("ablation trace policy identity mismatch")
    paired._sha256(trace.get("initial_actor_observation_sha256"), "initial observation")
    required_layout = {
        "reward_terms",
        "reward_internal_identity",
        "control_dt_s",
        "ee_body_names",
        "contact_site_names",
        "native_joint_names",
        "hardware_joint_names",
        "action_orders",
        "ee_body_pos_threshold_m",
    }
    expected_action_orders = {
        "raw_native_action": "native_isaaclab_23",
        "safe_native_action": "native_isaaclab_23",
        "final_target_hardware": "hardware_mujoco_23",
    }
    terms = layout.get("reward_terms")
    if (
        set(layout) != required_layout
        or layout.get("control_dt_s") != 0.02
        or tuple(layout.get("ee_body_names", ())) != paired.EE_BODY_NAMES
        or tuple(layout.get("contact_site_names", ())) != paired.CONTACT_SITE_NAMES
        or tuple(layout.get("native_joint_names", ())) != tuple(NATIVE_IL23_JOINT_NAMES)
        or tuple(layout.get("hardware_joint_names", ())) != tuple(HARDWARE_23_JOINT_NAMES)
        or layout.get("ee_body_pos_threshold_m") != 0.25
        or type(terms) is not list
    ):
        raise ValueError("ablation trace layout mismatch")
    if layout.get("action_orders") != expected_action_orders:
        raise ValueError("ablation trace action_orders mismatch")
    observed_reward_terms = _validate_reward_internal_identity(
        terms,
        _mapping(layout.get("reward_internal_identity"), "ablation reward_internal_identity"),
    )
    expected_reward_terms = contract.get("expected_reward_terms")
    if type(expected_reward_terms) is not list or len(expected_reward_terms) != len(observed_reward_terms):
        raise ValueError("ablation trace layout mismatch")
    for expected_term in expected_reward_terms:
        if (
            type(expected_term) is not list
            or len(expected_term) != 3
            or type(expected_term[0]) is not str
            or not expected_term[0]
            or type(expected_term[1]) is not float
            or not math.isfinite(expected_term[1])
            or type(expected_term[2]) is not str
            or not expected_term[2]
        ):
            raise ValueError("ablation trace expected reward term schema mismatch")
    if not _exact_json_equal(observed_reward_terms, expected_reward_terms):
        raise ValueError("ablation trace layout mismatch")

    series = _mapping(trace.get("series"), "ablation trace series")
    required_series = {
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
    if set(series) != required_series:
        raise ValueError("ablation trace series schema mismatch")
    completed = trace.get("completed_transitions")
    episode_return = trace.get("episode_return")
    if (
        type(completed) is not int
        or not 1 <= completed <= 510
        or type(trace.get("terminal_q9")) is not int
        or type(episode_return) is not float
        or not math.isfinite(episode_return)
        or type(trace.get("termination_names")) is not list
        or any(type(name) is not str for name in trace["termination_names"])
    ):
        raise ValueError("ablation trace terminal/return summary mismatch")
    q9s = series.get("q9")
    if not isinstance(q9s, list) or any(type(q9) is not int for q9 in q9s) or q9s != list(range(9, 9 + completed)):
        raise ValueError("ablation trace q9 mismatch")
    for name in required_series:
        if not isinstance(series[name], list) or len(series[name]) != completed:
            raise ValueError(f"ablation trace {name} length mismatch")
    if any(type(value) is not float or not math.isfinite(value) for value in series["reward_total"]):
        raise ValueError("ablation trace scalar series nonfinite")
    widths = {
        "reward_raw": len(terms),
        "reward_weighted": len(terms),
        "ee_z_error_m": len(paired.EE_BODY_NAMES),
        "contact_force_magnitude_n": len(paired.CONTACT_SITE_NAMES),
        "raw_native_action": paired.ACTION_DIM,
        "safe_native_action": paired.ACTION_DIM,
        "final_target_hardware": paired.ACTION_DIM,
        "contact_found": len(paired.CONTACT_SITE_NAMES),
    }
    for name, width in widths.items():
        if name == "contact_found":
            if any(
                not isinstance(row, list) or len(row) != width or any(type(value) is not bool for value in row)
                for row in series[name]
            ):
                raise ValueError(f"ablation trace {name} row mismatch")
            continue
        if any(
            not isinstance(row, list)
            or len(row) != width
            or any(type(value) is not float or not math.isfinite(value) for value in row)
            for row in series[name]
        ):
            raise ValueError(f"ablation trace {name} row mismatch")
    if any(
        type(value) is not float or value < 0.0 or not math.isfinite(value)
        for value in series["landing_force_mean_n"]
    ):
        raise ValueError("ablation trace landing_force_mean_n row mismatch")
    if any(value < 0.0 for row in series["ee_z_error_m"] for value in row):
        raise ValueError("ablation trace ee_z_error_m must be nonnegative")
    if any(value < 0.0 for row in series["contact_force_magnitude_n"] for value in row):
        raise ValueError("ablation trace contact_force_magnitude_n must be nonnegative")
    for index in range(completed):
        _validate_action_semantics(
            series["raw_native_action"][index],
            series["safe_native_action"][index],
            series["final_target_hardware"][index],
            context=f"ablation trace {policy_name} frame {index}",
        )
    weights = [term["weight"] for term in terms]
    for index in range(completed):
        names = series["termination_names"][index]
        if not isinstance(names, list) or any(type(value) is not str for value in names):
            raise ValueError("ablation trace termination names must be lists of strings")
        if not math.isclose(
            sum(series["reward_weighted"][index]),
            series["reward_total"][index],
            rel_tol=0.0,
            abs_tol=2e-5,
        ):
            raise ValueError("ablation trace reward sum mismatch")
        for raw, weighted, weight in zip(
            series["reward_raw"][index],
            series["reward_weighted"][index],
            weights,
            strict=True,
        ):
            if not math.isclose(raw * weight * 0.02, weighted, rel_tol=0.0, abs_tol=2e-5):
                raise ValueError("ablation trace raw/weighted reward mismatch")
    terminal = [index for index, names in enumerate(series["termination_names"]) if names]
    allowed = set(contract["execution"]["allowed_terminal_terms"])
    final_names = series["termination_names"][-1]
    if (
        terminal != [completed - 1]
        or not isinstance(final_names, list)
        or any(type(name) is not str for name in final_names)
        or not final_names
        or any(series["termination_names"][index] != [] for index in range(completed - 1))
        or final_names != sorted(set(final_names))
        or not set(final_names) <= allowed
    ):
        raise ValueError("ablation trace must have one allowed final terminal frame")
    threshold = 0.25
    if any(max(row) > threshold for row in series["ee_z_error_m"][:-1]):
        raise ValueError("ablation trace crossed EE threshold before terminal")
    final_crossed = max(series["ee_z_error_m"][-1]) > threshold
    if ("ee_body_pos" in final_names) != final_crossed:
        raise ValueError("ablation trace terminal EE semantics mismatch")
    timed_out = "time_out" in final_names
    if timed_out != (completed == 510 and series["q9"][-1] == 518):
        raise ValueError("ablation trace timeout semantics mismatch")
    if (
        trace.get("terminal_q9") != series["q9"][-1]
        or trace.get("termination_names") != final_names
        or not math.isclose(sum(series["reward_total"]), float(episode_return), rel_tol=0.0, abs_tol=2e-5)
    ):
        raise ValueError("ablation trace terminal/return summary mismatch")


def validate_endpoint_reproduction(
    trace: Mapping[str, Any], expected: Mapping[str, Any], policy_name: str
) -> dict[str, Any]:
    mismatch = any(
        trace.get(field) != expected.get(field)
        for field in ("completed_transitions", "terminal_q9", "termination_names")
    )
    if mismatch:
        evidence = {
            "stage": f"{policy_name}_endpoint_reproduction",
            "policy_name": policy_name,
            "observed_completed_transitions": trace.get("completed_transitions"),
            "expected_completed_transitions": expected.get("completed_transitions"),
            "observed_terminal_q9": trace.get("terminal_q9"),
            "expected_terminal_q9": expected.get("terminal_q9"),
            "observed_termination_names": ",".join(trace.get("termination_names", [])),
            "expected_termination_names": ",".join(expected.get("termination_names", [])),
        }
        raise EndpointReproductionError(f"{policy_name} endpoint reproduction failed", evidence)
    observed = float(trace["episode_return"])
    source = float(expected["source_trace_episode_return"])
    return {
        "policy_name": policy_name,
        "observed_episode_return": observed,
        "source_trace_episode_return": source,
        "delta_observed_minus_source_trace": observed - source,
        "exact_match": observed == source,
        "gate_applied": False,
        "reason": "absolute_replay_return_is_diagnostic_only",
    }


def run_ablation_trace(
    *,
    policy: Any,
    wrapped_env: Any,
    policy_name: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    raw_env = getattr(wrapped_env, "unwrapped", None)
    if (
        raw_env is None
        or int(getattr(raw_env, "num_envs", -1)) != 1
        or int(getattr(raw_env.cfg, "seed", -1)) != paired.FIXED_SEED
        or getattr(wrapped_env, "clip_actions", "missing") is not None
        or int(getattr(wrapped_env, "max_episode_length", -1)) != 510
        or int(raw_env.common_step_counter) != 0
        or int(raw_env._sim_step_counter) != 0
    ):
        raise ValueError("ablation environment is not fresh exact one-env task")
    command = raw_env.command_manager.get_term("motion")
    if paired._q9(command) != 9:
        raise ValueError("ablation trace did not start at q9=9")
    expected_hash = contract["policy_construction"]["policies"][policy_name]["policy_state_sha256"]
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy.export_true23_policy_state()},
        reference_profile="released_low_latency_step1_0p02s",
    )
    if policy_hash != expected_hash:
        raise ValueError("ablation live policy hash mismatch")
    layout, reward = _trace_layout(raw_env, contract)
    observations = wrapped_env.get_observations()
    initial_observation = _clone_initial_actor_observation(observations)
    recorder = paired._RewardComputeTraceRecorder(raw_env, reward)
    was_training = bool(policy.training)
    policy.eval()
    frames: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for transition in range(510):
                q9 = paired._q9(command)
                if q9 != 9 + transition:
                    raise ValueError("ablation trace q9 discontinuity")
                actor_action = policy(observations, stochastic_output=False)
                if (
                    type(actor_action) is not torch.Tensor
                    or tuple(actor_action.shape) != (1, paired.ACTION_DIM)
                    or actor_action.dtype is not torch.float32
                    or not bool(torch.isfinite(actor_action).all())
                ):
                    raise ValueError("ablation actor action metadata drift")
                if not bool(torch.all(torch.abs(actor_action) < SAFE_TARGET_RAW_ACTION_CLIP)):
                    raise ValueError("ablation raw_native_action strict clip threshold exceeded")
                recorder.arm(q9, actor_action)
                observations, rewards, dones, _extras = wrapped_env.step(actor_action.to(wrapped_env.device))
                frame = recorder.finish()
                if (
                    type(rewards) is not torch.Tensor
                    or rewards.dtype is not torch.float32
                    or tuple(rewards.shape) != (1,)
                ):
                    raise ValueError("ablation wrapped step schema drift")
                if type(dones) is not torch.Tensor or dones.dtype is not torch.long or tuple(dones.shape) != (1,):
                    raise ValueError("ablation wrapped step done dtype drift")
                wrapped_reward = float(rewards[0].detach().cpu().item())
                if not math.isfinite(wrapped_reward) or not math.isclose(
                    frame["reward_total"], wrapped_reward, rel_tol=0.0, abs_tol=1e-7
                ):
                    raise ValueError("ablation wrapped reward identity mismatch")
                done_value = int(dones[0].detach().cpu().item())
                if done_value not in (0, 1):
                    raise ValueError("ablation wrapped done value drift")
                done = bool(done_value)
                if done != bool(frame["termination_names"]):
                    raise ValueError("ablation done/termination mismatch")
                frames.append(frame)
                if done:
                    if frame["episode_length_pre_reset"] != len(frames):
                        raise ValueError("ablation terminal episode length mismatch")
                    break
    finally:
        recorder.restore()
        policy.train(was_training)
    if not frames or not frames[-1]["termination_names"]:
        raise ValueError("ablation policy did not terminate within exact 510 transitions")
    spec = contract["policy_construction"]["policies"][policy_name]
    trace = {
        "policy_name": policy_name,
        "module14_source_update": spec["module14_source_update"],
        "module16_source_update": spec["module16_source_update"],
        "evaluation_seed": paired.FIXED_SEED,
        "controller": "deterministic_actor_mean",
        "policy_state_sha256": policy_hash,
        "initial_actor_observation_sha256": _tensor_state_sha256(initial_observation),
        "completed_transitions": len(frames),
        "terminal_q9": frames[-1]["q9"],
        "termination_names": frames[-1]["termination_names"],
        "episode_return": sum(frame["reward_total"] for frame in frames),
        "series": paired.frames_to_series(frames),
    }
    validate_ablation_trace(trace, layout, contract)
    return {"layout": layout, "trace": trace}


def _scalar_divergence(
    left: Sequence[float],
    right: Sequence[float],
    q9s: Sequence[int],
    threshold: float,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    if not left or len(left) != len(right) or len(left) != len(q9s):
        raise ValueError("ablation scalar comparison rows mismatch")
    deltas = [abs(a - b) for a, b in zip(left, right, strict=True)]
    maximum_index = max(range(len(deltas)), key=deltas.__getitem__)
    first_index = next((index for index, delta in enumerate(deltas) if delta > threshold), None)

    def evidence(index: int | None) -> dict[str, Any] | None:
        if index is None:
            return None
        return {
            "q9": q9s[index],
            "absolute_delta": deltas[index],
            "left_value": left[index],
            "right_value": right[index],
            "right_minus_left": right[index] - left[index],
        }

    return {
        "left_policy": left_name,
        "right_policy": right_name,
        "threshold_strictly_greater_than": threshold,
        "first": evidence(first_index),
        "maximum": evidence(maximum_index),
    }


def _reward_sums(series: Mapping[str, Any], indices: Sequence[int]) -> list[float]:
    width = len(series["reward_weighted"][0])
    return [sum(series["reward_weighted"][row][column] for row in indices) for column in range(width)]


def _terminal_body_summary(trace: Mapping[str, Any]) -> dict[str, Any]:
    errors = trace["series"]["ee_z_error_m"][-1]
    index = max(range(len(errors)), key=errors.__getitem__)
    return {
        "q9": trace["terminal_q9"],
        "termination_names": trace["termination_names"],
        "worst_body_name": paired.EE_BODY_NAMES[index],
        "worst_body_z_error_m": errors[index],
        "ee_body_pos_threshold_m": 0.25,
        "threshold_crossing_bodies": [
            name for name, value in zip(paired.EE_BODY_NAMES, errors, strict=True) if value > 0.25
        ],
        "all_four_ee_z_error_m": {name: value for name, value in zip(paired.EE_BODY_NAMES, errors, strict=True)},
        "terminal_contact_found": {
            name: value
            for name, value in zip(
                paired.CONTACT_SITE_NAMES,
                trace["series"]["contact_found"][-1],
                strict=True,
            )
        },
        "terminal_contact_force_magnitude_n": {
            name: value
            for name, value in zip(
                paired.CONTACT_SITE_NAMES,
                trace["series"]["contact_force_magnitude_n"][-1],
                strict=True,
            )
        },
    }


def _suffix_summary(trace: Mapping[str, Any], common_last_q9: int, reward_names: Sequence[str]) -> dict[str, Any]:
    series = trace["series"]
    indices = [index for index, q9 in enumerate(series["q9"]) if q9 > common_last_q9]
    sums = _reward_sums(series, indices)
    return {
        "transition_count": len(indices),
        "q9_first": None if not indices else series["q9"][indices[0]],
        "q9_last": None if not indices else series["q9"][indices[-1]],
        "reward_total_sum": sum(series["reward_total"][index] for index in indices),
        "reward_weighted_sum_by_term": {name: value for name, value in zip(reward_names, sums, strict=True)},
    }


def compare_ablation_pair(
    left_name: str,
    left_trace: Mapping[str, Any],
    right_name: str,
    right_trace: Mapping[str, Any],
    layout: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two closed-loop traces only on shared q9 support."""

    validate_ablation_trace(left_trace, layout, contract)
    validate_ablation_trace(right_trace, layout, contract)
    if left_trace["policy_name"] != left_name or right_trace["policy_name"] != right_name:
        raise ValueError("ablation pair labels do not match traces")
    left = left_trace["series"]
    right = right_trace["series"]
    common_q9s = sorted(set(left["q9"]) & set(right["q9"]))
    expected_common = list(range(9, min(left_trace["terminal_q9"], right_trace["terminal_q9"]) + 1))
    if common_q9s != expected_common:
        raise ValueError("ablation pair common q9 mismatch")
    preterminal_q9s = common_q9s[:-1]
    left_index = {q9: index for index, q9 in enumerate(left["q9"])}
    right_index = {q9: index for index, q9 in enumerate(right["q9"])}

    def paired_rows(column: str) -> list[list[Sequence[float]]]:
        return [[left[column][left_index[q9]], right[column][right_index[q9]]] for q9 in common_q9s]

    comparison = contract["comparison"]
    action = {}
    for column, names in (
        ("raw_native_action", NATIVE_IL23_JOINT_NAMES),
        ("safe_native_action", NATIVE_IL23_JOINT_NAMES),
        ("final_target_hardware", HARDWARE_23_JOINT_NAMES),
    ):
        action[column] = paired._threshold_ladder(
            paired_rows(column),
            common_q9s,
            names,
            comparison["action_linf_thresholds"],
        )
    ee = paired._threshold_ladder(
        paired_rows("ee_z_error_m"),
        common_q9s,
        paired.EE_BODY_NAMES,
        comparison["ee_error_delta_thresholds_m"],
    )
    contact_state = {}
    for site_index, site in enumerate(paired.CONTACT_SITE_NAMES):
        contact_state[site] = next(
            (
                q9
                for q9 in common_q9s
                if left["contact_found"][left_index[q9]][site_index]
                != right["contact_found"][right_index[q9]][site_index]
            ),
            None,
        )
    contact_force = paired._threshold_ladder(
        paired_rows("contact_force_magnitude_n"),
        common_q9s,
        paired.CONTACT_SITE_NAMES,
        [comparison["contact_force_delta_atol_n"]],
    )
    reward_names = [term["name"] for term in layout["reward_terms"]]
    left_common_indices = [left_index[q9] for q9 in common_q9s]
    right_common_indices = [right_index[q9] for q9 in common_q9s]
    left_pre_indices = [left_index[q9] for q9 in preterminal_q9s]
    right_pre_indices = [right_index[q9] for q9 in preterminal_q9s]
    common_left = _reward_sums(left, left_common_indices)
    common_right = _reward_sums(right, right_common_indices)
    pre_left = _reward_sums(left, left_pre_indices)
    pre_right = _reward_sums(right, right_pre_indices)
    reward_total = _scalar_divergence(
        [left["reward_total"][left_index[q9]] for q9 in common_q9s],
        [right["reward_total"][right_index[q9]] for q9 in common_q9s],
        common_q9s,
        comparison["reward_delta_atol"],
        left_name,
        right_name,
    )
    reward_terms = {}
    for term_index, term_name in enumerate(reward_names):
        reward_terms[term_name] = _scalar_divergence(
            [left["reward_weighted"][left_index[q9]][term_index] for q9 in common_q9s],
            [right["reward_weighted"][right_index[q9]][term_index] for q9 in common_q9s],
            common_q9s,
            comparison["reward_delta_atol"],
            left_name,
            right_name,
        )
    landing = _scalar_divergence(
        [left["landing_force_mean_n"][left_index[q9]] for q9 in common_q9s],
        [right["landing_force_mean_n"][right_index[q9]] for q9 in common_q9s],
        common_q9s,
        comparison["landing_force_delta_atol_n"],
        left_name,
        right_name,
    )
    target_terms = {}
    for term_name in (comparison["barrier_term_name"], comparison["worst_ee_term_name"]):
        term_index = reward_names.index(term_name)
        target_terms[term_name] = {
            "left_first_positive_raw_q9": next(
                (q9 for q9, row in zip(left["q9"], left["reward_raw"], strict=True) if row[term_index] > 0.0),
                None,
            ),
            "right_first_positive_raw_q9": next(
                (q9 for q9, row in zip(right["q9"], right["reward_raw"], strict=True) if row[term_index] > 0.0),
                None,
            ),
            "preterminal_common_weighted_sum_left": pre_left[term_index],
            "preterminal_common_weighted_sum_right": pre_right[term_index],
            "preterminal_common_right_minus_left": pre_right[term_index] - pre_left[term_index],
        }
    return {
        "left_policy": left_name,
        "right_policy": right_name,
        "common_q9_first": common_q9s[0],
        "common_q9_last": common_q9s[-1],
        "common_transition_count": len(common_q9s),
        "preterminal_common_q9_last": None if not preterminal_q9s else preterminal_q9s[-1],
        "preterminal_common_transition_count": len(preterminal_q9s),
        "terminal_q9": {"left": left_trace["terminal_q9"], "right": right_trace["terminal_q9"]},
        "completed_transitions": {
            "left": left_trace["completed_transitions"],
            "right": right_trace["completed_transitions"],
            "right_minus_left": right_trace["completed_transitions"] - left_trace["completed_transitions"],
        },
        "action_divergence": action,
        "ee_error_divergence": ee,
        "contact_state_first_divergence_q9": contact_state,
        "contact_force_divergence": contact_force,
        "landing_force_divergence": landing,
        "reward_total_divergence": reward_total,
        "reward_term_divergence": reward_terms,
        "reward_by_term": {
            "names": reward_names,
            "common_through_shorter_terminal": {
                "left": common_left,
                "right": common_right,
                "right_minus_left": [
                    right_value - left_value
                    for left_value, right_value in zip(common_left, common_right, strict=True)
                ],
            },
            "preterminal_common_prefix": {
                "left": pre_left,
                "right": pre_right,
                "right_minus_left": [
                    right_value - left_value for left_value, right_value in zip(pre_left, pre_right, strict=True)
                ],
                "total_left": sum(pre_left),
                "total_right": sum(pre_right),
                "total_right_minus_left": sum(pre_right) - sum(pre_left),
            },
        },
        "target_reward_terms": target_terms,
        "exclusive_suffix": {
            "left": _suffix_summary(left_trace, common_q9s[-1], reward_names),
            "right": _suffix_summary(right_trace, common_q9s[-1], reward_names),
        },
        "terminal_body": {
            "left": _terminal_body_summary(left_trace),
            "right": _terminal_body_summary(right_trace),
        },
        "absolute_episode_return_diagnostic_only": {
            "left": left_trace["episode_return"],
            "right": right_trace["episode_return"],
            "right_minus_left": right_trace["episode_return"] - left_trace["episode_return"],
            "authorizes_attribution_or_candidate": False,
        },
    }


def _factorial_effect(values: Mapping[str, int | float]) -> dict[str, Any]:
    baseline = values["baseline"]
    block = values["block_only"]
    head = values["head_only"]
    full = values["full"]
    return {
        "values": dict(values),
        "module14_effect_when_module16_at_model0": block - baseline,
        "module14_effect_when_module16_at_model1": full - head,
        "module16_effect_when_module14_at_model0": head - baseline,
        "module16_effect_when_module14_at_model1": full - block,
        "module14_module16_interaction": full - block - head + baseline,
    }


def _global_preterminal_reward_factorial(
    traces: Mapping[str, Mapping[str, Any]], layout: Mapping[str, Any]
) -> dict[str, Any]:
    q9_last = min(int(traces[name]["terminal_q9"]) for name in POLICY_ORDER) - 1
    q9s = list(range(9, q9_last + 1))
    if not q9s:
        raise ValueError("ablation lacks global four-way preterminal reward support")
    reward_names = [term["name"] for term in layout["reward_terms"]]
    sums: dict[str, list[float]] = {}
    for policy_name in POLICY_ORDER:
        series = traces[policy_name]["series"]
        index = {q9: row for row, q9 in enumerate(series["q9"])}
        if any(q9 not in index for q9 in q9s):
            raise ValueError("ablation global reward support is not shared by all policies")
        sums[policy_name] = _reward_sums(series, [index[q9] for q9 in q9s])
    total = {name: sum(sums[name]) for name in POLICY_ORDER}
    by_term = {
        term_name: _factorial_effect({name: sums[name][term_index] for name in POLICY_ORDER})
        for term_index, term_name in enumerate(reward_names)
    }
    return {
        "q9_first": q9s[0],
        "q9_last": q9s[-1],
        "transition_count": len(q9s),
        "terminal_frames_excluded": True,
        "same_q9_support_for_all_four_policies": True,
        "total_weighted_reward": _factorial_effect(total),
        "weighted_reward_by_term": by_term,
    }


def _attribution_summary(
    traces: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    layout: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_pairs = {
        name: comparisons[f"baseline_vs_{name}"]["reward_by_term"]["preterminal_common_prefix"]
        for name in ("block_only", "head_only", "full")
    }
    return {
        "factorial_completed_transitions": _factorial_effect(
            {name: int(traces[name]["completed_transitions"]) for name in POLICY_ORDER}
        ),
        "factorial_terminal_q9": _factorial_effect(
            {name: int(traces[name]["terminal_q9"]) for name in POLICY_ORDER}
        ),
        "preterminal_common_reward_delta_vs_baseline": {
            name: evidence["total_right_minus_left"] for name, evidence in baseline_pairs.items()
        },
        "global_four_way_preterminal_reward_factorial": _global_preterminal_reward_factorial(traces, layout),
        "absolute_episode_returns_diagnostic_only": {
            name: traces[name]["episode_return"] for name in POLICY_ORDER
        },
        "absolute_episode_return_authorizes_attribution_or_candidate": False,
        "candidate_selected": False,
    }


def execute_update_delta_ablation(repository_root: Path, parent_run_dir: Path) -> dict[str, Any]:
    """Run four fresh deterministic evaluations; never optimize or train."""

    inputs = resolve_ablation_inputs(repository_root, parent_run_dir)
    starting_snapshot = inputs.provenance["snapshot_sha256"]
    checkpoints = paired._load_and_validate_checkpoints(inputs.parent_inputs)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    from gear_sonic.envs.mjlab.sonic_true23 import prime_sonic_true23_training_environment
    from gear_sonic.trl.mjlab.sonic_task_space_ppo_runner import (
        audit_task_space_ppo_env_cfg,
        make_task_space_ppo_env_cfg,
    )
    from gear_sonic.trl.mjlab.true23_actor import True23SonicActorModel

    bound_inputs = paired._bound_input_paths(inputs.repository_root)
    topology = bound_inputs["actor/topology.pt"]
    motion = bound_inputs["motion/B_DadDance.npz"]
    traces: dict[str, Mapping[str, Any]] = {}
    endpoint_diagnostics: dict[str, Mapping[str, Any]] = {}
    common_layout: Mapping[str, Any] | None = None
    common_audit: Mapping[str, Any] | None = None
    common_prime: Mapping[str, Any] | None = None
    for policy_name in POLICY_ORDER:
        if _current_provenance_snapshot(inputs) != starting_snapshot:
            raise RuntimeError(f"ablation provenance changed before {policy_name}")
        state = construct_hybrid_policy_state(checkpoints, policy_name, inputs.contract)
        actor = True23SonicActorModel(
            {
                "tokenizer": torch.zeros((1, 268), dtype=torch.float32),
                "policy": torch.zeros((1, 930), dtype=torch.float32),
            },
            {"actor": ["tokenizer", "policy"]},
            "actor",
            paired.ACTION_DIM,
            warm_start_path=str(topology),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.1,
                "std_type": "scalar",
            },
            hidden_dims=(),
            activation="silu",
            obs_normalization=False,
        ).to(paired.DEVICE)
        _load_live_policy(actor, state, POLICY_SHA256[policy_name])
        paired._seed_everything()
        cfg = make_task_space_ppo_env_cfg(motion_file=str(motion), num_envs=1)
        cfg.seed = paired.FIXED_SEED
        audit = audit_task_space_ppo_env_cfg(cfg, expected_num_envs=1)
        env = ManagerBasedRlEnv(cfg=cfg, device=paired.DEVICE)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
            prime = prime_sonic_true23_training_environment(wrapped)
            if not _exact_json_equal(prime, inputs.source_trace["prime"]) or not _exact_json_equal(
                audit,
                inputs.source_trace["task_audit"],
            ):
                raise ValueError(f"{policy_name} env/prime differs from source pair trace")
            executed = run_ablation_trace(
                policy=actor,
                wrapped_env=wrapped,
                policy_name=policy_name,
                contract=inputs.contract,
            )
            after_hash = inspect_true23_policy_state(
                {"policy_state_dict": actor.export_true23_policy_state()},
                reference_profile="released_low_latency_step1_0p02s",
            )
            if after_hash != POLICY_SHA256[policy_name]:
                raise RuntimeError(f"{policy_name} policy changed during diagnostic evaluation")
        finally:
            env.close()
        if common_layout is None:
            common_layout, common_audit, common_prime = executed["layout"], audit, prime
        elif (
            not _exact_json_equal(executed["layout"], common_layout)
            or not _exact_json_equal(audit, common_audit)
            or not _exact_json_equal(prime, common_prime)
        ):
            raise ValueError("ablation environments/layouts differ across policies")
        trace = executed["trace"]
        traces[policy_name] = trace
        if policy_name in inputs.contract["expected_endpoint_reproductions"]:
            endpoint_diagnostics[policy_name] = validate_endpoint_reproduction(
                trace,
                inputs.contract["expected_endpoint_reproductions"][policy_name],
                policy_name,
            )
        del state, actor, wrapped, env
        if _current_provenance_snapshot(inputs) != starting_snapshot:
            raise RuntimeError(f"ablation provenance changed after {policy_name}")
    assert common_layout is not None and common_audit is not None and common_prime is not None
    initial_hashes = {name: trace["initial_actor_observation_sha256"] for name, trace in traces.items()}
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError("ablation policies did not receive identical initial actor observations")
    pairwise: dict[str, Mapping[str, Any]] = {}
    for left_name, right_name in combinations(POLICY_ORDER, 2):
        label = f"{left_name}_vs_{right_name}"
        pairwise[label] = compare_ablation_pair(
            left_name,
            traces[left_name],
            right_name,
            traces[right_name],
            common_layout,
            inputs.contract,
        )
    if list(pairwise) != inputs.contract["comparison"]["pairwise_order"]:
        raise RuntimeError("ablation pairwise comparison order drift")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": ABLATION_KIND,
        "purpose": inputs.contract["purpose"],
        "evaluation_seed": paired.FIXED_SEED,
        "parent_run_dir": str(inputs.run_dir),
        "source_pair_trace": {
            "path": str(inputs.source_trace_path),
            "sha256": SOURCE_PAIR_TRACE_SHA256,
            "provenance_snapshot_sha256": SOURCE_PAIR_PROVENANCE_SHA256,
            "validated_comparison_exact": True,
        },
        "provenance": inputs.provenance,
        "capture_contract": paired.CAPTURE_CONTRACT,
        "policy_construction": inputs.provenance["hybrid_policy_construction"],
        "initial_actor_observation": {
            "identical_across_all_four_policies": True,
            "sha256": next(iter(initial_hashes.values())),
        },
        "layout": common_layout,
        "task_audit": common_audit,
        "prime": common_prime,
        "traces": traces,
        "endpoint_reproduction": endpoint_diagnostics,
        "pairwise_comparisons": pairwise,
        "attribution_summary": _attribution_summary(traces, pairwise, common_layout),
        **paired._boundary_receipt(simulator_constructed=True),
    }
    paired.assert_publication_boundary(report)
    if _current_provenance_snapshot(inputs) != starting_snapshot:
        raise RuntimeError("ablation provenance changed before publication")
    return report


def validate_publication_provenance(
    repository_root: Path,
    parent_run_dir: Path,
    report: Mapping[str, Any],
) -> str:
    expected = _validate_success_publication(report)
    inputs = resolve_ablation_inputs(repository_root, parent_run_dir)
    _validate_success_publication_report(_mapping(report, "ablation success publication"), inputs)
    actual = _sha256(
        _mapping(inputs.provenance, "ablation resolved inputs provenance").get("snapshot_sha256"),
        "ablation resolved input snapshot",
    )
    if actual != expected:
        raise RuntimeError("ablation publication provenance changed")
    latest = _sha256(_current_provenance_snapshot(inputs), "ablation live provenance snapshot")
    if latest != expected:
        raise RuntimeError("ablation publication provenance changed")
    return actual


def _publication_target_path(path: Path) -> Path:
    target = Path(path).expanduser().absolute()
    paired._regular_directory(target.parent, "ablation publication target parent")
    if os.path.lexists(target):
        raise FileExistsError(f"ablation publication target must not overwrite existing output: {target}")
    return target


def _write_payload_or_raise(stream: Any, payload: bytes) -> None:
    written = stream.write(payload)
    if written != len(payload):
        raise RuntimeError("ablation publication temp write was short")
    stream.flush()
    os.fsync(stream.fileno())


def write_json_exclusive(
    path: Path,
    value: Mapping[str, Any],
    *,
    publication_guard: Callable[[], str] | None = None,
    publication_receipt_guard: Callable[[], str] | None = None,
) -> None:
    kind = value.get("kind")
    if kind == ABLATION_KIND:
        snapshot = _validate_success_publication(_mapping(value, "ablation success publication"))
        if publication_guard is None:
            raise ValueError("ablation success report publication requires publication_guard")
        if publication_receipt_guard is None:
            raise ValueError("ablation success report publication requires publication_receipt_guard")
    elif kind != FAILURE_KIND:
        raise ValueError(f"unsupported ablation publication kind: {kind!r}")
    else:
        _validate_failure_publication(_mapping(value, "ablation failure publication"))
    payload = json.dumps(_mapping(value, "ablation publication"), sort_keys=True, allow_nan=False).encode("utf-8")
    target = _publication_target_path(path)
    temp_path: Path | None = None
    published = False
    try:
        handle, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temp_path = Path(temp_name)
        with os.fdopen(handle, "wb") as stream:
            _write_payload_or_raise(stream, payload)
        if kind == ABLATION_KIND:
            pre_publication_snapshot = _sha256(publication_guard(), "ablation publication guard return")
            if pre_publication_snapshot != snapshot:
                raise ValueError("ablation publication guard returned stale provenance")
            os.link(temp_path, target)
            published = True
            post_publication_snapshot = _sha256(
                publication_receipt_guard(), "ablation publication receipt guard return"
            )
            if post_publication_snapshot != snapshot:
                raise ValueError("ablation publication receipt guard returned stale provenance")
        else:
            os.link(temp_path, target)
            published = True
    except BaseException:
        if kind == ABLATION_KIND and published:
            _remove_publication_target_if_tied_to_temp(target, temp_path)
        raise
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except Exception:
                if not published:
                    raise


def failure_report(
    error: Exception,
    *,
    failure_stage: str = "unattributed_failure",
    parent_run_dir: str | None = None,
    provenance_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    if type(failure_stage) is not str or not failure_stage:
        raise ValueError("failure stage must be non-empty string")
    if provenance_snapshot_sha256 is not None:
        _sha256(provenance_snapshot_sha256, "failure provenance snapshot")
    if parent_run_dir is not None and (type(parent_run_dir) is not str or not parent_run_dir):
        raise ValueError("failure parent run must be non-empty string")
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": FAILURE_KIND,
        "ablation_contract_sha256": CONTRACT_SHA256,
        "source_pair_trace_sha256": SOURCE_PAIR_TRACE_SHA256,
        "failure_stage": failure_stage,
        "parent_run_dir": parent_run_dir,
        "preexecution_provenance_snapshot_sha256": provenance_snapshot_sha256,
        "error": {"type": type(error).__name__, "message": str(error)},
        "simulator_ablation_complete": False,
        **paired._boundary_receipt(simulator_constructed=None),
    }
    partial = getattr(error, "partial_evidence", None)
    if partial is not None:
        paired._assert_scalar_evidence(partial)
        report["partial_scalar_evidence"] = dict(partial)
    paired.assert_publication_boundary(report)
    return report


__all__ = [
    "ABLATION_KIND",
    "CONTRACT_SHA256",
    "EndpointReproductionError",
    "OUTPUT_FILENAME",
    "POLICY_ORDER",
    "POLICY_SHA256",
    "ablation_preflight",
    "compare_ablation_pair",
    "construct_hybrid_policy_state",
    "execute_update_delta_ablation",
    "failure_report",
    "load_ablation_contract",
    "resolve_ablation_inputs",
    "run_ablation_trace",
    "validate_ablation_trace",
    "validate_endpoint_reproduction",
    "validate_publication_provenance",
    "write_json_exclusive",
]
