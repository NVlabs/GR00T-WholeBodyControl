"""In-memory symmetric line-search attribution for full-support PPO updates.

This diagnostic crosses decoder module-14 and module-16 tensors from immutable
model 0 and model 1 checkpoints. It never creates an optimizer, trains, saves a
checkpoint, selects a candidate, or communicates with robot hardware.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
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
    g1_true23_sonic_task_space_ppo_full_support_update_delta_ablation as source_ablation,
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
LINE_SEARCH_KIND = "g1_true23_sonic_task_space_ppo_full_support_delta_line_search_v1"
ABLATION_KIND = LINE_SEARCH_KIND
PREFLIGHT_KIND = f"{LINE_SEARCH_KIND}_preflight"
FAILURE_KIND = f"{LINE_SEARCH_KIND}_failure"
CONTRACT_KIND = "g1_true23_sonic_task_space_ppo_full_support_delta_line_search_contract_v1"
CONTRACT_SHA256 = "e8fb32b6173c2a58f1beb11ae764572334d2da192e49d405a16a4f9aa7912cbd"
SAFE_TARGET_TRANSFORM_APPLICATION_COUNT = 1
SOURCE_PAIR_TRACE_SHA256 = "0fb71869eb1249958f7f4137df52b9b0bd92b4666d07c74a614ee11622a878d3"
SOURCE_PAIR_PROVENANCE_SHA256 = "3733a8f1248392f1e64ed7af01ef8a75c5f1e1409af343e30a6f9cee237ce849"
SOURCE_ABLATION_REPORT_SHA256 = "a44fcf7ca73e8bd03e99cb43568659c4a507f2327bdacf1924960f12dc47c1af"
SOURCE_ABLATION_REPORT_PROVENANCE_SHA256 = "4c16c4dd8716a1c19921a0d1648e8b96389159c558b1f85818ba1935c210661c"
DELTA_ABLATION_REPORT_SHA256 = "a44fcf7ca73e8bd03e99cb43568659c4a507f2327bdacf1924960f12dc47c1af"
DELTA_ABLATION_REPORT_PROVENANCE_SNAPSHOT_SHA256 = (
    "4c16c4dd8716a1c19921a0d1648e8b96389159c558b1f85818ba1935c210661c"
)
SOURCE_ABLATION_REPORT_LINUX_PATH = (
    "/root/g1_true23_runs/sonic_task_space_ppo_full_support_update_delta_ablation_v1.json"
)
DELTA_ABLATION_REPORT_LINUX_PATH = SOURCE_ABLATION_REPORT_LINUX_PATH
PRIOR_FAILURE_OUTPUT_FILENAME = "sonic_task_space_ppo_full_support_delta_line_search_v1.json"
PRIOR_FAILURE_REPORT_LINUX_PATH = (
    "/root/g1_true23_runs/sonic_task_space_ppo_full_support_delta_line_search_v1.json"
)
PRIOR_FAILURE_REPORT_SHA256 = "0be248f2405aded5ade65632e95eb7fc5eae26d79082b3bb43177e4d4316ec3a"
PRIOR_FAILURE_REPORT_SIZE_BYTES = 1410
PRIOR_FAILURE_CONTRACT_SHA256 = "d6ac14e63de0a74a31946a2d056b54451c307587d7f31f5a3383f363b9c104d0"
PRIOR_FAILURE_STAGE = "exclusive_publication"
PRIOR_FAILURE_KIND = "g1_true23_sonic_task_space_ppo_full_support_delta_line_search_v1_failure"
PRIOR_FAILURE_ERROR = {
    "type": "ValueError",
    "message": "ablation pairwise comparison order drift",
}
OUTPUT_FILENAME = "sonic_task_space_ppo_full_support_delta_line_search_v1_retry1.json"
LINE_SEARCH_REPORT_LINUX_PATH = f"/root/g1_true23_runs/{OUTPUT_FILENAME}"

CONTRACT_RELATIVE_PATH = (
    "gear_sonic/config/sim_validation/g1_true23_sonic_task_space_ppo_full_support_delta_line_search_v1.json"
)
SOURCE_PAIR_TRACE_LINUX_PATH = (
    "/root/g1_true23_runs/sonic_task_space_ppo_full_support_model0_vs_model1_trace_v1.json"
)
SOURCE_PAIR_TRACE_PATH = Path(SOURCE_PAIR_TRACE_LINUX_PATH)
SOURCE_RELATIVE_PATHS = (
    CONTRACT_RELATIVE_PATH,
    "gear_sonic/utils/g1_true23_sonic_task_space_ppo_full_support_delta_line_search.py",
    "gear_sonic/scripts/trace_g1_true23_sonic_task_space_ppo_full_support_delta_line_search.py",
    "gear_sonic/tests/test_g1_true23_sonic_task_space_ppo_full_support_delta_line_search.py",
)
POLICY_ORDER = (
    "alpha_minus_0_25",
    "baseline",
    "alpha_plus_0_25",
    "alpha_plus_0_5",
    "full",
)
PAIRWISE_POLICY_NAMES_BY_LABEL: dict[str, tuple[str, str]] = {
    f"{left_name}_vs_{right_name}": (left_name, right_name)
    for left_name, right_name in combinations(POLICY_ORDER, 2)
}
PAIRWISE_COMPARISON_ORDER = tuple(sorted(PAIRWISE_POLICY_NAMES_BY_LABEL))
if len(PAIRWISE_COMPARISON_ORDER) != 10 or len(set(PAIRWISE_COMPARISON_ORDER)) != len(PAIRWISE_COMPARISON_ORDER):
    raise RuntimeError("line-search pairwise comparison order malformed")
POLICY_ALPHAS = {
    "alpha_minus_0_25": (-1, 4),
    "baseline": (0, 1),
    "alpha_plus_0_25": (1, 4),
    "alpha_plus_0_5": (1, 2),
    "full": (1, 1),
}
MODULE14_TENSORS = (
    "actor_module.decoders.g1_dyn.module.14.bias",
    "actor_module.decoders.g1_dyn.module.14.weight",
)
MODULE16_TENSORS = (
    "actor_module.decoders.g1_dyn.module.16.bias",
    "actor_module.decoders.g1_dyn.module.16.weight",
)
UPDATED_TENSORS = (*MODULE14_TENSORS, *MODULE16_TENSORS)
POLICY_SHA256 = {
    "alpha_minus_0_25": "eae8d0287a9e17d44d33e7bd2ca86598e9a4b8bc0acb1669236d78a9e4a6e67c",
    "baseline": "358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61",
    "alpha_plus_0_25": "bb14ccc4fc465d3c61580c3f3b2f2512ddcd3fc0d9bf49ed1fa7cbd048585113",
    "alpha_plus_0_5": "dd56f4cd7b2c18e69b63aa4c7e537f89407dc18a4ffc17465bbebf8819919753",
    "full": "7299df1851c5b42256f170334e2d5afdc81603b17ae832381abadadd9ea48639",
}


@dataclass(frozen=True)
class LineSearchInputs:
    repository_root: Path
    run_dir: Path
    contract: Mapping[str, Any]
    source_ablation_inputs: source_ablation.AblationInputs
    source_ablation_report_path: Path
    source_ablation_report: Mapping[str, Any]
    provenance: Mapping[str, Any]
    source_pair_trace: Mapping[str, Any] | None = None
    parent_inputs: paired.TraceInputs | None = None
    source_trace_path: Path | None = None
    source_trace: Mapping[str, Any] | None = None


# Preserve legacy compatibility for downstream imports.
AblationInputs = LineSearchInputs


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


def _policy_expected_state_sha256(
    policy_spec: Mapping[str, Any],
    *,
    context: str,
) -> str:
    if not isinstance(policy_spec, Mapping):
        raise ValueError(f"{context} policy spec must be a mapping")
    if "expected_policy_state_sha256" not in policy_spec:
        raise ValueError(f"{context} missing expected policy state hash")
    return _sha256(policy_spec["expected_policy_state_sha256"], f"{context} expected policy state hash")


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
    if any(abs(value) >= 10.0 for value in raw):
        raise ValueError(f"{context} raw_native_action strict clip threshold exceeded")
    expected_safe_native, expected_final_target = safe_target_transform_numpy(np.array(raw, dtype=np.float32))
    if not np.allclose(np.asarray(safe, dtype=np.float32), expected_safe_native, rtol=0.0, atol=1e-6):
        raise ValueError(f"{context} safe_native_action action transform mismatch")
    if not np.allclose(np.asarray(final, dtype=np.float32), expected_final_target, rtol=0.0, atol=1e-6):
        raise ValueError(f"{context} final_target_hardware action transform mismatch")


def _assert_json_finite_scalars(value: Any, *, context: str) -> None:
    if value is None:
        return
    if type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{context} contains non-finite float")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _assert_json_finite_scalars(item, context=f"{context}[{index}]")
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError(f"{context} contains non-string key")
        for key, item in value.items():
            _assert_json_finite_scalars(item, context=f"{context}[{key!r}]")
        return
    raise ValueError(f"{context} contains unsupported scalar container type")


def _line_search_boundary_receipt(
    *, simulator_constructed: bool | None, include_simulator_transitions: bool = False
) -> dict[str, Any]:
    receipt = dict(paired._boundary_receipt(simulator_constructed=simulator_constructed))
    receipt.update(
        {
            "optimizer_steps": 0,
            "training_transitions": 0,
            "training_updates": 0,
            "teacher_queries": 0,
            "robot_commands_issued": 0,
            "network_commands_issued": 0,
            "hardware_actions": 0,
        }
    )
    if include_simulator_transitions:
        receipt["simulator_transitions"] = 0
    return receipt


def _validate_publication_receipt_boundary(
    value: Mapping[str, Any], *, simulator_constructed: bool | None
) -> None:
    expected = _line_search_boundary_receipt(simulator_constructed=simulator_constructed)
    for key, expected_value in expected.items():
        if key not in value:
            raise ValueError("ablation publication boundary receipt mismatch")
        actual_value = value[key]
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ValueError("ablation publication boundary receipt mismatch")


def _validate_comparison_pairwise_order(comparison: Mapping[str, Any], *, context: str) -> list[tuple[str, str]]:
    pairwise_order = comparison.get("pairwise_order")
    comparison_count = comparison.get("pairwise_count")
    expected_pairwise_order = list(PAIRWISE_COMPARISON_ORDER)
    if (
        type(comparison_count) is not int
        or comparison_count != len(PAIRWISE_COMPARISON_ORDER)
        or type(pairwise_order) is not list
        or tuple(pairwise_order) != tuple(expected_pairwise_order)
    ):
        raise ValueError(f"{context} comparison pairwise order mismatch")
    if set(pairwise_order) != set(expected_pairwise_order):
        raise ValueError(f"{context} comparison pairwise order drift")
    parsed: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for label in pairwise_order:
        if type(label) is not str:
            raise ValueError(f"{context} comparison pairwise label type drift")
        pair = PAIRWISE_POLICY_NAMES_BY_LABEL.get(label)
        if pair is None:
            raise ValueError(f"{context} comparison pairwise label drift")
        if pair in seen:
            raise ValueError(f"{context} comparison pairwise label duplicate {label!r}")
        seen.add(pair)
        parsed.append(pair)
    return parsed


def _build_pairwise_comparisons(
    traces: Mapping[str, Mapping[str, Any]],
    layout: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    pairwise_order = _validate_comparison_pairwise_order(
        _mapping(contract.get("comparison"), "line-search runtime comparison"),
        context="line-search runtime",
    )
    pairwise: dict[str, Any] = {}
    for left_name, right_name in pairwise_order:
        label = f"{left_name}_vs_{right_name}"
        pairwise[label] = compare_ablation_pair(
            left_name,
            _mapping(traces[left_name], f"line-search pairwise left {left_name}"),
            right_name,
            _mapping(traces[right_name], f"line-search pairwise right {right_name}"),
            layout,
            contract,
        )
    return pairwise


def _rebuild_pairwise_comparisons(
    *,
    expected_pairwise_order: Sequence[str] | None,
    pairwise_comparisons: Mapping[str, Mapping[str, Any]],
    traces: Mapping[str, Mapping[str, Any]],
    layout: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    contract_pairwise = _validate_comparison_pairwise_order(
        _mapping(runtime_contract.get("comparison"), "line-search runtime comparison"),
        context="line-search runtime",
    )
    expected_order = (
        [f"{left_name}_vs_{right_name}" for left_name, right_name in contract_pairwise]
        if expected_pairwise_order is None
        else list(expected_pairwise_order)
    )
    if len(expected_order) != len(set(expected_order)):
        raise ValueError("ablation pairwise comparison duplicate labels")
    if set(expected_order) != set(PAIRWISE_COMPARISON_ORDER):
        raise ValueError("ablation pairwise comparison key mismatch")

    pairwise = _mapping(pairwise_comparisons, "ablation pairwise comparisons")
    if set(pairwise) != set(expected_order):
        raise ValueError("ablation pairwise comparison key mismatch")
    if list(pairwise) != expected_order:
        raise ValueError("ablation pairwise comparison order drift")

    recomputed: dict[str, Any] = {}
    for label in expected_order:
        pair = PAIRWISE_POLICY_NAMES_BY_LABEL.get(label)
        if pair is None:
            raise ValueError("ablation pairwise comparison label drift")
        left_name, right_name = pair
        recomputed[label] = compare_ablation_pair(
            left_name,
            _mapping(traces[left_name], f"ablation pairwise left {left_name}"),
            right_name,
            _mapping(traces[right_name], f"ablation pairwise right {right_name}"),
            layout,
            runtime_contract,
        )
        if not _exact_json_equal(pairwise[label], recomputed[label]):
            raise ValueError("ablation pairwise comparison drift")
    return recomputed


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
    _assert_json_finite_scalars(value, context="ablation success publication")
    paired.assert_publication_boundary(value)
    return snapshot


def _validate_success_publication_report(
    value: Mapping[str, Any],
    inputs: AblationInputs,
) -> None:
    runtime_contract = _runtime_trace_contract(
        _mapping(inputs.contract, "line-search runtime contract"),
        _mapping(
            inputs.provenance.get("hybrid_policy_construction"),
            "line-search runtime policy construction",
        ),
    )
    provenance_requirements = _mapping(
        inputs.contract.get("provenance_requirements"),
        "line-search provenance requirements",
    )
    if provenance_requirements.get("source_pair_task_audit_layout_prime_exact_match") is not True:
        raise ValueError("line-search provenance requirement for source pair task/layout/prime disabled")
    boundary = _line_search_boundary_receipt(simulator_constructed=True)
    expected_keys = {
        "schema_version",
        "kind",
        "purpose",
        "evaluation_seed",
        "parent_run_dir",
        "source_pair_trace",
        "source_delta_ablation_report",
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
    _assert_json_finite_scalars(value, context="ablation success publication")

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
    source_delta = _mapping(value.get("source_delta_ablation_report"), "ablation source-delta receipt")
    if set(source_delta) != {
        "path",
        "sha256",
        "provenance_snapshot_sha256",
        "validated_publication_snapshot",
        "validated_publication",
    }:
        raise ValueError("ablation source-delta receipt schema mismatch")
    if (
        source_delta.get("path") != str(inputs.source_ablation_report_path)
        or _sha256(source_delta.get("sha256"), "ablation source-delta report hash")
        != SOURCE_ABLATION_REPORT_SHA256
        or _sha256(
            source_delta.get("provenance_snapshot_sha256"),
            "ablation source-delta provenance snapshot",
        )
        != SOURCE_ABLATION_REPORT_PROVENANCE_SHA256
        or type(source_delta.get("validated_publication_snapshot")) is not str
        or _sha256(
            source_delta.get("validated_publication_snapshot"), "ablation source-delta publication snapshot"
        )
        != SOURCE_ABLATION_REPORT_PROVENANCE_SHA256
        or source_delta.get("validated_publication") is not True
    ):
        raise ValueError("ablation source-delta receipt drift")

    provenance = _mapping(value.get("provenance"), "ablation report provenance")
    if not _exact_json_equal(provenance, inputs.provenance):
        raise ValueError("ablation report provenance drift")
    if not _exact_json_equal(value.get("capture_contract"), paired.CAPTURE_CONTRACT):
        raise ValueError("ablation capture contract drift")
    construction = _mapping(value.get("policy_construction"), "ablation policy construction")
    if not _exact_json_equal(construction, inputs.provenance["hybrid_policy_construction"]):
        raise ValueError("ablation policy construction drift")
    contract = runtime_contract
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
    if set(initial) != {"identical_across_all_five_policies", "sha256"}:
        raise ValueError("ablation initial actor observation schema mismatch")
    if (
        initial["identical_across_all_five_policies"] is not True
        or len(set(trace_initials.values())) != 1
        or _sha256(initial.get("sha256"), "ablation initial actor observation")
        != next(iter(trace_initials.values()))
    ):
        raise ValueError("ablation initial actor observation drift")

    expected_endpoints = _mapping(
        runtime_contract.get("expected_endpoint_reproductions"), "ablation expected endpoints"
    )
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

    pairwise = _mapping(value.get("pairwise_comparisons"), "ablation pairwise comparisons")
    recomputed_pairwise = _rebuild_pairwise_comparisons(
        expected_pairwise_order=list(runtime_contract["comparison"]["pairwise_order"]),
        pairwise_comparisons=pairwise,
        traces=traces,
        layout=source_layout,
        runtime_contract=runtime_contract,
    )

    attribution = _mapping(value.get("attribution_summary"), "ablation attribution summary")
    if not _exact_json_equal(
        attribution,
        _attribution_summary(traces, recomputed_pairwise, source_layout, runtime_contract),
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
    boundary = _line_search_boundary_receipt(simulator_constructed=None)
    expected = {
        "schema_version",
        "kind",
        "ablation_contract_sha256",
        "source_pair_trace_sha256",
        "source_delta_ablation_report_sha256",
        "source_delta_ablation_report_provenance_snapshot_sha256",
        "failure_stage",
        "parent_run_dir",
        "preexecution_provenance_snapshot_sha256",
        "error",
        "simulator_ablation_complete",
        *boundary,
    }
    partial_optional = {"partial_scalar_evidence"}
    if set(value) not in (expected, expected | partial_optional):
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
    if (
        _sha256(
            value.get("source_delta_ablation_report_sha256"),
            "ablation failure source delta report hash",
        )
        != DELTA_ABLATION_REPORT_SHA256
    ):
        raise ValueError("ablation failure source delta report hash mismatch")
    if (
        _sha256(
            value.get("source_delta_ablation_report_provenance_snapshot_sha256"),
            "ablation failure source delta report provenance",
        )
        != DELTA_ABLATION_REPORT_PROVENANCE_SNAPSHOT_SHA256
    ):
        raise ValueError("ablation failure source delta provenance mismatch")
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
            "alpha_numerator": POLICY_ALPHAS[name][0],
            "alpha_denominator": POLICY_ALPHAS[name][1],
            "policy_state_sha256": POLICY_SHA256[name],
        }
        for name in POLICY_ORDER
    }


def _legacy_load_ablation_contract(repository_root: str | Path | None = None) -> Mapping[str, Any]:
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    path = paired._regular_file(root / CONTRACT_RELATIVE_PATH, "line-search ablation contract")
    if sha256_file(path) != CONTRACT_SHA256:
        raise ValueError("line-search ablation contract SHA256 mismatch")
    contract = _strict_json(path, "line-search ablation contract")
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("purpose") != "diagnostic_only_symmetric_line_search_between_decoder_module14_and_module16"
    ):
        raise ValueError("line-search ablation contract identity mismatch")

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
        raise ValueError("line-search ablation parent binding mismatch")

    source = _mapping(contract.get("source_pair_trace"), "source pair trace")
    if source != {
        "linux_path": SOURCE_PAIR_TRACE_LINUX_PATH,
        "sha256": SOURCE_PAIR_TRACE_SHA256,
        "kind": paired.TRACE_KIND,
        "provenance_snapshot_sha256": SOURCE_PAIR_PROVENANCE_SHA256,
        "contract_sha256": paired.TRACE_CONTRACT_SHA256,
        "validated_comparison_exact": True,
        "utility_sha256": "f05e53a0eaaafd53fe718b6f530cee01be163ba8496642a04c1db9ee65f21104",
        "cli_sha256": "e07f51fabc49c37da137c0739fd93966cc43aea235e001cef04264c5bf878ad8",
        "test_sha256": "23043064d260d7544a90029813a748659381814412d97d2849b56cc30dfbc7bf",
    }:
        raise ValueError("source pair trace binding mismatch")

    delta_report = _mapping(contract.get("source_delta_ablation_report"), "line-search source ablation report")
    if (
        delta_report.get("linux_path") != DELTA_ABLATION_REPORT_LINUX_PATH
        or delta_report.get("sha256") != DELTA_ABLATION_REPORT_SHA256
        or delta_report.get("provenance_snapshot_sha256") != DELTA_ABLATION_REPORT_PROVENANCE_SNAPSHOT_SHA256
        or delta_report.get("config_sha256") != "542eb733f631349e0117f8a61d7ae6221ea13ab54fe8cb837a02745e106cec1e"
        or delta_report.get("utility_sha256") != "058094e74202118e4bb4579cb17d5c1d865ba323090aedacc8f528fdb5a7d935"
        or delta_report.get("cli_sha256") != "7938f3b7f616dda41d09a8a0bea6e8ea5cb64f9271415b28c7259d831911e53c"
        or delta_report.get("test_sha256") != "7fa0691a42b822508c7298f38388f6cb2890607169de6b3368d47dacd2a5b1b7"
    ):
        raise ValueError("line-search source ablation report binding mismatch")

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
        raise ValueError("line-search policy construction mismatch")

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
        raise ValueError("line-search endpoint contract mismatch")

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
        raise ValueError("line-search execution contract mismatch")

    comparison = _mapping(contract.get("comparison"), "ablation comparison")
    if (
        comparison.get("pairwise_count") != len(PAIRWISE_COMPARISON_ORDER)
        or comparison.get("pairwise_order") != list(PAIRWISE_COMPARISON_ORDER)
        or comparison.get("reward_attribution_uses_preterminal_common_q9_only") is not True
        or comparison.get("global_preterminal_line_search_reward_summary") is not True
        or comparison.get("global_reward_factorial_q9_first") != 9
        or comparison.get("global_reward_factorial_q9_last_rule") != "min(all terminal_q9)-1"
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
        raise ValueError("line-search comparison/boundary/output mismatch")
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
        "kind": "g1_true23_full_support_delta_line_search_executed_sources_v1",
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


def _verify_hybrid_policies(
    checkpoints: Mapping[int, Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = {}
    for name in POLICY_ORDER:
        state = construct_hybrid_policy_state(checkpoints, name, contract)
        observed_sha256 = inspect_true23_policy_state(
            {"policy_state_dict": state},
            reference_profile="released_low_latency_step1_0p02s",
        )
        observed_sha256 = _sha256(observed_sha256, f"line-search {name} policy hash")
        receipt[name] = {
            "alpha_numerator": POLICY_ALPHAS[name][0],
            "alpha_denominator": POLICY_ALPHAS[name][1],
            "policy_state_sha256": POLICY_SHA256[name],
            "tensor_count": len(state),
            "fully_cloned_no_checkpoint_alias": True,
        }
        if observed_sha256 != POLICY_SHA256[name]:
            raise ValueError(f"line-search {name} policy construction hash drift")
        del state
    return {
        "changed_tensor_names": list(UPDATED_TENSORS),
        "all_other_policy_tensors_identical": True,
        "policies": receipt,
    }


def _legacy_current_provenance_snapshot(inputs: AblationInputs) -> str:
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


def _legacy_resolve_ablation_inputs(repository_root: Path, parent_run_dir: Path) -> AblationInputs:
    root = repository_root.expanduser().resolve(strict=True)
    contract = _legacy_load_ablation_contract(root)
    run_dir = parent_run_dir.expanduser().resolve(strict=True)
    expected_run = Path(contract["parent_run"]["linux_path"]).expanduser().resolve(strict=True)
    if run_dir != expected_run or run_dir.is_symlink():
        raise ValueError("line-search ablation parent run mismatch")
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
    provenance["snapshot_sha256"] = _legacy_current_provenance_snapshot(provisional)
    return provisional


def _legacy_ablation_preflight(repository_root: Path, parent_run_dir: Path) -> dict[str, Any]:
    try:
        inputs = resolve_ablation_inputs(repository_root, parent_run_dir)
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PREFLIGHT_KIND,
            "ready": True,
            "parent_run_dir": str(inputs.run_dir),
            "source_pair_trace": str(inputs.source_trace_path),
            "provenance": inputs.provenance,
            **_line_search_boundary_receipt(simulator_constructed=False, include_simulator_transitions=False),
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PREFLIGHT_KIND,
            "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            **_line_search_boundary_receipt(simulator_constructed=False, include_simulator_transitions=False),
        }


def _load_live_policy(
    actor: Any,
    policy_state: Mapping[str, torch.Tensor],
) -> str:
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
    return observed


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


def _policy_construction_policies(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    if "policy_construction" in contract:
        specs = _mapping(
            _mapping(contract["policy_construction"], "line-search policy construction").get("policies"),
            "line-search policy construction specs",
        )
        return specs
    if "policies" in contract:
        return _mapping(contract["policies"], "line-search policy specs")
    raise ValueError("line-search policy construction specs missing")


def _json_deep_copy(value: Any) -> Any:
    if type(value) is dict:
        return {str(key): _json_deep_copy(item) for key, item in value.items()}
    if type(value) is list:
        return [_json_deep_copy(item) for item in value]
    if type(value) is tuple:
        return [_json_deep_copy(item) for item in value]
    return value


def _runtime_trace_contract(contract: Mapping[str, Any], construction: Mapping[str, Any]) -> dict[str, Any]:
    source_contract = _json_deep_copy(_mapping(contract, "line-search runtime contract source"))
    construction_specs = _mapping(
        construction.get("policies"),
        "line-search runtime policy construction",
    )
    source_specs = _policy_construction_policies(source_contract)
    if set(source_specs) != set(POLICY_ORDER) or set(construction_specs) != set(POLICY_ORDER):
        raise ValueError("line-search runtime policy construction set mismatch")
    endpoint_contract = _mapping(
        source_contract.get("expected_endpoint_reproductions"),
        "line-search runtime expected endpoint reproductions",
    )
    runtime_policies: dict[str, Any] = {}
    for name in POLICY_ORDER:
        source_spec = _mapping(
            source_specs[name],
            f"line-search runtime source policy spec {name}",
        )
        construction_spec = _mapping(
            construction_specs[name],
            f"line-search runtime construction policy spec {name}",
        )
        runtime_hash = _sha256(
            construction_spec.get("policy_state_sha256"),
            f"line-search runtime policy hash {name}",
        )
        expected_hash = _policy_expected_state_sha256(
            source_spec,
            context=f"line-search runtime policy expected {name}",
        )
        if expected_hash != POLICY_SHA256[name]:
            raise ValueError("line-search runtime policy hash mismatch")
        if runtime_hash != expected_hash:
            raise ValueError("line-search runtime policy hash mismatch with source contract")
        if (
            source_spec.get("alpha_numerator") != POLICY_ALPHAS[name][0]
            or source_spec.get("alpha_denominator") != POLICY_ALPHAS[name][1]
        ):
            raise ValueError("line-search runtime policy interpolation coefficients mismatch")
        runtime_spec = dict(source_spec)
        runtime_spec["policy_state_sha256"] = runtime_hash
        runtime_policies[name] = runtime_spec

    endpoint_baseline = _policy_expected_state_sha256(
        _mapping(endpoint_contract.get("baseline"), "line-search baseline endpoint contract"),
        context="line-search runtime baseline endpoint hash",
    )
    endpoint_full = _policy_expected_state_sha256(
        _mapping(endpoint_contract.get("full"), "line-search full endpoint contract"),
        context="line-search runtime full endpoint hash",
    )
    if runtime_policies["baseline"]["policy_state_sha256"] != endpoint_baseline:
        raise ValueError("line-search baseline endpoint hash mismatch")
    if runtime_policies["full"]["policy_state_sha256"] != endpoint_full:
        raise ValueError("line-search full endpoint hash mismatch")
    _validate_comparison_pairwise_order(
        _mapping(source_contract.get("comparison"), "line-search runtime comparison"),
        context="line-search runtime",
    )

    construction_block = _mapping(
        source_contract["policy_construction"], "line-search runtime policy construction metadata"
    )
    construction_block["policies"] = runtime_policies
    source_contract["policy_construction"] = construction_block
    return source_contract


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
        "alpha_numerator",
        "alpha_denominator",
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
    spec = _mapping(_policy_construction_policies(contract), "line-search trace policy specs").get(policy_name)
    if spec is None or type(spec) is not dict:
        raise ValueError("ablation trace policy spec missing")
    expected_policy_hash = _policy_expected_state_sha256(
        spec,
        context="line-search ablation policy expected hash",
    )
    observed_policy_hash = _sha256(trace.get("policy_state_sha256"), "ablation policy state hash")
    if expected_policy_hash != observed_policy_hash:
        raise ValueError("ablation trace policy hash mismatch")
    if (
        type(trace.get("alpha_numerator")) is not int
        or type(trace.get("alpha_denominator")) is not int
        or type(trace.get("evaluation_seed")) is not int
        or trace.get("alpha_numerator") != spec["alpha_numerator"]
        or trace.get("alpha_denominator") != spec["alpha_denominator"]
        or trace.get("evaluation_seed") != paired.FIXED_SEED
        or trace.get("controller") != "deterministic_actor_mean"
    ):
        raise ValueError("ablation trace policy identity mismatch")
    _sha256(trace.get("initial_actor_observation_sha256"), "ablation initial observation")
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
    if trace.get("terminal_q9") != max(series["q9"]):
        raise ValueError("ablation trace terminal_q9 mismatch")
    if len(q9s) != completed:
        raise ValueError("ablation trace q9 length mismatch")
    if any(row != [] for row in series["termination_names"][:-1]):
        raise ValueError("ablation trace extra terminal evidence before episode end")
    _terminal_body_summary(trace)


def validate_endpoint_reproduction(
    trace: Mapping[str, Any], expected: Mapping[str, Any], policy_name: str
) -> dict[str, Any]:
    required_fields = {
        "completed_transitions",
        "terminal_q9",
        "termination_names",
        "terminal_worst_ee_body_name",
        "expected_policy_state_sha256",
        "source_trace_episode_return",
        "historical_return",
    }
    if set(expected) != required_fields:
        raise ValueError("ablation endpoint expected schema mismatch")
    historical = _mapping(expected.get("historical_return"), "ablation endpoint historical return")
    if (
        type(historical.get("episode_return")) is not float
        or not math.isfinite(historical.get("episode_return"))
        or historical.get("structural_gate") is not False
    ):
        raise ValueError("ablation endpoint historical return schema mismatch")
    terminal = _terminal_body_summary(trace)
    if terminal["worst_body_name"] != expected.get("terminal_worst_ee_body_name"):
        raise ValueError("ablation endpoint wrong terminal worst body")
    expected_hash = _policy_expected_state_sha256(
        expected,
        context=f"line-search {policy_name} expected endpoint hash",
    )
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
    observed_hash = _sha256(trace["policy_state_sha256"], "line-search endpoint observed policy hash")
    if observed_hash != expected_hash:
        raise EndpointReproductionError(
            f"{policy_name} endpoint policy mismatch",
            {
                "stage": f"{policy_name}_endpoint_policy_reproduction",
                "policy_name": policy_name,
                "observed_policy_state_sha256": observed_hash,
                "expected_policy_state_sha256": expected_hash,
            },
        )
    observed = float(trace["episode_return"])
    source = float(expected["source_trace_episode_return"])
    return {
        "policy_name": policy_name,
        "observed_episode_return": observed,
        "source_trace_episode_return": source,
        "historical_return": historical,
        "delta_observed_minus_source_trace": observed - source,
        "exact_match": observed == source,
        "gate_applied": False,
        "reason": "endpoint_policy_and_terminal_reproducibility_structural_check",
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
    policy_specs = _policy_construction_policies(contract)
    policy_spec = _mapping(policy_specs, "line-search policy specs")[policy_name]
    policy_hash = inspect_true23_policy_state(
        {"policy_state_dict": policy.export_true23_policy_state()},
        reference_profile="released_low_latency_step1_0p02s",
    )
    expected_hash = _sha256(
        policy_spec["policy_state_sha256"],
        "line-search policy expected hash",
    )
    if expected_hash != policy_hash:
        raise ValueError("ablation live policy hash mismatch")
    layout, reward = _trace_layout(raw_env, contract)
    observations = wrapped_env.get_observations()
    initial_observation = _clone_initial_actor_observation(observations)
    recorder = paired._RewardComputeTraceRecorder(raw_env, reward)
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
                if not bool(torch.all(torch.abs(actor_action) < 10.0)):
                    raise ValueError("ablation raw_native_action strict clip exceeded")
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
    if not frames or not frames[-1]["termination_names"]:
        raise ValueError("ablation policy did not terminate within exact 510 transitions")
    spec = _mapping(_policy_construction_policies(contract), "line-search trace policy spec")[policy_name]
    trace = {
        "policy_name": policy_name,
        "alpha_numerator": spec["alpha_numerator"],
        "alpha_denominator": spec["alpha_denominator"],
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


def _policy_alpha_fraction(name: str) -> Fraction:
    numerator, denominator = POLICY_ALPHAS[name]
    if type(numerator) is not int or type(denominator) is not int:
        raise ValueError("line-search policy alpha must be integer ratio")
    return Fraction(numerator, denominator)


def _strict_float_scalar_map(
    value: Mapping[str, Any], *, expected: Sequence[str], context: str
) -> dict[str, float]:
    mapping = _mapping(value, context)
    if set(mapping) != set(expected):
        raise ValueError(f"{context} keys must be exact policy order")
    parsed: dict[str, float] = {}
    for key in expected:
        sample = mapping[key]
        if type(sample) is not float or not math.isfinite(sample):
            raise ValueError(f"{context} value for {key} must be finite float")
        parsed[key] = sample
    return parsed


def _strict_int_scalar_map(value: Mapping[str, Any], *, expected: Sequence[str], context: str) -> dict[str, int]:
    mapping = _mapping(value, context)
    if set(mapping) != set(expected):
        raise ValueError(f"{context} keys must be exact policy order")
    parsed: dict[str, int] = {}
    for key in expected:
        sample = mapping[key]
        if type(sample) is not int:
            raise ValueError(f"{context} value for {key} must be strict int")
        parsed[key] = sample
    return parsed


def _rational_evidence(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def signed_finite_differences(samples: Mapping[str, Any]) -> dict[str, Any]:
    values = _strict_float_scalar_map(
        samples, expected=POLICY_ORDER, context="line-search finite difference samples"
    )
    positive_order = ("baseline", "alpha_plus_0_25", "alpha_plus_0_5", "full")
    first_differences: list[dict[str, Any]] = []
    for left_name, right_name in zip(positive_order[:-1], positive_order[1:], strict=True):
        left_value = values[left_name]
        right_value = values[right_name]
        left_alpha = _policy_alpha_fraction(left_name)
        right_alpha = _policy_alpha_fraction(right_name)
        delta_alpha = right_alpha - left_alpha
        if delta_alpha == 0:
            raise ValueError("line-search finite difference alpha spacing invalid")
        signed_delta = right_value - left_value
        slope = signed_delta / float(delta_alpha)
        if not math.isfinite(slope):
            raise ValueError("line-search finite difference slope must be finite")
        first_differences.append(
            {
                "left_policy": left_name,
                "right_policy": right_name,
                "left_alpha": _rational_evidence(left_alpha),
                "right_alpha": _rational_evidence(right_alpha),
                "delta_alpha": _rational_evidence(delta_alpha),
                "signed_delta": signed_delta,
                "slope": slope,
            }
        )
    if len(first_differences) != 3:
        raise ValueError("line-search finite differences must include three positive-direction diffs")
    second_differences: list[dict[str, Any]] = []
    for left_difference, right_difference in zip(first_differences[:-1], first_differences[1:], strict=True):
        right_minus_left = right_difference["slope"] - left_difference["slope"]
        if not math.isfinite(right_minus_left):
            raise ValueError("line-search second finite difference must be finite")
        second_differences.append(
            {
                "left_slope_between": f"{left_difference['left_policy']}_vs_{left_difference['right_policy']}",
                "right_slope_between": f"{right_difference['left_policy']}_vs_{right_difference['right_policy']}",
                "right_slope_minus_left_slope": right_minus_left,
            }
        )
    if len(second_differences) != 2:
        raise ValueError("line-search finite differences must include two second-order diffs")
    return {
        "policy_order": list(POLICY_ORDER),
        "policy_alphas": [
            {"policy_name": name, "alpha": _rational_evidence(_policy_alpha_fraction(name))}
            for name in POLICY_ORDER
        ],
        "positive_direction_policy_order": list(positive_order),
        "policy_values": values,
        "first_differences": first_differences,
        "second_differences": second_differences,
    }


def _direction_classification_from_alpha_series(
    samples: Mapping[str, float],
) -> dict[str, Any]:
    finite_differences = signed_finite_differences(samples)
    return {
        "classification": "inconclusive",
        "reason": "survival_evidence_required",
        "policy_order": finite_differences["policy_order"],
        "alphas": [
            float(_policy_alpha_fraction(name))
            for name in ("baseline", "alpha_plus_0_25", "alpha_plus_0_5", "full")
        ],
        "values": [float(samples[name]) for name in ("baseline", "alpha_plus_0_25", "alpha_plus_0_5", "full")],
        "first_differences": finite_differences["first_differences"],
        "second_differences": finite_differences["second_differences"],
        "finite_differences": finite_differences,
    }


def classify_direction_vs_magnitude(
    completed_transitions_by_policy: Mapping[str, Any],
    aligned_reward_by_policy: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    completed = _strict_int_scalar_map(
        completed_transitions_by_policy,
        expected=POLICY_ORDER,
        context="line-search completed transitions by policy",
    )
    aligned = _strict_float_scalar_map(
        aligned_reward_by_policy,
        expected=POLICY_ORDER,
        context="line-search aligned preterminal reward by policy",
    )
    comparison = _mapping(contract.get("comparison"), "line-search contract comparison")
    reward_atol = comparison.get("reward_delta_atol")
    if type(reward_atol) is not float or not math.isfinite(reward_atol):
        raise ValueError("line-search reward_atol must be finite float")
    line_search_classification = _mapping(
        comparison.get("line_search_classification"),
        "line-search direction-vs-magnitude contract",
    )
    negative_probe = line_search_classification.get("negative_probe")
    if negative_probe is not None and negative_probe != "alpha_minus_0_25":
        raise ValueError("line-search negative probe contract mismatch")
    positive_small_probes = line_search_classification.get(
        "positive_small_probes", ["alpha_plus_0_25", "alpha_plus_0_5"]
    )
    if set(positive_small_probes) != {"alpha_plus_0_25", "alpha_plus_0_5"}:
        raise ValueError("line-search positive-small probe contract mismatch")
    if any(type(name) is not str for name in positive_small_probes):
        raise ValueError("line-search positive-small probe contract malformed")
    baseline_survival = completed["baseline"]
    baseline_reward = aligned["baseline"]
    per_probe: dict[str, Any] = {}
    for policy in POLICY_ORDER:
        survival_delta = completed[policy] - baseline_survival
        reward_delta = aligned[policy] - baseline_reward
        improves = (
            survival_delta >= 0
            and reward_delta >= -reward_atol
            and (survival_delta > 0 or reward_delta > reward_atol)
        )
        regresses = (
            survival_delta <= 0 and reward_delta <= 0 and (survival_delta < 0 or reward_delta < -reward_atol)
        )
        per_probe[policy] = {
            "survival_delta": survival_delta,
            "reward_delta": reward_delta,
            "survival_nonworse": survival_delta >= 0,
            "reward_not_worse_than_tolerance": reward_delta >= -reward_atol,
            "strict_reward_benefit": reward_delta > reward_atol,
            "strict_survival_benefit": survival_delta > 0,
            "improves": improves,
            "regresses": regresses,
        }
    negative_improves = per_probe["alpha_minus_0_25"]["improves"]
    positive_small_improves = any(per_probe[name]["improves"] for name in positive_small_probes)
    full_regresses = per_probe["full"]["regresses"]
    if full_regresses and positive_small_improves and not negative_improves:
        classification = "excessive_update_magnitude"
        reason = "positive_small_updates_improve_and_negative_does_not"
    elif full_regresses and negative_improves and not positive_small_improves:
        classification = "harmful_gradient_direction"
        reason = "negative_improves_and_no_positive_small_improve"
    elif full_regresses:
        classification = "inconclusive"
        reason = "positive_small_and_negative_overlap"
    else:
        classification = "inconclusive"
        reason = "no_full_regression"
    return {
        "policy_order": list(POLICY_ORDER),
        "reward_atol": reward_atol,
        "per_probe": per_probe,
        "full_regression": {
            "survival_delta": per_probe["full"]["survival_delta"],
            "reward_delta": per_probe["full"]["reward_delta"],
            "regresses": full_regresses,
            "survival_delta_max": float(0.0),
            "global_preterminal_aligned_reward_delta_max": 0.0,
        },
        "positive_small_improves": positive_small_improves,
        "negative_improves": negative_improves,
        "negative_probe": "alpha_minus_0_25",
        "positive_small_probes": list(positive_small_probes),
        "classification": classification,
        "reason": reason,
        "diagnostic_only": True,
        "candidate_selected": False,
    }


def _global_preterminal_reward_summary(
    traces: Mapping[str, Mapping[str, Any]],
    layout: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if set(traces) != set(POLICY_ORDER):
        raise ValueError("ablation global preterminal reward summary policy set mismatch")
    q9_last = min(int(traces[name]["terminal_q9"]) for name in POLICY_ORDER) - 1
    q9s = list(range(9, q9_last + 1))
    if not q9s:
        raise ValueError("ablation lacks global preterminal reward support")
    reward_names = [term["name"] for term in layout["reward_terms"]]
    if any(type(term_name) is not str or not term_name for term_name in reward_names):
        raise ValueError("ablation reward names malformed")
    comparison = _mapping(contract.get("comparison"), "line-search comparison") if contract is not None else {}
    barrier_term_name = comparison.get("barrier_term_name")
    worst_term_name = comparison.get("worst_ee_term_name")
    if contract is not None and (
        type(barrier_term_name) is not str
        or type(worst_term_name) is not str
        or barrier_term_name not in reward_names
        or worst_term_name not in reward_names
    ):
        raise ValueError("line-search target reward term names missing in layout")
    sums: dict[str, list[float]] = {}
    for policy_name in POLICY_ORDER:
        series = traces[policy_name]["series"]
        index = {q9: row for row, q9 in enumerate(series["q9"])}
        if any(q9 not in index for q9 in q9s):
            raise ValueError("ablation global reward support is not shared by all policies")
        sums[policy_name] = _reward_sums(series, [index[q9] for q9 in q9s])
    total = {name: sum(sums[name]) for name in POLICY_ORDER}
    by_term: dict[str, Any] = {}
    for term_index, term_name in enumerate(reward_names):
        values_by_policy = {name: float(sums[name][term_index]) for name in POLICY_ORDER}
        by_term[term_name] = {
            "policy_values": values_by_policy,
            "signed_effect_vs_baseline": {
                name: values_by_policy[name] - values_by_policy["baseline"] for name in POLICY_ORDER
            },
            "finite_differences": signed_finite_differences(values_by_policy),
        }
    signed_total_effect = {name: total[name] - total["baseline"] for name in POLICY_ORDER}
    return {
        "policy_order": list(POLICY_ORDER),
        "q9_first": q9s[0],
        "q9_last": q9s[-1],
        "transition_count": len(q9s),
        "terminal_frames_excluded": True,
        "same_q9_support_for_all_policies": True,
        "total_weighted_reward": {name: float(total[name]) for name in POLICY_ORDER},
        "signed_total_effect_vs_baseline": signed_total_effect,
        "finite_differences_total_weighted_reward": signed_finite_differences(total),
        "weighted_reward_by_term": by_term,
        "barrier_term_name": barrier_term_name,
        "worst_ee_term_name": worst_term_name,
        "target_reward_terms": {
            "barrier": by_term[barrier_term_name] if barrier_term_name is not None else None,
            "worst_ee": by_term[worst_term_name] if worst_term_name is not None else None,
        },
    }


def _attribution_summary(
    traces: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
    layout: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    pairwise = _validate_comparison_pairwise_order(
        _mapping(contract.get("comparison"), "line-search attribution comparison"),
        context="line-search attribution",
    )
    pairwise_order = [f"{left}_vs_{right}" for left, right in pairwise]
    pairwise_comparisons = _mapping(comparisons, "ablation pairwise comparisons")
    if set(pairwise_comparisons) != set(pairwise_order):
        raise ValueError("ablation attribution summary comparison key mismatch")
    if list(pairwise_comparisons) != pairwise_order:
        raise ValueError("ablation attribution summary comparison order mismatch")
    for label in pairwise_order:
        comparison = _mapping(pairwise_comparisons[label], f"ablation attribution summary pairwise {label}")
        expected_left, expected_right = PAIRWISE_POLICY_NAMES_BY_LABEL[label]
        if comparison.get("left_policy") != expected_left or comparison.get("right_policy") != expected_right:
            raise ValueError("ablation attribution summary pairwise policy mapping drift")
    global_summary = _global_preterminal_reward_summary(traces, layout, contract)
    completed_transitions_by_policy = _strict_int_scalar_map(
        {name: traces[name]["completed_transitions"] for name in POLICY_ORDER},
        expected=POLICY_ORDER,
        context="line-search completed transitions by policy",
    )
    terminal_q9_by_policy = _strict_int_scalar_map(
        {name: traces[name]["terminal_q9"] for name in POLICY_ORDER},
        expected=POLICY_ORDER,
        context="line-search terminal_q9_by_policy",
    )
    classification = classify_direction_vs_magnitude(
        completed_transitions_by_policy,
        global_summary["total_weighted_reward"],
        contract,
    )
    return {
        "global_preterminal_line_search_reward_summary": global_summary,
        "pairwise_comparisons_validated_exact": True,
        "validated_pairwise_comparison_order": pairwise_order,
        "preterminal_common_reward_delta_vs_baseline": {
            **{"baseline": 0.0},
            **{
                name: global_summary["signed_total_effect_vs_baseline"][name]
                for name in POLICY_ORDER
                if name != "baseline"
            },
        },
        "completed_transitions_by_policy": completed_transitions_by_policy,
        "terminal_q9_by_policy": terminal_q9_by_policy,
        "completed_transitions_finite_differences": signed_finite_differences(
            {name: float(value) for name, value in completed_transitions_by_policy.items()}
        ),
        "terminal_q9_finite_differences": signed_finite_differences(
            {name: float(value) for name, value in terminal_q9_by_policy.items()}
        ),
        "global_preterminal_total_weighted_reward_finite_differences": global_summary[
            "finite_differences_total_weighted_reward"
        ],
        "barrier_term_effect_vs_baseline": (
            global_summary["target_reward_terms"]["barrier"]["signed_effect_vs_baseline"]
            if global_summary["target_reward_terms"]["barrier"] is not None
            else {}
        ),
        "worst_ee_term_effect_vs_baseline": (
            global_summary["target_reward_terms"]["worst_ee"]["signed_effect_vs_baseline"]
            if global_summary["target_reward_terms"]["worst_ee"] is not None
            else {}
        ),
        "absolute_episode_returns_diagnostic_only": {
            name: traces[name]["episode_return"] for name in POLICY_ORDER
        },
        "absolute_episode_return_authorizes_attribution_or_candidate": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "candidate_selected": False,
        "direction_vs_magnitude": classification,
    }


def execute_update_delta_ablation(repository_root: Path, parent_run_dir: Path) -> dict[str, Any]:
    """Run five fresh deterministic evaluations; never optimize or train."""

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
    runtime_contract = _runtime_trace_contract(
        inputs.contract,
        _mapping(inputs.provenance.get("hybrid_policy_construction"), "line-search runtime policy construction"),
    )
    traces: dict[str, Mapping[str, Any]] = {}
    endpoint_diagnostics: dict[str, Mapping[str, Any]] = {}
    common_layout: Mapping[str, Any] | None = None
    common_audit: Mapping[str, Any] | None = None
    common_prime: Mapping[str, Any] | None = None
    construction = _mapping(
        inputs.provenance.get("hybrid_policy_construction"), "line-search runtime policy construction"
    )
    runtime_policy_specs = _mapping(
        construction.get("policies"), "line-search runtime policy construction policies"
    )
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
        policy_spec = _mapping(
            runtime_policy_specs.get(policy_name), f"line-search runtime policy construction {policy_name}"
        )
        expected_hash = _sha256(
            policy_spec["policy_state_sha256"],
            f"line-search {policy_name} runtime policy expected hash",
        )
        live_hash = _load_live_policy(actor, state)
        if expected_hash != live_hash:
            raise RuntimeError(f"{policy_name} policy runtime hash mismatch")
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
                contract=runtime_contract,
            )
            after_hash = inspect_true23_policy_state(
                {"policy_state_dict": actor.export_true23_policy_state()},
                reference_profile="released_low_latency_step1_0p02s",
            )
            if after_hash != expected_hash:
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
    pairwise = _build_pairwise_comparisons(
        traces=traces,
        layout=common_layout,
        contract=runtime_contract,
    )
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
        "source_delta_ablation_report": {
            "path": str(inputs.source_ablation_report_path),
            "sha256": SOURCE_ABLATION_REPORT_SHA256,
            "provenance_snapshot_sha256": SOURCE_ABLATION_REPORT_PROVENANCE_SHA256,
            "validated_publication_snapshot": SOURCE_ABLATION_REPORT_PROVENANCE_SHA256,
            "validated_publication": True,
        },
        "provenance": inputs.provenance,
        "capture_contract": paired.CAPTURE_CONTRACT,
        "policy_construction": inputs.provenance["hybrid_policy_construction"],
        "initial_actor_observation": {
            "identical_across_all_five_policies": True,
            "sha256": next(iter(initial_hashes.values())),
        },
        "layout": common_layout,
        "task_audit": common_audit,
        "prime": common_prime,
        "traces": traces,
        "endpoint_reproduction": endpoint_diagnostics,
        "pairwise_comparisons": pairwise,
        "attribution_summary": _attribution_summary(traces, pairwise, common_layout, runtime_contract),
        **_line_search_boundary_receipt(simulator_constructed=True),
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
    source_delta_ablation_report_sha256: str | None = None,
    source_delta_ablation_report_provenance_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    if type(failure_stage) is not str or not failure_stage:
        raise ValueError("failure stage must be non-empty string")
    if provenance_snapshot_sha256 is not None:
        _sha256(provenance_snapshot_sha256, "failure provenance snapshot")
    if parent_run_dir is not None and (type(parent_run_dir) is not str or not parent_run_dir):
        raise ValueError("failure parent run must be non-empty string")
    if source_delta_ablation_report_sha256 is None:
        source_delta_ablation_report_sha256 = DELTA_ABLATION_REPORT_SHA256
    if source_delta_ablation_report_provenance_snapshot_sha256 is None:
        source_delta_ablation_report_provenance_snapshot_sha256 = DELTA_ABLATION_REPORT_PROVENANCE_SNAPSHOT_SHA256
    _sha256(source_delta_ablation_report_sha256, "ablation failure source delta report")
    _sha256(
        source_delta_ablation_report_provenance_snapshot_sha256,
        "ablation failure source delta report provenance",
    )
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
        "source_delta_ablation_report_sha256": source_delta_ablation_report_sha256,
        "source_delta_ablation_report_provenance_snapshot_sha256": (
            source_delta_ablation_report_provenance_snapshot_sha256
        ),
        **_line_search_boundary_receipt(simulator_constructed=None),
    }
    partial = getattr(error, "partial_evidence", None)
    if partial is not None:
        paired._assert_scalar_evidence(partial)
        report["partial_scalar_evidence"] = dict(partial)
    paired.assert_publication_boundary(report)
    return report


def _load_strict_json(path: Path, context: str) -> Mapping[str, Any]:
    return _load_strict_json_from_bytes(
        paired._regular_file(path, context).read_bytes(),
        context,
    )


def _load_strict_json_from_bytes(payload: bytes, context: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, value in pairs:
            if type(key) is not str:
                raise ValueError(f"{context} contains non-string key")
            if key in seen:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            seen[key] = value
        return seen

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{context} contains non-finite token {token}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} contains invalid UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} contains malformed JSON") from error
    return _mapping(parsed, context)


def _read_and_validate_prior_failure_receipt(contract: Mapping[str, Any]) -> dict[str, Any]:
    contract_parent = _mapping(contract.get("parent_run"), "line-search prior failure receipt parent run")
    contract_parent_run_dir = contract_parent.get("linux_path")
    if type(contract_parent_run_dir) is not str or not contract_parent_run_dir:
        raise ValueError("line-search prior failure parent run directory missing")
    prior_failure_spec = _mapping(
        contract.get("prior_failure_receipt"), "line-search prior failure receipt specification"
    )
    if (
        set(prior_failure_spec)
        != {
            "path",
            "linux_path",
            "filename",
            "sha256",
            "size_bytes",
            "kind",
            "contract_sha256",
            "failure_stage",
            "error",
            "simulator_ablation_complete",
            "boundary_receipt",
            "byte_immutable",
            "trace_evidence_used",
            "discarded_simulation_evidence_reused",
        }
        or prior_failure_spec.get("path") != PRIOR_FAILURE_REPORT_LINUX_PATH
        or prior_failure_spec.get("linux_path") != PRIOR_FAILURE_REPORT_LINUX_PATH
        or prior_failure_spec.get("filename") != PRIOR_FAILURE_OUTPUT_FILENAME
        or _sha256(prior_failure_spec.get("sha256"), "line-search prior failure contract specification hash")
        != PRIOR_FAILURE_REPORT_SHA256
        or type(prior_failure_spec.get("size_bytes")) is not int
        or prior_failure_spec.get("size_bytes") != PRIOR_FAILURE_REPORT_SIZE_BYTES
        or prior_failure_spec.get("kind") != PRIOR_FAILURE_KIND
        or _sha256(
            prior_failure_spec.get("contract_sha256"),
            "line-search prior failure contract specification contract hash",
        )
        != PRIOR_FAILURE_CONTRACT_SHA256
        or prior_failure_spec.get("failure_stage") != PRIOR_FAILURE_STAGE
        or prior_failure_spec.get("simulator_ablation_complete") is not False
        or prior_failure_spec.get("byte_immutable") is not True
        or prior_failure_spec.get("trace_evidence_used") is not False
        or prior_failure_spec.get("discarded_simulation_evidence_reused") is not False
        or not _exact_json_equal(prior_failure_spec.get("error"), PRIOR_FAILURE_ERROR)
    ):
        raise ValueError("line-search prior failure receipt specification mismatch")

    prior_failure_path = paired._regular_file(
        Path(prior_failure_spec["linux_path"]), "line-search prior failure report"
    )
    if prior_failure_path != Path(PRIOR_FAILURE_REPORT_LINUX_PATH):
        raise ValueError("line-search prior failure report path mismatch")
    prior_failure_bytes = prior_failure_path.read_bytes()
    if len(prior_failure_bytes) != PRIOR_FAILURE_REPORT_SIZE_BYTES:
        raise ValueError("line-search prior failure report size mismatch")
    prior_failure_hash = hashlib.sha256(prior_failure_bytes).hexdigest()
    if prior_failure_hash != PRIOR_FAILURE_REPORT_SHA256:
        raise ValueError("line-search prior failure report hash mismatch")

    if Path(prior_failure_spec["path"]) != Path(PRIOR_FAILURE_REPORT_LINUX_PATH):
        raise ValueError("line-search prior failure report spec path mismatch")

    receipt = _load_strict_json_from_bytes(prior_failure_bytes, "line-search prior failure report")
    boundary_receipt = _line_search_boundary_receipt(simulator_constructed=None)
    if (
        set(receipt)
        != {"schema_version", "kind", "ablation_contract_sha256"}
        | set(boundary_receipt)
        | {
            "source_pair_trace_sha256",
            "source_delta_ablation_report_sha256",
            "source_delta_ablation_report_provenance_snapshot_sha256",
            "failure_stage",
            "parent_run_dir",
            "preexecution_provenance_snapshot_sha256",
            "error",
            "simulator_ablation_complete",
        }
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != SCHEMA_VERSION
        or receipt.get("kind") not in {PRIOR_FAILURE_KIND, FAILURE_KIND}
        or _sha256(receipt.get("ablation_contract_sha256"), "line-search prior failure contract hash")
        != PRIOR_FAILURE_CONTRACT_SHA256
        or _sha256(receipt.get("source_pair_trace_sha256"), "line-search prior failure source pair trace hash")
        != SOURCE_PAIR_TRACE_SHA256
        or _sha256(
            receipt.get("source_delta_ablation_report_sha256"),
            "line-search prior failure source delta report hash",
        )
        != SOURCE_ABLATION_REPORT_SHA256
        or _sha256(
            receipt.get("source_delta_ablation_report_provenance_snapshot_sha256"),
            "line-search prior failure source delta provenance",
        )
        != SOURCE_ABLATION_REPORT_PROVENANCE_SHA256
        or receipt.get("failure_stage") != PRIOR_FAILURE_STAGE
        or receipt.get("parent_run_dir") != contract_parent_run_dir
        or type(receipt.get("preexecution_provenance_snapshot_sha256")) is not str
        or type(receipt.get("error")) is not dict
        or type(receipt.get("simulator_ablation_complete")) is not bool
        or receipt["simulator_ablation_complete"] is not False
        or not _exact_json_equal(receipt.get("error"), PRIOR_FAILURE_ERROR)
    ):
        raise ValueError("line-search prior failure report schema mismatch")

    if not _exact_json_equal(receipt["parent_run_dir"], contract_parent_run_dir):
        raise ValueError("line-search prior failure parent run mismatch")
    _sha256(
        receipt["preexecution_provenance_snapshot_sha256"],
        "line-search prior failure preexecution provenance snapshot",
    )

    _assert_json_finite_scalars(receipt, context="line-search prior failure receipt")
    _validate_publication_receipt_boundary(receipt, simulator_constructed=None)
    paired.assert_publication_boundary(receipt)
    return {
        "path": str(prior_failure_path),
        "sha256": prior_failure_hash,
        "size_bytes": len(prior_failure_bytes),
        "kind": receipt["kind"],
        "contract_sha256": receipt["ablation_contract_sha256"],
        "failure_stage": receipt["failure_stage"],
        "error": dict(receipt["error"]),
        "validated": True,
        "byte_immutable": True,
        "trace_evidence_used": False,
        "discarded_simulation_evidence_reused": False,
    }


def load_line_search_contract(repository_root: str | Path | None = None) -> Mapping[str, Any]:
    root = (
        Path(repository_root).expanduser().resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    contract_path = root / CONTRACT_RELATIVE_PATH
    contract = _load_strict_json(contract_path, "line-search contract")
    if sha256_file(contract_path) != CONTRACT_SHA256:
        raise ValueError("line-search contract SHA256 mismatch")

    if (
        type(contract.get("schema_version")) is not int
        or contract["schema_version"] != SCHEMA_VERSION
        or contract.get("kind") != CONTRACT_KIND
        or contract.get("purpose") != "diagnostic_only_symmetric_line_search_between_decoder_module14_and_module16"
    ):
        raise ValueError("line-search contract identity mismatch")

    parent = _mapping(contract.get("parent_run"), "line-search parent run")
    parent_contract = paired.load_trace_contract(root)
    parent_reference = parent_contract["parent_run"]
    if (
        parent.get("linux_path") != parent_reference["linux_path"]
        or parent.get("directory_name") != parent_reference["directory_name"]
        or parent.get("artifact_sha256") != parent_reference["artifact_sha256"]
        or _sha256(parent.get("full_support_contract_sha256"), "line-search parent full support contract")
        != paired.FULL_SUPPORT_CONTRACT_SHA256
        or _sha256(parent.get("run_materials_sha256"), "line-search parent materials")
        != paired.RUN_MATERIALS_SHA256
        or _sha256(parent.get("rollout_evidence_sha256"), "line-search rollout evidence")
        != paired.ROLLOUT_EVIDENCE_SHA256
    ):
        raise ValueError("line-search parent run mismatch")

    source_pair = _mapping(contract.get("source_pair_trace"), "line-search source pair")
    if (
        source_pair.get("linux_path") != SOURCE_PAIR_TRACE_LINUX_PATH
        or source_pair.get("kind") != paired.TRACE_KIND
        or _sha256(source_pair.get("sha256"), "line-search source pair hash") != SOURCE_PAIR_TRACE_SHA256
        or _sha256(source_pair.get("provenance_snapshot_sha256"), "line-search source pair provenance")
        != SOURCE_PAIR_PROVENANCE_SHA256
        or source_pair.get("validated_comparison_exact") is not True
        or _sha256(source_pair.get("contract_sha256"), "line-search source pair contract")
        != paired.TRACE_CONTRACT_SHA256
    ):
        raise ValueError("line-search source pair binding mismatch")

    source_report = _mapping(contract.get("source_delta_ablation_report"), "line-search source ablation report")
    if (
        source_report.get("linux_path") != SOURCE_ABLATION_REPORT_LINUX_PATH
        or source_report.get("kind") != source_ablation.ABLATION_KIND
        or _sha256(source_report.get("sha256"), "line-search source report hash") != SOURCE_ABLATION_REPORT_SHA256
        or _sha256(source_report.get("provenance_snapshot_sha256"), "line-search source report provenance")
        != SOURCE_ABLATION_REPORT_PROVENANCE_SHA256
        or _sha256(source_report.get("contract_sha256"), "line-search source report contract")
        != source_ablation.CONTRACT_SHA256
        or _sha256(source_report.get("config_sha256"), "line-search source report config")
        != source_ablation.CONTRACT_SHA256
        or source_report.get("utility_sha256")
        != "058094e74202118e4bb4579cb17d5c1d865ba323090aedacc8f528fdb5a7d935"
        or source_report.get("cli_sha256") != "7938f3b7f616dda41d09a8a0bea6e8ea5cb64f9271415b28c7259d831911e53c"
        or source_report.get("test_sha256") != "7fa0691a42b822508c7298f38388f6cb2890607169de6b3368d47dacd2a5b1b7"
    ):
        raise ValueError("line-search source ablation report binding mismatch")

    prior_failure = _mapping(contract.get("prior_failure_receipt"), "line-search prior failure receipt")
    if (
        set(prior_failure)
        != {
            "path",
            "linux_path",
            "filename",
            "sha256",
            "size_bytes",
            "kind",
            "contract_sha256",
            "failure_stage",
            "error",
            "simulator_ablation_complete",
            "boundary_receipt",
            "byte_immutable",
            "trace_evidence_used",
            "discarded_simulation_evidence_reused",
        }
        or type(prior_failure.get("path")) is not str
        or type(prior_failure.get("linux_path")) is not str
        or type(prior_failure.get("filename")) is not str
        or prior_failure.get("path") != PRIOR_FAILURE_REPORT_LINUX_PATH
        or prior_failure.get("linux_path") != PRIOR_FAILURE_REPORT_LINUX_PATH
        or prior_failure.get("filename") != PRIOR_FAILURE_OUTPUT_FILENAME
        or prior_failure.get("failure_stage") != PRIOR_FAILURE_STAGE
        or prior_failure.get("simulator_ablation_complete") is not False
        or prior_failure.get("byte_immutable") is not True
        or prior_failure.get("trace_evidence_used") is not False
        or prior_failure.get("discarded_simulation_evidence_reused") is not False
        or _sha256(prior_failure.get("sha256"), "line-search prior failure report hash")
        != PRIOR_FAILURE_REPORT_SHA256
        or type(prior_failure.get("size_bytes")) is not int
        or prior_failure.get("size_bytes") != PRIOR_FAILURE_REPORT_SIZE_BYTES
        or prior_failure.get("kind") != PRIOR_FAILURE_KIND
        or _sha256(prior_failure.get("contract_sha256"), "line-search prior failure contract")
        != PRIOR_FAILURE_CONTRACT_SHA256
        or not _exact_json_equal(prior_failure.get("error"), PRIOR_FAILURE_ERROR)
        or not _exact_json_equal(
            prior_failure.get("boundary_receipt"), _line_search_boundary_receipt(simulator_constructed=None)
        )
    ):
        raise ValueError("line-search prior failure receipt mismatch")

    provenance_requirements = _mapping(
        contract.get("provenance_requirements"),
        "line-search provenance requirements",
    )
    source_ablation_requirements = _mapping(
        provenance_requirements.get("source_ablation_report"),
        "line-search source ablation requirements",
    )
    source_pair_requirements = _mapping(
        provenance_requirements.get("source_pair_trace"), "line-search source pair requirements"
    )
    parent_run_requirements = _mapping(
        provenance_requirements.get("parent_run"), "line-search parent run requirements"
    )
    checkpoint_requirements = _mapping(
        provenance_requirements.get("checkpoint_sha256"), "line-search checkpoint requirements"
    )
    endpoint_requirements = _mapping(
        provenance_requirements.get("endpoint_policy_state_sha256"),
        "line-search endpoint policy requirements",
    )
    if (
        provenance_requirements.get("recursive_source_ablation_publication_revalidation") is not True
        or provenance_requirements.get("recursive_source_ablation_live_input_revalidation") is not True
        or provenance_requirements.get("source_pair_task_audit_layout_prime_exact_match") is not True
        or provenance_requirements.get("inherit_physical_inputs_via_validated_source_ablation_provenance")
        is not True
        or provenance_requirements.get("inherit_runtime_inputs_via_validated_source_ablation_provenance")
        is not True
        or provenance_requirements.get("inherit_bound_inputs_via_validated_source_ablation_provenance") is not True
        or provenance_requirements.get("rehash_before_after_each_evaluation") is not True
        or provenance_requirements.get("rehash_immediately_before_publication") is not True
        or set(source_ablation_requirements) != {"sha256", "snapshot_sha256"}
        or _sha256(source_ablation_requirements.get("sha256"), "line-search source ablation requirement hash")
        != SOURCE_ABLATION_REPORT_SHA256
        or _sha256(
            source_ablation_requirements.get("snapshot_sha256"),
            "line-search source ablation requirement snapshot",
        )
        != SOURCE_ABLATION_REPORT_PROVENANCE_SHA256
        or set(source_pair_requirements) != {"sha256", "snapshot_sha256"}
        or _sha256(
            source_pair_requirements.get("sha256"),
            "line-search source pair requirement hash",
        )
        != SOURCE_PAIR_TRACE_SHA256
        or _sha256(
            source_pair_requirements.get("snapshot_sha256"),
            "line-search source pair requirement snapshot",
        )
        != SOURCE_PAIR_PROVENANCE_SHA256
        or set(parent_run_requirements)
        != {"run_materials_sha256", "full_support_result_sha256", "rollout_evidence_sha256"}
        or _sha256(
            parent_run_requirements.get("run_materials_sha256"), "line-search parent run materials requirement"
        )
        != paired.RUN_MATERIALS_SHA256
        or _sha256(
            parent_run_requirements.get("full_support_result_sha256"),
            "line-search parent full support result requirement",
        )
        != parent["artifact_sha256"].get("full_support_result.json")
        or _sha256(
            parent_run_requirements.get("rollout_evidence_sha256"),
            "line-search parent rollout evidence requirement",
        )
        != paired.ROLLOUT_EVIDENCE_SHA256
        or set(checkpoint_requirements)
        != {
            "checkpoints/sonic_task_space_full_support_model_0.pt",
            "checkpoints/sonic_task_space_full_support_model_1.pt",
        }
        or _sha256(
            checkpoint_requirements.get("checkpoints/sonic_task_space_full_support_model_0.pt"),
            "line-search source-pair model0 checkpoint",
        )
        != parent["artifact_sha256"].get("checkpoints/sonic_task_space_full_support_model_0.pt")
        or _sha256(
            checkpoint_requirements.get("checkpoints/sonic_task_space_full_support_model_1.pt"),
            "line-search source-pair model1 checkpoint",
        )
        != parent["artifact_sha256"].get("checkpoints/sonic_task_space_full_support_model_1.pt")
        or set(endpoint_requirements) != {"baseline", "full"}
        or _sha256(endpoint_requirements.get("baseline"), "line-search baseline endpoint policy requirement")
        != POLICY_SHA256["baseline"]
        or _sha256(endpoint_requirements.get("full"), "line-search full endpoint policy requirement")
        != POLICY_SHA256["full"]
    ):
        raise ValueError("line-search provenance requirements mismatch")

    policy_construction = _mapping(contract.get("policy_construction"), "line-search policy construction")
    canonical_algorithm = _mapping(
        policy_construction.get("canonical_algorithm"), "line-search canonical algorithm"
    )
    if (
        policy_construction.get("policy_order") != list(POLICY_ORDER)
        or policy_construction.get("module14_tensor_names") != list(MODULE14_TENSORS)
        or policy_construction.get("module16_tensor_names") != list(MODULE16_TENSORS)
        or policy_construction.get("all_other_policy_tensors_identical_between_model0_and_model1") is not True
        or policy_construction.get("hybrids_are_in_memory_only") is not True
        or policy_construction.get("intermediate_policy_hash_binding")
        != "immutable_contract_pinned_exact_all_five_policy_state_sha256"
        or canonical_algorithm.get("model0_keyset_sorted") is not True
        or canonical_algorithm.get("source_tensor_spec")
        != {"device": "cpu", "dtype": "torch.float32", "finite": True, "contiguous": True}
        or canonical_algorithm.get("result_tensor_spec")
        != {"device": "cpu", "dtype": "torch.float32", "finite": True, "contiguous": True, "clone": True}
        or canonical_algorithm.get("result_no_aliases_to_source") is not True
        or canonical_algorithm.get("frozen_source_byte_identity") is not True
        or type(canonical_algorithm.get("updated_tensor_names")) is not list
        or canonical_algorithm.get("updated_tensor_names") != list(UPDATED_TENSORS)
        or canonical_algorithm.get("alpha_conversion", {}).get("numerator_key") != "alpha_numerator"
        or canonical_algorithm.get("alpha_conversion", {}).get("denominator_key") != "alpha_denominator"
        or canonical_algorithm.get("alpha_conversion", {}).get("dtype") != "torch.float32"
        or canonical_algorithm.get("endpoints") != {"0": "model0_clone", "1": "model1_clone"}
        or canonical_algorithm.get("intermediate_ops") != ["torch.sub", "torch.mul", "torch.add"]
        or canonical_algorithm.get("no_source_mutation") is not True
    ):
        raise ValueError("line-search canonical construction mismatch")

    policies = _mapping(policy_construction.get("policies"), "line-search policy specs")
    if set(policies) != set(POLICY_ORDER):
        raise ValueError("line-search policy spec set mismatch")
    for name, (numerator, denominator) in POLICY_ALPHAS.items():
        policy_spec = _mapping(policies.get(name), f"line-search policy spec {name}")
        if (
            type(policy_spec.get("alpha_numerator")) is not int
            or type(policy_spec.get("alpha_denominator")) is not int
            or policy_spec.get("alpha_numerator") != numerator
            or policy_spec.get("alpha_denominator") != denominator
        ):
            raise ValueError("line-search policy alpha mismatch")
        policy_expected = _policy_expected_state_sha256(
            policy_spec,
            context=f"line-search expected policy {name} hash",
        )
        if policy_expected != POLICY_SHA256[name]:
            raise ValueError("line-search expected policy hash mismatch")

    action_path = _mapping(contract.get("action_path"), "line-search action path")
    if (
        action_path.get("actor_class") != "gear_sonic.trl.mjlab.true23_actor:True23SonicActorModel"
        or action_path.get("genuine_sonic") is not True
        or action_path.get("tokenizer_observation_dim") != 268
        or action_path.get("routing_index_dim") != 1
        or action_path.get("encoder_input_dim") != 267
        or action_path.get("teleop_encoder_dim") != 267
        or action_path.get("token_dim") != 64
        or action_path.get("h10_policy_dim") != 930
        or action_path.get("decoder_input_dim") != 994
        or action_path.get("policy_history_length") != 10
        or action_path.get("policy_layout") != "term-major_padded_il29"
        or action_path.get("raw_native_action_dim") != 23
        or action_path.get("decoder_concat_input_order") != ["fsq_token", "h10_policy"]
        or action_path.get("external_transform_kind") != SAFE_TARGET_TRANSFORM_KIND
        or action_path.get("external_transform_application_count") != SAFE_TARGET_TRANSFORM_APPLICATION_COUNT
        or action_path.get("raw") != {"strict_abs_less_than": SAFE_TARGET_RAW_ACTION_CLIP}
        or action_path.get("safe_target_application_pipeline")
        != [
            "raw_native23",
            "candidate_hardware23",
            "safe_native23",
            "final_hardware23",
        ]
        or action_path.get("chain")
        != [
            "tokenizer_observation",
            "routing_index",
            "encoder_input",
            "h10_policy",
            "decoder_input",
            "raw_native_action",
            "safe_native_action",
            "final_target_hardware",
        ]
        or len(action_path.get("chain", ())) != 8
        or action_path.get("output_order") != ["native23", "safe-native23", "hardware23"]
        or action_path.get("wrapper_clip") is not None
    ):
        raise ValueError("line-search action path mismatch")

    execution = _mapping(contract.get("execution"), "line-search execution")
    if (
        execution.get("evaluation_seed") != paired.FIXED_SEED
        or execution.get("num_envs") != 1
        or execution.get("policy_count") != 5
        or execution.get("device") != paired.DEVICE
        or execution.get("controller") != "deterministic_actor_mean"
        or execution.get("max_transitions") != 510
        or execution.get("initial_q9") != 9
        or execution.get("expected_max_episode_length") != 510
        or execution.get("wrapper_clip_actions") is not None
        or execution.get("capture_contract") != paired.CAPTURE_CONTRACT
        or execution.get("fresh_environment_per_policy") is not True
        or execution.get("fresh_actor_instance_per_policy") is not True
        or execution.get("task_inheritance") is not True
        or execution.get("prime_inheritance") is not True
        or execution.get("layout_inheritance") is not True
        or execution.get("allowed_terminal_terms") != ["anchor_ori", "anchor_pos", "ee_body_pos", "time_out"]
        or execution.get("all_absolute_episode_returns_are_diagnostic_only") is not True
        or execution.get("safe_target_transform_kind") != SAFE_TARGET_TRANSFORM_KIND
        or execution.get("safe_target_transform_application_count") != SAFE_TARGET_TRANSFORM_APPLICATION_COUNT
        or execution.get("raw_native_action_strict_abs_max") != SAFE_TARGET_RAW_ACTION_CLIP
        or execution.get("initial_actor_observation_clone_only_before_first_action") is not True
        or execution.get("initial_actor_observation_host_materialization_after_episode") is not True
        or execution.get("require_identical_initial_actor_observation_sha256") is not True
        or execution.get("full_provenance_rehash_before_and_after_every_evaluation") is not True
        or execution.get("full_provenance_rehash_immediately_before_publication") is not True
    ):
        raise ValueError("line-search execution mismatch")

    comparison = _mapping(contract.get("comparison"), "line-search comparison")
    _validate_comparison_pairwise_order(comparison, context="line-search")
    if (
        comparison.get("line_search_classification", {}).get("enum")
        != [
            "harmful_gradient_direction",
            "excessive_update_magnitude",
            "inconclusive",
        ]
        or comparison.get("line_search_classification", {}).get("diagnostic_only") is not True
        or comparison.get("line_search_classification", {}).get("negative_probe") != "alpha_minus_0_25"
        or comparison.get("line_search_classification", {}).get("positive_small_probes")
        != ["alpha_plus_0_25", "alpha_plus_0_5"]
        or comparison.get("line_search_classification", {}).get("baseline") != "baseline"
        or comparison.get("line_search_classification", {}).get("full") != "full"
        or comparison.get("line_search_classification", {}).get("improvement")
        != {
            "survival_delta_min": 0,
            "global_preterminal_aligned_reward_delta_min": -comparison.get("reward_delta_atol"),
            "at_least_one_strict_benefit": True,
        }
        or comparison.get("line_search_classification", {}).get("full_regression")
        != {
            "survival_delta_max": 0,
            "global_preterminal_aligned_reward_delta_max": 0.0,
            "at_least_one_strict_regression": True,
        }
        or comparison.get("line_search_classification", {}).get("excessive_if")
        != [
            "positive_small_improvement",
            "no_negative_improvement",
            "full_regression",
        ]
        or comparison.get("line_search_classification", {}).get("harmful_if")
        != [
            "negative_improvement",
            "no_positive_small_improvement",
            "full_regression",
        ]
        or comparison.get("line_search_classification", {}).get("ambiguous_if")
        != ["positive_small_and_negative_overlap"]
        or comparison.get("line_search_signed_finite_differences") is not True
        or comparison.get("global_preterminal_line_search_reward_summary") is not True
        or comparison.get("reward_attribution_uses_preterminal_common_q9_only") is not True
        or comparison.get("absolute_episode_return_never_authorizes_attribution_or_candidate") is not True
        or "global_reward_factorial_q9_last" in comparison
        or comparison.get("line_search_schedule_policies")
        != ["baseline", "alpha_plus_0_25", "alpha_plus_0_5", "full"]
        or comparison.get("action_linf_thresholds") != [1e-6, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
        or comparison.get("ee_error_delta_thresholds_m") != [0.0001, 0.001, 0.01, 0.025, 0.05]
        or comparison.get("reward_delta_atol") != 1e-6
        or comparison.get("contact_force_delta_atol_n") != 0.001
        or comparison.get("landing_force_delta_atol_n") != 0.001
        or comparison.get("ee_body_pos_threshold_m") != 0.25
        or comparison.get("barrier_term_name") != "right_wrist_prethreshold_barrier"
        or comparison.get("worst_ee_term_name") != "worst_ee_z_normalized_squared"
        or comparison.get("global_reward_factorial_q9_first") != 9
        or comparison.get("global_reward_factorial_q9_last_rule") != "min(all terminal_q9)-1"
        or comparison.get("diagnostic_only_pareto_logic") != "inconclusive"
        or comparison.get("absolute_episode_return_authorizes_candidate") is not False
    ):
        raise ValueError("line-search comparison mismatch")

    expected_endpoints = _mapping(
        contract.get("expected_endpoint_reproductions"), "line-search expected endpoints"
    )
    if set(expected_endpoints) != {"baseline", "full"}:
        raise ValueError("line-search expected endpoint set mismatch")
    for name, expected in expected_endpoints.items():
        endpoint = {
            "completed_transitions": expected.get("completed_transitions"),
            "terminal_q9": expected.get("terminal_q9"),
            "termination_names": expected.get("termination_names"),
            "terminal_worst_ee_body_name": expected.get("terminal_worst_ee_body_name"),
            "expected_policy_state_sha256": expected.get("expected_policy_state_sha256"),
            "source_trace_episode_return": expected.get("source_trace_episode_return"),
            "historical_return": expected.get("historical_return"),
        }
        if set(endpoint) != set(expected):
            raise ValueError("line-search expected endpoint schema mismatch")
        if (
            tuple(endpoint["termination_names"]) != ("ee_body_pos",)
            or endpoint.get("completed_transitions") not in {155, 151}
            or type(endpoint.get("terminal_q9")) is not int
            or endpoint.get("terminal_worst_ee_body_name") != "right_wrist_roll_rubber_hand"
            or type(endpoint.get("source_trace_episode_return")) is not float
            or not math.isfinite(endpoint.get("source_trace_episode_return"))
            or not isinstance(endpoint.get("historical_return"), Mapping)
            or endpoint.get("historical_return").get("structural_gate") is not False
        ):
            raise ValueError("line-search expected endpoint mismatch")
        if name == "baseline":
            if endpoint["completed_transitions"] != 155 or endpoint["terminal_q9"] != 163:
                raise ValueError("line-search expected baseline endpoint mismatch")
            if (
                _sha256(endpoint["expected_policy_state_sha256"], "line-search baseline policy hash")
                != POLICY_SHA256["baseline"]
            ):
                raise ValueError("line-search expected baseline hash mismatch")
        else:
            if endpoint["completed_transitions"] != 151 or endpoint["terminal_q9"] != 159:
                raise ValueError("line-search expected full endpoint mismatch")
            if (
                _sha256(endpoint["expected_policy_state_sha256"], "line-search full policy hash")
                != POLICY_SHA256["full"]
            ):
                raise ValueError("line-search expected full policy hash mismatch")

    if contract.get("boundaries") != _line_search_boundary_receipt(simulator_constructed=None):
        raise ValueError("line-search boundaries mismatch")

    output = _mapping(contract.get("output"), "line-search output")
    if (
        output.get("path") != LINE_SEARCH_REPORT_LINUX_PATH
        or output.get("linux_path") != LINE_SEARCH_REPORT_LINUX_PATH
        or output.get("filename") != OUTPUT_FILENAME
        or output.get("exclusive_create") is not True
        or output.get("atomic") is not True
        or output.get("no_replace") is not True
    ):
        raise ValueError("line-search output mismatch")

    reward_terms = _mapping(parent_contract, "line-search parent contract").get("expected_reward_terms")
    if contract.get("expected_reward_terms") != reward_terms:
        raise ValueError("line-search expected reward terms mismatch")

    sealed_sources = _mapping(contract.get("sealed_sources"), "line-search sealed sources")
    for logical, expected in sealed_sources.items():
        if _sha256(expected, f"line-search sealed source expected {logical}") != sha256_file(
            paired._regular_file(root / logical, f"line-search sealed source {logical}")
        ):
            raise ValueError(f"line-search sealed source mismatch for {logical}")

    return contract


load_ablation_contract = load_line_search_contract


def _strict_cpu_float32_contiguous_tensor(tensor: Any, *, context: str) -> torch.Tensor:
    if type(tensor) is not torch.Tensor:
        raise ValueError(f"{context} must be a torch Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{context} must be on cpu")
    if tensor.dtype is not torch.float32:
        raise ValueError(f"{context} must be torch.float32")
    if not tensor.is_contiguous():
        raise ValueError(f"{context} must be contiguous")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{context} must be finite")
    return tensor


def _tensor_profile(tensor: torch.Tensor, *, context: str) -> dict[str, Any]:
    tensor_cpu = _strict_cpu_float32_contiguous_tensor(
        tensor.detach(),
        context=context,
    )
    if tensor_cpu.device.type != "cpu":
        raise ValueError(f"{context} tensor must be CPU")
    if tensor_cpu.dtype is not torch.float32:
        raise ValueError(f"{context} tensor must be float32")
    if not tensor_cpu.is_contiguous():
        raise ValueError(f"{context} tensor must be contiguous")
    if not bool(torch.isfinite(tensor_cpu).all()):
        raise ValueError(f"{context} tensor must be finite")
    return {
        "shape": tuple(int(i) for i in tensor_cpu.shape),
        "dtype": str(tensor_cpu.dtype),
        "device": str(tensor_cpu.device),
        "contiguous": bool(tensor_cpu.is_contiguous()),
        "strides": tuple(int(i) for i in tensor_cpu.stride()),
        "data_ptr": int(tensor_cpu.data_ptr()),
        "bytes": tensor_cpu.numpy().tobytes(),
    }


def construct_interpolated_policy_state(
    checkpoints: Mapping[int, Mapping[str, Any]],
    policy_name: str,
    contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    if policy_name not in POLICY_ORDER:
        raise ValueError(f"unknown policy name {policy_name!r} for line-search interpolation")
    policy0 = {
        name: _strict_cpu_float32_contiguous_tensor(tensor, context=f"line-search model0 source {name}")
        for name, tensor in _policy_mapping(checkpoints[0], "model0 policy").items()
    }
    policy1 = {
        name: _strict_cpu_float32_contiguous_tensor(tensor, context=f"line-search model1 source {name}")
        for name, tensor in _policy_mapping(checkpoints[1], "model1 policy").items()
    }
    policy0_names = set(policy0)
    policy1_names = set(policy1)
    if policy0_names != policy1_names:
        raise ValueError("line-search policy namespace drift between checkpoints")
    specs = _mapping(contract["policy_construction"]["policies"], "line-search policy specs")
    spec = _mapping(specs[policy_name], f"line-search policy spec {policy_name}")
    numerator = spec["alpha_numerator"]
    denominator = spec["alpha_denominator"]
    expected_numerator, expected_denominator = POLICY_ALPHAS[policy_name]
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or numerator != expected_numerator
        or denominator != expected_denominator
    ):
        raise ValueError("line-search interpolation alpha mismatch")
    if denominator <= 0:
        raise ValueError("line-search interpolation denominator invalid")

    changed = [name for name in policy0 if not torch.equal(policy0[name], policy1[name])]
    if set(changed) != set(UPDATED_TENSORS):
        raise ValueError("line-search interpolation changed tensor set mismatch")

    result: dict[str, torch.Tensor] = {}
    before_model0 = {
        name: _tensor_profile(policy0[name], context=f"line-search model0 source {name}") for name in policy0
    }
    before_model1 = {
        name: _tensor_profile(policy1[name], context=f"line-search model1 source {name}") for name in policy1
    }
    ratio = torch.tensor(numerator, dtype=torch.float32) / torch.tensor(
        denominator,
        dtype=torch.float32,
    )
    for name in sorted(policy0):
        tensor0 = policy0[name]
        tensor1 = policy1[name]
        if tuple(int(i) for i in tensor0.shape) != tuple(int(i) for i in tensor1.shape):
            raise ValueError(f"line-search policy tensor shape mismatch for {name}")
        if name in UPDATED_TENSORS:
            if numerator == 0:
                candidate = tensor0.clone()
            elif numerator == denominator:
                candidate = tensor1.clone()
            else:
                delta = torch.sub(tensor1, tensor0)
                scaled = torch.mul(delta, ratio)
                candidate = torch.add(tensor0, scaled)
        else:
            candidate = tensor0.clone()
        candidate = candidate.contiguous().clone()
        if not bool(torch.isfinite(candidate).all()):
            raise ValueError(f"line-search policy tensor {name} must remain finite")
        if tuple(int(i) for i in candidate.shape) != tuple(int(i) for i in tensor0.shape):
            raise ValueError(f"line-search policy tensor {name} shape drift")
        if candidate.data_ptr() == policy0[name].data_ptr() or candidate.data_ptr() == policy1[name].data_ptr():
            raise ValueError(f"line-search policy tensor {name} aliases source tensor")
        result[name] = candidate

    if len({tensor.data_ptr() for tensor in result.values()}) != len(result):
        raise ValueError("line-search interpolation output alias detected")
    for name in policy0:
        if _tensor_profile(policy0[name], context=f"line-search model0 source {name}") != before_model0[name]:
            raise ValueError("line-search source tensor mutated during interpolation")
        if _tensor_profile(policy1[name], context=f"line-search model1 source {name}") != before_model1[name]:
            raise ValueError("line-search source tensor mutated during interpolation")

    observed = inspect_true23_policy_state(
        {"policy_state_dict": result},
        reference_profile="released_low_latency_step1_0p02s",
    )
    observed = _sha256(observed, f"line-search {policy_name} contract policy hash")
    expected = _policy_expected_state_sha256(
        spec,
        context=f"line-search {policy_name} expected hash",
    )
    if expected != POLICY_SHA256[policy_name]:
        raise ValueError(f"line-search {policy_name} expected hash mismatch")
    if observed != expected:
        raise ValueError(f"line-search {policy_name} contract policy hash mismatch")
    return result


construct_hybrid_policy_state = construct_interpolated_policy_state


def _verify_interpolated_policies(
    checkpoints: Mapping[int, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    source0 = {
        name: _strict_cpu_float32_contiguous_tensor(
            tensor,
            context=f"line-search model0 source tensor {name}",
        )
        for name, tensor in _policy_mapping(checkpoints[0], "line-search model0 source checkpoint").items()
    }
    source1 = {
        name: _strict_cpu_float32_contiguous_tensor(
            tensor,
            context=f"line-search model1 source tensor {name}",
        )
        for name, tensor in _policy_mapping(checkpoints[1], "line-search model1 source checkpoint").items()
    }
    source0_keyset = set(source0)
    if source0_keyset != set(source1):
        raise ValueError("line-search policy namespace drift between checkpoints")
    source0_before = {
        name: _tensor_profile(value, context=f"line-search model0 source {name}")
        for name, value in source0.items()
    }
    source1_before = {
        name: _tensor_profile(value, context=f"line-search model1 source {name}")
        for name, value in source1.items()
    }
    source_pointers = {int(value.data_ptr()) for value in source0.values()} | {
        int(value.data_ptr()) for value in source1.values()
    }
    frozen_tensors_byte_identical_to_model0 = all(
        source0_profile["bytes"] == source1_profile["bytes"]
        for name, source0_profile in source0_before.items()
        for source1_profile in (source1_before.get(name),)
        if name not in UPDATED_TENSORS
    )
    if not frozen_tensors_byte_identical_to_model0:
        raise ValueError("line-search source tensor byte drift for frozen tensors")

    policies = {name: construct_interpolated_policy_state(checkpoints, name, contract) for name in POLICY_ORDER}
    source0_after = {
        name: _tensor_profile(value, context=f"line-search model0 source {name}")
        for name, value in source0.items()
    }
    source1_after = {
        name: _tensor_profile(value, context=f"line-search model1 source {name}")
        for name, value in source1.items()
    }
    source_checkpoints_unchanged = source0_before == source0_after and source1_before == source1_after

    policy_pointers: set[int] = set()
    verified: dict[str, dict[str, Any]] = {}
    for policy_name in POLICY_ORDER:
        policy_state = policies[policy_name]
        policy_keyset = set(policy_state)
        if policy_keyset != source0_keyset:
            raise ValueError("line-search interpolated policy keyset mismatch")
        for tensor_name in policy_state:
            if tensor_name not in source0:
                raise ValueError("line-search interpolated policy keyset mismatch")

        for tensor_name in policy_keyset:
            checked = _strict_cpu_float32_contiguous_tensor(
                policy_state[tensor_name],
                context=f"line-search interpolated policy {policy_name} tensor {tensor_name}",
            )
            data_ptr = int(checked.data_ptr())
            if data_ptr in source_pointers:
                raise ValueError("line-search interpolated tensor aliases checkpoint tensor")
            if data_ptr in policy_pointers:
                raise ValueError("line-search interpolation states alias across policies")
            policy_pointers.add(data_ptr)

            if tensor_name not in UPDATED_TENSORS:
                if not torch.equal(checked, source0[tensor_name]):
                    raise ValueError("line-search interpolated policy frozen tensor changed")
                if checked.numpy().tobytes() != source0_before[tensor_name]["bytes"]:
                    raise ValueError("line-search interpolated policy frozen tensor byte drift")

        spec = _mapping(
            contract["policy_construction"]["policies"][policy_name],
            f"line-search policy spec {policy_name}",
        )
        changed = [name for name in policy_keyset if not torch.equal(policy_state[name], source0[name])]
        if not set(changed).issubset(UPDATED_TENSORS):
            raise ValueError("line-search interpolated policy changed undeclared tensor")

        observed = inspect_true23_policy_state(
            {"policy_state_dict": policy_state},
            reference_profile="released_low_latency_step1_0p02s",
        )
        observed = _sha256(observed, f"line-search {policy_name} observed policy hash")
        expected = _policy_expected_state_sha256(
            spec,
            context=f"line-search expected {policy_name} policy hash",
        )
        if expected != POLICY_SHA256[policy_name]:
            raise ValueError(f"line-search {policy_name} expected policy hash mismatch")
        if observed != expected:
            raise ValueError(f"line-search {policy_name} observed policy hash mismatch")
        verified[policy_name] = {
            "alpha_numerator": spec["alpha_numerator"],
            "alpha_denominator": spec["alpha_denominator"],
            "policy_state_sha256": observed,
            "tensor_count": len(policy_state),
            "exact_model0_keyset": True,
            "frozen_tensors_byte_identical_to_model0": frozen_tensors_byte_identical_to_model0,
            "only_declared_tensors_may_differ": True,
            "cpu_float32_contiguous_finite": True,
            "no_source_or_cross_policy_alias": True,
            "source_checkpoints_unchanged": source_checkpoints_unchanged,
        }
    return verified


def _source_ablation_report(root: Path) -> tuple[dict[str, Any], Path]:
    contract = load_line_search_contract(root)
    source_report_path = Path(contract["source_delta_ablation_report"]["linux_path"]).expanduser()
    if source_report_path != Path(SOURCE_ABLATION_REPORT_LINUX_PATH):
        raise ValueError("line-search source ablation report path mismatch")
    if (
        _sha256(sha256_file(source_report_path), "line-search source ablation report hash")
        != SOURCE_ABLATION_REPORT_SHA256
    ):
        raise ValueError("line-search source ablation report hash mismatch")
    source_report = _mapping(
        _load_strict_json(source_report_path, "source ablation report"), "source ablation report"
    )
    if source_report.get("kind") != source_ablation.ABLATION_KIND:
        raise ValueError("line-search source ablation report kind mismatch")
    if (
        _sha256(source_report["provenance"]["snapshot_sha256"], "source ablation report snapshot")
        != SOURCE_ABLATION_REPORT_PROVENANCE_SHA256
    ):
        raise ValueError("line-search source ablation provenance snapshot mismatch")
    return source_report, source_report_path


def resolve_line_search_inputs(repository_root: Path, parent_run_dir: Path) -> LineSearchInputs:
    root = Path(repository_root).expanduser().resolve(strict=True)
    contract = load_line_search_contract(root)
    expected_parent = Path(contract["parent_run"]["linux_path"]).expanduser().resolve(strict=True)
    normalized_parent = Path(parent_run_dir).expanduser().resolve(strict=True)
    if normalized_parent != expected_parent or normalized_parent.is_symlink():
        raise ValueError("line-search parent run mismatch")
    prior_failure_receipt = _read_and_validate_prior_failure_receipt(contract)
    source_ablation_inputs = source_ablation.resolve_ablation_inputs(root, expected_parent)
    source_report, source_report_path = _source_ablation_report(root)
    source_report_snapshot = source_ablation.validate_publication_provenance(root, expected_parent, source_report)
    if source_report_snapshot != _sha256(
        source_report["provenance"]["snapshot_sha256"], "line-search source report snapshot"
    ):
        raise ValueError("line-search source report snapshot mismatch")
    source_report_snapshot = _sha256(
        source_report["provenance"]["snapshot_sha256"], "line-search source report snapshot"
    )
    if source_report_snapshot != SOURCE_ABLATION_REPORT_PROVENANCE_SHA256:
        raise ValueError("line-search source report provenance snapshot invalid")
    source_report_inputs_snapshot = _sha256(
        source_ablation._current_provenance_snapshot(source_ablation_inputs),
        "line-search source ablation live snapshot",
    )
    if source_report_inputs_snapshot != source_report_snapshot:
        raise ValueError("line-search source ablation provenance snapshot mismatch")
    if source_ablation_inputs.run_dir != expected_parent:
        raise ValueError("line-search source ablation input run mismatch")

    if not source_ablation_inputs.source_trace_path.exists():
        raise ValueError("line-search source ablation trace missing")
    if (
        _sha256(sha256_file(source_ablation_inputs.source_trace_path), "line-search source pair trace snapshot")
        != SOURCE_PAIR_TRACE_SHA256
    ):
        raise ValueError("line-search source pair trace hash mismatch")

    if source_ablation_inputs.provenance["snapshot_sha256"] != source_report_snapshot:
        raise ValueError("line-search source ablation provenance snapshot mismatch")

    checkpoints = paired._load_and_validate_checkpoints(source_ablation_inputs.parent_inputs)
    cpu_checkpoints: dict[int, dict[str, Any]] = {}
    for index, checkpoint in checkpoints.items():
        policy = {
            name: _strict_cpu_float32_contiguous_tensor(
                tensor,
                context=f"line-search source checkpoint model {index} tensor {name}",
            )
            for name, tensor in _policy_mapping(
                checkpoint,
                f"line-search source checkpoint model {index}",
            ).items()
        }
        cpu_checkpoints[index] = {"policy_state_dict": {name: tensor.clone() for name, tensor in policy.items()}}

    construction = {
        "policy_order": list(POLICY_ORDER),
        "module14_tensor_names": list(MODULE14_TENSORS),
        "module16_tensor_names": list(MODULE16_TENSORS),
        "all_other_policy_tensors_identical_between_model0_and_model1": True,
        "hybrids_are_in_memory_only": True,
        "updated_tensor_names": list(UPDATED_TENSORS),
        "policies": {},
    }
    runtime_policy_info = _verify_interpolated_policies(cpu_checkpoints, contract)
    for name in POLICY_ORDER:
        construction["policies"][name] = dict(runtime_policy_info[name])
    dynamic_sources = _source_binding(root)
    provenance = {
        "line_search_contract_sha256": CONTRACT_SHA256,
        "parent_run_dir": str(expected_parent),
        "parent_run_source_trace_sha256": SOURCE_PAIR_TRACE_SHA256,
        "source_ablation_report_path": str(source_report_path),
        "source_ablation_report_sha256": SOURCE_ABLATION_REPORT_SHA256,
        "prior_failure_receipt": prior_failure_receipt,
        "source_ablation_report_provenance_sha256": SOURCE_ABLATION_REPORT_PROVENANCE_SHA256,
        "source_ablation_publication_snapshot_sha256": source_report_snapshot,
        "source_ablation_report_inputs_snapshot_sha256": source_report_inputs_snapshot,
        "source_ablation_provenance": source_ablation_inputs.provenance,
        "source_ablation_report": source_report,
        "source_ablation_inputs": {
            "snapshot_sha256": source_ablation_inputs.provenance["snapshot_sha256"],
        },
        "hybrid_policy_construction": construction,
        "executed_line_search_sources": _source_binding(root),
        "line_search_runtime_sources": dynamic_sources,
    }

    provisional = LineSearchInputs(
        repository_root=root,
        run_dir=expected_parent,
        contract=contract,
        source_ablation_inputs=source_ablation_inputs,
        source_ablation_report_path=source_report_path,
        source_ablation_report=source_report,
        provenance=provenance,
        source_pair_trace=source_ablation_inputs.source_trace
        if hasattr(source_ablation_inputs, "source_trace")
        else None,
        parent_inputs=source_ablation_inputs.parent_inputs,
        source_trace_path=source_ablation_inputs.source_trace_path,
        source_trace=_mapping(source_ablation_inputs.source_trace, "line-search source ablation source trace"),
    )
    provenance["snapshot_sha256"] = _current_provenance_snapshot(provisional)
    return provisional


def _resolve_line_search_checkpoints_contract(
    construction: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    expected_policy_specs = _policy_construction_policies(contract)
    if type(construction.get("policy_order")) is not list or construction.get("policy_order") != list(
        POLICY_ORDER
    ):
        raise ValueError("line-search construction policy order mismatch")
    if (
        type(construction.get("module14_tensor_names")) is not list
        or tuple(construction["module14_tensor_names"]) != MODULE14_TENSORS
    ):
        raise ValueError("line-search construction module14 tensor names mismatch")
    if (
        type(construction.get("module16_tensor_names")) is not list
        or tuple(construction["module16_tensor_names"]) != MODULE16_TENSORS
    ):
        raise ValueError("line-search construction module16 tensor names mismatch")
    if construction.get("all_other_policy_tensors_identical_between_model0_and_model1") is not True:
        raise ValueError("line-search construction all-other-tensors flag mismatch")
    if construction.get("hybrids_are_in_memory_only") is not True:
        raise ValueError("line-search construction in-memory flag mismatch")
    if type(construction.get("updated_tensor_names")) is not list or construction["updated_tensor_names"] != list(
        UPDATED_TENSORS
    ):
        raise ValueError("line-search construction updated tensors mismatch")

    policies = _mapping(construction.get("policies"), "line-search construction policies")
    if set(policies) != set(POLICY_ORDER):
        raise ValueError("line-search construction policy order mismatch")
    required_keys = {
        "alpha_numerator",
        "alpha_denominator",
        "policy_state_sha256",
        "tensor_count",
        "exact_model0_keyset",
        "frozen_tensors_byte_identical_to_model0",
        "only_declared_tensors_may_differ",
        "cpu_float32_contiguous_finite",
        "no_source_or_cross_policy_alias",
        "source_checkpoints_unchanged",
    }
    for name in POLICY_ORDER:
        policy_spec = _mapping(policies[name], f"line-search construction policy spec {name}")
        if set(policy_spec) != required_keys:
            raise ValueError("line-search construction policy spec keyset mismatch")
        if (
            type(policy_spec.get("policy_state_sha256")) is not str
            or type(policy_spec.get("tensor_count")) is not int
            or type(policy_spec.get("alpha_numerator")) is not int
            or type(policy_spec.get("alpha_denominator")) is not int
            or tuple((policy_spec["alpha_numerator"], policy_spec["alpha_denominator"])) != POLICY_ALPHAS[name]
            or policy_spec.get("exact_model0_keyset") is not True
            or policy_spec.get("frozen_tensors_byte_identical_to_model0") is not True
            or policy_spec.get("only_declared_tensors_may_differ") is not True
            or policy_spec.get("cpu_float32_contiguous_finite") is not True
            or policy_spec.get("no_source_or_cross_policy_alias") is not True
            or policy_spec.get("source_checkpoints_unchanged") is not True
        ):
            raise ValueError("line-search construction policy spec mismatch")
        policy_hash = _sha256(
            policy_spec.get("policy_state_sha256"),
            f"line-search construction policy hash {name}",
        )
        expected_hash = _policy_expected_state_sha256(
            _mapping(expected_policy_specs[name], f"line-search expected construction policy spec {name}"),
            context=f"line-search construction expected policy hash {name}",
        )
        if policy_hash != POLICY_SHA256[name] or policy_hash != expected_hash:
            raise ValueError("line-search construction policy hash mismatch")

    expected_endpoints = _mapping(
        contract.get("expected_endpoint_reproductions"),
        "line-search construction expected endpoints",
    )
    if set(expected_endpoints) != {"baseline", "full"}:
        raise ValueError("line-search construction expected endpoint key mismatch")
    for name in ("baseline", "full"):
        endpoint_spec = _mapping(
            expected_endpoints[name],
            f"line-search construction expected endpoint {name}",
        )
        endpoint_hash = _sha256(
            policies[name]["policy_state_sha256"],
            f"line-search construction endpoint binding {name}",
        )
        expected_hash = _policy_expected_state_sha256(
            endpoint_spec,
            context=f"line-search expected endpoint {name} hash",
        )
        if endpoint_hash != expected_hash:
            raise ValueError("line-search construction expected endpoint hash mismatch")


def _current_provenance_snapshot(inputs: LineSearchInputs) -> str:
    if inputs.parent_inputs is None:
        raise ValueError("line-search parent inputs missing")
    parent_snapshot = paired._current_provenance_snapshot(inputs.parent_inputs)
    if parent_snapshot != SOURCE_PAIR_PROVENANCE_SHA256:
        raise ValueError("line-search source pair provenance snapshot mismatch")
    prior_failure_receipt = _read_and_validate_prior_failure_receipt(inputs.contract)
    if not _exact_json_equal(prior_failure_receipt, inputs.provenance.get("prior_failure_receipt")):
        raise ValueError("line-search prior failure receipt drift")
    source_trace_path = Path(inputs.source_trace_path).expanduser().resolve(strict=True)
    if source_trace_path != Path(SOURCE_PAIR_TRACE_LINUX_PATH):
        raise ValueError("line-search source pair trace path drift")
    source_trace_hash = sha256_file(paired._regular_file(source_trace_path, "line-search source pair trace"))
    if _sha256(source_trace_hash, "line-search source pair trace hash") != SOURCE_PAIR_TRACE_SHA256:
        raise ValueError("line-search source pair trace hash mismatch")
    live_source_report, _ = _source_ablation_report(inputs.repository_root)
    construction = _mapping(
        inputs.provenance.get("hybrid_policy_construction"), "line-search resolved provenance construction"
    )
    source_report = _mapping(inputs.source_ablation_report, "line-search source ablation report")
    source_report_path = Path(inputs.source_ablation_report_path)
    if source_report_path != Path(SOURCE_ABLATION_REPORT_LINUX_PATH):
        raise ValueError("line-search source ablation report path drift")
    if (
        sha256_file(paired._regular_file(source_report_path, "line-search source ablation report"))
        != SOURCE_ABLATION_REPORT_SHA256
    ):
        raise ValueError("line-search source ablation report sha mismatch")
    if not _exact_json_equal(source_report, live_source_report):
        raise ValueError("line-search source ablation report drift")
    source_report_snapshot = _sha256(
        source_report["provenance"]["snapshot_sha256"],
        "line-search source report provenance snapshot",
    )
    if source_report_snapshot != SOURCE_ABLATION_REPORT_PROVENANCE_SHA256:
        raise ValueError("line-search source ablation report provenance snapshot mismatch")
    if source_report.get("kind") != source_ablation.ABLATION_KIND:
        raise ValueError("line-search source ablation report kind drift")
    if (
        source_report["provenance"]["snapshot_sha256"]
        != inputs.source_ablation_inputs.provenance["snapshot_sha256"]
    ):
        raise ValueError("line-search source ablation provenance mismatch")
    source_ablation_inputs_snapshot = _sha256(
        source_ablation._current_provenance_snapshot(inputs.source_ablation_inputs),
        "line-search source ablation live snapshot",
    )
    source_ablation_inputs_snapshot_expected = _sha256(
        inputs.provenance.get("source_ablation_report_inputs_snapshot_sha256"),
        "line-search source ablation inputs snapshot",
    )
    if source_ablation_inputs_snapshot != source_ablation_inputs_snapshot_expected:
        raise ValueError("line-search source ablation inputs snapshot mismatch")
    contract = _mapping(inputs.contract, "line-search resolved contract")
    _resolve_line_search_checkpoints_contract(construction, contract)
    return _canonical_sha256(
        {
            "kind": LINE_SEARCH_KIND,
            "prior_failure_receipt": prior_failure_receipt,
            "contract": _mapping(
                load_line_search_contract(inputs.repository_root),
                "line-search current resolved contract",
            ),
            "parent_run_dir": str(inputs.run_dir),
            "source_pair_trace_sha256": SOURCE_PAIR_TRACE_SHA256,
            "source_pair_provenance_sha256": SOURCE_PAIR_PROVENANCE_SHA256,
            "source_ablation_report_sha256": SOURCE_ABLATION_REPORT_SHA256,
            "source_ablation_report_provenance_sha256": SOURCE_ABLATION_REPORT_PROVENANCE_SHA256,
            "source_ablation_report_inputs_snapshot_sha256": source_ablation_inputs_snapshot,
            "source_ablation_report_snapshot_sha256": source_report_snapshot,
            "contract_sha256": CONTRACT_SHA256,
            "checkpoints_are_cpu": True,
            "hybrid_policy_construction": construction,
            "executed_line_search_sources": _source_binding(inputs.repository_root),
        }
    )


def _validate_line_search_bound_inputs(inputs: LineSearchInputs) -> None:
    provenance_requirements = _mapping(
        inputs.contract.get("provenance_requirements"),
        "line-search provenance requirements",
    )
    if (
        provenance_requirements.get("recursive_source_ablation_publication_revalidation") is not True
        or provenance_requirements.get("recursive_source_ablation_live_input_revalidation") is not True
        or provenance_requirements.get("inherit_physical_inputs_via_validated_source_ablation_provenance")
        is not True
        or provenance_requirements.get("inherit_runtime_inputs_via_validated_source_ablation_provenance")
        is not True
        or provenance_requirements.get("inherit_bound_inputs_via_validated_source_ablation_provenance") is not True
        or provenance_requirements.get("rehash_before_after_each_evaluation") is not True
        or provenance_requirements.get("rehash_immediately_before_publication") is not True
    ):
        raise ValueError("line-search provenance requirement policy mismatch")
    requirement_source_report = _sha256(
        provenance_requirements["source_ablation_report"]["snapshot_sha256"],
        "line-search provenance source ablation snapshot",
    )
    requirement_pair_snapshot = _sha256(
        provenance_requirements["source_pair_trace"]["snapshot_sha256"],
        "line-search provenance source pair snapshot",
    )
    if requirement_source_report != SOURCE_ABLATION_REPORT_PROVENANCE_SHA256:
        raise ValueError("line-search provenance source ablation snapshot requirement mismatch")
    if requirement_pair_snapshot != SOURCE_PAIR_PROVENANCE_SHA256:
        raise ValueError("line-search provenance source pair snapshot requirement mismatch")
    source_report_snapshot = _sha256(
        inputs.source_ablation_report["provenance"]["snapshot_sha256"],
        "line-search source report snapshot",
    )
    if source_report_snapshot != SOURCE_ABLATION_REPORT_PROVENANCE_SHA256:
        raise ValueError("line-search source ablation report provenance snapshot invalid")
    source_report_inputs_snapshot = _sha256(
        source_ablation._current_provenance_snapshot(inputs.source_ablation_inputs),
        "line-search source ablation live snapshot",
    )
    if source_report_inputs_snapshot != _sha256(
        inputs.source_ablation_inputs.provenance["snapshot_sha256"],
        "line-search source ablation provenance mismatch",
    ):
        raise ValueError("line-search source ablation provenance revalidation mismatch")
    if _current_provenance_snapshot(inputs) != inputs.provenance["snapshot_sha256"]:
        raise ValueError("line-search live provenance snapshot mismatch before ready")


def line_search_preflight(repository_root: Path, parent_run_dir: Path) -> dict[str, Any]:
    try:
        inputs = resolve_line_search_inputs(repository_root, parent_run_dir)
        if os.path.lexists(Path(inputs.contract["output"]["linux_path"])):
            output_path = Path(inputs.contract["output"]["linux_path"])
            return {
                "schema_version": SCHEMA_VERSION,
                "kind": PREFLIGHT_KIND,
                "ready": False,
                "error": {
                    "type": "FileExistsError",
                    "message": f"line-search preflight output exists: {output_path}",
                },
                **_line_search_boundary_receipt(simulator_constructed=False, include_simulator_transitions=True),
            }
        _validate_line_search_bound_inputs(inputs)
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": PREFLIGHT_KIND,
            "ready": True,
            "parent_run_dir": str(inputs.run_dir),
            "source_ablation_report": str(inputs.source_ablation_report_path),
            "prior_failure_receipt": inputs.provenance["prior_failure_receipt"],
            "source_ablation_inputs": {
                "snapshot_sha256": inputs.source_ablation_inputs.provenance["snapshot_sha256"],
            },
            "policy_construction": inputs.provenance["hybrid_policy_construction"],
            "provenance_snapshot_sha256": inputs.provenance["snapshot_sha256"],
            **_line_search_boundary_receipt(simulator_constructed=False, include_simulator_transitions=True),
        }
        if report["simulator_constructed"] is not False:
            raise ValueError("line-search simulator constructor leaked into preflight")
        if inputs.contract.get("boundaries") != _line_search_boundary_receipt(simulator_constructed=None):
            raise ValueError("line-search preflight boundaries drift")
        return report
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": PREFLIGHT_KIND,
            "ready": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "provenance": {"snapshot_sha256": None},
            **_line_search_boundary_receipt(simulator_constructed=False, include_simulator_transitions=True),
        }


def ablation_preflight(repository_root: Path, parent_run_dir: Path) -> dict[str, Any]:
    return line_search_preflight(repository_root, parent_run_dir)


def resolve_ablation_inputs(repository_root: Path, parent_run_dir: Path) -> LineSearchInputs:
    return resolve_line_search_inputs(repository_root, parent_run_dir)


def validate_line_search_trace(
    trace: Mapping[str, Any], layout: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    return validate_ablation_trace(trace, layout, contract)


def run_line_search_trace(
    *,
    policy: Any,
    wrapped_env: Any,
    policy_name: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return run_ablation_trace(
        policy=policy,
        wrapped_env=wrapped_env,
        policy_name=policy_name,
        contract=contract,
    )


def compare_line_search_pair(
    left_name: str,
    left_trace: Mapping[str, Any],
    right_name: str,
    right_trace: Mapping[str, Any],
    layout: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return compare_ablation_pair(
        left_name,
        left_trace,
        right_name,
        right_trace,
        layout,
        contract,
    )


def execute_line_search(repository_root: Path, parent_run_dir: Path) -> dict[str, Any]:
    return execute_update_delta_ablation(repository_root, parent_run_dir)


__all__ = [
    "LINE_SEARCH_KIND",
    "ABLATION_KIND",
    "CONTRACT_SHA256",
    "EndpointReproductionError",
    "OUTPUT_FILENAME",
    "POLICY_ORDER",
    "POLICY_ALPHAS",
    "POLICY_SHA256",
    "LineSearchInputs",
    "ablation_preflight",
    "line_search_preflight",
    "compare_ablation_pair",
    "construct_interpolated_policy_state",
    "construct_hybrid_policy_state",
    "failure_report",
    "load_line_search_contract",
    "load_ablation_contract",
    "resolve_line_search_inputs",
    "resolve_ablation_inputs",
    "validate_line_search_trace",
    "run_ablation_trace",
    "run_line_search_trace",
    "validate_ablation_trace",
    "compare_line_search_pair",
    "compare_ablation_pair",
    "validate_endpoint_reproduction",
    "execute_line_search",
    "execute_update_delta_ablation",
    "signed_finite_differences",
    "classify_direction_vs_magnitude",
    "validate_publication_provenance",
    "write_json_exclusive",
]
