from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import torch

from gear_sonic.scripts import (
    trace_g1_true23_sonic_task_space_ppo_full_support_delta_line_search as cli,
)
from gear_sonic.utils import (
    g1_true23_sonic_task_space_ppo_full_support_checkpoint_trace as paired,
    g1_true23_sonic_task_space_ppo_full_support_delta_line_search as subject,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_POLICY_STATE_SHA256: dict[str, str] = {
    "alpha_minus_0_25": "eae8d0287a9e17d44d33e7bd2ca86598e9a4b8bc0acb1669236d78a9e4a6e67c",
    "baseline": "358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61",
    "alpha_plus_0_25": "bb14ccc4fc465d3c61580c3f3b2f2512ddcd3fc0d9bf49ed1fa7cbd048585113",
    "alpha_plus_0_5": "dd56f4cd7b2c18e69b63aa4c7e537f89407dc18a4ffc17465bbebf8819919753",
    "full": "7299df1851c5b42256f170334e2d5afdc81603b17ae832381abadadd9ea48639",
}
LINE_SEARCH_PRIOR_FAILURE_PATH = "/root/g1_true23_runs/sonic_task_space_ppo_full_support_delta_line_search_v1.json"
LINE_SEARCH_PRIOR_FAILURE_SHA = "0be248f2405aded5ade65632e95eb7fc5eae26d79082b3bb43177e4d4316ec3a"
LINE_SEARCH_PRIOR_FAILURE_KIND = "g1_true23_sonic_task_space_ppo_full_support_delta_line_search_v1_failure"
LINE_SEARCH_PRIOR_FAILURE_CONTRACT_SHA = "d6ac14e63de0a74a31946a2d056b54451c307587d7f31f5a3383f363b9c104d0"
LINE_SEARCH_RETRY1_OUTPUT_PATH = (
    "/root/g1_true23_runs/sonic_task_space_ppo_full_support_delta_line_search_v1_retry1.json"
)
LINE_SEARCH_RETRY1_OUTPUT_FILENAME = "sonic_task_space_ppo_full_support_delta_line_search_v1_retry1.json"
LINE_SEARCH_PREV_OUTPUT_FILENAME = "sonic_task_space_ppo_full_support_delta_line_search_v1.json"
LINE_SEARCH_PRIOR_FAILURE_ERROR: dict[str, str] = {
    "type": "ValueError",
    "message": "ablation pairwise comparison order drift",
}


def _contract() -> dict[str, Any]:
    return dict(subject.load_line_search_contract(ROOT))


def _synthetic_checkpoints() -> dict[int, Mapping[str, Any]]:
    fixed = torch.tensor([7.0], dtype=torch.float32)
    checkpoints = {
        0: {
            "policy_state_dict": {
                name: torch.tensor([float(index)], dtype=torch.float32)
                for index, name in enumerate(subject.UPDATED_TENSORS)
            }
        },
        1: {
            "policy_state_dict": {
                name: torch.tensor([float(index + 10)], dtype=torch.float32)
                for index, name in enumerate(subject.UPDATED_TENSORS)
            }
        },
    }
    for index in (0, 1):
        checkpoints[index]["policy_state_dict"]["actor_module.teleop.module.0.bias"] = fixed.clone()
    return {index: {"policy_state_dict": state["policy_state_dict"]} for index, state in checkpoints.items()}


def _line_search_checkpoint_construction(policy_state_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    checkpoints = _synthetic_checkpoints()
    tensor_count = len(checkpoints[0]["policy_state_dict"])
    resolved_policy_state_hashes = dict(EXPECTED_POLICY_STATE_SHA256)
    if policy_state_hashes is not None:
        resolved_policy_state_hashes = {
            **resolved_policy_state_hashes,
            **policy_state_hashes,
        }
    return {
        "policy_order": list(subject.POLICY_ORDER),
        "module14_tensor_names": list(subject.MODULE14_TENSORS),
        "module16_tensor_names": list(subject.MODULE16_TENSORS),
        "all_other_policy_tensors_identical_between_model0_and_model1": True,
        "hybrids_are_in_memory_only": True,
        "updated_tensor_names": list(subject.UPDATED_TENSORS),
        "policies": {
            name: {
                "alpha_numerator": subject.POLICY_ALPHAS[name][0],
                "alpha_denominator": subject.POLICY_ALPHAS[name][1],
                "policy_state_sha256": resolved_policy_state_hashes[name],
                "tensor_count": tensor_count,
                "exact_model0_keyset": True,
                "frozen_tensors_byte_identical_to_model0": True,
                "only_declared_tensors_may_differ": True,
                "cpu_float32_contiguous_finite": True,
                "no_source_or_cross_policy_alias": True,
                "source_checkpoints_unchanged": True,
            }
            for index, name in enumerate(subject.POLICY_ORDER)
        },
    }


def test_policy_expected_state_sha256_requires_explicit_expected_hash_field() -> None:
    explicit = {"expected_policy_state_sha256": "f" * 64}
    assert subject._policy_expected_state_sha256(explicit, context="line-search test") == "f" * 64
    with pytest.raises(ValueError, match="missing expected policy state hash"):
        subject._policy_expected_state_sha256(
            {"policy_state_sha256": "f" * 64},
            context="line-search test",
        )


def _expected_policy_value(
    policy_name: str,
    tensor_name: str,
    checkpoints: Mapping[int, Mapping[str, Any]],
) -> torch.Tensor:
    if tensor_name not in subject.UPDATED_TENSORS:
        return checkpoints[0]["policy_state_dict"][tensor_name].clone()
    numerator, denominator = subject.POLICY_ALPHAS[policy_name]
    source = checkpoints[0]["policy_state_dict"][tensor_name]
    target = checkpoints[1]["policy_state_dict"][tensor_name]
    alpha = numerator / denominator
    return source + source.new_full(source.shape, alpha) * (target - source)


def _fake_policy_hash_factory(checkpoints: Mapping[int, Mapping[str, Any]]) -> Any:
    model0 = checkpoints[0]["policy_state_dict"]
    model1 = checkpoints[1]["policy_state_dict"]
    base = float(model0[subject.MODULE14_TENSORS[0]][0])
    delta = float((model1[subject.MODULE14_TENSORS[0]] - model0[subject.MODULE14_TENSORS[0]])[0])

    def _fake_policy_hash(body: Any, *args: Any, **kwargs: Any) -> str:
        payload = body["policy_state_dict"]
        sample = float(payload[subject.MODULE14_TENSORS[0]][0])
        alpha = (sample - base) / delta if delta else 0.0
        for policy_name, spec in subject.POLICY_ALPHAS.items():
            target = spec[0] / spec[1]
            if math.isclose(alpha, target, rel_tol=0.0, abs_tol=1e-6):
                return subject.POLICY_SHA256[policy_name]
        return "f" * 64

    return _fake_policy_hash


def _minimal_ablation_trace_reward_terms(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(name),
            "weight": float(weight),
            "callable_identity": str(callable_identity),
        }
        for name, weight, callable_identity in contract["expected_reward_terms"]
    ]


def _minimal_ablation_layout(contract: Mapping[str, Any]) -> dict[str, Any]:
    terms = _minimal_ablation_trace_reward_terms(contract)
    return {
        "reward_terms": terms,
        "reward_internal_identity": {
            "manager_class": "test-minimal-layout",
            "step_reward_shape": [1, len(terms)],
            "column_order_verified": True,
            "terms": [
                {
                    "column_index": index,
                    "name": term["name"],
                    "weight": term["weight"],
                    "callable_identity": term["callable_identity"],
                    "parameter_names": [f"param_{index}"],
                    "public_internal_cfg_object_identical": True,
                }
                for index, term in enumerate(terms)
            ],
        },
        "control_dt_s": 0.02,
        "ee_body_names": list(paired.EE_BODY_NAMES),
        "contact_site_names": list(paired.CONTACT_SITE_NAMES),
        "native_joint_names": list(paired.NATIVE_IL23_JOINT_NAMES),
        "hardware_joint_names": list(paired.HARDWARE_23_JOINT_NAMES),
        "action_orders": {
            "raw_native_action": "native_isaaclab_23",
            "safe_native_action": "native_isaaclab_23",
            "final_target_hardware": "hardware_mujoco_23",
        },
        "ee_body_pos_threshold_m": 0.25,
    }


def _minimal_ablation_trace(
    contract: Mapping[str, Any],
    policy_name: str,
    *,
    completed: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_native_action = [0.0] * paired.ACTION_DIM
    safe_native_action, final_target_hardware = safe_target_transform_numpy(
        np.array(raw_native_action, dtype=np.float32)
    )
    safe_native_action = [float(item) for item in safe_native_action]
    final_target_hardware = [float(item) for item in final_target_hardware]
    contract_spec = contract["policy_construction"]["policies"][policy_name]
    policy_state = contract_spec["expected_policy_state_sha256"]
    policy = contract["policy_construction"]["policies"][policy_name]
    q9 = list(range(9, 9 + completed))
    width_reward = len(contract["expected_reward_terms"])
    zero_reward = [0.0] * width_reward
    layout = _minimal_ablation_layout(contract)
    ee_zero = [0.0] * len(paired.EE_BODY_NAMES)
    ee_terminal = [0.0] * len(paired.EE_BODY_NAMES)
    ee_terminal[-1] = 0.4

    trace = {
        "policy_name": policy_name,
        "alpha_numerator": int(policy["alpha_numerator"]),
        "alpha_denominator": int(policy["alpha_denominator"]),
        "evaluation_seed": paired.FIXED_SEED,
        "controller": "deterministic_actor_mean",
        "policy_state_sha256": policy_state,
        "initial_actor_observation_sha256": "0" * 64,
        "completed_transitions": completed,
        "terminal_q9": q9[-1],
        "termination_names": ["ee_body_pos"],
        "episode_return": 0.0,
        "series": {
            "q9": q9,
            "reward_total": [0.0] * completed,
            "reward_raw": [zero_reward.copy() for _ in range(completed)],
            "reward_weighted": [zero_reward.copy() for _ in range(completed)],
            "ee_z_error_m": [ee_zero.copy() for _ in range(max(0, completed - 1))] + [ee_terminal],
            "raw_native_action": [raw_native_action.copy() for _ in range(completed)],
            "safe_native_action": [safe_native_action.copy() for _ in range(completed)],
            "final_target_hardware": [final_target_hardware.copy() for _ in range(completed)],
            "contact_found": [[False] * len(paired.CONTACT_SITE_NAMES) for _ in range(completed)],
            "contact_force_magnitude_n": [[0.0] * len(paired.CONTACT_SITE_NAMES) for _ in range(completed)],
            "landing_force_mean_n": [0.0] * completed,
            "termination_names": [[] for _ in range(max(0, completed - 1))] + [["ee_body_pos"]],
        },
    }
    return trace, layout


def _endpoint_trace(policy_name: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    expected = contract["expected_endpoint_reproductions"][policy_name]
    return {
        "completed_transitions": expected["completed_transitions"],
        "terminal_q9": expected["terminal_q9"],
        "termination_names": expected["termination_names"],
        "policy_state_sha256": expected["expected_policy_state_sha256"],
        "episode_return": expected["source_trace_episode_return"],
        "series": {
            "ee_z_error_m": [[0.0, 0.0, 0.0, 0.4]],
            "contact_found": [[False] * len(paired.CONTACT_SITE_NAMES)],
            "contact_force_magnitude_n": [[0.0] * len(paired.CONTACT_SITE_NAMES)],
        },
    }


def _pairwise_and_attribution_fixtures(
    *,
    completed_by_policy: Mapping[str, int] | None = None,
    episode_return_by_policy: Mapping[str, float] | None = None,
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    contract = _contract()
    traces: dict[str, Mapping[str, Any]] = {}
    layout: Mapping[str, Any] | None = None
    for policy_name in subject.POLICY_ORDER:
        completed = 3
        if completed_by_policy is not None:
            completed = int(completed_by_policy[policy_name])
        trace, policy_layout = _minimal_ablation_trace(contract, policy_name, completed=completed)
        if episode_return_by_policy is not None:
            trace["episode_return"] = float(episode_return_by_policy[policy_name])
        traces[policy_name] = trace
        if layout is None:
            layout = policy_layout
    if layout is None:
        raise RuntimeError("failed to build pairwise traces")
    return traces, layout, contract


def test_contract_and_execution_schema_are_exact_and_reachable() -> None:
    contract = _contract()
    assert subject.POLICY_SHA256 == EXPECTED_POLICY_STATE_SHA256
    assert subject.CONTRACT_SHA256 == sha256_file(ROOT / subject.CONTRACT_RELATIVE_PATH)
    assert contract["kind"] == subject.CONTRACT_KIND
    assert contract["comparison"]["pairwise_count"] == len(subject.PAIRWISE_COMPARISON_ORDER)
    assert contract["comparison"]["pairwise_order"] == list(subject.PAIRWISE_COMPARISON_ORDER)
    assert contract["output"]["filename"] == subject.OUTPUT_FILENAME
    assert contract["output"]["linux_path"] == f"/root/g1_true23_runs/{subject.OUTPUT_FILENAME}"
    assert contract["output"]["path"] == contract["output"]["linux_path"]
    assert subject.UPDATED_TENSORS == (
        "actor_module.decoders.g1_dyn.module.14.bias",
        "actor_module.decoders.g1_dyn.module.14.weight",
        "actor_module.decoders.g1_dyn.module.16.bias",
        "actor_module.decoders.g1_dyn.module.16.weight",
    )
    assert contract["policy_construction"]["policy_order"] == list(subject.POLICY_ORDER)
    assert tuple(contract["policy_construction"]["policies"]) == subject.POLICY_ORDER
    expected_policy_hashes = [EXPECTED_POLICY_STATE_SHA256[name] for name in subject.POLICY_ORDER]
    contract_policy_hashes = []
    for name in subject.POLICY_ORDER:
        mapping = contract["policy_construction"]["policies"][name]
        contract_policy_hashes.append(mapping["expected_policy_state_sha256"])
    assert contract_policy_hashes == expected_policy_hashes
    boundary = subject._line_search_boundary_receipt(simulator_constructed=None)
    assert (
        contract["policy_construction"]["intermediate_policy_hash_binding"]
        == "immutable_contract_pinned_exact_all_five_policy_state_sha256"
    )
    assert contract["execution"]["policy_count"] == len(subject.POLICY_ORDER)
    assert contract["source_pair_trace"]["sha256"] == subject.SOURCE_PAIR_TRACE_SHA256
    assert contract["source_pair_trace"]["provenance_snapshot_sha256"] == subject.SOURCE_PAIR_PROVENANCE_SHA256
    assert contract["source_delta_ablation_report"]["sha256"] == subject.DELTA_ABLATION_REPORT_SHA256
    assert contract["source_delta_ablation_report"]["sha256"] == subject.SOURCE_ABLATION_REPORT_SHA256
    assert contract["expected_endpoint_reproductions"].keys() == {"baseline", "full"}
    assert contract["policy_construction"]["policy_order"] == list(subject.POLICY_ORDER)
    assert contract["policy_construction"]["canonical_algorithm"]["updated_tensor_names"] == list(
        subject.UPDATED_TENSORS
    )
    assert contract["prior_failure_receipt"]["path"] == LINE_SEARCH_PRIOR_FAILURE_PATH
    assert contract["prior_failure_receipt"]["linux_path"] == LINE_SEARCH_PRIOR_FAILURE_PATH
    assert contract["prior_failure_receipt"]["filename"] == LINE_SEARCH_PREV_OUTPUT_FILENAME
    assert contract["prior_failure_receipt"]["sha256"] == LINE_SEARCH_PRIOR_FAILURE_SHA
    assert contract["prior_failure_receipt"]["size_bytes"] == 1410
    assert contract["prior_failure_receipt"]["kind"] == LINE_SEARCH_PRIOR_FAILURE_KIND
    assert contract["prior_failure_receipt"]["failure_stage"] == "exclusive_publication"
    assert contract["prior_failure_receipt"]["error"] == LINE_SEARCH_PRIOR_FAILURE_ERROR
    assert contract["prior_failure_receipt"]["byte_immutable"] is True
    assert contract["prior_failure_receipt"]["trace_evidence_used"] is False
    assert contract["prior_failure_receipt"]["discarded_simulation_evidence_reused"] is False
    assert contract["prior_failure_receipt"]["boundary_receipt"] == boundary
    assert subject.PRIOR_FAILURE_CONTRACT_SHA256 == LINE_SEARCH_PRIOR_FAILURE_CONTRACT_SHA
    assert contract["prior_failure_receipt"]["contract_sha256"] == LINE_SEARCH_PRIOR_FAILURE_CONTRACT_SHA
    assert contract["output"]["filename"] == LINE_SEARCH_RETRY1_OUTPUT_FILENAME
    assert contract["output"]["path"] == contract["output"]["linux_path"] == LINE_SEARCH_RETRY1_OUTPUT_PATH
    assert contract["output"]["no_replace"] is True
    assert contract["output"]["atomic"] is True
    assert boundary["teacher_queries"] == 0
    assert boundary["training_transitions"] == 0
    assert boundary["training_updates"] == 0
    assert boundary["optimizer_steps"] == 0
    assert boundary["checkpoints_written"] == 0
    assert boundary["robot_commands_issued"] == 0
    assert boundary["network_commands_issued"] == 0
    assert boundary["hardware_actions"] == 0
    assert boundary["candidate_selected"] is False
    assert boundary["hardware_authorized"] is False
    assert boundary["robot_or_network_commands_permitted"] is False
    assert boundary["diagnostic_only"] is True
    assert boundary["failed_model5_loaded"] is False
    assert boundary["failed_model5_resumed"] is False
    assert boundary["support_qualified"] is False
    assert boundary["promotion_eligible"] is False
    assert boundary["teacher_labels_used"] is False
    assert boundary["deployment_ready"] is False
    assert contract["expected_reward_terms"], "contract reward terms missing"
    policies = contract["policy_construction"]["policies"]
    for policy_name in subject.POLICY_ORDER:
        assert policies[policy_name]["alpha_numerator"] == subject.POLICY_ALPHAS[policy_name][0]
        assert policies[policy_name]["alpha_denominator"] == subject.POLICY_ALPHAS[policy_name][1]
        assert policies[policy_name]["expected_policy_state_sha256"] == EXPECTED_POLICY_STATE_SHA256[policy_name]


def _prior_failure_receipt_contract_spec() -> dict[str, Any]:
    return {
        "path": subject.PRIOR_FAILURE_REPORT_LINUX_PATH,
        "linux_path": subject.PRIOR_FAILURE_REPORT_LINUX_PATH,
        "filename": subject.PRIOR_FAILURE_OUTPUT_FILENAME,
        "sha256": subject.PRIOR_FAILURE_REPORT_SHA256,
        "size_bytes": subject.PRIOR_FAILURE_REPORT_SIZE_BYTES,
        "kind": subject.PRIOR_FAILURE_KIND,
        "contract_sha256": subject.PRIOR_FAILURE_CONTRACT_SHA256,
        "failure_stage": subject.PRIOR_FAILURE_STAGE,
        "error": dict(subject.PRIOR_FAILURE_ERROR),
        "simulator_ablation_complete": False,
        "boundary_receipt": subject._line_search_boundary_receipt(simulator_constructed=None),
        "byte_immutable": True,
        "trace_evidence_used": False,
        "discarded_simulation_evidence_reused": False,
    }


def _prior_failure_receipt_runtime_identity() -> dict[str, Any]:
    return {
        "path": subject.PRIOR_FAILURE_REPORT_LINUX_PATH,
        "sha256": subject.PRIOR_FAILURE_REPORT_SHA256,
        "size_bytes": subject.PRIOR_FAILURE_REPORT_SIZE_BYTES,
        "kind": subject.PRIOR_FAILURE_KIND,
        "contract_sha256": subject.PRIOR_FAILURE_CONTRACT_SHA256,
        "failure_stage": subject.PRIOR_FAILURE_STAGE,
        "error": dict(subject.PRIOR_FAILURE_ERROR),
        "validated": True,
        "byte_immutable": True,
        "trace_evidence_used": False,
        "discarded_simulation_evidence_reused": False,
    }


def _build_synthetic_prior_failure_receipt_body(
    contract: Mapping[str, Any], preexecution_provenance_snapshot_sha256: str
) -> bytes:
    parent = contract.get("parent_run")
    parent_run_dir = ""
    if type(parent) is dict:
        parent_run_dir = parent.get("linux_path", "")
    if type(parent_run_dir) is not str:
        parent_run_dir = ""
    receipt = {
        "schema_version": subject.SCHEMA_VERSION,
        "kind": subject.PRIOR_FAILURE_KIND,
        "ablation_contract_sha256": subject.PRIOR_FAILURE_CONTRACT_SHA256,
        "source_pair_trace_sha256": subject.SOURCE_PAIR_TRACE_SHA256,
        "failure_stage": subject.PRIOR_FAILURE_STAGE,
        "parent_run_dir": parent_run_dir,
        "preexecution_provenance_snapshot_sha256": preexecution_provenance_snapshot_sha256,
        "error": dict(subject.PRIOR_FAILURE_ERROR),
        "simulator_ablation_complete": False,
        "source_delta_ablation_report_sha256": subject.SOURCE_ABLATION_REPORT_SHA256,
        "source_delta_ablation_report_provenance_snapshot_sha256": (
            subject.SOURCE_ABLATION_REPORT_PROVENANCE_SHA256
        ),
        **subject._line_search_boundary_receipt(simulator_constructed=None),
    }
    return json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def test_line_search_checkpoints_contract_validates_exact_rational_schedule() -> None:
    contract = _contract()
    construction = _line_search_checkpoint_construction()
    subject._resolve_line_search_checkpoints_contract(construction, contract)


def test_line_search_checkpoints_contract_rejects_rational_tamper() -> None:
    contract = _contract()
    construction = _line_search_checkpoint_construction()
    construction["policies"]["full"]["alpha_denominator"] = 7
    with pytest.raises(ValueError):
        subject._resolve_line_search_checkpoints_contract(construction, contract)


def test_line_search_checkpoints_contract_rejects_updated_tensor_order_tamper() -> None:
    contract = _contract()
    construction = _line_search_checkpoint_construction()
    construction["updated_tensor_names"] = list(reversed(construction["updated_tensor_names"]))
    with pytest.raises(ValueError, match="line-search construction updated tensors mismatch"):
        subject._resolve_line_search_checkpoints_contract(construction, contract)


def test_line_search_checkpoints_contract_rejects_non_list_updated_tensors() -> None:
    contract = _contract()
    construction = _line_search_checkpoint_construction()
    construction["updated_tensor_names"] = tuple(construction["updated_tensor_names"])
    with pytest.raises(ValueError, match="line-search construction updated tensors mismatch"):
        subject._resolve_line_search_checkpoints_contract(construction, contract)


def test_line_search_runtime_contract_rejects_non_canonical_pairwise_order() -> None:
    contract = _contract()
    construction = _line_search_checkpoint_construction()
    noncanonical_order = [
        "alpha_minus_0_25_vs_baseline",
        "alpha_minus_0_25_vs_alpha_plus_0_25",
        "alpha_minus_0_25_vs_alpha_plus_0_5",
        "alpha_minus_0_25_vs_full",
        "baseline_vs_alpha_plus_0_25",
        "baseline_vs_alpha_plus_0_5",
        "baseline_vs_full",
        "alpha_plus_0_25_vs_alpha_plus_0_5",
        "alpha_plus_0_25_vs_full",
        "alpha_plus_0_5_vs_full",
    ]
    contract["comparison"]["pairwise_order"] = noncanonical_order
    with pytest.raises(ValueError, match="line-search runtime comparison"):
        subject._runtime_trace_contract(contract, construction)


@pytest.mark.parametrize(
    ("mutator", "expected_message"),
    (
        ("reordered", "line-search runtime comparison pairwise order mismatch"),
        ("too_few", "line-search runtime comparison pairwise order mismatch"),
        ("too_many", "line-search runtime comparison pairwise order mismatch"),
        ("not_list", "line-search runtime comparison pairwise order mismatch"),
    ),
)
def test_validate_comparison_pairwise_order_rejects_mutated_orders(mutator: str, expected_message: str) -> None:
    contract = copy.deepcopy(_contract())
    if mutator == "reordered":
        contract["comparison"]["pairwise_order"] = [
            f"{left}_vs_{right}" for left, right in itertools.combinations(subject.POLICY_ORDER, 2)
        ]
    elif mutator == "too_few":
        contract["comparison"]["pairwise_order"] = list(subject.PAIRWISE_COMPARISON_ORDER[:-1])
        contract["comparison"]["pairwise_count"] = len(subject.PAIRWISE_COMPARISON_ORDER)
    elif mutator == "too_many":
        contract["comparison"]["pairwise_order"] = [
            *subject.PAIRWISE_COMPARISON_ORDER,
            subject.PAIRWISE_COMPARISON_ORDER[0],
        ]
        contract["comparison"]["pairwise_count"] = len(contract["comparison"]["pairwise_order"])
    else:
        contract["comparison"]["pairwise_order"] = tuple(subject.PAIRWISE_COMPARISON_ORDER)
    with pytest.raises(ValueError, match=expected_message):
        comparison = contract["comparison"]
        subject._validate_comparison_pairwise_order(
            comparison,
            context="line-search runtime",
        )


def test_build_pairwise_comparisons_is_canonical_and_round_trips_with_sorted_json() -> None:
    traces, layout, contract = _pairwise_and_attribution_fixtures()
    pairwise = subject._build_pairwise_comparisons(traces=traces, layout=layout, contract=contract)
    assert list(pairwise) == list(subject.PAIRWISE_COMPARISON_ORDER)

    encoded = json.dumps(pairwise, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    decoded = json.loads(encoded.decode("utf-8"))
    assert list(decoded) == list(pairwise)
    recomputed = subject._rebuild_pairwise_comparisons(
        expected_pairwise_order=None,
        pairwise_comparisons=decoded,
        traces=traces,
        layout=layout,
        runtime_contract=contract,
    )
    assert subject._exact_json_equal(recomputed, pairwise)


def test_rebuild_pairwise_comparisons_rejects_reversed_noncanonical_order_and_expected_order_drift() -> None:
    traces, layout, contract = _pairwise_and_attribution_fixtures()
    pairwise = subject._build_pairwise_comparisons(traces=traces, layout=layout, contract=contract)
    canonical_order = list(subject.PAIRWISE_COMPARISON_ORDER)
    reversed_payload = {label: pairwise[label] for label in reversed(canonical_order)}
    with pytest.raises(ValueError, match="ablation pairwise comparison order drift"):
        subject._rebuild_pairwise_comparisons(
            expected_pairwise_order=canonical_order,
            pairwise_comparisons=reversed_payload,
            traces=traces,
            layout=layout,
            runtime_contract=contract,
        )


def test_rebuild_pairwise_comparisons_rejects_legacy_nonlex_order_and_expected_order_drift() -> None:
    traces, layout, contract = _pairwise_and_attribution_fixtures()
    pairwise = subject._build_pairwise_comparisons(traces=traces, layout=layout, contract=contract)
    legacy_order = [f"{left}_vs_{right}" for left, right in itertools.combinations(subject.POLICY_ORDER, 2)]
    legacy_payload = {label: pairwise[label] for label in legacy_order}
    encoded = json.dumps(
        legacy_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    decoded = json.loads(encoded.decode("utf-8"))
    with pytest.raises(ValueError, match="ablation pairwise comparison order drift"):
        subject._rebuild_pairwise_comparisons(
            expected_pairwise_order=legacy_order,
            pairwise_comparisons=decoded,
            traces=traces,
            layout=layout,
            runtime_contract=contract,
        )


def test_rebuild_pairwise_comparisons_rejects_key_or_mapping_drifts() -> None:
    traces, layout, contract = _pairwise_and_attribution_fixtures()
    pairwise = subject._build_pairwise_comparisons(traces=traces, layout=layout, contract=contract)
    drifted_keys = dict(pairwise)
    drifted_keys.pop(next(iter(drifted_keys)))
    with pytest.raises(ValueError, match="ablation pairwise comparison key mismatch"):
        subject._rebuild_pairwise_comparisons(
            expected_pairwise_order=None,
            pairwise_comparisons=drifted_keys,
            traces=traces,
            layout=layout,
            runtime_contract=contract,
        )
    with pytest.raises(ValueError, match="ablation pairwise comparison duplicate labels"):
        subject._rebuild_pairwise_comparisons(
            expected_pairwise_order=[*subject.PAIRWISE_COMPARISON_ORDER, subject.PAIRWISE_COMPARISON_ORDER[0]],
            pairwise_comparisons=pairwise,
            traces=traces,
            layout=layout,
            runtime_contract=contract,
        )
    drifted_extra = dict(pairwise)
    drifted_extra["unexpected"] = pairwise[next(iter(pairwise))]
    with pytest.raises(ValueError, match="ablation pairwise comparison key mismatch"):
        subject._rebuild_pairwise_comparisons(
            expected_pairwise_order=None,
            pairwise_comparisons=drifted_extra,
            traces=traces,
            layout=layout,
            runtime_contract=contract,
        )
    tampered = copy.deepcopy(pairwise)
    first = next(iter(tampered))
    tampered[first] = {**tampered[first], "tampered": True}
    with pytest.raises(ValueError, match="ablation pairwise comparison drift"):
        subject._rebuild_pairwise_comparisons(
            expected_pairwise_order=None,
            pairwise_comparisons=tampered,
            traces=traces,
            layout=layout,
            runtime_contract=contract,
        )


@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        (b'{"a": 1, "a": 2}', "contains duplicate key 'a'"),
        (b"\xff", "contains invalid UTF-8"),
        (b"{", "contains malformed JSON"),
        (b'{"a": NaN}', "contains non-finite token NaN"),
    ],
)
def test_load_strict_json_from_bytes_rejects_duplicate_or_invalid_encodings_and_tokens(
    payload: bytes, expected_message: str
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        subject._load_strict_json_from_bytes(payload, "line-search strict json tests")


def _configure_prior_failure_contract_and_report(
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    preexecution_provenance_snapshot_sha256: str = "0" * 64,
    mutator: str | None = None,
) -> tuple[Mapping[str, Any], Path, bytes]:
    contract = copy.deepcopy(_contract())
    parent_run = tmp_path / "parent"
    parent_run.mkdir()
    contract["parent_run"]["linux_path"] = str(parent_run)
    report_path = tmp_path / subject.PRIOR_FAILURE_OUTPUT_FILENAME
    body = _build_synthetic_prior_failure_receipt_body(contract, preexecution_provenance_snapshot_sha256)
    if mutator == "wrong_body_stage":
        report = json.loads(body.decode("utf-8"))
        report["failure_stage"] = "wrong_stage"
        body = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    elif mutator == "bad_body_error":
        report = json.loads(body.decode("utf-8"))
        report["error"] = {"type": "bad", "message": "wrong"}
        body = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    elif mutator == "extra_body_key":
        report = json.loads(body.decode("utf-8"))
        report["unexpected"] = 1
        body = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    elif mutator == "partial_body_evidence":
        report = json.loads(body.decode("utf-8"))
        report["partial_scalar_evidence"] = {"type": "overflow"}
        body = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    report_path.write_bytes(body)
    report_hash = hashlib.sha256(body).hexdigest()
    receipt = _prior_failure_receipt_contract_spec()
    receipt["path"] = str(report_path)
    receipt["linux_path"] = str(report_path)
    receipt["filename"] = report_path.name
    receipt["sha256"] = report_hash
    receipt["size_bytes"] = len(body)
    contract["prior_failure_receipt"] = dict(receipt)
    if mutator == "wrong_spec_stage":
        contract["prior_failure_receipt"]["failure_stage"] = "wrong_stage"
    monkeypatch.setattr(subject, "PRIOR_FAILURE_REPORT_LINUX_PATH", str(report_path))
    monkeypatch.setattr(subject, "PRIOR_FAILURE_REPORT_SHA256", report_hash)
    monkeypatch.setattr(subject, "PRIOR_FAILURE_REPORT_SIZE_BYTES", len(body))
    return contract, report_path, body


def test_read_and_validate_prior_failure_receipt_accepts_canonical_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, report_path, body = _configure_prior_failure_contract_and_report(tmp_path, monkeypatch=monkeypatch)
    received = subject._read_and_validate_prior_failure_receipt(contract)
    assert received["path"] == str(report_path)
    assert received["sha256"] == hashlib.sha256(body).hexdigest()
    assert received["size_bytes"] == len(body)
    assert received["kind"] == subject.PRIOR_FAILURE_KIND
    assert received["contract_sha256"] == subject.PRIOR_FAILURE_CONTRACT_SHA256
    assert received["failure_stage"] == subject.PRIOR_FAILURE_STAGE
    assert received["byte_immutable"] is True
    assert received["trace_evidence_used"] is False
    assert received["discarded_simulation_evidence_reused"] is False
    assert isinstance(received["error"], dict)


@pytest.mark.parametrize(
    ("mutator", "expected_message"),
    (
        ("wrong_report_hash", "line-search prior failure report hash mismatch"),
        ("wrong_report_size", "line-search prior failure report size mismatch"),
        ("wrong_spec_stage", "line-search prior failure receipt specification mismatch"),
        ("wrong_body_stage", "line-search prior failure report schema mismatch"),
        ("bad_body_error", "line-search prior failure report schema mismatch"),
        ("extra_body_key", "line-search prior failure report schema mismatch"),
        ("partial_body_evidence", "line-search prior failure report schema mismatch"),
    ),
)
def test_read_and_validate_prior_failure_receipt_rejects_mutated_specifications_and_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: str,
    expected_message: str,
) -> None:
    contract, _, body = _configure_prior_failure_contract_and_report(
        tmp_path,
        monkeypatch=monkeypatch,
        mutator=(
            mutator
            if mutator.startswith("wrong_body")
            or mutator.startswith("extra")
            or mutator.startswith("partial")
            or mutator.startswith("bad_body")
            else None
        ),
    )
    if mutator == "wrong_report_hash":
        monkeypatch.setattr(subject, "PRIOR_FAILURE_REPORT_SHA256", "0" * 64)
        monkeypatch.setattr(subject, "PRIOR_FAILURE_REPORT_SIZE_BYTES", len(body))
        contract["prior_failure_receipt"]["sha256"] = "0" * 64
    elif mutator == "wrong_report_size":
        monkeypatch.setattr(subject, "PRIOR_FAILURE_REPORT_SIZE_BYTES", len(body) + 1)
        contract["prior_failure_receipt"]["size_bytes"] = len(body) + 1
    elif mutator == "wrong_spec_stage":
        contract["prior_failure_receipt"]["failure_stage"] = "wrong_stage"
    with pytest.raises(ValueError, match=expected_message):
        subject._read_and_validate_prior_failure_receipt(contract)


def test_read_and_validate_prior_failure_receipt_rejects_symlink_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract, report_path, body = _configure_prior_failure_contract_and_report(tmp_path, monkeypatch=monkeypatch)
    target = tmp_path / "real_prior_failure_report.json"
    target.write_bytes(body)
    if report_path.exists():
        report_path.unlink()
    report_path.symlink_to(target)
    with pytest.raises(ValueError, match="may not traverse symlinks"):
        subject._read_and_validate_prior_failure_receipt(contract)
    if report_path.exists():
        report_path.unlink()
    target.unlink()


def _mutate_preterminal_weighted_reward_by_policy(
    traces: dict[str, Mapping[str, Any]],
    layout: Mapping[str, Any],
    policy_by_value: Mapping[str, float],
) -> None:
    width = len(layout["reward_terms"])
    reward_terms = layout["reward_terms"]
    for policy_name, value in policy_by_value.items():
        weight = reward_terms[0]["weight"] if reward_terms else 0.0
        zero = [0.0] * width
        weighted_row = [float(value), *zero[1:]]
        raw_scale = float(value) / (weight * 0.02) if weight else 0.0
        raw_row = [raw_scale, *zero[1:]]
        trace = traces[policy_name]
        preterminal_rows = len(trace["series"]["reward_weighted"])
        trace["series"]["reward_raw"] = [raw_row.copy() for _ in range(preterminal_rows)]
        trace["series"]["reward_weighted"] = [weighted_row.copy() for _ in range(preterminal_rows)]
        trace["series"]["reward_total"] = [sum(weighted_row) for _ in range(preterminal_rows)]


def test_attribution_summary_accepts_zero_delta_baseline_and_reports_zero_effects() -> None:
    traces, layout, contract = _pairwise_and_attribution_fixtures()
    pairwise = subject._build_pairwise_comparisons(traces=traces, layout=layout, contract=contract)
    attribution = subject._attribution_summary(
        traces=traces,
        comparisons=pairwise,
        layout=layout,
        contract=contract,
    )

    assert attribution["pairwise_comparisons_validated_exact"] is True
    assert attribution["candidate_selected"] is False
    assert attribution["support_qualified"] is False
    assert attribution["promotion_eligible"] is False
    assert attribution["validated_pairwise_comparison_order"] == list(subject.PAIRWISE_COMPARISON_ORDER)
    assert attribution["completed_transitions_by_policy"] == {name: 3 for name in subject.POLICY_ORDER}
    assert attribution["terminal_q9_by_policy"] == {name: 11 for name in subject.POLICY_ORDER}
    assert attribution["absolute_episode_returns_diagnostic_only"] == {name: 0.0 for name in subject.POLICY_ORDER}
    global_summary = attribution["global_preterminal_line_search_reward_summary"]
    assert global_summary["policy_order"] == list(subject.POLICY_ORDER)
    assert global_summary["same_q9_support_for_all_policies"] is True
    assert global_summary["transition_count"] == 2
    assert global_summary["q9_first"] == 9
    assert global_summary["q9_last"] == 10
    direction = attribution["direction_vs_magnitude"]
    assert direction["classification"] == "inconclusive"
    assert direction["reason"] == "no_full_regression"
    assert direction["full_regression"]["regresses"] is False


def test_attribution_summary_rejects_pairwise_comparison_key_set_or_order_or_mapping_drifts() -> None:
    traces, layout, contract = _pairwise_and_attribution_fixtures()
    pairwise = subject._build_pairwise_comparisons(traces=traces, layout=layout, contract=contract)

    missing_key = dict(pairwise)
    missing_key.pop(next(iter(missing_key)))
    with pytest.raises(ValueError, match="ablation attribution summary comparison key mismatch"):
        subject._attribution_summary(
            traces=traces,
            comparisons=missing_key,
            layout=layout,
            contract=contract,
        )

    reversed_order = dict(reversed(list(pairwise.items())))
    with pytest.raises(ValueError, match="ablation attribution summary comparison order mismatch"):
        subject._attribution_summary(
            traces=traces,
            comparisons=reversed_order,
            layout=layout,
            contract=contract,
        )

    drifted = {name: dict(entry) for name, entry in pairwise.items()}
    first_label = subject.PAIRWISE_COMPARISON_ORDER[0]
    drifted[first_label]["left_policy"] = "alpha_plus_0_25"
    with pytest.raises(ValueError, match="ablation attribution summary pairwise policy mapping drift"):
        subject._attribution_summary(
            traces=traces,
            comparisons=drifted,
            layout=layout,
            contract=contract,
        )


def test_attribution_summary_detects_non_shared_q9_support_and_terminal_anchor_changes() -> None:
    traces, layout, contract = _pairwise_and_attribution_fixtures()
    pairwise = subject._build_pairwise_comparisons(traces=traces, layout=layout, contract=contract)
    full = traces["full"]
    full["series"]["q9"] = [10, 11, 12]
    full["terminal_q9"] = 12
    with pytest.raises(ValueError, match="ablation global reward support is not shared by all policies"):
        subject._attribution_summary(
            traces=traces,
            comparisons=pairwise,
            layout=layout,
            contract=contract,
        )


def test_attribution_summary_reflects_reward_and_transition_variation() -> None:
    completed_by_policy = {
        "baseline": 3,
        "alpha_minus_0_25": 3,
        "alpha_plus_0_25": 4,
        "alpha_plus_0_5": 3,
        "full": 2,
    }
    reward_by_policy = {
        "baseline": 0.0,
        "alpha_minus_0_25": 0.0,
        "alpha_plus_0_25": 1.0,
        "alpha_plus_0_5": 0.2,
        "full": -1.0,
    }
    episode_return_by_policy = {
        name: reward_by_policy[name] * completed_by_policy[name] for name in subject.POLICY_ORDER
    }
    traces, layout, contract = _pairwise_and_attribution_fixtures(
        completed_by_policy=completed_by_policy,
        episode_return_by_policy=episode_return_by_policy,
    )
    _mutate_preterminal_weighted_reward_by_policy(traces, layout, reward_by_policy)
    pairwise = subject._build_pairwise_comparisons(traces=traces, layout=layout, contract=contract)
    attribution = subject._attribution_summary(
        traces=traces,
        comparisons=pairwise,
        layout=layout,
        contract=contract,
    )
    direction = attribution["direction_vs_magnitude"]
    assert direction["classification"] == "excessive_update_magnitude"
    assert direction["reason"] == "positive_small_updates_improve_and_negative_does_not"
    assert direction["positive_small_improves"] is True
    assert direction["negative_improves"] is False
    assert direction["full_regression"]["regresses"] is True
    assert attribution["completed_transitions_by_policy"]["full"] == 2
    assert attribution["terminal_q9_by_policy"]["alpha_plus_0_25"] == 12
    assert attribution["completed_transitions_finite_differences"]["policy_order"] == list(subject.POLICY_ORDER)
    assert attribution["terminal_q9_finite_differences"]["policy_order"] == list(subject.POLICY_ORDER)
    global_summary = attribution["global_preterminal_line_search_reward_summary"]
    assert global_summary["q9_first"] == 9
    assert global_summary["q9_last"] == 9
    assert global_summary["transition_count"] == 1
    assert global_summary["total_weighted_reward"]["full"] == -1.0
    assert attribution["absolute_episode_returns_diagnostic_only"] == episode_return_by_policy


@pytest.mark.parametrize("mutator", ("missing", "tampered"))
def test_load_line_search_contract_rejects_alpha_plus_0_25_expected_policy_state_sha256(
    mutator: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = copy.deepcopy(_contract())
    policy_spec = contract["policy_construction"]["policies"]["alpha_plus_0_25"]
    if mutator == "missing":
        policy_spec.pop("expected_policy_state_sha256")
        expected_message = "missing expected policy state hash"
    else:
        policy_spec["expected_policy_state_sha256"] = "0" * 64
        expected_message = "line-search expected policy hash mismatch"
    mutated_contract_path = tmp_path / "corrupted_delta_line_search_contract.json"
    mutated_contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    fake_hash = "f" * 64
    monkeypatch.setattr(subject, "CONTRACT_RELATIVE_PATH", str(mutated_contract_path))
    monkeypatch.setattr(subject, "CONTRACT_SHA256", fake_hash)
    monkeypatch.setattr(subject, "sha256_file", lambda *_args: fake_hash)
    with pytest.raises(ValueError, match=expected_message):
        subject.load_line_search_contract(ROOT)


def test_load_line_search_contract_rejects_reordered_canonical_updated_tensors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = copy.deepcopy(_contract())
    contract["policy_construction"]["canonical_algorithm"]["updated_tensor_names"] = list(
        reversed(contract["policy_construction"]["canonical_algorithm"]["updated_tensor_names"])
    )
    mutated_contract_path = tmp_path / "corrupted_delta_line_search_contract.json"
    mutated_contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    fake_hash = "f" * 64
    monkeypatch.setattr(subject, "CONTRACT_RELATIVE_PATH", str(mutated_contract_path))
    monkeypatch.setattr(subject, "CONTRACT_SHA256", fake_hash)
    monkeypatch.setattr(subject, "sha256_file", lambda *_args: fake_hash)
    with pytest.raises(ValueError):
        subject.load_line_search_contract(ROOT)


@pytest.mark.parametrize(
    ("mutator", "expected_message"),
    (
        ("missing", "line-search prior failure receipt mismatch"),
        ("extra", "line-search prior failure receipt mismatch"),
        ("tampered", "line-search prior failure receipt mismatch"),
    ),
)
def test_load_line_search_contract_rejects_mutated_prior_failure_receipt_metadata(
    mutator: str,
    expected_message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = copy.deepcopy(_contract())
    prior_failure_receipt = dict(contract["prior_failure_receipt"])
    if mutator == "missing":
        prior_failure_receipt.pop("sha256")
    elif mutator == "extra":
        prior_failure_receipt["extra_key"] = "unexpected"
    else:
        prior_failure_receipt["failure_stage"] = "wrong_stage"
    contract["prior_failure_receipt"] = prior_failure_receipt
    mutated_contract_path = tmp_path / "corrupted_delta_line_search_contract.json"
    mutated_contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    fake_hash = "f" * 64
    monkeypatch.setattr(subject, "CONTRACT_RELATIVE_PATH", str(mutated_contract_path))
    monkeypatch.setattr(subject, "CONTRACT_SHA256", fake_hash)
    monkeypatch.setattr(subject, "sha256_file", lambda *_args: fake_hash)
    with pytest.raises(ValueError, match=expected_message):
        subject.load_line_search_contract(ROOT)


def test_construct_interpolated_policy_state_matches_alpha_and_clones_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    checkpoints = _synthetic_checkpoints()
    monkeypatch.setattr(subject, "inspect_true23_policy_state", _fake_policy_hash_factory(checkpoints))

    original_model0 = checkpoints[0]["policy_state_dict"][subject.MODULE14_TENSORS[0]].clone()
    for policy_name in subject.POLICY_ORDER:
        policy = subject.construct_interpolated_policy_state(checkpoints, policy_name, contract)
        assert set(policy) == set(checkpoints[0]["policy_state_dict"])
        for tensor_name, tensor in policy.items():
            expected = _expected_policy_value(policy_name, tensor_name, checkpoints)
            assert tensor.dtype == torch.float32
            assert tensor.device.type == "cpu"
            assert tensor.is_contiguous()
            assert bool(torch.isfinite(tensor).all())
            assert torch.allclose(tensor, expected)
            assert tensor.data_ptr() != checkpoints[0]["policy_state_dict"][tensor_name].data_ptr()
            assert tensor.data_ptr() != checkpoints[1]["policy_state_dict"][tensor_name].data_ptr()

    assert torch.equal(
        checkpoints[0]["policy_state_dict"][subject.MODULE14_TENSORS[0]],
        original_model0,
    )


def test_construct_interpolated_policy_state_preserves_frozen_tensors_and_no_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    checkpoints = _synthetic_checkpoints()
    monkeypatch.setattr(subject, "inspect_true23_policy_state", _fake_policy_hash_factory(checkpoints))
    source0 = checkpoints[0]["policy_state_dict"]
    source1 = checkpoints[1]["policy_state_dict"]
    source0_snapshot = {name: tensor.clone() for name, tensor in source0.items()}
    source1_snapshot = {name: tensor.clone() for name, tensor in source1.items()}
    source0_snapshot_bytes = {name: tensor.numpy().tobytes() for name, tensor in source0_snapshot.items()}
    source1_snapshot_bytes = {name: tensor.numpy().tobytes() for name, tensor in source1_snapshot.items()}
    policies = {
        name: subject.construct_interpolated_policy_state(checkpoints, name, contract)
        for name in subject.POLICY_ORDER
    }
    source0_ptr_set = {int(tensor.data_ptr()) for tensor in source0.values()}
    source1_ptr_set = {int(tensor.data_ptr()) for tensor in source1.values()}

    for policy_name in subject.POLICY_ORDER:
        policy = policies[policy_name]
        assert set(policy) == set(source0_snapshot)
        for tensor_name, tensor in policy.items():
            observed_ptr = int(tensor.data_ptr())
            assert observed_ptr not in source0_ptr_set
            assert observed_ptr not in source1_ptr_set
            if tensor_name not in subject.UPDATED_TENSORS:
                assert torch.equal(tensor, source0_snapshot[tensor_name])
                assert tensor.numpy().tobytes() == source0_snapshot_bytes[tensor_name]
            else:
                assert torch.allclose(tensor, _expected_policy_value(policy_name, tensor_name, checkpoints))

    policy_ptrs = {
        policy_name: [int(tensor.data_ptr()) for tensor in policies[policy_name].values()]
        for policy_name in subject.POLICY_ORDER
    }
    all_ptrs = [ptr for ptrs in policy_ptrs.values() for ptr in ptrs]
    assert len(set(all_ptrs)) == len(all_ptrs)
    policy_ptr_sets = {policy_name: set(ptrs) for policy_name, ptrs in policy_ptrs.items()}
    for policy_name in subject.POLICY_ORDER:
        for other_name in subject.POLICY_ORDER:
            if policy_name == other_name:
                continue
            assert not (policy_ptr_sets[policy_name] & policy_ptr_sets[other_name])

    for name, snapshot in source0_snapshot_bytes.items():
        assert torch.equal(source0[name], source0_snapshot[name])
        assert source0[name].numpy().tobytes() == snapshot
    for name, snapshot in source1_snapshot_bytes.items():
        assert torch.equal(source1[name], source1_snapshot[name])
        assert source1[name].numpy().tobytes() == snapshot


def test_verify_interpolated_policies_produces_boundary_hash_records_with_monkeypatched_inspector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    checkpoints = _synthetic_checkpoints()
    monkeypatch.setattr(subject, "inspect_true23_policy_state", _fake_policy_hash_factory(checkpoints))
    info = subject._verify_interpolated_policies(checkpoints, contract)
    assert info["baseline"]["policy_state_sha256"] == subject.POLICY_SHA256["baseline"]
    assert info["full"]["policy_state_sha256"] == subject.POLICY_SHA256["full"]
    for policy_name in subject.POLICY_ORDER:
        expected_hash = subject.POLICY_SHA256[policy_name]
        assert info[policy_name]["policy_state_sha256"] == expected_hash


def test_line_search_checkpoint_construction_provenance_hash_deterministic_across_pythonhashseed() -> None:
    script = """
import json
import os
from pathlib import Path
from gear_sonic.utils import g1_true23_sonic_task_space_ppo_full_support_delta_line_search as subject

contract = subject.load_line_search_contract(Path(os.environ['REPO_ROOT']))
construction = {
    'policy_order': list(subject.POLICY_ORDER),
    'module14_tensor_names': list(subject.MODULE14_TENSORS),
    'module16_tensor_names': list(subject.MODULE16_TENSORS),
    'all_other_policy_tensors_identical_between_model0_and_model1': True,
    'hybrids_are_in_memory_only': True,
    'updated_tensor_names': list(subject.UPDATED_TENSORS),
    'policies': {
        name: {
            'alpha_numerator': subject.POLICY_ALPHAS[name][0],
            'alpha_denominator': subject.POLICY_ALPHAS[name][1],
            'policy_state_sha256': subject.POLICY_SHA256[name],
            'tensor_count': 4,
            'exact_model0_keyset': True,
            'frozen_tensors_byte_identical_to_model0': True,
            'only_declared_tensors_may_differ': True,
            'cpu_float32_contiguous_finite': True,
            'no_source_or_cross_policy_alias': True,
            'source_checkpoints_unchanged': True,
        }
        for name in subject.POLICY_ORDER
    },
}
subject._resolve_line_search_checkpoints_contract(construction, contract)
print(
    json.dumps(
        {
            'updated_tensor_names': construction['updated_tensor_names'],
            'policy_state_hashes': [
                construction['policies'][name]['policy_state_sha256']
                for name in subject.POLICY_ORDER
            ],
            'hybrid_policy_construction_sha256': subject._canonical_sha256(
                {'hybrid_policy_construction': construction}
            ),
        },
        sort_keys=True,
        separators=(',', ':'),
    )
)
"""
    env = dict(os.environ)
    env["REPO_ROOT"] = str(ROOT)
    outputs: list[bytes] = []
    payloads: list[dict[str, Any]] = []
    for seed in ("1", "2", "3"):
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            check=True,
        )
        outputs.append(result.stdout)
        payloads.append(json.loads(result.stdout.decode("utf-8")))

    assert outputs[0] == outputs[1] == outputs[2]
    payload = payloads[0]
    assert payload["updated_tensor_names"] == list(subject.UPDATED_TENSORS)
    assert payload["policy_state_hashes"] == [subject.POLICY_SHA256[name] for name in subject.POLICY_ORDER]
    assert payload["hybrid_policy_construction_sha256"] == subject._canonical_sha256(
        {
            "hybrid_policy_construction": {
                "policy_order": list(subject.POLICY_ORDER),
                "module14_tensor_names": list(subject.MODULE14_TENSORS),
                "module16_tensor_names": list(subject.MODULE16_TENSORS),
                "all_other_policy_tensors_identical_between_model0_and_model1": True,
                "hybrids_are_in_memory_only": True,
                "updated_tensor_names": list(subject.UPDATED_TENSORS),
                "policies": {
                    name: {
                        "alpha_numerator": subject.POLICY_ALPHAS[name][0],
                        "alpha_denominator": subject.POLICY_ALPHAS[name][1],
                        "policy_state_sha256": subject.POLICY_SHA256[name],
                        "tensor_count": 4,
                        "exact_model0_keyset": True,
                        "frozen_tensors_byte_identical_to_model0": True,
                        "only_declared_tensors_may_differ": True,
                        "cpu_float32_contiguous_finite": True,
                        "no_source_or_cross_policy_alias": True,
                        "source_checkpoints_unchanged": True,
                    }
                    for name in subject.POLICY_ORDER
                },
            }
        }
    )


def test_construct_interpolated_policy_state_rejects_non_identical_source_tensors() -> None:
    contract = _contract()
    checkpoints = _synthetic_checkpoints()
    checkpoints[1]["policy_state_dict"]["actor_module.teleop.module.0.bias"] = torch.tensor(
        [8.0], dtype=torch.float32
    )
    with pytest.raises(ValueError):
        subject.construct_interpolated_policy_state(checkpoints, "baseline", contract)


@pytest.mark.parametrize(
    "mutator",
    (
        "dtype",
        "nonfinite",
        "noncontiguous",
        "keyset",
        "frozen",
    ),
)
def test_construct_interpolated_policy_state_rejects_tampered_source_tensors(mutator: str) -> None:
    contract = _contract()
    checkpoints = copy.deepcopy(_synthetic_checkpoints())
    candidate = subject.UPDATED_TENSORS[0]
    if mutator == "dtype":
        checkpoints[0]["policy_state_dict"][candidate] = torch.tensor([0.0], dtype=torch.float64)
    elif mutator == "nonfinite":
        checkpoints[1]["policy_state_dict"][candidate] = torch.tensor([float("nan")], dtype=torch.float32)
    elif mutator == "noncontiguous":
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).t()
        checkpoints[0]["policy_state_dict"][candidate] = source
        checkpoints[1]["policy_state_dict"][candidate] = source + 10.0
    elif mutator == "keyset":
        checkpoints[1]["policy_state_dict"].pop(candidate)
    else:
        checkpoints[1]["policy_state_dict"]["actor_module.teleop.module.0.bias"] = torch.tensor(
            [9.0], dtype=torch.float32
        )
    with pytest.raises(ValueError):
        subject.construct_interpolated_policy_state(checkpoints, "baseline", contract)


def test_construct_interpolated_policy_state_rejects_mismatched_alpha_plus_0_25_observed_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    checkpoints = _synthetic_checkpoints()

    def _tampered_policy_hash(body: Any, *args: Any, **kwargs: Any) -> str:
        payload = body["policy_state_dict"]
        sample = float(payload[subject.MODULE14_TENSORS[0]][0])
        model0 = checkpoints[0]["policy_state_dict"]
        model1 = checkpoints[1]["policy_state_dict"]
        base = float(model0[subject.MODULE14_TENSORS[0]][0])
        delta = float((model1[subject.MODULE14_TENSORS[0]] - model0[subject.MODULE14_TENSORS[0]])[0])
        alpha = (sample - base) / delta if delta else 0.0
        for policy_name, spec in subject.POLICY_ALPHAS.items():
            target = spec[0] / spec[1]
            if math.isclose(alpha, target, rel_tol=0.0, abs_tol=1e-6):
                if policy_name == "alpha_plus_0_25":
                    return "0" * 64
                return subject.POLICY_SHA256[policy_name]
        return "f" * 64

    monkeypatch.setattr(subject, "inspect_true23_policy_state", _tampered_policy_hash)
    with pytest.raises(ValueError, match="line-search alpha_plus_0_25 contract policy hash mismatch"):
        subject.construct_interpolated_policy_state(checkpoints, "alpha_plus_0_25", contract)


def test_verify_interpolated_policies_rejects_tampered_intermediate_policy_state_sha256() -> None:
    contract = _contract()
    construction = _line_search_checkpoint_construction()
    construction["policies"]["alpha_plus_0_25"]["policy_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="line-search construction policy hash mismatch"):
        subject._resolve_line_search_checkpoints_contract(construction, contract)


def test_validate_action_semantics_matches_safe_target_transform_v2_and_tampering_cases() -> None:
    raw_native_action = [0.0] * 23
    safe_native_action, final_target_hardware = safe_target_transform_numpy(
        np.array(raw_native_action, dtype=np.float32)
    )
    safe_native_action = [float(item) for item in safe_native_action]
    final_target_hardware = [float(item) for item in final_target_hardware]

    subject._validate_action_semantics(
        raw_native_action,
        safe_native_action,
        final_target_hardware,
        context="line-search",
    )

    raw_native_action[-1] = 10.0
    with pytest.raises(ValueError, match="clip threshold"):
        subject._validate_action_semantics(
            raw_native_action,
            safe_native_action,
            final_target_hardware,
            context="line-search",
        )

    mutated_safe = safe_native_action.copy()
    mutated_safe[0] = 10.0
    with pytest.raises(ValueError, match="transform"):
        subject._validate_action_semantics(
            [0.0] * 23,
            mutated_safe,
            final_target_hardware,
            context="line-search",
        )

    mutated_final = final_target_hardware.copy()
    mutated_final[0] = 10.0
    with pytest.raises(ValueError, match="transform"):
        subject._validate_action_semantics(
            [0.0] * 23,
            safe_native_action,
            mutated_final,
            context="line-search",
        )

    with pytest.raises(ValueError):
        subject._validate_action_semantics(
            [True] * 23,
            safe_native_action,
            final_target_hardware,
            context="line-search",
        )
    with pytest.raises(ValueError):
        subject._validate_action_semantics(
            [0] * 23,
            safe_native_action,
            final_target_hardware,
            context="line-search",
        )
    with pytest.raises(ValueError):
        subject._validate_action_semantics(
            raw_native_action,
            [0] * 23,
            final_target_hardware,
            context="line-search",
        )
    with pytest.raises(ValueError):
        subject._validate_action_semantics(
            raw_native_action,
            safe_native_action,
            [0] * 23,
            context="line-search",
        )


def test_validate_endpoint_reproduction_matches_baseline_and_full_exact() -> None:
    contract = _contract()
    for policy_name in ("baseline", "full"):
        expected = contract["expected_endpoint_reproductions"][policy_name]
        observed = subject.validate_endpoint_reproduction(
            _endpoint_trace(policy_name, contract),
            expected,
            policy_name,
        )
        assert observed["policy_name"] == policy_name
        assert observed["exact_match"]
        assert observed["observed_episode_return"] == expected["source_trace_episode_return"]
        assert observed["historical_return"] == expected["historical_return"]


@pytest.mark.parametrize(
    "mutator", ("completed", "terminal_q9", "termination_names", "endpoint_hash", "terminal_wrist")
)
def test_validate_endpoint_reproduction_detects_tampering(mutator: str) -> None:
    contract = _contract()
    expected = contract["expected_endpoint_reproductions"]["baseline"]
    trace = _endpoint_trace("baseline", contract)
    if mutator == "completed":
        expected_trace = copy.deepcopy(trace)
        expected_trace["completed_transitions"] += 1
        with pytest.raises(subject.EndpointReproductionError):
            subject.validate_endpoint_reproduction(expected_trace, expected, "baseline")
        return
    if mutator == "terminal_q9":
        expected_trace = copy.deepcopy(trace)
        expected_trace["terminal_q9"] += 1
        with pytest.raises(subject.EndpointReproductionError):
            subject.validate_endpoint_reproduction(expected_trace, expected, "baseline")
        return
    if mutator == "termination_names":
        expected_trace = copy.deepcopy(trace)
        expected_trace["termination_names"] = ["time_out"]
        with pytest.raises(subject.EndpointReproductionError):
            subject.validate_endpoint_reproduction(expected_trace, expected, "baseline")
        return
    if mutator == "endpoint_hash":
        expected_trace = copy.deepcopy(trace)
        expected_trace["policy_state_sha256"] = "f" * 64
        with pytest.raises(subject.EndpointReproductionError):
            subject.validate_endpoint_reproduction(expected_trace, expected, "baseline")
        return
    mismatched_expected = dict(expected)
    mismatched_expected["terminal_worst_ee_body_name"] = "left_wrist_roll_rubber_hand"
    with pytest.raises(ValueError):
        subject.validate_endpoint_reproduction(trace, mismatched_expected, "baseline")


def test_validate_ablation_trace_minimal_pure_dict_tamper_paths() -> None:
    contract = _contract()
    trace, layout = _minimal_ablation_trace(contract, "baseline")
    subject.validate_ablation_trace(trace, layout, contract)

    for mutator in (
        "raw_bool",
        "safe_nonfloat",
        "final_nonfloat",
        "q9_float",
        "timeout",
        "reward_sum",
        "reward_weight",
    ):
        tampered = copy.deepcopy(trace)
        if mutator == "raw_bool":
            tampered["series"]["raw_native_action"][0][0] = True
        elif mutator == "safe_nonfloat":
            tampered["series"]["safe_native_action"][0][0] = 0
        elif mutator == "final_nonfloat":
            tampered["series"]["final_target_hardware"][0][0] = 0
        elif mutator == "q9_float":
            tampered["series"]["q9"][1] = float(tampered["series"]["q9"][1])
        elif mutator == "timeout":
            tampered["series"]["termination_names"][-1] = ["time_out"]
            tampered["termination_names"] = ["time_out"]
        elif mutator == "reward_sum":
            tampered["series"]["reward_total"][0] = 1.0
        else:
            tampered["series"]["reward_weighted"][0][0] = 1.0
        with pytest.raises(ValueError):
            subject.validate_ablation_trace(tampered, layout, contract)


def test_cli_parser_source_and_ast_avoid_training_mutations() -> None:
    parser = cli._parser()
    assert parser.parse_args(["preflight"]).command == "preflight"
    parsed = parser.parse_args(
        [
            "trace",
            "--output-json",
            str(ROOT / subject.OUTPUT_FILENAME),
            "--repository-root",
            str(ROOT),
            "--parent-run-dir",
            str(ROOT),
        ]
    )
    assert parsed.command == "trace"
    cli_source = inspect.getsource(cli)
    assert "validate_publication_provenance(root, args.parent_run_dir, report)" in cli_source
    assert cli_source.index(
        "validate_publication_provenance(root, args.parent_run_dir, report)"
    ) < cli_source.index("write_json_exclusive(")
    execute_source = inspect.getsource(subject.execute_line_search)
    for token in ("learn", "backward", "save", "train_step", "robot", "network"):
        assert token not in execute_source.lower()


def test_cli_validate_output_rejects_any_non_retry_output_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    target_parent = tmp_path / "runs"
    target_parent_run = target_parent / "run"
    target_parent.mkdir()
    target_parent_run.mkdir()
    contract["parent_run"]["linux_path"] = str(target_parent_run)
    contract["output"]["linux_path"] = str(target_parent / subject.OUTPUT_FILENAME)
    contract["output"]["path"] = str(target_parent / subject.OUTPUT_FILENAME)
    contract["output"]["filename"] = subject.OUTPUT_FILENAME
    monkeypatch.setattr(cli, "load_line_search_contract", lambda *_args: contract)

    assert cli._validate_output(ROOT, target_parent_run, target_parent / subject.OUTPUT_FILENAME) == (
        target_parent / subject.OUTPUT_FILENAME
    )
    (target_parent / subject.OUTPUT_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="line-search output already exists"):
        cli._validate_output(ROOT, target_parent_run, target_parent / subject.OUTPUT_FILENAME)
    with pytest.raises(ValueError, match="line-search output must be exact sibling"):
        cli._validate_output(
            ROOT,
            target_parent_run,
            target_parent / subject.OUTPUT_FILENAME.replace("_retry1", ""),
        )
    with pytest.raises(ValueError, match="line-search output must be exact sibling"):
        cli._validate_output(
            ROOT,
            target_parent_run,
            target_parent / subject.OUTPUT_FILENAME.replace("_retry1", "_retry2"),
        )
    with pytest.raises(ValueError, match="line-search parent run mismatch"):
        other_parent = tmp_path / "other-parent"
        other_parent.mkdir()
        cli._validate_output(ROOT, other_parent, target_parent / subject.OUTPUT_FILENAME)


def test_cli_validate_output_rejects_symlink_output_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract()
    target_parent = tmp_path / "runs"
    linked_parent = tmp_path / "linked_runs"
    target_parent.mkdir()
    linked_parent.symlink_to(target_parent)
    linked_parent_run = linked_parent / "run"
    linked_parent_run.mkdir()
    contract["parent_run"]["linux_path"] = str(linked_parent_run)
    contract["output"]["linux_path"] = str(linked_parent / subject.OUTPUT_FILENAME)
    contract["output"]["path"] = str(linked_parent / subject.OUTPUT_FILENAME)
    contract["output"]["filename"] = subject.OUTPUT_FILENAME
    monkeypatch.setattr(cli, "load_line_search_contract", lambda *_args: contract)
    monkeypatch.setattr(
        cli,
        "_strict_directory",
        lambda path, context="": path.expanduser().absolute(),
    )

    with pytest.raises(ValueError, match="line-search output parent must be regular directory"):
        cli._validate_output(ROOT, linked_parent_run, linked_parent / subject.OUTPUT_FILENAME)


def test_cli_provenance_failure_falls_back_to_failure_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / subject.OUTPUT_FILENAME
    monkeypatch.setattr(cli, "_validate_output", lambda *_args: output)
    monkeypatch.setattr(
        cli,
        "resolve_line_search_inputs",
        lambda *_args: SimpleNamespace(provenance={"snapshot_sha256": "a" * 64}),
    )
    monkeypatch.setattr(cli, "_configure_simulator_runtime", lambda *_args: {})
    monkeypatch.setattr(
        cli,
        "execute_line_search",
        lambda *_args: {
            "schema_version": subject.SCHEMA_VERSION,
            "kind": subject.ABLATION_KIND,
            "evaluation_seed": 20260805,
            "provenance": {"snapshot_sha256": "a" * 64},
            **subject._line_search_boundary_receipt(simulator_constructed=True),
        },
    )
    monkeypatch.setattr(
        cli,
        "validate_publication_provenance",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("publication drift")),
    )
    assert (
        cli.main(
            [
                "trace",
                "--repository-root",
                str(ROOT),
                "--parent-run-dir",
                str(tmp_path),
                "--output-json",
                str(output),
            ]
        )
        == 1
    )
    persisted = {}
    persisted = json.loads(output.read_text(encoding="utf-8"))
    emitted = sorted(entry.name for entry in tmp_path.iterdir() if entry.is_file())
    assert emitted == [subject.OUTPUT_FILENAME]
    assert not any(
        entry.name.startswith(f".{subject.OUTPUT_FILENAME}.") and entry.name.endswith(".tmp")
        for entry in tmp_path.iterdir()
    )
    assert persisted["kind"] == subject.FAILURE_KIND
    assert persisted["kind"] != subject.ABLATION_KIND
    assert "purpose" not in persisted
    assert "traces" not in persisted
    subject._validate_failure_publication(persisted)
    assert persisted["simulator_ablation_complete"] is False
    for key, expected_value in subject._line_search_boundary_receipt(simulator_constructed=None).items():
        assert persisted[key] == expected_value


def _minimal_success_publication(snapshot: str = "a" * 64) -> dict[str, Any]:
    return {
        "schema_version": subject.SCHEMA_VERSION,
        "kind": subject.ABLATION_KIND,
        "evaluation_seed": paired.FIXED_SEED,
        "provenance": {"snapshot_sha256": snapshot},
        **subject._line_search_boundary_receipt(simulator_constructed=True),
    }


def test_write_json_exclusive_rejects_missing_publication_guards_for_success_kind(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    report = _minimal_success_publication(snapshot="f" * 64)
    with pytest.raises(ValueError, match="ablation success report publication requires publication_guard"):
        subject.write_json_exclusive(output, report)
    with pytest.raises(
        ValueError,
        match="ablation success report publication requires publication_receipt_guard",
    ):
        subject.write_json_exclusive(output, report, publication_guard=lambda: "f" * 64)


def test_write_json_exclusive_rejects_stale_publication_guards_without_leaving_outputs(tmp_path: Path) -> None:
    output = tmp_path / "stale.json"
    valid = "a" * 64
    with pytest.raises(ValueError, match="ablation publication guard returned stale provenance"):
        subject.write_json_exclusive(
            output,
            _minimal_success_publication(snapshot=valid),
            publication_guard=lambda: "b" * 64,
            publication_receipt_guard=lambda: valid,
        )
    assert not output.exists()

    output = tmp_path / "stale-receipt.json"
    with pytest.raises(ValueError, match="ablation publication receipt guard returned stale provenance"):
        subject.write_json_exclusive(
            output,
            _minimal_success_publication(snapshot=valid),
            publication_guard=lambda: valid,
            publication_receipt_guard=lambda: "b" * 64,
        )
    assert not output.exists()


def test_write_json_exclusive_rejects_overwrite_for_success_and_failure_reports(tmp_path: Path) -> None:
    failure_output = tmp_path / "failure.json"
    failure_output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        subject.write_json_exclusive(failure_output, subject.failure_report(RuntimeError("intentional")))
    assert failure_output.read_text(encoding="utf-8") == "{}"

    success_output = tmp_path / "success.json"
    success_output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="ablation publication target must not overwrite existing output"):
        subject.write_json_exclusive(
            success_output,
            _minimal_success_publication(snapshot="0" * 64),
            publication_guard=lambda: "0" * 64,
            publication_receipt_guard=lambda: "0" * 64,
        )
    assert success_output.read_text(encoding="utf-8") == "{}"


def _line_search_inputs_for_provenance_snapshot(
    contract: Mapping[str, Any],
    construction: Mapping[str, Any],
    *,
    prior_failure_receipt: Mapping[str, Any] | None = None,
) -> subject.LineSearchInputs:
    source_snapshot = subject.SOURCE_ABLATION_REPORT_PROVENANCE_SHA256
    source_ablation_snapshot = "e" * 64
    resolved_prior_failure_receipt = (
        dict(prior_failure_receipt)
        if prior_failure_receipt is not None
        else dict(_prior_failure_receipt_runtime_identity())
    )
    return subject.LineSearchInputs(
        repository_root=ROOT,
        run_dir=ROOT,
        contract=contract,
        source_ablation_inputs=SimpleNamespace(
            provenance={"snapshot_sha256": source_snapshot},
            run_dir=ROOT,
        ),
        source_ablation_report_path=Path(subject.SOURCE_ABLATION_REPORT_LINUX_PATH),
        source_ablation_report={
            "kind": subject.source_ablation.ABLATION_KIND,
            "provenance": {"snapshot_sha256": source_snapshot},
        },
        provenance={
            "source_ablation_report_inputs_snapshot_sha256": source_ablation_snapshot,
            "hybrid_policy_construction": construction,
            "prior_failure_receipt": resolved_prior_failure_receipt,
            "snapshot_sha256": "f" * 64,
        },
        parent_inputs=SimpleNamespace(),
        source_pair_trace=None,
        source_trace_path=Path(subject.SOURCE_PAIR_TRACE_LINUX_PATH),
        source_trace=None,
    )


def test_current_provenance_snapshot_contract_branch_returns_lowercase_hex_snapshot_when_stubbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    construction = _line_search_checkpoint_construction()
    inputs = _line_search_inputs_for_provenance_snapshot(contract, construction)
    load_calls = {"count": 0}
    source_binding_calls = {"count": 0}
    prior_failure_calls = {"count": 0}

    prior_failure_identity = _prior_failure_receipt_runtime_identity()

    def _counting_read_prior_failure_receipt(_value: Mapping[str, Any]) -> dict[str, Any]:
        prior_failure_calls["count"] += 1
        return copy.deepcopy(prior_failure_identity)

    def _stubbed_load_line_search_contract(_: Any) -> Mapping[str, Any]:
        load_calls["count"] += 1
        return contract

    def _stubbed_regular_file(path: Path, _label: str) -> Path:
        return path

    def _stubbed_sha256_file(path: Path) -> str:
        if str(path) == str(Path(subject.SOURCE_PAIR_TRACE_LINUX_PATH)):
            return subject.SOURCE_PAIR_TRACE_SHA256
        return subject.SOURCE_ABLATION_REPORT_SHA256

    def _stubbed_source_binding(_root: Path) -> Mapping[str, Any]:
        source_binding_calls["count"] += 1
        return {
            "schema_version": subject.SCHEMA_VERSION,
            "kind": "g1_true23_full_support_delta_line_search_executed_sources_v1",
            "file_count": len(subject.SOURCE_RELATIVE_PATHS),
            "total_bytes": 0,
            "manifest_sha256": "f" * 64,
            "files": [],
        }

    monkeypatch.setattr(
        subject,
        "_source_ablation_report",
        lambda _path: (inputs.source_ablation_report, inputs.source_ablation_report_path),
    )
    monkeypatch.setattr(
        subject.paired,
        "_current_provenance_snapshot",
        lambda *_args, **_kwargs: subject.SOURCE_PAIR_PROVENANCE_SHA256,
    )
    monkeypatch.setattr(subject.paired, "_regular_file", _stubbed_regular_file)
    monkeypatch.setattr(
        subject.source_ablation,
        "_current_provenance_snapshot",
        lambda *_args, **_kwargs: inputs.provenance["source_ablation_report_inputs_snapshot_sha256"],
    )
    monkeypatch.setattr(subject, "load_line_search_contract", _stubbed_load_line_search_contract)
    monkeypatch.setattr(subject, "sha256_file", _stubbed_sha256_file)
    monkeypatch.setattr(subject, "_source_binding", _stubbed_source_binding)
    monkeypatch.setattr(
        subject,
        "_read_and_validate_prior_failure_receipt",
        _counting_read_prior_failure_receipt,
    )
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda self, strict=False: self,
    )

    snapshot = subject._current_provenance_snapshot(inputs)
    assert len(snapshot) == 64
    assert all(char in "0123456789abcdef" for char in snapshot)
    assert load_calls["count"] == 1
    assert source_binding_calls["count"] == 1
    assert prior_failure_calls["count"] == 1


def test_mapping_calls_in_line_search_utility_use_context() -> None:
    tree = ast.parse(inspect.getsource(subject))
    bad_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_mapping":
            has_context_kw = any(
                isinstance(keyword, ast.keyword) and keyword.arg == "context" for keyword in node.keywords
            )
            if len(node.args) < 2 and not has_context_kw:
                bad_lines.append(node.lineno)

    assert not bad_lines, f"line-search utility _mapping calls missing context at line(s): {bad_lines}"


def test_line_search_preflight_reports_ready_and_zero_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    construction = _line_search_checkpoint_construction()
    inputs = _line_search_inputs_for_provenance_snapshot(contract, construction)
    preflight_report_path = Path(subject.SOURCE_ABLATION_REPORT_LINUX_PATH)

    def fake_resolve_inputs(_repository_root: Path, _parent_run_dir: Path) -> subject.LineSearchInputs:
        return inputs

    def _no_op_validate(_inputs: subject.LineSearchInputs) -> None:
        return None

    monkeypatch.setattr(subject, "resolve_line_search_inputs", fake_resolve_inputs)
    monkeypatch.setattr(subject, "_validate_line_search_bound_inputs", _no_op_validate)
    monkeypatch.setattr(subject.os.path, "lexists", lambda *_args, **_kwargs: False)
    report = subject.line_search_preflight(ROOT, preflight_report_path)

    assert report["ready"] is True
    assert report["simulator_constructed"] is False
    expected = subject._line_search_boundary_receipt(
        simulator_constructed=False, include_simulator_transitions=True
    )
    for key, expected_value in expected.items():
        assert key in report
        if isinstance(expected_value, bool):
            assert report[key] is expected_value
        else:
            assert report[key] == expected_value == 0
    for key in (
        "teacher_queries",
        "simulator_transitions",
        "training_transitions",
        "training_updates",
        "optimizer_steps",
        "checkpoints_written",
        "robot_commands_issued",
        "network_commands_issued",
        "hardware_actions",
    ):
        assert report[key] == 0
    for key in ("candidate_selected", "hardware_authorized", "robot_or_network_commands_permitted"):
        assert report[key] is False
