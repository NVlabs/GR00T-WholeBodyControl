from __future__ import annotations

import ast
import copy
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import torch

from gear_sonic.scripts import (
    ablate_g1_true23_sonic_task_space_ppo_full_support_update_delta as cli,
)
from gear_sonic.utils import (
    g1_true23_sonic_task_space_ppo_full_support_checkpoint_trace as paired,
    g1_true23_sonic_task_space_ppo_full_support_update_delta_ablation as subject,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    NATIVE_IL23_JOINT_NAMES,
)
from gear_sonic.utils.g1_23dof_safe_target_transform import safe_target_transform_numpy

ROOT = Path(__file__).resolve().parents[2]


def _contract() -> dict[str, Any]:
    return dict(subject.load_ablation_contract(ROOT))


def _layout(contract: dict[str, Any]) -> dict[str, Any]:
    reward_terms = [
        {"name": name, "weight": weight, "callable_identity": identity}
        for name, weight, identity in contract["expected_reward_terms"]
    ]
    return {
        "reward_terms": reward_terms,
        "reward_internal_identity": {
            "manager_class": "ablation_test.RewardManager",
            "step_reward_shape": [1, len(reward_terms)],
            "column_order_verified": True,
            "terms": [
                {
                    "column_index": index,
                    "name": term["name"],
                    "weight": term["weight"],
                    "callable_identity": term["callable_identity"],
                    "parameter_names": [],
                    "public_internal_cfg_object_identical": True,
                }
                for index, term in enumerate(reward_terms)
            ],
        },
        "control_dt_s": 0.02,
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
    }


def _trace(
    policy_name: str,
    completed: int,
    *,
    termination_names: list[str] | None = None,
    terminal_body_index: int = 3,
) -> dict[str, Any]:
    if termination_names is None:
        termination_names = ["ee_body_pos"]
    termination_names = list(termination_names)
    q9s = list(range(9, 9 + completed))
    reward_width = 23
    raw_native = [0.0] * 23
    safe_native, final_target = safe_target_transform_numpy(np.array(raw_native, dtype=np.float32))
    safe_row = [float(value) for value in safe_native.tolist()]
    final_row = [float(value) for value in final_target.tolist()]
    series = {
        "q9": q9s,
        "reward_total": [0.0 for _ in q9s],
        "reward_raw": [[0.0] * reward_width for _ in q9s],
        "reward_weighted": [[0.0] * reward_width for _ in q9s],
        "ee_z_error_m": [[0.0] * len(paired.EE_BODY_NAMES) for _ in q9s],
        "raw_native_action": [list(raw_native) for _ in q9s],
        "safe_native_action": [list(safe_row) for _ in q9s],
        "final_target_hardware": [list(final_row) for _ in q9s],
        "contact_found": [[True, True] for _ in q9s],
        "contact_force_magnitude_n": [[100.0, 100.0] for _ in q9s],
        "landing_force_mean_n": [100.0 for _ in q9s],
        "termination_names": [[] for _ in q9s],
    }
    series["termination_names"][-1] = termination_names
    if "ee_body_pos" in termination_names:
        series["ee_z_error_m"][-1][terminal_body_index] = 0.3
    spec = subject.POLICY_SOURCES[policy_name]
    return {
        "policy_name": policy_name,
        "module14_source_update": spec[0],
        "module16_source_update": spec[1],
        "evaluation_seed": paired.FIXED_SEED,
        "controller": "deterministic_actor_mean",
        "policy_state_sha256": subject.POLICY_SHA256[policy_name],
        "initial_actor_observation_sha256": "a" * 64,
        "completed_transitions": completed,
        "terminal_q9": q9s[-1],
        "termination_names": list(termination_names),
        "episode_return": 0.0,
        "series": series,
    }


def _set_raw_reward(trace: dict[str, Any], layout: dict[str, Any], q9: int, term_name: str, raw: float) -> None:
    row = trace["series"]["q9"].index(q9)
    column = next(index for index, term in enumerate(layout["reward_terms"]) if term["name"] == term_name)
    weight = layout["reward_terms"][column]["weight"]
    weighted = raw * weight * 0.02
    trace["series"]["reward_raw"][row][column] = raw
    trace["series"]["reward_weighted"][row][column] = weighted
    trace["series"]["reward_total"][row] += weighted
    trace["episode_return"] = sum(trace["series"]["reward_total"])


def _success_publication_report(*, provenance_snapshot: str = "a" * 64) -> dict[str, Any]:
    return {
        "schema_version": subject.SCHEMA_VERSION,
        "kind": subject.ABLATION_KIND,
        "evaluation_seed": paired.FIXED_SEED,
        "provenance": {"snapshot_sha256": provenance_snapshot},
        **paired._boundary_receipt(simulator_constructed=True),
    }


def _build_valid_publication_artifact(
    *,
    contract: dict[str, Any] | None = None,
    snapshot: str = "a" * 64,
    source_layout: dict[str, Any] | None = None,
) -> tuple[subject.AblationInputs, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if contract is None:
        contract = _contract()
    trace_layout = _layout(contract)
    traces = {
        "baseline": _trace("baseline", 155),
        "block_only": _trace("block_only", 155),
        "head_only": _trace("head_only", 155),
        "full": _trace("full", 151),
    }
    report_layout = _layout(contract) if source_layout is None else source_layout
    hybrid_policy_construction = {
        "changed_tensor_names": sorted(subject.UPDATED_TENSORS),
        "all_other_policy_tensors_identical": True,
        "policies": {
            name: {
                "module14_source_update": subject.POLICY_SOURCES[name][0],
                "module16_source_update": subject.POLICY_SOURCES[name][1],
                "policy_state_sha256": subject.POLICY_SHA256[name],
                "tensor_count": 1000,
                "fully_cloned_no_checkpoint_alias": True,
            }
            for name in subject.POLICY_ORDER
        },
    }
    sources = {
        "schema_version": 1,
        "kind": "g1_true23_full_support_update_delta_ablation_executed_sources_v1",
        "file_count": 0,
        "total_bytes": 0,
        "manifest_sha256": "c" * 64,
        "files": [],
    }
    provenance = {
        "ablation_contract_sha256": subject.CONTRACT_SHA256,
        "parent_run_dir": str(Path(contract["parent_run"]["linux_path"])),
        "parent_provenance_snapshot_sha256": subject.SOURCE_PAIR_PROVENANCE_SHA256,
        "source_pair_trace_sha256": subject.SOURCE_PAIR_TRACE_SHA256,
        "source_pair_trace_provenance_sha256": subject.SOURCE_PAIR_PROVENANCE_SHA256,
        "executed_ablation_sources": sources,
        "hybrid_policy_construction": hybrid_policy_construction,
    }
    run_dir = Path(contract["parent_run"]["linux_path"])
    source_trace_path = Path(subject.SOURCE_PAIR_TRACE_LINUX_PATH)
    source_trace = {
        "layout": trace_layout,
        "task_audit": {"audit": 1},
        "prime": {"seed": paired.FIXED_SEED},
    }
    provenance["snapshot_sha256"] = snapshot
    inputs = subject.AblationInputs(
        repository_root=ROOT,
        run_dir=run_dir,
        contract=contract,
        parent_inputs=SimpleNamespace(),
        source_trace_path=source_trace_path,
        source_trace=source_trace,
        provenance=provenance,
    )
    initial_actor_observation = {
        "identical_across_all_four_policies": True,
        "sha256": traces["baseline"]["initial_actor_observation_sha256"],
    }
    endpoint_reproduction = {
        name: subject.validate_endpoint_reproduction(
            traces[name],
            contract["expected_endpoint_reproductions"][name],
            name,
        )
        for name in contract["expected_endpoint_reproductions"]
    }
    pairwise_comparisons = {
        label: subject.compare_ablation_pair(
            left,
            traces[left],
            right,
            traces[right],
            report_layout,
            contract,
        )
        for left, right in (("baseline", "block_only"), ("baseline", "head_only"), ("baseline", "full"))
        for label in [f"{left}_vs_{right}"]
    }
    remaining = (
        ("block_only", "head_only"),
        ("block_only", "full"),
        ("head_only", "full"),
    )
    for left, right in remaining:
        pairwise_comparisons[f"{left}_vs_{right}"] = subject.compare_ablation_pair(
            left,
            traces[left],
            right,
            traces[right],
            report_layout,
            contract,
        )
    attribution_summary = subject._attribution_summary(traces, pairwise_comparisons, report_layout)
    report = {
        "schema_version": subject.SCHEMA_VERSION,
        "kind": subject.ABLATION_KIND,
        "purpose": contract["purpose"],
        "evaluation_seed": paired.FIXED_SEED,
        "parent_run_dir": str(run_dir),
        "source_pair_trace": {
            "path": str(source_trace_path),
            "sha256": subject.SOURCE_PAIR_TRACE_SHA256,
            "provenance_snapshot_sha256": subject.SOURCE_PAIR_PROVENANCE_SHA256,
            "validated_comparison_exact": True,
        },
        "provenance": provenance,
        "capture_contract": paired.CAPTURE_CONTRACT,
        "policy_construction": provenance["hybrid_policy_construction"],
        "initial_actor_observation": initial_actor_observation,
        "layout": report_layout,
        "task_audit": source_trace["task_audit"],
        "prime": source_trace["prime"],
        "traces": traces,
        "endpoint_reproduction": endpoint_reproduction,
        "pairwise_comparisons": pairwise_comparisons,
        "attribution_summary": attribution_summary,
        **paired._boundary_receipt(simulator_constructed=True),
    }
    return inputs, report, traces, source_trace


def _synthetic_checkpoints() -> dict[int, dict[str, Any]]:
    frozen = {
        "std": torch.full((23,), 0.1),
        "actor_module.encoders.teleop.module.0.bias": torch.tensor([8.0]),
    }
    model0 = {
        **{name: torch.tensor([float(index)]) for index, name in enumerate(subject.UPDATED_TENSORS)},
        **{name: value.clone() for name, value in frozen.items()},
    }
    model1 = {name: value.clone() for name, value in model0.items()}
    for name in subject.UPDATED_TENSORS:
        model1[name].add_(10.0)
    return {0: {"policy_state_dict": model0}, 1: {"policy_state_dict": model1}}


def _publication_snapshot(
    inputs: subject.AblationInputs,
    report: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    current_snapshot: str = "a" * 64,
) -> str:
    monkeypatch.setattr(subject, "resolve_ablation_inputs", lambda *_args: inputs)
    monkeypatch.setattr(subject, "_current_provenance_snapshot", lambda *_args: current_snapshot)
    return subject.validate_publication_provenance(ROOT, inputs.run_dir, report)


def test_contract_binds_source_trace_hybrids_and_fail_closed_scope() -> None:
    contract = _contract()
    assert subject.CONTRACT_SHA256 == sha256_file(ROOT / subject.CONTRACT_RELATIVE_PATH)
    assert contract["source_pair_trace"]["sha256"] == subject.SOURCE_PAIR_TRACE_SHA256
    assert contract["source_pair_trace"]["provenance_snapshot_sha256"] == (subject.SOURCE_PAIR_PROVENANCE_SHA256)
    assert contract["policy_construction"]["policies"] == {
        name: {
            "module14_source_update": subject.POLICY_SOURCES[name][0],
            "module16_source_update": subject.POLICY_SOURCES[name][1],
            "policy_state_sha256": subject.POLICY_SHA256[name],
        }
        for name in subject.POLICY_ORDER
    }
    assert contract["execution"]["allowed_terminal_terms"] == [
        "anchor_ori",
        "anchor_pos",
        "ee_body_pos",
        "time_out",
    ]
    assert contract["execution"]["safe_target_transform_kind"] == subject.SAFE_TARGET_TRANSFORM_KIND
    assert (
        contract["execution"]["safe_target_transform_application_count"]
        == subject.SAFE_TARGET_TRANSFORM_APPLICATION_COUNT
    )
    assert contract["execution"]["raw_native_action_strict_abs_max"] == subject.SAFE_TARGET_RAW_ACTION_CLIP
    assert contract["comparison"]["pairwise_order"] == [
        "baseline_vs_block_only",
        "baseline_vs_head_only",
        "baseline_vs_full",
        "block_only_vs_head_only",
        "block_only_vs_full",
        "head_only_vs_full",
    ]
    assert contract["comparison"]["global_four_way_preterminal_reward_factorial"] is True
    assert contract["boundaries"] == paired._boundary_receipt(simulator_constructed=None)
    for relative, expected in contract["sealed_sources"].items():
        assert sha256_file(ROOT / relative) == expected


def test_hybrid_construction_replaces_exact_keys_and_never_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = _contract()
    checkpoints = _synthetic_checkpoints()

    def fake_inspect(body: dict[str, Any], **_kwargs: Any) -> str:
        state = body["policy_state_dict"]
        model0 = checkpoints[0]["policy_state_dict"]
        sources = (
            int(not torch.equal(state[subject.MODULE14_TENSORS[0]], model0[subject.MODULE14_TENSORS[0]])),
            int(not torch.equal(state[subject.MODULE16_TENSORS[0]], model0[subject.MODULE16_TENSORS[0]])),
        )
        return next(
            subject.POLICY_SHA256[name] for name, expected in subject.POLICY_SOURCES.items() if expected == sources
        )

    monkeypatch.setattr(subject, "inspect_true23_policy_state", fake_inspect)
    for policy_name, (module14_source, module16_source) in subject.POLICY_SOURCES.items():
        state = subject.construct_hybrid_policy_state(checkpoints, policy_name, contract)
        for name, value in state.items():
            source_update = (
                module14_source
                if name in subject.MODULE14_TENSORS
                else module16_source
                if name in subject.MODULE16_TENSORS
                else 0
            )
            source = checkpoints[source_update]["policy_state_dict"][name]
            assert torch.equal(value, source)
            assert value.data_ptr() != source.data_ptr()
        original = checkpoints[module14_source]["policy_state_dict"][subject.MODULE14_TENSORS[0]].clone()
        state[subject.MODULE14_TENSORS[0]].add_(99.0)
        assert torch.equal(
            checkpoints[module14_source]["policy_state_dict"][subject.MODULE14_TENSORS[0]], original
        )


def test_hybrid_construction_rejects_any_extra_checkpoint_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoints = _synthetic_checkpoints()
    checkpoints[1]["policy_state_dict"]["actor_module.encoders.teleop.module.0.bias"].add_(1.0)
    monkeypatch.setattr(subject, "inspect_true23_policy_state", lambda *_args, **_kwargs: "0" * 64)
    with pytest.raises(ValueError, match="changed tensor set"):
        subject.construct_hybrid_policy_state(checkpoints, "baseline", _contract())


def test_initial_observation_and_imported_reward_seams_are_clone_only() -> None:
    source = inspect.getsource(subject._clone_initial_actor_observation)
    for forbidden in (".cpu(", ".item(", ".tolist(", "bool("):
        assert forbidden not in source
    observations = {"tokenizer": torch.zeros((1, 268)), "policy": torch.ones((1, 930))}
    snapshot = subject._clone_initial_actor_observation(observations)
    observations["policy"].add_(1.0)
    assert torch.all(snapshot["policy"] == 1.0)
    assert snapshot["policy"].data_ptr() != observations["policy"].data_ptr()
    assert subject.paired._RewardComputeTraceRecorder is paired._RewardComputeTraceRecorder
    recorder_source = inspect.getsource(paired._RewardComputeTraceRecorder._snapshot)
    for forbidden in (".cpu(", ".item(", ".tolist(", "bool("):
        assert forbidden not in recorder_source


def test_run_ablation_trace_uses_nested_action_frames_without_flat_key_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert subject.paired._RewardComputeTraceRecorder is paired._RewardComputeTraceRecorder

    class FakeRawEnv:
        def __init__(self) -> None:
            self.num_envs = 1
            self.cfg = SimpleNamespace(seed=paired.FIXED_SEED)
            self.command_manager = SimpleNamespace(get_term=lambda _name: None)
            self.common_step_counter = 0
            self._sim_step_counter = 0

    class FakeWrappedEnv:
        def __init__(self, raw_env: FakeRawEnv) -> None:
            self.unwrapped = raw_env
            self.clip_actions = None
            self.max_episode_length = 510
            self.device = torch.device("cpu")
            self._observations = {
                "tokenizer": torch.zeros((1, 268), dtype=torch.float32),
                "policy": torch.ones((1, 930), dtype=torch.float32),
            }

        def get_observations(self) -> dict[str, torch.Tensor]:
            return dict(self._observations)

        def step(self, _action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
            return (
                dict(self._observations),
                torch.tensor([3.0], dtype=torch.float32),
                torch.tensor([1], dtype=torch.long),
                {},
            )

    class FakePolicy:
        def __init__(self) -> None:
            self.training = True

        def __call__(
            self, _observations: Mapping[str, torch.Tensor], stochastic_output: bool = False
        ) -> torch.Tensor:
            assert stochastic_output is False
            return torch.zeros((1, paired.ACTION_DIM), dtype=torch.float32)

        def eval(self) -> None:
            self.training = False

        def train(self, mode: bool) -> None:
            self.training = mode

        def export_true23_policy_state(self) -> dict[str, torch.Tensor]:
            return {}

    events = {"frame_calls": 0, "recorder_calls": 0}
    raw_action = [0.0] * paired.ACTION_DIM
    reward = {"kind": "nested-action-seam"}

    def fake_q9(_command: Any) -> int:
        return 9

    class FakeRecorder:
        def __init__(self, _env: Any, _reward: Any) -> None:
            events["recorder_calls"] += 1

        def arm(self, q9: int, _actor_action: torch.Tensor) -> None:
            assert q9 == 9 + events["frame_calls"]

        def finish(self) -> dict[str, Any]:
            events["frame_calls"] += 1
            return {
                "q9": 9,
                "reward_total": 3.0,
                "reward_raw": [0.0],
                "reward_weighted": [0.0],
                "ee_z_error_m": [0.0, 0.0, 0.0, 0.0],
                "termination_names": ["ee_body_pos"],
                "episode_length_pre_reset": events["frame_calls"],
                "contact": {
                    "found": [True, True],
                    "force_magnitude_n": [100.0, 100.0],
                    "landing_force_mean_n": 100.0,
                },
                "actions": {
                    "raw_native": list(raw_action),
                    "safe_native": list(raw_action),
                    "final_target_hardware": list(raw_action),
                },
            }

        def restore(self) -> None:
            pass

    contract = {
        "policy_construction": {
            "policies": {
                "baseline": {
                    "policy_state_sha256": "a" * 64,
                    "module14_source_update": 0,
                    "module16_source_update": 0,
                }
            }
        }
    }
    fake_layout = {"reward_internal_identity": {"terms": []}}

    monkeypatch.setattr(subject, "_trace_layout", lambda *_args: (fake_layout, reward))
    monkeypatch.setattr(subject, "validate_ablation_trace", lambda *_args: None)
    monkeypatch.setattr(
        subject,
        "inspect_true23_policy_state",
        lambda *_args, **_kwargs: "a" * 64,
    )
    monkeypatch.setattr(subject.paired, "_q9", fake_q9)
    monkeypatch.setattr(subject.paired, "_RewardComputeTraceRecorder", FakeRecorder)

    output = subject.run_ablation_trace(
        policy=FakePolicy(),
        wrapped_env=FakeWrappedEnv(FakeRawEnv()),
        policy_name="baseline",
        contract=contract,
    )

    assert events["recorder_calls"] == 1
    assert output["layout"] is fake_layout
    assert output["trace"]["completed_transitions"] == 1


def test_trace_validator_enforces_reward_terminal_and_timeout_semantics() -> None:
    contract = _contract()
    layout = _layout(contract)
    trace = _trace("baseline", 5)
    subject.validate_ablation_trace(trace, layout, contract)
    assert all(type(value) is float for value in trace["series"]["landing_force_mean_n"])
    assert trace["termination_names"] == ["ee_body_pos"]
    anchor = _trace("block_only", 5, termination_names=["anchor_pos"])
    subject.validate_ablation_trace(anchor, layout, contract)
    early_timeout = _trace("head_only", 5, termination_names=["time_out"])
    with pytest.raises(ValueError, match="timeout semantics"):
        subject.validate_ablation_trace(early_timeout, layout, contract)
    missing_timeout = _trace("full", 510, termination_names=["anchor_pos"])
    with pytest.raises(ValueError, match="timeout semantics"):
        subject.validate_ablation_trace(missing_timeout, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["reward_raw"][0][0] = 1.0
    with pytest.raises(ValueError, match="raw/weighted"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["q9"][1] = 99
    with pytest.raises(ValueError, match="q9"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["ee_z_error_m"][-1] = [0.1] * 4
    with pytest.raises(ValueError, match="terminal EE"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["q9"][1] = 9.0
    with pytest.raises(ValueError, match="q9"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["q9"][0] = True
    with pytest.raises(ValueError, match="q9"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["reward_total"][0] = True
    with pytest.raises(ValueError, match="scalar series nonfinite"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["reward_total"][0] = torch.tensor(1.0)
    with pytest.raises(ValueError, match="scalar series nonfinite"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["termination_names"][0] = ["ee_body_pos"]
    with pytest.raises(ValueError, match="final terminal"):
        subject.validate_ablation_trace(bad, layout, contract)


def test_trace_validator_enforces_terminal_names_shape_and_summary_scalar_types() -> None:
    contract = _contract()
    layout = _layout(contract)
    trace = _trace("baseline", 5)
    bad = copy.deepcopy(trace)
    bad["series"]["termination_names"][0] = "ee_body_pos"
    with pytest.raises(ValueError, match="termination names must be"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["termination_names"][0] = ("ee_body_pos",)
    with pytest.raises(ValueError, match="termination names must be"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["module14_source_update"] = True
    with pytest.raises(ValueError, match="policy identity mismatch"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["module16_source_update"] = 3.0
    with pytest.raises(ValueError, match="policy identity mismatch"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["evaluation_seed"] = "0"
    with pytest.raises(ValueError, match="policy identity mismatch"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["evaluation_seed"] = True
    with pytest.raises(ValueError, match="policy identity mismatch"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["terminal_q9"] = True
    with pytest.raises(ValueError, match="terminal/return summary mismatch"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["terminal_q9"] = 163.0
    with pytest.raises(ValueError, match="terminal/return summary mismatch"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["episode_return"] = False
    with pytest.raises(ValueError, match="terminal/return summary mismatch"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["episode_return"] = "0.0"
    with pytest.raises(ValueError, match="terminal/return summary mismatch"):
        subject.validate_ablation_trace(bad, layout, contract)


def test_trace_validator_enforces_action_order_reward_internal_and_raw_native_boundaries() -> None:
    contract = _contract()
    layout = _layout(contract)
    trace = _trace("baseline", 5)
    subject.validate_ablation_trace(trace, layout, contract)
    bad = copy.deepcopy(layout)
    bad["action_orders"]["raw_native_action"] = "bad_order"
    with pytest.raises(ValueError, match="action_orders"):
        subject.validate_ablation_trace(trace, bad, contract)
    bad = copy.deepcopy(layout)
    bad["reward_internal_identity"]["manager_class"] = ""
    with pytest.raises(ValueError, match="reward_internal_identity"):
        subject.validate_ablation_trace(trace, bad, contract)
    bad = copy.deepcopy(layout)
    bad["reward_internal_identity"]["step_reward_shape"][0] = True
    with pytest.raises(ValueError, match="step_reward_shape"):
        subject.validate_ablation_trace(trace, bad, contract)
    unit_weight_index = next(
        index
        for index, term in enumerate(bad["reward_terms"])
        if type(term["weight"]) is float and term["weight"] == 1.0
    )
    bad = copy.deepcopy(layout)
    bad["reward_terms"][unit_weight_index]["weight"] = True
    bad["reward_internal_identity"]["terms"][unit_weight_index]["weight"] = True
    with pytest.raises(ValueError, match="weight must be finite float"):
        subject.validate_ablation_trace(trace, bad, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["raw_native_action"][0][0] = subject.SAFE_TARGET_RAW_ACTION_CLIP
    with pytest.raises(ValueError, match="strict clip"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["safe_native_action"][0][0] += 1e-3
    with pytest.raises(ValueError, match="action transform mismatch"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["final_target_hardware"][0][0] += 1e-3
    with pytest.raises(ValueError, match="action transform mismatch"):
        subject.validate_ablation_trace(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["termination_names"][-1] = ["time_out", "ee_body_pos"]
    with pytest.raises(ValueError, match="final"):
        subject.validate_ablation_trace(bad, layout, contract)


def test_trace_validator_enforces_action_transform_consistency_and_nonnegative_metrics() -> None:
    contract = _contract()
    layout = _layout(contract)
    trace = _trace("baseline", 5)

    bad = copy.deepcopy(trace)
    bad["series"]["ee_z_error_m"][0][0] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        subject.validate_ablation_trace(bad, layout, contract)

    bad = copy.deepcopy(trace)
    bad["series"]["contact_force_magnitude_n"][0][0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        subject.validate_ablation_trace(bad, layout, contract)

    bad = copy.deepcopy(trace)
    bad["series"]["landing_force_mean_n"][0] = -1.0
    with pytest.raises(ValueError, match="landing_force_mean_n"):
        subject.validate_ablation_trace(bad, layout, contract)


def test_endpoint_reproduction_is_structural_and_return_is_diagnostic() -> None:
    expected = _contract()["expected_endpoint_reproductions"]["baseline"]
    trace = {
        "completed_transitions": 155,
        "terminal_q9": 163,
        "termination_names": ["ee_body_pos"],
        "episode_return": expected["source_trace_episode_return"] + 3.0,
    }
    diagnostic = subject.validate_endpoint_reproduction(trace, expected, "baseline")
    assert diagnostic["gate_applied"] is False
    assert diagnostic["delta_observed_minus_source_trace"] == 3.0
    trace["terminal_q9"] = 162
    with pytest.raises(subject.EndpointReproductionError) as captured:
        subject.validate_endpoint_reproduction(trace, expected, "baseline")
    assert captured.value.partial_evidence["stage"] == "baseline_endpoint_reproduction"


def test_pair_comparison_is_q9_aligned_and_names_joint_body_site_reward() -> None:
    contract = _contract()
    layout = _layout(contract)
    baseline = _trace("baseline", 5, terminal_body_index=2)
    head = _trace("head_only", 4, terminal_body_index=3)
    row = head["series"]["q9"].index(10)
    head["series"]["raw_native_action"][row][3] = 0.02
    safe_native_action, final_target_hardware = safe_target_transform_numpy(
        np.array(head["series"]["raw_native_action"][row], dtype=np.float32)
    )
    head["series"]["safe_native_action"][row] = [float(value) for value in safe_native_action.tolist()]
    head["series"]["final_target_hardware"][row] = [float(value) for value in final_target_hardware.tolist()]
    row = head["series"]["q9"].index(11)
    head["series"]["ee_z_error_m"][row][1] = 0.026
    head["series"]["contact_found"][row][1] = False
    head["series"]["contact_force_magnitude_n"][row][0] = 120.0
    _set_raw_reward(head, layout, 10, "right_wrist_prethreshold_barrier", 2.0)
    result = subject.compare_ablation_pair("baseline", baseline, "head_only", head, layout, contract)
    assert result["common_q9_first"] == 9
    assert result["common_q9_last"] == 12
    assert result["preterminal_common_q9_last"] == 11
    assert result["exclusive_suffix"]["left"]["q9_first"] == 13
    assert result["exclusive_suffix"]["right"]["transition_count"] == 0
    assert result["action_divergence"]["raw_native_action"]["first_at_or_above"]["0.01"] == {
        "q9": 10,
        "delta": 0.02,
        "culprit_name": NATIVE_IL23_JOINT_NAMES[3],
    }
    assert result["ee_error_divergence"]["first_at_or_above"]["0.025"]["culprit_name"] == (paired.EE_BODY_NAMES[1])
    assert result["contact_state_first_divergence_q9"]["right_foot"] == 11
    assert result["contact_force_divergence"]["maximum"]["culprit_name"] == "left_foot"
    assert result["reward_term_divergence"]["right_wrist_prethreshold_barrier"]["first"]["q9"] == 10
    assert result["terminal_body"]["left"]["worst_body_name"] == paired.EE_BODY_NAMES[2]
    assert result["terminal_body"]["right"]["worst_body_name"] == paired.EE_BODY_NAMES[3]
    assert result["absolute_episode_return_diagnostic_only"]["authorizes_attribution_or_candidate"] is False


def test_global_four_way_reward_factorial_uses_one_shared_preterminal_horizon() -> None:
    contract = _contract()
    layout = _layout(contract)
    traces = {
        "baseline": _trace("baseline", 6),
        "block_only": _trace("block_only", 5),
        "head_only": _trace("head_only", 5),
        "full": _trace("full", 5),
    }
    desired_weighted = {"baseline": 0.0, "block_only": 1.0, "head_only": 2.0, "full": 5.0}
    for name, weighted in desired_weighted.items():
        _set_raw_reward(traces[name], layout, 10, "motion_global_root_pos", weighted / 0.01)
    evidence = subject._global_preterminal_reward_factorial(traces, layout)
    assert evidence["q9_first"] == 9
    assert evidence["q9_last"] == 12
    assert evidence["same_q9_support_for_all_four_policies"] is True
    total = evidence["total_weighted_reward"]
    assert total["values"] == desired_weighted
    assert total["module14_effect_when_module16_at_model0"] == 1.0
    assert total["module16_effect_when_module14_at_model0"] == 2.0
    assert total["module14_module16_interaction"] == 2.0
    term = evidence["weighted_reward_by_term"]["motion_global_root_pos"]
    assert term["module14_module16_interaction"] == 2.0


def test_current_provenance_rehashes_parent_pair_sealed_and_new_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_trace = tmp_path / "pair.json"
    source_trace.write_text("{}")
    sealed_file = tmp_path / "sealed.py"
    sealed_file.write_text("sealed")
    calls = {"parent": 0}

    def parent_snapshot(_inputs: Any) -> str:
        calls["parent"] += 1
        return subject.SOURCE_PAIR_PROVENANCE_SHA256

    monkeypatch.setattr(subject.paired, "_current_provenance_snapshot", parent_snapshot)
    monkeypatch.setattr(subject.paired, "_regular_file", lambda path, _context: Path(path))
    monkeypatch.setattr(
        subject,
        "sha256_file",
        lambda path: subject.SOURCE_PAIR_TRACE_SHA256 if Path(path) == source_trace else "b" * 64,
    )
    monkeypatch.setattr(subject, "_source_binding", lambda _root: {"manifest_sha256": "c" * 64})
    inputs = subject.AblationInputs(
        repository_root=tmp_path,
        run_dir=tmp_path,
        contract={"sealed_sources": {sealed_file.name: "b" * 64}},
        parent_inputs=SimpleNamespace(),
        source_trace_path=source_trace,
        source_trace={},
        provenance={},
    )
    observed = subject._current_provenance_snapshot(inputs)
    assert len(observed) == 64
    assert calls["parent"] == 1


def test_publication_failure_is_scalar_and_writer_is_exclusive(tmp_path: Path) -> None:
    success = _success_publication_report(provenance_snapshot="a" * 64)
    output = tmp_path / "ablation.json"
    with pytest.raises(ValueError, match="requires publication_guard"):
        subject.write_json_exclusive(output, success)
    with pytest.raises(ValueError, match="requires publication_receipt_guard"):
        subject.write_json_exclusive(output, success, publication_guard=lambda: "a" * 64)
    assert not output.exists()
    with pytest.raises(ValueError, match="guard return"):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: None,
            publication_receipt_guard=lambda: "a" * 64,
        )
    with pytest.raises(ValueError, match="guard return"):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: "b" * 64,
            publication_receipt_guard=lambda: "a" * 64,
        )
    with pytest.raises(ValueError, match="guard return"):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: "a" * 64,
            publication_receipt_guard=lambda: "",
        )
    subject.write_json_exclusive(
        output,
        success,
        publication_guard=lambda: "a" * 64,
        publication_receipt_guard=lambda: "a" * 64,
    )
    assert json.loads(output.read_text()) == success
    with pytest.raises(FileExistsError):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: "a" * 64,
            publication_receipt_guard=lambda: "a" * 64,
        )
    failure_output = tmp_path / "failure.json"
    failure = subject.failure_report(
        ValueError("boom"),
        failure_stage="input_resolution",
        parent_run_dir="/root/run",
        provenance_snapshot_sha256="a" * 64,
    )
    subject.write_json_exclusive(failure_output, failure)
    assert failure_output.exists()

    class BadError(ValueError):
        partial_evidence = {"array": [1, 2]}

    with pytest.raises(ValueError, match="scalar"):
        subject.failure_report(BadError("bad"))
    failure = subject.failure_report(
        ValueError("boom"),
        failure_stage="input_resolution",
        parent_run_dir="/root/run",
        provenance_snapshot_sha256="a" * 64,
    )
    assert failure["kind"] == subject.FAILURE_KIND
    assert failure["source_pair_trace_sha256"] == subject.SOURCE_PAIR_TRACE_SHA256
    assert failure["candidate_selected"] is False
    empty_message = subject.failure_report(
        ValueError(),
        failure_stage="input_resolution",
        parent_run_dir="/root/run",
        provenance_snapshot_sha256="a" * 64,
    )
    assert type(empty_message["error"]["message"]) is str


def test_publication_writer_fails_without_partial_artifact_or_temp_leftover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / subject.OUTPUT_FILENAME
    original = set(item.name for item in tmp_path.iterdir())
    success = _success_publication_report(provenance_snapshot="a" * 64)
    original_dumps = subject.json.dumps
    monkeypatch.setattr(
        subject.json,
        "dumps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("unsupported value")),
    )
    with pytest.raises(ValueError, match="unsupported value"):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: "a" * 64,
            publication_receipt_guard=lambda: "a" * 64,
        )
    monkeypatch.setattr(subject.json, "dumps", original_dumps)
    assert not output.exists()
    assert set(item.name for item in tmp_path.iterdir()) == original
    with pytest.raises(RuntimeError, match="pre-publication provenance changed"):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: (_ for _ in ()).throw(RuntimeError("pre-publication provenance changed")),
            publication_receipt_guard=lambda: "a" * 64,
        )
    assert not output.exists()
    assert set(item.name for item in tmp_path.iterdir()) == original


def test_publication_writer_rejects_short_write_fsync_and_prelink_failures_without_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / subject.OUTPUT_FILENAME
    original = set(item.name for item in tmp_path.iterdir())
    success = _success_publication_report(provenance_snapshot="a" * 64)
    original_write = subject._write_payload_or_raise

    monkeypatch.setattr(
        subject,
        "_write_payload_or_raise",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("ablation publication temp write was short")),
    )
    with pytest.raises(RuntimeError, match="temp write was short"):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: "a" * 64,
            publication_receipt_guard=lambda: "a" * 64,
        )
    assert not output.exists()
    assert set(item.name for item in tmp_path.iterdir()) == original

    def short_write_with_partial(stream: Any, payload: bytes) -> None:
        stream.write(payload[:-1])
        raise RuntimeError("ablation publication temp write was short")

    monkeypatch.setattr(subject, "_write_payload_or_raise", short_write_with_partial)
    with pytest.raises(RuntimeError, match="temp write was short"):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: "a" * 64,
            publication_receipt_guard=lambda: "a" * 64,
        )
    assert not output.exists()
    assert set(item.name for item in tmp_path.iterdir()) == original

    monkeypatch.setattr(subject, "_write_payload_or_raise", original_write)
    original_fsync = subject.os.fsync
    monkeypatch.setattr(subject.os, "fsync", lambda *_args: (_ for _ in ()).throw(RuntimeError("fsync failed")))
    with pytest.raises(RuntimeError, match="fsync failed"):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: "a" * 64,
            publication_receipt_guard=lambda: "a" * 64,
        )
    assert not output.exists()
    assert set(item.name for item in tmp_path.iterdir()) == original

    monkeypatch.setattr(subject.os, "fsync", original_fsync)
    monkeypatch.setattr(subject.os, "link", lambda *_args: (_ for _ in ()).throw(RuntimeError("link failed")))
    with pytest.raises(RuntimeError, match="link failed"):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: "a" * 64,
            publication_receipt_guard=lambda: "a" * 64,
        )
    assert not output.exists()
    assert set(item.name for item in tmp_path.iterdir()) == original


def test_publication_writer_rejects_stale_receipt_and_withdraws_target(tmp_path: Path) -> None:
    output = tmp_path / subject.OUTPUT_FILENAME
    success = _success_publication_report(provenance_snapshot="a" * 64)
    with pytest.raises(ValueError, match="receipt guard"):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: "a" * 64,
            publication_receipt_guard=lambda: "b" * 64,
        )
    assert not output.exists()


def test_publication_writer_cleanup_failure_after_publish_does_not_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / subject.OUTPUT_FILENAME
    success = _success_publication_report(provenance_snapshot="a" * 64)

    cleanup_calls = {"count": 0}

    def cleanup_fails(_path: Path) -> None:
        cleanup_calls["count"] += 1
        raise OSError("cleanup failed")

    monkeypatch.setattr(subject.os, "unlink", cleanup_fails)
    subject.write_json_exclusive(
        output,
        success,
        publication_guard=lambda: "a" * 64,
        publication_receipt_guard=lambda: "a" * 64,
    )
    assert output.exists()
    assert json.loads(output.read_text()) == success
    assert cleanup_calls["count"] >= 1


def test_publication_writer_preserves_existing_output_and_raises_on_publish_collision(tmp_path: Path) -> None:
    output = tmp_path / subject.OUTPUT_FILENAME
    success = _success_publication_report(provenance_snapshot="a" * 64)
    output.write_text("{}", encoding="utf-8")
    original_contents = output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: "a" * 64,
            publication_receipt_guard=lambda: "a" * 64,
        )
    assert output.read_text(encoding="utf-8") == original_contents


def test_publication_writer_preserves_existing_symlink_output(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks unsupported")
    output = tmp_path / subject.OUTPUT_FILENAME
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    try:
        os.symlink(target, output)
    except (FileNotFoundError, NotImplementedError, OSError) as exc:
        pytest.skip(str(exc))
    original_contents = target.read_text(encoding="utf-8")
    success = _success_publication_report(provenance_snapshot="a" * 64)
    with pytest.raises(FileExistsError):
        subject.write_json_exclusive(
            output,
            success,
            publication_guard=lambda: "a" * 64,
            publication_receipt_guard=lambda: "a" * 64,
        )
    assert target.read_text(encoding="utf-8") == original_contents


def test_publication_validation_passes_with_full_synthetic_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, report, _, _ = _build_valid_publication_artifact()
    assert _publication_snapshot(inputs, report, monkeypatch) == report["provenance"]["snapshot_sha256"]


def test_exact_json_equality_rejects_recursive_bool_numeric_aliases() -> None:
    expected = {"outer": [{"integer": 0, "float": 0.0}, True]}
    assert subject._exact_json_equal(expected, copy.deepcopy(expected))
    assert not subject._exact_json_equal(expected, {"outer": [{"integer": False, "float": 0.0}, True]})
    assert not subject._exact_json_equal(expected, {"outer": [{"integer": 0, "float": False}, True]})


def test_validate_publication_provenance_rejects_malformed_success_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, report, _, _ = _build_valid_publication_artifact()
    bad_top = dict(report)
    bad_top.pop("purpose")
    with pytest.raises(ValueError, match="schema mismatch"):
        _publication_snapshot(inputs, bad_top, monkeypatch)

    bad_top = dict(report)
    bad_top["unexpected"] = 1
    with pytest.raises(ValueError, match="schema mismatch"):
        _publication_snapshot(inputs, bad_top, monkeypatch)

    bad_initial = copy.deepcopy(report)
    bad_initial["initial_actor_observation"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="initial actor observation"):
        _publication_snapshot(inputs, bad_initial, monkeypatch)

    bad_trace = copy.deepcopy(report)
    bad_trace["traces"]["baseline"]["initial_actor_observation_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="initial actor observation"):
        _publication_snapshot(inputs, bad_trace, monkeypatch)

    bad_endpoint = copy.deepcopy(report)
    bad_endpoint["endpoint_reproduction"]["baseline"]["observed_episode_return"] += 1.0
    with pytest.raises(ValueError, match="endpoint reproduction"):
        _publication_snapshot(inputs, bad_endpoint, monkeypatch)

    bool_alias_endpoint = copy.deepcopy(report)
    observed_return = bool_alias_endpoint["endpoint_reproduction"]["baseline"]["observed_episode_return"]
    assert type(observed_return) is float and observed_return == 0.0
    bool_alias_endpoint["endpoint_reproduction"]["baseline"]["observed_episode_return"] = False
    with pytest.raises(ValueError, match="endpoint reproduction"):
        _publication_snapshot(inputs, bool_alias_endpoint, monkeypatch)

    bad_pairwise = copy.deepcopy(report)
    bad_pairwise["pairwise_comparisons"]["baseline_vs_full"]["action_divergence"]["raw_native_action"]["maximum"][
        "delta"
    ] += 9.9
    with pytest.raises(ValueError, match="pairwise comparison"):
        _publication_snapshot(inputs, bad_pairwise, monkeypatch)

    bad_pairwise = copy.deepcopy(report)
    pre_alias_pairwise_max = bad_pairwise["pairwise_comparisons"]["baseline_vs_full"]["action_divergence"][
        "raw_native_action"
    ]["maximum"]
    pre_alias_delta = pre_alias_pairwise_max["delta"]
    assert type(pre_alias_delta) is float and pre_alias_delta == 0.0
    bad_pairwise["pairwise_comparisons"]["baseline_vs_full"]["action_divergence"]["raw_native_action"]["maximum"][
        "delta"
    ] = False
    with pytest.raises(ValueError, match="pairwise comparison"):
        _publication_snapshot(inputs, bad_pairwise, monkeypatch)

    bad_attribution = copy.deepcopy(report)
    bad_attribution["attribution_summary"]["candidate_selected"] = True
    with pytest.raises(ValueError, match="attribution"):
        _publication_snapshot(inputs, bad_attribution, monkeypatch)

    bad_attribution = copy.deepcopy(report)
    pre_alias_baseline_return = bad_attribution["attribution_summary"]["absolute_episode_returns_diagnostic_only"][
        "baseline"
    ]
    assert type(pre_alias_baseline_return) is float and pre_alias_baseline_return == 0.0
    bad_attribution["attribution_summary"]["absolute_episode_returns_diagnostic_only"]["baseline"] = False
    with pytest.raises(ValueError, match="attribution"):
        _publication_snapshot(inputs, bad_attribution, monkeypatch)

    bad_provenance = copy.deepcopy(report)
    bad_provenance["provenance"]["snapshot_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="report provenance drift"):
        _publication_snapshot(inputs, bad_provenance, monkeypatch)

    bad_policy = copy.deepcopy(report)
    bad_policy["policy_construction"] = copy.deepcopy(report["policy_construction"])
    bad_policy["policy_construction"]["policies"]["baseline"]["policy_state_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="policy construction"):
        _publication_snapshot(inputs, bad_policy, monkeypatch)

    stale_latest = copy.deepcopy(report)
    with pytest.raises(RuntimeError, match="publication provenance changed"):
        _publication_snapshot(inputs, stale_latest, monkeypatch, current_snapshot="b" * 64)


def test_validate_failure_publication_is_strict_and_no_unexpected_fields() -> None:
    failure = subject.failure_report(
        ValueError("boom"),
        failure_stage="input_resolution",
        parent_run_dir="/root/run",
        provenance_snapshot_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        subject._validate_failure_publication({k: v for k, v in failure.items() if k != "candidate_selected"})
    with pytest.raises(ValueError, match="schema mismatch"):
        subject._validate_failure_publication({**failure, "unexpected": 1})
    bad_parent = dict(failure)
    bad_parent["parent_run_dir"] = 123
    with pytest.raises(ValueError, match="parent run"):
        subject._validate_failure_publication(bad_parent)
    bad_snapshot = dict(failure)
    bad_snapshot["preexecution_provenance_snapshot_sha256"] = "not_hex"
    with pytest.raises(ValueError, match="preexecution"):
        subject._validate_failure_publication(bad_snapshot)
    bad_partial = dict(failure)
    bad_partial["partial_scalar_evidence"] = None
    with pytest.raises(ValueError, match="partial scalar evidence"):
        subject._validate_failure_publication(bad_partial)


def test_publication_validators_reject_stale_inputs_and_bool_overrides() -> None:
    success = _success_publication_report(provenance_snapshot="a" * 64)
    with pytest.raises(ValueError, match="schema mismatch"):
        subject._validate_success_publication({"kind": subject.ABLATION_KIND, **success, "schema_version": True})
    with pytest.raises(ValueError, match="seed drift"):
        subject._validate_success_publication({"kind": subject.ABLATION_KIND, **success, "evaluation_seed": 1.0})

    failure = subject.failure_report(
        ValueError("boom"),
        failure_stage="input_resolution",
        parent_run_dir="/root/run",
        provenance_snapshot_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="ablation failure contract mismatch"):
        subject._validate_failure_publication(
            {"kind": subject.FAILURE_KIND, **failure, "ablation_contract_sha256": "b" * 64}
        )
    with pytest.raises(ValueError, match="ablation failure source pair trace mismatch"):
        subject._validate_failure_publication(
            {"kind": subject.FAILURE_KIND, **failure, "source_pair_trace_sha256": "b" * 64}
        )
    with pytest.raises(ValueError, match="schema mismatch"):
        subject._validate_failure_publication(
            {"kind": subject.FAILURE_KIND, **failure, "simulator_constructed": True}
        )


def test_cli_is_pinned_and_runtime_ast_has_no_training_or_checkpoint_calls() -> None:
    parser = cli._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ablate", "--seed", "1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["ablate", "--output-json", "/tmp/x", "--checkpoint", "x"])
    execute_source = inspect.getsource(subject.execute_update_delta_ablation)
    assert execute_source.count("_current_provenance_snapshot(inputs)") == 3
    tree = ast.parse(execute_source)
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not any(
        token in call.lower()
        for call in calls
        for token in ("optim", "backward", "save", "train_step", "robot", "network")
    )
    cli_source = inspect.getsource(cli)
    assert "validate_publication_provenance(root, args.parent_run_dir, report)" in cli_source
    assert cli_source.index(
        "validate_publication_provenance(root, args.parent_run_dir, report)"
    ) < cli_source.index("write_json_exclusive(")


def test_cli_provenance_failure_never_leaves_success_kind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / subject.OUTPUT_FILENAME
    monkeypatch.setattr(cli, "_validate_output", lambda *_args: output)
    monkeypatch.setattr(
        cli,
        "resolve_ablation_inputs",
        lambda *_args: SimpleNamespace(provenance={"snapshot_sha256": "a" * 64}),
    )
    monkeypatch.setattr(cli, "_configure_simulator_runtime", lambda *_args: {})
    monkeypatch.setattr(
        cli,
        "execute_update_delta_ablation",
        lambda *_args: {"kind": subject.ABLATION_KIND, **paired._boundary_receipt(simulator_constructed=True)},
    )
    monkeypatch.setattr(
        cli,
        "validate_publication_provenance",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("changed")),
    )
    assert (
        cli.main(
            [
                "ablate",
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
    persisted = json.loads(output.read_text())
    assert persisted["kind"] == subject.FAILURE_KIND
    assert persisted["simulator_ablation_complete"] is False


def test_cli_validate_success_receipt_returns_snapshot_sha256(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / subject.OUTPUT_FILENAME
    snapshot = "a" * 64
    output.write_text(
        json.dumps(
            {
                "schema_version": subject.SCHEMA_VERSION,
                "kind": subject.ABLATION_KIND,
                "evaluation_seed": paired.FIXED_SEED,
                "provenance": {"snapshot_sha256": snapshot},
                **paired._boundary_receipt(simulator_constructed=True),
            },
            sort_keys=True,
        )
    )
    monkeypatch.setattr(
        cli,
        "validate_publication_provenance",
        lambda *_args: snapshot,
    )
    persisted = json.loads(output.read_text())
    assert cli._validate_success_receipt(output, ROOT, tmp_path) == snapshot
    assert persisted["candidate_selected"] is False


def test_cli_publication_guard_catches_second_stage_provenance_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / subject.OUTPUT_FILENAME
    monkeypatch.setattr(cli, "_validate_output", lambda *_args: output)
    monkeypatch.setattr(
        cli,
        "resolve_ablation_inputs",
        lambda *_args: SimpleNamespace(provenance={"snapshot_sha256": "a" * 64}),
    )
    monkeypatch.setattr(cli, "_configure_simulator_runtime", lambda *_args: {})
    monkeypatch.setattr(
        cli,
        "execute_update_delta_ablation",
        lambda *_args: {
            "schema_version": subject.SCHEMA_VERSION,
            "kind": subject.ABLATION_KIND,
            "evaluation_seed": paired.FIXED_SEED,
            "provenance": {"snapshot_sha256": "a" * 64},
            **paired._boundary_receipt(simulator_constructed=True),
        },
    )
    calls = {"validation": 0}

    def drifting_validation(*_args: Any) -> str:
        calls["validation"] += 1
        if calls["validation"] == 1:
            return "a" * 64
        raise RuntimeError("ablation publication provenance changed")

    monkeypatch.setattr(cli, "validate_publication_provenance", drifting_validation)
    assert (
        cli.main(
            [
                "ablate",
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
    persisted = json.loads(output.read_text())
    assert persisted["kind"] == subject.FAILURE_KIND
    assert persisted["simulator_ablation_complete"] is False
    assert persisted["failure_stage"] == "exclusive_publication"


def test_cli_success_receipt_rejects_persisted_failure_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / subject.OUTPUT_FILENAME
    monkeypatch.setattr(cli, "_validate_output", lambda *_args: output)
    monkeypatch.setattr(
        cli,
        "resolve_ablation_inputs",
        lambda *_args: SimpleNamespace(provenance={"snapshot_sha256": "a" * 64}),
    )
    monkeypatch.setattr(cli, "_configure_simulator_runtime", lambda *_args: {})
    report = {
        "schema_version": subject.SCHEMA_VERSION,
        "kind": subject.ABLATION_KIND,
        "evaluation_seed": paired.FIXED_SEED,
        "provenance": {"snapshot_sha256": "a" * 64},
        **paired._boundary_receipt(simulator_constructed=True),
    }
    monkeypatch.setattr(cli, "execute_update_delta_ablation", lambda *_args: report)
    calls = {"validation": 0}

    def publication_validation(*_args: Any) -> str:
        calls["validation"] += 1
        if calls["validation"] == 1:
            return "a" * 64
        raise RuntimeError("published artifact malformed")

    monkeypatch.setattr(cli, "validate_publication_provenance", publication_validation)

    def write_bad_output(path: Path, *_args: Any, **kwargs: Any) -> None:
        receipt = kwargs.get("publication_receipt_guard")
        path.write_text(
            json.dumps(
                {
                    "schema_version": subject.SCHEMA_VERSION,
                    "kind": subject.FAILURE_KIND,
                    "simulator_ablation_complete": False,
                    **paired._boundary_receipt(simulator_constructed=True),
                }
            )
        )
        try:
            if callable(receipt):
                receipt()
        except Exception:
            if path.exists():
                path.unlink()
            raise

    monkeypatch.setattr(cli, "write_json_exclusive", write_bad_output)
    assert (
        cli.main(
            [
                "ablate",
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
    persisted = json.loads(output.read_text())
    assert persisted["kind"] == subject.FAILURE_KIND
    assert persisted["simulator_ablation_complete"] is False
