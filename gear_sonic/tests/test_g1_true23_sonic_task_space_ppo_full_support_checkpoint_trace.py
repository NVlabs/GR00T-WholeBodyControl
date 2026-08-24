from __future__ import annotations

import ast
import copy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from gear_sonic.scripts import (
    trace_g1_true23_sonic_task_space_ppo_full_support_checkpoints as cli,
)
from gear_sonic.utils import (
    g1_true23_sonic_task_space_ppo_full_support_checkpoint_trace as subject,
)
from gear_sonic.utils.g1_23dof_artifact import sha256_file
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_JOINT_NAMES,
    NATIVE_IL23_JOINT_NAMES,
)
from gear_sonic.utils.g1_true23_sonic_task_space_checkpoint_trace import (
    CONTACT_SITE_NAMES,
    EE_BODY_NAMES,
    _RewardComputeTraceRecorder,
)

ROOT = Path(__file__).resolve().parents[2]


def _contract() -> dict[str, Any]:
    return dict(subject.load_trace_contract(ROOT))


def _layout(contract: dict[str, Any]) -> dict[str, Any]:
    terms = [
        {"name": name, "weight": weight, "callable_identity": identity}
        for name, weight, identity in contract["expected_reward_terms"]
    ]
    return {
        "reward_terms": terms,
        "reward_internal_identity": {"column_order_verified": True},
        "control_dt_s": 0.02,
        "ee_body_names": list(EE_BODY_NAMES),
        "contact_site_names": list(CONTACT_SITE_NAMES),
        "native_joint_names": list(NATIVE_IL23_JOINT_NAMES),
        "hardware_joint_names": list(HARDWARE_23_JOINT_NAMES),
        "action_orders": {
            "raw_native_action": "native_isaaclab_23",
            "safe_native_action": "native_isaaclab_23",
            "final_target_hardware": "hardware_mujoco_23",
        },
        "ee_body_pos_threshold_m": 0.25,
    }


def _trace(update: int, completed: int, policy_hash: str) -> dict[str, Any]:
    q9s = list(range(9, 9 + completed))
    reward_width = 23
    series = {
        "q9": q9s,
        "reward_total": [0.0 for _ in q9s],
        "reward_raw": [[0.0] * reward_width for _ in q9s],
        "reward_weighted": [[0.0] * reward_width for _ in q9s],
        "ee_z_error_m": [[0.0] * len(EE_BODY_NAMES) for _ in q9s],
        "raw_native_action": [[0.0] * 23 for _ in q9s],
        "safe_native_action": [[0.0] * 23 for _ in q9s],
        "final_target_hardware": [[0.0] * 23 for _ in q9s],
        "contact_found": [[True, True] for _ in q9s],
        "contact_force_magnitude_n": [[100.0, 100.0] for _ in q9s],
        "landing_force_mean_n": [100.0 for _ in q9s],
        "termination_names": [[] for _ in q9s],
    }
    series["termination_names"][-1] = ["ee_body_pos"]
    series["ee_z_error_m"][-1][2 if update == 0 else 3] = 0.3
    return {
        "update_count": update,
        "evaluation_seed": 20260805,
        "controller": "deterministic_actor_mean",
        "policy_state_sha256": policy_hash,
        "completed_transitions": completed,
        "terminal_q9": q9s[-1],
        "termination_names": ["ee_body_pos"],
        "episode_return": 0.0,
        "series": series,
    }


def _set_raw_reward(trace: dict[str, Any], layout: dict[str, Any], q9: int, term_name: str, raw: float) -> None:
    row = trace["series"]["q9"].index(q9)
    term = next(index for index, value in enumerate(layout["reward_terms"]) if value["name"] == term_name)
    weight = layout["reward_terms"][term]["weight"]
    weighted = raw * weight * 0.02
    trace["series"]["reward_raw"][row][term] = raw
    trace["series"]["reward_weighted"][row][term] = weighted
    trace["series"]["reward_total"][row] += weighted
    trace["episode_return"] = sum(trace["series"]["reward_total"])


def test_contract_and_parent_source_chain_are_exact() -> None:
    contract = _contract()
    assert subject.TRACE_CONTRACT_SHA256 == sha256_file(ROOT / subject.TRACE_CONTRACT_RELATIVE_PATH)
    assert contract["execution"]["updates"] == [0, 1]
    assert contract["expected_reproductions"] == {
        "0": {
            "policy_state_sha256": "358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61",
            "completed_transitions": 155,
            "terminal_q9": 163,
            "termination_names": ["ee_body_pos"],
            "historical_episode_return": -123.29479291848838,
        },
        "1": {
            "policy_state_sha256": "7299df1851c5b42256f170334e2d5afdc81603b17ae832381abadadd9ea48639",
            "completed_transitions": 151,
            "terminal_q9": 159,
            "termination_names": ["ee_body_pos"],
            "historical_episode_return": -115.07444967143238,
        },
    }
    assert len(contract["parent_run"]["artifact_sha256"]) == 13
    assert "full_support_failure.json" not in contract["parent_run"]["artifact_sha256"]
    for relative, expected in contract["sealed_sources"].items():
        assert sha256_file(ROOT / relative) == expected


def test_imported_reward_seam_is_exact_clone_only_v3() -> None:
    assert subject._RewardComputeTraceRecorder is _RewardComputeTraceRecorder
    snapshot_source = inspect.getsource(_RewardComputeTraceRecorder._snapshot)
    assert "_clone_tensor_snapshot" in snapshot_source
    for forbidden in (".cpu(", ".item(", ".tolist(", "bool("):
        assert forbidden not in snapshot_source
    observed_compute = inspect.getsource(_RewardComputeTraceRecorder.__init__)
    assert "reward = self._original_compute(dt)" in observed_compute
    assert observed_compute.count("self._original_compute(dt)") == 1
    assert "self._captured_snapshot = self._snapshot" in observed_compute


def test_reward_internal_identity_binds_public_object_callable_and_column() -> None:
    def first(_env: Any) -> torch.Tensor:
        return torch.zeros(1)

    def second(_env: Any) -> torch.Tensor:
        return torch.zeros(1)

    cfgs = [
        SimpleNamespace(func=first, weight=1.0, params={"alpha": 1}),
        SimpleNamespace(func=second, weight=-2.0, params={}),
    ]

    class Manager:
        active_terms = ["first", "second"]
        _term_cfgs = cfgs
        _step_reward = torch.zeros((1, 2))

        def get_term_cfg(self, name: str) -> Any:
            return self._term_cfgs[self.active_terms.index(name)]

    contract = {
        "expected_reward_terms": [
            ["first", 1.0, f"{first.__module__}:{first.__qualname__}"],
            ["second", -2.0, f"{second.__module__}:{second.__qualname__}"],
        ]
    }
    identity = subject.reward_internal_identity(SimpleNamespace(reward_manager=Manager()), contract)
    assert identity["column_order_verified"] is True
    assert [term["column_index"] for term in identity["terms"]] == [0, 1]
    assert all(term["public_internal_cfg_object_identical"] for term in identity["terms"])
    bad = copy.deepcopy(contract)
    bad["expected_reward_terms"][1][2] = "wrong:callable"
    with pytest.raises(ValueError, match="callable identity"):
        subject.reward_internal_identity(SimpleNamespace(reward_manager=Manager()), bad)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_transitions", 154),
        ("terminal_q9", 162),
        ("termination_names", ["anchor_pos"]),
        ("policy_state_sha256", "f" * 64),
    ],
)
def test_structural_reproduction_is_exact_and_return_is_diagnostic(field: str, value: Any) -> None:
    expected = _contract()["expected_reproductions"]["0"]
    trace = {
        "completed_transitions": 155,
        "terminal_q9": 163,
        "termination_names": ["ee_body_pos"],
        "policy_state_sha256": expected["policy_state_sha256"],
        "episode_return": expected["historical_episode_return"] + 1.0,
    }
    diagnostic = subject.validate_structural_reproduction(trace, expected, 0)
    assert diagnostic["gate_applied"] is False
    assert diagnostic["delta_observed_minus_historical"] == 1.0
    trace[field] = value
    with pytest.raises(subject.TraceReproductionError) as captured:
        subject.validate_structural_reproduction(trace, expected, 0)
    assert captured.value.partial_evidence["stage"] == "update0_structural_reproduction"
    subject._assert_scalar_evidence(captured.value.partial_evidence)


def test_trace_series_rejects_reward_and_q9_corruption() -> None:
    contract = _contract()
    layout = _layout(contract)
    trace = _trace(0, 155, contract["expected_reproductions"]["0"]["policy_state_sha256"])
    subject.validate_trace_series(trace, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["q9"][5] = 99
    with pytest.raises(ValueError, match="q9 discontinuity"):
        subject.validate_trace_series(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["reward_raw"][0][0] = 1.0
    with pytest.raises(ValueError, match="raw/weighted"):
        subject.validate_trace_series(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["termination_names"][-2] = ["ee_body_pos"]
    with pytest.raises(ValueError, match="one final terminal"):
        subject.validate_trace_series(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["evaluation_seed"] = 1
    with pytest.raises(ValueError, match="pair"):
        subject.validate_trace_series(bad, layout, contract)
    bad = copy.deepcopy(trace)
    bad["series"]["ee_z_error_m"][-1] = [0.1] * 4
    with pytest.raises(ValueError, match="EE threshold"):
        subject.validate_trace_series(bad, layout, contract)


def test_pair_comparison_is_q9_keyed_and_names_joint_body_site_and_suffix() -> None:
    contract = _contract()
    layout = _layout(contract)
    model0 = _trace(0, 155, contract["expected_reproductions"]["0"]["policy_state_sha256"])
    model1 = _trace(1, 151, contract["expected_reproductions"]["1"]["policy_state_sha256"])
    row = model1["series"]["q9"].index(20)
    model1["series"]["raw_native_action"][row][3] = 0.02
    row = model1["series"]["q9"].index(30)
    model1["series"]["safe_native_action"][row][4] = 0.06
    row = model1["series"]["q9"].index(40)
    model1["series"]["final_target_hardware"][row][5] = 0.11
    row = model1["series"]["q9"].index(50)
    model1["series"]["ee_z_error_m"][row][1] = 0.026
    row = model1["series"]["q9"].index(60)
    model1["series"]["contact_found"][row][1] = False
    row = model1["series"]["q9"].index(70)
    model1["series"]["contact_force_magnitude_n"][row][0] = 110.0
    _set_raw_reward(model0, layout, 155, "right_wrist_prethreshold_barrier", 2.0)
    _set_raw_reward(model1, layout, 150, "right_wrist_prethreshold_barrier", 3.0)
    _set_raw_reward(model0, layout, 160, "worst_ee_z_normalized_squared", 4.0)
    subject.validate_trace_series(model0, layout, contract)
    subject.validate_trace_series(model1, layout, contract)
    result = subject.compare_trace_pair(model0, model1, layout, contract)
    assert result["common_q9_first"] == 9
    assert result["common_q9_last"] == 159
    assert result["preterminal_common_q9_last"] == 158
    assert result["model0_only_suffix_q9_first"] == 160
    assert result["model0_only_suffix_q9_last"] == 163
    assert result["action_divergence"]["raw_native_action"]["first_at_or_above"]["0.01"] == {
        "q9": 20,
        "delta": 0.02,
        "culprit_name": NATIVE_IL23_JOINT_NAMES[3],
    }
    assert (
        result["action_divergence"]["safe_native_action"]["first_at_or_above"]["0.05"]["culprit_name"]
        == NATIVE_IL23_JOINT_NAMES[4]
    )
    assert (
        result["action_divergence"]["final_target_hardware"]["first_at_or_above"]["0.1"]["culprit_name"]
        == HARDWARE_23_JOINT_NAMES[5]
    )
    assert result["ee_error_divergence"]["first_at_or_above"]["0.025"]["culprit_name"] == (EE_BODY_NAMES[1])
    assert result["contact_state_first_divergence_q9"]["right_foot"] == 60
    assert result["contact_force_divergence"]["maximum"]["culprit_name"] == "left_foot"
    assert result["landing_force_divergence"]["maximum_q9"] == 9
    assert result["reward_total_divergence"]["first_q9"] == 150
    assert result["reward_term_divergence"]["right_wrist_prethreshold_barrier"]["first_q9"] == 150
    assert result["terminal_ee_culprit"]["model0"]["body_name"] == EE_BODY_NAMES[2]
    assert result["terminal_ee_culprit"]["model1"]["body_name"] == EE_BODY_NAMES[3]
    assert result["model0_only_suffix"]["q9"] == [160, 161, 162, 163]


def test_publication_boundary_failure_scalars_and_exclusive_writer(tmp_path: Path) -> None:
    report = {"kind": subject.TRACE_KIND, **subject._boundary_receipt(simulator_constructed=False)}
    output = tmp_path / "trace.json"
    subject.write_json_exclusive(output, report)
    assert json.loads(output.read_text()) == report
    with pytest.raises(FileExistsError):
        subject.write_json_exclusive(output, report)
    for key, bad_value in (
        ("candidate_selected", True),
        ("optimizer_steps", 1),
        ("training_transitions", 1),
        ("checkpoints_written", 1),
    ):
        bad = dict(report)
        bad[key] = bad_value
        with pytest.raises(ValueError, match="boundary"):
            subject.assert_publication_boundary(bad)

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
    assert failure["trace_contract_sha256"] == subject.TRACE_CONTRACT_SHA256
    assert failure["failure_stage"] == "input_resolution"
    assert failure["preexecution_provenance_snapshot_sha256"] == "a" * 64
    assert "simulator_constructed" not in failure


def test_cli_is_pinned_and_runtime_ast_has_no_mutating_calls() -> None:
    parser = cli._parser()
    assert parser.parse_args(["preflight"]).command == "preflight"
    parsed = parser.parse_args(["trace", "--output-json", f"/root/g1_true23_runs/{subject.OUTPUT_FILENAME}"])
    assert parsed.command == "trace"
    actions = {option for action in parser._actions for option in action.option_strings}
    assert actions <= {"-h", "--help"}
    for subparser in parser._subparsers._group_actions[0].choices.values():
        options = {option for action in subparser._actions for option in action.option_strings}
        assert options <= {"-h", "--help", "--repository-root", "--parent-run-dir", "--output-json"}

    utility_tree = ast.parse(inspect.getsource(subject))
    cli_tree = ast.parse(inspect.getsource(cli))
    forbidden_attributes = {"learn", "backward"}
    for tree in (utility_tree, cli_tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attributes
            if isinstance(node, ast.Attribute):
                assert not (
                    isinstance(node.value, ast.Attribute)
                    and node.value.attr == "optim"
                    and node.attr in {"Adam", "AdamW", "SGD"}
                )
    execute_source = inspect.getsource(subject.execute_checkpoint_trace)
    assert "for update in EXPECTED_UPDATES:" in execute_source
    assert execute_source.index("_seed_everything()") < execute_source.index("env = ManagerBasedRlEnv")
    assert "finally:\n            env.close()" in execute_source
    run_source = inspect.getsource(subject.run_policy_trace)
    assert "full-support trace wrapped reward identity mismatch" in run_source
    main_source = inspect.getsource(cli.main)
    assert main_source.count("validate_publication_provenance(") == 1


def test_cli_provenance_failure_never_leaves_success_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / subject.OUTPUT_FILENAME
    fake_inputs = SimpleNamespace(provenance={"snapshot_sha256": "a" * 64})
    monkeypatch.setattr(cli, "_validate_output", lambda *_args: output)
    monkeypatch.setattr(cli, "resolve_trace_inputs", lambda *_args: fake_inputs)
    monkeypatch.setattr(cli, "_configure_simulator_runtime", lambda *_args: {})
    monkeypatch.setattr(
        cli,
        "execute_checkpoint_trace",
        lambda *_args: {"kind": subject.TRACE_KIND, "provenance": fake_inputs.provenance},
    )

    def reject(*_args: Any) -> None:
        raise RuntimeError("adversarial provenance drift")

    monkeypatch.setattr(cli, "validate_publication_provenance", reject)
    result = cli.main(
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
    assert result == 1
    written = json.loads(output.read_text())
    assert written["kind"] == subject.FAILURE_KIND
    assert written["failure_stage"] == "prepublication_provenance"
    assert written["preexecution_provenance_snapshot_sha256"] == "a" * 64
    assert written["kind"] != subject.TRACE_KIND


def test_current_source_snapshot_contains_all_additive_files() -> None:
    manifest = subject._trace_source_binding(ROOT)
    logical = {record["logical_path"] for record in manifest["files"]}
    assert set(subject.TRACE_SOURCE_RELATIVE_PATHS).issubset(logical)
    assert manifest["file_count"] == len(logical)
    assert len(manifest["manifest_sha256"]) == 64
    assert math.isfinite(float(manifest["total_bytes"]))
    snapshot_source = inspect.getsource(subject._current_provenance_snapshot)
    assert "_verify_material_execution_inputs(inputs.repository_root, inputs.material)" in (snapshot_source)
