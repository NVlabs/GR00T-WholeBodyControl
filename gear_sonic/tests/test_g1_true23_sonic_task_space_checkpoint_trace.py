from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from gear_sonic.scripts import trace_g1_true23_sonic_task_space_checkpoints as cli
from gear_sonic.utils.g1_23dof_artifact import sha256_file
import gear_sonic.utils.g1_true23_sonic_task_space_checkpoint_trace as trace

ROOT = Path(__file__).resolve().parents[2]


def _layout() -> dict[str, Any]:
    return {
        "reward_terms": [
            {"name": "alive", "weight": 5.0},
            {"name": "non_timeout_termination", "weight": -5000.0},
        ],
        "control_dt_s": 0.02,
        "ee_body_names": list(trace.EE_BODY_NAMES),
        "contact_site_names": list(trace.CONTACT_SITE_NAMES),
        "action_orders": {
            "raw_native_action": "native_isaaclab_23",
            "safe_native_action": "native_isaaclab_23",
            "final_target_hardware": "hardware_mujoco_23",
        },
    }


def _row(value: float, width: int = trace.ACTION_DIM) -> list[float]:
    return [value] * width


def _synthetic_trace(update: int, completed: int) -> dict[str, Any]:
    reward_total: list[float] = []
    reward_raw: list[list[float]] = []
    reward_weighted: list[list[float]] = []
    terminations: list[list[str]] = []
    for index in range(completed):
        terminal = index == completed - 1
        reward_total.append(-100.0 if terminal else 0.1)
        reward_raw.append([0.0, 1.0] if terminal else [1.0, 0.0])
        reward_weighted.append([0.0, -100.0] if terminal else [0.1, -0.0])
        terminations.append(["ee_body_pos"] if terminal else [])
    raw_actions = [_row(0.0) for _ in range(completed)]
    ee = [[0.01, 0.01, 0.01, 0.01] for _ in range(completed)]
    contacts = [[True, True] for _ in range(completed)]
    forces = [[100.0, 100.0] for _ in range(completed)]
    landing = [0.0] * completed
    if update == 5:
        raw_actions[0][0] = 2.0e-6
        ee[-1][-1] = 0.26
        contacts[-1] = [True, False]
        forces[-1][-1] = 130.0
        landing[-1] = 130.0
    series = {
        "q9": list(range(9, 9 + completed)),
        "reward_total": reward_total,
        "reward_raw": reward_raw,
        "reward_weighted": reward_weighted,
        "ee_z_error_m": ee,
        "raw_native_action": raw_actions,
        "safe_native_action": copy.deepcopy(raw_actions),
        "final_target_hardware": copy.deepcopy(raw_actions),
        "contact_found": contacts,
        "contact_force_magnitude_n": forces,
        "landing_force_mean_n": landing,
        "termination_names": terminations,
    }
    return {
        "update_count": update,
        "evaluation_seed": trace.FIXED_SEED,
        "controller": "deterministic_actor_mean",
        "policy_state_sha256": trace.EXPECTED_POLICY_SHA256[update],
        "completed_transitions": completed,
        "terminal_q9": 9 + completed - 1,
        "termination_names": ["ee_body_pos"],
        "episode_return": sum(reward_total),
        "series": series,
    }


def test_parent_artifact_and_sealed_control_hashes_are_exact() -> None:
    assert trace.EXPECTED_CONTRACT_SHA256 == ("a0225ad7d08d7037ec3782d545961a1340b1efef540bda5737b37b7c987d20eb")
    assert trace.EXPECTED_MATERIAL_SHA256 == ("92b934c91fb58fd8955054e1ccbfa92b3d49e56aa3c8948e64e791fce17a3ef2")
    assert trace.PARENT_RUN_FILE_SHA256["checkpoints/sonic_task_space_model_0.pt"] == (
        "c448882d753f6a594dee578fac437ed5cc17889378a037d47095e9a1958ec8f2"
    )
    assert trace.PARENT_RUN_FILE_SHA256["checkpoints/sonic_task_space_model_5.pt"] == (
        "99b9a19d26f9a1ae9d685bde73faaf2842983b64551d447c65879ba97071cdbd"
    )
    assert trace.PRIOR_FAILED_TRACE_SHA256 == ("f982aceb0a2ba996193ae8c02e8f27c75b6b86f12bd6a96dd66a71e55d961401")
    assert trace.SECOND_FAILED_TRACE_FILENAME == "sonic_task_space_ckpt0_vs_ckpt5_trace_v2.json"
    assert trace.SECOND_FAILED_TRACE_SHA256 == ("eef3441cf4e9af1c343bbc43862eec40ed52c20ceb31bf553209dc92bc852059")
    assert trace.RETRY_TRACE_FILENAME == "sonic_task_space_ckpt0_vs_ckpt5_trace_v3.json"
    assert trace.CAPTURE_CONTRACT == "gpu_clone_at_reward_compute_cpu_materialize_after_wrapped_step_v1"
    for logical, expected in trace.SEALED_CURRENT_CONTROL_SHA256.items():
        assert sha256_file(ROOT / logical) == expected


def test_reward_decoder_recovers_raw_and_dt_weighted_contributions() -> None:
    raw, weighted = trace.decode_reward_terms(
        weighted_rates=[5.0, -5000.0],
        weights=[5.0, -5000.0],
        dt=0.02,
        reward_total=-99.9,
    )
    assert raw == [1.0, 1.0]
    assert weighted == [0.1, -100.0]
    with pytest.raises(ValueError, match="do not sum"):
        trace.decode_reward_terms(
            weighted_rates=[5.0],
            weights=[5.0],
            dt=0.02,
            reward_total=2.0,
        )
    with pytest.raises(ValueError, match="zero-weight"):
        trace.decode_reward_terms(
            weighted_rates=[0.0],
            weights=[0.0],
            dt=0.02,
            reward_total=0.0,
        )


def test_trace_pair_comparison_uses_common_and_preterminal_horizons() -> None:
    update0 = _synthetic_trace(0, 3)
    update5 = _synthetic_trace(5, 2)
    comparison = trace.compare_trace_pair(update0, update5, _layout())
    assert comparison["common_transition_count"] == 2
    assert comparison["common_q9_last"] == 10
    assert comparison["preterminal_common_transition_count"] == 1
    assert comparison["preterminal_common_q9_last"] == 9
    assert comparison["first_divergence_q9"] == 9
    assert comparison["first_divergence_by_category_q9"] == {
        "raw_action": 9,
        "safe_action": 9,
        "final_target": 9,
        "ee_z_error": 10,
        "reward_total": 10,
        "contact_state": 10,
        "contact_force": 10,
        "landing_force": 10,
        "termination": 10,
    }
    common_rewards = comparison["common_reward_by_term"]
    assert common_rewards["preterminal_common_prefix"]["update5_minus_update0"] == [0.0, 0.0]
    assert common_rewards["through_terminal_update5"]["update5_minus_update0"] == [-0.1, -100.0]
    assert comparison["update5_terminal_worst_ee"] == {
        "body_name": "right_wrist_roll_rubber_hand",
        "z_error_m": 0.26,
    }


def test_trace_layout_is_exact_and_zero_weight_is_forbidden() -> None:
    trace.validate_trace_layout(_layout())
    broken = _layout()
    broken["ee_body_names"] = list(reversed(broken["ee_body_names"]))
    with pytest.raises(ValueError, match="EE bodies"):
        trace.validate_trace_layout(broken)
    broken = _layout()
    broken["reward_terms"][0]["weight"] = 0.0
    with pytest.raises(ValueError, match="weight"):
        trace.validate_trace_layout(broken)


def test_reproduction_failure_publishes_only_scalar_diagnostics() -> None:
    result = _synthetic_trace(0, 3)
    expected = {
        "completed_transitions": 4,
        "terminal_q9": 12,
        "termination_names": ["ee_body_pos"],
        "episode_return": -123.0,
        "policy_state_sha256": trace.EXPECTED_POLICY_SHA256[0],
    }
    evidence = trace.reproduction_scalar_evidence(
        result=result,
        layout=_layout(),
        expected_evaluation=expected,
        update_count=0,
        provenance_snapshot_sha256="a" * 64,
    )
    assert evidence["observed"]["completed_transitions"] == 3
    assert evidence["expected"]["completed_transitions"] == 4
    assert evidence["delta_observed_minus_expected"]["completed_transitions"] == -1
    assert evidence["mismatch"]["completed_transitions"] is True
    assert "episode_return" not in evidence["mismatch"]
    assert evidence["episode_return_diagnostic"] == {
        "update_count": 0,
        "observed_episode_return": -99.8,
        "historical_parent_episode_return": -123.0,
        "delta_observed_minus_historical": pytest.approx(23.2),
        "exact_match": False,
        "gate_applied": False,
        "reason": "mujoco_warp_cuda_return_is_not_cross_replay_deterministic",
    }
    assert evidence["terminal_ee_z_error_m"]["right_wrist_roll_rubber_hand"] == 0.01
    assert (
        evidence["hashes"]["checkpoint_file_sha256"]
        == (trace.PARENT_RUN_FILE_SHA256["checkpoints/sonic_task_space_model_0.pt"])
    )

    def reject_sequences(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                reject_sequences(child)
            return
        assert not isinstance(value, (list, tuple))

    reject_sequences(evidence)
    error = trace.TraceReproductionError("update0 mismatch", evidence)
    failure = trace.failure_report(error)
    assert failure["partial_scalar_evidence"] == evidence
    trace.assert_publication_boundary(failure)


def test_return_only_parent_difference_is_diagnostic_and_not_a_gate() -> None:
    result = _synthetic_trace(0, 3)
    expected = {
        "completed_transitions": 3,
        "terminal_q9": 11,
        "termination_names": ["ee_body_pos"],
        "episode_return": result["episode_return"] - 0.19601054675877094,
        "policy_state_sha256": trace.EXPECTED_POLICY_SHA256[0],
    }
    diagnostic = trace.validate_parent_evaluation_reproduction(
        result=result,
        layout=_layout(),
        expected_evaluation=expected,
        update_count=0,
        provenance_snapshot_sha256="a" * 64,
    )
    assert diagnostic["delta_observed_minus_historical"] == pytest.approx(0.19601054675877094)
    assert diagnostic["exact_match"] is False
    assert diagnostic["gate_applied"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_transitions", 4),
        ("terminal_q9", 12),
        ("termination_names", ["time_out"]),
        ("policy_state_sha256", "f" * 64),
    ],
)
def test_parent_structural_mismatch_still_fails_closed(field: str, value: Any) -> None:
    result = _synthetic_trace(0, 3)
    expected = {
        "completed_transitions": 3,
        "terminal_q9": 11,
        "termination_names": ["ee_body_pos"],
        "episode_return": result["episode_return"],
        "policy_state_sha256": trace.EXPECTED_POLICY_SHA256[0],
    }
    expected[field] = value
    with pytest.raises(trace.TraceReproductionError, match="structure") as caught:
        trace.validate_parent_evaluation_reproduction(
            result=result,
            layout=_layout(),
            expected_evaluation=expected,
            update_count=0,
            provenance_snapshot_sha256="a" * 64,
        )
    assert caught.value.partial_evidence["mismatch"][field] is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["series"]["q9"].__setitem__(1, 99), "q9 series"),
        (
            lambda value: value["series"]["reward_total"].__setitem__(0, float("nan")),
            "scalar series",
        ),
        (
            lambda value: value["series"]["reward_weighted"][0].__setitem__(0, 99.0),
            "do not sum",
        ),
        (
            lambda value: value["series"]["termination_names"].__setitem__(0, ["ee_body_pos"]),
            "one final terminal",
        ),
        (
            lambda value: value["series"]["raw_native_action"][0].pop(),
            "row mismatch",
        ),
    ],
)
def test_trace_validation_fails_closed_on_corruption(mutation: Any, match: str) -> None:
    value = _synthetic_trace(0, 3)
    mutation(value)
    with pytest.raises(ValueError, match=match):
        trace.validate_trace_series(value, _layout())


def test_frame_columnarization_preserves_only_declared_fields() -> None:
    frame = {
        "q9": 9,
        "reward_total": 0.1,
        "reward_raw": [1.0, 0.0],
        "reward_weighted": [0.1, -0.0],
        "ee_z_error_m": [0.0] * 4,
        "actions": {
            "raw_native": _row(0.0),
            "safe_native": _row(0.0),
            "final_target_hardware": _row(0.0),
        },
        "contact": {
            "found": [True, True],
            "force_magnitude_n": [10.0, 10.0],
            "landing_force_mean_n": 0.0,
        },
        "termination_names": [],
        "episode_length_pre_reset": 1,
    }
    series = trace.frames_to_series([frame])
    assert series["q9"] == [9]
    assert series["raw_native_action"] == [_row(0.0)]
    assert "episode_length_pre_reset" not in series
    assert set(series) == {
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


def test_reward_seam_snapshots_are_unaliased_and_defer_all_host_reads() -> None:
    source = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    snapshot = trace._clone_tensor_snapshot(
        source,
        shape=(1, 2),
        context="test snapshot",
        floating=True,
    )
    source.add_(10.0)
    assert snapshot.tolist() == [[1.0, 2.0]]
    assert snapshot.data_ptr() != source.data_ptr()

    snapshot_helpers = (
        trace._RewardComputeTraceRecorder._snapshot,
        trace._snapshot_action_chain,
        trace._snapshot_contact_state,
        trace._snapshot_ee_z_errors,
        trace._snapshot_termination_state,
    )
    for helper in snapshot_helpers:
        helper_source = inspect.getsource(helper)
        assert ".cpu(" not in helper_source
        assert ".item(" not in helper_source
        assert ".tolist(" not in helper_source


def test_failed_trace_chain_requires_exact_v1_and_v2_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base = {
        "schema_version": 1,
        "kind": trace.FAILURE_KIND,
        "error": {
            "type": "TraceReproductionError",
            "message": "update0 trace did not reproduce parent evaluation",
        },
        "simulator_trace_complete": False,
        "training_updates": 0,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
    }
    v1_path = tmp_path / trace.PRIOR_FAILED_TRACE_FILENAME
    v2_path = tmp_path / trace.SECOND_FAILED_TRACE_FILENAME
    v1_path.write_text(json.dumps(base), encoding="utf-8")
    v2 = copy.deepcopy(base)
    v2["partial_scalar_evidence"] = {
        "stage": "update0_parent_evaluation_reproduction",
        "mismatch": {
            "completed_transitions": False,
            "terminal_q9": False,
            "termination_names": False,
            "episode_return": True,
            "policy_state_sha256": False,
        },
    }
    v2_path.write_text(json.dumps(v2), encoding="utf-8")
    monkeypatch.setattr(trace, "PRIOR_FAILED_TRACE_SHA256", sha256_file(v1_path))
    monkeypatch.setattr(trace, "SECOND_FAILED_TRACE_SHA256", sha256_file(v2_path))
    bindings = trace._validate_prior_failed_traces(tmp_path / "sonic_task_space_ppo_seed20260805_v2")
    assert bindings["v1"]["contains_partial_scalar_evidence"] is False
    assert bindings["v2"]["contains_partial_scalar_evidence"] is True
    assert bindings["v1"]["immutable"] is True
    assert bindings["v2"]["immutable"] is True

    v2_path.write_text(json.dumps({**v2, "deployment_ready": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        trace._validate_prior_failed_traces(tmp_path / "sonic_task_space_ppo_seed20260805_v2")


def test_control_plane_drift_is_exactly_pinned_and_disclosed() -> None:
    required = set(trace.KNOWN_CONTROL_PLANE_PATHS) | set(trace.EXPECTED_UNCHANGED_CONTROL_PATHS)
    parent = {}
    current = {}
    for index, logical in enumerate(sorted(required), start=1):
        current[logical] = trace.SEALED_CURRENT_CONTROL_SHA256[logical]
        parent_hash = current[logical]
        if logical in trace.KNOWN_CONTROL_PLANE_PATHS:
            parent_hash = f"{index:064x}"
        parent[logical] = {"sha256": parent_hash}
    report = trace.classify_control_plane_drift(parent, current)
    assert len(report["known_drift"]) == 3
    assert report["unchanged"] == list(trace.EXPECTED_UNCHANGED_CONTROL_PATHS)
    assert report["historical_full_source_identity_claimed"] is False

    changed = dict(current)
    changed[next(iter(required))] = "f" * 64
    with pytest.raises(ValueError, match="current sealed control-plane source drift"):
        trace.classify_control_plane_drift(parent, changed)


def test_publication_boundary_and_exclusive_compact_writer(tmp_path: Path) -> None:
    value = {
        "kind": trace.TRACE_KIND,
        "training_updates": 0,
        "teacher_labels_used": False,
        "support_qualified": False,
        "promotion_eligible": False,
        "hardware_authorized": False,
        "deployment_ready": False,
        "values": [1.0, 2.0],
    }
    output = tmp_path / "trace.json"
    trace.write_json_exclusive(output, value)
    payload = output.read_bytes()
    assert payload.endswith(b"\n")
    assert b": " not in payload
    assert json.loads(payload) == value
    with pytest.raises(FileExistsError, match="overwrite"):
        trace.write_json_exclusive(output, value)
    broken = copy.deepcopy(value)
    broken["hardware_authorized"] = True
    with pytest.raises(ValueError, match="boundary violated"):
        trace.assert_publication_boundary(broken)
    broken = copy.deepcopy(value)
    broken["training_updates"] = 1
    with pytest.raises(ValueError, match="training boundary"):
        trace.assert_publication_boundary(broken)
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    symlink_parent = tmp_path / "linked"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        trace.write_json_exclusive(symlink_parent / "trace.json", value)


def test_preflight_failure_and_failure_report_preserve_all_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError("bound source drift")

    monkeypatch.setattr(trace, "resolve_trace_inputs", fail)
    report = trace.trace_preflight(tmp_path, tmp_path)
    assert report["ready"] is False
    assert report["simulator_constructed"] is False
    assert report["training_updates"] == 0
    assert report["hardware_authorized"] is False
    failure = trace.failure_report(RuntimeError("boom"))
    trace.assert_publication_boundary(failure)
    assert failure["simulator_trace_complete"] is False
    assert failure["training_updates"] == 0


def test_cli_has_only_preflight_trace_and_no_training_or_checkpoint_overrides() -> None:
    parser = cli._parser()
    preflight = parser.parse_args(["preflight"])
    assert preflight.command == "preflight"
    args = parser.parse_args(["trace", "--output-json", "trace.json"])
    assert args.command == "trace"
    assert args.output_json == Path("trace.json")
    for forbidden in (
        "--checkpoint",
        "--seed",
        "--learning-rate",
        "--updates",
        "--device",
        "--hardware",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(["trace", "--output-json", "trace.json", forbidden, "x"])


def test_cli_pins_v3_sibling_output_and_never_reuses_v1_or_v2(tmp_path: Path) -> None:
    parent_run = tmp_path / "sonic_task_space_ppo_seed20260805_v2"
    parent_run.mkdir()
    expected = tmp_path / trace.RETRY_TRACE_FILENAME
    assert cli._validate_retry_output(parent_run, expected) == expected.absolute()
    with pytest.raises(ValueError, match="immutable sibling"):
        cli._validate_retry_output(parent_run, tmp_path / trace.PRIOR_FAILED_TRACE_FILENAME)
    with pytest.raises(ValueError, match="immutable sibling"):
        cli._validate_retry_output(parent_run, tmp_path / trace.SECOND_FAILED_TRACE_FILENAME)
    with pytest.raises(ValueError, match="immutable sibling"):
        cli._validate_retry_output(parent_run, tmp_path / "arbitrary.json")


def test_runtime_source_contains_no_training_or_hardware_execution() -> None:
    utility = (ROOT / "gear_sonic/utils/g1_true23_sonic_task_space_checkpoint_trace.py").read_text()
    launcher = (ROOT / "gear_sonic/scripts/trace_g1_true23_sonic_task_space_checkpoints.py").read_text()
    assert ".learn(" not in utility
    assert "optimizer.step(" not in utility
    assert "unitree_sdk" not in utility
    assert "unitree_sdk" not in launcher
    assert 'training_updates": 0' in utility
    assert 'hardware_authorized": False' in utility
