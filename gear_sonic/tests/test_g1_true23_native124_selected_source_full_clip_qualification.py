from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.scripts.qualify_g1_true23_native124_selected_source_full_clip import (
    _parser,
)
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTOR_STATE_SHA256,
    CHECKPOINT_SHA256,
    ONNX_SHA256,
)
import gear_sonic.utils.g1_true23_native124_selected_source_full_clip_qualification as qualification
import gear_sonic.utils.g1_true23_native124_selected_v2_ankle_evaluation as evaluation
from gear_sonic.utils.g1_true23_native124_selected_v2_full_clip_schedule import (
    CONTINUOUS_CONTRACT_SHA256,
    PHASE_SCHEDULE_SHA256,
    QUALIFICATION_SCHEDULE_SHA256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _request(
    tmp_path: Path,
    *,
    mode: str = "continuous",
    phase_window_id: str | None = None,
) -> qualification.FullClipQualificationRequest:
    evidence = tmp_path / "artifacts" / "g1_true23"
    evidence.mkdir(parents=True)
    return qualification.FullClipQualificationRequest(
        repository_root=tmp_path,
        output=Path("artifacts/g1_true23/full_clip.json"),
        mode=mode,
        phase_window_id=phase_window_id,
    )


def test_only_pinned_continuous_and_named_phase_windows_resolve() -> None:
    continuous = qualification.resolve_qualification_window("continuous", None)
    assert (
        continuous.window_id,
        continuous.anchor_q9,
        continuous.transitions,
        continuous.burn_in_transitions,
        continuous.last_q9,
    ) == ("continuous", 9, 2080, 0, 2088)

    expected = {
        "w0": (9, 500, 0, 508),
        "w1": (409, 500, 100, 908),
        "w2": (809, 500, 100, 1308),
        "w3": (1209, 500, 100, 1708),
        "w4": (1609, 480, 100, 2088),
    }
    for window_id, values in expected.items():
        window = qualification.resolve_qualification_window("phase", window_id)
        assert (
            window.anchor_q9,
            window.transitions,
            window.burn_in_transitions,
            window.last_q9,
        ) == values

    with pytest.raises(ValueError, match="forbids"):
        qualification.resolve_qualification_window("continuous", "w0")
    with pytest.raises(ValueError, match="one of"):
        qualification.resolve_qualification_window("phase", "w5")
    with pytest.raises(ValueError, match="continuous or phase"):
        qualification.resolve_qualification_window("custom", None)


def test_gate_binds_schedule_hashes_and_unchanged_support_thresholds() -> None:
    window = qualification.resolve_qualification_window("phase", "w4")
    gate = qualification.full_clip_gate_contract(REPO_ROOT, window)
    assert gate["schedule_sha256"] == QUALIFICATION_SCHEDULE_SHA256
    assert gate["continuous_contract_sha256"] == CONTINUOUS_CONTRACT_SHA256
    assert gate["phase_schedule_sha256"] == PHASE_SCHEDULE_SHA256
    assert gate["window"] == {
        "id": "w4",
        "anchor_q9": 1609,
        "transitions": 480,
        "burn_in_transitions": 100,
        "first_scored_q9": 1709,
        "last_q9": 2088,
        "q10_proof_last": 2089,
    }
    assert gate["minimum_base_height_m"] == 0.45
    assert gate["maximum_base_tilt_rad"] == 1.0
    assert gate["maximum_joint_velocity_ratio"] == 1.0
    assert gate["maximum_tracking_rmse_rad"] == 0.75
    assert gate["plain_sonic_raw_abs_strict_max"] == 10.0
    assert gate["actor_onnx_parity_max_absolute_error"] == 1.0e-5
    assert gate["inference_duration_threshold_applied"] is False
    assert gate["projection_threshold_applied"] is False


@pytest.mark.parametrize(
    ("mode", "window_id"),
    (("continuous", None), ("phase", "w1"), ("phase", "w4")),
)
def test_scheduled_parity_requires_every_actual_q9(
    mode: str,
    window_id: str | None,
) -> None:
    window = qualification.resolve_qualification_window(mode, window_id)
    accumulator = qualification.ScheduledActorOnnxParityAccumulator(
        window,
        1.0e-5,
    )
    for transition in range(window.transitions):
        rsl = np.full(23, transition * 1.0e-5, dtype=np.float32)
        onnx = rsl.copy()
        accumulator.add(
            transition=transition,
            q9=window.anchor_q9 + transition,
            rsl_action=rsl,
            onnx_action=onnx,
        )
    report = accumulator.report()
    assert report["check_count"] == window.transitions
    assert report["violation_count"] == 0
    assert report["passed"] is True

    bad = qualification.ScheduledActorOnnxParityAccumulator(window, 1.0e-5)
    rsl = np.zeros(23, dtype=np.float32)
    onnx = rsl.copy()
    onnx[3] = np.float32(1.1e-5)
    bad.add(
        transition=0,
        q9=window.anchor_q9,
        rsl_action=rsl,
        onnx_action=onnx,
    )
    assert bad.report()["violation_count"] == 1
    assert bad.report()["passed"] is False


def _passing_inputs(
    mode: str,
    window_id: str | None,
) -> dict[str, object]:
    window = qualification.resolve_qualification_window(mode, window_id)
    gate = qualification.full_clip_gate_contract(REPO_ROOT, window)
    support_summary = {name: 0 for name in gate["required_zero_counts"]}
    support_summary.update(
        {
            "minimum_base_height_m": 0.78,
            "maximum_base_tilt_rad": 0.14,
            "maximum_joint_velocity_ratio": 0.18,
            "maximum_tracking_rmse_rad": 0.19,
            "maximum_plain_sonic_raw_native_abs": 3.4,
        }
    )
    return {
        "mode": mode,
        "window": window,
        "gate": gate,
        "reset_seam": {
            "prime_q9": window.anchor_q9,
            "first_deterministic_actor_q9": window.anchor_q9,
            "fixed_warmup_or_action_substitution_steps": 0,
            "action_substitution": False,
            "reset_buffer_proof": {
                "reset_virtual_torso_mask": True,
                "actor_previous_action_slice_is_zero": True,
            },
        },
        "first_done": {
            "transition": window.transitions - 1,
            "q9_before": window.last_q9,
            "q9_after_autoreset": window.anchor_q9,
            "episode_length_pre_reset": window.transitions,
            "termination_names": ["time_out"],
            "is_timeout": True,
            "is_terminated": False,
        },
        "attempted": window.transitions,
        "partition_counts": {
            "burn_in_transition_count": window.burn_in_transitions,
            "scored_transition_count": (window.transitions - window.burn_in_transitions),
            "unexpected_done_before_final_count": 0,
        },
        "support_summary": support_summary,
        "rollout_summary": {
            "transition_count": window.transitions,
            "hard_safety_violation_count": 0,
            "soft_safety_warning_count": 0,
        },
        "parity": {
            "passed": True,
            "check_count": window.transitions,
            "violation_count": 0,
            "maximum_absolute_error": 2.0e-6,
        },
        "action_semantics": {
            "passed": True,
            "check_count": window.transitions,
            "mismatch_count": 0,
        },
        "source_identity": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "actor_state_sha256": ACTOR_STATE_SHA256,
            "onnx_sha256": ONNX_SHA256,
            "restart_or_derivative_loaded": False,
            "adaptation_delta_required": False,
        },
        "frozen_state": {
            "actor_unchanged": True,
            "critic_unchanged": True,
            "optimizer_unchanged": True,
            "optimizer_steps": 0,
            "current_learning_iteration": 0,
            "rollout_storage_step": 0,
        },
    }


@pytest.mark.parametrize(
    ("mode", "window_id"),
    (("continuous", None), ("phase", "w0"), ("phase", "w2"), ("phase", "w4")),
)
def test_continuous_and_phase_claims_are_separate(
    mode: str,
    window_id: str | None,
) -> None:
    result = qualification.assess_full_clip_qualification(**_passing_inputs(mode, window_id))
    assert result["qualified_requested_mode"] is True
    continuous = result["claims"]["continuous_full_clip_reachability"]
    restart = result["claims"]["phase_restartability"]
    if mode == "continuous":
        assert continuous == {"performed": True, "qualified": True}
        assert restart == {
            "performed": False,
            "window_id": None,
            "qualified": None,
        }
    else:
        assert continuous == {"performed": False, "qualified": None}
        assert restart == {
            "performed": True,
            "window_id": window_id,
            "qualified": True,
        }
    assert result["claims"]["cross_mode_inference_permitted"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "reset_anchor",
        "early_burn_in_done",
        "early_score_done",
        "wrong_final_reset",
        "partition",
        "support_count",
        "tracking",
        "soft_safety",
        "parity",
        "action_semantics",
        "derivative",
        "actor_changed",
    ),
)
def test_phase_fails_every_early_done_and_independent_gate(mutation: str) -> None:
    inputs = _passing_inputs("phase", "w2")
    window = inputs["window"]
    mutated = copy.deepcopy(inputs)
    if mutation == "reset_anchor":
        mutated["reset_seam"]["prime_q9"] += 1
    elif mutation == "early_burn_in_done":
        mutated["first_done"]["transition"] = 50
        mutated["first_done"]["q9_before"] = window.anchor_q9 + 50
        mutated["first_done"]["episode_length_pre_reset"] = 51
        mutated["attempted"] = 51
        mutated["rollout_summary"]["transition_count"] = 51
        mutated["partition_counts"] = {
            "burn_in_transition_count": 51,
            "scored_transition_count": 0,
            "unexpected_done_before_final_count": 1,
        }
    elif mutation == "early_score_done":
        mutated["first_done"]["transition"] = 200
        mutated["first_done"]["q9_before"] = window.anchor_q9 + 200
        mutated["first_done"]["episode_length_pre_reset"] = 201
        mutated["attempted"] = 201
        mutated["rollout_summary"]["transition_count"] = 201
        mutated["partition_counts"] = {
            "burn_in_transition_count": 100,
            "scored_transition_count": 101,
            "unexpected_done_before_final_count": 1,
        }
    elif mutation == "wrong_final_reset":
        mutated["first_done"]["q9_after_autoreset"] = 9
    elif mutation == "partition":
        mutated["partition_counts"]["scored_transition_count"] -= 1
    elif mutation == "support_count":
        mutated["support_summary"]["raw_clip_required_count"] = 1
    elif mutation == "tracking":
        mutated["support_summary"]["maximum_tracking_rmse_rad"] = 0.750001
    elif mutation == "soft_safety":
        mutated["rollout_summary"]["soft_safety_warning_count"] = 1
    elif mutation == "parity":
        mutated["parity"]["maximum_absolute_error"] = 1.0001e-5
    elif mutation == "action_semantics":
        mutated["action_semantics"]["mismatch_count"] = 1
    elif mutation == "derivative":
        mutated["source_identity"]["restart_or_derivative_loaded"] = True
    elif mutation == "actor_changed":
        mutated["frozen_state"]["actor_unchanged"] = False
    result = qualification.assess_full_clip_qualification(**mutated)
    assert result["qualified_requested_mode"] is False
    assert result["claims"]["phase_restartability"]["qualified"] is False


def test_writer_is_exclusive_and_failure_claims_stay_nondeploying(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, mode="phase", phase_window_id="w3")
    report = qualification.failure_report(RuntimeError("boom"), request)
    output = qualification.write_full_clip_qualification_new(request, report)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["kind"] == qualification.QUALIFICATION_KIND
    assert written["qualified_requested_mode"] is False
    assert written["claims"]["phase_restartability"] == {
        "performed": True,
        "window_id": "w3",
        "qualified": False,
    }
    assert written["scope"]["deployment_authorized"] is False
    with pytest.raises(FileExistsError):
        qualification.write_full_clip_qualification_new(request, report)


@pytest.mark.parametrize(
    ("args", "mode", "window_id"),
    (
        (["--continuous"], "continuous", None),
        (["--phase-window", "w0"], "phase", "w0"),
        (["--phase-window", "w4"], "phase", "w4"),
    ),
)
def test_cli_exposes_only_pinned_modes(
    args: list[str],
    mode: str,
    window_id: str | None,
) -> None:
    parser = _parser()
    parsed = parser.parse_args(
        [
            "--output",
            "artifacts/g1_true23/full.json",
            *args,
        ]
    )
    actual_mode = "continuous" if parsed.continuous else "phase"
    assert actual_mode == mode
    assert parsed.phase_window == window_id
    destinations = {
        action.dest
        for action in parser._actions  # noqa: SLF001
        if action.dest != "help"
    }
    assert destinations == {
        "repository_root",
        "output",
        "continuous",
        "phase_window",
    }


def test_cli_rejects_missing_conflicting_or_arbitrary_schedule_controls() -> None:
    parser = _parser()
    output = ["--output", "artifacts/g1_true23/full.json"]
    with pytest.raises(SystemExit):
        parser.parse_args(output)
    with pytest.raises(SystemExit):
        parser.parse_args([*output, "--continuous", "--phase-window", "w0"])
    with pytest.raises(SystemExit):
        parser.parse_args([*output, "--phase-window", "w5"])
    for forbidden in ("--anchor", "--transitions", "--burn-in", "--checkpoint"):
        with pytest.raises(SystemExit):
            parser.parse_args([*output, "--continuous", forbidden, "1"])


def test_preflight_binds_exact_schedule_source_and_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qualification,
        "resolve_rsl_runtime_binding",
        lambda: {"test_only": "pinned-runtime-binding-witness"},
    )
    request = qualification.FullClipQualificationRequest(
        repository_root=REPO_ROOT,
        output=Path(f"artifacts/g1_true23/{tmp_path.name}_full_clip_never_written.json"),
        mode="phase",
        phase_window_id="w4",
    )
    preflight = qualification.preflight_full_clip_qualification(request)
    assert preflight["ready"] is True
    assert preflight["selected_source"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert preflight["selected_source"]["actor_state_sha256"] == ACTOR_STATE_SHA256
    assert preflight["selected_source"]["onnx_sha256"] == ONNX_SHA256
    assert preflight["window"]["anchor_q9"] == 1609
    assert preflight["window"]["transitions"] == 480
    assert preflight["fixed"]["fixed_warmup_or_action_substitution_steps"] == 0
    assert preflight["scope"]["phase_restartability_claim_permitted"] is True
    assert preflight["scope"]["continuous_full_clip_reachability_claim_permitted"] is False


def test_runtime_source_uses_direct_selected_actor_and_no_custom_schedule() -> None:
    source = Path(qualification.__file__).read_text(encoding="utf-8")
    assert "configure_selected_v2_ankle_rsl_runner" in source
    assert "restart_path=None" in source
    assert "Native124Checkpoint21204Policy" in source
    assert "fixed_anchor_index=window.anchor_q9" in source
    assert "env_cfg.episode_length_s = window.transitions * CONTROL_DT_S" in source
    assert "configure_hash_locked_warm_evaluation_runner" not in source
    assert "prove_warmup_action_equivalence" not in source
    assert "initial_model.pt" not in source
    assert "wrapped.step(rsl_action_tensor)" in source
    assert '"partial_action_contract_failure"' in source
    assert "selected_source_action_contract_failure_quarantined" in source


def _fake_action_contract_env(selected: np.ndarray) -> SimpleNamespace:
    action = SimpleNamespace(
        raw_action=torch.from_numpy(selected.copy()).reshape(1, 23),
        candidate_target_hardware=torch.from_numpy(selected + np.float32(1.0)).reshape(1, 23),
        plain_sonic_raw_action_native=torch.from_numpy(selected + np.float32(2.0)).reshape(1, 23),
        safe_native_action=torch.from_numpy(selected + np.float32(3.0)).reshape(1, 23),
        processed_action=torch.from_numpy(selected + np.float32(4.0)).reshape(1, 23),
        raw_clip_mask_native=torch.zeros((1, 23), dtype=torch.bool),
    )
    return SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda name: action),
    )


def _step_evidence() -> evaluation.StepEvidence:
    return evaluation.StepEvidence(
        reward=0.25,
        scalars={name: 0.0 for name in evaluation._SAFETY_SCALAR_KEYS},  # noqa: SLF001
        counts={name: 0 for name in evaluation._SAFETY_COUNT_KEYS},  # noqa: SLF001
        reward_rates={"test_reward": 0.5},
    )


@pytest.mark.parametrize(
    ("message", "reason"),
    (
        (
            "teacher composite safe inverse does not recover plain SONIC raw label",
            "safe_inverse_round_trip_failure",
        ),
        (
            "teacher composite safe native action does not recover V2 forward transform",
            "safe_inverse_round_trip_failure",
        ),
        (
            "teacher composite final hardware target does not recover V2 forward transform",
            "safe_inverse_round_trip_failure",
        ),
        (
            "teacher composite requires plain SONIC raw clipping",
            "plain_sonic_raw_clip_required",
        ),
    ),
)
def test_expected_action_contract_failure_keeps_exact_action_chain(
    message: str,
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
    env = _fake_action_contract_env(selected)

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError(message)

    monkeypatch.setattr(qualification.nominal, "_runtime_action_semantics", fail)
    observation = qualification.capture_action_semantics_or_expected_failure(
        env,
        selected,
        {},
    )
    assert observation["semantics"]["match"] is False
    assert observation["semantics"]["contract_failure"] is True
    assert observation["failure"]["reason"] == reason
    chain = observation["failure"]["action_chain"]
    assert chain["selected_actor_raw_action_hardware"] == pytest.approx(selected)
    assert chain["actual_selected_raw_action_hardware"] == pytest.approx(selected)
    assert chain["candidate_target_hardware"] == pytest.approx(selected + 1.0)
    assert chain["plain_sonic_raw_action_native"] == pytest.approx(selected + 2.0)
    assert chain["safe_native_action"] == pytest.approx(selected + 3.0)
    assert chain["final_target_hardware"] == pytest.approx(selected + 4.0)
    assert chain["raw_clip_coordinate_count"] == 0

    window = qualification.resolve_qualification_window("continuous", None)
    partial = qualification.build_partial_action_contract_failure(
        transition=123,
        q9_before=132,
        q9_after=133,
        window=window,
        environment_done=False,
        parity_max_absolute_error=2.0e-6,
        action_observation=observation,
        step_evidence=_step_evidence(),
    )
    assert partial["transition"] == 123
    assert partial["q9_before"] == 132
    assert partial["q10_proof_before"] == 133
    assert partial["partition"] == "scored"
    assert partial["partial_terminal"] is True
    assert partial["qualification_stop_reason"] == "action_contract_failure"
    assert partial["simulator_termination"] is False
    assert partial["environment_done"] is False
    assert partial["action_chain"] == chain
    assert partial["step_evidence"]["reward"] == 0.25
    assert partial["gate_weakened"] is False
    assert partial["qualification_must_fail"] is True

    accumulator = qualification._ScheduledActionSemanticsAccumulator(2080)  # noqa: SLF001
    accumulator.add(observation["semantics"])
    report = accumulator.report()
    assert report["check_count"] == 1
    assert report["mismatch_count"] == 1
    assert report["contract_failure_count"] == 1
    assert report["maximum_absolute_error_by_link"]["raw"] == 0.0
    assert report["maximum_absolute_error_by_link"]["candidate"] is None
    assert report["unavailable_error_count_by_link"]["candidate"] == 1
    assert report["passed"] is False


def test_unexpected_action_semantics_programming_error_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = np.zeros(23, dtype=np.float32)
    env = _fake_action_contract_env(selected)

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("unrecognized action implementation drift")

    monkeypatch.setattr(qualification.nominal, "_runtime_action_semantics", fail)
    with pytest.raises(ValueError, match="implementation drift"):
        qualification.capture_action_semantics_or_expected_failure(
            env,
            selected,
            {},
        )


def test_continuous_and_w4_task_horizons_construct_without_simulator() -> None:
    from gear_sonic.envs.mjlab import (
        native124_selected_v2_ankle_task as task,
        native124_selected_v2_causal_adaptation as causal,
    )

    if causal._MJLAB_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab config runtime required")
    for mode, window_id, expected_base_steps in (
        ("continuous", None, 500),
        ("phase", "w4", 480),
    ):
        window = qualification.resolve_qualification_window(mode, window_id)
        cfg = task.make_native124_selected_v2_ankle_task_env_cfg(
            motion_file=str(REPO_ROOT / causal.DAD_DANCE_RELATIVE_PATH),
            num_envs=1,
            play=False,
            fixed_anchor_index=window.anchor_q9,
        )
        base_audit = task.audit_native124_selected_v2_ankle_task_env_cfg(cfg)
        assert base_audit["reset_anchor_q9"] == window.anchor_q9
        assert base_audit["episode_steps"] == expected_base_steps
        assert cfg.episode_length_s == pytest.approx(expected_base_steps * task.CONTROL_DT_S)
        audit = qualification.configure_scheduled_task_horizon(cfg, window)
        assert cfg.commands["motion"].fixed_anchor_index == window.anchor_q9
        assert cfg.episode_length_s == pytest.approx(window.transitions * task.CONTROL_DT_S)
        assert audit["scheduled_episode_steps"] == window.transitions
        assert audit["scheduled_last_q9"] == window.last_q9
        assert audit["scheduled_last_q10_proof"] == window.last_q9 + 1
