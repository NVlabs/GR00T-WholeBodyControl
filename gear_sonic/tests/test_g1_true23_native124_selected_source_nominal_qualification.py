from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.scripts.qualify_g1_true23_native124_selected_source_nominal import (
    _parser,
)
from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
    ACTOR_STATE_SHA256,
    CHECKPOINT_SHA256,
    ONNX_SHA256,
)
import gear_sonic.utils.g1_true23_native124_selected_source_nominal_qualification as qualification
from gear_sonic.utils.g1_true23_teacher_support import (
    compose_checkpoint21204_teacher_action,
    load_teacher_support_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _request(tmp_path: Path) -> qualification.SelectedSourceNominalQualificationRequest:
    evidence = tmp_path / "artifacts" / "g1_true23"
    evidence.mkdir(parents=True)
    return qualification.SelectedSourceNominalQualificationRequest(
        repository_root=tmp_path,
        output=Path("artifacts/g1_true23/selected_source_nominal.json"),
    )


def test_request_writer_and_scope_are_exclusive_and_non_deploying(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    scope = qualification.qualification_scope()
    assert scope["classification"] == ("selected_source_daddance_nominal_slice_candidate_only")
    assert scope["original_selected_actor_required"] is True
    assert scope["adaptation_delta_required"] is False
    for key in (
        "teacher_labels_admitted",
        "promotion_authorized",
        "deployment_authorized",
        "hardware_authorized",
    ):
        assert scope[key] is False

    report = qualification.failure_report(RuntimeError("boom"), request)
    output = qualification.write_selected_source_nominal_qualification_new(
        request,
        report,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["kind"] == qualification.QUALIFICATION_KIND
    assert written["qualified_nominal_slice"] is False
    assert written["scope"] == scope
    with pytest.raises(FileExistsError):
        qualification.write_selected_source_nominal_qualification_new(request, report)

    outside = qualification.SelectedSourceNominalQualificationRequest(
        repository_root=tmp_path,
        output=Path("outside.json"),
    )
    with pytest.raises(ValueError, match="must stay under"):
        _ = outside.output_path


def test_gate_is_exact_frozen_support_nominal_slice() -> None:
    gate = qualification.selected_source_nominal_gate_contract(REPO_ROOT)
    assert gate["minimum_base_height_m"] == 0.45
    assert gate["maximum_base_tilt_rad"] == 1.0
    assert gate["maximum_joint_velocity_ratio"] == 1.0
    assert gate["maximum_tracking_rmse_rad"] == 0.75
    assert gate["plain_sonic_raw_abs_strict_max"] == 10.0
    assert gate["actor_onnx_parity_max_absolute_error"] == 1.0e-5
    assert gate["required_transitions"] == 500
    assert gate["initial_actor_q9"] == 9
    assert gate["expected_done_transition"] == 499
    assert gate["expected_done_q9"] == 508
    assert gate["inference_duration_threshold_applied"] is False
    assert gate["projection_threshold_applied"] is False
    assert gate["task_reward_rate_thresholds_applied"] is False
    assert gate["threshold_sources"]["support_config"]["sha256"] == (
        "28d29c4204a3a0c4952723a9bfb40327ea2fa4aa50ca39bb510ea6121c6119e9"
    )


def test_actor_onnx_parity_accumulator_is_strict_per_step() -> None:
    accumulator = qualification.ActorOnnxParityAccumulator(1.0e-5)
    for transition in range(500):
        rsl = np.full(23, transition * 1.0e-4, dtype=np.float32)
        onnx = rsl.copy()
        if transition == 123:
            onnx[7] += np.float32(9.0e-6)
        accumulator.add(
            transition=transition,
            q9=9 + transition,
            rsl_action=rsl,
            onnx_action=onnx,
        )
    report = accumulator.report()
    assert report["check_count"] == 500
    assert report["violation_count"] == 0
    assert report["worst_transition"] == 123
    assert report["worst_q9"] == 132
    assert report["passed"] is True

    failed = qualification.ActorOnnxParityAccumulator(1.0e-5)
    rsl = np.zeros(23, dtype=np.float32)
    onnx = rsl.copy()
    onnx[0] = np.float32(1.1e-5)
    failed.add(transition=0, q9=9, rsl_action=rsl, onnx_action=onnx)
    assert failed.report()["violation_count"] == 1
    assert failed.report()["passed"] is False
    with pytest.raises(ValueError, match="sequence drift"):
        failed.add(transition=2, q9=11, rsl_action=rsl, onnx_action=onnx)


def test_runtime_action_semantics_matches_frozen_composite() -> None:
    selected = np.linspace(-0.5, 0.5, 23, dtype=np.float32)
    expected = compose_checkpoint21204_teacher_action(
        selected,
        repository_root=REPO_ROOT,
    )
    action = SimpleNamespace(
        raw_action=torch.from_numpy(selected.copy()).reshape(1, 23),
        candidate_target_hardware=torch.from_numpy(expected.teacher_candidate_target_hardware.copy()).reshape(
            1, 23
        ),
        plain_sonic_raw_action_native=torch.from_numpy(expected.teacher_action_native.copy()).reshape(1, 23),
        safe_native_action=torch.from_numpy(expected.teacher_applied_safe_action_native.copy()).reshape(1, 23),
        processed_action=torch.from_numpy(expected.teacher_target_hardware.copy()).reshape(1, 23),
    )
    env = SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda name: action),
    )
    support = load_teacher_support_contract(REPO_ROOT)

    matched = qualification._runtime_action_semantics(  # noqa: SLF001
        env,
        selected,
        support,
    )
    assert matched["match"] is True
    action.processed_action[0, 3] += 2.0e-5
    mismatched = qualification._runtime_action_semantics(  # noqa: SLF001
        env,
        selected,
        support,
    )
    assert mismatched["match"] is False


def _passing_decision_inputs() -> dict[str, object]:
    gate = qualification.selected_source_nominal_gate_contract(REPO_ROOT)
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
        "gate": gate,
        "reset_seam": {
            "prime_q9": 9,
            "first_deterministic_actor_q9": 9,
            "fixed_warmup_steps": 0,
            "action_substitution": False,
            "reset_buffer_proof": {
                "reset_virtual_torso_mask": True,
                "actor_previous_action_slice_is_zero": True,
            },
        },
        "first_done": {
            "transition": 499,
            "q9_before": 508,
            "q9_after_autoreset": 9,
            "episode_length_pre_reset": 500,
            "termination_names": ["time_out"],
            "is_timeout": True,
            "is_terminated": False,
        },
        "attempted": 500,
        "support_summary": support_summary,
        "rollout_summary": {
            "transition_count": 500,
            "hard_safety_violation_count": 0,
            "soft_safety_warning_count": 0,
        },
        "parity": {
            "passed": True,
            "check_count": 500,
            "violation_count": 0,
            "maximum_absolute_error": 2.0e-6,
        },
        "action_semantics": {
            "passed": True,
            "check_count": 500,
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
    "mutation",
    (
        "warmup",
        "timeout",
        "support_count",
        "tracking",
        "soft_safety",
        "parity",
        "action_semantics",
        "derivative",
        "actor_changed",
    ),
)
def test_qualification_decision_fails_each_independent_gate(mutation: str) -> None:
    inputs = _passing_decision_inputs()
    assert qualification.assess_selected_source_nominal_slice(**inputs)["qualified_nominal_slice"] is True

    mutated = copy.deepcopy(inputs)
    if mutation == "warmup":
        mutated["reset_seam"]["fixed_warmup_steps"] = 2
    elif mutation == "timeout":
        mutated["first_done"]["q9_before"] = 507
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
    result = qualification.assess_selected_source_nominal_slice(**mutated)
    assert result["qualified_nominal_slice"] is False


def test_preflight_binds_original_selected_pt_and_onnx_without_derivative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qualification,
        "resolve_rsl_runtime_binding",
        lambda: {"test_only": "pinned-runtime-binding-witness"},
    )
    request = qualification.SelectedSourceNominalQualificationRequest(
        repository_root=REPO_ROOT,
        output=Path(f"artifacts/g1_true23/{tmp_path.name}_selected_source_nominal_never_written.json"),
    )
    preflight = qualification.preflight_selected_source_nominal_qualification(request)
    assert preflight["ready"] is True
    assert preflight["selected_source"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert preflight["selected_source"]["actor_state_sha256"] == ACTOR_STATE_SHA256
    assert preflight["selected_source"]["onnx_sha256"] == ONNX_SHA256
    assert preflight["fixed"]["fixed_warmup_steps"] == 0
    assert preflight["fixed"]["action_substitution"] is False
    assert preflight["scope"]["adaptation_delta_required"] is False


def test_cli_has_no_checkpoint_threshold_warmup_or_deploy_controls() -> None:
    parser = _parser()
    destinations = {
        action.dest
        for action in parser._actions  # noqa: SLF001
        if action.dest != "help"
    }
    assert destinations == {"repository_root", "output"}
    args = parser.parse_args(["--output", "artifacts/g1_true23/selected_source_nominal.json"])
    assert args.output == Path("artifacts/g1_true23/selected_source_nominal.json")
    for forbidden in (
        "--checkpoint",
        "--expected-sha256",
        "--warmup-steps",
        "--threshold",
        "--deploy",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--output",
                    "artifacts/g1_true23/selected_source_nominal.json",
                    forbidden,
                    "x",
                ]
            )


def test_runtime_source_has_direct_selected_load_and_zero_substitutions() -> None:
    source = Path(qualification.__file__).read_text(encoding="utf-8")
    assert "configure_selected_v2_ankle_rsl_runner" in source
    assert "restart_path=None" in source
    assert "configure_hash_locked_warm_evaluation_runner" not in source
    assert "prove_warmup_action_equivalence" not in source
    assert "initial_model.pt" not in source
    assert "fixed_anchor_index=CAUSAL_HISTORY_ANCHOR_INDEX" in source
    assert "wrapped.step(rsl_action_tensor)" in source


def test_qualifier_task_config_binds_fixed_q9_without_simulator() -> None:
    from gear_sonic.envs.mjlab import (
        native124_selected_v2_ankle_task as task,
        native124_selected_v2_causal_adaptation as causal,
    )

    if causal._MJLAB_IMPORT_ERROR is not None:  # noqa: SLF001
        pytest.skip("MJLab config runtime required")
    cfg = task.make_native124_selected_v2_ankle_task_env_cfg(
        motion_file=str(REPO_ROOT / causal.DAD_DANCE_RELATIVE_PATH),
        num_envs=1,
        play=False,
        fixed_anchor_index=9,
    )
    command = cfg.commands["motion"]
    assert command.fixed_anchor_index == 9


def test_pinned_rsl_constructs_exact_original_selected_actor_and_cpu_onnx(
    tmp_path: Path,
) -> None:
    try:
        from mjlab.rl.runner import MjlabOnPolicyRunner
        from tensordict import TensorDict

        from gear_sonic.scripts.train_g1_true23_native124_selected_v2_ankle import (
            SMOKE_ITERATIONS,
            SMOKE_NUM_ENVS,
            Stage1LaunchPlan,
            stage1_agent_config,
        )
        from gear_sonic.trl.mjlab.native124_selected_v2_ankle_rsl import (
            configure_selected_v2_ankle_rsl_runner,
        )
        from gear_sonic.trl.mjlab.native124_selected_v2_ankle_runner import (
            tensor_state_sha256,
        )
        from gear_sonic.utils.g1_23dof_native124_21204_adapter import (
            Native124Checkpoint21204Policy,
            load_checkpoint21204_binding,
        )
    except ImportError:
        pytest.skip("pinned WSL RSL-RL runtime required")

    observations = TensorDict(
        {
            "actor": torch.zeros(1, 124, dtype=torch.float32),
            "critic": torch.zeros(1, 256, dtype=torch.float32),
        },
        batch_size=[1],
    )

    class FakeEnv:
        num_actions = 23
        num_envs = 1
        cfg: dict[str, object] = {}

        def get_observations(self) -> TensorDict:
            return observations

    plan = Stage1LaunchPlan(
        mode="smoke",
        motion_file=REPO_ROOT / "data" / "B_DadDance.npz",
        run_dir=tmp_path,
        num_envs=SMOKE_NUM_ENVS,
        iterations=SMOKE_ITERATIONS,
    )
    runner = MjlabOnPolicyRunner(
        FakeEnv(),
        copy.deepcopy(stage1_agent_config(plan)),
        None,
        "cpu",
    )
    integration = configure_selected_v2_ankle_rsl_runner(
        runner,
        repository_root=REPO_ROOT,
        restart_path=None,
        expected_restart_sha256=None,
    )
    assert tensor_state_sha256(integration.actor.state_dict()) == ACTOR_STATE_SHA256
    assert integration.optimizer_steps == 0
    assert runner.current_learning_iteration == 0
    assert getattr(runner.alg.storage, "step", 0) == 0

    integration.actor.eval()
    with torch.inference_mode():
        rsl = integration.actor(observations, stochastic_output=False).detach().cpu().numpy()[0]
    binding = load_checkpoint21204_binding(REPO_ROOT)
    onnx = Native124Checkpoint21204Policy(binding).run(np.zeros((1, 124), dtype=np.float32))
    assert float(np.max(np.abs(rsl.astype(np.float64) - onnx))) <= 1.0e-5
