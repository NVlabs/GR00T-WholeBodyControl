from __future__ import annotations

import copy
import json
from pathlib import Path
import random
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from gear_sonic.scripts import train_g1_true23_sonic_task_space_ppo as launcher
import gear_sonic.trl.mjlab.sonic_task_space_ppo_runner as task_space

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "gear_sonic" / "config" / "sim_validation" / "g1_true23_sonic_task_space_ppo_v1.json"


def _evaluation(
    update: int,
    *,
    completed: int,
    episode_return: float = -100.0,
    termination_names: list[str] | None = None,
) -> dict[str, Any]:
    if termination_names is None:
        termination_names = ["time_out"] if completed == 510 else ["ee_body_pos"]
    return {
        "update_count": update,
        "policy_state_sha256": f"policy-{update}",
        "completed_transitions": completed,
        "terminal_q9": 9 + completed - 1,
        "termination_names": termination_names,
        "episode_return": episode_return,
        "nonfinite_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": 0,
        "q9_discontinuity_count": 0,
        "teacher_labels_used": False,
    }


def test_contract_is_hash_bound_and_pins_exact_pilot() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert task_space.sha256_file(CONTRACT_PATH) == task_space.CONTRACT_SHA256
    task_space.validate_task_space_ppo_contract(contract)
    assert task_space.NUM_ENVS == 128
    assert task_space.NUM_STEPS_PER_ENV == 16
    assert task_space.MAXIMUM_UPDATES == 25
    assert task_space.TRANSITIONS_PER_UPDATE == task_space.NUM_ENVS * task_space.NUM_STEPS_PER_ENV
    assert task_space.TRANSITIONS_PER_UPDATE == 2_048
    assert task_space.OPTIMIZER_STEPS_PER_UPDATE == task_space.PPO_LEARNING_EPOCHS * task_space.PPO_MINI_BATCHES
    assert task_space.OPTIMIZER_STEPS_PER_UPDATE == 8
    assert task_space.MAXIMUM_TRAINING_TRANSITIONS == 51_200
    expected_training_transitions = task_space.TRANSITIONS_PER_UPDATE * task_space.MAXIMUM_UPDATES
    assert task_space.MAXIMUM_TRAINING_TRANSITIONS == expected_training_transitions
    expected_optimizer_steps = task_space.OPTIMIZER_STEPS_PER_UPDATE * task_space.MAXIMUM_UPDATES
    assert task_space.MAXIMUM_OPTIMIZER_STEPS == expected_optimizer_steps
    assert task_space.MAXIMUM_OPTIMIZER_STEPS == 200
    assert contract["ppo_pilot"] == {
        "num_envs": 128,
        "num_steps_per_env": 16,
        "maximum_updates": 25,
        "maximum_training_transitions": 51_200,
        "learning_rate": 1.0e-5,
        "schedule": "fixed",
        "clip_param": 0.1,
        "num_learning_epochs": 2,
        "num_mini_batches": 4,
        "entropy_coef": 0.0,
        "gamma": 0.99,
        "lam": 0.95,
        "max_grad_norm": 0.25,
        "critic": "fresh_random_512_256_128",
        "optimizer": "fresh_torch_adam_trainable_actor_plus_fresh_critic",
        "training_start_update_count": 0,
        "checkpoint_updates": [0, 5, 10, 25],
        "evaluation_updates": [0, 5, 10, 25],
        "random_episode_length_initialization": False,
    }
    assert contract["pilot_gates"]["required_zero_counts"] == [
        "nonfinite_count",
        "raw_clip_required_count",
        "action_semantics_mismatch_count",
        "q9_discontinuity_count",
    ]
    assert contract["pilot_gates"]["reward_divergence_relative_tolerance"] == 0.2
    assert contract["pilot_gates"]["reward_divergence_absolute_tolerance"] == 50.0
    assert contract["pilot_gates"]["single_pilot_pass_is_not_support_or_deployment_evidence"] is True
    assert contract["pilot_gates"]["update5_episode_length_regression_is_diagnostic_only"] is True
    assert "no_episode_length_regression_from_baseline" not in contract["pilot_gates"]


def test_assessment_allows_update5_diagnostic_regression_and_still_requires_update10_q163() -> None:
    baseline = _evaluation(0, completed=155)
    update5 = _evaluation(5, completed=125)
    update10 = _evaluation(10, completed=156)
    assessment = task_space.assess_pilot_evaluations([baseline, update5, update10])
    assert assessment["stop"] is False
    assert assessment["update5_episode_length_regression_is_diagnostic_only"] is True
    assert assessment["q163_update10_gate_reached"] is True
    assert assessment["q163_update10_gate_passed"] is True
    assert assessment["update25_permitted"] is True
    assert assessment["passed_so_far"] is True
    assert assessment["pilot_complete"] is False

    assessment = task_space.assess_pilot_evaluations([baseline, _evaluation(5, completed=155)])
    assert assessment["q163_update10_gate_reached"] is False
    assert assessment["q163_update10_gate_passed"] is False

    regressed10 = _evaluation(10, completed=155)
    assessment = task_space.assess_pilot_evaluations([baseline, update5, regressed10])
    assert assessment["stop"] is True
    assert assessment["stop_reason"] == "update10_no_q163_improvement"
    assert assessment["q163_update10_gate_reached"] is True
    assert assessment["q163_update10_gate_passed"] is False
    assert assessment["update25_permitted"] is False

    with pytest.raises(ValueError, match="sequential prefix"):
        task_space.assess_pilot_evaluations([baseline, _evaluation(10, completed=156)])
    with pytest.raises(ValueError, match="sequential prefix"):
        task_space.assess_pilot_evaluations([baseline, update5, _evaluation(25, completed=156)])
    with pytest.raises(ValueError, match="update coverage invalid"):
        task_space.assess_pilot_evaluations([baseline, update5, update5, _evaluation(25, completed=156)])
    with pytest.raises(ValueError, match="sequential prefix"):
        task_space.assess_pilot_evaluations([baseline, _evaluation(10, completed=156), update5])


def test_baseline_identity_failure_marks_gate_state_as_unreached_unpassed() -> None:
    assessment = task_space.assess_pilot_evaluations([_evaluation(0, completed=154)])
    assert assessment["stop"] is True
    assert assessment["stop_reason"] == "update0_baseline_identity_failed"
    assert assessment["q163_update10_gate_reached"] is False
    assert assessment["q163_update10_gate_passed"] is False


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"q9_discontinuity_count": 1}, "episode_length_or_safety"),
        ({"nonfinite_count": 1}, "episode_length_or_safety"),
        ({"raw_clip_required_count": 1}, "episode_length_or_safety"),
        ({"action_semantics_mismatch_count": 1}, "episode_length_or_safety"),
        ({"terminal_q9": None}, "episode_length_or_safety"),
        ({"termination_names": []}, "episode_length_or_safety"),
        ({"termination_names": ["time_out"]}, "episode_length_or_safety"),
        ({"termination_names": ["v2_raw_clip"]}, "episode_length_or_safety"),
        ({"termination_names": ["unknown_term"]}, "episode_length_or_safety"),
    ],
)
def test_assessment_rejects_malformed_terminal_or_individual_counts(
    mutation: dict[str, Any],
    reason: str,
) -> None:
    baseline = _evaluation(0, completed=155)
    update5 = _evaluation(5, completed=155)
    update5.update(mutation)
    assessment = task_space.assess_pilot_evaluations([baseline, update5])
    assert assessment["stop"] is True
    assert reason in assessment["stop_reason"]


def test_assessment_rejects_timeout_and_reward_divergence_on_update5() -> None:
    baseline = _evaluation(0, completed=155)
    timeout = _evaluation(5, completed=510, termination_names=["ee_body_pos"])
    assessment = task_space.assess_pilot_evaluations([baseline, timeout])
    assert assessment["stop"] is True
    assert assessment["stop_reason"] == "update5_episode_length_or_safety_divergence"

    low_return = _evaluation(5, completed=125, episode_return=-10000.0)
    assessment = task_space.assess_pilot_evaluations([baseline, low_return])
    assert assessment["stop"] is True
    assert assessment["stop_reason"] == "update5_reward_divergence"


def test_assessment_requires_update25_to_retain_improvement() -> None:
    records = [
        _evaluation(0, completed=155),
        _evaluation(5, completed=125),
        _evaluation(10, completed=156),
        _evaluation(25, completed=155),
    ]
    assessment = task_space.assess_pilot_evaluations(records)
    assert assessment["stop_reason"] == "update25_q163_regression"
    assert assessment["pilot_complete"] is False


def test_assessment_reaches_update25_only_when_q163_holds() -> None:
    baseline = _evaluation(0, completed=155)
    records = [
        baseline,
        _evaluation(5, completed=125),
        _evaluation(10, completed=156),
        _evaluation(25, completed=156),
    ]
    assessment = task_space.assess_pilot_evaluations(records)
    assert assessment["stop"] is False
    assert assessment["pilot_complete"] is True
    assert assessment["q163_update10_gate_reached"] is True
    assert assessment["q163_update10_gate_passed"] is True


def test_candidate_selection_requires_passed_update10_or_update25_and_hard_closes_invalid_selection() -> None:
    baseline = _evaluation(0, completed=155)
    update5 = _evaluation(5, completed=125)
    update10 = _evaluation(10, completed=156, episode_return=1.0)
    good25 = _evaluation(25, completed=156, episode_return=2.0)
    bad25 = _evaluation(25, completed=156)
    bad25["action_semantics_mismatch_count"] = 1
    paths = ["model0", "model5", "model10", "model25"]
    selected = task_space._select_pilot_candidate([baseline, update5, update10, good25], paths)
    assert selected["selected_candidate_update"] == 25
    assert selected["selected_candidate_checkpoint_path"] == "model25"

    selected = task_space._select_pilot_candidate([baseline, update5, update10, bad25], paths)
    assert selected["selected_candidate_update"] == 10
    assert selected["selected_candidate_checkpoint_path"] == "model10"

    bad10 = copy.deepcopy(update10)
    bad10["nonfinite_count"] = 1
    selected = task_space._select_pilot_candidate(
        [baseline, update5, bad10],
        paths[:3],
    )
    assert selected["selected_candidate_update"] is None
    assert selected["selected_candidate_checkpoint_path"] is None

    bad0 = copy.deepcopy(baseline)
    bad0["terminal_q9"] = None
    selected = task_space._select_pilot_candidate([bad0], paths[:1])
    assert selected["selected_candidate_update"] is None

    with pytest.raises(ValueError, match="update coverage invalid"):
        task_space._select_pilot_candidate(
            [baseline, update5, _evaluation(25, completed=156)],
            ["model0", "model5", "model25"],
        )
    with pytest.raises(ValueError, match="update coverage invalid"):
        task_space._select_pilot_candidate(
            [baseline, update10, update5, good25],
            paths,
        )
    with pytest.raises(ValueError, match="update coverage invalid"):
        task_space._select_pilot_candidate(
            [baseline, update5, update5, good25],
            ["model0", "model5", "model55", "model25"],
        )
    with pytest.raises(ValueError, match="checkpoint path must be unique and non-empty"):
        task_space._select_pilot_candidate(
            [baseline, update5, update10],
            ["m0", "m5", "m5"],
        )
    with pytest.raises(ValueError, match="requires at least one evaluation"):
        task_space._select_pilot_candidate([], [])


def test_action_chain_checks_raw_candidate_and_v2_links() -> None:
    raw = torch.zeros((1, 23), dtype=torch.float32)
    safe, target = task_space.safe_target_transform_torch(raw)
    candidate = task_space.native_actions_to_hardware_targets(
        raw,
        task_space.SAFE_TARGET_DEFAULT_Q_HARDWARE,
    )
    chain = {
        "raw_native": raw.clone(),
        "candidate_target_hardware": candidate.clone(),
        "safe_native": safe.clone(),
        "final_target_hardware": target.clone(),
        "raw_clip_mask_native": torch.zeros_like(raw, dtype=torch.bool),
    }
    assert task_space._action_chain_mismatch_count(raw, chain) == (0, 0)
    chain["candidate_target_hardware"][0, 0] += 1.0e-3
    assert task_space._action_chain_mismatch_count(raw, chain)[0] == 1


def test_death_proof_reward_terms_are_exact() -> None:
    body_names = task_space.EE_TERMINATION_BODY_NAMES
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=body_names),
        body_pos_relative_w=torch.zeros((2, 4, 3), dtype=torch.float32),
        robot_body_pos_w=torch.zeros((2, 4, 3), dtype=torch.float32),
    )
    command.robot_body_pos_w[0, 2, 2] = 0.25
    command.robot_body_pos_w[1, 3, 2] = 0.20
    env = SimpleNamespace(
        num_envs=2,
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )
    worst = task_space.worst_ee_z_normalized_squared(env)
    barrier = task_space.right_wrist_prethreshold_barrier(env)
    assert torch.allclose(worst, torch.tensor([1.0, 0.64]), atol=1.0e-6)
    assert torch.allclose(barrier, torch.tensor([0.0, 0.5]), atol=1.0e-6)


def _checkpoint_body(update: int) -> dict[str, Any]:
    optimizer_steps = update * task_space.OPTIMIZER_STEPS_PER_UPDATE
    state: dict[int, dict[str, torch.Tensor]] = {}
    if update:
        for index, shape in enumerate(task_space.OPTIMIZER_PARAMETER_SHAPES):
            state[index] = {
                "step": torch.tensor(float(optimizer_steps)),
                "exp_avg": torch.zeros(shape),
                "exp_avg_sq": torch.zeros(shape),
            }
    return {
        "g1_true23_sonic_task_space_ppo_checkpoint": task_space._checkpoint_header(),
        "policy_state_dict": {"policy": torch.zeros(1)},
        "critic_state_dict": {"critic": torch.zeros(1)},
        "optimizer_state_dict": {
            "state": state,
            "param_groups": [
                {
                    "lr": 1.0e-5,
                    "betas": (0.9, 0.999),
                    "eps": 1.0e-8,
                    "weight_decay": 0,
                    "amsgrad": False,
                    "maximize": False,
                    "foreach": None,
                    "capturable": False,
                    "differentiable": False,
                    "fused": None,
                    "decoupled_weight_decay": False,
                    "params": list(range(task_space.EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT)),
                }
            ],
        },
        "update_count": update,
        "trainer_state": {
            "completed_update_count": update,
            "current_learning_iteration": update,
            "env_common_step_counter": update * task_space.NUM_STEPS_PER_ENV,
            "algorithm_learning_rate": 1.0e-5,
        },
        "initial_critic_state_sha256": "c" * 64,
        "optimizer_parameter_tensor_count": (task_space.EXPECTED_OPTIMIZER_PARAMETER_TENSOR_COUNT),
        "optimizer_step_count": optimizer_steps,
        "executed_training_transitions": update * task_space.TRANSITIONS_PER_UPDATE,
        "policy_state_sha256": ("358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61"),
        "critic_state_sha256": "d" * 64,
        "frozen_actor_state_sha256": (task_space.INITIAL_FROZEN_ACTOR_STATE_SHA256),
        "trainable_actor_state_sha256": (task_space.INITIAL_TRAINABLE_ACTOR_STATE_SHA256),
        "initial_overlay_policy_state_sha256": (
            "358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61"
        ),
        "contract_sha256": task_space.CONTRACT_SHA256,
        "run_materials_sha256": "e" * 64,
        "source_actor": {
            "checkpoint_sha256": ("85bd6de646905a44190dbf32c79737082bb604ab007a90a62e4fd2fdeeee6bd9"),
            "policy_state_sha256": ("c3bfcb5c42929293b62425f155b59ccb731f57c98e8852c7f1e97094525684af"),
            "lineage_sha256": ("08bbd03d0df751328e449d3624d79167587b461d6835fa1f4a8742aad9ffa82a"),
            "checkpoint_update_count": 250,
            "v2_decoder_sha256": ("011740f86483323fc0f1c39ab25b784cf9411b401e56fee8b7a716664e921ee1"),
            "critic_reused": False,
            "optimizer_reused": False,
            "counters_reused": False,
        },
        "training_boundary": {
            "trainable_actor_parameters": list(task_space.TRAINABLE_ACTOR_PARAMETERS),
            "trainable_actor_parameter_count": (task_space.EXPECTED_TRAINABLE_PARAMETER_COUNT),
            "exploration_std": 0.1,
            "exploration_std_trainable": False,
            "teacher_labels_used": False,
            "hardware_authorized": False,
            "deployment_ready": False,
        },
    }


def test_checkpoint_validator_binds_provenance_counters_and_adam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        task_space,
        "inspect_true23_policy_state",
        lambda *_args, **_kwargs: "358310ececeff0177386ae28f60b513a94902465b7e99ac480d40ba21578af61",
    )
    monkeypatch.setattr(task_space, "_validate_critic_state_schema", lambda _state: None)
    monkeypatch.setattr(
        task_space,
        "_policy_state_subsets",
        lambda _state: ({"frozen": torch.zeros(1)}, {"trainable": torch.zeros(1)}),
    )

    def fake_state_hash(state: dict[str, torch.Tensor]) -> str:
        if "frozen" in state:
            return task_space.INITIAL_FROZEN_ACTOR_STATE_SHA256
        if "trainable" in state:
            return task_space.INITIAL_TRAINABLE_ACTOR_STATE_SHA256
        return "d" * 64

    monkeypatch.setattr(task_space, "_state_sha256", fake_state_hash)
    body = _checkpoint_body(5)
    task_space.validate_task_space_checkpoint(
        body,
        expected_initial_critic_sha256="c" * 64,
        expected_run_materials_sha256="e" * 64,
    )

    missing_moment = copy.deepcopy(body)
    missing_moment["optimizer_state_dict"]["state"][0].pop("exp_avg")
    with pytest.raises(ValueError, match="entry schema"):
        task_space.validate_task_space_checkpoint(missing_moment)

    wrong_source = copy.deepcopy(body)
    wrong_source["source_actor"]["lineage_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="source boundary"):
        task_space.validate_task_space_checkpoint(wrong_source)

    with pytest.raises(ValueError, match="run materials"):
        task_space.validate_task_space_checkpoint(
            body,
            expected_run_materials_sha256="f" * 64,
        )


def test_bounded_schedule_stops_at_failed_update10_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = object.__new__(task_space.SonicTaskSpacePpoRunner)
    runner.completed_update_count = 0
    runner.current_learning_iteration = 0
    runner._optimizer_step_count = 0
    runner._executed_training_transitions = 0
    runner._initial_critic_state_sha256 = "c" * 64
    runner._run_materials_sha256 = "e" * 64
    runner.alg = SimpleNamespace(
        critic=torch.nn.Linear(1, 1),
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]),
    )
    runner._policy_state_adapter = SimpleNamespace(state_dict=lambda: {"policy": torch.zeros(1)})

    def coherence(self: Any) -> int:
        assert self.completed_update_count == self.current_learning_iteration
        return self.completed_update_count

    def save(self: Any) -> Path:
        return tmp_path / f"model_{self.completed_update_count}.pt"

    learned: list[int] = []

    def learn(self: Any, count: int, random_length: bool) -> None:
        assert random_length is False
        learned.append(count)
        self.completed_update_count += count
        self.current_learning_iteration += count
        self._optimizer_step_count += count * task_space.OPTIMIZER_STEPS_PER_UPDATE
        self._executed_training_transitions += count * task_space.TRANSITIONS_PER_UPDATE

    runner._require_counter_coherence = MethodType(coherence, runner)
    runner.save_current_checkpoint = MethodType(save, runner)
    runner.learn = MethodType(learn, runner)
    runner._assert_training_boundary = MethodType(lambda _self: None, runner)
    monkeypatch.setattr(
        task_space.torch,
        "load",
        lambda *_args, **_kwargs: {"update": runner.completed_update_count},
    )
    monkeypatch.setattr(
        task_space,
        "validate_task_space_checkpoint",
        lambda *_args, **_kwargs: {
            "update_count": runner.completed_update_count,
            "policy_state_sha256": f"policy-{runner.completed_update_count}",
        },
    )
    monkeypatch.setattr(
        task_space,
        "inspect_true23_policy_state",
        lambda *_args, **_kwargs: f"policy-{runner.completed_update_count}",
    )
    monkeypatch.setattr(task_space, "_state_sha256", lambda _state: "critic")
    monkeypatch.setattr(task_space, "_nested_state_equal", lambda *_args: True)

    def evaluator(_runner: Any, update: int) -> dict[str, Any]:
        if update == 5:
            return _evaluation(update, completed=125)
        return _evaluation(update, completed=155)

    phases: list[str] = []
    result = task_space.execute_bounded_pilot_schedule(
        runner,
        evaluator,
        phase_boundary=phases.append,
    )
    assert learned == [5, 5]
    assert runner.completed_update_count == 10
    assert result["assessment"]["stop_reason"] == "update10_no_q163_improvement"
    assert result["assessment"]["q163_update10_gate_reached"] is True
    assert result["assessment"]["q163_update10_gate_passed"] is False
    assert result["executed_training_transitions"] == 20_480
    assert result["optimizer_step_count"] == 80
    assert result["selected_candidate_update"] is None
    assert all("update_25" not in phase for phase in phases)


def test_bounded_schedule_reaches_update25_after_passing_update10_and_selects_only_passing_10_25_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = object.__new__(task_space.SonicTaskSpacePpoRunner)
    runner.completed_update_count = 0
    runner.current_learning_iteration = 0
    runner._optimizer_step_count = 0
    runner._executed_training_transitions = 0
    runner._initial_critic_state_sha256 = "c" * 64
    runner._run_materials_sha256 = "e" * 64
    runner.alg = SimpleNamespace(
        critic=torch.nn.Linear(1, 1),
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]),
    )
    runner._policy_state_adapter = SimpleNamespace(state_dict=lambda: {"policy": torch.zeros(1)})

    def coherence(self: Any) -> int:
        assert self.completed_update_count == self.current_learning_iteration
        return self.completed_update_count

    def save(self: Any) -> Path:
        return tmp_path / f"model_{self.completed_update_count}.pt"

    learned: list[int] = []

    def learn(self: Any, count: int, random_length: bool) -> None:
        assert random_length is False
        learned.append(count)
        self.completed_update_count += count
        self.current_learning_iteration += count
        self._optimizer_step_count += count * task_space.OPTIMIZER_STEPS_PER_UPDATE
        self._executed_training_transitions += count * task_space.TRANSITIONS_PER_UPDATE

    runner._require_counter_coherence = MethodType(coherence, runner)
    runner.save_current_checkpoint = MethodType(save, runner)
    runner.learn = MethodType(learn, runner)
    runner._assert_training_boundary = MethodType(lambda _self: None, runner)
    monkeypatch.setattr(
        task_space.torch,
        "load",
        lambda *_args, **_kwargs: {"update": runner.completed_update_count},
    )
    monkeypatch.setattr(
        task_space,
        "validate_task_space_checkpoint",
        lambda *_args, **_kwargs: {
            "update_count": runner.completed_update_count,
            "policy_state_sha256": f"policy-{runner.completed_update_count}",
        },
    )
    monkeypatch.setattr(
        task_space,
        "inspect_true23_policy_state",
        lambda *_args, **_kwargs: f"policy-{runner.completed_update_count}",
    )
    monkeypatch.setattr(task_space, "_state_sha256", lambda _state: "critic")
    monkeypatch.setattr(task_space, "_nested_state_equal", lambda *_args: True)

    def evaluator(_runner: Any, update: int) -> dict[str, Any]:
        if update == 5:
            return _evaluation(update, completed=125)
        if update == 10:
            return _evaluation(update, completed=156, episode_return=1.0)
        if update == 0:
            return _evaluation(update, completed=155)
        return _evaluation(update, completed=156, episode_return=2.0)

    phases: list[str] = []
    result = task_space.execute_bounded_pilot_schedule(
        runner,
        evaluator,
        phase_boundary=phases.append,
    )
    assert learned == [5, 5, 15]
    assert runner.completed_update_count == 25
    assert result["assessment"]["stop"] is False
    assert result["assessment"]["q163_update10_gate_reached"] is True
    assert result["assessment"]["q163_update10_gate_passed"] is True
    assert result["assessment"]["pilot_complete"] is True
    assert result["executed_training_transitions"] == 51_200
    assert result["optimizer_step_count"] == 200
    assert result["selected_candidate_update"] == 25
    assert result["selected_candidate_checkpoint_path"].endswith("model_25.pt")
    assert "before_training_to_update_25" in phases
    assert "after_training_to_update_25" in phases


def test_cli_exposes_no_training_override_controls() -> None:
    parser = launcher._parser()
    parsed = parser.parse_args(["pilot", "--run-dir", "pilot-output"])
    assert parsed.command == "pilot"
    for forbidden in ("--seed", "--resume", "--num-envs", "--updates", "--gpu"):
        with pytest.raises(SystemExit):
            parser.parse_args(["pilot", "--run-dir", "pilot-output", forbidden, "1"])


def test_evaluation_rng_context_restores_python_numpy_and_torch() -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    with launcher._preserved_rng_for_evaluation():
        random.random()
        np.random.random()
        torch.rand(4)
        if torch.cuda.is_available():
            torch.rand(4, device="cuda:0")
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert all(torch.equal(after, before) for after, before in zip(torch.cuda.get_rng_state_all(), cuda_before))
