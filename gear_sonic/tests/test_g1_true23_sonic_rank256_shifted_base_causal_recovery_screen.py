from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gear_sonic.scripts import screen_g1_true23_sonic_rank256_shifted_base_causal_recovery as screen
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def _row(scale: float, scenario: str, steps: int, policy: str) -> dict[str, object]:
    return {
        "scale": scale,
        "scenario": scenario,
        "completed_transitions": steps,
        "policy_state_sha256": policy,
        **{name: 0 for name in screen.disturbance.ZERO_COUNTS},
    }


def test_contract_is_hash_locked_and_bounded() -> None:
    path = ROOT / screen.CONTRACT_RELATIVE_PATH
    body = json.loads(path.read_text(encoding="utf-8"))
    assert sha256_file(path) == screen.CONTRACT_SHA256
    assert body["screen"]["scales"] == list(screen.SCALES)
    assert body["screen"]["scenarios"] == list(screen.SCENARIOS)
    assert body["boundaries"]["additional_training_transitions"] == 0
    assert body["boundaries"]["optimizer_steps"] == 0
    assert body["boundaries"]["hardware_authorized"] is False


def test_state_changes_only_direction_tensors() -> None:
    baseline = {"frozen": torch.tensor([4.0])}
    direction: dict[str, torch.Tensor] = {}
    for index, name in enumerate(screen.STATE_PARAMETER_NAMES, start=1):
        baseline[name] = torch.tensor([float(index)])
        direction[name] = torch.tensor([0.25 * index])
    state = screen._state(baseline, direction, -2.0)  # noqa: SLF001
    assert torch.equal(state["frozen"], baseline["frozen"])
    for index, name in enumerate(screen.STATE_PARAMETER_NAMES, start=1):
        assert torch.equal(state[name], torch.tensor([0.5 * index]))


def test_assess_requires_clean_joint_improvement() -> None:
    records = []
    for scale in screen.SCALES:
        policy = f"{int((scale + 8) * 10):064d}"
        nominal = 507 if scale == 0.0 else (490 if scale == 1.0 else 480)
        push = 287 if scale == 0.0 else (289 if scale == 1.0 else 300)
        records.extend((_row(scale, "nominal", nominal, policy), _row(scale, "disturbance", push, policy)))
    result = screen._assess(records)  # noqa: SLF001
    assert result["baseline_passed"] is True
    assert result["candidate_selected"] is True
    assert result["selected_scale"] == 1.0
    assert result["selected_nominal_completed_transitions"] == 490
    assert result["selected_disturbance_completed_transitions"] == 289


def test_assess_rejects_bad_baseline() -> None:
    records = []
    for scale in screen.SCALES:
        policy = f"{int((scale + 8) * 10):064d}"
        nominal = 485 if scale == 0.0 else 510
        push = 287 if scale == 0.0 else 300
        records.extend((_row(scale, "nominal", nominal, policy), _row(scale, "disturbance", push, policy)))
    result = screen._assess(records)  # noqa: SLF001
    assert result["baseline_passed"] is False
    assert result["candidate_selected"] is False


def test_names_encode_sign_stably() -> None:
    assert screen._name(-0.5, "nominal") == "scale_m0p5_nominal.json"  # noqa: SLF001
    assert screen._name(8.0, "disturbance") == "scale_8_disturbance.json"  # noqa: SLF001


def test_preflight_is_read_only_when_bound_linux_inputs_exist() -> None:
    if not Path("/root/g1_true23_runs/sonic_rank256_shifted_base_causal_recovery_score_v1").is_dir():
        pytest.skip("bound WSL artifacts unavailable")
    report = screen.preflight(ROOT)
    assert report["ready"] is True
    assert report["simulator_constructed"] is False
    assert report["evaluation_runs"] == 0
    assert report["optimizer_steps"] == 0
    assert report["hardware_authorized"] is False
