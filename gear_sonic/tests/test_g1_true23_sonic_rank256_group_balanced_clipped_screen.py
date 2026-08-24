from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gear_sonic.scripts import screen_g1_true23_sonic_rank256_group_balanced_clipped_update as screen
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def _record(scale: float, scenario: str, steps: int, policy: str) -> dict[str, object]:
    return {
        "screen_scale": scale,
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


def test_state_interpolates_only_trainable_tensors() -> None:
    baseline = {"frozen": torch.tensor([3.0])}
    delta: dict[str, torch.Tensor] = {}
    for index, name in enumerate(screen.STATE_PARAMETER_NAMES, start=1):
        baseline[name] = torch.tensor([float(index)])
        delta[name] = torch.tensor([0.5 * index])
    observed = screen._state(baseline, delta, 2.0)  # noqa: SLF001
    assert torch.equal(observed["frozen"], baseline["frozen"])
    for index, name in enumerate(screen.STATE_PARAMETER_NAMES, start=1):
        assert torch.equal(observed[name], torch.tensor([2.0 * index]))


def test_assess_selects_joint_improvement_only() -> None:
    records = []
    for scale in screen.SCALES:
        policy = f"{int(scale * 100):064d}"
        nominal = 505 if scale == 0.0 else {0.25: 487, 0.5: 490, 1.0: 480, 2.0: 500}[scale]
        disturbance = 287 if scale == 0.0 else {0.25: 288, 0.5: 289, 1.0: 300, 2.0: 288}[scale]
        records.extend(
            (
                _record(scale, "nominal", nominal, policy),
                _record(scale, "disturbance", disturbance, policy),
            )
        )
    result = screen._assess(records)  # noqa: SLF001
    assert result["candidate_selected"] is True
    assert result["selected_scale"] == 0.5
    assert result["selected_nominal_completed_transitions"] == 490
    assert result["selected_disturbance_completed_transitions"] == 289
    assert result["support_qualified"] is False


def test_assess_rejects_nominal_disturbance_tradeoff() -> None:
    records = []
    for scale in screen.SCALES:
        policy = f"{int(scale * 100):064d}"
        records.extend(
            (
                _record(scale, "nominal", 505 if scale == 0.0 else 485, policy),
                _record(scale, "disturbance", 287 if scale == 0.0 else 288, policy),
            )
        )
    assert screen._assess(records)["candidate_selected"] is False  # noqa: SLF001


def test_names_are_stable() -> None:
    assert screen._name(0.0, "nominal") == "scale_0_nominal.json"  # noqa: SLF001
    assert screen._name(0.25, "disturbance") == "scale_0p25_disturbance.json"  # noqa: SLF001


def test_preflight_is_read_only_when_bound_linux_inputs_exist() -> None:
    if not Path("/root/g1_true23_runs/sonic_rank256_group_balanced_clipped_update_v1").is_dir():
        pytest.skip("bound WSL artifacts unavailable")
    report = screen.preflight(ROOT)
    assert report["ready"] is True
    assert report["simulator_constructed"] is False
    assert report["evaluation_runs"] == 0
    assert report["optimizer_steps"] == 0
    assert report["hardware_authorized"] is False
