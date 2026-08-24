from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch

from gear_sonic.scripts import qualify_g1_true23_sonic_rank256_phase_balance as phase


def _contract() -> dict:
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / phase.CONTRACT_RELATIVE_PATH).read_text(encoding="utf-8"))


def test_contract_hash_and_fixed_controller() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / phase.CONTRACT_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == phase.CONTRACT_SHA256
    body = _contract()
    assert body["controller"]["phase_window_q9_inclusive"] == [180, 300]
    assert body["controller"]["ankle_pitch_coefficient"] == -8.0
    assert body["controller"]["signal"]["deadband"] == 0.06
    assert body["controller"]["maximum_applied_raw_residual_abs"] == 1.0


def test_contract_has_disjoint_twenty_run_cohort() -> None:
    qualification = _contract()["qualification"]
    assert len(set(qualification["nominal_seeds"])) == 10
    assert len(set(qualification["disturbance_seeds"])) == 10
    assert not set(qualification["nominal_seeds"]) & set(qualification["disturbance_seeds"])
    assert len(qualification["disturbance_vectors"]) == 10
    assert qualification["disturbance_vectors"][0] == [
        -0.109284965708,
        0.348842165152,
        -0.026012141987,
        0.137065557322,
        -0.499675156302,
        -0.717607152875,
    ]


def _install_fake_mjlab(monkeypatch) -> None:
    modules = {
        name: ModuleType(name)
        for name in ("mjlab", "mjlab.utils", "mjlab.utils.lab_api", "mjlab.utils.lab_api.math")
    }
    modules["mjlab.utils.lab_api.math"].quat_apply_inverse = lambda _quat, vector: vector
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


class _Base(torch.nn.Module):
    def forward(self, observations, stochastic_output=False):
        del observations, stochastic_output
        return torch.zeros((1, 23), dtype=torch.float32)

    def export_true23_policy_state(self):
        return {}


class _Commands:
    def __init__(self, q9: int) -> None:
        self.time_steps = torch.tensor([q9], dtype=torch.long)
        self.anchor_quat_w = torch.zeros((1, 4))
        self.robot_anchor_quat_w = torch.zeros((1, 4))
        self.anchor_lin_vel_w = torch.tensor([[0.2, 0.0, 0.0]])
        self.robot_anchor_lin_vel_w = torch.zeros((1, 3))
        self.anchor_ang_vel_w = torch.tensor([[0.0, 0.2, 0.0]])
        self.robot_anchor_ang_vel_w = torch.zeros((1, 3))


class _RawEnv:
    def __init__(self, q9: int) -> None:
        self.command = _Commands(q9)
        data = SimpleNamespace(gravity_vec_w=torch.tensor([[0.2, 0.0, -1.0]]))
        self.scene = {"robot": SimpleNamespace(data=data)}
        self.command_manager = SimpleNamespace(get_term=lambda _name: self.command)


def _policy(raw: _RawEnv) -> phase.PhaseBalancePolicy:
    signal = {
        "orientation_error_gain": 1.5,
        "angular_velocity_error_gain": 0.5,
        "linear_velocity_error_gain": 0.1,
        "deadband": 0.06,
        "signal_clip": 1.0,
    }
    return phase.PhaseBalancePolicy(_Base(), raw, signal, 0.4, 0.4, 1.0, -8.0, 0.0, start_q9=180, stop_q9=300)


def test_phase_residual_disabled_outside_window(monkeypatch) -> None:
    _install_fake_mjlab(monkeypatch)
    raw = _RawEnv(179)
    policy = _policy(raw)
    output = policy({})
    assert torch.equal(output, torch.zeros_like(output))
    assert policy.active_q9_first is None
    assert policy.activation_count == 0


def test_phase_residual_applies_only_ankle_pitch_inside_window(monkeypatch) -> None:
    _install_fake_mjlab(monkeypatch)
    raw = _RawEnv(180)
    policy = _policy(raw)
    output = policy({})
    changed = torch.nonzero(output[0], as_tuple=False).flatten().tolist()
    assert changed == [15, 16]
    assert output[0, 15].item() == output[0, 16].item()
    assert abs(output[0, 15].item()) <= 1.0
    assert policy.active_q9_first == 180
    assert policy.active_q9_last == 180
    assert policy.activation_count == 1


def test_assessment_requires_exact_timeout_and_clean_counts() -> None:
    contract = _contract()
    record = {
        "mode": "disturbance",
        "completed_transitions": 510,
        "terminal_q9": 518,
        "termination_names": ["time_out"],
        "nonfinite_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": 0,
        "q9_discontinuity_count": 0,
        "active_q9_first": 180,
        "active_q9_last": 300,
        "residual_activation_count": 1,
        "maximum_abs_residual": 1.0,
        "impulse_applied": True,
    }
    assert phase._clean(record, contract)  # noqa: SLF001
    record["terminal_q9"] = 517
    assert not phase._clean(record, contract)  # noqa: SLF001


def test_boundaries_forbid_training_hardware_and_network() -> None:
    boundaries = _contract()["boundaries"]
    assert boundaries["training_transitions"] == 0
    assert boundaries["optimizer_steps"] == 0
    assert boundaries["robot_or_network_commands_permitted"] is False
    assert boundaries["hardware_authorized"] is False
