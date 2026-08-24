from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from gear_sonic.scripts import (
    qualify_g1_true23_sonic_rank256_phase_balance as core,
    qualify_g1_true23_sonic_rank256_phase_balance_v2 as v2,
)


def _contract() -> dict:
    root = Path(__file__).resolve().parents[2]
    path = root / v2.CONTRACT_RELATIVE_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def test_v2_contract_hash_and_late_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / v2.CONTRACT_RELATIVE_PATH
    assert hashlib.sha256(path.read_bytes()).hexdigest() == v2.CONTRACT_SHA256
    body = _contract()
    late = body["controller"]["late_policy"]
    assert late["switch_after_q9"] == 400
    assert late["direction_scale"] == -1.0
    assert late["first_late_action_q9"] == 401
    assert late["last_late_action_q9"] == 518


def test_v2_scope_restores_core_globals() -> None:
    before = (
        core.CONTRACT_RELATIVE_PATH,
        core.CONTRACT_SHA256,
        core.CONTRACT_KIND,
        core.RUN_DIR_DEFAULT,
    )
    with v2._scope():  # noqa: SLF001
        assert core.CONTRACT_RELATIVE_PATH == v2.CONTRACT_RELATIVE_PATH
        assert core.CONTRACT_SHA256 == v2.CONTRACT_SHA256
        assert core.CONTRACT_KIND == v2.CONTRACT_KIND
        assert core.RUN_DIR_DEFAULT == v2.RUN_DIR_DEFAULT
    assert before == (
        core.CONTRACT_RELATIVE_PATH,
        core.CONTRACT_SHA256,
        core.CONTRACT_KIND,
        core.RUN_DIR_DEFAULT,
    )


class _Actor(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, observations, stochastic_output=False):
        del observations, stochastic_output
        return torch.full((1, 23), self.value)

    def export_true23_policy_state(self):
        return {"value": torch.tensor(self.value)}


def test_switch_is_exactly_after_q400() -> None:
    command = SimpleNamespace(time_steps=torch.tensor([400], dtype=torch.long))
    raw = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: command))
    policy = core.SwitchedBasePolicy(_Actor(0.0), _Actor(1.0), raw, 400)
    assert torch.equal(policy({}), torch.zeros((1, 23)))
    assert policy.late_policy_first_q9 is None
    command.time_steps.fill_(401)
    assert torch.equal(policy({}), torch.ones((1, 23)))
    assert policy.late_policy_first_q9 == 401
    assert policy.late_policy_last_q9 == 401


def test_v2_clean_requires_late_actor_span() -> None:
    contract = _contract()
    record = {
        "mode": "nominal",
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
        "impulse_applied": False,
        "late_policy_first_q9": 401,
        "late_policy_last_q9": 518,
        "late_policy_state_sha256": contract["controller"]["late_policy"]["policy_state_sha256"],
    }
    assert core._clean(record, contract)  # noqa: SLF001
    record["late_policy_first_q9"] = 402
    assert not core._clean(record, contract)  # noqa: SLF001


def test_v2_binds_failed_v1_and_forbids_hardware() -> None:
    body = _contract()
    assert body["parents"]["failed_v1_nominal"]["sha256"] == (
        "f94e0fea2d79c4fc2793c668e2d8c3b612a719e2bd9d1d8e6d539aaef1a35972"
    )
    assert body["boundaries"]["robot_or_network_commands_permitted"] is False
    assert body["boundaries"]["hardware_authorized"] is False
