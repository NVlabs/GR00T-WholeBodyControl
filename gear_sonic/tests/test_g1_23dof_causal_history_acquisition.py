"""Fail-closed source tests for causal acquisition v2."""

from __future__ import annotations

from gear_sonic.envs.mjlab.sonic_true23_causal_history_acquisition import (
    causal_acquisition_contract,
)


def test_causal_acquisition_contract_keeps_strict_termination() -> None:
    contract = causal_acquisition_contract()
    assert contract["non_timeout_termination"] == {
        "function": "mjlab.envs.mdp.is_terminated",
        "weight": -200.0,
    }
    assert contract["ee_body_pos_termination_threshold_m"] == 0.25
    assert contract["ee_termination_weakened"] is False


def test_causal_acquisition_is_new_lineage_stage() -> None:
    contract = causal_acquisition_contract()
    assert contract["restart_from_approved_initialization"] is True
    assert contract["collapsed_model_250_reused"] is False
    assert contract["critic_reused"] is False
    assert contract["optimizer_reused"] is False
    assert contract["interval_pushes_enabled"] is False
    assert contract["disturbance_finetune_required_before_promotion"] is True
