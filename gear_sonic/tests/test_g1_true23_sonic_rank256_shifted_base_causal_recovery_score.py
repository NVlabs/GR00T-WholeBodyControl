from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gear_sonic.scripts import collect_g1_true23_sonic_rank256_shifted_base_causal_recovery_score as shifted
from gear_sonic.utils.g1_23dof_artifact import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def test_contract_is_hash_locked_and_post_push_only() -> None:
    path = ROOT / shifted.CONTRACT_RELATIVE_PATH
    body = json.loads(path.read_text(encoding="utf-8"))
    assert sha256_file(path) == shifted.CONTRACT_SHA256
    assert body["collection"]["all_environments_receive_exact_impulse"] is True
    assert body["collection"]["first_credited_transition"] == 242
    assert body["collection"]["first_credited_q9"] == 251
    assert body["objective"]["precredited_nonzero_weights_required"] == 0
    assert body["gradient"]["optimizer_steps"] == 0
    assert body["boundaries"]["hardware_authorized"] is False


def test_return_weights_are_zero_before_push_and_balanced() -> None:
    generator = torch.Generator().manual_seed(9)
    scores = torch.randn((shifted.COLLECTION_STEPS, shifted.NUM_ENVS), generator=generator)
    active = torch.ones_like(scores, dtype=torch.bool)
    weights, evidence = shifted._return_weights(scores, active)  # noqa: SLF001
    assert torch.count_nonzero(weights[: shifted.FIRST_CREDITED_TRANSITION]) == 0
    assert evidence["precredited_nonzero_weight_count"] == 0
    assert [row["valid_transition_count"] for row in evidence["replicate_groups"]] == [268, 268]
    count_a = torch.count_nonzero(weights[:, :64])
    count_b = torch.count_nonzero(weights[:, 64:])
    assert count_a == count_b


def test_return_weights_fail_when_replicates_lack_variance() -> None:
    scores = torch.zeros((shifted.COLLECTION_STEPS, shifted.NUM_ENVS))
    active = torch.ones_like(scores, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="replicate gate failed"):
        shifted._return_weights(scores, active)  # noqa: SLF001


def test_replicate_groups_cover_all_environments_exactly_once() -> None:
    indices = []
    for _name, start, stop in shifted.GROUPS:
        indices.extend(range(start, stop))
    assert indices == list(range(shifted.NUM_ENVS))
    assert shifted.GROUPS == (
        ("exact_impulse_a", 0, 64),
        ("exact_impulse_b", 64, 128),
    )


def test_preflight_is_read_only_when_bound_linux_inputs_exist() -> None:
    if not Path("/root/g1_true23_runs/sonic_rank256_group_balanced_clipped_screen_v1").is_dir():
        pytest.skip("bound WSL artifacts unavailable")
    report = shifted.preflight(ROOT)
    assert report["ready"] is True
    assert report["simulator_constructed"] is False
    assert report["training_transitions"] == 0
    assert report["gradient_computations"] == 0
    assert report["optimizer_steps"] == 0
    assert report["hardware_authorized"] is False
