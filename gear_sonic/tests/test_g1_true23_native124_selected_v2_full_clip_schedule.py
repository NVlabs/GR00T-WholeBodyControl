from __future__ import annotations

import copy

import pytest

from gear_sonic.envs.mjlab import native124_selected_v2_causal_adaptation as causal
from gear_sonic.utils import g1_true23_native124_selected_v2_full_clip_schedule as schedule
from gear_sonic.utils.g1_23dof_artifact import canonical_json_bytes, sha256_bytes


def test_continuous_contract_covers_every_valid_causal_action_once() -> None:
    contract = schedule.continuous_contract()

    assert contract == {
        "anchor_q9": 9,
        "causal_max_q9": 2088,
        "causal_min_q9": 9,
        "control_hz": 50,
        "motion_frame_count": 2090,
        "q10_proof_last": 2089,
        "schema": "g1_true23_daddance_full_clip_continuous_v1",
        "score_q9": [9, 2088],
        "transitions": 2080,
    }
    assert schedule.FULL_CLIP_CONTINUOUS_WINDOW.last_q9 == 2088
    assert schedule.FULL_CLIP_CONTINUOUS_WINDOW.last_q9 + 1 == 2089
    assert sha256_bytes(canonical_json_bytes(contract)) == schedule.CONTINUOUS_CONTRACT_SHA256
    assert schedule.validate_continuous_contract(contract) == contract


def test_phase_schedule_has_exact_overlap_and_unique_primary_ownership() -> None:
    contract = schedule.phase_schedule_contract()

    assert contract["windows"] == [
        {
            "anchor_q9": 9,
            "burn_in_transitions": 0,
            "id": "w0",
            "score_q9": [9, 508],
            "transitions": 500,
        },
        {
            "anchor_q9": 409,
            "burn_in_transitions": 100,
            "id": "w1",
            "score_q9": [509, 908],
            "transitions": 500,
        },
        {
            "anchor_q9": 809,
            "burn_in_transitions": 100,
            "id": "w2",
            "score_q9": [909, 1308],
            "transitions": 500,
        },
        {
            "anchor_q9": 1209,
            "burn_in_transitions": 100,
            "id": "w3",
            "score_q9": [1309, 1708],
            "transitions": 500,
        },
        {
            "anchor_q9": 1609,
            "burn_in_transitions": 100,
            "id": "w4",
            "score_q9": [1709, 2088],
            "transitions": 480,
        },
    ]
    assert sha256_bytes(canonical_json_bytes(contract)) == schedule.PHASE_SCHEDULE_SHA256

    primary_owners: dict[int, list[str]] = {q9: [] for q9 in range(9, 2089)}
    for window in schedule.PHASE_WINDOWS:
        for q9 in range(window.first_scored_q9, window.last_q9 + 1):
            primary_owners[q9].append(window.window_id)
    assert all(len(owners) == 1 for owners in primary_owners.values())

    for previous, current in zip(
        schedule.PHASE_WINDOWS[:-1],
        schedule.PHASE_WINDOWS[1:],
        strict=True,
    ):
        assert previous.last_q9 - current.anchor_q9 + 1 == 100
        assert current.first_scored_q9 == previous.last_q9 + 1
    assert schedule.PHASE_WINDOWS[-1].last_q9 + 1 == schedule.FINAL_Q10_PROOF


def test_aggregate_schedule_binds_but_does_not_conflate_both_evidence_types() -> None:
    contract = schedule.qualification_schedule_contract()

    assert contract["continuous"] == schedule.continuous_contract()
    assert contract["phase_schedule"] == schedule.phase_schedule_contract()
    assert sha256_bytes(canonical_json_bytes(contract)) == schedule.QUALIFICATION_SCHEDULE_SHA256
    assert schedule.validate_qualification_schedule_contract(contract) == contract


def test_schedule_contracts_are_fresh_and_fail_closed_on_tamper() -> None:
    first = schedule.qualification_schedule_contract()
    second = schedule.qualification_schedule_contract()
    assert first == second
    assert first is not second
    assert first["phase_schedule"] is not second["phase_schedule"]

    tampered = copy.deepcopy(first)
    tampered["phase_schedule"]["windows"][4]["transitions"] = 481
    with pytest.raises(ValueError, match="differs from the pinned schedule"):
        schedule.validate_qualification_schedule_contract(tampered)
    with pytest.raises(TypeError, match="must be a mapping"):
        schedule.validate_phase_schedule_contract([])  # type: ignore[arg-type]


def test_schedule_constants_match_causal_runtime_contract() -> None:
    assert schedule.CAUSAL_MIN_Q9 == causal.CAUSAL_HISTORY_ANCHOR_INDEX == 9
    assert schedule.CAUSAL_MAX_Q9 == causal.DAD_DANCE_FRAME_COUNT - 2 == 2088
    assert schedule.MOTION_FRAME_COUNT == causal.DAD_DANCE_FRAME_COUNT == 2090
    assert schedule.CONTROL_HZ == int(causal.CONTROL_HZ) == 50


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_id": "", "anchor_q9": 9, "transitions": 1, "burn_in_transitions": 0}, "window_id"),
        ({"window_id": "x", "anchor_q9": True, "transitions": 1, "burn_in_transitions": 0}, "anchor_q9"),
        ({"window_id": "x", "anchor_q9": 8, "transitions": 1, "burn_in_transitions": 0}, "causal q9"),
        ({"window_id": "x", "anchor_q9": 9, "transitions": 0, "burn_in_transitions": 0}, "positive"),
        ({"window_id": "x", "anchor_q9": 9, "transitions": 2, "burn_in_transitions": 2}, "within"),
        ({"window_id": "x", "anchor_q9": 2088, "transitions": 2, "burn_in_transitions": 0}, "q10 proof"),
    ],
)
def test_window_rejects_invalid_or_unprovable_ranges(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        schedule.CausalQualificationWindow(**kwargs)
