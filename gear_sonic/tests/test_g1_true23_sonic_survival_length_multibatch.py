from __future__ import annotations

from pathlib import Path

import torch

from gear_sonic.scripts import train_g1_true23_sonic_survival_length_multibatch as multibatch
from gear_sonic.utils.g1_true23_sonic_survival_score_line_search import STATE_PARAMETER_NAMES

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _record(scale: float, completed: int, policy: str, termination: str = "anchor_pos") -> dict[str, object]:
    return {
        "scale": scale,
        "completed_transitions": completed,
        "terminal_q9": 8 + completed,
        "termination_names": [termination],
        "episode_return": -90.0,
        "policy_state_sha256": policy,
        "nonfinite_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": 0,
        "q9_discontinuity_count": 0,
    }


def test_multibatch_contract_is_hash_locked() -> None:
    contract = multibatch._load_contract(REPOSITORY_ROOT)
    assert contract["collection"]["batch_count"] == 4
    assert contract["gradient"]["screen_scales"] == list(multibatch.SCALES)


def test_multibatch_direction_average_is_renormalized() -> None:
    directions = []
    for index in range(multibatch.BATCH_COUNT):
        directions.append(
            {name: torch.full((2,), float(index + 1), dtype=torch.float32) for name in STATE_PARAMETER_NAMES}
        )
    averaged, evidence = multibatch.aggregate_directions(directions)
    assert set(averaged) == set(STATE_PARAMETER_NAMES)
    assert abs(evidence["normalized_direction_l2"] - multibatch.TARGET_DIRECTION_L2) < 1e-9
    assert len(evidence["pairwise_direction_cosines"]) == multibatch.BATCH_COUNT


def test_multibatch_screen_and_confirmation_require_real_margin() -> None:
    records = [_record(scale, 350, f"{index + 1:064x}") for index, scale in enumerate(multibatch.SCALES)]
    baseline_index = multibatch.SCALES.index(0.0)
    records[baseline_index] = _record(0.0, 358, multibatch.single.SOURCE_POLICY_SHA256)
    candidate_index = multibatch.SCALES.index(0.1)
    records[candidate_index] = _record(0.1, 380, "a" * 64, "ee_body_pos")
    assessment = multibatch.assess_screen(records)
    assert assessment["screen_candidate_found"] is True
    assert assessment["selected_scale"] == 0.1
    assert (
        multibatch.assess_confirmation(
            _record(0.0, 357, multibatch.single.SOURCE_POLICY_SHA256),
            _record(0.1, 370, "a" * 64, "ee_body_pos"),
            "a" * 64,
        )
        is True
    )


def test_multibatch_screen_rejects_one_frame_noise() -> None:
    records = [_record(scale, 350, f"{index + 1:064x}") for index, scale in enumerate(multibatch.SCALES)]
    baseline_index = multibatch.SCALES.index(0.0)
    records[baseline_index] = _record(0.0, 358, multibatch.single.SOURCE_POLICY_SHA256)
    candidate_index = multibatch.SCALES.index(0.05)
    records[candidate_index] = _record(0.05, 359, "b" * 64)
    assert multibatch.assess_screen(records)["screen_candidate_found"] is False


def test_multibatch_creates_all_output_subdirectories(tmp_path: Path) -> None:
    (tmp_path / "evaluations").mkdir()
    (tmp_path / "checkpoints").mkdir()
    multibatch._create_output_subdirectories(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "batches",
        "checkpoints",
        "evaluations",
        "gradient",
    ]
