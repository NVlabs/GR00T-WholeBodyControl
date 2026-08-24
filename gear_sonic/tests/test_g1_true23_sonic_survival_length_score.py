from __future__ import annotations

from pathlib import Path

from gear_sonic.scripts import train_g1_true23_sonic_survival_length_score as length_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _record(scale: float, completed: int, q9: int, termination: str = "anchor_pos") -> dict[str, object]:
    return {
        "scale": scale,
        "completed_transitions": completed,
        "terminal_q9": q9,
        "termination_names": [termination],
        "episode_return": -90.0,
        "policy_state_sha256": f"{int((scale + 1) * 100):064x}",
        "nonfinite_count": 0,
        "raw_clip_required_count": 0,
        "action_semantics_mismatch_count": 0,
        "q9_discontinuity_count": 0,
    }


def test_survival_length_contract_is_hash_locked() -> None:
    contract = length_score._load_contract(REPOSITORY_ROOT)
    assert contract["collection"]["num_steps"] == 360
    assert contract["gradient"]["scales"] == list(length_score.SCALES)


def test_survival_length_assessment_requires_exact_baseline_and_improvement() -> None:
    records = [_record(scale, 358, 366) for scale in length_score.SCALES]
    records[list(length_score.SCALES).index(0.0)] = _record(0.0, 358, 366)
    records[list(length_score.SCALES).index(0.5)] = _record(0.5, 510, 518, "time_out")
    result = length_score.assess(records)
    assert result["baseline_passed"] is True
    assert result["candidate_selected"] is True
    assert result["selected_scale"] == 0.5
    assert result["selected_terminal_q9"] == 518


def test_survival_length_assessment_rejects_dirty_candidate() -> None:
    records = [_record(scale, 358, 366) for scale in length_score.SCALES]
    records[list(length_score.SCALES).index(0.0)] = _record(0.0, 358, 366)
    index = list(length_score.SCALES).index(1.0)
    records[index] = _record(1.0, 500, 508)
    records[index]["action_semantics_mismatch_count"] = 1
    result = length_score.assess(records)
    assert result["candidate_selected"] is False


def test_survival_length_local_scale_zero_is_not_original_update_zero(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_parent(**kwargs):
        captured.update(kwargs)
        return {"completed_transitions": 358}

    monkeypatch.setattr(length_score, "_evaluate_parent_state", fake_parent)
    result = length_score._evaluate_state(
        state={},
        scale=0.0,
        topology=Path("topology.pt"),
        motion=Path("motion.npz"),
        material_manifest_sha256="0" * 64,
    )
    assert result == {"completed_transitions": 358}
    assert captured["scale"] == 0.0
    assert captured["evaluation_update_count"] == 1
